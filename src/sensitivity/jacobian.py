"""Jacobian evaluation, finite differences, and first-order maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import DifferentiableModel, as_vector

Array = NDArray[np.floating]
JacobianSource = Literal["analytical", "central", "forward", "complex", "jax", "auto"]


@dataclass(frozen=True)
class JacobianEstimate:
    """A Jacobian together with the method used to obtain it."""

    matrix: Array
    source: str
    point: Array
    step: Array | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.matrix.shape)  # type: ignore[return-value]


def jvp(jacobian: ArrayLike, dx: ArrayLike) -> Array:
    """Push a perturbation through ``J``: ``dy ~ J dx``."""
    jac = np.asarray(jacobian, dtype=float)
    delta = as_vector(dx, "dx")
    if jac.ndim != 2:
        raise ValueError(f"Jacobian must be 2-D, got {jac.shape}")
    if jac.shape[1] != delta.size:
        raise ValueError(f"J is {jac.shape}, dx has length {delta.size}")
    return jac @ delta


def vjp(jacobian: ArrayLike, dy: ArrayLike) -> Array:
    """Pull a cotangent through ``J``: ``dxtilde = J^T dy``."""
    jac = np.asarray(jacobian, dtype=float)
    cotan = as_vector(dy, "dy")
    if jac.ndim != 2:
        raise ValueError(f"Jacobian must be 2-D, got {jac.shape}")
    if jac.shape[0] != cotan.size:
        raise ValueError(f"J is {jac.shape}, dy has length {cotan.size}")
    return jac.T @ cotan


def _default_step(x: Array, rel: float) -> Array:
    return rel * np.maximum(1.0, np.abs(x))


def finite_difference_jacobian(
    model: DifferentiableModel,
    x: ArrayLike,
    *,
    method: Literal["central", "forward", "complex"] = "central",
    relative_step: float | None = None,
) -> JacobianEstimate:
    """Approximate ``df/dx`` without requiring an analytical formula.

    Central differences are the default. Complex-step differentiation is
    available when ``f`` accepts a complex probe and returns a complex
    value whose real part is ``f(x)``. It is exact to first order in the
    rounding of ``f`` for holomorphic real-on-real maps.
    """
    point = as_vector(x, "x")
    if point.size != model.input_dim:
        raise ValueError(f"expected input dim {model.input_dim}, got {point.size}")

    if method == "complex":
        step_scale = 1e-20 if relative_step is None else relative_step
        steps = _default_step(point, step_scale)
        jac = np.zeros((model.output_dim, model.input_dim), dtype=float)
        for i in range(model.input_dim):
            probe = point.astype(complex)
            probe[i] += 1j * steps[i]
            value = np.asarray(model.forward(probe), dtype=complex)
            if value.ndim == 0:
                value = value.reshape(1)
            jac[:, i] = np.real(value) * 0.0 + np.imag(value) / steps[i]
        return JacobianEstimate(matrix=jac, source="complex", point=point, step=steps)

    if relative_step is None:
        relative_step = 1.5e-6 if method == "central" else 1.0e-6
    steps = _default_step(point, relative_step)
    y0 = model.evaluate(point)
    jac = np.zeros((model.output_dim, model.input_dim), dtype=float)
    for i in range(model.input_dim):
        plus = point.copy()
        plus[i] += steps[i]
        y_plus = model.evaluate(plus)
        if method == "forward":
            jac[:, i] = (y_plus - y0) / steps[i]
        else:
            minus = point.copy()
            minus[i] -= steps[i]
            y_minus = model.evaluate(minus)
            jac[:, i] = (y_plus - y_minus) / (2.0 * steps[i])
    return JacobianEstimate(matrix=jac, source=method, point=point, step=steps)


def _jax_jacobian(model: DifferentiableModel, x: Array) -> JacobianEstimate:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional backend
        raise RuntimeError("jax is not installed; install the [ad] extra") from exc

    def wrapped(vec):
        return jnp.asarray(model.forward(np.asarray(vec, dtype=float)))

    try:
        jac = np.asarray(jax.jacfwd(wrapped)(jnp.asarray(x, dtype=float)), dtype=float)
        return JacobianEstimate(matrix=jac, source="jax", point=x)
    except Exception:
        return finite_difference_jacobian(model, x, method="central")


def jacobian_at(
    model: DifferentiableModel,
    x: ArrayLike,
    *,
    source: JacobianSource = "auto",
) -> JacobianEstimate:
    """Return a Jacobian at ``x``.

    ``source='auto'`` prefers a declared analytical Jacobian, then central
    finite differences. An explicit method name forces that estimator.
    """
    point = as_vector(x, "x")
    if source == "analytical":
        return JacobianEstimate(
            matrix=model.analytical_jacobian(point),
            source="analytical",
            point=point,
        )
    if source in {"central", "forward", "complex"}:
        return finite_difference_jacobian(model, point, method=source)
    if source == "jax":
        return _jax_jacobian(model, point)
    if source != "auto":
        raise ValueError(f"unknown Jacobian source {source!r}")
    if model.jacobian is not None:
        return JacobianEstimate(
            matrix=model.analytical_jacobian(point),
            source="analytical",
            point=point,
        )
    return finite_difference_jacobian(model, point, method="central")
