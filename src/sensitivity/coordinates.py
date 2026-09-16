"""Invertible linear coordinate changes and the induced Jacobian law.

For ``x' = T x`` and ``y' = S y`` the transformed Jacobian is

    J' = S J T^{-1}.

Raw matrix entries and unscaled singular values are not invariants.
The invariant is the physical prediction after both the model and the
perturbation have been translated between coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import DifferentiableModel, as_vector

Array = NDArray[np.floating]


@dataclass(frozen=True)
class AffineCoordinates:
    """Invertible linear maps on the input and output spaces."""

    T: Array
    S: Array
    name: str = "coordinates"

    def __post_init__(self) -> None:
        t = np.asarray(self.T, dtype=float)
        s = np.asarray(self.S, dtype=float)
        object.__setattr__(self, "T", t)
        object.__setattr__(self, "S", s)
        if t.ndim != 2 or t.shape[0] != t.shape[1]:
            raise ValueError("T must be a square matrix")
        if s.ndim != 2 or s.shape[0] != s.shape[1]:
            raise ValueError("S must be a square matrix")
        if abs(np.linalg.det(t)) < 1e-14:
            raise ValueError("T must be invertible")
        if abs(np.linalg.det(s)) < 1e-14:
            raise ValueError("S must be invertible")

    @property
    def T_inv(self) -> Array:
        return np.linalg.inv(self.T)

    @property
    def S_inv(self) -> Array:
        return np.linalg.inv(self.S)

    @classmethod
    def scale(cls, input_scales: ArrayLike, output_scales: ArrayLike, name: str = "units") -> AffineCoordinates:
        tin = np.asarray(input_scales, dtype=float)
        tout = np.asarray(output_scales, dtype=float)
        if np.any(tin == 0.0) or np.any(tout == 0.0):
            raise ValueError("scale factors must be nonzero")
        return cls(T=np.diag(tin), S=np.diag(tout), name=name)

    @classmethod
    def permutation(
        cls,
        input_order: ArrayLike,
        output_order: ArrayLike,
        name: str = "permutation",
    ) -> AffineCoordinates:
        inp = np.asarray(input_order, dtype=int)
        out = np.asarray(output_order, dtype=int)
        t = np.eye(inp.size)[inp]
        s = np.eye(out.size)[out]
        return cls(T=t, S=s, name=name)


def transform_jacobian(jacobian: ArrayLike, coordinates: AffineCoordinates) -> Array:
    """``J' = S J T^{-1}``."""
    jac = np.asarray(jacobian, dtype=float)
    if jac.shape != (coordinates.S.shape[0], coordinates.T.shape[0]):
        raise ValueError(
            f"J shape {jac.shape} incompatible with S {coordinates.S.shape} "
            f"and T {coordinates.T.shape}"
        )
    return coordinates.S @ jac @ coordinates.T_inv


def apply_input_map(x: ArrayLike, coordinates: AffineCoordinates) -> Array:
    return coordinates.T @ as_vector(x, "x")


def apply_output_map(y: ArrayLike, coordinates: AffineCoordinates) -> Array:
    return coordinates.S @ as_vector(y, "y")


def transform_model(
    model: DifferentiableModel,
    coordinates: AffineCoordinates,
    *,
    name: str | None = None,
) -> DifferentiableModel:
    """Return the same physical map expressed in primed coordinates.

    ``y' = S f(T^{-1} x')``.
    """
    if coordinates.T.shape[0] != model.input_dim:
        raise ValueError("T does not match model input dimension")
    if coordinates.S.shape[0] != model.output_dim:
        raise ValueError("S does not match model output dimension")

    def forward(xp: Array) -> Array:
        x = coordinates.T_inv @ xp
        y = model.evaluate(x)
        return coordinates.S @ y

    jac_fn = None
    if model.jacobian is not None:

        def jac_fn(xp: Array) -> Array:  # type: ignore[misc]
            x = coordinates.T_inv @ xp
            return transform_jacobian(model.analytical_jacobian(x), coordinates)

    return DifferentiableModel(
        name=name or f"{model.name}[{coordinates.name}]",
        forward=forward,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        jacobian=jac_fn,
        notes=f"re-expressed under {coordinates.name}: y' = S f(T^{{-1}} x')",
        tags=model.tags + ("transformed",),
    )
