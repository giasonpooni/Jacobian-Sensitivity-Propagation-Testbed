"""Declared reference maps with known analytical Jacobians."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import DifferentiableModel

Array = NDArray[np.floating]


def affine_map(
    matrix: ArrayLike,
    offset: ArrayLike | None = None,
    *,
    name: str = "affine",
) -> DifferentiableModel:
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2:
        raise ValueError("affine matrix must be 2-D")
    b = np.zeros(a.shape[0], dtype=float) if offset is None else np.asarray(offset, dtype=float)
    if b.shape != (a.shape[0],):
        raise ValueError("offset shape must match the number of rows")

    def forward(x: Array) -> Array:
        return a @ x + b

    def jacobian(_x: Array) -> Array:
        return a.copy()

    return DifferentiableModel(
        name=name,
        forward=forward,
        input_dim=a.shape[1],
        output_dim=a.shape[0],
        jacobian=jacobian,
        notes="affine; first-order perturbation and covariance maps are exact",
        tags=("affine", "analytical"),
    )


def quadratic_form_gradient(*, hess: ArrayLike, grad: ArrayLike, name: str = "quadratic") -> DifferentiableModel:
    """Scalar map ``f(x) = 0.5 x^T H x + g^T x``.

    The remainder of the linear approximation is exactly ``0.5 dx^T H dx``.
    """
    h = np.asarray(hess, dtype=float)
    g = np.asarray(grad, dtype=float)
    h = 0.5 * (h + h.T)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("H must be square")
    if g.shape != (h.shape[0],):
        raise ValueError("g must match H")

    def forward(x: Array) -> Array:
        return np.array([0.5 * float(x @ h @ x) + float(g @ x)])

    def jacobian(x: Array) -> Array:
        return (h @ x + g).reshape(1, -1)

    return DifferentiableModel(
        name=name,
        forward=forward,
        input_dim=h.shape[0],
        output_dim=1,
        jacobian=jacobian,
        notes="scalar quadratic; linear remainder is 0.5 dx^T H dx",
        tags=("quadratic", "analytical", "nonlinear"),
    )


def componentwise_exp(*, dim: int = 2, name: str = "exp") -> DifferentiableModel:
    def forward(x: Array) -> Array:
        return np.exp(x)

    def jacobian(x: Array) -> Array:
        return np.diag(np.exp(x))

    return DifferentiableModel(
        name=name,
        forward=forward,
        input_dim=dim,
        output_dim=dim,
        jacobian=jacobian,
        notes="componentwise exponential; holomorphic on R^n",
        tags=("nonlinear", "analytical", "holomorphic"),
    )


def polar_from_cartesian(*, name: str = "polar") -> DifferentiableModel:
    """``(x, y) -> (r, theta)`` on the right half-plane."""

    def forward(vec: Array) -> Array:
        x, y = vec
        return np.array([np.hypot(x, y), np.arctan2(y, x)])

    def jacobian(vec: Array) -> Array:
        x, y = vec
        r2 = x * x + y * y
        r = np.sqrt(r2)
        if r2 <= 0.0:
            raise ValueError("polar Jacobian is undefined at the origin")
        return np.array([[x / r, y / r], [-y / r2, x / r2]])

    return DifferentiableModel(
        name=name,
        forward=forward,
        input_dim=2,
        output_dim=2,
        jacobian=jacobian,
        input_names=("x", "y"),
        output_names=("r", "theta"),
        notes="undefined at the origin; angle branch is atan2",
        tags=("nonlinear", "analytical", "coordinates"),
    )


def scaled_rotation(angle: float = np.pi / 5, scale: float = 1.5, *, name: str = "scaled-rotation") -> DifferentiableModel:
    c, s = np.cos(angle), np.sin(angle)
    matrix = scale * np.array([[c, -s], [s, c]])
    return affine_map(matrix, name=name)


def two_tank_mass_balance(*, name: str = "two-tank") -> DifferentiableModel:
    """Tiny physical composition used as a portable example.

    Inputs are two tank masses and a shared density scale. Outputs are
    total mass and a density-weighted imbalance. The map is affine in the
    masses at fixed density and nonlinear once density is an input.
    """

    def forward(vec: Array) -> Array:
        m1, m2, density = vec
        total = m1 + m2
        imbalance = density * (m1 - m2)
        return np.array([total, imbalance])

    def jacobian(vec: Array) -> Array:
        _m1, _m2, density = vec
        return np.array(
            [
                [1.0, 1.0, 0.0],
                [density, -density, vec[0] - vec[1]],
            ]
        )

    return DifferentiableModel(
        name=name,
        forward=forward,
        input_dim=3,
        output_dim=2,
        jacobian=jacobian,
        input_names=("m1", "m2", "density"),
        output_names=("total", "imbalance"),
        notes="illustrative plant fragment, not a closed fluid model",
        tags=("analytical", "physical-example"),
    )


def reference_catalogue() -> dict[str, DifferentiableModel]:
    return {
        "affine2": affine_map([[2.0, 0.5], [0.0, 1.5]], [0.1, -0.2], name="affine2"),
        "quadratic": quadratic_form_gradient(
            hess=[[2.0, 0.4], [0.4, 1.2]],
            grad=[0.3, -0.1],
        ),
        "exp2": componentwise_exp(dim=2),
        "polar": polar_from_cartesian(),
        "scaled-rotation": scaled_rotation(),
        "two-tank": two_tank_mass_balance(),
    }
