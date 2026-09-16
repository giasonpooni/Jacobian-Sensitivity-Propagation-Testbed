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

MAX_CONDITION_NUMBER = 1e12
ROUND_TRIP_TOLERANCE = 1e-8


def _require_chart(matrix: ArrayLike, name: str) -> Array:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain finite numbers")
    try:
        condition = float(np.linalg.cond(value))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} condition could not be resolved") from exc
    if not np.isfinite(condition) or condition > MAX_CONDITION_NUMBER:
        raise ValueError(f"{name} must be invertible with condition number <= 1e12")
    identity = np.eye(value.shape[0])
    recovered = np.linalg.solve(value, value @ identity)
    if float(np.max(np.abs(recovered - identity))) > ROUND_TRIP_TOLERANCE:
        raise ValueError(f"{name} loses precision in the coordinate round trip")
    return value


@dataclass(frozen=True)
class AffineCoordinates:
    """Invertible linear maps on the input and output spaces."""

    T: Array
    S: Array
    name: str = "coordinates"

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", _require_chart(self.T, "T"))
        object.__setattr__(self, "S", _require_chart(self.S, "S"))

    @property
    def T_inv(self) -> Array:
        return np.linalg.solve(self.T, np.eye(self.T.shape[0]))

    @property
    def S_inv(self) -> Array:
        return np.linalg.solve(self.S, np.eye(self.S.shape[0]))

    @property
    def T_condition(self) -> float:
        return float(np.linalg.cond(self.T))

    @property
    def S_condition(self) -> float:
        return float(np.linalg.cond(self.S))

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
