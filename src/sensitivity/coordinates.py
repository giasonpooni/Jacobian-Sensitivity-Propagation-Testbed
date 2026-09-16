"""Invertible affine charts and the induced Jacobian law.

For x' = T x + c and y' = S y + b the transformed Jacobian is

    J' = S J T^{-1}.

Offsets move the origin. They are not automorphisms of (R^n, +).
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
    T: Array
    S: Array
    name: str = "coordinates"
    input_offset: Array | None = None
    output_offset: Array | None = None

    def __post_init__(self) -> None:
        t = _require_chart(self.T, "T")
        s = _require_chart(self.S, "S")
        object.__setattr__(self, "T", t)
        object.__setattr__(self, "S", s)
        cin = np.zeros(t.shape[0]) if self.input_offset is None else as_vector(self.input_offset, "input_offset")
        cout = np.zeros(s.shape[0]) if self.output_offset is None else as_vector(self.output_offset, "output_offset")
        if cin.size != t.shape[0]:
            raise ValueError("input_offset must match T")
        if cout.size != s.shape[0]:
            raise ValueError("output_offset must match S")
        object.__setattr__(self, "input_offset", cin)
        object.__setattr__(self, "output_offset", cout)

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
        return cls(T=np.eye(inp.size)[inp], S=np.eye(out.size)[out], name=name)


def transform_jacobian(jacobian: ArrayLike, coordinates: AffineCoordinates) -> Array:
    jac = np.asarray(jacobian, dtype=float)
    if jac.shape != (coordinates.S.shape[0], coordinates.T.shape[0]):
        raise ValueError(
            f"J shape {jac.shape} incompatible with S {coordinates.S.shape} "
            f"and T {coordinates.T.shape}"
        )
    return coordinates.S @ jac @ coordinates.T_inv


def apply_input_map(x: ArrayLike, coordinates: AffineCoordinates) -> Array:
    return coordinates.T @ as_vector(x, "x") + coordinates.input_offset


def apply_output_map(y: ArrayLike, coordinates: AffineCoordinates) -> Array:
    return coordinates.S @ as_vector(y, "y") + coordinates.output_offset


def invert_input_map(xp: ArrayLike, coordinates: AffineCoordinates) -> Array:
    return coordinates.T_inv @ (as_vector(xp, "x'") - coordinates.input_offset)


def invert_output_map(yp: ArrayLike, coordinates: AffineCoordinates) -> Array:
    return coordinates.S_inv @ (as_vector(yp, "y'") - coordinates.output_offset)


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
        x = invert_input_map(xp, coordinates)
        y = model.evaluate(x)
        return apply_output_map(y, coordinates)

    jac_fn = None
    if model.jacobian is not None:

        def jac_fn(xp: Array) -> Array:  # type: ignore[misc]
            x = invert_input_map(xp, coordinates)
            return transform_jacobian(model.analytical_jacobian(x), coordinates)

    return DifferentiableModel(
        name=name or f"{model.name}[{coordinates.name}]",
        forward=forward,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        jacobian=jac_fn,
        notes=f"re-expressed under {coordinates.name}",
        tags=model.tags + ("transformed",),
    )
