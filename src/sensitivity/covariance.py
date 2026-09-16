"""First-order covariance propagation and Monte Carlo comparison.

The map

    Sigma_y ~ J(mu_x) Sigma_x J(mu_x)^T

is exact for an affine model with the stated covariance. For a nonlinear
model it inherits the same local limitation as ``dy ~ J dx``. The testbed
measures that gap; it does not treat the first-order formula as a second
interpretation of covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .jacobian import jacobian_at
from .models import DifferentiableModel, as_vector

Array = NDArray[np.floating]


def _require_spd(matrix: Array, name: str) -> Array:
    cov = np.asarray(matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eig = np.linalg.eigvalsh(cov)
    if np.any(eig < -1e-10):
        raise ValueError(f"{name} must be positive semidefinite")
    return 0.5 * (cov + cov.T)


def first_order_covariance(jacobian: ArrayLike, sigma_x: ArrayLike) -> Array:
    jac = np.asarray(jacobian, dtype=float)
    cov = _require_spd(np.asarray(sigma_x, dtype=float), "Sigma_x")
    if jac.ndim != 2 or jac.shape[1] != cov.shape[0]:
        raise ValueError(f"J is {jac.shape}, Sigma_x is {cov.shape}")
    return jac @ cov @ jac.T


def monte_carlo_covariance(
    model: DifferentiableModel,
    mean: ArrayLike,
    sigma_x: ArrayLike,
    *,
    samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> Array:
    mu = as_vector(mean, "mean")
    cov = _require_spd(np.asarray(sigma_x, dtype=float), "Sigma_x")
    if mu.size != model.input_dim or cov.shape[0] != model.input_dim:
        raise ValueError("mean and Sigma_x must match the model input dimension")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    engine = rng or np.random.default_rng(0)
    draws = engine.multivariate_normal(mu, cov, size=samples)
    outputs = np.vstack([model.evaluate(row) for row in draws])
    centered = outputs - outputs.mean(axis=0)
    return (centered.T @ centered) / (samples - 1)


@dataclass(frozen=True)
class CovarianceExperiment:
    model: str
    first_order: Array
    monte_carlo: Array
    frobenius_gap: float
    relative_gap: float
    samples: int
    notes: str


def run_covariance_experiment(
    model: DifferentiableModel,
    mean: ArrayLike,
    sigma_x: ArrayLike,
    *,
    samples: int = 4000,
    source: str = "auto",
    rng: np.random.Generator | None = None,
) -> CovarianceExperiment:
    mu = as_vector(mean, "mean")
    jac = jacobian_at(model, mu, source=source)  # type: ignore[arg-type]
    linear = first_order_covariance(jac.matrix, sigma_x)
    empirical = monte_carlo_covariance(
        model, mu, sigma_x, samples=samples, rng=rng
    )
    gap = linear - empirical
    fro = float(np.linalg.norm(gap, ord="fro"))
    denom = max(float(np.linalg.norm(empirical, ord="fro")), 1e-16)
    return CovarianceExperiment(
        model=model.name,
        first_order=linear,
        monte_carlo=empirical,
        frobenius_gap=fro,
        relative_gap=fro / denom,
        samples=samples,
        notes=(
            "First-order covariance is exact for affine maps. "
            "The Monte Carlo comparison measures the local remainder."
        ),
    )
