"""Reusable first-order sensitivity core for the Jacobian testbed.

This package is the importable runtime fragment. Experiments, reference
models, and reports live beside it in the repository; consuming projects
can depend on ``sensitivity`` without adopting the experiment suite.
"""

from .checks import (
    CheckResult,
    check_composition,
    check_coordinate_consistency,
    check_covariance_affine,
    check_derivative,
    check_metric_consistency,
)
from .composition import compose
from .coordinates import (
    AffineCoordinates,
    apply_input_map,
    apply_output_map,
    transform_jacobian,
)
from .covariance import (
    CovarianceExperiment,
    first_order_covariance,
    monte_carlo_covariance,
    run_covariance_experiment,
)
from .jacobian import (
    JacobianEstimate,
    finite_difference_jacobian,
    jacobian_at,
    jax_is_available,
    jvp,
    vjp,
)
from .metrics import DeclaredMetric
from .models import DifferentiableModel
from .perturbation import (
    PerturbationSweep,
    local_validity,
    perturbation_error,
    sweep_perturbation_scale,
)
from .reference_models import reference_catalogue

__all__ = [
    "AffineCoordinates",
    "CheckResult",
    "CovarianceExperiment",
    "DeclaredMetric",
    "DifferentiableModel",
    "JacobianEstimate",
    "PerturbationSweep",
    "apply_input_map",
    "apply_output_map",
    "check_composition",
    "check_coordinate_consistency",
    "check_covariance_affine",
    "check_derivative",
    "check_metric_consistency",
    "compose",
    "finite_difference_jacobian",
    "first_order_covariance",
    "jacobian_at",
    "jax_is_available",
    "jvp",
    "local_validity",
    "monte_carlo_covariance",
    "perturbation_error",
    "reference_catalogue",
    "run_covariance_experiment",
    "sweep_perturbation_scale",
    "transform_jacobian",
    "vjp",
]

__version__ = "0.1.0"
