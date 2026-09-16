from __future__ import annotations

import numpy as np

from sensitivity.checks import check_coordinate_consistency
from sensitivity.coordinates import AffineCoordinates, transform_jacobian
from sensitivity.jacobian import jacobian_at
from sensitivity.reference_models import reference_catalogue


def test_unit_change_preserves_physical_prediction():
    model = reference_catalogue()["two-tank"]
    x = np.array([10.0, 4.0, 0.9])
    dx = np.array([0.05, -0.02, 0.0])
    coords = AffineCoordinates.scale([1000.0, 1000.0, 1.0], [1000.0, 1000.0], name="milli")
    raw = jacobian_at(model, x).matrix
    primed = transform_jacobian(raw, coords)
    assert not np.allclose(raw, primed)
    result = check_coordinate_consistency(model, x, coords, dx)
    assert result.passed, result.details


def test_permutation_consistency():
    model = reference_catalogue()["affine2"]
    coords = AffineCoordinates.permutation([1, 0], [1, 0], name="swap")
    result = check_coordinate_consistency(model, [0.4, -0.3], coords, [0.1, 0.2])
    assert result.passed, result.details
