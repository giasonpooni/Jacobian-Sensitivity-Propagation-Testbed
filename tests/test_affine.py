from __future__ import annotations

import numpy as np

from sensitivity.affine import AffinePlant, GaussianState, predict, update
from sensitivity.checks import check_filter_chart_equivariance, check_translation_invariance
from sensitivity.coordinates import AffineCoordinates


def _plant():
    state = GaussianState([40.0, 60.0], [[1.0, 0.2], [0.2, 1.5]])
    plant = AffinePlant(
        F=[[0.96, 0.03], [0.04, 0.97]],
        drift=[0.2, -0.2],
        Q=0.01 * np.array([[1.0, -1.0], [-1.0, 1.0]]),
        H=np.eye(2),
        offset=[0.7, -0.4],
        R=[[0.25, 0.08], [0.08, 0.36]],
    )
    return state, plant, np.array([40.9, 59.6])


def test_translation_invariance_of_additive_error():
    result = check_translation_invariance([40.0, 60.0], [41.0, 58.5], [100.0, -20.0])
    assert result.passed, result.details


def test_predict_is_exact_affine_pushforward():
    state, plant, _ = _plant()
    prior = predict(state, plant)
    expected_mean = plant.F @ state.mean + plant.drift
    expected_cov = plant.F @ state.covariance @ plant.F.T + plant.Q
    np.testing.assert_allclose(prior.mean, expected_mean)
    np.testing.assert_allclose(prior.covariance, expected_cov)


def test_chart_with_origin_preserves_posterior_and_nis():
    state, plant, measurement = _plant()
    chart = AffineCoordinates(
        T=np.diag([1000.0, 1000.0]),
        S=np.diag([1000.0, 1000.0]),
        name="grams-datum",
        input_offset=[0.0, 100.0],
        output_offset=[0.0, 100.0],
    )
    result = check_filter_chart_equivariance(state, plant, measurement, chart)
    assert result.passed, result.details


def test_update_reduces_covariance_trace():
    state, plant, measurement = _plant()
    prior = predict(state, plant)
    posterior = update(prior, plant, measurement).posterior
    assert np.trace(posterior.covariance) < np.trace(prior.covariance)
