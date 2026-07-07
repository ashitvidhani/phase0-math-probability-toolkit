import numpy as np
import pytest

from math_toolkit.probability import (
    normal_pdf,
    normal_cdf,
    sample_normal,
    simulate_binomial,
)

from math_toolkit.monte_carlo import (
    run_monte_carlo,
    estimate_expected_value,
    estimate_event_probability,
    monte_carlo_summary,
    simulate_random_walk,
    simulate_asset_price_paths,
    percentile_value,
    value_at_risk,
)

from math_toolkit.multivariate_gaussian import (
    estimate_mean_vector,
    estimate_covariance_matrix,
    validate_covariance_matrix,
    multivariate_normal_pdf,
    sample_multivariate_normal,
    detect_anomalies,
)


def test_normal_pdf_rejects_zero_standard_deviation():
    with pytest.raises(ValueError):
        normal_pdf(0, standard_deviation=0)


def test_normal_cdf_rejects_zero_standard_deviation():
    with pytest.raises(ValueError):
        normal_cdf(0, standard_deviation=0)


def test_sample_normal_rejects_zero_samples():
    with pytest.raises(ValueError):
        sample_normal(samples=0)


def test_simulate_binomial_rejects_zero_simulations():
    with pytest.raises(ValueError):
        simulate_binomial(trials=10, probability_success=0.5, simulations=0)


def test_run_monte_carlo_rejects_zero_simulations():
    with pytest.raises(ValueError):
        run_monte_carlo(lambda rng: 1.0, simulations=0)


def test_estimate_expected_value_rejects_empty_samples():
    with pytest.raises(ValueError):
        estimate_expected_value([])


def test_estimate_event_probability_rejects_empty_samples():
    with pytest.raises(ValueError):
        estimate_event_probability([], lambda x: x > 0)


def test_monte_carlo_summary_rejects_empty_samples():
    with pytest.raises(ValueError):
        monte_carlo_summary([])


def test_simulate_random_walk_rejects_negative_volatility():
    with pytest.raises(ValueError):
        simulate_random_walk(steps=10, volatility=-1)


def test_simulate_asset_price_paths_rejects_invalid_initial_price():
    with pytest.raises(ValueError):
        simulate_asset_price_paths(
            initial_price=0,
            drift=0.05,
            volatility=0.2,
            time_horizon=1,
            steps=10,
            simulations=100,
        )


def test_percentile_value_rejects_invalid_percentile():
    with pytest.raises(ValueError):
        percentile_value([1, 2, 3], percentile=101)


def test_value_at_risk_rejects_invalid_confidence_level():
    with pytest.raises(ValueError):
        value_at_risk([0.01, -0.02, 0.03], confidence_level=1.0)


def test_estimate_mean_vector_basic_case():
    data = [[1, 2], [3, 4], [5, 6]]
    result = estimate_mean_vector(data)

    assert np.allclose(result, [3, 4])


def test_estimate_covariance_matrix_rejects_single_observation():
    with pytest.raises(ValueError):
        estimate_covariance_matrix([[1, 2]])


def test_validate_covariance_matrix_rejects_non_symmetric_matrix():
    with pytest.raises(ValueError):
        validate_covariance_matrix([[1, 2], [3, 4]])


def test_multivariate_normal_pdf_basic_case():
    density = multivariate_normal_pdf(
        point=[0, 0],
        mean=[0, 0],
        covariance_matrix=[[1, 0], [0, 1]],
    )

    assert density > 0


def test_sample_multivariate_normal_shape():
    samples = sample_multivariate_normal(
        mean=[0, 0],
        covariance_matrix=[[1, 0], [0, 1]],
        samples=10,
        random_seed=42,
    )

    assert samples.shape == (10, 2)


def test_detect_anomalies_returns_boolean_array():
    data = [[0, 0], [1, 1], [10, 10]]
    mean = [0, 0]
    covariance_matrix = [[1, 0], [0, 1]]

    result = detect_anomalies(
        data=data,
        mean=mean,
        covariance_matrix=covariance_matrix,
        threshold=3,
    )

    assert result.dtype == bool
    assert result.shape == (3,)