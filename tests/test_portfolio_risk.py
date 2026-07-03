import numpy as np
import pytest

from math_toolkit.portfolio_risk import (
    correlation_to_covariance,
    covariance_to_correlation,
    historical_portfolio_summary,
    portfolio_expected_return,
    portfolio_returns_from_asset_returns,
    portfolio_summary,
    portfolio_variance,
    portfolio_volatility,
    validate_weights,
    variance_contribution_percentages,
    variance_contributions,
)


def test_validate_weights_accepts_sum_one():
    weights = [0.6, 0.4]

    result = validate_weights(weights)

    assert np.allclose(result, np.array(weights, dtype=float))


def test_validate_weights_rejects_sum_not_one():
    with pytest.raises(ValueError):
        validate_weights([0.6, 0.5])


def test_validate_weights_rejects_negative_when_short_not_allowed():
    with pytest.raises(ValueError):
        validate_weights([1.2, -0.2], allow_short=False)


def test_portfolio_expected_return():
    weights = [0.6, 0.4]
    expected_returns = [0.10, 0.05]

    result = portfolio_expected_return(weights, expected_returns)

    assert result == pytest.approx(0.08)


def test_portfolio_expected_return_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        portfolio_expected_return([0.6, 0.4], [0.1, 0.05, 0.02])


def test_portfolio_variance():
    weights = [0.6, 0.4]

    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    result = portfolio_variance(weights, covariance_matrix)

    assert result == pytest.approx(0.03168)


def test_portfolio_variance_rejects_non_square_covariance():
    weights = [0.6, 0.4]

    covariance_matrix = [
        [0.04, 0.006, 0.01],
        [0.006, 0.09, 0.02],
    ]

    with pytest.raises(ValueError):
        portfolio_variance(weights, covariance_matrix)


def test_portfolio_volatility():
    weights = [0.6, 0.4]

    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    result = portfolio_volatility(weights, covariance_matrix)

    assert result == pytest.approx(np.sqrt(0.03168))


def test_portfolio_summary():
    weights = [0.6, 0.4]
    expected_returns = [0.10, 0.05]

    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    summary = portfolio_summary(weights, expected_returns, covariance_matrix)

    assert summary["expected_return"] == pytest.approx(0.08)
    assert summary["variance"] == pytest.approx(0.03168)
    assert summary["volatility"] == pytest.approx(np.sqrt(0.03168))


def test_covariance_to_correlation():
    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    result = covariance_to_correlation(covariance_matrix)

    expected = np.array(
        [
            [1.0, 0.1],
            [0.1, 1.0],
        ]
    )

    assert np.allclose(result, expected)


def test_correlation_to_covariance():
    correlation_matrix = [
        [1.0, 0.1],
        [0.1, 1.0],
    ]

    standard_deviations = [0.20, 0.30]

    result = correlation_to_covariance(correlation_matrix, standard_deviations)

    expected = np.array(
        [
            [0.04, 0.006],
            [0.006, 0.09],
        ]
    )

    assert np.allclose(result, expected)


def test_portfolio_returns_from_asset_returns():
    asset_returns = [
        [0.01, 0.02],
        [-0.01, 0.00],
        [0.03, 0.01],
    ]

    weights = [0.6, 0.4]

    result = portfolio_returns_from_asset_returns(asset_returns, weights)

    expected = np.array(
        [
            0.014,
            -0.006,
            0.022,
        ]
    )

    assert np.allclose(result, expected)


def test_historical_portfolio_summary():
    asset_returns = [
        [0.01, 0.02],
        [-0.01, 0.00],
        [0.03, 0.01],
    ]

    weights = [0.6, 0.4]

    portfolio_returns = np.array([0.014, -0.006, 0.022])

    summary = historical_portfolio_summary(asset_returns, weights)

    assert summary["count"] == pytest.approx(3.0)
    assert summary["mean_return"] == pytest.approx(np.mean(portfolio_returns))
    assert summary["variance"] == pytest.approx(np.var(portfolio_returns))
    assert summary["volatility"] == pytest.approx(np.std(portfolio_returns))
    assert summary["min_return"] == pytest.approx(np.min(portfolio_returns))
    assert summary["max_return"] == pytest.approx(np.max(portfolio_returns))


def test_variance_contributions_sum_to_portfolio_variance():
    weights = [0.6, 0.4]

    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    contributions = variance_contributions(weights, covariance_matrix)
    variance = portfolio_variance(weights, covariance_matrix)

    assert np.sum(contributions) == pytest.approx(variance)


def test_variance_contribution_percentages_sum_to_one():
    weights = [0.6, 0.4]

    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    percentages = variance_contribution_percentages(weights, covariance_matrix)

    assert np.sum(percentages) == pytest.approx(1.0)