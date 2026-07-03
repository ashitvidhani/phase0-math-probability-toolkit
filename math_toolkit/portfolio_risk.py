"""
Portfolio Risk Toolkit

This module contains reusable portfolio risk functions for the
Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- Portfolio expected return
- Portfolio variance
- Portfolio volatility
- Covariance to correlation conversion
- Correlation to covariance conversion
- Historical portfolio returns
- Historical portfolio summary
- Variance contribution by asset

Core memory:

Portfolio return:
    R_p = w^T R

Portfolio expected return:
    E[R_p] = w^T μ

Portfolio variance:
    Var(R_p) = w^T Σ w

Where:
    w = portfolio weight vector
    μ = expected return vector
    Σ = covariance matrix
"""

from typing import Dict, Sequence

import numpy as np


def to_vector(values: Sequence[float]) -> np.ndarray:
    """
    Convert a list-like object into a one-dimensional NumPy vector.
    """
    vector = np.array(values, dtype=float)

    if vector.ndim != 1:
        raise ValueError("Input must be a one-dimensional vector.")

    if vector.size == 0:
        raise ValueError("Vector cannot be empty.")

    return vector


def to_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Convert a nested list-like object into a two-dimensional NumPy matrix.
    """
    matrix = np.array(values, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")

    if matrix.size == 0:
        raise ValueError("Matrix cannot be empty.")

    return matrix


def validate_weights(
    weights: Sequence[float],
    allow_short: bool = True,
) -> np.ndarray:
    """
    Validate portfolio weights.

    Portfolio weights should usually sum to 1.

    If allow_short is False:
        negative weights are rejected.

    If allow_short is True:
        negative weights are allowed, representing short positions.
    """
    w = to_vector(weights)

    if not allow_short and np.any(w < 0):
        raise ValueError("Negative weights are not allowed when allow_short=False.")

    if not np.isclose(np.sum(w), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return w


def portfolio_expected_return(
    weights: Sequence[float],
    expected_returns: Sequence[float],
) -> float:
    """
    Calculate portfolio expected return.

    Formula:
        E[R_p] = w^T μ

    Where:
        w = portfolio weights
        μ = expected returns of assets
    """
    w = validate_weights(weights)
    mu = to_vector(expected_returns)

    if w.shape != mu.shape:
        raise ValueError("Weights and expected returns must have the same shape.")

    return float(w @ mu)


def portfolio_variance(
    weights: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate portfolio variance.

    Formula:
        Var(R_p) = w^T Σ w

    Where:
        w = portfolio weights
        Σ = covariance matrix of asset returns
    """
    w = validate_weights(weights)
    sigma = to_matrix(covariance_matrix)

    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if sigma.shape[0] != w.shape[0]:
        raise ValueError("Covariance matrix size must match number of weights.")

    variance = float(w.T @ sigma @ w)

    if variance < 0 and np.isclose(variance, 0.0):
        variance = 0.0

    if variance < 0:
        raise ValueError("Portfolio variance cannot be negative.")

    return variance


def portfolio_volatility(
    weights: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate portfolio volatility.

    Formula:
        volatility = sqrt(portfolio variance)

    Volatility is the standard deviation of portfolio returns.
    """
    variance = portfolio_variance(weights, covariance_matrix)

    return float(np.sqrt(variance))


def portfolio_summary(
    weights: Sequence[float],
    expected_returns: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """
    Return portfolio expected return, variance, and volatility.
    """
    expected_return = portfolio_expected_return(weights, expected_returns)
    variance = portfolio_variance(weights, covariance_matrix)
    volatility = float(np.sqrt(variance))

    return {
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
    }


def covariance_to_correlation(
    covariance_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Convert covariance matrix to correlation matrix.

    Formula:
        Corr(X, Y) = Cov(X, Y) / (σ_X σ_Y)

    Diagonal of correlation matrix should be 1.
    """
    sigma = to_matrix(covariance_matrix)

    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    variances = np.diag(sigma)

    if np.any(variances <= 0):
        raise ValueError("All variances must be positive.")

    standard_deviations = np.sqrt(variances)

    correlation_matrix = sigma / np.outer(standard_deviations, standard_deviations)

    return correlation_matrix


def correlation_to_covariance(
    correlation_matrix: Sequence[Sequence[float]],
    standard_deviations: Sequence[float],
) -> np.ndarray:
    """
    Convert correlation matrix to covariance matrix.

    Formula:
        Cov(X, Y) = Corr(X, Y) * σ_X * σ_Y
    """
    corr = to_matrix(correlation_matrix)
    std = to_vector(standard_deviations)

    if corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    if corr.shape[0] != std.shape[0]:
        raise ValueError(
            "Correlation matrix size must match number of standard deviations."
        )

    if np.any(std <= 0):
        raise ValueError("Standard deviations must be positive.")

    covariance_matrix = corr * np.outer(std, std)

    return covariance_matrix


def portfolio_returns_from_asset_returns(
    asset_returns: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> np.ndarray:
    """
    Calculate historical / simulated portfolio returns from asset returns.

    Input:
        asset_returns:
            rows = observations
            columns = assets

        weights:
            portfolio weights

    Formula:
        portfolio_returns = asset_returns @ weights
    """
    returns = to_matrix(asset_returns)
    w = validate_weights(weights)

    if returns.shape[1] != w.shape[0]:
        raise ValueError("Number of asset return columns must match weights.")

    return returns @ w


def historical_portfolio_summary(
    asset_returns: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> Dict[str, float]:
    """
    Summarize historical / simulated portfolio returns.

    Returns:
        count
        mean_return
        variance
        volatility
        min_return
        max_return
    """
    portfolio_returns = portfolio_returns_from_asset_returns(asset_returns, weights)

    return {
        "count": float(portfolio_returns.size),
        "mean_return": float(np.mean(portfolio_returns)),
        "variance": float(np.var(portfolio_returns)),
        "volatility": float(np.std(portfolio_returns)),
        "min_return": float(np.min(portfolio_returns)),
        "max_return": float(np.max(portfolio_returns)),
    }


def variance_contributions(
    weights: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Calculate each asset's contribution to portfolio variance.

    Formula:
        component_i = w_i * (Σw)_i

    Sum of all components equals portfolio variance:

        sum(contributions) = w^T Σ w
    """
    w = validate_weights(weights)
    sigma = to_matrix(covariance_matrix)

    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if sigma.shape[0] != w.shape[0]:
        raise ValueError("Covariance matrix size must match number of weights.")

    sigma_w = sigma @ w

    contributions = w * sigma_w

    return contributions


def variance_contribution_percentages(
    weights: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Calculate percentage contribution of each asset to total portfolio variance.

    Formula:
        percentage_i = variance_contribution_i / total_portfolio_variance
    """
    contributions = variance_contributions(weights, covariance_matrix)
    total_variance = float(np.sum(contributions))

    if np.isclose(total_variance, 0.0):
        raise ValueError("Total variance is zero; contribution percentages undefined.")

    return contributions / total_variance