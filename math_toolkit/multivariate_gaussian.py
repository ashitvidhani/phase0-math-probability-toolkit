"""
Multivariate Gaussian Toolkit

This module contains reusable multivariate Gaussian and anomaly detection
functions for the Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- Mean vector estimation
- Covariance matrix estimation
- Multivariate Gaussian PDF
- Multivariate Gaussian sampling
- Mahalanobis distance
- Anomaly scoring
- Anomaly detection

Core memory:

Multivariate Gaussian:
    X ~ N(μ, Σ)

Where:
    X = random vector
    μ = mean vector
    Σ = covariance matrix

Mahalanobis distance:
    d(x, μ) = sqrt((x - μ)^T Σ^-1 (x - μ))

Meaning:
    distance from mean adjusted for variance and covariance
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


def validate_covariance_matrix(
    covariance_matrix: Sequence[Sequence[float]],
    dimension: int | None = None,
    require_positive_definite: bool = False,
) -> np.ndarray:
    """
    Validate a covariance matrix.

    A covariance matrix should be:
        1. square
        2. symmetric
        3. positive semi-definite

    For multivariate Gaussian PDF, we require positive definite covariance,
    because the inverse and determinant must behave properly.
    """
    sigma = to_matrix(covariance_matrix)

    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    if dimension is not None and sigma.shape[0] != dimension:
        raise ValueError("Covariance matrix dimension does not match vector dimension.")

    if not np.allclose(sigma, sigma.T):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(sigma)

    if require_positive_definite:
        if np.any(eigenvalues <= 0):
            raise ValueError("Covariance matrix must be positive definite.")
    else:
        if np.any(eigenvalues < -1e-10):
            raise ValueError("Covariance matrix must be positive semi-definite.")

    return sigma


def estimate_mean_vector(data: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Estimate mean vector from data.

    Input:
        rows = observations
        columns = variables / features

    Output:
        mean vector μ
    """
    X = to_matrix(data)

    if X.shape[0] < 1:
        raise ValueError("Data must contain at least one observation.")

    return np.mean(X, axis=0)


def estimate_covariance_matrix(data: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Estimate sample covariance matrix from data.

    Input:
        rows = observations
        columns = variables / features

    Output:
        sample covariance matrix Σ

    Uses sample covariance with denominator n - 1.
    """
    X = to_matrix(data)

    if X.shape[0] < 2:
        raise ValueError("At least two observations are required to estimate covariance.")

    covariance = np.cov(X, rowvar=False, ddof=1)

    return np.atleast_2d(covariance)


def fit_multivariate_gaussian(
    data: Sequence[Sequence[float]],
) -> Dict[str, np.ndarray]:
    """
    Fit a multivariate Gaussian model to data.

    Output:
        {
            "mean": mean vector,
            "covariance": covariance matrix
        }
    """
    mean = estimate_mean_vector(data)
    covariance = estimate_covariance_matrix(data)

    return {
        "mean": mean,
        "covariance": covariance,
    }


def squared_mahalanobis_distance(
    point: Sequence[float],
    mean: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate squared Mahalanobis distance.

    Formula:
        d^2 = (x - μ)^T Σ^-1 (x - μ)

    This measures distance from the mean adjusted for covariance.
    """
    x = to_vector(point)
    mu = to_vector(mean)

    if x.shape != mu.shape:
        raise ValueError("Point and mean must have the same shape.")

    sigma = validate_covariance_matrix(
        covariance_matrix,
        dimension=x.shape[0],
    )

    difference = x - mu

    inverse_sigma = np.linalg.pinv(sigma)

    distance_squared = difference.T @ inverse_sigma @ difference

    return float(distance_squared)


def mahalanobis_distance(
    point: Sequence[float],
    mean: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate Mahalanobis distance.

    Formula:
        d = sqrt((x - μ)^T Σ^-1 (x - μ))
    """
    distance_squared = squared_mahalanobis_distance(
        point=point,
        mean=mean,
        covariance_matrix=covariance_matrix,
    )

    return float(np.sqrt(distance_squared))


def multivariate_normal_pdf(
    point: Sequence[float],
    mean: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate multivariate normal probability density at a point.

    Formula:
        f(x) =
        1 / sqrt((2π)^k |Σ|)
        *
        exp(-0.5 * (x - μ)^T Σ^-1 (x - μ))

    Important:
        This returns density, not direct probability at a point.
    """
    x = to_vector(point)
    mu = to_vector(mean)

    if x.shape != mu.shape:
        raise ValueError("Point and mean must have the same shape.")

    dimension = x.shape[0]

    sigma = validate_covariance_matrix(
        covariance_matrix,
        dimension=dimension,
        require_positive_definite=True,
    )

    determinant = np.linalg.det(sigma)

    if determinant <= 0:
        raise ValueError("Covariance matrix determinant must be positive.")

    distance_squared = squared_mahalanobis_distance(x, mu, sigma)

    normalization = 1 / np.sqrt(((2 * np.pi) ** dimension) * determinant)

    density = normalization * np.exp(-0.5 * distance_squared)

    return float(density)


def sample_multivariate_normal(
    mean: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
    samples: int = 1000,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples from a multivariate Gaussian distribution.

    Output shape:
        samples x number_of_variables
    """
    mu = to_vector(mean)

    if samples <= 0:
        raise ValueError("Number of samples must be positive.")

    sigma = validate_covariance_matrix(
        covariance_matrix,
        dimension=mu.shape[0],
    )

    rng = np.random.default_rng(random_seed)

    return rng.multivariate_normal(
        mean=mu,
        cov=sigma,
        size=samples,
    )


def anomaly_scores(
    data: Sequence[Sequence[float]],
    mean: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """
    Calculate Mahalanobis anomaly scores for each observation.

    Higher score = farther from normal region.
    """
    X = to_matrix(data)
    mu = to_vector(mean)

    if X.shape[1] != mu.shape[0]:
        raise ValueError("Data column count must match mean vector dimension.")

    scores = [
        mahalanobis_distance(row, mu, covariance_matrix)
        for row in X
    ]

    return np.array(scores, dtype=float)


def detect_anomalies(
    data: Sequence[Sequence[float]],
    mean: Sequence[float],
    covariance_matrix: Sequence[Sequence[float]],
    threshold: float = 3.0,
) -> np.ndarray:
    """
    Detect anomalies using Mahalanobis distance.

    Rule:
        anomaly = score > threshold

    Common intuition:
        threshold around 3 means points roughly more than 3 covariance-adjusted
        standard-distance units away from the mean.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")

    scores = anomaly_scores(
        data=data,
        mean=mean,
        covariance_matrix=covariance_matrix,
    )

    return scores > threshold