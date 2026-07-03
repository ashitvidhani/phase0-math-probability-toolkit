import numpy as np
import pytest

from math_toolkit.multivariate_gaussian import (
    anomaly_scores,
    detect_anomalies,
    estimate_covariance_matrix,
    estimate_mean_vector,
    fit_multivariate_gaussian,
    mahalanobis_distance,
    multivariate_normal_pdf,
    sample_multivariate_normal,
    squared_mahalanobis_distance,
    validate_covariance_matrix,
)


def test_validate_covariance_matrix_accepts_valid_matrix():
    covariance_matrix = [
        [1.0, 0.2],
        [0.2, 1.0],
    ]

    result = validate_covariance_matrix(covariance_matrix)

    assert result.shape == (2, 2)


def test_validate_covariance_matrix_rejects_non_square_matrix():
    covariance_matrix = [
        [1.0, 0.2, 0.1],
        [0.2, 1.0, 0.3],
    ]

    with pytest.raises(ValueError):
        validate_covariance_matrix(covariance_matrix)


def test_validate_covariance_matrix_rejects_non_symmetric_matrix():
    covariance_matrix = [
        [1.0, 0.8],
        [0.2, 1.0],
    ]

    with pytest.raises(ValueError):
        validate_covariance_matrix(covariance_matrix)


def test_estimate_mean_vector():
    data = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    result = estimate_mean_vector(data)

    assert np.allclose(result, [3, 4])


def test_estimate_covariance_matrix():
    data = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    result = estimate_covariance_matrix(data)

    expected = np.array(
        [
            [4.0, 4.0],
            [4.0, 4.0],
        ]
    )

    assert np.allclose(result, expected)


def test_fit_multivariate_gaussian_returns_mean_and_covariance():
    data = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    model = fit_multivariate_gaussian(data)

    assert "mean" in model
    assert "covariance" in model
    assert model["mean"].shape == (2,)
    assert model["covariance"].shape == (2, 2)


def test_squared_mahalanobis_distance_identity_covariance():
    point = [3, 4]
    mean = [0, 0]
    covariance_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = squared_mahalanobis_distance(point, mean, covariance_matrix)

    assert result == pytest.approx(25.0)


def test_mahalanobis_distance_identity_covariance():
    point = [3, 4]
    mean = [0, 0]
    covariance_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = mahalanobis_distance(point, mean, covariance_matrix)

    assert result == pytest.approx(5.0)


def test_mahalanobis_distance_zero_at_mean():
    point = [1, 2]
    mean = [1, 2]
    covariance_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = mahalanobis_distance(point, mean, covariance_matrix)

    assert result == pytest.approx(0.0)


def test_multivariate_normal_pdf_standard_2d_at_mean():
    point = [0, 0]
    mean = [0, 0]
    covariance_matrix = [
        [1, 0],
        [0, 1],
    ]

    result = multivariate_normal_pdf(point, mean, covariance_matrix)

    expected = 1 / (2 * np.pi)

    assert result == pytest.approx(expected)


def test_multivariate_normal_pdf_rejects_singular_covariance():
    point = [0, 0]
    mean = [0, 0]
    covariance_matrix = [
        [1, 1],
        [1, 1],
    ]

    with pytest.raises(ValueError):
        multivariate_normal_pdf(point, mean, covariance_matrix)


def test_sample_multivariate_normal_shape():
    samples = sample_multivariate_normal(
        mean=[0, 0],
        covariance_matrix=[
            [1, 0.2],
            [0.2, 1],
        ],
        samples=100,
        random_seed=42,
    )

    assert samples.shape == (100, 2)


def test_sample_multivariate_normal_rejects_invalid_sample_count():
    with pytest.raises(ValueError):
        sample_multivariate_normal(
            mean=[0, 0],
            covariance_matrix=[
                [1, 0],
                [0, 1],
            ],
            samples=0,
        )


def test_anomaly_scores_shape():
    data = [
        [0, 0],
        [1, 1],
        [4, 4],
    ]

    mean = [0, 0]

    covariance_matrix = [
        [1, 0],
        [0, 1],
    ]

    scores = anomaly_scores(data, mean, covariance_matrix)

    assert scores.shape == (3,)


def test_detect_anomalies():
    data = [
        [0, 0],
        [1, 1],
        [4, 4],
    ]

    mean = [0, 0]

    covariance_matrix = [
        [1, 0],
        [0, 1],
    ]

    anomalies = detect_anomalies(
        data=data,
        mean=mean,
        covariance_matrix=covariance_matrix,
        threshold=3.0,
    )

    expected = np.array([False, False, True])

    assert np.array_equal(anomalies, expected)