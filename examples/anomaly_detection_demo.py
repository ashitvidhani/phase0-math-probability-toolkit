"""
Demo for Multivariate Gaussian + Anomaly Detection Toolkit

Run from project root:

    python -m examples.anomaly_detection_demo
"""

import numpy as np

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
)


def main():
    print("=== Multivariate Gaussian + Anomaly Detection Demo ===")

    true_mean = [0.0, 0.0]

    true_covariance = [
        [1.0, 0.8],
        [0.8, 1.5],
    ]

    samples = sample_multivariate_normal(
        mean=true_mean,
        covariance_matrix=true_covariance,
        samples=1000,
        random_seed=42,
    )

    fitted_model = fit_multivariate_gaussian(samples)

    estimated_mean = fitted_model["mean"]
    estimated_covariance = fitted_model["covariance"]

    print("\nTrue Mean:")
    print(np.array(true_mean))

    print("\nEstimated Mean:")
    print(estimated_mean)

    print("\nTrue Covariance:")
    print(np.array(true_covariance))

    print("\nEstimated Covariance:")
    print(estimated_covariance)

    point = [1.0, 1.0]

    print("\nPoint:")
    print(point)

    print("\nSquared Mahalanobis Distance:")
    print(squared_mahalanobis_distance(point, estimated_mean, estimated_covariance))

    print("\nMahalanobis Distance:")
    print(mahalanobis_distance(point, estimated_mean, estimated_covariance))

    print("\nMultivariate Gaussian PDF at point:")
    print(multivariate_normal_pdf(point, estimated_mean, estimated_covariance))

    normal_points = samples[:10]

    suspicious_points = np.array(
        [
            [4.0, 4.0],
            [-4.0, -3.0],
        ],
        dtype=float,
    )

    data_to_check = np.vstack([normal_points, suspicious_points])

    scores = anomaly_scores(
        data=data_to_check,
        mean=estimated_mean,
        covariance_matrix=estimated_covariance,
    )

    anomalies = detect_anomalies(
        data=data_to_check,
        mean=estimated_mean,
        covariance_matrix=estimated_covariance,
        threshold=3.0,
    )

    print("\nData to Check:")
    print(data_to_check)

    print("\nAnomaly Scores:")
    print(scores)

    print("\nAnomaly Flags:")
    print(anomalies)

    print("\nRows detected as anomalies:")
    print(data_to_check[anomalies])


if __name__ == "__main__":
    main()