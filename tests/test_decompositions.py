import numpy as np
import pytest

from math_toolkit.decompositions import (
    low_rank_approximation,
    lu_decomposition,
    pca_variance_analysis,
    qr_decomposition,
    reconstruct_from_svd,
    svd_decomposition,
)


def test_lu_decomposition_reconstructs_matrix():
    A = [
        [1, 2],
        [3, 4],
    ]

    P, L, U = lu_decomposition(A)

    reconstructed = P @ L @ U

    assert np.allclose(reconstructed, np.array(A, dtype=float))


def test_qr_decomposition_reconstructs_matrix():
    A = [
        [1, 2],
        [3, 4],
    ]

    Q, R = qr_decomposition(A)

    reconstructed = Q @ R

    assert np.allclose(reconstructed, np.array(A, dtype=float))


def test_qr_q_is_orthogonal():
    A = [
        [1, 2],
        [3, 4],
    ]

    Q, R = qr_decomposition(A)

    identity = np.eye(Q.shape[1])

    assert np.allclose(Q.T @ Q, identity)


def test_svd_decomposition_reconstructs_matrix():
    A = [
        [1, 2],
        [3, 4],
    ]

    U, singular_values, Vt = svd_decomposition(A)

    reconstructed = reconstruct_from_svd(U, singular_values, Vt)

    assert np.allclose(reconstructed, np.array(A, dtype=float))


def test_svd_singular_values_are_non_negative():
    A = [
        [1, 2],
        [3, 4],
    ]

    U, singular_values, Vt = svd_decomposition(A)

    assert np.all(singular_values >= 0)


def test_low_rank_approximation_shape():
    A = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    approx = low_rank_approximation(A, rank=1)

    assert approx.shape == (3, 3)


def test_low_rank_approximation_invalid_rank_zero():
    A = [
        [1, 2],
        [3, 4],
    ]

    with pytest.raises(ValueError):
        low_rank_approximation(A, rank=0)


def test_low_rank_approximation_invalid_rank_too_high():
    A = [
        [1, 2],
        [3, 4],
    ]

    with pytest.raises(ValueError):
        low_rank_approximation(A, rank=3)


def test_pca_variance_analysis_outputs_valid_shapes():
    data = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ]

    components, explained_variance, explained_variance_ratio = pca_variance_analysis(data)

    assert components.shape == (2, 2)
    assert explained_variance.shape == (2,)
    assert explained_variance_ratio.shape == (2,)


def test_pca_variance_ratio_sums_to_one():
    data = [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ]

    components, explained_variance, explained_variance_ratio = pca_variance_analysis(data)

    assert np.sum(explained_variance_ratio) == pytest.approx(1.0)


def test_pca_variance_analysis_requires_two_observations():
    data = [
        [1, 2],
    ]

    with pytest.raises(ValueError):
        pca_variance_analysis(data)