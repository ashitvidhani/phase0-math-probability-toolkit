import numpy as np
import pytest

from math_toolkit.linear_algebra import (
    determinant,
    dot_product,
    eigen_analysis,
    inverse,
    matrix_multiply,
    matrix_rank,
    matrix_vector_multiply,
    vector_magnitude,
)


def test_vector_magnitude():
    assert vector_magnitude([3, 4]) == pytest.approx(5.0)


def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)


def test_dot_product_dimension_mismatch():
    with pytest.raises(ValueError):
        dot_product([1, 2], [1, 2, 3])


def test_matrix_vector_multiply():
    A = [
        [1, 2],
        [3, 4],
    ]

    x = [10, 20]

    result = matrix_vector_multiply(A, x)

    expected = np.array([50, 110], dtype=float)

    assert np.allclose(result, expected)


def test_matrix_multiply():
    A = [
        [1, 2],
        [3, 4],
    ]

    B = [
        [5, 6],
        [7, 8],
    ]

    result = matrix_multiply(A, B)

    expected = np.array(
        [
            [19, 22],
            [43, 50],
        ],
        dtype=float,
    )

    assert np.allclose(result, expected)


def test_matrix_rank_full_rank():
    A = [
        [1, 2],
        [3, 4],
    ]

    assert matrix_rank(A) == 2


def test_matrix_rank_rank_deficient():
    A = [
        [1, 2],
        [2, 4],
    ]

    assert matrix_rank(A) == 1


def test_determinant():
    A = [
        [1, 2],
        [3, 4],
    ]

    assert determinant(A) == pytest.approx(-2.0)


def test_inverse():
    A = [
        [1, 2],
        [3, 4],
    ]

    result = inverse(A)

    expected = np.array(
        [
            [-2.0, 1.0],
            [1.5, -0.5],
        ]
    )

    assert np.allclose(result, expected)


def test_inverse_singular_matrix():
    A = [
        [1, 2],
        [2, 4],
    ]

    with pytest.raises(ValueError):
        inverse(A)


def test_eigen_analysis_diagonal_matrix():
    A = [
        [2, 0],
        [0, 3],
    ]

    eigenvalues, eigenvectors = eigen_analysis(A)

    sorted_eigenvalues = sorted(eigenvalues)

    assert np.allclose(sorted_eigenvalues, [2, 3])
    assert eigenvectors.shape == (2, 2)