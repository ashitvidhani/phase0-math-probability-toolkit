"""

Matrix Decomposition Toolkit

This module contains reusable matrix decomposition functions
for the Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- LU decomposition
- QR decomposition
- Singular Value Decomposition
- Low-rank approximation
- PCA-style variance analysis

Core memory:

LU  = factor matrix into lower and upper triangular parts
QR  = factor matrix into orthogonal directions and triangular structure
SVD = rotate -> scale -> rotate
PCA = find directions of maximum variance in data
"""

from typing import Sequence,Tuple

import numpy as np
from scipy.linalg import lu

Matrixlike = Sequence[Sequence[float]]

def to_matrix(values: Matrixlike) -> np.ndarray:
    """
    Convert a Python nested list/tuple into a NumPy matrix of floats.
    """
    matrix = np.array(values, dtype=float)

    if matrix.ndim != 2:
        raise ValueError ("Input must be two-dimensional matrix.")
    
    return matrix

def lu_decomposition(matrix: Matrixlike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU decomposition.

    For a matrix A, SciPy returns:

        P, L, U

    such that:

        A = P @ L @ U

    Where:
        P = permutation matrix
        L = lower triangular matrix
        U = upper triangular matrix

    LU is useful for efficiently solving systems of linear equations.
    """
    A = to_matrix(matrix)

    P, L, U = lu(A)

    return P, L, U

def qr_decomposition(matrix: Matrixlike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform QR decomposition.

    For a matrix A:

        A = Q @ R

    Where:
        Q = orthogonal matrix
        R = upper triangular matrix

    QR is useful for:
        - least squares
        - numerical stability
        - orthogonal basis construction
    """
    A = to_matrix(matrix)

    Q, R = np.linalg.qr(A)

    return Q, R

def svd_decomposition(
        matrix: Matrixlike,
        full_matrices : bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Singular Value Decomposition.

    For a matrix A:

        A = U @ Sigma @ Vt

    Where:
        U = output directions
        singular_values = strength of each direction
        Vt = transposed input directions

    NumPy returns singular values as a 1D array, not as a full diagonal matrix.
    """
    A = to_matrix(matrix)

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=full_matrices)

    return U, singular_values, Vt

def reconstruct_from_svd(
    U: np.ndarray,
    singular_values: Sequence[float],
    Vt: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct a matrix from SVD components.

    If:
        A = U @ Sigma @ Vt

    then this function rebuilds A from U, singular values, and Vt.
    """
    s = np.array(singular_values, dtype=float)

    Sigma = np.diag(s)

    return U @ Sigma @ Vt

def low_rank_approximation(matrix: Matrixlike, rank: int) -> np.ndarray:
    """
    Create a low-rank approximation of a matrix using SVD.

    Idea:
        Keep only the strongest singular values.

    This is useful for:
        - compression
        - denoising
        - dimensionality reduction
        - factor models

    Example:
        rank = 1 keeps only the strongest direction.
    """
    A = to_matrix(matrix)

    if rank <= 0:
        raise ValueError("Rank must be positive.")

    max_rank = min(A.shape)

    if rank > max_rank:
        raise ValueError(f"Rank cannot exceed min(matrix shape) = {max_rank}.")

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

    U_k = U[:, :rank]
    s_k = singular_values[:rank]
    Vt_k = Vt[:rank, :]

    return U_k @ np.diag(s_k) @ Vt_k


def pca_variance_analysis(
    data: Matrixlike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA-style variance analysis using SVD.

    Input:
        data = rows are observations, columns are features

    Output:
        components = principal directions
        explained_variance = variance captured by each direction
        explained_variance_ratio = percentage of total variance captured

    Important:
        This function centers the data first.

    PCA memory:
        PCA finds directions where the data varies the most.
    """
    X = to_matrix(data)

    if X.shape[0] < 2:
        raise ValueError("PCA variance analysis needs at least two observations.")

    centered_X = X - np.mean(X, axis=0)

    U, singular_values, Vt = np.linalg.svd(centered_X, full_matrices=False)

    explained_variance = (singular_values**2) / (X.shape[0] - 1)

    total_variance = np.sum(explained_variance)

    if np.isclose(total_variance, 0.0):
        explained_variance_ratio = np.zeros_like(explained_variance)
    else:
        explained_variance_ratio = explained_variance / total_variance

    components = Vt

    return components, explained_variance, explained_variance_ratio