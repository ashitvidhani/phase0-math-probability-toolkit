"""
Demo for Matrix Decomposition Toolkit

Run from project root:

    python -m examples.decompositions_demo
"""

import numpy as np

from math_toolkit.decompositions import (
    low_rank_approximation,
    lu_decomposition,
    pca_variance_analysis,
    qr_decomposition,
    reconstruct_from_svd,
    svd_decomposition,
)


def main():
    print("=== Matrix Decomposition Toolkit Demo ===")

    A = [
        [1, 2],
        [3, 4],
    ]

    print("\nMatrix A:")
    print(np.array(A, dtype=float))

    P, L, U = lu_decomposition(A)

    print("\nLU Decomposition")
    print("P:")
    print(P)
    print("L:")
    print(L)
    print("U:")
    print(U)
    print("Reconstructed A from P @ L @ U:")
    print(P @ L @ U)

    Q, R = qr_decomposition(A)

    print("\nQR Decomposition")
    print("Q:")
    print(Q)
    print("R:")
    print(R)
    print("Reconstructed A from Q @ R:")
    print(Q @ R)

    U_svd, singular_values, Vt = svd_decomposition(A)

    print("\nSVD")
    print("U:")
    print(U_svd)
    print("Singular values:")
    print(singular_values)
    print("Vt:")
    print(Vt)
    print("Reconstructed A from SVD:")
    print(reconstruct_from_svd(U_svd, singular_values, Vt))

    B = [
        [5, 5, 5],
        [5, 5, 5],
        [1, 1, 1],
    ]

    print("\nMatrix B:")
    print(np.array(B, dtype=float))

    print("\nRank-1 approximation of B:")
    print(low_rank_approximation(B, rank=1))

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

    print("\nPCA Variance Analysis")
    print("Principal components:")
    print(components)
    print("Explained variance:")
    print(explained_variance)
    print("Explained variance ratio:")
    print(explained_variance_ratio)
    print("Explained variance ratio percentage:")
    print(explained_variance_ratio * 100)


if __name__ == "__main__":
    main()