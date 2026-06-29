"""
Demo for Linear Algebra Toolkit

Run from project root:

    python -m examples.linear_algebra_demo
"""

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


def main():
    print("=== Linear Algebra Toolkit Demo ===")

    vector = [3, 4]
    print("\nVector:", vector)
    print("Magnitude:", vector_magnitude(vector))

    a = [1, 2, 3]
    b = [4, 5, 6]
    print("\nVector A:", a)
    print("Vector B:", b)
    print("Dot product:", dot_product(a, b))

    A = [
        [1, 2],
        [3, 4],
    ]

    x = [10, 20]

    print("\nMatrix A:")
    print(A)

    print("\nVector x:")
    print(x)

    print("\nA x:")
    print(matrix_vector_multiply(A, x))

    B = [
        [5, 6],
        [7, 8],
    ]

    print("\nMatrix B:")
    print(B)

    print("\nA B:")
    print(matrix_multiply(A, B))

    print("\nRank of A:")
    print(matrix_rank(A))

    print("\nDeterminant of A:")
    print(determinant(A))

    print("\nInverse of A:")
    print(inverse(A))

    eigenvalues, eigenvectors = eigen_analysis(A)

    print("\nEigenvalues of A:")
    print(eigenvalues)

    print("\nEigenvectors of A:")
    print(eigenvectors)


if __name__ == "__main__":
    main()