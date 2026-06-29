"""
Linear Algebra Toolkit

This module contains reusable linear algebra functions for the
Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- Vector Magnitude
- Dot Product
- Matrix-Vector Multiplication
- Matrix Multiplication
- Matrix Rank
- Determinant
- Matrix Inverse
Eigenvalue/ eigenvector analysis
"""

from typing import Sequence, Tuple
import numpy as np

Arraylike = Sequence[float] | Sequence[Sequence[float]]

def to_array(values: Arraylike) -> np.ndarray:
    """
    Convert a python list/tuple into Numpy array of floats.
    Example [1,2,3] becomes np.array([1.0,2.0,3.0])
    """
    return np.array(values, dtype=float)

def vector_magnitude(vector: Sequence[float]) -> float:
    """
    Calculate the magnitude/length of a vector.
    
    Formula:
    ||v|| = sqrt(v1^2+v2^2+v3^2...+vn^2)

    vector_magnitude([3,4]) = 5
    """
    v = to_array(vector)
    if v.ndim !=1:
        raise ValueError("Input must be a 1 dimensional vector")
    
    return float(np.linalg.norm(v))

def dot_product(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
   """
    Calculate the dot product of 2 vectors.

    Formula:
    a · b = a1b1 + a2b2+...+anbn

    The dot product measures alignment/similarity between two vectors.
    """
   a = to_array(vector_a)
   b = to_array(vector_b)

   if a.ndim != 1 or b.ndim != 1:
       raise ValueError("Both input must be a 1 dimensional vector ")
   if a.shape != b.shape:
       raise ValueError("Vectors must have same length")
   
   return float(np.dot(a,b))

def matrix_vector_multiply(matrix: Sequence[Sequence[float]],
                           vector: Sequence[float],) -> np.ndarray:
    """
    Multiply a matrix by a vector
    
    Formula:
    Ax =b
    
    Meaning: 
    Matrix A transforms input vector x into output vector b.
    """
    A = to_array(matrix)
    v = to_array(vector)

    if A.ndim != 2:
        raise ValueError("Matrix input must be two-dimensional.")
    
    if v.ndim != 1:
        raise ValueError("Vector input must be one-dimensional.")
    
    if A.shape[1] != v.shape[0]:
        raise ValueError(
            "Matrix columns must match vector length."
            f"Got matrix shape {A.shape} and vector shape {v.shape}."
            )
    
    return A @ v

def matrix_multiply(
        matrix_a: Sequence[Sequence[float]],
        matrix_b: Sequence[Sequence[float]],
) -> np.ndarray:
    """

    Multiply two matrices.

    Rule:
        If A is m x n and B is n x p,
        then AB is m x p.

    Matrix multiplication means composition of transformations.
    """

    A = to_array(matrix_a)
    B = to_array(matrix_b)

    if A.ndim != 2 or B.ndim !=2:
        raise ValueError("Both inputs must be two-dimensional matrices.")
    
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            "Number of columns of A must match number of rows of B. "
            f"Got A shape {A.shape} and B shape {B.shape}."
        )
    
    return A @ B

def matrix_rank(matrix: Sequence[Sequence[float]]) -> int:
    """
    
    Calculate the rank of a matrix.

    Rank = number of independent directions preserved by the matrix.
    Rank = dimension of column space.
    """

    A = to_array(matrix)

    if A.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")
    
    return int(np.linalg.matrix_rank(A))

def determinant(matrix: Sequence[Sequence[float]]) -> float:
    """
    Calculate determinant of a square matrix.

    Determinant = volume scaling factor.

    If determinant = 0, the matrix is singular and not invertible.
    """

    A = to_array(matrix)

    if A.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Determinant is defined only for square matrices.")
    
    return float(np.linalg.det(A))

def inverse(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Calculate inverse of a square matrix.

    Inverse means undoing the transformation.

    If A is singular, inverse does not exist.
    """
    A = to_array(matrix)

    if A.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Inverse is defined only for square matrices.")

    det = np.linalg.det(A)

    if np.isclose(det, 0.0):
        raise ValueError("Matrix is singular. Inverse does not exist.")

    return np.linalg.inv(A)

def eigen_analysis(
    matrix: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate eigenvalues and eigenvectors of a square matrix.

    Formula:
        Av = lambda v

    Meaning:
        v = eigenvector
        lambda = eigenvalue

    A transforms v by scaling it without changing its direction.
    """
    A = to_array(matrix)

    if A.ndim != 2:
        raise ValueError("Input must be a two-dimensional matrix.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Eigen analysis is defined only for square matrices.")

    eigenvalues, eigenvectors = np.linalg.eig(A)

    return eigenvalues, eigenvectors
