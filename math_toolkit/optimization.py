"""
Optimization Toolkit

This module contains reusable optimization functions for the
Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- Numerical derivative
- Numerical gradient
- 1D gradient descent
- Multi-variable gradient descent
- Mean squared error
- Linear regression loss

Core memory:

Derivative = slope / sensitivity
Gradient = direction of steepest increase
Gradient descent = move opposite the gradient to reduce loss
Optimization = finding the input that minimizes or maximizes an objective
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def numerical_derivative(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-5,
) -> float:
    """
    Estimate derivative of a single-variable function using central difference.

    Formula:
        f'(x) ≈ [f(x + h) - f(x - h)] / (2h)

    This is useful when we do not have the exact derivative formula.
    """
    if step <= 0:
        raise ValueError("Step size must be positive.")

    derivative = (function(x + step) - function(x - step)) / (2 * step)

    return float(derivative)


def numerical_gradient(
    function: Callable[[np.ndarray], float],
    point: Sequence[float],
    step: float = 1e-5,
) -> np.ndarray:
    """
    Estimate gradient of a multi-variable function using central difference.

    Gradient:
        ∇f = vector of partial derivatives

    For each variable, we slightly move forward and backward,
    then estimate how the function changes.
    """
    if step <= 0:
        raise ValueError("Step size must be positive.")

    x = np.array(point, dtype=float)

    if x.ndim != 1:
        raise ValueError("Point must be a one-dimensional vector.")

    gradient = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        forward = x.copy()
        backward = x.copy()

        forward[i] += step
        backward[i] -= step

        gradient[i] = (function(forward) - function(backward)) / (2 * step)

    return gradient


def gradient_descent_1d(
    function: Callable[[float], float],
    initial_x: float,
    learning_rate: float = 0.1,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Minimize a single-variable function using gradient descent.

    Update rule:
        x_new = x_old - learning_rate * derivative

    Returns:
        best_x
        best_value
        history of (x, function_value)
    """
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    if iterations <= 0:
        raise ValueError("Iterations must be positive.")

    x = float(initial_x)
    history: List[Tuple[float, float]] = []

    for _ in range(iterations):
        value = float(function(x))
        history.append((x, value))

        derivative = numerical_derivative(function, x)

        new_x = x - learning_rate * derivative

        if abs(new_x - x) < tolerance:
            x = new_x
            history.append((x, float(function(x))))
            break

        x = new_x

    best_x = x
    best_value = float(function(best_x))

    return best_x, best_value, history


def gradient_descent(
    function: Callable[[np.ndarray], float],
    initial_point: Sequence[float],
    gradient_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    learning_rate: float = 0.1,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
    """
    Minimize a multi-variable function using gradient descent.

    If gradient_function is provided, it uses the exact gradient.
    If not, it estimates the gradient numerically.

    Update rule:
        x_new = x_old - learning_rate * gradient
    """
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    if iterations <= 0:
        raise ValueError("Iterations must be positive.")

    x = np.array(initial_point, dtype=float)

    if x.ndim != 1:
        raise ValueError("Initial point must be a one-dimensional vector.")

    history: List[Tuple[np.ndarray, float]] = []

    for _ in range(iterations):
        value = float(function(x))
        history.append((x.copy(), value))

        if gradient_function is None:
            gradient = numerical_gradient(function, x)
        else:
            gradient = np.array(gradient_function(x), dtype=float)

        if gradient.shape != x.shape:
            raise ValueError("Gradient shape must match point shape.")

        new_x = x - learning_rate * gradient

        if np.linalg.norm(new_x - x) < tolerance:
            x = new_x
            history.append((x.copy(), float(function(x))))
            break

        x = new_x

    best_point = x
    best_value = float(function(best_point))

    return best_point, best_value, history


def mean_squared_error(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """
    Calculate mean squared error.

    Formula:
        MSE = mean((actual - predicted)^2)

    MSE is a common loss function in machine learning and regression.
    """
    y = np.array(actual, dtype=float)
    y_hat = np.array(predicted, dtype=float)

    if y.shape != y_hat.shape:
        raise ValueError("Actual and predicted values must have the same shape.")

    return float(np.mean((y - y_hat) ** 2))


def linear_regression_predictions(
    params: Sequence[float],
    x_values: Sequence[float],
) -> np.ndarray:
    """
    Generate predictions for a simple linear regression model.

    Model:
        y_hat = m*x + b

    params:
        [m, b]

    where:
        m = slope
        b = intercept
    """
    p = np.array(params, dtype=float)

    if p.shape != (2,):
        raise ValueError("Params must contain exactly two values: [slope, intercept].")

    x = np.array(x_values, dtype=float)

    slope = p[0]
    intercept = p[1]

    return slope * x + intercept


def linear_regression_loss(
    params: Sequence[float],
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    """
    Calculate mean squared error loss for simple linear regression.

    This allows us to optimize slope and intercept using gradient descent.
    """
    predictions = linear_regression_predictions(params, x_values)

    return mean_squared_error(y_values, predictions)