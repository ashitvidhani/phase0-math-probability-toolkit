"""
Demo for Optimization Toolkit

Run from project root:

    python -m examples.optimization_demo
"""

import numpy as np

from math_toolkit.optimization import (
    gradient_descent,
    gradient_descent_1d,
    linear_regression_loss,
    linear_regression_predictions,
    mean_squared_error,
    numerical_derivative,
    numerical_gradient,
)


def main():
    print("=== Optimization Toolkit Demo ===")

    # ------------------------------------------------------------
    # 1D derivative example
    # ------------------------------------------------------------
    def f_1d(x: float) -> float:
        return (x - 3) ** 2 + 2

    print("\n1D Function:")
    print("f(x) = (x - 3)^2 + 2")
    print("Minimum should be near x = 3")

    derivative_at_5 = numerical_derivative(f_1d, 5)

    print("\nNumerical derivative at x = 5:")
    print(derivative_at_5)

    best_x, best_value, history_1d = gradient_descent_1d(
        f_1d,
        initial_x=10,
        learning_rate=0.1,
        iterations=100,
    )

    print("\nGradient Descent 1D Result")
    print("Best x:", best_x)
    print("Best value:", best_value)
    print("Iterations used:", len(history_1d))

    # ------------------------------------------------------------
    # Multi-variable gradient example
    # ------------------------------------------------------------
    def f_multi(point: np.ndarray) -> float:
        x = point[0]
        y = point[1]
        return (x - 2) ** 2 + (y + 1) ** 2

    def grad_multi(point: np.ndarray) -> np.ndarray:
        x = point[0]
        y = point[1]
        return np.array(
            [
                2 * (x - 2),
                2 * (y + 1),
            ],
            dtype=float,
        )

    initial_point = [10, 10]

    print("\nMulti-variable Function:")
    print("f(x, y) = (x - 2)^2 + (y + 1)^2")
    print("Minimum should be near [2, -1]")

    numerical_grad = numerical_gradient(f_multi, initial_point)

    print("\nNumerical gradient at [10, 10]:")
    print(numerical_grad)

    best_point, best_multi_value, history_multi = gradient_descent(
        f_multi,
        initial_point=initial_point,
        gradient_function=grad_multi,
        learning_rate=0.1,
        iterations=200,
    )

    print("\nGradient Descent Multi-variable Result")
    print("Best point:", best_point)
    print("Best value:", best_multi_value)
    print("Iterations used:", len(history_multi))

    # ------------------------------------------------------------
    # Linear regression loss example
    # ------------------------------------------------------------
    x_values = np.array([1, 2, 3, 4, 5], dtype=float)
    y_values = np.array([3, 5, 7, 9, 11], dtype=float)

    print("\nLinear Regression Data")
    print("x:", x_values)
    print("y:", y_values)
    print("True relationship: y = 2x + 1")

    def regression_objective(params: np.ndarray) -> float:
        return linear_regression_loss(params, x_values, y_values)

    best_params, best_loss, history_regression = gradient_descent(
        regression_objective,
        initial_point=[0, 0],
        learning_rate=0.01,
        iterations=1000,
    )

    predictions = linear_regression_predictions(best_params, x_values)

    print("\nLinear Regression Optimization Result")
    print("Best params [slope, intercept]:", best_params)
    print("Best loss:", best_loss)
    print("Predictions:", predictions)
    print("MSE:", mean_squared_error(y_values, predictions))
    print("Iterations used:", len(history_regression))


if __name__ == "__main__":
    main()