import numpy as np
import pytest

from math_toolkit.optimization import (
    gradient_descent,
    gradient_descent_1d,
    linear_regression_loss,
    linear_regression_predictions,
    mean_squared_error,
    numerical_derivative,
    numerical_gradient,
)


def test_numerical_derivative_quadratic():
    def f(x):
        return x**2

    assert numerical_derivative(f, 3) == pytest.approx(6.0, rel=1e-4)


def test_numerical_derivative_rejects_negative_step():
    def f(x):
        return x**2

    with pytest.raises(ValueError):
        numerical_derivative(f, 3, step=-1e-5)


def test_numerical_gradient_quadratic():
    def f(point):
        x = point[0]
        y = point[1]
        return x**2 + y**2

    gradient = numerical_gradient(f, [3, 4])

    assert np.allclose(gradient, [6, 8], atol=1e-4)


def test_numerical_gradient_rejects_negative_step():
    def f(point):
        return point[0] ** 2

    with pytest.raises(ValueError):
        numerical_gradient(f, [1, 2], step=-1e-5)


def test_gradient_descent_1d_finds_minimum():
    def f(x):
        return (x - 3) ** 2

    best_x, best_value, history = gradient_descent_1d(
        f,
        initial_x=10,
        learning_rate=0.1,
        iterations=200,
    )

    assert best_x == pytest.approx(3.0, abs=1e-3)
    assert best_value == pytest.approx(0.0, abs=1e-5)
    assert len(history) > 0


def test_gradient_descent_1d_rejects_invalid_learning_rate():
    def f(x):
        return x**2

    with pytest.raises(ValueError):
        gradient_descent_1d(f, initial_x=1, learning_rate=0)


def test_gradient_descent_multivariable_with_exact_gradient():
    def f(point):
        x = point[0]
        y = point[1]
        return (x - 2) ** 2 + (y + 1) ** 2

    def grad(point):
        x = point[0]
        y = point[1]
        return np.array(
            [
                2 * (x - 2),
                2 * (y + 1),
            ],
            dtype=float,
        )

    best_point, best_value, history = gradient_descent(
        f,
        initial_point=[10, 10],
        gradient_function=grad,
        learning_rate=0.1,
        iterations=300,
    )

    assert np.allclose(best_point, [2, -1], atol=1e-3)
    assert best_value == pytest.approx(0.0, abs=1e-5)
    assert len(history) > 0


def test_gradient_descent_multivariable_with_numerical_gradient():
    def f(point):
        x = point[0]
        y = point[1]
        return (x - 1) ** 2 + (y - 4) ** 2

    best_point, best_value, history = gradient_descent(
        f,
        initial_point=[8, -2],
        learning_rate=0.1,
        iterations=300,
    )

    assert np.allclose(best_point, [1, 4], atol=1e-3)
    assert best_value == pytest.approx(0.0, abs=1e-5)
    assert len(history) > 0


def test_gradient_descent_rejects_invalid_iterations():
    def f(point):
        return point[0] ** 2

    with pytest.raises(ValueError):
        gradient_descent(f, initial_point=[1], iterations=0)


def test_mean_squared_error():
    actual = [1, 2, 3]
    predicted = [1, 2, 5]

    assert mean_squared_error(actual, predicted) == pytest.approx(4 / 3)


def test_mean_squared_error_rejects_shape_mismatch():
    actual = [1, 2, 3]
    predicted = [1, 2]

    with pytest.raises(ValueError):
        mean_squared_error(actual, predicted)


def test_linear_regression_predictions():
    params = [2, 1]
    x_values = [1, 2, 3]

    predictions = linear_regression_predictions(params, x_values)

    assert np.allclose(predictions, [3, 5, 7])


def test_linear_regression_loss_zero_for_perfect_line():
    params = [2, 1]
    x_values = [1, 2, 3]
    y_values = [3, 5, 7]

    loss = linear_regression_loss(params, x_values, y_values)

    assert loss == pytest.approx(0.0)


def test_linear_regression_predictions_rejects_bad_params():
    params = [2, 1, 0]
    x_values = [1, 2, 3]

    with pytest.raises(ValueError):
        linear_regression_predictions(params, x_values)