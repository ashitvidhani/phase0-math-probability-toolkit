import numpy as np
import pytest

from math_toolkit.monte_carlo import (
    estimate_event_probability,
    estimate_expected_value,
    monte_carlo_summary,
    percentile_value,
    run_monte_carlo,
    simulate_asset_price_paths,
    simulate_random_walk,
    value_at_risk,
)


def test_run_monte_carlo_shape():
    def simulation_function(rng):
        return rng.normal()

    results = run_monte_carlo(
        simulation_function=simulation_function,
        simulations=100,
        random_seed=42,
    )

    assert results.shape == (100,)


def test_run_monte_carlo_rejects_invalid_simulations():
    def simulation_function(rng):
        return rng.normal()

    with pytest.raises(ValueError):
        run_monte_carlo(simulation_function, simulations=0)


def test_estimate_expected_value():
    samples = [1, 2, 3, 4, 5]

    assert estimate_expected_value(samples) == pytest.approx(3.0)


def test_estimate_event_probability():
    samples = [-1, 2, 3, -4]

    probability = estimate_event_probability(samples, lambda x: x > 0)

    assert probability == pytest.approx(0.5)


def test_monte_carlo_summary():
    samples = [1, 2, 3, 4, 5]

    summary = monte_carlo_summary(samples)

    assert summary["count"] == pytest.approx(5.0)
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["variance"] == pytest.approx(2.0)
    assert summary["standard_deviation"] == pytest.approx(np.sqrt(2.0))
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(5.0)
    assert summary["p50"] == pytest.approx(3.0)


def test_simulate_random_walk_shape_and_start():
    path = simulate_random_walk(
        steps=10,
        start_value=100,
        drift=0.0,
        volatility=1.0,
        random_seed=42,
    )

    assert path.shape == (11,)
    assert path[0] == pytest.approx(100.0)


def test_simulate_random_walk_rejects_invalid_steps():
    with pytest.raises(ValueError):
        simulate_random_walk(steps=0)


def test_simulate_asset_price_paths_shape_and_initial_price():
    paths = simulate_asset_price_paths(
        initial_price=100,
        drift=0.05,
        volatility=0.2,
        time_horizon=1,
        steps=10,
        simulations=20,
        random_seed=42,
    )

    assert paths.shape == (20, 11)
    assert np.allclose(paths[:, 0], 100.0)


def test_simulate_asset_price_paths_rejects_invalid_initial_price():
    with pytest.raises(ValueError):
        simulate_asset_price_paths(
            initial_price=0,
            drift=0.05,
            volatility=0.2,
            time_horizon=1,
            steps=10,
            simulations=20,
        )


def test_percentile_value():
    samples = [1, 2, 3, 4, 5]

    assert percentile_value(samples, 50) == pytest.approx(3.0)


def test_percentile_value_rejects_invalid_percentile():
    samples = [1, 2, 3]

    with pytest.raises(ValueError):
        percentile_value(samples, 101)


def test_value_at_risk():
    returns = [-0.10, -0.05, 0.00, 0.05, 0.10]

    var_95 = value_at_risk(returns, confidence_level=0.95)

    losses = -np.array(returns, dtype=float)
    expected = np.percentile(losses, 95)

    assert var_95 == pytest.approx(expected)


def test_value_at_risk_rejects_invalid_confidence_level():
    returns = [-0.1, 0.0, 0.1]

    with pytest.raises(ValueError):
        value_at_risk(returns, confidence_level=1.0)