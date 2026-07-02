"""
Monte Carlo Simulation Toolkit

This module contains reusable Monte Carlo simulation functions for the
Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- Generic Monte Carlo simulation
- Expected value estimation
- Event probability estimation
- Random walk simulation
- Asset price path simulation
- Percentile / Value-at-Risk style risk estimation

Core memory:

Monte Carlo = use repeated random simulation to estimate uncertain outcomes.

Instead of solving probability analytically, we simulate many possible worlds
and measure the results.
"""

from typing import Callable, Dict, Sequence

import numpy as np


def run_monte_carlo(
    simulation_function: Callable[[np.random.Generator], float],
    simulations: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Run a generic Monte Carlo simulation.

    Input:
        simulation_function:
            A function that accepts a NumPy random generator and returns one number.

        simulations:
            Number of repeated simulations.

        random_seed:
            Optional seed for repeatable results.

    Output:
        NumPy array of simulation results.

    Example:
        Estimate expected payoff of a random trade.
    """
    if simulations <= 0:
        raise ValueError("Number of simulations must be positive.")

    rng = np.random.default_rng(random_seed)

    results = [
        simulation_function(rng)
        for _ in range(simulations)
    ]

    return np.array(results, dtype=float)


def estimate_expected_value(samples: Sequence[float]) -> float:
    """
    Estimate expected value from simulated samples.

    Formula:
        estimated E[X] = average of simulated outcomes

    Monte Carlo idea:
        If simulations are large enough, sample average approaches expectation.
    """
    data = np.array(samples, dtype=float)

    if data.ndim != 1:
        raise ValueError("Samples must be one-dimensional.")

    if data.size == 0:
        raise ValueError("Samples cannot be empty.")

    return float(np.mean(data))


def estimate_event_probability(
    samples: Sequence[float],
    event_condition: Callable[[float], bool],
) -> float:
    """
    Estimate probability of an event from simulated samples.

    Example:
        P(profit > 0)

    We count how many simulations satisfy the condition
    and divide by total simulations.
    """
    data = np.array(samples, dtype=float)

    if data.ndim != 1:
        raise ValueError("Samples must be one-dimensional.")

    if data.size == 0:
        raise ValueError("Samples cannot be empty.")

    event_results = np.array(
        [event_condition(value) for value in data],
        dtype=bool,
    )

    return float(np.mean(event_results))


def monte_carlo_summary(samples: Sequence[float]) -> Dict[str, float]:
    """
    Summarize Monte Carlo simulation outcomes.

    Returns:
        count
        mean
        variance
        standard deviation
        min
        max
        5th percentile
        50th percentile
        95th percentile
    """
    data = np.array(samples, dtype=float)

    if data.ndim != 1:
        raise ValueError("Samples must be one-dimensional.")

    if data.size == 0:
        raise ValueError("Samples cannot be empty.")

    return {
        "count": float(data.size),
        "mean": float(np.mean(data)),
        "variance": float(np.var(data)),
        "standard_deviation": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p5": float(np.percentile(data, 5)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }


def simulate_random_walk(
    steps: int,
    start_value: float = 0.0,
    drift: float = 0.0,
    volatility: float = 1.0,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulate a simple random walk.

    Formula:
        X_t = X_{t-1} + drift + volatility * random_noise

    Output:
        Array of length steps + 1

    The first value is start_value.
    """
    if steps <= 0:
        raise ValueError("Steps must be positive.")

    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")

    rng = np.random.default_rng(random_seed)

    shocks = rng.normal(
        loc=drift,
        scale=volatility,
        size=steps,
    )

    path = np.empty(steps + 1, dtype=float)
    path[0] = start_value
    path[1:] = start_value + np.cumsum(shocks)

    return path


def simulate_asset_price_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    steps: int,
    simulations: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulate asset price paths using geometric Brownian motion.

    Model:
        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Where:
        S = asset price
        mu = drift
        sigma = volatility
        Z = standard normal random shock

    Output:
        Matrix with shape:
            simulations x (steps + 1)

    Each row is one possible future price path.
    """
    if initial_price <= 0:
        raise ValueError("Initial price must be positive.")

    if volatility < 0:
        raise ValueError("Volatility cannot be negative.")

    if time_horizon <= 0:
        raise ValueError("Time horizon must be positive.")

    if steps <= 0:
        raise ValueError("Steps must be positive.")

    if simulations <= 0:
        raise ValueError("Number of simulations must be positive.")

    rng = np.random.default_rng(random_seed)

    dt = time_horizon / steps

    random_shocks = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(simulations, steps),
    )

    log_returns = (
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * random_shocks
    )

    paths = np.empty((simulations, steps + 1), dtype=float)
    paths[:, 0] = initial_price

    paths[:, 1:] = initial_price * np.exp(np.cumsum(log_returns, axis=1))

    return paths


def percentile_value(
    samples: Sequence[float],
    percentile: float,
) -> float:
    """
    Calculate percentile value from samples.

    Example:
        percentile_value(samples, 5)
        gives the 5th percentile.
    """
    data = np.array(samples, dtype=float)

    if data.ndim != 1:
        raise ValueError("Samples must be one-dimensional.")

    if data.size == 0:
        raise ValueError("Samples cannot be empty.")

    if percentile < 0 or percentile > 100:
        raise ValueError("Percentile must be between 0 and 100.")

    return float(np.percentile(data, percentile))


def value_at_risk(
    returns: Sequence[float],
    confidence_level: float = 0.95,
) -> float:
    """
    Estimate historical / simulated Value-at-Risk.

    Input:
        returns:
            Simulated return outcomes.

        confidence_level:
            Example: 0.95 for 95% VaR.

    Output:
        Positive loss number.

    Logic:
        Convert returns into losses:
            loss = -return

        Then take the confidence percentile of losses.

    Example:
        If 95% VaR = 0.03,
        then loss should not exceed 3% in about 95% of simulated cases,
        under the model assumptions.
    """
    data = np.array(returns, dtype=float)

    if data.ndim != 1:
        raise ValueError("Returns must be one-dimensional.")

    if data.size == 0:
        raise ValueError("Returns cannot be empty.")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    losses = -data

    percentile = confidence_level * 100

    return float(np.percentile(losses, percentile))