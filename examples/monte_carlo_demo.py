"""
Demo for Monte Carlo Simulation Toolkit

Run from project root:

    python -m examples.monte_carlo_demo
"""

import numpy as np

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


def main():
    print("=== Monte Carlo Simulation Toolkit Demo ===")

    # ------------------------------------------------------------
    # Generic Monte Carlo: trade PnL simulation
    # ------------------------------------------------------------
    def simulated_trade_pnl(rng: np.random.Generator) -> float:
        """
        Simulate one trade PnL.

        Assume:
            average profit = 10
            standard deviation = 25
        """
        return float(rng.normal(loc=10, scale=25))

    pnl_samples = run_monte_carlo(
        simulation_function=simulated_trade_pnl,
        simulations=10_000,
        random_seed=42,
    )

    print("\nTrade PnL Monte Carlo")
    print("Estimated expected PnL:", estimate_expected_value(pnl_samples))
    print("Probability of profit:", estimate_event_probability(pnl_samples, lambda x: x > 0))
    print("PnL summary:")
    print(monte_carlo_summary(pnl_samples))

    # ------------------------------------------------------------
    # Random walk
    # ------------------------------------------------------------
    random_walk = simulate_random_walk(
        steps=10,
        start_value=100,
        drift=0.1,
        volatility=1.0,
        random_seed=42,
    )

    print("\nRandom Walk")
    print(random_walk)

    # ------------------------------------------------------------
    # Asset price paths
    # ------------------------------------------------------------
    paths = simulate_asset_price_paths(
        initial_price=100,
        drift=0.08,
        volatility=0.20,
        time_horizon=1.0,
        steps=252,
        simulations=1000,
        random_seed=42,
    )

    final_prices = paths[:, -1]

    print("\nAsset Price Simulation")
    print("Paths shape:", paths.shape)
    print("Average final price:", np.mean(final_prices))
    print("5th percentile final price:", percentile_value(final_prices, 5))
    print("95th percentile final price:", percentile_value(final_prices, 95))

    simulated_returns = (final_prices - 100) / 100

    print("\nRisk Estimate")
    print("Simulated 95% VaR:", value_at_risk(simulated_returns, confidence_level=0.95))


if __name__ == "__main__":
    main()