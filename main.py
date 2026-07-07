"""
Command-line entry point for the Phase 0 Mathematics + Probability Simulation Toolkit.
"""

import argparse

from math_toolkit.linear_algebra import dot_product, vector_magnitude
from math_toolkit.probability import binomial_probability, normal_cdf
from math_toolkit.monte_carlo import simulate_random_walk, monte_carlo_summary
from math_toolkit.portfolio_risk import portfolio_summary


def run_linear_algebra_demo() -> None:
    """Run a small linear algebra demo."""
    print("\nLinear Algebra Demo")
    print("-------------------")
    print("vector_magnitude([3, 4]) =", vector_magnitude([3, 4]))
    print(
        "dot_product([1, 2, 3], [4, 5, 6]) =",
        dot_product([1, 2, 3], [4, 5, 6]),
    )


def run_probability_demo() -> None:
    """Run a small probability demo."""
    print("\nProbability Demo")
    print("----------------")
    print(
        "P(X = 2) for Binomial(n=3, p=0.5) =",
        binomial_probability(3, 0.5, 2),
    )
    print("Standard normal CDF at 0 =", normal_cdf(0))


def run_monte_carlo_demo() -> None:
    """Run a small Monte Carlo simulation demo."""
    print("\nMonte Carlo Demo")
    print("----------------")
    path = simulate_random_walk(
        steps=10,
        start_value=0.0,
        drift=0.0,
        volatility=1.0,
        random_seed=42,
    )
    print("Random walk path:", path)
    print("Summary:", monte_carlo_summary(path))


def run_portfolio_demo() -> None:
    """Run a small portfolio risk demo."""
    print("\nPortfolio Risk Demo")
    print("-------------------")

    weights = [0.6, 0.4]
    expected_returns = [0.10, 0.06]
    covariance_matrix = [
        [0.04, 0.01],
        [0.01, 0.02],
    ]

    print(
        "Portfolio summary:",
        portfolio_summary(weights, expected_returns, covariance_matrix),
    )


def main() -> None:
    """Parse command-line arguments and run selected demos."""
    parser = argparse.ArgumentParser(
        description="Phase 0 Mathematics + Probability Simulation Toolkit"
    )

    parser.add_argument(
        "--demo",
        choices=["all", "linear-algebra", "probability", "monte-carlo", "portfolio"],
        default="all",
        help="Choose which demo to run.",
    )

    args = parser.parse_args()

    print("Phase 0 Mathematics + Probability Simulation Toolkit")
    print("Status: Professional open-source/package-grade demo runner")

    if args.demo in ["all", "linear-algebra"]:
        run_linear_algebra_demo()

    if args.demo in ["all", "probability"]:
        run_probability_demo()

    if args.demo in ["all", "monte-carlo"]:
        run_monte_carlo_demo()

    if args.demo in ["all", "portfolio"]:
        run_portfolio_demo()


if __name__ == "__main__":
    main()