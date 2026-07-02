"""
Demo for Probability Distribution Toolkit

Run from project root:

    python -m examples.probability_demo
"""

import numpy as np

from math_toolkit.probability import (
    bernoulli_trial,
    binomial_probability,
    empirical_summary,
    expected_value,
    normal_cdf,
    normal_pdf,
    sample_normal,
    simulate_bernoulli,
    simulate_binomial,
    standard_deviation,
    variance,
)


def main():
    print("=== Probability Distribution Toolkit Demo ===")

    # ------------------------------------------------------------
    # Bernoulli trial
    # ------------------------------------------------------------
    print("\nBernoulli Trial")
    print("One yes/no experiment with p = 0.5")
    print("Result:", bernoulli_trial(0.5, random_seed=42))

    bernoulli_samples = simulate_bernoulli(
        probability_success=0.5,
        trials=20,
        random_seed=42,
    )

    print("\n20 Bernoulli trials:")
    print(bernoulli_samples)
    print("Empirical success rate:", np.mean(bernoulli_samples))

    # ------------------------------------------------------------
    # Binomial probability
    # ------------------------------------------------------------
    print("\nBinomial Probability")
    print("X ~ Binomial(n=3, p=0.5)")
    print("P(X = 2):", binomial_probability(3, 0.5, 2))

    binomial_samples = simulate_binomial(
        trials=10,
        probability_success=0.5,
        simulations=20,
        random_seed=42,
    )

    print("\n20 Binomial simulations:")
    print(binomial_samples)
    print("Empirical average successes:", np.mean(binomial_samples))

    # ------------------------------------------------------------
    # Normal distribution
    # ------------------------------------------------------------
    print("\nNormal Distribution")
    print("Standard normal PDF at x = 0:")
    print(normal_pdf(0))

    print("Standard normal CDF at x = 0:")
    print(normal_cdf(0))

    normal_samples = sample_normal(
        mean=0,
        standard_deviation=1,
        samples=1000,
        random_seed=42,
    )

    print("\nNormal sample summary:")
    print(empirical_summary(normal_samples))

    # ------------------------------------------------------------
    # Expected value, variance, standard deviation
    # ------------------------------------------------------------
    values = [0, 10]
    probabilities = [0.5, 0.5]

    print("\nDiscrete Random Variable")
    print("Values:", values)
    print("Probabilities:", probabilities)

    print("Expected value:", expected_value(values, probabilities))
    print("Variance:", variance(values, probabilities))
    print("Standard deviation:", standard_deviation(values, probabilities))

    # ------------------------------------------------------------
    # Fair die example
    # ------------------------------------------------------------
    die_values = [1, 2, 3, 4, 5, 6]
    die_probabilities = [1 / 6] * 6

    print("\nFair Die")
    print("Expected value:", expected_value(die_values, die_probabilities))
    print("Variance:", variance(die_values, die_probabilities))
    print("Standard deviation:", standard_deviation(die_values, die_probabilities))


if __name__ == "__main__":
    main()