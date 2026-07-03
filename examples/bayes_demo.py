"""
Demo for Bayesian Updating Toolkit

Run from project root:

    python -m examples.bayes_demo
"""

import numpy as np

from math_toolkit.bayes import (
    bayes_theorem,
    binary_bayes_summary,
    binary_bayes_update,
    odds_to_probability,
    posterior_from_likelihoods,
    probability_to_odds,
    total_probability_binary,
    update_probability_with_likelihood_ratio,
)


def main():
    print("=== Bayesian Updating Toolkit Demo ===")

    # ------------------------------------------------------------
    # Classic medical diagnosis example
    # ------------------------------------------------------------
    disease_prior = 0.01
    sensitivity = 0.99
    false_positive_rate = 0.05

    evidence = total_probability_binary(
        prior=disease_prior,
        likelihood_if_true=sensitivity,
        likelihood_if_false=false_positive_rate,
    )

    posterior = bayes_theorem(
        prior=disease_prior,
        likelihood=sensitivity,
        evidence=evidence,
    )

    print("\nMedical Diagnosis Example")
    print("Prior P(disease):", disease_prior)
    print("P(positive | disease):", sensitivity)
    print("P(positive | no disease):", false_positive_rate)
    print("Evidence P(positive):", evidence)
    print("Posterior P(disease | positive):", posterior)

    print("\nSame result using binary_bayes_update:")
    print(binary_bayes_update(disease_prior, sensitivity, false_positive_rate))

    print("\nFull binary Bayes summary:")
    print(binary_bayes_summary(disease_prior, sensitivity, false_positive_rate))

    # ------------------------------------------------------------
    # Finance default-risk example
    # ------------------------------------------------------------
    default_prior = 0.02
    likelihood_income_fall_if_default = 0.80
    likelihood_income_fall_if_not_default = 0.10

    updated_default_probability = binary_bayes_update(
        prior=default_prior,
        likelihood_if_true=likelihood_income_fall_if_default,
        likelihood_if_false=likelihood_income_fall_if_not_default,
    )

    print("\nFinance Default Risk Example")
    print("Prior P(default):", default_prior)
    print("Posterior P(default | income fall):", updated_default_probability)

    # ------------------------------------------------------------
    # HFT signal example
    # ------------------------------------------------------------
    prior_next_tick_up = 0.50
    likelihood_signal_if_up = 0.70
    likelihood_signal_if_not_up = 0.40

    updated_tick_probability = binary_bayes_update(
        prior=prior_next_tick_up,
        likelihood_if_true=likelihood_signal_if_up,
        likelihood_if_false=likelihood_signal_if_not_up,
    )

    print("\nHFT Signal Example")
    print("Prior P(next tick up):", prior_next_tick_up)
    print("Posterior P(next tick up | buy signal):", updated_tick_probability)

    # ------------------------------------------------------------
    # Multi-hypothesis regime example
    # ------------------------------------------------------------
    priors = [0.70, 0.25, 0.05]
    likelihoods = [0.10, 0.50, 0.90]

    posteriors = posterior_from_likelihoods(priors, likelihoods)

    print("\nMarket Regime Example")
    print("Hypotheses: [normal, volatile, crash]")
    print("Priors:", priors)
    print("Likelihoods of volatility spike:", likelihoods)
    print("Posteriors after evidence:", posteriors)
    print("Posterior sum:", np.sum(posteriors))

    # ------------------------------------------------------------
    # Odds and likelihood ratio example
    # ------------------------------------------------------------
    probability = 0.75
    odds = probability_to_odds(probability)

    print("\nProbability and Odds")
    print("Probability:", probability)
    print("Odds:", odds)
    print("Back to probability:", odds_to_probability(odds))

    prior = 0.20
    likelihood_ratio = 3.0

    updated_probability = update_probability_with_likelihood_ratio(
        prior=prior,
        likelihood_ratio=likelihood_ratio,
    )

    print("\nLikelihood Ratio Update")
    print("Prior probability:", prior)
    print("Likelihood ratio:", likelihood_ratio)
    print("Updated probability:", updated_probability)


if __name__ == "__main__":
    main()