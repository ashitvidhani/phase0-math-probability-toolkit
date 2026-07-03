"""
Bayesian Updating Toolkit

This module contains reusable Bayesian probability functions for the
Phase 0 Mathematics + Probability Simulation Toolkit.

It covers:
- Basic Bayes theorem
- Binary Bayes update
- Total probability calculation
- Multi-hypothesis Bayesian updating
- Likelihood-ratio update
- Bayesian diagnostic summary

Core memory:

Bayes theorem:
    P(A|B) = P(B|A)P(A) / P(B)

Where:
    P(A)   = prior
    P(B|A) = likelihood
    P(B)   = evidence
    P(A|B) = posterior

Bayes = update belief after evidence.
"""

from typing import Dict, Sequence

import numpy as np


def validate_probability(
    probability: float,
    name: str = "probability",
) -> None:
    """
    Validate that a probability is between 0 and 1.

    Probability rule:
        0 <= p <= 1
    """
    if probability < 0 or probability > 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def bayes_theorem(
    prior: float,
    likelihood: float,
    evidence: float,
) -> float:
    """
    Apply Bayes theorem.

    Formula:
        posterior = likelihood * prior / evidence

    Meaning:
        P(A|B) = P(B|A)P(A) / P(B)

    Inputs:
        prior:
            P(A)

        likelihood:
            P(B|A)

        evidence:
            P(B)

    Output:
        posterior:
            P(A|B)
    """
    validate_probability(prior, "Prior")
    validate_probability(likelihood, "Likelihood")
    validate_probability(evidence, "Evidence")

    if evidence == 0:
        raise ValueError("Evidence cannot be zero.")

    posterior = (likelihood * prior) / evidence

    return float(posterior)


def total_probability_binary(
    prior: float,
    likelihood_if_true: float,
    likelihood_if_false: float,
) -> float:
    """
    Calculate total probability of evidence for a binary case.

    Formula:
        P(B) = P(B|A)P(A) + P(B|A^c)P(A^c)

    Where:
        A   = hypothesis is true
        A^c = hypothesis is false
        B   = evidence observed
    """
    validate_probability(prior, "Prior")
    validate_probability(likelihood_if_true, "Likelihood if true")
    validate_probability(likelihood_if_false, "Likelihood if false")

    probability_false = 1 - prior

    evidence = (
        likelihood_if_true * prior
        + likelihood_if_false * probability_false
    )

    return float(evidence)


def binary_bayes_update(
    prior: float,
    likelihood_if_true: float,
    likelihood_if_false: float,
) -> float:
    """
    Apply Bayes theorem in a binary true/false hypothesis setting.

    Formula:
        P(A|B) =
        P(B|A)P(A) /
        [P(B|A)P(A) + P(B|A^c)P(A^c)]

    Example:
        disease diagnosis
        default-risk updating
        HFT signal confidence updating
    """
    evidence = total_probability_binary(
        prior=prior,
        likelihood_if_true=likelihood_if_true,
        likelihood_if_false=likelihood_if_false,
    )

    return bayes_theorem(
        prior=prior,
        likelihood=likelihood_if_true,
        evidence=evidence,
    )


def binary_bayes_summary(
    prior: float,
    likelihood_if_true: float,
    likelihood_if_false: float,
) -> Dict[str, float]:
    """
    Return a full Bayesian update summary for a binary case.

    Returns:
        prior
        probability_false
        likelihood_if_true
        likelihood_if_false
        evidence
        posterior
        posterior_false
    """
    evidence = total_probability_binary(
        prior=prior,
        likelihood_if_true=likelihood_if_true,
        likelihood_if_false=likelihood_if_false,
    )

    posterior = binary_bayes_update(
        prior=prior,
        likelihood_if_true=likelihood_if_true,
        likelihood_if_false=likelihood_if_false,
    )

    posterior_false = 1 - posterior

    return {
        "prior": float(prior),
        "probability_false": float(1 - prior),
        "likelihood_if_true": float(likelihood_if_true),
        "likelihood_if_false": float(likelihood_if_false),
        "evidence": float(evidence),
        "posterior": float(posterior),
        "posterior_false": float(posterior_false),
    }


def posterior_from_likelihoods(
    priors: Sequence[float],
    likelihoods: Sequence[float],
) -> np.ndarray:
    """
    Calculate posterior probabilities for multiple hypotheses.

    Formula:
        posterior_i = likelihood_i * prior_i / evidence

    where:

        evidence = sum(likelihood_i * prior_i)

    Inputs:
        priors:
            Prior probabilities for all hypotheses.
            Must sum to 1.

        likelihoods:
            Probability of seeing evidence under each hypothesis.

    Output:
        posterior probabilities for all hypotheses.
    """
    prior_array = np.array(priors, dtype=float)
    likelihood_array = np.array(likelihoods, dtype=float)

    if prior_array.ndim != 1 or likelihood_array.ndim != 1:
        raise ValueError("Priors and likelihoods must be one-dimensional.")

    if prior_array.shape != likelihood_array.shape:
        raise ValueError("Priors and likelihoods must have the same shape.")

    if prior_array.size == 0:
        raise ValueError("Priors and likelihoods cannot be empty.")

    if np.any(prior_array < 0) or np.any(prior_array > 1):
        raise ValueError("All priors must be between 0 and 1.")

    if not np.isclose(np.sum(prior_array), 1.0):
        raise ValueError("Priors must sum to 1.")

    if np.any(likelihood_array < 0) or np.any(likelihood_array > 1):
        raise ValueError("All likelihoods must be between 0 and 1.")

    evidence = float(np.sum(prior_array * likelihood_array))

    if evidence == 0:
        raise ValueError("Evidence cannot be zero.")

    posteriors = (prior_array * likelihood_array) / evidence

    return posteriors


def probability_to_odds(probability: float) -> float:
    """
    Convert probability to odds.

    Formula:
        odds = p / (1 - p)

    Example:
        p = 0.75
        odds = 0.75 / 0.25 = 3
    """
    validate_probability(probability)

    if probability == 1:
        raise ValueError("Probability of 1 gives infinite odds.")

    return float(probability / (1 - probability))


def odds_to_probability(odds: float) -> float:
    """
    Convert odds back to probability.

    Formula:
        probability = odds / (1 + odds)
    """
    if odds < 0:
        raise ValueError("Odds cannot be negative.")

    return float(odds / (1 + odds))


def update_probability_with_likelihood_ratio(
    prior: float,
    likelihood_ratio: float,
) -> float:
    """
    Update probability using a likelihood ratio.

    Formula:
        posterior odds = prior odds * likelihood ratio

    Then:
        posterior probability = posterior odds / (1 + posterior odds)

    This is useful when evidence is expressed as a likelihood ratio.
    """
    validate_probability(prior, "Prior")

    if likelihood_ratio < 0:
        raise ValueError("Likelihood ratio cannot be negative.")

    prior_odds = probability_to_odds(prior)

    posterior_odds = prior_odds * likelihood_ratio

    return odds_to_probability(posterior_odds)