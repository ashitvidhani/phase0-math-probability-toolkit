import numpy as np
import pytest

from math_toolkit.bayes import (
    bayes_theorem,
    binary_bayes_summary,
    binary_bayes_update,
    odds_to_probability,
    posterior_from_likelihoods,
    probability_to_odds,
    total_probability_binary,
    update_probability_with_likelihood_ratio,
    validate_probability,
)


def test_validate_probability_accepts_valid_value():
    validate_probability(0.5)


def test_validate_probability_rejects_invalid_value():
    with pytest.raises(ValueError):
        validate_probability(1.5)


def test_bayes_theorem_medical_example():
    posterior = bayes_theorem(
        prior=0.01,
        likelihood=0.99,
        evidence=0.0594,
    )

    assert posterior == pytest.approx(0.1666666667)


def test_bayes_theorem_rejects_zero_evidence():
    with pytest.raises(ValueError):
        bayes_theorem(
            prior=0.01,
            likelihood=0.99,
            evidence=0.0,
        )


def test_total_probability_binary():
    evidence = total_probability_binary(
        prior=0.01,
        likelihood_if_true=0.99,
        likelihood_if_false=0.05,
    )

    assert evidence == pytest.approx(0.0594)


def test_binary_bayes_update_medical_example():
    posterior = binary_bayes_update(
        prior=0.01,
        likelihood_if_true=0.99,
        likelihood_if_false=0.05,
    )

    assert posterior == pytest.approx(0.1666666667)


def test_binary_bayes_update_finance_example():
    posterior = binary_bayes_update(
        prior=0.02,
        likelihood_if_true=0.80,
        likelihood_if_false=0.10,
    )

    assert posterior == pytest.approx(0.1403508772)


def test_binary_bayes_update_hft_example():
    posterior = binary_bayes_update(
        prior=0.50,
        likelihood_if_true=0.70,
        likelihood_if_false=0.40,
    )

    assert posterior == pytest.approx(0.6363636364)


def test_binary_bayes_summary():
    summary = binary_bayes_summary(
        prior=0.01,
        likelihood_if_true=0.99,
        likelihood_if_false=0.05,
    )

    assert summary["prior"] == pytest.approx(0.01)
    assert summary["probability_false"] == pytest.approx(0.99)
    assert summary["evidence"] == pytest.approx(0.0594)
    assert summary["posterior"] == pytest.approx(0.1666666667)
    assert summary["posterior_false"] == pytest.approx(0.8333333333)


def test_posterior_from_likelihoods_sums_to_one():
    priors = [0.70, 0.25, 0.05]
    likelihoods = [0.10, 0.50, 0.90]

    posteriors = posterior_from_likelihoods(priors, likelihoods)

    assert np.sum(posteriors) == pytest.approx(1.0)


def test_posterior_from_likelihoods_values():
    priors = [0.70, 0.25, 0.05]
    likelihoods = [0.10, 0.50, 0.90]

    posteriors = posterior_from_likelihoods(priors, likelihoods)

    evidence = 0.70 * 0.10 + 0.25 * 0.50 + 0.05 * 0.90
    expected = np.array(
        [
            0.70 * 0.10 / evidence,
            0.25 * 0.50 / evidence,
            0.05 * 0.90 / evidence,
        ]
    )

    assert np.allclose(posteriors, expected)


def test_posterior_from_likelihoods_rejects_priors_not_sum_one():
    priors = [0.70, 0.20, 0.05]
    likelihoods = [0.10, 0.50, 0.90]

    with pytest.raises(ValueError):
        posterior_from_likelihoods(priors, likelihoods)


def test_probability_to_odds():
    odds = probability_to_odds(0.75)

    assert odds == pytest.approx(3.0)


def test_odds_to_probability():
    probability = odds_to_probability(3.0)

    assert probability == pytest.approx(0.75)


def test_update_probability_with_likelihood_ratio():
    posterior = update_probability_with_likelihood_ratio(
        prior=0.20,
        likelihood_ratio=3.0,
    )

    prior_odds = 0.20 / 0.80
    posterior_odds = prior_odds * 3.0
    expected = posterior_odds / (1 + posterior_odds)

    assert posterior == pytest.approx(expected)