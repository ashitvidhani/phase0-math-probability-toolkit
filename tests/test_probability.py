import numpy as np
import pytest

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
    validate_probability,
    variance,
)


def test_validate_probability_accepts_valid_probability():
    validate_probability(0.5)


def test_validate_probability_rejects_invalid_probability():
    with pytest.raises(ValueError):
        validate_probability(1.5)


def test_bernoulli_trial_returns_zero_or_one():
    result = bernoulli_trial(0.5, random_seed=42)

    assert result in [0, 1]


def test_simulate_bernoulli_shape_and_values():
    samples = simulate_bernoulli(
        probability_success=0.5,
        trials=100,
        random_seed=42,
    )

    assert samples.shape == (100,)
    assert set(np.unique(samples)).issubset({0, 1})


def test_binomial_probability():
    probability = binomial_probability(
        trials=3,
        probability_success=0.5,
        successes=2,
    )

    assert probability == pytest.approx(0.375)


def test_binomial_probability_invalid_success_count_returns_zero():
    probability = binomial_probability(
        trials=3,
        probability_success=0.5,
        successes=5,
    )

    assert probability == pytest.approx(0.0)


def test_simulate_binomial_shape_and_bounds():
    samples = simulate_binomial(
        trials=10,
        probability_success=0.5,
        simulations=100,
        random_seed=42,
    )

    assert samples.shape == (100,)
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)


def test_normal_pdf_standard_normal_at_zero():
    density = normal_pdf(0)

    assert density == pytest.approx(1 / np.sqrt(2 * np.pi))


def test_normal_cdf_standard_normal_at_zero():
    cumulative_probability = normal_cdf(0)

    assert cumulative_probability == pytest.approx(0.5)


def test_sample_normal_shape():
    samples = sample_normal(
        mean=0,
        standard_deviation=1,
        samples=100,
        random_seed=42,
    )

    assert samples.shape == (100,)


def test_expected_value():
    values = [0, 10]
    probabilities = [0.5, 0.5]

    assert expected_value(values, probabilities) == pytest.approx(5.0)


def test_expected_value_rejects_probabilities_not_summing_to_one():
    values = [0, 10]
    probabilities = [0.5, 0.4]

    with pytest.raises(ValueError):
        expected_value(values, probabilities)


def test_variance():
    values = [0, 10]
    probabilities = [0.5, 0.5]

    assert variance(values, probabilities) == pytest.approx(25.0)


def test_standard_deviation():
    values = [0, 10]
    probabilities = [0.5, 0.5]

    assert standard_deviation(values, probabilities) == pytest.approx(5.0)


def test_empirical_summary():
    samples = [1, 2, 3, 4, 5]

    summary = empirical_summary(samples)

    assert summary["count"] == pytest.approx(5.0)
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["variance"] == pytest.approx(2.0)
    assert summary["standard_deviation"] == pytest.approx(np.sqrt(2.0))
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(5.0)


def test_empirical_summary_rejects_empty_samples():
    with pytest.raises(ValueError):
        empirical_summary([])