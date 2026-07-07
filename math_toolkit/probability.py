"""
Probability Distribution Toolkit

This module contains reusable probability and statistic functions for the
Phase 0 Mathematics + Probability Simulation Toolkit

It Covers:
- Bernaulli Simulation
- Binomial Probability
- Binomial Simulation
- Normal PDF
- Normal CDF
- Normal Sampling
- Expected Value
- Variance
- Standard Deviation
- Empirical Sample Summary

Core Memory:

Random Variable = Uncertain Numerical Quantity
Distribution - Probability pattern over possible values
Expectation = Probability-Weighted average
Variance = average squared spread around the mean
Standard Deviation = redeable spread in original units
"""

from math import comb, erf, exp, pi, sqrt
from typing import Dict, Sequence

import numpy as np

def validate_probability(probability: float) -> None:
    """
    Validate that the probability is between 0 and 1.

    Probability must satisfy:

    0 <= p <= 1
    """
    if probability<0 or probability>1:
        raise ValueError("Proabability must be between 0 and 1.")
    
def bernoulli_trial(
        probability_success: float,
        random_seed: int | None = None,
) -> int:
    """
    Simulate one Bernoulli trial.

    Bernaoulli trial = one yes/no experiment.

    Output:
    1 = Success
    0 = Failure

    Example:
    Coin toss with probability_success = 0.5
    """
    validate_probability(probability_success)

    rng = np.random.default_rng(random_seed)

    return int(rng.binomial(n=1, p=probability_success))

def simulate_bernoulli(
        probability_success: float,
        trials: int,
        random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulate many Bernoulli trials.

    Output:
    NumPy array of 0s and 1s

    Example:
    simulate_bernoulli(0.5,10)
    """
    validate_probability(probability_success)

    if trials <= 0:
        raise ValueError("Number of trials must be positive.")
    
    rng = np.random.default_rng(random_seed)

    return rng.binomial(n=1, p=probability_success, size=trials)

def binomial_probability(
        trials: int,
        probability_success: float,
        successes: int,
) -> float:
    """
    Calulate binomial probability.

    Formula:
    P(X = k) = C(n,k) p^n (1-p)^(n-k)

    Where
    n = number of trials
    k = successes
    p = probability of success

    Binomial distribution counts the number of successes in repeated independent Bernoulli trials
    """

    if trials <= 0:
        raise ValueError("Number of trials must be positive.")
    
    validate_probability(probability_success)

    if successes<0 or successes>trials:
        return 0.0
    
    combinations = comb(trials, successes)

    probability = (
        combinations
        *(probability_success**successes)
        *((1 - probability_success)**(trials - successes))
    )

    return float(probability)

def simulate_binomial(
        trials: int,
        probability_success: float,
        simulations: int,
        random_seed: int | None = None
) -> np.ndarray:
    """
    Simulate a binomial random variable many times.

    Each simulation counts successes across 'trials' Bernoulli experiments.

    Example:
        simulate_binomial(trials=10, probability_success=0.5, simulations=1000)
    """
    
    if trials < 0:
        raise ValueError("Number of trials must be positive.")
    if simulations <= 0:
        raise ValueError("Number of simulations must be positive.")
    
    validate_probability(probability_success)

    rng = np.random.default_rng(random_seed)

    return rng.binomial(
        n = trials,
        p = probability_success,
        size= simulations, 
    )

def normal_pdf(
        x: float,
        mean: float = 0.0,
        standard_deviation: float = 1.0
) -> float:
    
    """
    Calculate the probability density of a normal distribution at x.

    Normal distribution:

        X ~ N(mean, standard_deviation^2)

    PDF formula:

        f(x) = 1 / (sigma * sqrt(2pi)) *
               exp(-0.5 * ((x - mean) / sigma)^2)

    Important:
        PDF is density, not direct probability at a point.
    """

    if standard_deviation <= 0:
        raise ValueError("Standard Deviation must be positive.")
    
    z = (x - mean)/ standard_deviation

    density = (
        1
        / (standard_deviation * sqrt(2 * pi))
        * exp(-0.5 * z**2)
    )

    return float(density)

def normal_cdf(
        x: float,
        mean: float = 0.0,
        standard_deviation: float = 1.0,
)-> float:
    
    """
    Calculate cumulative probability P(X <= x) for a normal distribution.

    Uses the error function erf.

    For standard normal:

        normal_cdf(0) = 0.5
    """

    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be positive.")
    
    z = (x - mean)/ (standard_deviation * sqrt(2))

    cumulative_probability = 0.5 * (1 + erf(z))

    return float(cumulative_probability)

def sample_normal(
        mean: float = 0.0,
        standard_deviation: float = 1.0,
        samples: int = 1000,
        random_seed: int | None = None,
)-> np.ndarray:
    
    """
    Generate samples from a normal distribution.

    Example:
        sample_normal(mean=0, standard_deviation=1, samples=1000)
    """

    if standard_deviation <= 0:
        raise ValueError("Standard deviation must be positive.")
    
    if samples <= 0:
        raise ValueError("Number of samples must be psotive.")
    
    rng = np.random.default_rng(random_seed)

    return rng.normal(
        loc=mean,
        scale=standard_deviation,
        size=samples,
    )

def expected_value(
        values: Sequence[float],
        probabilities: Sequence[float],
)-> float:
    
    """
    Calculate expected value of a discrete random variable.

    Formula:
        E[X] = sum of x * P(X = x)

    Example:
        values = [0, 10]
        probabilities = [0.5, 0.5]

        E[X] = 0*0.5 + 10*0.5 = 5
    """
    x = np.array(values, dtype=float)
    p = np.array(probabilities, dtype=float)

    if x.ndim != 1 or p.ndim != 1:
        raise ValueError("Values and Probabilites must be one dimensional.")
    
    if x.shape != p.shape:
        raise ValueError("Values and Probabilities must have same shape.")
    if np.any(p<0):
        raise ValueError("Probabilities cannot be negative.")
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    
    return float(np.sum(x * p))

def variance(
        values: Sequence[float],
        probabilities: Sequence[float],
)-> float:
    """
    Calculate variance of a discrete random variable.

    Formula:
        Var(X) = E[(X - E[X])^2]
    """
    x = np.array(values, dtype=float)
    p = np.array(probabilities, dtype=float)

    mean = expected_value(x, p)

    return float(np.sum(((x - mean) ** 2) * p))

def standard_deviation(
    values: Sequence[float],
    probabilities: Sequence[float],
) -> float:
    """
    Calculate standard deviation of a discrete random variable.

    Formula:
        SD(X) = sqrt(Var(X))
    """
    return float(sqrt(variance(values, probabilities)))

def empirical_summary(samples: Sequence[float]) -> Dict[str, float]:
    """
    Summarize empirical samples.

    Empirical means:
        observed from simulated or real data.

    Returns:
        count
        mean
        variance
        standard deviation
        min
        max
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
    }


