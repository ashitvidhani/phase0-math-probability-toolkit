# API Reference

This document describes the public functions available in the Phase 0 Mathematics + Probability Simulation Toolkit.

## Package

```python
import math_toolkit
```

## Modules

- `math_toolkit.linear_algebra`
- `math_toolkit.decompositions`
- `math_toolkit.optimization`
- `math_toolkit.probability`
- `math_toolkit.monte_carlo`
- `math_toolkit.portfolio_risk`
- `math_toolkit.bayes`
- `math_toolkit.multivariate_gaussian`

---

# `math_toolkit.linear_algebra`

## `vector_magnitude(vector)`

Calculates the Euclidean magnitude of a vector.

```python
from math_toolkit.linear_algebra import vector_magnitude

vector_magnitude([3, 4])
# 5.0
```

## `dot_product(vector_a, vector_b)`

Calculates the dot product of two vectors.

```python
from math_toolkit.linear_algebra import dot_product

dot_product([1, 2, 3], [4, 5, 6])
# 32.0
```

## `matrix_vector_multiply(matrix, vector)`

Multiplies a matrix by a vector.

```python
from math_toolkit.linear_algebra import matrix_vector_multiply

matrix_vector_multiply([[1, 2], [3, 4]], [10, 20])
# array([50., 110.])
```

## `matrix_multiply(matrix_a, matrix_b)`

Multiplies two matrices.

```python
from math_toolkit.linear_algebra import matrix_multiply

matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
# array([[19., 22.],
#        [43., 50.]])
```

## `matrix_rank(matrix)`

Calculates the rank of a matrix.

```python
from math_toolkit.linear_algebra import matrix_rank

matrix_rank([[1, 2], [2, 4]])
# 1
```

## `determinant(matrix)`

Calculates the determinant of a square matrix.

```python
from math_toolkit.linear_algebra import determinant

determinant([[1, 2], [3, 4]])
# -2.0
```

## `inverse(matrix)`

Calculates the inverse of a non-singular square matrix.

```python
from math_toolkit.linear_algebra import inverse

inverse([[1, 2], [3, 4]])
# array([[-2. ,  1. ],
#        [ 1.5, -0.5]])
```

## `eigen_analysis(matrix)`

Returns eigenvalues and eigenvectors of a square matrix.

```python
from math_toolkit.linear_algebra import eigen_analysis

eigenvalues, eigenvectors = eigen_analysis([[2, 0], [0, 3]])
```

---

# `math_toolkit.decompositions`

## `lu_decomposition(matrix)`

Performs LU decomposition.

```python
from math_toolkit.decompositions import lu_decomposition

P, L, U = lu_decomposition([[1, 2], [3, 4]])
```

## `qr_decomposition(matrix)`

Performs QR decomposition.

```python
from math_toolkit.decompositions import qr_decomposition

Q, R = qr_decomposition([[1, 2], [3, 4]])
```

## `svd_decomposition(matrix, full_matrices=False)`

Performs Singular Value Decomposition.

```python
from math_toolkit.decompositions import svd_decomposition

U, singular_values, Vt = svd_decomposition([[1, 2], [3, 4]])
```

## `reconstruct_from_svd(U, singular_values, Vt)`

Reconstructs a matrix from SVD components.

```python
from math_toolkit.decompositions import svd_decomposition, reconstruct_from_svd

U, s, Vt = svd_decomposition([[1, 2], [3, 4]])
A_rebuilt = reconstruct_from_svd(U, s, Vt)
```

## `low_rank_approximation(matrix, rank)`

Creates a low-rank approximation using SVD.

```python
from math_toolkit.decompositions import low_rank_approximation

low_rank_approximation([[1, 2], [3, 4]], rank=1)
```

## `pca_variance_analysis(data)`

Performs PCA-style variance analysis using SVD.

```python
from math_toolkit.decompositions import pca_variance_analysis

components, explained_variance, ratio = pca_variance_analysis([[1, 2], [3, 4], [5, 6]])
```

---

# `math_toolkit.optimization`

## `numerical_derivative(function, x, step=1e-5)`

Estimates the derivative of a single-variable function using central difference.

```python
from math_toolkit.optimization import numerical_derivative

numerical_derivative(lambda x: x**2, x=3)
# approximately 6.0
```

## `numerical_gradient(function, point, step=1e-5)`

Estimates the gradient of a multi-variable function using central difference.

```python
from math_toolkit.optimization import numerical_gradient

numerical_gradient(lambda x: x[0]**2 + x[1]**2, [3, 4])
# approximately array([6., 8.])
```

## `gradient_descent_1d(function, initial_x, learning_rate=0.1, iterations=100, tolerance=1e-8)`

Minimizes a single-variable function using gradient descent.

```python
from math_toolkit.optimization import gradient_descent_1d

best_x, best_value, history = gradient_descent_1d(
    function=lambda x: (x - 3) ** 2,
    initial_x=0,
)
```

## `gradient_descent(function, initial_point, gradient_function=None, learning_rate=0.1, iterations=100, tolerance=1e-8)`

Minimizes a multi-variable function using gradient descent.

```python
from math_toolkit.optimization import gradient_descent

best_point, best_value, history = gradient_descent(
    function=lambda x: x[0]**2 + x[1]**2,
    initial_point=[3, 4],
)
```

## `mean_squared_error(actual, predicted)`

Calculates mean squared error.

```python
from math_toolkit.optimization import mean_squared_error

mean_squared_error([1, 2, 3], [1, 2, 4])
```

## `linear_regression_predictions(params, x_values)`

Generates predictions for a simple linear regression model.

```python
from math_toolkit.optimization import linear_regression_predictions

linear_regression_predictions(params=[2, 1], x_values=[1, 2, 3])
# array([3., 5., 7.])
```

## `linear_regression_loss(params, x_values, y_values)`

Calculates mean squared error loss for simple linear regression.

```python
from math_toolkit.optimization import linear_regression_loss

linear_regression_loss(params=[2, 1], x_values=[1, 2, 3], y_values=[3, 5, 7])
# 0.0
```

---

# `math_toolkit.probability`

## `validate_probability(probability)`

Validates that a probability lies between 0 and 1.

```python
from math_toolkit.probability import validate_probability

validate_probability(0.5)
```

## `bernoulli_trial(probability_success, random_seed=None)`

Simulates one Bernoulli trial.

```python
from math_toolkit.probability import bernoulli_trial

bernoulli_trial(0.5, random_seed=42)
```

## `simulate_bernoulli(probability_success, trials, random_seed=None)`

Simulates many Bernoulli trials.

```python
from math_toolkit.probability import simulate_bernoulli

simulate_bernoulli(probability_success=0.5, trials=10, random_seed=42)
```

## `binomial_probability(trials, probability_success, successes)`

Calculates exact binomial probability.

```python
from math_toolkit.probability import binomial_probability

binomial_probability(trials=3, probability_success=0.5, successes=2)
# 0.375
```

## `simulate_binomial(trials, probability_success, simulations, random_seed=None)`

Simulates binomial random variables.

```python
from math_toolkit.probability import simulate_binomial

simulate_binomial(trials=10, probability_success=0.5, simulations=100, random_seed=42)
```

## `normal_pdf(x, mean=0.0, standard_deviation=1.0)`

Calculates normal distribution probability density.

```python
from math_toolkit.probability import normal_pdf

normal_pdf(0)
# approximately 0.3989
```

## `normal_cdf(x, mean=0.0, standard_deviation=1.0)`

Calculates cumulative normal probability.

```python
from math_toolkit.probability import normal_cdf

normal_cdf(0)
# 0.5
```

## `sample_normal(mean=0.0, standard_deviation=1.0, samples=1000, random_seed=None)`

Generates samples from a normal distribution.

```python
from math_toolkit.probability import sample_normal

sample_normal(mean=0, standard_deviation=1, samples=100, random_seed=42)
```

## `expected_value(values, probabilities)`

Calculates expected value of a discrete random variable.

```python
from math_toolkit.probability import expected_value

expected_value([0, 10], [0.5, 0.5])
# 5.0
```

## `variance(values, probabilities)`

Calculates variance of a discrete random variable.

```python
from math_toolkit.probability import variance

variance([0, 10], [0.5, 0.5])
# 25.0
```

## `standard_deviation(values, probabilities)`

Calculates standard deviation.

```python
from math_toolkit.probability import standard_deviation

standard_deviation([0, 10], [0.5, 0.5])
# 5.0
```

## `empirical_summary(samples)`

Returns summary statistics for empirical samples.

```python
from math_toolkit.probability import empirical_summary

empirical_summary([1, 2, 3, 4, 5])
```

---

# `math_toolkit.monte_carlo`

## `run_monte_carlo(simulation_function, simulations, random_seed=None)`

Runs a generic Monte Carlo simulation.

```python
from math_toolkit.monte_carlo import run_monte_carlo

results = run_monte_carlo(lambda rng: rng.normal(), simulations=1000, random_seed=42)
```

## `estimate_expected_value(samples)`

Estimates expected value from simulation samples.

```python
from math_toolkit.monte_carlo import estimate_expected_value

estimate_expected_value([1, 2, 3])
# 2.0
```

## `estimate_event_probability(samples, event_condition)`

Estimates probability of an event from samples.

```python
from math_toolkit.monte_carlo import estimate_event_probability

estimate_event_probability([1, -1, 2, -2], lambda x: x > 0)
# 0.5
```

## `monte_carlo_summary(samples)`

Summarizes Monte Carlo outcomes.

```python
from math_toolkit.monte_carlo import monte_carlo_summary

monte_carlo_summary([1, 2, 3, 4, 5])
```

## `simulate_random_walk(steps, start_value=0.0, drift=0.0, volatility=1.0, random_seed=None)`

Simulates a random walk.

```python
from math_toolkit.monte_carlo import simulate_random_walk

simulate_random_walk(steps=10, random_seed=42)
```

## `simulate_asset_price_paths(initial_price, drift, volatility, time_horizon, steps, simulations, random_seed=None)`

Simulates asset price paths using geometric Brownian motion.

```python
from math_toolkit.monte_carlo import simulate_asset_price_paths

paths = simulate_asset_price_paths(
    initial_price=100,
    drift=0.05,
    volatility=0.2,
    time_horizon=1,
    steps=252,
    simulations=1000,
    random_seed=42,
)
```

## `percentile_value(samples, percentile)`

Calculates a percentile value from samples.

```python
from math_toolkit.monte_carlo import percentile_value

percentile_value([1, 2, 3, 4, 5], percentile=5)
```

## `value_at_risk(returns, confidence_level=0.95)`

Estimates Value-at-Risk from return samples.

```python
from math_toolkit.monte_carlo import value_at_risk

value_at_risk([0.01, -0.02, 0.03, -0.01], confidence_level=0.95)
```

---

# `math_toolkit.portfolio_risk`

## `validate_weights(weights, allow_short=True)`

Validates portfolio weights.

```python
from math_toolkit.portfolio_risk import validate_weights

validate_weights([0.6, 0.4])
```

## `portfolio_expected_return(weights, expected_returns)`

Calculates portfolio expected return.

```python
from math_toolkit.portfolio_risk import portfolio_expected_return

portfolio_expected_return([0.6, 0.4], [0.10, 0.06])
# 0.084
```

## `portfolio_variance(weights, covariance_matrix)`

Calculates portfolio variance.

```python
from math_toolkit.portfolio_risk import portfolio_variance

portfolio_variance([0.6, 0.4], [[0.04, 0.01], [0.01, 0.02]])
```

## `portfolio_volatility(weights, covariance_matrix)`

Calculates portfolio volatility.

```python
from math_toolkit.portfolio_risk import portfolio_volatility

portfolio_volatility([0.6, 0.4], [[0.04, 0.01], [0.01, 0.02]])
```

## `portfolio_summary(weights, expected_returns, covariance_matrix)`

Returns expected return, variance, and volatility.

```python
from math_toolkit.portfolio_risk import portfolio_summary

portfolio_summary(
    weights=[0.6, 0.4],
    expected_returns=[0.10, 0.06],
    covariance_matrix=[[0.04, 0.01], [0.01, 0.02]],
)
```

## `covariance_to_correlation(covariance_matrix)`

Converts covariance matrix to correlation matrix.

```python
from math_toolkit.portfolio_risk import covariance_to_correlation

covariance_to_correlation([[0.04, 0.01], [0.01, 0.02]])
```

## `correlation_to_covariance(correlation_matrix, standard_deviations)`

Converts correlation matrix to covariance matrix.

```python
from math_toolkit.portfolio_risk import correlation_to_covariance

correlation_to_covariance([[1.0, 0.5], [0.5, 1.0]], [0.2, 0.1])
```

## `portfolio_returns_from_asset_returns(asset_returns, weights)`

Calculates portfolio returns from asset returns.

```python
from math_toolkit.portfolio_risk import portfolio_returns_from_asset_returns

portfolio_returns_from_asset_returns([[0.01, 0.02], [0.03, -0.01]], [0.6, 0.4])
```

## `historical_portfolio_summary(asset_returns, weights)`

Summarizes historical or simulated portfolio returns.

```python
from math_toolkit.portfolio_risk import historical_portfolio_summary

historical_portfolio_summary([[0.01, 0.02], [0.03, -0.01]], [0.6, 0.4])
```

## `variance_contributions(weights, covariance_matrix)`

Calculates each asset's contribution to portfolio variance.

```python
from math_toolkit.portfolio_risk import variance_contributions

variance_contributions([0.6, 0.4], [[0.04, 0.01], [0.01, 0.02]])
```

## `variance_contribution_percentages(weights, covariance_matrix)`

Calculates percentage contribution to portfolio variance.

```python
from math_toolkit.portfolio_risk import variance_contribution_percentages

variance_contribution_percentages([0.6, 0.4], [[0.04, 0.01], [0.01, 0.02]])
```

---

# `math_toolkit.bayes`

## `validate_probability(probability, name="probability")`

Validates that a probability lies between 0 and 1.

```python
from math_toolkit.bayes import validate_probability

validate_probability(0.5)
```

## `bayes_theorem(prior, likelihood, evidence)`

Applies Bayes theorem.

```python
from math_toolkit.bayes import bayes_theorem

bayes_theorem(prior=0.01, likelihood=0.99, evidence=0.05)
```

## `total_probability_binary(prior, likelihood_if_true, likelihood_if_false)`

Calculates total probability of evidence in a binary hypothesis case.

```python
from math_toolkit.bayes import total_probability_binary

total_probability_binary(
    prior=0.01,
    likelihood_if_true=0.99,
    likelihood_if_false=0.05,
)
```

## `binary_bayes_update(prior, likelihood_if_true, likelihood_if_false)`

Applies Bayesian updating for a binary true/false hypothesis.

```python
from math_toolkit.bayes import binary_bayes_update

binary_bayes_update(
    prior=0.01,
    likelihood_if_true=0.99,
    likelihood_if_false=0.05,
)
```

## `binary_bayes_summary(prior, likelihood_if_true, likelihood_if_false)`

Returns a full Bayesian update summary.

```python
from math_toolkit.bayes import binary_bayes_summary

binary_bayes_summary(
    prior=0.01,
    likelihood_if_true=0.99,
    likelihood_if_false=0.05,
)
```

## `posterior_from_likelihoods(priors, likelihoods)`

Calculates posterior probabilities for multiple hypotheses.

```python
from math_toolkit.bayes import posterior_from_likelihoods

posterior_from_likelihoods(
    priors=[0.5, 0.5],
    likelihoods=[0.8, 0.2],
)
```

## `probability_to_odds(probability)`

Converts probability to odds.

```python
from math_toolkit.bayes import probability_to_odds

probability_to_odds(0.75)
# 3.0
```

## `odds_to_probability(odds)`

Converts odds to probability.

```python
from math_toolkit.bayes import odds_to_probability

odds_to_probability(3)
# 0.75
```

## `likelihood_ratio_update(prior_probability, likelihood_ratio)`

Updates probability using a likelihood ratio.

```python
from math_toolkit.bayes import likelihood_ratio_update

likelihood_ratio_update(prior_probability=0.2, likelihood_ratio=3)
```

---

# `math_toolkit.multivariate_gaussian`

## `validate_covariance_matrix(covariance_matrix, dimension=None, require_positive_definite=False)`

Validates that a covariance matrix is square, symmetric, and positive semi-definite or positive definite.

```python
from math_toolkit.multivariate_gaussian import validate_covariance_matrix

validate_covariance_matrix([[1, 0], [0, 1]])
```

## `estimate_mean_vector(data)`

Estimates mean vector from data.

```python
from math_toolkit.multivariate_gaussian import estimate_mean_vector

estimate_mean_vector([[1, 2], [3, 4], [5, 6]])
# array([3., 4.])
```

## `estimate_covariance_matrix(data)`

Estimates covariance matrix from data.

```python
from math_toolkit.multivariate_gaussian import estimate_covariance_matrix

estimate_covariance_matrix([[1, 2], [3, 4], [5, 6]])
```

## `fit_multivariate_gaussian(data)`

Fits a multivariate Gaussian model.

```python
from math_toolkit.multivariate_gaussian import fit_multivariate_gaussian

model = fit_multivariate_gaussian([[1, 2], [3, 4], [5, 6]])
```

## `squared_mahalanobis_distance(point, mean, covariance_matrix)`

Calculates squared Mahalanobis distance.

```python
from math_toolkit.multivariate_gaussian import squared_mahalanobis_distance

squared_mahalanobis_distance(
    point=[1, 1],
    mean=[0, 0],
    covariance_matrix=[[1, 0], [0, 1]],
)
```

## `mahalanobis_distance(point, mean, covariance_matrix)`

Calculates Mahalanobis distance.

```python
from math_toolkit.multivariate_gaussian import mahalanobis_distance

mahalanobis_distance(
    point=[1, 1],
    mean=[0, 0],
    covariance_matrix=[[1, 0], [0, 1]],
)
```

## `multivariate_normal_pdf(point, mean, covariance_matrix)`

Calculates multivariate normal probability density.

```python
from math_toolkit.multivariate_gaussian import multivariate_normal_pdf

multivariate_normal_pdf(
    point=[0, 0],
    mean=[0, 0],
    covariance_matrix=[[1, 0], [0, 1]],
)
```

## `sample_multivariate_normal(mean, covariance_matrix, samples=1000, random_seed=None)`

Generates samples from a multivariate Gaussian distribution.

```python
from math_toolkit.multivariate_gaussian import sample_multivariate_normal

sample_multivariate_normal(
    mean=[0, 0],
    covariance_matrix=[[1, 0], [0, 1]],
    samples=100,
    random_seed=42,
)
```

## `anomaly_scores(data, mean, covariance_matrix)`

Calculates anomaly scores using Mahalanobis distance.

```python
from math_toolkit.multivariate_gaussian import anomaly_scores

anomaly_scores(
    data=[[0, 0], [1, 1], [10, 10]],
    mean=[0, 0],
    covariance_matrix=[[1, 0], [0, 1]],
)
```

## `detect_anomalies(data, mean, covariance_matrix, threshold=3.0)`

Detects anomalies using a Mahalanobis-distance threshold.

```python
from math_toolkit.multivariate_gaussian import detect_anomalies

detect_anomalies(
    data=[[0, 0], [1, 1], [10, 10]],
    mean=[0, 0],
    covariance_matrix=[[1, 0], [0, 1]],
    threshold=3.0,
)
```