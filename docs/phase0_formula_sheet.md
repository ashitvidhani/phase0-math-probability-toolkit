# Phase 0 Formula Sheet

## Linear Algebra

Dot product:

a · b = sum(ai bi)

Matrix-vector multiplication:

Ax = b

Rank-nullity:

rank(A) + nullity(A) = number of columns

Eigen equation:

Av = lambda v

SVD:

A = U Sigma V^T

## Optimization

Numerical derivative:

f'(x) ≈ [f(x+h) - f(x-h)] / 2h

Gradient descent:

x_new = x_old - learning_rate × gradient

Mean squared error:

MSE = mean((actual - predicted)^2)

## Probability

Expected value:

E[X] = sum x P(X=x)

Variance:

Var(X) = E[(X - E[X])^2]

Standard deviation:

SD(X) = sqrt(Var(X))

Binomial probability:

P(X=k) = C(n,k) p^k (1-p)^(n-k)

Normal PDF:

f(x) = 1 / (sigma sqrt(2pi)) × exp(-0.5 z^2)

## Portfolio Risk

Portfolio expected return:

E[Rp] = w^T mu

Portfolio variance:

Var(Rp) = w^T Sigma w

Portfolio volatility:

Volatility = sqrt(w^T Sigma w)

## Bayes

Bayes theorem:

P(A|B) = P(B|A)P(A) / P(B)

Binary evidence:

P(B) = P(B|A)P(A) + P(B|not A)P(not A)

## Multivariate Gaussian

X ~ N(mu, Sigma)

Mahalanobis distance:

d = sqrt((x - mu)^T Sigma^-1 (x - mu))