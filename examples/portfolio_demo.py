"""
Demo for Portfolio Risk Toolkit

Run from project root:

    python -m examples.portfolio_demo
"""

import numpy as np

from math_toolkit.portfolio_risk import (
    correlation_to_covariance,
    covariance_to_correlation,
    historical_portfolio_summary,
    portfolio_expected_return,
    portfolio_returns_from_asset_returns,
    portfolio_summary,
    portfolio_variance,
    portfolio_volatility,
    variance_contribution_percentages,
    variance_contributions,
)


def main():
    print("=== Portfolio Risk Toolkit Demo ===")

    weights = [0.6, 0.4]
    expected_returns = [0.10, 0.05]

    covariance_matrix = [
        [0.04, 0.006],
        [0.006, 0.09],
    ]

    print("\nPortfolio Inputs")
    print("Weights:", weights)
    print("Expected returns:", expected_returns)
    print("Covariance matrix:")
    print(np.array(covariance_matrix))

    print("\nPortfolio Expected Return")
    print(portfolio_expected_return(weights, expected_returns))

    print("\nPortfolio Variance")
    print(portfolio_variance(weights, covariance_matrix))

    print("\nPortfolio Volatility")
    print(portfolio_volatility(weights, covariance_matrix))

    print("\nPortfolio Summary")
    print(portfolio_summary(weights, expected_returns, covariance_matrix))

    print("\nCorrelation Matrix")
    correlation_matrix = covariance_to_correlation(covariance_matrix)
    print(correlation_matrix)

    print("\nCovariance Matrix Reconstructed From Correlation")
    standard_deviations = [0.20, 0.30]
    reconstructed_covariance = correlation_to_covariance(
        correlation_matrix,
        standard_deviations,
    )
    print(reconstructed_covariance)

    print("\nVariance Contributions")
    contributions = variance_contributions(weights, covariance_matrix)
    print(contributions)
    print("Sum of contributions:", np.sum(contributions))

    print("\nVariance Contribution Percentages")
    contribution_percentages = variance_contribution_percentages(
        weights,
        covariance_matrix,
    )
    print(contribution_percentages)
    print("Sum of percentages:", np.sum(contribution_percentages))

    asset_returns = [
        [0.01, 0.02],
        [-0.01, 0.00],
        [0.03, 0.01],
        [0.02, -0.02],
        [0.00, 0.01],
    ]

    print("\nAsset Returns")
    print(np.array(asset_returns))

    portfolio_returns = portfolio_returns_from_asset_returns(
        asset_returns,
        weights,
    )

    print("\nPortfolio Returns")
    print(portfolio_returns)

    print("\nHistorical Portfolio Summary")
    print(historical_portfolio_summary(asset_returns, weights))


if __name__ == "__main__":
    main()