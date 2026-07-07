# Architecture

The Phase 0 Mathematics + Probability Simulation Toolkit is organized as a reusable Python package.

```text
phase0-math-probability-toolkit/
│
├── math_toolkit/
│   ├── linear_algebra.py
│   ├── decompositions.py
│   ├── optimization.py
│   ├── probability.py
│   ├── monte_carlo.py
│   ├── portfolio_risk.py
│   ├── bayes.py
│   └── multivariate_gaussian.py
│
├── examples/
│   ├── linear_algebra_demo.py
│   ├── decompositions_demo.py
│   ├── optimization_demo.py
│   ├── probability_demo.py
│   ├── monte_carlo_demo.py
│   ├── portfolio_demo.py
│   ├── bayes_demo.py
│   └── anomaly_detection_demo.py
│
├── tests/
│   └── test_*.py
│
├── docs/
│   ├── future_integration_notes.md
│   ├── phase0_formula_sheet.md
│   ├── project_explanation.md
│   ├── API_REFERENCE.md
│   └── ARCHITECTURE.md
│
├── main.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## Design Principle

The project separates reusable mathematical logic from examples, tests, and documentation.

```text
math_toolkit/  -> reusable library code
examples/      -> demonstration scripts
tests/         -> correctness checks
docs/          -> user/developer documentation
main.py        -> command-line demo entry point
```

## Module Flow

```text
Mathematical primitives
        ↓
Reusable toolkit modules
        ↓
Examples and demos
        ↓
Tests and coverage
        ↓
Future quant / AI / finance / industrial intelligence systems
```

## Core Package

The `math_toolkit/` package contains the reusable mathematical engine.

### `linear_algebra.py`

Covers vector and matrix fundamentals:

- Vector magnitude
- Dot product
- Matrix-vector multiplication
- Matrix multiplication
- Matrix rank
- Determinant
- Matrix inverse
- Eigenvalue and eigenvector analysis

### `decompositions.py`

Covers matrix decomposition techniques:

- LU decomposition
- QR decomposition
- Singular Value Decomposition
- SVD reconstruction
- Low-rank approximation
- PCA-style variance analysis

### `optimization.py`

Covers numerical optimization tools:

- Numerical derivative
- Numerical gradient
- One-dimensional gradient descent
- Multi-variable gradient descent
- Mean squared error
- Linear regression predictions
- Linear regression loss

### `probability.py`

Covers probability and statistics primitives:

- Probability validation
- Bernoulli trials
- Binomial probability
- Binomial simulation
- Normal PDF
- Normal CDF
- Normal sampling
- Expected value
- Variance
- Standard deviation
- Empirical summary

### `monte_carlo.py`

Covers simulation-based estimation:

- Generic Monte Carlo simulation
- Expected value estimation
- Event probability estimation
- Monte Carlo summary
- Random walk simulation
- Asset price path simulation
- Percentile calculation
- Value-at-Risk style estimation

### `portfolio_risk.py`

Covers portfolio mathematics:

- Portfolio weight validation
- Portfolio expected return
- Portfolio variance
- Portfolio volatility
- Portfolio summary
- Covariance to correlation conversion
- Correlation to covariance conversion
- Portfolio returns from asset returns
- Historical portfolio summary
- Variance contributions
- Variance contribution percentages

### `bayes.py`

Covers Bayesian probability updating:

- Probability validation
- Bayes theorem
- Binary Bayes update
- Total probability
- Multi-hypothesis posterior calculation
- Probability-to-odds conversion
- Odds-to-probability conversion
- Likelihood-ratio update
- Bayesian diagnostic summaries

### `multivariate_gaussian.py`

Covers multivariate probability and anomaly detection:

- Mean vector estimation
- Covariance matrix estimation
- Covariance matrix validation
- Multivariate Gaussian fitting
- Squared Mahalanobis distance
- Mahalanobis distance
- Multivariate normal PDF
- Multivariate normal sampling
- Anomaly scoring
- Anomaly detection

## Examples Layer

The `examples/` folder contains runnable demonstration scripts.

These scripts show how the toolkit modules can be used in practice without modifying the core library code.

```text
examples/
├── linear_algebra_demo.py
├── decompositions_demo.py
├── optimization_demo.py
├── probability_demo.py
├── monte_carlo_demo.py
├── portfolio_demo.py
├── bayes_demo.py
└── anomaly_detection_demo.py
```

## Tests Layer

The `tests/` folder validates correctness.

The test suite covers:

- Normal use cases
- Edge cases
- Invalid inputs
- Numerical correctness
- Error handling
- Probability and statistical behavior
- Portfolio and risk calculations
- Multivariate Gaussian and anomaly detection behavior

Current professionalization target:

```text
Minimum coverage gate: 80%
Current verified status: 128 tests passed with 81.82% coverage
```

## Documentation Layer

The `docs/` folder contains supporting documentation.

```text
docs/
├── future_integration_notes.md
├── phase0_formula_sheet.md
├── project_explanation.md
├── API_REFERENCE.md
└── ARCHITECTURE.md
```

Purpose of each file:

- `project_explanation.md` explains the project purpose and roadmap context.
- `phase0_formula_sheet.md` captures important formulas.
- `future_integration_notes.md` explains how this toolkit can support later roadmap systems.
- `API_REFERENCE.md` documents public functions.
- `ARCHITECTURE.md` explains the project structure and design.

## Command-Line Entry Point

`main.py` acts as the project entry point.

It should provide a simple way to run selected demos from the terminal.

Example usage:

```bash
python main.py
python main.py --demo probability
python main.py --demo portfolio
```

## Packaging Layer

The project uses `pyproject.toml` for modern Python packaging.

This allows the project to be installed locally using:

```bash
pip install -e ".[dev]"
```

This makes the package importable as:

```python
import math_toolkit
```

## Continuous Integration

GitHub Actions is used to run tests automatically on pushes and pull requests.

Expected CI behavior:

```text
1. Checkout repository
2. Set up Python
3. Install package with development dependencies
4. Run pytest
5. Run coverage checks
```

## Quality Gates

The professional-grade project standard includes:

- Clean package structure
- Installable local package
- Test suite
- Coverage gate
- Documentation
- API reference
- Architecture documentation
- Example demos
- Changelog
- License
- GitHub Actions CI
- Reproducible setup instructions

## Future Integration

This toolkit can later support:

- Quant research engines
- Portfolio risk systems
- HFT signal research
- AI-native financial intelligence tools
- Industrial anomaly detection
- Digital twin simulation systems

## Design Philosophy

This project is not only a collection of formulas.

It converts Phase 0 mathematics into reusable software components.

```text
Mathematics learned
        ↓
Mathematics implemented
        ↓
Mathematics tested
        ↓
Mathematics reused in future systems
```

## Current Status

```text
Phase 0 Mathematics Core: Completed 91 / 91
Phase 0 Capstone: Completed and locked
Original roadmap lock: 110 tests passed
Professionalization sprint: 128 tests passed with 81.82% coverage
```