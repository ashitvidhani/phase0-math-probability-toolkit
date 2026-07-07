# Phase 0 Mathematics + Probability Simulation Toolkit

![Tests](https://github.com/ashitvidhani/phase0-math-probability-toolkit/actions/workflows/tests.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-81.82%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This project is the Phase 0 capstone for my Super Roadmap.

It converts core mathematics into working Python software modules for:

- Linear algebra
- Matrix decompositions
- Optimization
- Probability distributions
- Monte Carlo simulation
- Portfolio risk
- Bayesian updating
- Multivariate Gaussian modeling
- Anomaly detection

---

## Roadmap Context

Current layer: Elite Spine  
Current phase: Phase 0 — Mathematics Core  
Status: Completed 91 / 91 topics  

This project acts as the first mathematical engine that will later support:

- Quant research tools
- Portfolio risk engines
- HFT signal research
- AI-native financial intelligence systems
- Industrial process simulation systems
- Digital twin simulation systems

---

## Project Status

```text
Phase 0 Mathematics Core: Completed 91 / 91
Phase 0 Capstone: Completed and locked
Original roadmap lock: 110 tests passed
Professionalization sprint: 128 tests passed with 81.82% coverage
```

The original Phase 0 capstone remains completed and locked.  
The professionalization sprint improves packaging, documentation, testing, coverage, and open-source readiness.

---

## Project Modules

The reusable package lives in `math_toolkit/`.

```text
math_toolkit/
├── linear_algebra.py
├── decompositions.py
├── optimization.py
├── probability.py
├── monte_carlo.py
├── portfolio_risk.py
├── bayes.py
└── multivariate_gaussian.py
```

### Module Summary

| Module | Purpose |
|---|---|
| `linear_algebra.py` | Vector and matrix operations |
| `decompositions.py` | LU, QR, SVD, low-rank approximation, PCA-style variance analysis |
| `optimization.py` | Numerical derivatives, gradients, gradient descent, regression loss |
| `probability.py` | Bernoulli, binomial, normal distribution, expected value, variance |
| `monte_carlo.py` | Monte Carlo simulation, random walks, asset paths, VaR-style estimates |
| `portfolio_risk.py` | Portfolio return, variance, volatility, covariance/correlation tools |
| `bayes.py` | Bayes theorem, Bayesian updating, odds, likelihood ratios |
| `multivariate_gaussian.py` | Multivariate Gaussian modeling and anomaly detection |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ashitvidhani/phase0-math-probability-toolkit.git
cd phase0-math-probability-toolkit
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

---

## How to Run

Run the main project demo runner:

```bash
python main.py
```

Run selected demos:

```bash
python main.py --demo linear-algebra
python main.py --demo probability
python main.py --demo monte-carlo
python main.py --demo portfolio
```

Run individual example scripts:

```bash
python -m examples.linear_algebra_demo
python -m examples.decompositions_demo
python -m examples.optimization_demo
python -m examples.probability_demo
python -m examples.monte_carlo_demo
python -m examples.portfolio_demo
python -m examples.bayes_demo
python -m examples.anomaly_detection_demo
```

---

## Testing

Run all tests:

```bash
python -m pytest
```

Run tests with coverage:

```bash
pytest --cov=math_toolkit --cov-report=term-missing
```

Generate HTML coverage report:

```bash
pytest --cov=math_toolkit --cov-report=term-missing --cov-report=html
```

The project currently uses an 80% minimum coverage gate.

---

## Documentation

- [Project Explanation](docs/project_explanation.md)
- [Formula Sheet](docs/phase0_formula_sheet.md)
- [Future Integration Notes](docs/future_integration_notes.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)

---

## Quality / Professionalization Features

This repository includes:

- Reusable Python package structure
- `pyproject.toml` package configuration
- GitHub Actions CI
- Test coverage gate
- Documentation folder
- API reference
- Architecture documentation
- Changelog
- MIT License
- Example demo scripts
- Expanded command-line demo runner
- Edge-case tests
- Coverage badge

---

## Final Test Status

Original roadmap lock:

```text
110 passed
```

Professionalization sprint status:

```text
128 passed
Coverage: 81.82%
Coverage gate: 80% minimum
```

---

## Future Integration

This toolkit will later support:

- Quant research engines
- Portfolio risk dashboards
- HFT signal research tools
- AI-native financial intelligence systems
- Industrial anomaly detection systems
- Digital twin simulation systems

---

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.