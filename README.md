# The Intelligence Bound

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/viridis-llc/intelligence-bound/actions/workflows/ci.yml/badge.svg)](https://github.com/viridis-llc/intelligence-bound/actions/workflows/ci.yml)

**Thermodynamic Limits on Learning Rate and Implications for Biosphere Information**

This repository contains the paper, computational validation code, and supplementary materials for "The Intelligence Bound" by Justin Hart (Viridis LLC).

## Abstract

We derive a fundamental upper bound on the rate at which any physical system can create intelligence:

$$\dot{I} \leq \min\left(D \cdot B, \frac{P}{k_B T \ln 2}\right)$$

where:
- $D \in [0,1]$ is the **data richness** (predictive fraction of observations)
- $B$ is observation bandwidth [bits/s]
- $P$ is available power [W]
- $T$ is temperature [K]
- $k_B$ is Boltzmann's constant

### Key Results

1. **The Intelligence Bound**: All current systems (brains, GPUs) operate 10¹²–10¹⁵× below the thermodynamic ceiling. Data quality, not compute, is the bottleneck.

2. **Gaia-Intelligence Proposition**: Earth's biosphere (~10¹⁵ bits of genetic information) constitutes the highest-D information source available to terrestrial intelligence.

3. **Convergent Conservation Theorem**: Any sufficiently advanced AI system optimizing long-term learning rate must preserve biosphere data richness as an instrumental goal—not from ethical programming but from information-theoretic necessity.

## Repository Structure

```
intelligence-bound/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── paper/
│   ├── intelligence_bound_v11.tex    # LaTeX source
│   ├── intelligence_bound_v11.pdf    # Compiled paper
│   └── figures/                       # Paper figures
├── code/
│   ├── __init__.py
│   ├── intelligence_bound.py         # Core computations
│   ├── estimators.py                 # D estimators
│   ├── simulations.py                # Biodiversity simulations
│   └── validation.py                 # Reproduce all paper results
├── tests/
│   ├── test_bounds.py                # Unit tests
│   ├── test_estimators.py
│   └── test_simulations.py
├── data/
│   └── README.md                     # Data sources documentation
└── notebooks/
    └── reproduce_results.ipynb       # Interactive reproduction
```

## Installation

```bash
# Clone the repository
git clone https://github.com/viridis-llc/intelligence-bound.git
cd intelligence-bound

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Reproduce All Paper Results

```bash
python -m code.validation
```

This will generate all tables from the paper and save them to `output/`.

### Python API

```python
from code.intelligence_bound import (
    landauer_bound,
    data_bound,
    intelligence_bound,
    system_analysis
)

# Calculate Landauer bound for 1W at room temperature
I_max = landauer_bound(power=1.0, temperature=300)
print(f"Landauer bound: {I_max:.2e} bits/s")

# Analyze a real system
analysis = system_analysis(
    name="NVIDIA H100",
    power=700,           # Watts
    temperature=350,     # Kelvin
    bandwidth=3e12,      # bits/s (HBM3)
    D=0.05              # Internet text data richness
)
print(analysis)
```

### Estimate Data Richness D

```python
from code.estimators import (
    compression_estimator,
    prediction_estimator,
    mutual_information_estimator
)
import numpy as np

# Generate sample data
observations = np.random.randn(10000)  # Replace with real data

# Compression-based estimate (upper bound)
D_compress = compression_estimator(observations)

# Prediction-based estimate (tighter)
targets = observations[1:]  # Next-step prediction
D_predict = prediction_estimator(observations[:-1], targets)

print(f"D_compress: {D_compress:.3f}")
print(f"D_predict: {D_predict:.3f}")
```

### Run Biodiversity Simulations

```python
from code.simulations import (
    biodiversity_impact,
    strategy_comparison,
    plot_results
)

# Simulate biodiversity loss impact on D
results = biodiversity_impact(
    D0=0.5,
    loss_fractions=[0.1, 0.25, 0.5, 0.75, 0.9]
)
print(results)

# Compare exploitation vs. preservation strategies
strategies = strategy_comparison(
    D0=0.5,
    time_horizon=200,  # years
    bandwidth=1e12     # bits/s
)
print(strategies)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=code --cov-report=html

# Run specific test file
pytest tests/test_bounds.py -v
```

## Falsifiable Predictions

The paper makes several testable predictions:

| Prediction | Test Method | Status |
|------------|-------------|--------|
| Learning rate ∝ D (data-limited regime) | Train models on datasets with measured D | Open |
| İ × T = constant (Landauer regime) | Cryogenic computing experiments | Open |
| Phase transition at P* = D·B·k_B·T·ln(2) | Vary compute intensity | Open |
| Higher biodiversity → higher D | Compare intact vs. degraded ecosystems | Open |
| AlphaFold extraction ≤ D_bio · B_seq | Measure info gain per structure | Open |

## Citation

If you use this work, please cite:

```bibtex
@article{hart2025intelligence,
  title={The Intelligence Bound: Thermodynamic Limits on Learning Rate 
         and Implications for Biosphere Information},
  author={Hart, Justin},
  journal={arXiv preprint},
  year={2025},
  note={Viridis LLC}
}
```

## Related Work

- Landauer, R. (1961). Irreversibility and heat generation in the computing process
- Bérut, A. et al. (2012). Experimental verification of Landauer's principle
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Bostrom, N. (2014). Superintelligence: Paths, Dangers, Strategies
- Hoffmann, J. et al. (2022). Training compute-optimal large language models

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Justin Hart
- **Organization**: Viridis LLC, Vermont, USA
- **Email**: viridisnorthllc@gmail.com

## Acknowledgments

Computational validation code reproducing all numerical results in the paper is provided in this repository. The theoretical derivations are fully contained within the paper.
