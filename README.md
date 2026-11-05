# The Intelligence Bound

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/jdhart81/viridis-llc-intelligence-bound/actions/workflows/ci.yml/badge.svg)](https://github.com/jdhart81/viridis-llc-intelligence-bound/actions/workflows/ci.yml)

**Thermodynamic Limits on Learning Rate and Implications for Biosphere Information**

> **Status: the Intelligence Bound is a conjecture. This code evaluates its formula; it does not test it.**
>
> - **What the code does:** it computes P/(k_B T ln 2), D·B and their minimum for
>   input values written into the code (brain, H100, a 1 MW data center), and runs
>   simple models of how D might change under biodiversity loss or different
>   strategies. The "validation" checks confirm that the code reproduces the paper's
>   tables and arithmetic. They do not compare the bound with any measured learning
>   rate.
> - **Assumed:** the bound itself, and every input value (D, B, P, T for each system,
>   the cascade exponent, decay and growth rates). These are illustrative, not
>   measured.
> - **Machine-checked elsewhere:** [intelligence-bound-lean](https://github.com/jdhart81/intelligence-bound-lean)
>   checks the conditional mathematics in Lean 4. The thermodynamic arm there is the
>   assumed premise `SatisfiesLandauerLimit` (P ≥ İ · k_B T ln 2), not a derived
>   fact. Landauer's principle prices erasing a bit, not acquiring one, so the
>   thermodynamic arm applies only to a learner that must erase at least as fast as
>   it learns.
>
> Active development continues in [Viridis Canon](https://github.com/jdhart81/viridis-canon), the Lean 4 research spine of the Viridis program.

This repository contains reference computations and supplementary materials for "The Intelligence Bound" by Justin Hart (Viridis LLC).

## Abstract

We propose an upper bound on the rate at which a physical system can create intelligence (the Intelligence Bound conjecture):

$$\dot{I} \leq \min\left(D \cdot B, \frac{P}{k_B T \ln 2}\right)$$

where:
- $D \in [0,1]$ is the **data richness** (predictive fraction of observations)
- $B$ is observation bandwidth [bits/s]
- $P$ is available power [W]
- $T$ is temperature [K]
- $k_B$ is Boltzmann's constant

### Key Results

1. **The Intelligence Bound (conjecture)**: Under the illustrative inputs in Table II, the Landauer term P/(k_B T ln 2) exceeds the data term D·B by about 3×10¹¹ to 7×10¹⁵ (H100 2.9×10¹¹, 1 MW data center 7.0×10¹¹, human brain 6.7×10¹⁵). If the bound holds, data quality rather than power is the binding term for these systems. This compares two upper bounds; it is not a measurement of how fast any system learns.

2. **Gaia-Intelligence Proposition (hypothesis)**: Earth's biosphere (~10¹⁵ bits of genetic information, an order-of-magnitude estimate) is proposed as the highest-D information source available to terrestrial intelligence. D has not been measured for the biosphere or for text corpora, so this is untested.

3. **Conditional conservation (model result)**: In a model where preservation keeps D constant and exploitation makes it decay exponentially, cumulative learning from preservation eventually exceeds exploitation. The Lean version (`conditional_conservation`) proves that a linear function eventually exceeds a bounded one. Whether advanced AI systems would preserve the biosphere depends on whether real systems match the model; that is a hypothesis, not a theorem.

## Repository Structure

```
intelligence-bound/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── code/
│   ├── __init__.py
│   ├── intelligence_bound.py         # Core computations
│   ├── estimators.py                 # D estimators
│   ├── simulations.py                # Biodiversity simulations
│   └── validation.py                 # Reproduce the paper's tables
├── tests/
│   ├── test_bounds.py                # Unit tests
│   ├── test_estimators.py
│   └── test_simulations.py
├── data/
│   └── README.md                     # Data sources documentation
├── output/                   # Generated tables
└── notebooks/
    └── reproduce_results.ipynb       # Interactive reproduction
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jdhart81/viridis-llc-intelligence-bound.git
cd viridis-llc-intelligence-bound

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### Reproduce the Paper's Tables

```bash
python -m code.validation
```

This regenerates Tables I, II, IV and V and saves them to `output/`. Each check
confirms that the code reproduces a number or ordering stated in the paper from
the paper's own inputs. None of them is an empirical test of the bound.

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
    bandwidth=2.4e13,    # bits/s (3 TB/s = 3e12 bytes/s = 2.4e13 bits/s)
    D=0.03              # D_text→world (text as channel for physical-world prediction)
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

If the conjecture is right, these should hold. None has been tested yet:

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
@misc{hart2025intelligence,
  title={The Intelligence Bound: Thermodynamic Limits on Learning Rate
         and Implications for Biosphere Information},
  author={Hart, Justin D.},
  year={2025},
  doi={10.5281/zenodo.23141592},
  note={Viridis LLC. The physical bound is stated as a conjecture.}
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

Code reproducing the paper's numerical tables is provided in this repository. Run `python -m code.validation` to regenerate them. The tables follow from the bound's formula and the stated inputs; the conditional mathematics is checked in Lean in [intelligence-bound-lean](https://github.com/jdhart81/intelligence-bound-lean).
