"""
The Intelligence Bound
======================

A Python package for computing thermodynamic bounds on learning rate
and simulating biosphere-intelligence coupling.

Main components:
- intelligence_bound: Core bound calculations
- estimators: Data richness (D) estimators
- simulations: Biodiversity impact simulations
- validation: Reproduce all paper results

Example usage:
    from code import landauer_bound, intelligence_bound
    
    # Calculate bounds
    I_max = landauer_bound(power=1.0, temperature=300)
    I_data = intelligence_bound(D=0.05, bandwidth=1e12, power=700, temperature=350)
"""

from .intelligence_bound import (
    landauer_bound,
    data_bound,
    intelligence_bound,
    system_analysis,
    BOLTZMANN_CONSTANT,
    LN2,
)

from .estimators import (
    compression_estimator,
    prediction_estimator,
    mutual_information_estimator,
)

from .simulations import (
    biodiversity_impact,
    strategy_comparison,
    biosphere_information_potential,
)

__version__ = "1.0.0"
__author__ = "Justin Hart"
__email__ = "viridisnorthllc@gmail.com"

__all__ = [
    # Constants
    "BOLTZMANN_CONSTANT",
    "LN2",
    # Core bounds
    "landauer_bound",
    "data_bound", 
    "intelligence_bound",
    "system_analysis",
    # Estimators
    "compression_estimator",
    "prediction_estimator",
    "mutual_information_estimator",
    # Simulations
    "biodiversity_impact",
    "strategy_comparison",
    "biosphere_information_potential",
]
