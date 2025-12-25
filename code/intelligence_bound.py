"""
Intelligence Bound: Core Computations
=====================================

This module implements the Intelligence Bound theorem:

    İ ≤ min(D · B, P / (k_B T ln 2))

where:
    İ = intelligence creation rate [bits/s]
    D = data richness (predictive fraction) ∈ [0, 1]
    B = observation bandwidth [bits/s]
    P = power [W]
    T = temperature [K]
    k_B = Boltzmann constant [J/K]

Author: Justin Hart, Viridis LLC
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K (exact, SI definition)
LN2 = np.log(2)  # ≈ 0.693147


@dataclass
class SystemAnalysis:
    """Results of analyzing a physical learning system."""

    name: str
    power: float  # W
    temperature: float  # K
    bandwidth: float  # bits/s
    D: float  # data richness ∈ [0, 1]

    landauer_bound: float  # bits/s
    data_bound: float  # bits/s
    effective_bound: float  # bits/s (minimum)
    headroom: float  # ratio of Landauer to data bound
    limiting_factor: str  # "DATA" or "POWER"

    def __str__(self) -> str:
        return (
            f"System Analysis: {self.name}\n"
            f"{'=' * 40}\n"
            f"Power:          {self.power:.1f} W\n"
            f"Temperature:    {self.temperature:.1f} K\n"
            f"Bandwidth:      {self.bandwidth:.2e} bits/s\n"
            f"Data Richness:  {self.D:.4f}\n"
            f"\n"
            f"Landauer Bound: {self.landauer_bound:.2e} bits/s\n"
            f"Data Bound:     {self.data_bound:.2e} bits/s\n"
            f"Effective:      {self.effective_bound:.2e} bits/s\n"
            f"\n"
            f"Headroom:       {self.headroom:.2e}×\n"
            f"Limiting:       {self.limiting_factor}\n"
        )


def landauer_bound(power: float, temperature: float) -> float:
    """
    Calculate the Landauer (thermodynamic) bound on intelligence creation rate.

    The Landauer bound arises from the minimum energy cost of erasing
    one bit of information: E_min = k_B T ln(2).

    Parameters
    ----------
    power : float
        Available power in Watts [W]
    temperature : float
        Operating temperature in Kelvin [K]

    Returns
    -------
    float
        Maximum intelligence creation rate [bits/s]

    Examples
    --------
    >>> landauer_bound(power=1.0, temperature=300)
    3.48e+20

    Notes
    -----
    The bound is: İ_max = P / (k_B T ln 2)

    At room temperature (300 K), 1 Watt allows ~3.5×10²⁰ bit operations/s.
    This is an absolute ceiling that no physical computer can exceed.
    """
    if power < 0:
        raise ValueError(f"Power must be non-negative, got {power}")
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return power / (BOLTZMANN_CONSTANT * temperature * LN2)


def data_bound(D: float, bandwidth: float) -> float:
    """
    Calculate the data (informational) bound on intelligence creation rate.

    The data bound arises from the fact that you can only learn as fast
    as your environment provides learnable structure.

    Parameters
    ----------
    D : float
        Data richness (predictive fraction) ∈ [0, 1]
    bandwidth : float
        Observation bandwidth in bits per second [bits/s]

    Returns
    -------
    float
        Maximum intelligence creation rate [bits/s]

    Examples
    --------
    >>> data_bound(D=0.05, bandwidth=1e12)
    5e+10

    Notes
    -----
    The bound is: İ_max = D · B

    D measures what fraction of observations contain predictive structure.
    Random noise has D ≈ 0; highly structured signals have D ≈ 1.
    """
    if not 0 <= D <= 1:
        raise ValueError(f"D must be in [0, 1], got {D}")
    if bandwidth < 0:
        raise ValueError(f"Bandwidth must be non-negative, got {bandwidth}")

    return D * bandwidth


def intelligence_bound(
    D: float, bandwidth: float, power: float, temperature: float
) -> Tuple[float, str]:
    """
    Calculate the Intelligence Bound (Theorem 1).

    The Intelligence Bound is the minimum of the data bound and Landauer bound:

        İ ≤ min(D · B, P / (k_B T ln 2))

    Parameters
    ----------
    D : float
        Data richness (predictive fraction) ∈ [0, 1]
    bandwidth : float
        Observation bandwidth [bits/s]
    power : float
        Available power [W]
    temperature : float
        Operating temperature [K]

    Returns
    -------
    bound : float
        Maximum intelligence creation rate [bits/s]
    limiting_factor : str
        "DATA" if data-limited, "POWER" if Landauer-limited

    Examples
    --------
    >>> bound, limiting = intelligence_bound(D=0.05, bandwidth=1e12, power=700, temperature=350)
    >>> print(f"{bound:.2e}, {limiting}")
    5.00e+10, DATA
    """
    I_landauer = landauer_bound(power, temperature)
    I_data = data_bound(D, bandwidth)

    if I_data <= I_landauer:
        return I_data, "DATA"
    else:
        return I_landauer, "POWER"


def system_analysis(
    name: str, power: float, temperature: float, bandwidth: float, D: float
) -> SystemAnalysis:
    """
    Perform complete analysis of a physical learning system.

    Parameters
    ----------
    name : str
        System name for display
    power : float
        Available power [W]
    temperature : float
        Operating temperature [K]
    bandwidth : float
        Observation bandwidth [bits/s]
    D : float
        Data richness ∈ [0, 1]

    Returns
    -------
    SystemAnalysis
        Dataclass containing all computed quantities

    Examples
    --------
    >>> analysis = system_analysis(
    ...     name="Human Brain",
    ...     power=20,
    ...     temperature=310,
    ...     bandwidth=1e9,
    ...     D=1e-3
    ... )
    >>> print(analysis)
    """
    I_landauer = landauer_bound(power, temperature)
    I_data = data_bound(D, bandwidth)
    effective, limiting = intelligence_bound(D, bandwidth, power, temperature)
    headroom = I_landauer / I_data if I_data > 0 else float("inf")

    return SystemAnalysis(
        name=name,
        power=power,
        temperature=temperature,
        bandwidth=bandwidth,
        D=D,
        landauer_bound=I_landauer,
        data_bound=I_data,
        effective_bound=effective,
        headroom=headroom,
        limiting_factor=limiting,
    )


def temperature_scaling_table(
    power: float = 1.0, temperatures: Optional[list] = None
) -> Dict[str, list]:
    """
    Generate Table I: Temperature scaling validation.

    Demonstrates that İ_max × T = constant (Landauer prediction).

    Parameters
    ----------
    power : float
        Power in Watts (default: 1.0 W)
    temperatures : list, optional
        List of temperatures in Kelvin
        Default: [4, 77, 300, 1000] (liquid He, liquid N₂, room temp, high)

    Returns
    -------
    dict
        Dictionary with columns: temperature, I_max, I_max_times_T
    """
    if temperatures is None:
        temperatures = [4, 77, 300, 1000]

    results = {"temperature_K": [], "I_max_bits_per_s": [], "I_max_times_T": []}

    for T in temperatures:
        I_max = landauer_bound(power, T)
        results["temperature_K"].append(T)
        results["I_max_bits_per_s"].append(I_max)
        results["I_max_times_T"].append(I_max * T)

    return results


def real_systems_table() -> Dict[str, SystemAnalysis]:
    """
    Generate Table II: Real system analysis.

    Analyzes human brain, NVIDIA H100, and theoretical maximum.

    Returns
    -------
    dict
        Dictionary mapping system name to SystemAnalysis
    """
    systems = {}

    # Human Brain
    systems["Human Brain"] = system_analysis(
        name="Human Brain",
        power=20,  # W (metabolic)
        temperature=310,  # K (body temperature)
        bandwidth=1e9,  # bits/s (sensory bandwidth)
        D=1e-3,  # Most sensory input is redundant
    )

    # NVIDIA H100
    systems["NVIDIA H100"] = system_analysis(
        name="NVIDIA H100",
        power=700,  # W (TDP)
        temperature=350,  # K (operating)
        bandwidth=2.4e13,  # bits/s (3 TB/s = 3e12 bytes/s = 2.4e13 bits/s)
        D=0.03,  # D_text→world (text as channel for physical-world prediction)
    )

    # Theoretical Maximum (optimistic assumptions)
    systems["Theoretical Max"] = system_analysis(
        name="Theoretical Max",
        power=1e6,  # W (1 MW data center)
        temperature=300,  # K
        bandwidth=1e15,  # bits/s (petabit/s)
        D=0.5,  # High-quality curated data
    )

    return systems


def critical_power(D: float, bandwidth: float, temperature: float) -> float:
    """
    Calculate the critical power P* at which the system transitions
    from data-limited to power-limited regime.

    At P = P*, the data bound equals the Landauer bound:
        D · B = P* / (k_B T ln 2)

    Parameters
    ----------
    D : float
        Data richness ∈ [0, 1]
    bandwidth : float
        Observation bandwidth [bits/s]
    temperature : float
        Operating temperature [K]

    Returns
    -------
    float
        Critical power P* [W]

    Notes
    -----
    For typical systems, P* is astronomically high, which is why
    all current systems are data-limited, not power-limited.
    """
    return D * bandwidth * BOLTZMANN_CONSTANT * temperature * LN2


if __name__ == "__main__":
    # Quick validation
    print("Temperature Scaling (Table I)")
    print("=" * 50)
    table = temperature_scaling_table()
    for i in range(len(table["temperature_K"])):
        print(
            f"T = {table['temperature_K'][i]:4d} K: "
            f"İ_max = {table['I_max_bits_per_s'][i]:.2e} bits/s, "
            f"İ×T = {table['I_max_times_T'][i]:.2e}"
        )

    print("\nReal Systems Analysis (Table II)")
    print("=" * 50)
    systems = real_systems_table()
    for name, analysis in systems.items():
        print(f"\n{analysis}")
