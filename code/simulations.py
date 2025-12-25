"""
Biodiversity-Intelligence Simulations
======================================

This module implements simulations for the Gaia-Intelligence Proposition
and Convergent Conservation Theorem.

Key simulations:
1. Biodiversity loss impact on D (Table IV)
2. Strategy comparison: Exploit vs. Sustain vs. Regenerate (Table V)
3. Biosphere information potential trajectories

Author: Justin Hart, Viridis LLC
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class BiodiversityImpact:
    """Results of biodiversity loss simulation."""

    loss_fraction: float
    D_before: float
    D_after: float
    ceiling_before: float
    ceiling_after: float
    ceiling_reduction_pct: float


@dataclass
class StrategyResult:
    """Results of strategy simulation over time horizon."""

    name: str
    D_trajectory: np.ndarray
    time_points: np.ndarray
    total_intelligence: float
    final_D: float
    outcome: str  # "Collapse", "Stable", "Growth"


def cascade_model(
    D0: float, loss_fraction: float, cascade_exponent: float = 2.0
) -> float:
    """
    Model nonlinear biodiversity-D coupling with cascade effects.

    As species are lost, ecological networks degrade nonlinearly,
    causing accelerating loss of predictive structure.

    Parameters
    ----------
    D0 : float
        Initial data richness
    loss_fraction : float
        Fraction of species lost ∈ [0, 1]
    cascade_exponent : float
        Controls nonlinearity (higher = more severe cascades)

    Returns
    -------
    float
        Data richness after loss

    Notes
    -----
    Model: D_after = D0 * (1 - f)^(1 + cascade_exponent * f)

    This captures:
    - Linear loss for small f (ecosystem resilience)
    - Accelerating loss for large f (cascade failures)
    """
    f = loss_fraction
    exponent = 1 + cascade_exponent * f
    return D0 * (1 - f) ** exponent


def biodiversity_impact(
    D0: float = 0.5,
    loss_fractions: Optional[List[float]] = None,
    bandwidth: float = 1e12,
    cascade_exponent: float = 2.0,
) -> List[BiodiversityImpact]:
    """
    Simulate impact of biodiversity loss on intelligence ceiling.

    Reproduces Table IV from the paper.

    Parameters
    ----------
    D0 : float
        Initial biosphere data richness
    loss_fractions : list, optional
        Fractions of biodiversity lost to simulate
        Default: [0.1, 0.25, 0.5, 0.75, 0.9]
    bandwidth : float
        Observation bandwidth [bits/s]
    cascade_exponent : float
        Nonlinearity parameter for cascade model

    Returns
    -------
    list[BiodiversityImpact]
        Results for each loss fraction

    Examples
    --------
    >>> results = biodiversity_impact(D0=0.5)
    >>> for r in results:
    ...     print(f"{r.loss_fraction:.0%}: D = {r.D_after:.3f}, "
    ...           f"Reduction = {r.ceiling_reduction_pct:.1f}%")
    """
    if loss_fractions is None:
        loss_fractions = [0.10, 0.25, 0.50, 0.75, 0.90]

    ceiling_before = D0 * bandwidth
    results = []

    for f in loss_fractions:
        D_after = cascade_model(D0, f, cascade_exponent)
        ceiling_after = D_after * bandwidth
        reduction_pct = 100 * (1 - ceiling_after / ceiling_before)

        results.append(
            BiodiversityImpact(
                loss_fraction=f,
                D_before=D0,
                D_after=D_after,
                ceiling_before=ceiling_before,
                ceiling_after=ceiling_after,
                ceiling_reduction_pct=reduction_pct,
            )
        )

    return results


def exploit_trajectory(
    t: np.ndarray, D0: float, decay_rate: float = 0.02
) -> np.ndarray:
    """
    Exploitation strategy: D decays exponentially.

    D(t) = D0 * exp(-λt)

    Parameters
    ----------
    t : array
        Time points [years]
    D0 : float
        Initial data richness
    decay_rate : float
        Exponential decay rate λ [1/year]

    Returns
    -------
    array
        D values at each time point
    """
    return D0 * np.exp(-decay_rate * t)


def sustain_trajectory(t: np.ndarray, D0: float) -> np.ndarray:
    """
    Sustain strategy: D remains constant.

    D(t) = D0

    Parameters
    ----------
    t : array
        Time points [years]
    D0 : float
        Initial data richness

    Returns
    -------
    array
        D values at each time point (constant)
    """
    return np.full_like(t, D0, dtype=float)


def regenerate_trajectory(
    t: np.ndarray, D0: float, growth_rate: float = 0.005, D_max: float = 1.0
) -> np.ndarray:
    """
    Regeneration strategy: D grows linearly (capped at D_max).

    D(t) = min(D0 * (1 + rt), D_max)

    Parameters
    ----------
    t : array
        Time points [years]
    D0 : float
        Initial data richness
    growth_rate : float
        Linear growth rate r [1/year]
    D_max : float
        Maximum possible D

    Returns
    -------
    array
        D values at each time point
    """
    D = D0 * (1 + growth_rate * t)
    return np.minimum(D, D_max)


def strategy_comparison(
    D0: float = 0.5,
    time_horizon: float = 200,
    bandwidth: float = 1e12,
    decay_rate: float = 0.02,
    growth_rate: float = 0.005,
    n_points: int = 1000,
) -> Dict[str, StrategyResult]:
    """
    Compare exploitation, sustain, and regeneration strategies.

    Reproduces Table V from the paper.

    Parameters
    ----------
    D0 : float
        Initial biosphere data richness
    time_horizon : float
        Simulation duration [years]
    bandwidth : float
        Observation bandwidth [bits/s]
    decay_rate : float
        Exponential decay rate for exploitation [1/year]
    growth_rate : float
        Linear growth rate for regeneration [1/year]
    n_points : int
        Number of time points for integration

    Returns
    -------
    dict[str, StrategyResult]
        Results for each strategy

    Examples
    --------
    >>> results = strategy_comparison(D0=0.5, time_horizon=200)
    >>> for name, r in results.items():
    ...     print(f"{name}: Total I = {r.total_intelligence:.2e}, "
    ...           f"Final D = {r.final_D:.3f}")
    """
    t = np.linspace(0, time_horizon, n_points)

    # Seconds per year
    SECONDS_PER_YEAR = 365.25 * 24 * 3600

    strategies = {}

    # Exploit
    D_exploit = exploit_trajectory(t, D0, decay_rate)
    I_exploit = np.trapezoid(D_exploit * bandwidth, t) * SECONDS_PER_YEAR
    strategies["Exploit"] = StrategyResult(
        name="Exploit",
        D_trajectory=D_exploit,
        time_points=t,
        total_intelligence=I_exploit,
        final_D=D_exploit[-1],
        outcome="Collapse",
    )

    # Sustain
    D_sustain = sustain_trajectory(t, D0)
    I_sustain = np.trapezoid(D_sustain * bandwidth, t) * SECONDS_PER_YEAR
    strategies["Sustain"] = StrategyResult(
        name="Sustain",
        D_trajectory=D_sustain,
        time_points=t,
        total_intelligence=I_sustain,
        final_D=D_sustain[-1],
        outcome="Stable",
    )

    # Regenerate
    D_regen = regenerate_trajectory(t, D0, growth_rate)
    I_regen = np.trapezoid(D_regen * bandwidth, t) * SECONDS_PER_YEAR
    strategies["Regenerate"] = StrategyResult(
        name="Regenerate",
        D_trajectory=D_regen,
        time_points=t,
        total_intelligence=I_regen,
        final_D=D_regen[-1],
        outcome="Growth",
    )

    return strategies


def biosphere_information_potential(
    D: float, bandwidth: float, discount_rate: float = 0.01, time_horizon: float = 1000
) -> float:
    """
    Calculate biosphere information potential Φ_bio.

    Φ_bio = ∫₀^∞ D(τ) · B · e^(-δτ) dτ

    For constant D:
    Φ_bio = D · B / δ

    Parameters
    ----------
    D : float
        Current biosphere data richness
    bandwidth : float
        Observation bandwidth [bits/s]
    discount_rate : float
        Discount rate δ [1/year]
    time_horizon : float
        Integration horizon [years] (approximates ∞)

    Returns
    -------
    float
        Biosphere information potential [bits]

    Notes
    -----
    This is the discounted total extractable intelligence from
    the biosphere at current state.
    """
    SECONDS_PER_YEAR = 365.25 * 24 * 3600

    # For constant D, analytical solution
    if discount_rate > 0:
        Phi = D * bandwidth * SECONDS_PER_YEAR / discount_rate
        # Apply finite horizon correction
        Phi *= 1 - np.exp(-discount_rate * time_horizon)
    else:
        # No discounting
        Phi = D * bandwidth * SECONDS_PER_YEAR * time_horizon

    return Phi


def crossover_time(D0: float, decay_rate: float, bandwidth: float = 1e12) -> float:
    """
    Calculate time at which Exploit strategy total intelligence
    falls below Sustain strategy.

    After t*, cumulative intelligence from Sustain exceeds Exploit.

    Parameters
    ----------
    D0 : float
        Initial data richness
    decay_rate : float
        Exploitation decay rate λ [1/year]
    bandwidth : float
        Observation bandwidth [bits/s]

    Returns
    -------
    float
        Crossover time [years]

    Notes
    -----
    For Exploit: I_total(t) = D0·B·(1 - e^(-λt))/λ → D0·B/λ
    For Sustain: I_total(t) = D0·B·t

    Crossover when: t = (1 - e^(-λt))/λ
    Approximately: t* ≈ 1/λ
    """
    # Numerical solution
    from scipy.optimize import brentq

    def diff(t):
        I_exploit = D0 * bandwidth * (1 - np.exp(-decay_rate * t)) / decay_rate
        I_sustain = D0 * bandwidth * t
        return I_sustain - I_exploit

    # Find crossover
    try:
        t_star = brentq(diff, 0.01, 1000 / decay_rate)
    except ValueError:
        t_star = 1 / decay_rate  # Approximate

    return t_star


def generate_paper_tables() -> Dict[str, str]:
    """
    Generate all tables from the paper as formatted strings.

    Returns
    -------
    dict
        Dictionary with table names and formatted content
    """
    tables = {}

    # Table IV: Biodiversity Impact
    results = biodiversity_impact()
    lines = [
        "Table IV: Biodiversity Loss Impact on Intelligence Ceiling",
        "=" * 60,
        f"{'Loss %':>10} {'D_after':>10} {'Ceiling':>15} {'Reduction':>12}",
        "-" * 60,
    ]
    for r in results:
        lines.append(
            f"{r.loss_fraction:>10.0%} {r.D_after:>10.3f} "
            f"{r.ceiling_after:>15.2e} {r.ceiling_reduction_pct:>11.1f}%"
        )
    tables["biodiversity_impact"] = "\n".join(lines)

    # Table V: Strategy Comparison
    strategies = strategy_comparison()
    lines = [
        "\nTable V: Strategy Comparison over 200 Years",
        "=" * 60,
        f"{'Strategy':>12} {'Final D':>10} {'Total I (bits)':>18} {'Outcome':>12}",
        "-" * 60,
    ]
    for name, r in strategies.items():
        lines.append(
            f"{name:>12} {r.final_D:>10.3f} "
            f"{r.total_intelligence:>18.2e} {r.outcome:>12}"
        )
    tables["strategy_comparison"] = "\n".join(lines)

    return tables


def plot_strategy_comparison(
    strategies: Dict[str, StrategyResult], save_path: Optional[str] = None
):
    """
    Plot D trajectories and cumulative intelligence for all strategies.

    Parameters
    ----------
    strategies : dict
        Output from strategy_comparison()
    save_path : str, optional
        Path to save figure (if None, displays)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"Exploit": "red", "Sustain": "blue", "Regenerate": "green"}

    # D trajectories
    ax1 = axes[0]
    for name, result in strategies.items():
        ax1.plot(
            result.time_points,
            result.D_trajectory,
            color=colors[name],
            label=name,
            linewidth=2,
        )
    ax1.set_xlabel("Time [years]")
    ax1.set_ylabel("Data Richness D")
    ax1.set_title("Biosphere D Trajectories")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # Cumulative intelligence
    ax2 = axes[1]
    SECONDS_PER_YEAR = 365.25 * 24 * 3600
    bandwidth = 1e12  # Assumed

    for name, result in strategies.items():
        cumulative = np.zeros_like(result.time_points)
        for i in range(1, len(result.time_points)):
            dt = result.time_points[i] - result.time_points[i - 1]
            cumulative[i] = (
                cumulative[i - 1]
                + result.D_trajectory[i] * bandwidth * dt * SECONDS_PER_YEAR
            )
        ax2.plot(
            result.time_points, cumulative, color=colors[name], label=name, linewidth=2
        )

    ax2.set_xlabel("Time [years]")
    ax2.set_ylabel("Cumulative Intelligence [bits]")
    ax2.set_title("Total Intelligence Acquired")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_biodiversity_cascade(
    loss_fractions: Optional[List[float]] = None, save_path: Optional[str] = None
):
    """
    Plot the cascade model showing nonlinear D reduction.

    Parameters
    ----------
    loss_fractions : list, optional
        x-axis values (default: fine grid 0-1)
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    if loss_fractions is None:
        loss_fractions = np.linspace(0, 0.99, 100)

    D0 = 0.5

    # Different cascade strengths
    cascade_exponents = [0, 1, 2, 4]

    fig, ax = plt.subplots(figsize=(8, 6))

    for exp in cascade_exponents:
        D_values = [cascade_model(D0, f, exp) for f in loss_fractions]
        label = "Linear" if exp == 0 else f"Cascade (β={exp})"
        ax.plot(loss_fractions * 100, D_values, label=label, linewidth=2)

    ax.set_xlabel("Biodiversity Loss [%]")
    ax.set_ylabel("Data Richness D")
    ax.set_title("Nonlinear Biodiversity-D Coupling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, D0 * 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Generate and print tables
    tables = generate_paper_tables()
    for name, content in tables.items():
        print(content)
        print()

    # Calculate crossover time
    t_cross = crossover_time(D0=0.5, decay_rate=0.02)
    print(f"\nCrossover time (Sustain beats Exploit): {t_cross:.1f} years")

    # Calculate biosphere information potential
    Phi = biosphere_information_potential(D=0.5, bandwidth=1e12)
    print(f"Biosphere information potential: {Phi:.2e} bits")
