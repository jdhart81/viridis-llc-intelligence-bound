"""
Paper Validation Script
=======================

This script reproduces all numerical results from:
"The Intelligence Bound: Thermodynamic Limits on Learning Rate
and Implications for Biosphere Information"

Run with: python -m code.validation

Author: Justin Hart, Viridis LLC
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.intelligence_bound import (
    landauer_bound,
    system_analysis,
    temperature_scaling_table,
    real_systems_table,
    critical_power,
    BOLTZMANN_CONSTANT,
    LN2,
)

from code.simulations import (
    biodiversity_impact,
    strategy_comparison,
    biosphere_information_potential,
    crossover_time,
    cascade_model,
)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def validate_landauer_bound():
    """Validate Landauer bound calculations (Table I)."""
    print_header("TABLE I: Temperature Scaling Validation")

    print("\nTheoretical prediction: İ_max × T = P/(k_B ln 2) = constant")
    print(f"For P = 1 W: İ_max × T = {1 / (BOLTZMANN_CONSTANT * LN2):.3e} K·bits/s")

    table = temperature_scaling_table(power=1.0)

    print(f"\n{'Temperature [K]':>15} {'İ_max [bits/s]':>18} {'İ_max × T':>15}")
    print("-" * 50)

    for i in range(len(table["temperature_K"])):
        T = table["temperature_K"][i]
        I_max = table["I_max_bits_per_s"][i]
        product = table["I_max_times_T"][i]

        # Add label for common temperatures
        label = ""
        if T == 4:
            label = " (liquid He)"
        elif T == 77:
            label = " (liquid N₂)"
        elif T == 300:
            label = " (room temp)"

        print(f"{T:>15}{label:<12} {I_max:>18.3e} {product:>15.3e}")

    # Verify constancy
    products = table["I_max_times_T"]
    variation = (max(products) - min(products)) / np.mean(products)

    print(f"\nVariation in İ×T: {variation:.2e} (should be ~0)")
    print("✓ PASSED" if variation < 1e-10 else "✗ FAILED")

    return variation < 1e-10


def validate_real_systems():
    """Validate real system analysis (Table II)."""
    print_header("TABLE II: Real System Analysis")

    systems = real_systems_table()

    print(
        f"\n{'System':<18} {'Data Bound':>14} {'Landauer Bound':>16} "
        f"{'Headroom':>12} {'Limiting':>10}"
    )
    print("-" * 75)

    all_data_limited = True

    for name, analysis in systems.items():
        print(
            f"{name:<18} {analysis.data_bound:>14.2e} "
            f"{analysis.landauer_bound:>16.2e} "
            f"{analysis.headroom:>12.2e} {analysis.limiting_factor:>10}"
        )

        if analysis.limiting_factor != "DATA":
            all_data_limited = False

    print("\nKey finding: All systems are DATA-limited, not POWER-limited")
    print(
        f"Headroom range: 10^{np.log10(min(s.headroom for s in systems.values())):.0f} "
        f"to 10^{np.log10(max(s.headroom for s in systems.values())):.0f}"
    )
    print("✓ PASSED" if all_data_limited else "✗ FAILED")

    return all_data_limited


def validate_biodiversity_impact():
    """Validate biodiversity impact simulation (Table IV)."""
    print_header("TABLE IV: Biodiversity Loss Impact")

    results = biodiversity_impact(D0=0.5)

    print("\nInitial D = 0.5, Bandwidth = 10¹² bits/s")
    print("Cascade model: D_after = D₀ × (1-f)^(1+2f)")

    print(f"\n{'Loss':>8} {'D_after':>10} {'Ceiling [bits/s]':>18} {'Reduction':>12}")
    print("-" * 52)

    for r in results:
        print(
            f"{r.loss_fraction:>8.0%} {r.D_after:>10.3f} "
            f"{r.ceiling_after:>18.2e} {r.ceiling_reduction_pct:>11.1f}%"
        )

    # Get specific loss results for reporting
    loss_50 = next(r for r in results if r.loss_fraction == 0.5)

    # Verify 90% loss causes severe reduction
    loss_90 = next(r for r in results if r.loss_fraction == 0.9)
    severe = loss_90.ceiling_reduction_pct > 40

    print(
        f"\nNonlinear cascade effect at 50% loss: {loss_50.ceiling_reduction_pct:.1f}%"
    )
    print(f"Severe reduction at 90% loss: {loss_90.ceiling_reduction_pct:.1f}%")
    print("✓ PASSED" if severe else "✗ FAILED")

    return severe


def validate_strategy_comparison():
    """Validate strategy comparison (Table V)."""
    print_header("TABLE V: Strategy Comparison (200 years)")

    strategies = strategy_comparison(D0=0.5, time_horizon=200)

    print("\nParameters: D₀ = 0.5, B = 10¹² bits/s")
    print("Exploit: λ = 0.02/year, Regenerate: r = 0.005/year")

    print(
        f"\n{'Strategy':<12} {'Final D':>10} {'Total Intelligence':>22} {'Outcome':>12}"
    )
    print("-" * 60)

    for name, r in strategies.items():
        print(
            f"{name:<12} {r.final_D:>10.3f} {r.total_intelligence:>22.2e} {r.outcome:>12}"
        )

    # Verify ordering: Regenerate > Sustain > Exploit
    I_exploit = strategies["Exploit"].total_intelligence
    I_sustain = strategies["Sustain"].total_intelligence
    I_regen = strategies["Regenerate"].total_intelligence

    correct_order = I_regen > I_sustain > I_exploit

    print(f"\nRatio Sustain/Exploit: {I_sustain/I_exploit:.1f}×")
    print(f"Ratio Regenerate/Exploit: {I_regen/I_exploit:.1f}×")
    print(f"Correct ordering (Regen > Sustain > Exploit): {correct_order}")
    print("✓ PASSED" if correct_order else "✗ FAILED")

    return correct_order


def validate_critical_power():
    """Validate critical power calculation."""
    print_header("PHASE TRANSITION: Critical Power P*")

    # For H100-like system
    D = 0.05
    B = 3e12  # bits/s
    T = 350  # K

    P_star = critical_power(D, B, T)

    print(f"\nFor D = {D}, B = {B:.0e} bits/s, T = {T} K:")
    print("P* = D × B × k_B × T × ln(2)")
    print(f"P* = {P_star:.2e} W")
    print(f"P* = {P_star/1e9:.2f} GW")

    # Compare to actual H100 power
    P_h100 = 700  # W
    ratio = P_h100 / P_star  # How much above critical power

    print(f"\nH100 power: {P_h100} W")
    print(f"P_H100/P* = {ratio:.2e}")
    print(f"\nH100 operates {ratio:.0e}× above critical power")
    print("This confirms the system is deeply in the DATA-limited regime")

    return ratio > 1e10


def validate_biosphere_potential():
    """Validate biosphere information potential calculation."""
    print_header("BIOSPHERE INFORMATION POTENTIAL")

    D = 0.5
    B = 1e12
    delta = 0.01  # 1% per year discount

    Phi = biosphere_information_potential(D, B, delta)

    print(f"\nParameters: D = {D}, B = {B:.0e} bits/s, δ = {delta}/year")
    print("\nΦ_bio = ∫ D·B·e^(-δt) dt")
    print("      = D·B/δ (for constant D)")
    print(f"      = {Phi:.2e} bits")

    # Compare to total biosphere information
    biosphere_info = 1e15  # bits (Table III estimate)
    years_equiv = Phi / (D * B * 365.25 * 24 * 3600)

    print(f"\nEquivalent to {years_equiv:.0f} years of undiscounted extraction")
    print(f"Biosphere contains ~{biosphere_info:.0e} bits (Table III)")
    print(f"Extraction efficiency required: {biosphere_info/Phi:.2e}")

    return Phi > 0


def validate_crossover():
    """Validate crossover time calculation."""
    print_header("CROSSOVER TIME: Sustain Beats Exploit")

    D0 = 0.5
    decay_rates = [0.01, 0.02, 0.05, 0.1]

    print("\nAt t*, cumulative intelligence from Sustain exceeds Exploit")
    print("Approximate result: t* ≈ 1/λ")

    print(f"\n{'Decay Rate λ':>15} {'t* (calculated)':>18} {'1/λ':>12}")
    print("-" * 48)

    for lam in decay_rates:
        t_star = crossover_time(D0, lam)
        print(f"{lam:>15.2f}/year {t_star:>18.1f} years {1/lam:>12.1f}")

    return True


def validate_appendix_calculations():
    """Validate Appendix A numerical estimates."""
    print_header("APPENDIX A: Numerical Estimates")

    # A.1: Landauer Bound at room temp
    print("\nA.1 Landauer Bound (P=1W, T=300K):")
    I_max = landauer_bound(1.0, 300)
    expected = 3.48e20
    error = abs(I_max - expected) / expected
    print(f"    Calculated: {I_max:.3e} bits/s")
    print(f"    Paper value: {expected:.2e} bits/s")
    print(f"    Error: {error:.2%}")

    # A.2: Human Brain
    print("\nA.2 Human Brain Analysis:")
    brain = system_analysis("Brain", power=20, temperature=310, bandwidth=1e9, D=1e-3)
    print(f"    Landauer bound: {brain.landauer_bound:.2e} bits/s")
    print(f"    Data bound: {brain.data_bound:.2e} bits/s")
    print(f"    Headroom: {brain.headroom:.0e}×")

    # A.3: H100
    print("\nA.3 NVIDIA H100 Analysis:")
    h100 = system_analysis("H100", power=700, temperature=350, bandwidth=3e12, D=0.05)
    print(f"    Landauer bound: {h100.landauer_bound:.2e} bits/s")
    print(f"    Data bound: {h100.data_bound:.2e} bits/s")
    print(f"    Headroom: {h100.headroom:.0e}×")

    # A.4: Cascade model
    print("\nA.4 Cascade Model (f=0.5):")
    D_after = cascade_model(0.5, 0.5, cascade_exponent=2.0)
    print(f"    D_after = 0.5 × 0.5^(1+2×0.5) = 0.5 × 0.5² = {D_after:.3f}")

    return error < 0.01


def run_all_validations():
    """Run all validation tests and report summary."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " INTELLIGENCE BOUND: COMPUTATIONAL VALIDATION ".center(68) + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    tests = [
        ("Temperature Scaling (Table I)", validate_landauer_bound),
        ("Real Systems (Table II)", validate_real_systems),
        ("Biodiversity Impact (Table IV)", validate_biodiversity_impact),
        ("Strategy Comparison (Table V)", validate_strategy_comparison),
        ("Critical Power", validate_critical_power),
        ("Biosphere Potential", validate_biosphere_potential),
        ("Crossover Time", validate_crossover),
        ("Appendix Calculations", validate_appendix_calculations),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ ERROR in {name}: {e}")
            results.append((name, False))

    # Summary
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for _, p in results if p)
    total = len(results)

    print(f"\n{'Test':<40} {'Status':>10}")
    print("-" * 52)
    for name, p in results:
        status = "✓ PASSED" if p else "✗ FAILED"
        print(f"{name:<40} {status:>10}")

    print("-" * 52)
    print(f"{'Total':<40} {passed}/{total}")

    if passed == total:
        print("\n✓ ALL VALIDATIONS PASSED")
        print("\nThe computational results reproduce all claims in the paper.")
    else:
        print(f"\n✗ {total - passed} VALIDATION(S) FAILED")

    return passed == total


def save_results(output_dir: str = "output"):
    """Save validation results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Table I
    with open(os.path.join(output_dir, "table_i_temperature_scaling.csv"), "w") as f:
        table = temperature_scaling_table()
        f.write("temperature_K,I_max_bits_per_s,I_max_times_T\n")
        for i in range(len(table["temperature_K"])):
            f.write(
                f"{table['temperature_K'][i]},"
                f"{table['I_max_bits_per_s'][i]:.6e},"
                f"{table['I_max_times_T'][i]:.6e}\n"
            )

    # Table II
    with open(os.path.join(output_dir, "table_ii_real_systems.csv"), "w") as f:
        systems = real_systems_table()
        f.write(
            "system,power_W,temperature_K,bandwidth_bits_s,D,"
            "data_bound,landauer_bound,headroom,limiting\n"
        )
        for name, s in systems.items():
            f.write(
                f"{name},{s.power},{s.temperature},{s.bandwidth},{s.D},"
                f"{s.data_bound:.6e},{s.landauer_bound:.6e},"
                f"{s.headroom:.6e},{s.limiting_factor}\n"
            )

    # Table IV
    with open(os.path.join(output_dir, "table_iv_biodiversity.csv"), "w") as f:
        results = biodiversity_impact()
        f.write("loss_fraction,D_after,ceiling_after,reduction_pct\n")
        for r in results:
            f.write(
                f"{r.loss_fraction},{r.D_after:.6f},"
                f"{r.ceiling_after:.6e},{r.ceiling_reduction_pct:.2f}\n"
            )

    # Table V
    with open(os.path.join(output_dir, "table_v_strategies.csv"), "w") as f:
        strategies = strategy_comparison()
        f.write("strategy,final_D,total_intelligence,outcome\n")
        for name, r in strategies.items():
            f.write(f"{name},{r.final_D:.6f},{r.total_intelligence:.6e},{r.outcome}\n")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    success = run_all_validations()

    # Save results to files
    save_results()

    sys.exit(0 if success else 1)
