"""
Unit Tests for Intelligence Bound Core Calculations
====================================================

Tests for:
- Landauer bound
- Data bound
- Combined intelligence bound
- System analysis

Author: Justin Hart, Viridis LLC
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.intelligence_bound import (
    landauer_bound,
    data_bound,
    intelligence_bound,
    system_analysis,
    temperature_scaling_table,
    real_systems_table,
    critical_power,
    BOLTZMANN_CONSTANT,
    LN2,
)


class TestLandauerBound:
    """Tests for Landauer bound calculation."""

    def test_room_temperature(self):
        """Verify Landauer bound at room temperature (300K)."""
        I_max = landauer_bound(power=1.0, temperature=300)
        expected = 1.0 / (BOLTZMANN_CONSTANT * 300 * LN2)
        assert np.isclose(I_max, expected, rtol=1e-10)

    def test_temperature_scaling(self):
        """Verify İ × T = constant."""
        P = 1.0
        temperatures = [4, 77, 300, 1000]
        products = [landauer_bound(P, T) * T for T in temperatures]

        # All products should be equal
        assert all(np.isclose(p, products[0], rtol=1e-10) for p in products)

    def test_power_scaling(self):
        """Verify İ ∝ P."""
        T = 300
        powers = [1, 10, 100, 1000]
        bounds = [landauer_bound(P, T) for P in powers]

        # Ratios should match power ratios
        for i in range(1, len(powers)):
            assert np.isclose(bounds[i] / bounds[0], powers[i] / powers[0], rtol=1e-10)

    def test_negative_power_raises(self):
        """Negative power should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            landauer_bound(power=-1.0, temperature=300)

    def test_zero_temperature_raises(self):
        """Zero temperature should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            landauer_bound(power=1.0, temperature=0)

    def test_negative_temperature_raises(self):
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            landauer_bound(power=1.0, temperature=-100)


class TestDataBound:
    """Tests for data bound calculation."""

    def test_basic_calculation(self):
        """Verify D × B calculation."""
        D = 0.1
        B = 1e12
        assert data_bound(D, B) == D * B

    def test_zero_D(self):
        """D = 0 should give zero bound."""
        assert data_bound(0, 1e12) == 0

    def test_full_D(self):
        """D = 1 should give full bandwidth."""
        B = 1e12
        assert data_bound(1.0, B) == B

    def test_D_out_of_range_raises(self):
        """D outside [0,1] should raise ValueError."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            data_bound(1.5, 1e12)
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            data_bound(-0.1, 1e12)

    def test_negative_bandwidth_raises(self):
        """Negative bandwidth should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            data_bound(0.5, -1e12)


class TestIntelligenceBound:
    """Tests for combined intelligence bound."""

    def test_data_limited_regime(self):
        """Most systems should be data-limited."""
        bound, limiting = intelligence_bound(
            D=0.05, bandwidth=1e12, power=700, temperature=350
        )
        assert limiting == "DATA"
        assert bound == 0.05 * 1e12

    def test_power_limited_regime(self):
        """Very high D and B should be power-limited."""
        bound, limiting = intelligence_bound(
            D=1.0, bandwidth=1e30, power=1.0, temperature=300
        )
        assert limiting == "POWER"
        assert np.isclose(bound, landauer_bound(1.0, 300))

    def test_minimum_of_bounds(self):
        """Result should be minimum of two bounds."""
        D, B, P, T = 0.1, 1e12, 100, 300

        I_data = data_bound(D, B)
        I_landauer = landauer_bound(P, T)

        bound, _ = intelligence_bound(D, B, P, T)
        assert bound == min(I_data, I_landauer)


class TestSystemAnalysis:
    """Tests for system analysis."""

    def test_human_brain(self):
        """Analyze human brain."""
        analysis = system_analysis(
            name="Human Brain", power=20, temperature=310, bandwidth=1e9, D=1e-3
        )

        assert analysis.name == "Human Brain"
        assert analysis.limiting_factor == "DATA"
        assert analysis.headroom > 1e10  # Many orders of magnitude

    def test_nvidia_h100(self):
        """Analyze NVIDIA H100."""
        analysis = system_analysis(
            name="NVIDIA H100",
            power=700,
            temperature=350,
            bandwidth=2.4e13,  # 3 TB/s = 2.4e13 bits/s
            D=0.03,  # D_text→world
        )

        assert analysis.limiting_factor == "DATA"
        assert analysis.headroom > 1e10

    def test_headroom_calculation(self):
        """Verify headroom = Landauer / Data."""
        analysis = system_analysis(
            name="Test", power=100, temperature=300, bandwidth=1e10, D=0.1
        )

        expected_headroom = analysis.landauer_bound / analysis.data_bound
        assert np.isclose(analysis.headroom, expected_headroom)


class TestTemperatureScalingTable:
    """Tests for Table I generation."""

    def test_default_temperatures(self):
        """Default temperatures should be [4, 77, 300, 1000]."""
        table = temperature_scaling_table()
        assert table["temperature_K"] == [4, 77, 300, 1000]

    def test_custom_temperatures(self):
        """Custom temperatures should work."""
        temps = [100, 200, 400]
        table = temperature_scaling_table(temperatures=temps)
        assert table["temperature_K"] == temps

    def test_constant_product(self):
        """İ × T should be constant."""
        table = temperature_scaling_table()
        products = table["I_max_times_T"]

        assert all(np.isclose(p, products[0], rtol=1e-10) for p in products)


class TestRealSystemsTable:
    """Tests for Table II generation."""

    def test_three_systems(self):
        """Should have Human Brain, H100, and Theoretical Max."""
        systems = real_systems_table()
        assert len(systems) == 3
        assert "Human Brain" in systems
        assert "NVIDIA H100" in systems
        assert "Theoretical Max" in systems

    def test_all_data_limited(self):
        """All systems should be data-limited."""
        systems = real_systems_table()
        for name, analysis in systems.items():
            assert analysis.limiting_factor == "DATA", f"{name} should be DATA-limited"


class TestCriticalPower:
    """Tests for critical power calculation."""

    def test_calculation(self):
        """Verify P* = D × B × k_B × T × ln(2)."""
        D, B, T = 0.1, 1e12, 300
        P_star = critical_power(D, B, T)
        expected = D * B * BOLTZMANN_CONSTANT * T * LN2
        assert np.isclose(P_star, expected)

    def test_phase_transition(self):
        """At P*, data bound should equal Landauer bound."""
        D, B, T = 0.1, 1e12, 300
        P_star = critical_power(D, B, T)

        I_data = data_bound(D, B)
        I_landauer = landauer_bound(P_star, T)

        assert np.isclose(I_data, I_landauer, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
