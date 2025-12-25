"""
Unit Tests for Biodiversity Simulations
========================================

Tests for:
- Cascade model
- Biodiversity impact
- Strategy comparison
- Biosphere information potential

Author: Justin Hart, Viridis LLC
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.simulations import (
    cascade_model,
    biodiversity_impact,
    strategy_comparison,
    biosphere_information_potential,
    crossover_time,
    exploit_trajectory,
    sustain_trajectory,
    regenerate_trajectory
)


class TestCascadeModel:
    """Tests for the biodiversity-D cascade model."""
    
    def test_no_loss(self):
        """Zero loss should preserve D."""
        D0 = 0.5
        D_after = cascade_model(D0, 0.0)
        assert D_after == D0
    
    def test_total_loss(self):
        """100% loss should give D = 0."""
        D0 = 0.5
        D_after = cascade_model(D0, 1.0)
        assert D_after == 0.0
    
    def test_partial_loss(self):
        """Partial loss should reduce D."""
        D0 = 0.5
        for f in [0.1, 0.25, 0.5, 0.75]:
            D_after = cascade_model(D0, f)
            assert 0 < D_after < D0
    
    def test_nonlinearity(self):
        """Higher cascade exponent should cause more severe loss."""
        D0 = 0.5
        f = 0.5
        
        D_linear = cascade_model(D0, f, cascade_exponent=0)
        D_cascade = cascade_model(D0, f, cascade_exponent=2)
        
        assert D_cascade < D_linear
    
    def test_monotonic_decrease(self):
        """D should decrease monotonically with loss."""
        D0 = 0.5
        losses = np.linspace(0, 0.99, 100)
        D_values = [cascade_model(D0, f) for f in losses]
        
        for i in range(1, len(D_values)):
            assert D_values[i] <= D_values[i-1]


class TestBiodiversityImpact:
    """Tests for biodiversity impact simulation."""
    
    def test_default_loss_fractions(self):
        """Should simulate 5 loss levels by default."""
        results = biodiversity_impact()
        assert len(results) == 5
    
    def test_custom_loss_fractions(self):
        """Should accept custom loss fractions."""
        losses = [0.2, 0.4, 0.6]
        results = biodiversity_impact(loss_fractions=losses)
        assert len(results) == 3
        assert results[0].loss_fraction == 0.2
    
    def test_ceiling_reduction(self):
        """Higher loss should cause greater ceiling reduction."""
        results = biodiversity_impact()
        reductions = [r.ceiling_reduction_pct for r in results]
        
        for i in range(1, len(reductions)):
            assert reductions[i] >= reductions[i-1]
    
    def test_severe_loss_at_90_percent(self):
        """90% biodiversity loss should cause >40% ceiling reduction."""
        results = biodiversity_impact()
        loss_90 = next(r for r in results if r.loss_fraction == 0.9)
        assert loss_90.ceiling_reduction_pct > 40


class TestTrajectories:
    """Tests for D trajectory functions."""
    
    def test_exploit_decay(self):
        """Exploit trajectory should decay exponentially."""
        t = np.array([0, 50, 100])
        D0 = 0.5
        decay = 0.02
        
        D = exploit_trajectory(t, D0, decay)
        
        assert D[0] == D0
        assert D[1] < D0
        assert D[2] < D[1]
        assert np.isclose(D[2], D0 * np.exp(-decay * 100))
    
    def test_sustain_constant(self):
        """Sustain trajectory should be constant."""
        t = np.array([0, 50, 100, 200])
        D0 = 0.5
        
        D = sustain_trajectory(t, D0)
        
        assert all(d == D0 for d in D)
    
    def test_regenerate_growth(self):
        """Regenerate trajectory should grow linearly."""
        t = np.array([0, 50, 100])
        D0 = 0.5
        rate = 0.005
        
        D = regenerate_trajectory(t, D0, rate)
        
        assert D[0] == D0
        assert D[1] > D0
        assert D[2] > D[1]
    
    def test_regenerate_cap(self):
        """Regenerate should cap at D_max."""
        t = np.linspace(0, 1000, 100)
        D0 = 0.5
        D_max = 0.9
        
        D = regenerate_trajectory(t, D0, growth_rate=0.01, D_max=D_max)
        
        assert max(D) <= D_max


class TestStrategyComparison:
    """Tests for strategy comparison simulation."""
    
    def test_three_strategies(self):
        """Should return results for all three strategies."""
        results = strategy_comparison()
        assert len(results) == 3
        assert "Exploit" in results
        assert "Sustain" in results
        assert "Regenerate" in results
    
    def test_outcome_labels(self):
        """Strategies should have correct outcome labels."""
        results = strategy_comparison()
        assert results["Exploit"].outcome == "Collapse"
        assert results["Sustain"].outcome == "Stable"
        assert results["Regenerate"].outcome == "Growth"
    
    def test_intelligence_ordering(self):
        """Regenerate > Sustain > Exploit for total intelligence."""
        results = strategy_comparison()
        
        I_exploit = results["Exploit"].total_intelligence
        I_sustain = results["Sustain"].total_intelligence
        I_regen = results["Regenerate"].total_intelligence
        
        assert I_regen > I_sustain
        assert I_sustain > I_exploit
    
    def test_final_D_ordering(self):
        """Final D should reflect strategy type."""
        results = strategy_comparison()
        
        assert results["Exploit"].final_D < results["Sustain"].final_D
        assert results["Regenerate"].final_D > results["Sustain"].final_D
    
    def test_trajectories_have_correct_length(self):
        """Trajectories should have n_points elements."""
        n = 500
        results = strategy_comparison(n_points=n)
        
        for name, r in results.items():
            assert len(r.D_trajectory) == n
            assert len(r.time_points) == n


class TestBiosphereInformationPotential:
    """Tests for biosphere information potential calculation."""
    
    def test_positive_result(self):
        """Potential should be positive."""
        Phi = biosphere_information_potential(D=0.5, bandwidth=1e12)
        assert Phi > 0
    
    def test_scales_with_D(self):
        """Potential should scale linearly with D."""
        Phi_1 = biosphere_information_potential(D=0.5, bandwidth=1e12)
        Phi_2 = biosphere_information_potential(D=1.0, bandwidth=1e12)
        
        assert np.isclose(Phi_2 / Phi_1, 2.0, rtol=0.01)
    
    def test_scales_with_bandwidth(self):
        """Potential should scale linearly with bandwidth."""
        Phi_1 = biosphere_information_potential(D=0.5, bandwidth=1e12)
        Phi_2 = biosphere_information_potential(D=0.5, bandwidth=2e12)
        
        assert np.isclose(Phi_2 / Phi_1, 2.0, rtol=0.01)
    
    def test_higher_discount_reduces_potential(self):
        """Higher discount rate should reduce potential."""
        Phi_low = biosphere_information_potential(D=0.5, bandwidth=1e12, discount_rate=0.01)
        Phi_high = biosphere_information_potential(D=0.5, bandwidth=1e12, discount_rate=0.1)
        
        assert Phi_low > Phi_high


class TestCrossoverTime:
    """Tests for crossover time calculation."""
    
    def test_positive_result(self):
        """Crossover time should be positive."""
        t = crossover_time(D0=0.5, decay_rate=0.02)
        assert t > 0
    
    def test_scales_with_decay_rate(self):
        """Higher decay rate should reduce crossover time."""
        t_slow = crossover_time(D0=0.5, decay_rate=0.01)
        t_fast = crossover_time(D0=0.5, decay_rate=0.1)
        
        assert t_fast < t_slow
    
    def test_approximately_inverse(self):
        """Crossover time should be approximately 1/λ."""
        decay = 0.05
        t = crossover_time(D0=0.5, decay_rate=decay)
        
        # Should be close to 1/λ
        assert 0.5 / decay < t < 2 / decay


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
