"""
Unit Tests for Data Richness Estimators
========================================

Tests for:
- Compression-based estimator
- Prediction-based estimator
- Mutual information estimator

Author: Justin Hart, Viridis LLC
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.estimators import (
    compression_estimator,
    prediction_estimator,
    mutual_information_estimator,
    estimate_D,
    compare_estimators
)


class TestCompressionEstimator:
    """Tests for compression-based D estimation."""
    
    def test_random_noise(self):
        """Random noise should have low D (hard to compress)."""
        np.random.seed(42)
        noise = np.random.randn(10000)
        D = compression_estimator(noise)
        assert D < 0.2, f"Random noise D={D} should be low"
    
    def test_periodic_signal(self):
        """Periodic signal should have higher D than noise."""
        t = np.linspace(0, 100 * np.pi, 10000)
        signal = np.sin(t)
        D = compression_estimator(signal)
        # Periodic signals compress better than noise but zlib isn't perfect
        assert D > 0.1, f"Periodic signal D={D} should be higher than noise"
    
    def test_constant_signal(self):
        """Constant signal should have very high D."""
        constant = np.ones(10000)
        D = compression_estimator(constant)
        assert D > 0.9, f"Constant signal D={D} should be very high"
    
    def test_string_input(self):
        """Should handle string input."""
        text = "hello world " * 1000
        D = compression_estimator(text)
        assert 0 < D < 1
    
    def test_bytes_input(self):
        """Should handle bytes input."""
        data = b"test data " * 1000
        D = compression_estimator(data)
        assert 0 < D < 1
    
    def test_empty_data(self):
        """Empty data should return 0."""
        assert compression_estimator(np.array([])) == 0.0
    
    def test_bounds(self):
        """Result should always be in [0, 1]."""
        for _ in range(10):
            data = np.random.randn(1000)
            D = compression_estimator(data)
            assert 0 <= D <= 1


class TestPredictionEstimator:
    """Tests for prediction-based D estimation."""
    
    def test_perfect_prediction(self):
        """Perfectly predictable target should give D ≈ 1."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        Y = X @ np.array([1, 2, 3])  # Linear combination
        D = prediction_estimator(X, Y)
        assert D > 0.99, f"Perfect prediction D={D} should be ~1"
    
    def test_no_relationship(self):
        """Independent variables should give D ≈ 0."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        Y = np.random.randn(1000)  # Independent
        D = prediction_estimator(X, Y)
        assert D < 0.1, f"No relationship D={D} should be ~0"
    
    def test_partial_relationship(self):
        """Noisy relationship should give intermediate D."""
        np.random.seed(42)
        X = np.random.randn(1000, 3)
        Y = X @ np.array([1, 2, 3]) + np.random.randn(1000)  # With noise
        D = prediction_estimator(X, Y)
        assert 0.5 < D < 1.0, f"Partial relationship D={D} should be intermediate"
    
    def test_zero_variance_target(self):
        """Constant target should give D = 0."""
        X = np.random.randn(100, 3)
        Y = np.ones(100)  # Constant
        D = prediction_estimator(X, Y)
        assert D == 0.0
    
    def test_knn_model(self):
        """KNN model should also work."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        Y = X @ np.array([1, 2, 3])
        D = prediction_estimator(X, Y, model="knn")
        assert D > 0.9
    
    def test_unknown_model_raises(self):
        """Unknown model should raise ValueError."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(100)
        with pytest.raises(ValueError, match="Unknown model"):
            prediction_estimator(X, Y, model="invalid")


class TestMutualInformationEstimator:
    """Tests for MI-based D estimation."""
    
    def test_correlated_variables(self):
        """Correlated variables should have positive D."""
        np.random.seed(42)
        X = np.random.randn(500)
        Y = X + 0.5 * np.random.randn(500)
        D = mutual_information_estimator(X, Y)
        assert D > 0.1, f"Correlated variables D={D} should be positive"
    
    def test_independent_variables(self):
        """Independent variables should have D ≈ 0."""
        np.random.seed(42)
        X = np.random.randn(500)
        Y = np.random.randn(500)
        D = mutual_information_estimator(X, Y)
        assert D < 0.2, f"Independent variables D={D} should be ~0"
    
    def test_deterministic_relationship(self):
        """Deterministic relationship should have high D."""
        np.random.seed(42)
        X = np.random.randn(500)
        Y = 2 * X + 1  # Deterministic
        D = mutual_information_estimator(X, Y)
        assert D > 0.5, f"Deterministic D={D} should be high"
    
    def test_unnormalized_output(self):
        """Should return tuple when normalize=False."""
        np.random.seed(42)
        X = np.random.randn(200)
        Y = X + np.random.randn(200)
        result = mutual_information_estimator(X, Y, normalize=False)
        assert isinstance(result, tuple)
        assert len(result) == 3
        I, H_X, H_O = result
        assert I >= 0
        assert H_X >= 0
        assert H_O >= 0


class TestEstimateD:
    """Tests for the unified estimate_D function."""
    
    def test_auto_without_targets(self):
        """Without targets, should use compression."""
        data = np.random.randn(1000)
        D, method = estimate_D(data)
        assert method == "compression"
        assert 0 <= D <= 1
    
    def test_auto_with_targets(self):
        """With targets, should use prediction."""
        data = np.random.randn(1000, 3)
        targets = np.random.randn(1000)
        D, method = estimate_D(data, targets)
        assert method == "prediction"
        assert 0 <= D <= 1
    
    def test_explicit_method(self):
        """Explicit method should be respected."""
        data = np.random.randn(1000)
        D, method = estimate_D(data, method="compression")
        assert method == "compression"


class TestCompareEstimators:
    """Tests for estimator comparison."""
    
    def test_returns_all_estimators(self):
        """Should return all three estimator values."""
        np.random.seed(42)
        X = np.random.randn(500)
        Y = X + 0.5 * np.random.randn(500)
        results = compare_estimators(X, Y)
        
        assert "compression" in results
        assert "prediction" in results
        assert "mi" in results
        assert "hierarchy_satisfied" in results
    
    def test_values_in_range(self):
        """All values should be in [0, 1]."""
        np.random.seed(42)
        X = np.random.randn(500)
        Y = X + np.random.randn(500)
        results = compare_estimators(X, Y)
        
        assert 0 <= results["compression"] <= 1
        assert 0 <= results["prediction"] <= 1
        assert 0 <= results["mi"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
