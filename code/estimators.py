"""
Data Richness Estimators
========================

This module implements operational estimators for the predictive fraction D,
which measures what fraction of observation information contains learnable
structure relevant to predicting the environment.

Three estimators are provided:

1. Compression-based: D̂_compress = 1 - H_compressed/H_raw
   - Measures structured fraction via compressibility
   - Provides upper bound (some structure may be non-predictive)
   - Easiest to compute

2. Prediction-based: D̂_predict = (Var(X) - Var(X|O)) / Var(X)
   - Measures variance reduction from prediction
   - Tighter bound than compression
   - Requires knowing target variable

3. Mutual information: D̂_MI = I(X_{t+τ}; O_t) / H(O_t)
   - Directly estimates predictive fraction
   - Most accurate but hardest to compute

Estimator hierarchy: D̂_compress ≥ D̂_MI ≥ D̂_predict

Author: Justin Hart, Viridis LLC
"""

import zlib
from typing import Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def compression_estimator(
    data: Union[np.ndarray, bytes, str],
    level: int = 9
) -> float:
    """
    Compression-based estimator for data richness D.
    
    Measures the structured fraction of observations via compressibility.
    Compressibility implies pattern; random noise is incompressible.
    
    D̂_compress = 1 - len(compressed) / len(raw)
    
    Parameters
    ----------
    data : array-like, bytes, or str
        Observation data to analyze
    level : int
        Compression level (1-9, higher = more compression)
    
    Returns
    -------
    float
        Estimated D ∈ [0, 1]
    
    Notes
    -----
    This provides an UPPER BOUND on true predictive richness.
    Some structure may be non-predictive (e.g., repetitive noise
    that doesn't predict environmental states).
    
    Examples
    --------
    >>> import numpy as np
    >>> # Highly structured (periodic)
    >>> structured = np.sin(np.linspace(0, 100*np.pi, 10000))
    >>> compression_estimator(structured)
    0.85  # High D
    
    >>> # Random noise
    >>> noise = np.random.randn(10000)
    >>> compression_estimator(noise)
    0.02  # Low D
    """
    # Convert to bytes if needed
    if isinstance(data, str):
        raw_bytes = data.encode('utf-8')
    elif isinstance(data, np.ndarray):
        raw_bytes = data.tobytes()
    elif isinstance(data, bytes):
        raw_bytes = data
    else:
        raw_bytes = np.array(data).tobytes()
    
    if len(raw_bytes) == 0:
        return 0.0
    
    compressed = zlib.compress(raw_bytes, level=level)
    
    # Compression ratio
    ratio = len(compressed) / len(raw_bytes)
    
    # D = 1 - ratio (fully random would have ratio ≈ 1)
    # Clamp to [0, 1] (compression can slightly exceed original for small data)
    D = max(0.0, min(1.0, 1.0 - ratio))
    
    return D


def prediction_estimator(
    observations: np.ndarray,
    targets: np.ndarray,
    model: str = "linear"
) -> float:
    """
    Prediction-based estimator for data richness D.
    
    Measures what fraction of target variability can be predicted
    from observations.
    
    D̂_predict = (Var(X) - Var(X|O)) / Var(X) = R²
    
    Parameters
    ----------
    observations : array-like
        Observation data O (features)
    targets : array-like
        Target environmental states X to predict
    model : str
        Prediction model: "linear" (default), "knn"
    
    Returns
    -------
    float
        Estimated D ∈ [0, 1]
    
    Notes
    -----
    For Gaussian variables, this equals the squared correlation ρ²(X,O).
    This is typically a LOWER BOUND on D_MI since it only captures
    linear predictability.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Perfect prediction
    >>> O = np.random.randn(1000, 5)
    >>> X = O @ np.array([1, 2, 3, 4, 5])
    >>> prediction_estimator(O, X)
    1.0
    
    >>> # No relationship
    >>> X_noise = np.random.randn(1000)
    >>> prediction_estimator(O, X_noise)
    0.01  # Near zero
    """
    observations = np.atleast_2d(observations)
    targets = np.atleast_1d(targets).ravel()
    
    if observations.shape[0] != len(targets):
        # Try transpose
        if observations.shape[1] == len(targets):
            observations = observations.T
        else:
            raise ValueError(
                f"Shape mismatch: observations {observations.shape}, "
                f"targets {len(targets)}"
            )
    
    if len(targets) < 10:
        raise ValueError("Need at least 10 samples for prediction estimation")
    
    var_X = np.var(targets)
    if var_X == 0:
        return 0.0  # No variance to predict
    
    if model == "linear":
        reg = LinearRegression()
        reg.fit(observations, targets)
        predictions = reg.predict(observations)
        
    elif model == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        k = min(5, len(targets) // 10)
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(observations, targets)
        predictions = knn.predict(observations)
        
    else:
        raise ValueError(f"Unknown model: {model}")
    
    residual_var = np.var(targets - predictions)
    D = max(0.0, min(1.0, 1.0 - residual_var / var_X))
    
    return D


def mutual_information_estimator(
    observations: np.ndarray,
    targets: np.ndarray,
    k: int = 3,
    normalize: bool = True
) -> Union[float, Tuple[float, float, float]]:
    """
    Mutual information estimator for data richness D.
    
    Uses k-nearest neighbor estimation of mutual information.
    
    D̂_MI = I(X; O) / H(O)
    
    Parameters
    ----------
    observations : array-like
        Observation data O
    targets : array-like
        Target environmental states X
    k : int
        Number of neighbors for KNN estimation
    normalize : bool
        If True, return D = I(X;O)/H(O)
        If False, return (I, H_X, H_O) tuple
    
    Returns
    -------
    float or tuple
        If normalize=True: D ∈ [0, 1]
        If normalize=False: (I(X;O), H(X), H(O))
    
    Notes
    -----
    Uses the Kraskov-Stögbauer-Grassberger estimator [1].
    
    [1] Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
        Estimating mutual information. Physical Review E, 69(6).
    
    Examples
    --------
    >>> import numpy as np
    >>> # Correlated variables
    >>> O = np.random.randn(1000)
    >>> X = O + 0.5 * np.random.randn(1000)
    >>> mutual_information_estimator(O, X)
    0.62
    """
    observations = np.atleast_2d(observations)
    targets = np.atleast_2d(targets)
    
    if observations.shape[0] == 1:
        observations = observations.T
    if targets.shape[0] == 1:
        targets = targets.T
    
    n = observations.shape[0]
    if n != targets.shape[0]:
        raise ValueError("observations and targets must have same length")
    
    if n < k + 1:
        raise ValueError(f"Need at least {k+1} samples, got {n}")
    
    # Combine for joint entropy estimation
    joint = np.hstack([observations, targets])
    
    # KNN distances
    def knn_distance(X, k):
        nn = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        return distances[:, -1]  # k-th neighbor distance
    
    # Count points within distance
    def count_within(X, epsilon):
        nn = NearestNeighbors(metric='chebyshev')
        nn.fit(X)
        counts = np.array([
            len(nn.radius_neighbors([x], epsilon[i], return_distance=False)[0]) - 1
            for i, x in enumerate(X)
        ])
        return counts
    
    # KSG estimator
    from scipy.special import digamma
    
    epsilon = knn_distance(joint, k)
    
    # Count neighbors in marginals
    n_x = count_within(targets, epsilon) + 1
    n_o = count_within(observations, epsilon) + 1
    
    # MI estimate
    I_est = digamma(k) - np.mean(digamma(n_x) + digamma(n_o)) + digamma(n)
    I_est = max(0, I_est)  # MI is non-negative
    
    if not normalize:
        # Estimate marginal entropies
        H_X = digamma(n) - np.mean(digamma(n_x)) + np.log(np.prod(np.ptp(targets, axis=0) + 1e-10))
        H_O = digamma(n) - np.mean(digamma(n_o)) + np.log(np.prod(np.ptp(observations, axis=0) + 1e-10))
        return I_est, max(0, H_X), max(0, H_O)
    
    # Estimate H(O) for normalization
    n_o_for_H = count_within(observations, knn_distance(observations, k)) + 1
    H_O = digamma(n) - np.mean(digamma(n_o_for_H))
    H_O += observations.shape[1] * np.log(2)  # Rough correction
    
    if H_O <= 0:
        return 0.0
    
    D = min(1.0, I_est / H_O)
    return max(0.0, D)


def estimate_D(
    data: np.ndarray,
    targets: Optional[np.ndarray] = None,
    method: str = "auto"
) -> Tuple[float, str]:
    """
    Estimate data richness D using the most appropriate method.
    
    Parameters
    ----------
    data : array-like
        Observation data
    targets : array-like, optional
        Target environmental states (if available)
    method : str
        "compression", "prediction", "mi", or "auto"
    
    Returns
    -------
    D : float
        Estimated data richness ∈ [0, 1]
    method_used : str
        Name of method actually used
    
    Examples
    --------
    >>> # Without targets, uses compression
    >>> D, method = estimate_D(data)
    
    >>> # With targets, uses prediction
    >>> D, method = estimate_D(data, targets)
    """
    if method == "auto":
        if targets is not None:
            method = "prediction"
        else:
            method = "compression"
    
    if method == "compression":
        D = compression_estimator(data)
        return D, "compression"
    
    elif method == "prediction":
        if targets is None:
            # Use next-step prediction
            data = np.atleast_1d(data)
            targets = data[1:]
            data = data[:-1]
        D = prediction_estimator(data, targets)
        return D, "prediction"
    
    elif method == "mi":
        if targets is None:
            data = np.atleast_1d(data)
            targets = data[1:]
            data = data[:-1]
        D = mutual_information_estimator(data, targets)
        return D, "mi"
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compare_estimators(
    observations: np.ndarray,
    targets: np.ndarray
) -> dict:
    """
    Compare all three D estimators on the same data.
    
    Parameters
    ----------
    observations : array-like
        Observation data
    targets : array-like
        Target environmental states
    
    Returns
    -------
    dict
        Dictionary with keys: compression, prediction, mi
        
    Notes
    -----
    Expected hierarchy: compression ≥ mi ≥ prediction
    """
    results = {
        "compression": compression_estimator(observations),
        "prediction": prediction_estimator(
            observations.reshape(-1, 1) if observations.ndim == 1 else observations,
            targets
        ),
        "mi": mutual_information_estimator(observations, targets)
    }
    
    # Verify hierarchy
    results["hierarchy_satisfied"] = (
        results["compression"] >= results["mi"] * 0.9 and  # Allow 10% tolerance
        results["mi"] >= results["prediction"] * 0.9
    )
    
    return results


if __name__ == "__main__":
    # Demonstration
    print("Data Richness Estimator Demo")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Case 1: Highly structured (periodic signal)
    print("\n1. Periodic signal (sin wave):")
    t = np.linspace(0, 100 * np.pi, 10000)
    structured = np.sin(t)
    D = compression_estimator(structured)
    print(f"   D_compress = {D:.3f}")
    
    # Case 2: Random noise
    print("\n2. Random noise:")
    noise = np.random.randn(10000)
    D = compression_estimator(noise)
    print(f"   D_compress = {D:.3f}")
    
    # Case 3: Mixed signal
    print("\n3. Signal + noise:")
    mixed = np.sin(t) + 0.5 * np.random.randn(10000)
    D = compression_estimator(mixed)
    print(f"   D_compress = {D:.3f}")
    
    # Case 4: Prediction comparison
    print("\n4. Prediction estimator comparison:")
    X = np.random.randn(1000, 3)
    Y = X @ np.array([1, 2, 3]) + 0.5 * np.random.randn(1000)
    
    D_pred = prediction_estimator(X, Y)
    D_comp = compression_estimator(np.hstack([X, Y.reshape(-1, 1)]))
    
    print(f"   D_compress = {D_comp:.3f}")
    print(f"   D_predict  = {D_pred:.3f}")
    print(f"   R² score   = {r2_score(Y, LinearRegression().fit(X, Y).predict(X)):.3f}")
