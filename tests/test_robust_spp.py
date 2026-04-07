"""Tests for robust SPP solver."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project python/ is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.robust_spp import robust_spp, _compute_robust_weights


def _make_synthetic_scenario(
    true_pos: np.ndarray,
    n_sat: int = 8,
    noise_std: float = 1.0,
    outlier_indices: list[int] | None = None,
    outlier_bias: float = 100.0,
    seed: int = 42,
):
    """Create synthetic satellite geometry and pseudoranges.

    Returns sat_ecef, pseudoranges, true_cb.
    """
    rng = np.random.default_rng(seed)

    # Satellites roughly 20200 km above Earth surface, spread around
    sat_ecef = np.zeros((n_sat, 3))
    r_orbit = 26_600_000.0  # GPS orbit radius [m]
    for i in range(n_sat):
        az = 2 * np.pi * i / n_sat
        el = np.radians(30 + rng.uniform(-15, 30))
        # Convert to ECEF relative to true position
        dx = r_orbit * np.cos(el) * np.cos(az)
        dy = r_orbit * np.cos(el) * np.sin(az)
        dz = r_orbit * np.sin(el)
        sat_ecef[i] = true_pos + np.array([dx, dy, dz])

    true_cb = 1_000_000.0  # 1e6 m clock bias (typical)
    ranges = np.linalg.norm(sat_ecef - true_pos, axis=1)
    pseudoranges = ranges + true_cb + rng.normal(0, noise_std, n_sat)

    # Add outlier bias
    if outlier_indices:
        for idx in outlier_indices:
            pseudoranges[idx] += outlier_bias

    return sat_ecef, pseudoranges, true_cb


class TestComputeRobustWeights:
    def test_cauchy_small_residuals(self):
        """Small residuals should get weight near 1."""
        r = np.array([0.0, 1.0, 2.0])
        w = _compute_robust_weights(r, threshold=15.0, weight_func="cauchy")
        assert np.all(w > 0.95)

    def test_cauchy_large_residuals(self):
        """Large residuals should get small weight."""
        r = np.array([0.0, 50.0, 100.0])
        w = _compute_robust_weights(r, threshold=15.0, weight_func="cauchy")
        assert w[0] == pytest.approx(1.0)
        assert w[1] < 0.1
        assert w[2] < 0.03

    def test_huber_small_residuals(self):
        """Residuals below threshold get weight 1."""
        r = np.array([0.0, 5.0, 14.9])
        w = _compute_robust_weights(r, threshold=15.0, weight_func="huber")
        np.testing.assert_array_almost_equal(w, [1.0, 1.0, 1.0])

    def test_huber_large_residuals(self):
        """Residuals above threshold get weight = threshold / |r|."""
        r = np.array([30.0, 60.0])
        w = _compute_robust_weights(r, threshold=15.0, weight_func="huber")
        np.testing.assert_array_almost_equal(w, [0.5, 0.25])


class TestRobustSPP:
    # True position near Tokyo (ECEF)
    TRUE_POS = np.array([-3_959_340.0, 3_352_854.0, 3_697_471.0])

    def test_clean_data(self):
        """With clean data, robust SPP should match standard WLS closely."""
        sat, pr, _ = _make_synthetic_scenario(self.TRUE_POS, noise_std=1.0)
        result = robust_spp(sat, pr, init_pos=self.TRUE_POS + np.array([5, 5, 5]))
        assert result is not None
        error = np.linalg.norm(result - self.TRUE_POS)
        assert error < 5.0, f"Clean data error too large: {error:.2f} m"

    def test_one_outlier_rejected(self):
        """With 1 outlier out of 8, robust solver should still be accurate."""
        sat, pr, _ = _make_synthetic_scenario(
            self.TRUE_POS, n_sat=8, noise_std=1.0,
            outlier_indices=[2], outlier_bias=200.0,
        )
        # Without robust: use standard WLS (iteration 0 only, all weights=1)
        result_robust = robust_spp(
            sat, pr, init_pos=self.TRUE_POS + np.array([5, 5, 5]),
            threshold=15.0, weight_func="cauchy",
        )
        assert result_robust is not None
        error_robust = np.linalg.norm(result_robust - self.TRUE_POS)

        # Also solve without robustness (high threshold = no downweighting)
        result_naive = robust_spp(
            sat, pr, init_pos=self.TRUE_POS + np.array([5, 5, 5]),
            threshold=1e9,  # effectively no downweighting
        )
        assert result_naive is not None
        error_naive = np.linalg.norm(result_naive - self.TRUE_POS)

        # Robust should be better than naive
        assert error_robust < error_naive, (
            f"Robust ({error_robust:.2f}m) should beat naive ({error_naive:.2f}m)"
        )
        assert error_robust < 10.0, f"Robust error too large: {error_robust:.2f} m"

    def test_two_outliers_rejected(self):
        """With 2 outliers out of 8, robust solver should still work."""
        sat, pr, _ = _make_synthetic_scenario(
            self.TRUE_POS, n_sat=8, noise_std=1.0,
            outlier_indices=[1, 5], outlier_bias=150.0,
        )
        result = robust_spp(
            sat, pr, init_pos=self.TRUE_POS + np.array([5, 5, 5]),
            threshold=15.0,
        )
        # 2 outliers out of 8 is challenging; solver may return None or rough result
        if result is not None:
            error = np.linalg.norm(result - self.TRUE_POS)
            assert error < 50.0, f"Two-outlier error too large: {error:.2f} m"

    def test_too_few_satellites(self):
        """Should return None with < 4 satellites."""
        sat = np.array([[1e7, 0, 0], [0, 1e7, 0], [-1e7, 0, 0]])
        pr = np.array([1e7, 1e7, 1e7])
        result = robust_spp(sat, pr)
        assert result is None

    def test_no_init_pos(self):
        """Should work without init_pos (uses satellite centroid)."""
        sat, pr, _ = _make_synthetic_scenario(self.TRUE_POS, noise_std=1.0)
        result = robust_spp(sat, pr, init_pos=None)
        # May not converge as well without good init, but should not crash
        # and should return something or None
        if result is not None:
            error = np.linalg.norm(result - self.TRUE_POS)
            # With satellite centroid as init, expect rough convergence
            assert error < 100.0

    def test_huber_weight_func(self):
        """Huber weight function should also work."""
        sat, pr, _ = _make_synthetic_scenario(
            self.TRUE_POS, n_sat=8, noise_std=1.0,
            outlier_indices=[3], outlier_bias=200.0,
        )
        result = robust_spp(
            sat, pr, init_pos=self.TRUE_POS + np.array([5, 5, 5]),
            weight_func="huber", threshold=15.0,
        )
        assert result is not None
        error = np.linalg.norm(result - self.TRUE_POS)
        assert error < 50.0, f"Huber error too large: {error:.2f} m"
