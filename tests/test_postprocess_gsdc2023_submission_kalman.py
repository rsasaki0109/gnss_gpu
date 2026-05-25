"""Unit tests for 1D RTS Kalman smoother."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.postprocess_gsdc2023_submission_kalman import (
    apply_kalman_smoothing_to_submission,
    rts_smooth_1d,
)


def test_rts_smoother_returns_same_length():
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    dt = np.ones(5)
    out = rts_smooth_1d(z, dt, sigma_a=1.0, sigma_z=2.0)
    assert out.shape == z.shape


def test_rts_smoother_passes_through_constant_velocity():
    # Perfect CV motion -> smoother should not change the trajectory shape
    # materially (CV is exactly what the motion model expects).
    n = 50
    z = np.arange(n, dtype=np.float64)  # 1 m/s
    dt = np.ones(n - 1)
    out = rts_smooth_1d(z, dt, sigma_a=1.0, sigma_z=2.0)
    # Average position should be very close to original at interior points.
    diff = np.abs(out - z)
    assert diff[5:-5].max() < 1.0  # within ~1m at interior


def test_rts_smoother_reduces_iid_noise_on_static_target():
    # Stationary target with 1m std measurement noise -> smoother should
    # collapse the noise toward the mean.
    rng = np.random.default_rng(0)
    n = 100
    z = rng.normal(loc=0.0, scale=1.0, size=n)
    dt = np.ones(n - 1)
    out = rts_smooth_1d(z, dt, sigma_a=0.05, sigma_z=1.0)  # small process noise
    # Output std should be substantially smaller than input std
    assert np.std(out) < 0.6 * np.std(z)


def test_apply_kalman_per_trip_isolation():
    rng = np.random.default_rng(1)
    n = 60
    rows = []
    # Trip A: noisy stationary
    for i in range(n):
        rows.append(("A", i * 1000, 37.0 + rng.normal(0, 1e-6), -122.0 + rng.normal(0, 1e-6)))
    # Trip B: smooth constant velocity
    for i in range(n):
        rows.append(("B", i * 1000, 37.0 + i * 5e-6, -121.0))
    df = pd.DataFrame(rows, columns=["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"])
    out, stats = apply_kalman_smoothing_to_submission(df, sigma_a=0.5, sigma_z=2.0)
    assert stats["trips"] == 2
    a = out[out["tripId"] == "A"]
    # A should have lower noise variance after smoothing.
    assert a["LatitudeDegrees"].std() < df[df["tripId"] == "A"]["LatitudeDegrees"].std()


def test_apply_kalman_short_trip_passthrough():
    # n < 3 -> skipped (kept as-is)
    df = pd.DataFrame({
        "tripId": ["A", "A"],
        "UnixTimeMillis": [0, 1000],
        "LatitudeDegrees": [37.0, 37.0001],
        "LongitudeDegrees": [-122.0, -122.0001],
    })
    out, stats = apply_kalman_smoothing_to_submission(df, sigma_a=1.0, sigma_z=2.0)
    # No change because trip too short to smooth.
    np.testing.assert_array_equal(out["LatitudeDegrees"].to_numpy(), df["LatitudeDegrees"].to_numpy())
