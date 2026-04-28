from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.smooth_gsdc2023_submission import (
    SmoothConfig,
    haversine_m,
    latlon_to_local_m,
    local_m_to_latlon,
    score_dataframe,
    smooth_dataframe,
)


def _df_from_local(east_m: np.ndarray, north_m: np.ndarray, *, gap_index: int | None = None) -> pd.DataFrame:
    lat, lon = local_m_to_latlon(east_m, north_m, 37.0, -122.0)
    times = np.arange(len(east_m), dtype=np.int64) * 1000
    if gap_index is not None:
        times[gap_index:] += 10_000
    return pd.DataFrame(
        {
            "tripId": "trip/phone",
            "UnixTimeMillis": times,
            "LatitudeDegrees": lat,
            "LongitudeDegrees": lon,
            "GroundTruthLatitudeDegrees": lat,
            "GroundTruthLongitudeDegrees": lon,
        },
    )


def test_latlon_local_roundtrip() -> None:
    lat = np.array([37.0, 37.0001, 36.9999])
    lon = np.array([-122.0, -121.9999, -122.0002])

    east, north, lat0, lon0 = latlon_to_local_m(lat, lon)
    out_lat, out_lon = local_m_to_latlon(east, north, lat0, lon0)

    np.testing.assert_allclose(out_lat, lat, atol=1e-12)
    np.testing.assert_allclose(out_lon, lon, atol=1e-12)


def test_hampel_repairs_isolated_outlier_and_improves_score() -> None:
    truth_east = np.arange(9, dtype=np.float64) * 2.0
    truth_north = np.zeros(9, dtype=np.float64)
    observed_east = truth_east.copy()
    observed_north = truth_north.copy()
    observed_north[4] = 30.0
    df = _df_from_local(observed_east, observed_north)
    truth_lat, truth_lon = local_m_to_latlon(truth_east, truth_north, 37.0, -122.0)
    df["GroundTruthLatitudeDegrees"] = truth_lat
    df["GroundTruthLongitudeDegrees"] = truth_lon

    before = score_dataframe(df)
    smoothed, stats = smooth_dataframe(
        df,
        SmoothConfig(
            median_window=5,
            smooth_window=1,
            max_correction_m=50.0,
            hampel_sigma=3.0,
            hampel_min_m=5.0,
        ),
    )
    after = score_dataframe(smoothed)

    assert before is not None and after is not None
    assert stats.hampel_rows == 1
    assert after["score_m"] < before["score_m"]
    fixed_error = haversine_m(
        smoothed.loc[[4], "LatitudeDegrees"].to_numpy(),
        smoothed.loc[[4], "LongitudeDegrees"].to_numpy(),
        truth_lat[[4]],
        truth_lon[[4]],
    )[0]
    assert fixed_error < 1.0


def test_segment_gap_prevents_cross_reset_smoothing() -> None:
    east = np.array([0.0, 2.0, 4.0, 200.0, 202.0, 204.0])
    north = np.zeros_like(east)
    df = _df_from_local(east, north, gap_index=3)

    smoothed, stats = smooth_dataframe(
        df,
        SmoothConfig(
            median_window=5,
            smooth_window=5,
            blend=1.0,
            max_correction_m=100.0,
            segment_gap_ms=3000.0,
            segment_step_m=1000.0,
            min_segment_points=3,
        ),
    )

    out_east, _, _, _ = latlon_to_local_m(
        smoothed["LatitudeDegrees"].to_numpy(),
        smoothed["LongitudeDegrees"].to_numpy(),
        origin_lat_deg=df.loc[0, "LatitudeDegrees"],
        origin_lon_deg=df.loc[0, "LongitudeDegrees"],
    )
    assert stats.segments == 2
    np.testing.assert_allclose(out_east, east, atol=1e-6)


def test_max_correction_cap_limits_outlier_repair() -> None:
    truth_east = np.arange(7, dtype=np.float64)
    truth_north = np.zeros(7, dtype=np.float64)
    observed_north = truth_north.copy()
    observed_north[3] = 100.0
    df = _df_from_local(truth_east, observed_north)

    smoothed, stats = smooth_dataframe(
        df,
        SmoothConfig(
            median_window=5,
            smooth_window=1,
            max_correction_m=10.0,
            hampel_sigma=3.0,
            hampel_min_m=5.0,
        ),
    )
    out_east, out_north, _, _ = latlon_to_local_m(
        smoothed["LatitudeDegrees"].to_numpy(),
        smoothed["LongitudeDegrees"].to_numpy(),
        origin_lat_deg=df.loc[0, "LatitudeDegrees"],
        origin_lon_deg=df.loc[0, "LongitudeDegrees"],
    )
    original_east, original_north, _, _ = latlon_to_local_m(
        df["LatitudeDegrees"].to_numpy(),
        df["LongitudeDegrees"].to_numpy(),
        origin_lat_deg=df.loc[0, "LatitudeDegrees"],
        origin_lon_deg=df.loc[0, "LongitudeDegrees"],
    )
    correction = np.hypot(out_east - original_east, out_north - original_north)

    assert stats.hampel_rows == 1
    assert correction[3] <= 10.0 + 1e-6


def test_gaussian_kernel_is_supported_and_unknown_kernel_fails() -> None:
    east = np.array([0.0, 1.0, 4.0, 3.0, 4.0, 5.0, 6.0])
    north = np.zeros_like(east)
    df = _df_from_local(east, north)

    smoothed, stats = smooth_dataframe(
        df,
        SmoothConfig(
            median_window=5,
            smooth_window=5,
            blend=1.0,
            max_correction_m=10.0,
            smooth_kernel="gaussian",
            gaussian_sigma=1.0,
        ),
    )

    assert stats.corrected_rows > 0
    assert np.isfinite(smoothed["LatitudeDegrees"]).all()
    with pytest.raises(ValueError, match="unsupported smooth kernel"):
        smooth_dataframe(df, SmoothConfig(smooth_window=5, smooth_kernel="bad"))
