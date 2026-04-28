from __future__ import annotations

import numpy as np

from experiments.gsdc2023_clock_state import (
    clean_clock_drift,
    clock_aid_enabled,
    clock_drift_seed_enabled,
    combine_clock_jump_masks,
    detect_clock_jumps_from_clock_bias,
    effective_multi_gnss_enabled,
    effective_position_source,
    factor_break_mask,
    segment_ranges,
)


def test_phone_policy_disables_mi8_multi_gnss_and_auto_source() -> None:
    assert effective_multi_gnss_enabled("test/course/mi8", True) is False
    assert effective_multi_gnss_enabled("test/course/xiaomimi8", True) is False
    assert effective_multi_gnss_enabled("test/course/pixel5", True) is True
    assert effective_multi_gnss_enabled("test/course/mi8", False) is False
    assert effective_position_source("test/course/xiaomimi8", "auto") == "raw_wls"
    assert effective_position_source("test/course/mi8", "gated") == "gated"


def test_clock_aid_and_seed_phone_policy() -> None:
    assert clock_aid_enabled("pixel4") is True
    assert clock_aid_enabled("sm-a505u") is True
    assert clock_aid_enabled("pixel5") is False
    assert clock_drift_seed_enabled("pixel4") is True
    assert clock_drift_seed_enabled("sm-a505u") is False
    assert clock_drift_seed_enabled("pixel5") is False


def test_clean_clock_drift_derives_mi8_drift_from_clock_bias() -> None:
    times_ms = np.array([0.0, 1000.0, 2000.0, 3000.0], dtype=np.float64)
    clock_bias_m = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)

    drift = clean_clock_drift(times_ms, clock_bias_m, None, "mi8")

    np.testing.assert_allclose(drift, np.full(4, -10.0), atol=1e-12)


def test_clean_clock_drift_filters_large_jumps_and_interpolates() -> None:
    times_ms = np.arange(5, dtype=np.float64) * 1000.0
    drift = np.array([1.0, 2.0, 80.0, 3.0, 4.0], dtype=np.float64)

    cleaned = clean_clock_drift(times_ms, None, drift, "pixel5")

    np.testing.assert_allclose(cleaned, np.array([1.0, 1.75, 2.5, 3.25, 4.0]), atol=1e-12)


def test_detect_and_combine_clock_jumps() -> None:
    jumps = detect_clock_jumps_from_clock_bias(np.array([0.0, 50.0, 151.0, 151.0]), "pixel4xl")
    combined = combine_clock_jump_masks(jumps, np.array([False, True, False, False]))

    np.testing.assert_array_equal(jumps, np.array([False, False, True, False]))
    np.testing.assert_array_equal(combined, np.array([False, True, True, False]))
    assert combine_clock_jump_masks(None, None) is None


def test_segment_ranges_split_on_clock_jump_inside_window() -> None:
    clock_jump = np.array([False, False, True, False, True, False], dtype=bool)

    assert segment_ranges(0, 6, clock_jump) == [(0, 2), (2, 4), (4, 6)]
    assert segment_ranges(2, 2, clock_jump) == []
    assert segment_ranges(1, 3, None) == [(1, 3)]


def test_factor_break_mask_combines_clock_jump_and_invalid_dt() -> None:
    clock_jump = np.array([False, False, True], dtype=bool)
    dt = np.array([1.0, 0.0, np.nan, 2.0], dtype=np.float64)

    mask = factor_break_mask(clock_jump, dt, 5)

    np.testing.assert_array_equal(mask, np.array([False, False, True, True, False]))
