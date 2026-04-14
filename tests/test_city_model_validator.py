"""Tests for city_model_validator.classify_satellite()."""

import math

import numpy as np
import pytest

from gnss_gpu.city_model_validator import (
    classify_satellite,
    validate_epoch,
    CN0_LOS_THRESHOLD,
    CN0_NLOS_THRESHOLD,
    ELEVATION_MASK_DEG,
)


class TestClassifySatellite:
    """Test the 4-way classification logic."""

    def test_los_predicted_strong_signal(self):
        """Model says LOS, signal is strong -> consistent."""
        assert classify_satellite(True, 40.0, 45.0) == "consistent"

    def test_nlos_predicted_weak_signal(self):
        """Model says NLOS, signal is weak -> consistent."""
        assert classify_satellite(False, 20.0, 45.0) == "consistent"

    def test_los_predicted_weak_signal(self):
        """Model says LOS, signal is weak -> missing building."""
        assert classify_satellite(True, 15.0, 45.0) == "missing_building"

    def test_nlos_predicted_strong_signal(self):
        """Model says NLOS, signal is strong -> phantom building."""
        assert classify_satellite(False, 40.0, 45.0) == "phantom_building"

    def test_low_elevation_is_ambiguous(self):
        """Satellites below elevation mask are always ambiguous."""
        assert classify_satellite(True, 40.0, 5.0) == "ambiguous"
        assert classify_satellite(False, 15.0, 5.0) == "ambiguous"

    def test_borderline_cn0_is_ambiguous(self):
        """C/N0 between thresholds is ambiguous."""
        mid_cn0 = (CN0_LOS_THRESHOLD + CN0_NLOS_THRESHOLD) / 2
        assert classify_satellite(True, mid_cn0, 45.0) == "ambiguous"
        assert classify_satellite(False, mid_cn0, 45.0) == "ambiguous"

    def test_nan_cn0_is_ambiguous(self):
        """NaN C/N0 is ambiguous."""
        assert classify_satellite(True, float("nan"), 45.0) == "ambiguous"
        assert classify_satellite(False, float("nan"), 45.0) == "ambiguous"

    def test_exact_threshold_los(self):
        """C/N0 exactly at LOS threshold."""
        assert classify_satellite(True, CN0_LOS_THRESHOLD, 45.0) == "consistent"

    def test_just_below_nlos_threshold(self):
        """C/N0 just below NLOS threshold."""
        assert classify_satellite(True, CN0_NLOS_THRESHOLD - 0.1, 45.0) == "missing_building"

    def test_elevation_at_mask(self):
        """Elevation exactly at mask -> ambiguous (< not <=)."""
        assert classify_satellite(True, 40.0, ELEVATION_MASK_DEG) == "consistent"


class TestValidateEpoch:
    """Test validate_epoch with synthetic data."""

    def test_all_consistent(self):
        """All LOS satellites with strong signal."""
        from gnss_gpu.raytrace import BuildingModel

        # No buildings -> all LOS
        building = BuildingModel.create_box(
            center=[1000, 1000, 1000], width=1, depth=1, height=1)

        rx = np.array([0.0, 0.0, 6378137.0])  # on Earth surface (north pole approx)
        sats = np.array([
            [0.0, 0.0, 6378137.0 + 20200e3],  # directly above
            [5e6, 0.0, 6378137.0 + 20200e3],
        ])
        cn0 = np.array([42.0, 38.0])

        result = validate_epoch(rx, sats, ["G01", "G02"], cn0, building)
        assert result.n_consistent == 2
        assert result.n_missing == 0
        assert result.n_phantom == 0
        assert result.model_score == 1.0
