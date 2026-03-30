"""Tests for cycle slip detection (pure Python, no CUDA required)."""

import numpy as np
import pytest

from gnss_gpu.cycle_slip import (
    detect_geometry_free,
    detect_melbourne_wubbena,
    detect_time_difference,
    L1_WAVELENGTH,
    L2_WAVELENGTH,
    WIDELANE_WAVELENGTH,
    _F1,
    _F2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_carrier(n_epoch=100, n_sat=6, seed=0):
    """Generate smooth carrier phase series with no cycle slips.

    Returns L1 and L2 carrier phase arrays in cycles, plus pseudoranges
    in metres that are consistent with the geometry.
    """
    rng = np.random.default_rng(seed)

    # Simulate slowly varying geometric range per satellite [m]
    # Start from a realistic range and add a gentle drift per epoch
    ranges = 22_000_000.0 + rng.uniform(-1e6, 1e6, size=(1, n_sat))
    drift = rng.uniform(-0.01, 0.01, size=(1, n_sat))  # m/epoch (slow motion)
    epoch_idx = np.arange(n_epoch).reshape(-1, 1)
    geo_range = ranges + drift * epoch_idx  # (n_epoch, n_sat)

    # Integer ambiguities (constant over time)
    N1 = rng.integers(-10, 11, size=(1, n_sat)).astype(float)
    N2 = rng.integers(-10, 11, size=(1, n_sat)).astype(float)

    # Carrier phase [cycles] = range / wavelength + N  (+ small noise)
    carrier_L1 = geo_range / L1_WAVELENGTH + N1 + rng.normal(0, 0.002, (n_epoch, n_sat))
    carrier_L2 = geo_range / L2_WAVELENGTH + N2 + rng.normal(0, 0.002, (n_epoch, n_sat))

    # Pseudoranges [m] = range + noise (small for MW test stability)
    pr_L1 = geo_range + rng.normal(0, 0.1, (n_epoch, n_sat))
    pr_L2 = geo_range + rng.normal(0, 0.1, (n_epoch, n_sat))

    return carrier_L1, carrier_L2, pr_L1, pr_L2


def _inject_slip(carrier, epoch, sat, slip_cycles, wavelength):
    """Inject a cycle slip of *slip_cycles* on *sat* starting at *epoch*.

    Modifies carrier in-place and returns the modified array.
    """
    carrier = carrier.copy()
    carrier[epoch:, sat] += slip_cycles
    return carrier


# ---------------------------------------------------------------------------
# Geometry-free detector
# ---------------------------------------------------------------------------

class TestGeometryFree:
    def test_clean_no_detection(self):
        L1, L2, _, _ = _make_clean_carrier()
        mask = detect_geometry_free(L1, L2, threshold=0.05)
        assert mask.shape == L1.shape
        assert not mask.any(), "No slips should be detected on clean data"

    def test_1_cycle_slip_L1(self):
        L1, L2, _, _ = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=50, sat=2, slip_cycles=1.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_geometry_free(L1_slipped, L2, threshold=0.05)
        assert mask[50, 2], "1-cycle L1 slip at epoch 50, sat 2 not detected"
        # No false detections on other satellites at epoch 50
        others = np.delete(mask[50], 2)
        assert not others.any(), "False detections on other satellites"

    def test_10_cycle_slip_L1(self):
        L1, L2, _, _ = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=30, sat=0, slip_cycles=10.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_geometry_free(L1_slipped, L2, threshold=0.05)
        assert mask[30, 0], "10-cycle slip not detected"

    def test_multiple_sats_independent(self):
        """Slip on one satellite should not flag others."""
        L1, L2, _, _ = _make_clean_carrier(n_sat=8)
        L1_slipped = _inject_slip(L1, epoch=40, sat=5, slip_cycles=3.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_geometry_free(L1_slipped, L2, threshold=0.05)
        assert mask[40, 5]
        # Only sat 5 flagged at epoch 40
        assert mask[40].sum() == 1

    def test_first_epoch_always_false(self):
        L1, L2, _, _ = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=0, sat=0, slip_cycles=5.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_geometry_free(L1_slipped, L2, threshold=0.05)
        assert not mask[0].any(), "First epoch should never be flagged"


# ---------------------------------------------------------------------------
# Melbourne-Wubbena detector
# ---------------------------------------------------------------------------

class TestMelbourneWubbena:
    def test_clean_no_detection(self):
        L1, L2, P1, P2 = _make_clean_carrier()
        mask = detect_melbourne_wubbena(L1, L2, P1, P2, threshold=1.0)
        assert mask.shape == L1.shape
        assert not mask.any(), "No slips on clean data"

    def test_1_cycle_slip_detected(self):
        L1, L2, P1, P2 = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=60, sat=1, slip_cycles=1.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_melbourne_wubbena(L1_slipped, L2, P1, P2, threshold=0.5)
        assert mask[60, 1], "1-cycle slip not detected by MW"

    def test_10_cycle_slip_detected(self):
        L1, L2, P1, P2 = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=20, sat=3, slip_cycles=10.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_melbourne_wubbena(L1_slipped, L2, P1, P2, threshold=0.5)
        assert mask[20, 3], "10-cycle slip not detected by MW"

    def test_multiple_sats_independent(self):
        L1, L2, P1, P2 = _make_clean_carrier(n_sat=8)
        L1_slipped = _inject_slip(L1, epoch=45, sat=7, slip_cycles=2.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_melbourne_wubbena(L1_slipped, L2, P1, P2, threshold=1.0)
        assert mask[45, 7], "Injected slip not detected"

    def test_first_epoch_always_false(self):
        L1, L2, P1, P2 = _make_clean_carrier()
        mask = detect_melbourne_wubbena(L1, L2, P1, P2, threshold=0.5)
        assert not mask[0].any()


# ---------------------------------------------------------------------------
# Time-difference detector
# ---------------------------------------------------------------------------

class TestTimeDifference:
    def test_clean_no_detection(self):
        L1, _, _, _ = _make_clean_carrier()
        mask = detect_time_difference(L1, threshold=0.5)
        assert mask.shape == L1.shape
        assert not mask.any(), "No slips on clean data"

    def test_1_cycle_slip_detected(self):
        L1, _, _, _ = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=70, sat=4, slip_cycles=1.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_time_difference(L1_slipped, threshold=0.5)
        assert mask[70, 4], "1-cycle slip not detected"

    def test_10_cycle_slip_detected(self):
        L1, _, _, _ = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=10, sat=0, slip_cycles=10.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_time_difference(L1_slipped, threshold=0.5)
        assert mask[10, 0], "10-cycle slip not detected"

    def test_multiple_sats_independent(self):
        L1, _, _, _ = _make_clean_carrier(n_sat=8)
        L1_slipped = _inject_slip(L1, epoch=55, sat=6, slip_cycles=5.0,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_time_difference(L1_slipped, threshold=0.5)
        assert mask[55, 6]
        assert mask[55].sum() == 1

    def test_first_epoch_always_false(self):
        L1, _, _, _ = _make_clean_carrier()
        mask = detect_time_difference(L1, threshold=0.5)
        assert not mask[0].any()

    def test_sub_threshold_no_detection(self):
        """A slip smaller than the threshold should not be detected."""
        L1, _, _, _ = _make_clean_carrier()
        L1_slipped = _inject_slip(L1, epoch=50, sat=2, slip_cycles=0.3,
                                  wavelength=L1_WAVELENGTH)
        mask = detect_time_difference(L1_slipped, threshold=0.5)
        assert not mask[50, 2], "Sub-threshold slip should not be flagged"
