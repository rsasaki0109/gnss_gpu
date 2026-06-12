"""Tests for the transmission-time / Sagnac satellite-position correction."""

import math

import numpy as np
import pytest

from gnss_gpu.validation.real_residuals import (
    EARTH_ROTATION_RATE,
    sagnac_rotate,
    transmission_time_sat_positions,
)


def test_sagnac_zero_travel_is_identity():
    p = np.array([1.0e7, 2.0e7, 3.0e6])
    np.testing.assert_allclose(sagnac_rotate(p, 0.0), p, rtol=0, atol=0)


def test_sagnac_rotation_about_z_only():
    p = np.array([1.0e7, 0.0, 5.0e6])
    tau = 0.07
    out = sagnac_rotate(p, tau)
    theta = EARTH_ROTATION_RATE * tau
    # +Z component preserved; x/y rotated by -theta (clockwise about +Z).
    assert out[2] == pytest.approx(p[2])
    assert out[0] == pytest.approx(math.cos(theta) * p[0])
    assert out[1] == pytest.approx(-math.sin(theta) * p[0])
    np.testing.assert_allclose(np.linalg.norm(out[:2]), np.linalg.norm(p[:2]))


def test_sagnac_magnitude_is_tens_of_metres():
    # A GPS-scale tangential position with a ~0.07 s transit rotates by ~tens of m.
    p = np.array([2.0e7, 0.0, 0.0])
    shift = np.linalg.norm(sagnac_rotate(p, 0.07) - p)
    assert 50.0 < shift < 150.0


class _FakeEph:
    """Returns a satellite moving at constant ECEF velocity; records query time."""

    def __init__(self, pos0, vel, t0):
        self.pos0 = np.asarray(pos0, float)
        self.vel = np.asarray(vel, float)
        self.t0 = t0
        self.queried_at = []

    def compute(self, t, sid_list, obs_codes=None):
        self.queried_at.append(t)
        pos = self.pos0 + self.vel * (t - self.t0)
        return np.array([pos]), np.array([0.0]), list(sid_list)


def test_transmission_time_queries_earlier_epoch():
    rx = np.array([-3.0e6, 4.0e6, 3.7e6])
    tow = 100.0
    # reception-time satellite position ~2.2e7 m away -> travel ~0.073 s.
    sat_recep = rx + np.array([1.5e7, 1.0e7, 1.0e7])
    eph = _FakeEph(sat_recep, vel=np.array([2000.0, -1000.0, 500.0]), t0=tow)
    out = transmission_time_sat_positions(
        eph, tow, rx, ["G01"], sat_recep[None, :])
    tau = float(np.linalg.norm(sat_recep - rx)) / 299792458.0
    # Queried the ephemeris at the earlier transmission epoch.
    assert eph.queried_at[0] == pytest.approx(tow - tau)
    assert 0.06 < tau < 0.09
    assert out.shape == (1, 3)
    assert np.all(np.isfinite(out[0]))


def test_unavailable_satellite_is_nan():
    class _Empty:
        def compute(self, t, sid_list, obs_codes=None):
            return np.empty((0, 3)), np.empty((0,)), []

    rx = np.zeros(3)
    sat_recep = np.array([[2.0e7, 0.0, 0.0]])
    out = transmission_time_sat_positions(_Empty(), 0.0, rx, ["G01"], sat_recep)
    assert out.shape == (1, 3)
    assert np.all(np.isnan(out[0]))
