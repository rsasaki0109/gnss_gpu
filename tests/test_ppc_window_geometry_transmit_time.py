"""Unit tests for experiments/ppc_window_geometry.py's ``_compute_at_transmit_time``.

Synthetic-only: uses a small fake ephemeris stub instead of the PPC dataset,
so this exercises the per-satellite transmit-time iteration logic in
isolation. See PROGRESS.md TrackF for why this helper exists locally (the
module always imported it from ``gnss_gpu.io.ppc``, but that function does
not exist there at this repo revision).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from ppc_window_geometry import _compute_at_transmit_time  # noqa: E402
from gnss_gpu.io.ppc import C_LIGHT  # noqa: E402


class _FakeMovingSatEphemeris:
    """A satellite moving at constant velocity along +x, with a linear clock.

    ``position(t) = pos0 + vel * t``, ``clock(t) = clk0`` (constant, so the
    transmit-time iteration should converge to the exact geometric transmit
    time within a couple of iterations).
    """

    def __init__(self, pos0: np.ndarray, vel: np.ndarray, clk0: float) -> None:
        self.pos0 = np.asarray(pos0, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)
        self.clk0 = float(clk0)
        self.calls: list[float] = []

    def compute(self, gps_time, prn_list, obs_codes=None):
        self.calls.append(float(gps_time))
        n = len(prn_list)
        pos = self.pos0[None, :] + self.vel[None, :] * float(gps_time)
        pos = np.repeat(pos, n, axis=0)
        clk = np.full(n, self.clk0, dtype=np.float64)
        return pos, clk, list(prn_list)


def test_compute_at_transmit_time_converges_to_exact_transmit_epoch() -> None:
    # Satellite ~20,200 km away (typical MEO range), receiver clock-free.
    tow = 1000.0
    true_range_m = 2.02e7
    true_transmit_time = tow - true_range_m / C_LIGHT
    vel = np.array([500.0, 0.0, 0.0])  # m/s along-track
    # Choose pos0 so that position(true_transmit_time) is exactly at true_range_m on the x axis.
    pos0 = np.array([true_range_m - vel[0] * true_transmit_time, 0.0, 0.0])
    eph = _FakeMovingSatEphemeris(pos0=pos0, vel=vel, clk0=0.0)

    pseudorange_m = true_range_m  # zero clock bias, noiseless
    sat_ecef, sat_clk, used = _compute_at_transmit_time(
        eph, tow, ["G01"], ["C1C"], [pseudorange_m], n_iterations=3
    )
    assert used == ["G01"]
    expected_pos = pos0 + vel * true_transmit_time
    np.testing.assert_allclose(sat_ecef[0], expected_pos, atol=1e-6)
    assert sat_clk[0] == 0.0
    # Iterating refines the estimate; more than one eph.compute call per satellite.
    assert len(eph.calls) == 3


def test_compute_at_transmit_time_zero_iterations_uses_reception_time() -> None:
    eph = _FakeMovingSatEphemeris(pos0=np.array([1.0, 2.0, 3.0]), vel=np.zeros(3), clk0=0.0)
    sat_ecef, _sat_clk, used = _compute_at_transmit_time(
        eph, 500.0, ["G01"], ["C1C"], [2.0e7], n_iterations=0
    )
    assert used == ["G01"]
    assert eph.calls == [500.0]


def test_compute_at_transmit_time_drops_satellites_with_no_ephemeris() -> None:
    class _NoEph:
        def compute(self, gps_time, prn_list, obs_codes=None):
            return np.zeros((0, 3)), np.zeros(0), []

    sat_ecef, sat_clk, used = _compute_at_transmit_time(
        _NoEph(), 500.0, ["G01"], ["C1C"], [2.0e7], n_iterations=2
    )
    assert used == []
    assert sat_ecef.shape == (0, 3)
    assert sat_clk.shape == (0,)


def test_compute_at_transmit_time_empty_sat_list() -> None:
    eph = _FakeMovingSatEphemeris(pos0=np.zeros(3), vel=np.zeros(3), clk0=0.0)
    sat_ecef, sat_clk, used = _compute_at_transmit_time(eph, 500.0, [], [], [], n_iterations=2)
    assert used == []
    assert sat_ecef.shape == (0, 3)
    assert sat_clk.shape == (0,)
