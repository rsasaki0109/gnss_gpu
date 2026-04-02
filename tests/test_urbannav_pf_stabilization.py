from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from exp_urbannav_pf3d import (
    _reference_predict_velocity,
    _select_guide_velocity,
    _select_pf_epoch_measurements,
    _should_rescue_pf_epoch,
)
from gnss_gpu.multi_gnss import SYSTEM_GALILEO, SYSTEM_GPS
from gnss_gpu.multi_gnss_quality import MultiGNSSQualityVetoConfig


def _quality_scenario():
    true_pos = np.array([-3957199.0, 3310205.0, 3737911.0], dtype=np.float64)
    gps_cb = 3000.0
    gal_bias = 12.0
    gps_sats = np.array(
        [
            [-14985000.0, -3988000.0, 21474000.0],
            [-9575000.0, 15498000.0, 19457000.0],
            [7624000.0, -16218000.0, 19843000.0],
            [16305000.0, 12037000.0, 17183000.0],
        ],
        dtype=np.float64,
    )
    gal_sats = np.array(
        [
            [-12500000.0, 8700000.0, 22100000.0],
            [18300000.0, -5200000.0, 20400000.0],
        ],
        dtype=np.float64,
    )
    sat_ecef = np.vstack([gps_sats, gal_sats])
    system_ids = np.array(
        [SYSTEM_GPS] * len(gps_sats) + [SYSTEM_GALILEO] * len(gal_sats),
        dtype=np.int32,
    )
    ranges = np.linalg.norm(sat_ecef - true_pos.reshape(1, 3), axis=1)
    pseudoranges = ranges.copy()
    pseudoranges[: len(gps_sats)] += gps_cb
    pseudoranges[len(gps_sats) :] += gps_cb + gal_bias
    weights = np.ones(len(pseudoranges), dtype=np.float64)
    return true_pos, sat_ecef, pseudoranges, weights, system_ids


class _FakeSolver:
    def __init__(self, position: np.ndarray, biases: dict[int, float]):
        self._position = np.asarray(position, dtype=np.float64)
        self._biases = dict(biases)

    def solve(self, sat_ecef, pseudoranges, system_ids, weights):
        del sat_ecef, pseudoranges, system_ids, weights
        return self._position, self._biases, 3


def test_select_pf_epoch_measurements_falls_back_to_reference_system():
    true_pos, sat_ecef, pseudoranges, weights, system_ids = _quality_scenario()
    bad_position = true_pos + np.array([900.0, 400.0, -300.0], dtype=np.float64)
    sat_sel, pr_sel, w_sel, use_multi = _select_pf_epoch_measurements(
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        weights=weights,
        system_ids=system_ids,
        multi_solver=_FakeSolver(
            bad_position,
            {SYSTEM_GPS: 3000.0, SYSTEM_GALILEO: 3300.0},
        ),
        quality_veto_config=MultiGNSSQualityVetoConfig(
            residual_p95_max_m=100.0,
            residual_max_abs_m=250.0,
            bias_delta_max_m=100.0,
            extra_satellite_min=2,
        ),
    )
    assert not use_multi
    assert sat_sel.shape == (4, 3)
    assert pr_sel.shape == (4,)
    assert w_sel.shape == (4,)


def test_select_pf_epoch_measurements_keeps_clean_multi_epoch():
    true_pos, sat_ecef, pseudoranges, weights, system_ids = _quality_scenario()
    sat_sel, pr_sel, w_sel, use_multi = _select_pf_epoch_measurements(
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        weights=weights,
        system_ids=system_ids,
        multi_solver=_FakeSolver(
            true_pos,
            {SYSTEM_GPS: 3000.0, SYSTEM_GALILEO: 3012.0},
        ),
        quality_veto_config=MultiGNSSQualityVetoConfig(
            residual_p95_max_m=100.0,
            residual_max_abs_m=250.0,
            bias_delta_max_m=100.0,
            extra_satellite_min=2,
        ),
    )
    assert use_multi
    assert sat_sel.shape == sat_ecef.shape
    assert pr_sel.shape == pseudoranges.shape
    assert w_sel.shape == weights.shape


def test_should_rescue_pf_epoch_uses_reference_gap_threshold():
    estimate_state = np.array([100.0, 0.0, 0.0, 25.0], dtype=np.float64)
    reference_position = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    assert _should_rescue_pf_epoch(estimate_state, reference_position, 80.0)
    assert not _should_rescue_pf_epoch(estimate_state, reference_position, 120.0)


def test_reference_predict_velocity_uses_consecutive_reference_positions():
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 5.0, -2.0],
            [16.0, 8.0, -2.0],
        ],
        dtype=np.float64,
    )
    assert _reference_predict_velocity(reference_positions, 0, 1.0) is None
    velocity = _reference_predict_velocity(reference_positions, 1, 2.0)
    assert np.allclose(velocity, np.array([5.0, 2.5, -1.0], dtype=np.float64))


def test_select_guide_velocity_respects_init_only_and_fallback_modes():
    reference_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 5.0, -2.0],
        ],
        dtype=np.float64,
    )
    assert (
        _select_guide_velocity(
            reference_positions,
            epoch_index=1,
            dt=2.0,
            guide_mode="init_only",
            use_multi=True,
            selected_satellite_count=6,
        )
        is None
    )
    fallback_velocity = _select_guide_velocity(
        reference_positions,
        epoch_index=1,
        dt=2.0,
        guide_mode="fallback_only",
        use_multi=False,
        selected_satellite_count=4,
    )
    assert np.allclose(fallback_velocity, np.array([5.0, 2.5, -1.0], dtype=np.float64))
    assert (
        _select_guide_velocity(
            reference_positions,
            epoch_index=1,
            dt=2.0,
            guide_mode="fallback_only",
            use_multi=True,
            selected_satellite_count=4,
        )
        is None
    )
