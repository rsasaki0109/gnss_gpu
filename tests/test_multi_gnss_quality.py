from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gnss_gpu.range_model import geometric_ranges_sagnac

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "experiments"))

from exp_urbannav_baseline import run_wls
from gnss_gpu.multi_gnss import MultiGNSSSolver, SYSTEM_GALILEO, SYSTEM_GPS
from gnss_gpu.multi_gnss_quality import (
    MultiGNSSQualityVetoConfig,
    accept_multi_gnss_solution,
    compute_multi_gnss_quality_metrics,
    select_multi_gnss_solution,
)


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
    ranges = geometric_ranges_sagnac(true_pos, sat_ecef)
    pseudoranges = ranges.copy()
    pseudoranges[: len(gps_sats)] += gps_cb
    pseudoranges[len(gps_sats) :] += gps_cb + gal_bias
    reference_solution = np.array([*true_pos, gps_cb], dtype=np.float64)
    multi_position = np.asarray(true_pos, dtype=np.float64)
    multi_biases = {SYSTEM_GPS: gps_cb, SYSTEM_GALILEO: gps_cb + gal_bias}
    return sat_ecef, pseudoranges, system_ids, reference_solution, multi_position, multi_biases


def test_compute_multi_gnss_quality_metrics_counts_and_bias():
    sat_ecef, pseudoranges, system_ids, reference_solution, multi_position, multi_biases = _quality_scenario()
    metrics = compute_multi_gnss_quality_metrics(
        reference_solution=reference_solution,
        multi_position=multi_position,
        multi_biases=multi_biases,
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        system_ids=system_ids,
    )
    assert metrics.reference_satellite_count == 4
    assert metrics.multi_satellite_count == 6
    assert metrics.extra_satellite_count == 2
    assert metrics.multi_bias_range_m == 12.0
    assert metrics.multi_residual_p95_abs_m < 1e-6
    assert metrics.multi_residual_max_abs_m < 1e-6


def test_accept_multi_gnss_solution_rejects_large_bias_range():
    sat_ecef, pseudoranges, system_ids, reference_solution, multi_position, _ = _quality_scenario()
    metrics = compute_multi_gnss_quality_metrics(
        reference_solution=reference_solution,
        multi_position=multi_position,
        multi_biases={SYSTEM_GPS: 3000.0, SYSTEM_GALILEO: 3205.0},
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        system_ids=system_ids,
    )
    config = MultiGNSSQualityVetoConfig(
        residual_p95_max_m=100.0,
        residual_max_abs_m=250.0,
        bias_delta_max_m=100.0,
        extra_satellite_min=2,
    )
    assert not accept_multi_gnss_solution(metrics, config)


def test_select_multi_gnss_solution_falls_back_to_reference_on_bad_multi():
    sat_ecef, pseudoranges, system_ids, reference_solution, _, _ = _quality_scenario()
    config = MultiGNSSQualityVetoConfig(
        residual_p95_max_m=100.0,
        residual_max_abs_m=250.0,
        bias_delta_max_m=100.0,
        extra_satellite_min=2,
    )
    decision = select_multi_gnss_solution(
        reference_solution=reference_solution,
        multi_position=reference_solution[:3] + np.array([800.0, 0.0, 0.0]),
        multi_biases={SYSTEM_GPS: 3000.0, SYSTEM_GALILEO: 3300.0},
        sat_ecef=sat_ecef,
        pseudoranges=pseudoranges,
        system_ids=system_ids,
        config=config,
    )
    assert not decision.use_multi
    assert np.allclose(decision.position, reference_solution[:3])
    assert decision.clock_bias_m == reference_solution[3]


def test_run_wls_quality_veto_rejects_bad_multi_solution(monkeypatch):
    sat_ecef, pseudoranges, system_ids, reference_solution, _, _ = _quality_scenario()
    bad_position = reference_solution[:3] + np.array([900.0, 400.0, -300.0])

    def _fake_solve(self, sat_ecef, pseudoranges, system_ids, weights=None):
        del self, sat_ecef, pseudoranges, system_ids, weights
        return bad_position, {SYSTEM_GPS: 3000.0, SYSTEM_GALILEO: 3300.0}, 3

    monkeypatch.setattr(MultiGNSSSolver, "solve", _fake_solve)

    data = {
        "n_epochs": 1,
        "sat_ecef": [sat_ecef],
        "pseudoranges": [pseudoranges],
        "weights": [np.ones(len(pseudoranges), dtype=np.float64)],
        "system_ids": [system_ids],
        "constellations": ("G", "E"),
    }
    positions, _ = run_wls(
        data,
        quality_veto_config=MultiGNSSQualityVetoConfig(
            residual_p95_max_m=100.0,
            residual_max_abs_m=250.0,
            bias_delta_max_m=100.0,
            extra_satellite_min=2,
        ),
    )

    assert np.linalg.norm(positions[0, :3] - bad_position) > 100.0
    assert np.linalg.norm(positions[0, :3] - reference_solution[:3]) < 10.0
