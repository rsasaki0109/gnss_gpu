from __future__ import annotations

from experiments.analyze_wp176_surplus_validation import analyze


def _telemetry(
    tow: float,
    *,
    status: int,
    candidate_x: float,
    distance: float,
    subset_size: int,
    ratio: float,
    promoted: bool = False,
) -> dict[str, str]:
    return {
        "tow": f"{tow:.1f}",
        "library_status": str(status),
        "satellite_par_candidate_ecef_x": str(candidate_x),
        "satellite_par_candidate_ecef_y": "0",
        "satellite_par_candidate_ecef_z": "0",
        "satellite_par_surplus_evaluated": "1",
        "satellite_par_surplus_passed": "1",
        "satellite_par_surplus_max_distance_cycles": str(distance),
        "satellite_par_subset_size": str(subset_size),
        "satellite_par_ratio": str(ratio),
        "float_update_nis_per_observation": "1.0",
        "float_update_prefit_residual_rms_m": "1.0",
        "processing_runtime_ms": "20.0",
        "quality_gate_satellite_par_promoted": "1" if promoted else "0",
    }


def _position(tow: float, x: float, status: int) -> dict[str, float | int]:
    return {
        "tow": tow,
        "ecef_x": x,
        "ecef_y": 0.0,
        "ecef_z": 0.0,
        "status": status,
    }


def test_analyze_reports_additive_fix_and_strict_candidate_integrity() -> None:
    truth = {1.0: (0.0, 0.0, 0.0), 2.0: (0.0, 0.0, 0.0)}
    monitor_rows = [
        _telemetry(
            1.0,
            status=3,
            candidate_x=0.1,
            distance=0.05,
            subset_size=8,
            ratio=2.0,
        ),
        _telemetry(
            2.0,
            status=3,
            candidate_x=2.0,
            distance=0.20,
            subset_size=6,
            ratio=1.0,
        ),
    ]
    active_rows = [
        _telemetry(
            1.0,
            status=4,
            candidate_x=0.1,
            distance=0.05,
            subset_size=8,
            ratio=2.0,
            promoted=True,
        ),
        _telemetry(
            2.0,
            status=3,
            candidate_x=2.0,
            distance=0.20,
            subset_size=6,
            ratio=1.0,
        ),
    ]
    payload = analyze(
        "tokyo",
        monitor_rows,
        {1.0: _position(1.0, 0.1, 3), 2.0: _position(2.0, 2.0, 3)},
        active_rows,
        {1.0: _position(1.0, 0.1, 4), 2.0: _position(2.0, 2.0, 3)},
        truth,
    )

    assert payload["fixed_epoch_delta"] == 1
    assert payload["strict_candidates"] == 1
    assert payload["strict_correct_candidates"] == 1
    assert payload["strict_wrong_candidates"] == 0
    assert payload["satellite_par_promoted_epochs"] == 1
    assert payload["active_runtime_p95_100ms_pass"] is True
