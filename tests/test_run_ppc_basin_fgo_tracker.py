from __future__ import annotations

import csv

import numpy as np

from experiments.run_ppc_basin_fgo_tracker import (
    track_basin_rows,
    write_validated_pf_feedback,
)
from gnss_gpu.basin_imu_bridge import CausalBasinImuPredictor, PPCImuSamples


def _row(epoch: int, rank: int, fixed: int, evidence: float, passed: bool) -> dict:
    integers = []
    for satellite in range(2, 8):
        integers.append(
            {
                "satellite": f"G{satellite:02d}",
                "reference_satellite": "G01",
                "signal": 0,
                "segment_index": 0,
                "reference_segment_index": 0,
                "wavelength_m": 0.1902936728,
                "fixed_cycles": fixed + satellite,
            }
        )
    return {
        "schema": "gnsspp_multisd_basin_v1",
        "epoch_index": epoch,
        "tow": float(epoch),
        "rank": rank,
        "group_index": 0,
        "group_rank": rank,
        "evaluated": True,
        "pass": passed,
        "position_ecef": [float(rank), 0.0, 0.0],
        "velocity_valid": False,
        "velocity_ecef_mps": [0.0, 0.0, 0.0],
        "position_covariance_valid": True,
        "position_covariance_m2": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "relative_log_evidence": evidence,
        "incremental_log_likelihood": evidence,
        "incremental_likelihood_rows": 8,
        "fixed_integers": integers,
    }


def _with_carrier_partition_costs(
    row: dict, left_cost: float, right_cost: float
) -> dict:
    residuals = []
    for index, satellite in enumerate(("G20", "G21", "G22", "G23")):
        normalized = left_cost if index % 2 == 0 else right_cost
        residuals.append(
            {
                "epoch_index": row["epoch_index"],
                "satellite": satellite,
                "reference_satellite": "G01",
                "signal": 0,
                "kind": "carrier",
                "normalized_residual": normalized,
                "pass": True,
            }
        )
    row["validation_residuals"] = residuals
    return row


def _with_native_imu(row: dict, y_m: float) -> dict:
    row["imu_fgo"] = {
        "available": True,
        "converged": True,
        "position_ecef": [6_378_137.0, y_m, 0.0],
        "velocity_nav_mps": [1.0, 0.0, 0.0],
    }
    return row


def test_tracker_requires_posterior_streak_and_unique_independent_pass() -> None:
    rows = {
        epoch: [
            _row(epoch, 0, 8, 0.0, True),
            _row(epoch, 1, 9, -20.0, False),
        ]
        for epoch in range(3)
    }
    output = track_basin_rows(
        rows,
        likelihood_temperature=1.0,
        fix_gamma_threshold=0.99,
        fix_min_streak=3,
    )
    assert [row["shadow_fixed"] for row in output] == [0, 0, 1]
    assert all(row["unique_holdout_pass"] == 1 for row in output)


def test_tracker_abstains_when_two_holdout_hypotheses_pass() -> None:
    rows = {
        epoch: [
            _row(epoch, 0, 8, 0.0, True),
            _row(epoch, 1, 9, -20.0, True),
        ]
        for epoch in range(3)
    }
    output = track_basin_rows(rows, likelihood_temperature=1.0)
    assert all(row["shadow_fixed"] == 0 for row in output)
    assert all(row["unique_holdout_pass"] == 0 for row in output)


def test_disjoint_holdout_consensus_selects_only_shared_partition_winner() -> None:
    rows = {
        epoch: [
            _with_carrier_partition_costs(_row(epoch, 0, 8, 0.0, True), 0.1, 0.1),
            _with_carrier_partition_costs(_row(epoch, 1, 9, -20.0, True), 0.5, 0.5),
        ]
        for epoch in range(3)
    }
    output = track_basin_rows(
        rows,
        likelihood_temperature=1.0,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
        disjoint_holdout_consensus=True,
        disjoint_holdout_margin=0.02,
    )
    assert [row["disjoint_holdout_selected"] for row in output] == [1, 1, 1]
    assert [row["selected_rank"] for row in output] == [0, 0, 0]
    assert [row["shadow_fixed"] for row in output] == [0, 1, 1]
    assert all(row["unique_holdout_pass"] == 0 for row in output)


def test_disjoint_holdout_consensus_abstains_on_partition_disagreement() -> None:
    rows = {
        epoch: [
            _with_carrier_partition_costs(_row(epoch, 0, 8, 0.0, True), 0.1, 0.5),
            _with_carrier_partition_costs(_row(epoch, 1, 9, -20.0, True), 0.5, 0.1),
        ]
        for epoch in range(3)
    }
    output = track_basin_rows(rows, disjoint_holdout_consensus=True)
    assert all(row["disjoint_holdout_selected"] == 0 for row in output)
    assert all(row["shadow_fixed"] == 0 for row in output)


def test_disjoint_holdout_consensus_abstains_on_inconsistent_carrier_pass() -> None:
    rows = {
        epoch: [
            _with_carrier_partition_costs(_row(epoch, 0, 8, 0.0, True), 0.1, 0.1),
            _with_carrier_partition_costs(_row(epoch, 1, 9, -20.0, True), 0.5, 0.5),
        ]
        for epoch in range(3)
    }
    for epoch_rows in rows.values():
        for residual in epoch_rows[0]["validation_residuals"][2:]:
            residual["pass"] = False
    output = track_basin_rows(rows, disjoint_holdout_consensus=True)
    assert all(row["disjoint_holdout_selected"] == 0 for row in output)
    assert all(row["shadow_fixed"] == 0 for row in output)


def test_tracker_treats_missing_evaluated_group_as_fail_closed_gap() -> None:
    rows = {
        0: [_row(0, 0, 8, 0.0, True), _row(0, 1, 9, -20.0, False)],
        1: [{**_row(1, 0, 8, 0.0, True), "evaluated": False}],
        2: [_row(2, 0, 8, 0.0, True), _row(2, 1, 9, -20.0, False)],
    }
    output = track_basin_rows(
        rows,
        likelihood_temperature=1.0,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
    )
    assert [row["shadow_fixed"] for row in output] == [0, 0, 0]
    assert output[1]["candidate_count"] == 0
    assert output[1]["transition_branches"] == 0
    assert output[1]["fix_streak"] == 0


def test_tracker_matches_holdout_by_assignment_not_dedup_source_rank() -> None:
    rows = {}
    for epoch in range(2):
        failed = _row(epoch, 0, 8, 0.0, False)
        passed = _row(epoch, 1, 8, -0.01, True)
        # Same assignment and nearby state merge into one basin whose source
        # ordering is not a reliable current-hypothesis identity.
        passed["position_ecef"] = [0.01, 0.0, 0.0]
        rows[epoch] = [failed, passed]
    output = track_basin_rows(
        rows,
        likelihood_temperature=1.0,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
    )
    assert output[-1]["unique_holdout_pass"] == 1
    assert output[-1]["shadow_fixed"] == 1
    assert output[-1]["selected_rank"] == 1


def test_validation_conditioning_requires_consecutive_unique_passes() -> None:
    rows = {
        0: [_row(0, 0, 8, 0.0, True), _row(0, 1, 9, -20.0, False)],
        1: [_row(1, 0, 8, 0.0, True), _row(1, 1, 9, -20.0, True)],
        2: [_row(2, 0, 8, 0.0, True), _row(2, 1, 9, -20.0, False)],
    }
    output = track_basin_rows(
        rows,
        likelihood_temperature=1.0,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
    )
    assert [row["fix_streak"] for row in output] == [1, 0, 1]
    assert all(row["shadow_fixed"] == 0 for row in output)


def test_validation_gap_tolerance_preserves_only_compatible_strict_passes() -> None:
    rows = {
        0: [_row(0, 0, 8, 0.0, True), _row(0, 1, 9, -20.0, False)],
        1: [_row(1, 0, 8, 0.0, True), _row(1, 1, 9, -20.0, True)],
        2: [_row(2, 0, 8, 0.0, True), _row(2, 1, 9, -20.0, False)],
    }
    output = track_basin_rows(
        rows,
        likelihood_temperature=1.0,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
        validation_gap_tolerance_epochs=1,
    )
    assert [row["validated_fix_streak"] for row in output] == [1, 1, 2]
    assert [row["validation_gap_epochs"] for row in output] == [0, 1, 0]
    assert [row["shadow_fixed"] for row in output] == [0, 0, 1]


def test_tracker_applies_causal_imu_prediction_between_native_epochs() -> None:
    tow = np.arange(-1.0, 2.01, 0.01)
    samples = PPCImuSamples(
        tow,
        np.tile([0.0, 0.0, 9.81], (tow.size, 1)),
        np.zeros((tow.size, 3)),
    )
    rows = {}
    for epoch in range(3):
        row = _row(epoch, 0, 8, 0.0, True)
        row["position_ecef"] = [6_378_137.0, 0.0, 0.0]
        rows[epoch] = [row]
    output = track_basin_rows(
        rows,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
        imu_predictor=CausalBasinImuPredictor(samples),
    )
    assert [row["imu_used"] for row in output] == [0, 1, 1]
    assert all(row["imu_samples"] >= 99 for row in output[1:])
    assert all(row["imu_position_correction_m"] < 1.0e-8 for row in output[1:])


def test_tracker_uses_embedded_native_imu_fgo_as_proposal_only() -> None:
    rows = {}
    for epoch in range(3):
        row = _row(epoch, 0, 8, 0.0, True)
        row["position_ecef"] = [6_378_137.0, float(epoch), 0.0]
        row["imu_fgo"] = {
            "available": True,
            "converged": True,
            "position_ecef": [6_378_137.0, float(epoch), 0.0],
            "velocity_nav_mps": [1.0, 0.0, 0.0],
        }
        rows[epoch] = [row]
    output = track_basin_rows(
        rows,
        fix_gamma_threshold=0.99,
        fix_min_streak=2,
        native_imu_fgo=True,
    )
    assert [row["native_imu_fgo_available"] for row in output] == [1, 1, 1]
    assert [row["native_imu_motion_used"] for row in output] == [0, 1, 1]
    assert [row["imu_source"] for row in output] == [
        "none",
        "native_fgo",
        "native_fgo",
    ]
    # IMU feedback cannot bypass the unchanged unique-holdout/streak contract.
    assert [row["shadow_fixed"] for row in output] == [0, 1, 1]


def test_tracker_fails_closed_on_malformed_native_imu_fgo() -> None:
    row = _row(0, 0, 8, 0.0, True)
    row["imu_fgo"] = {
        "available": True,
        "converged": True,
        "position_ecef": [None, 0.0, 0.0],
        "velocity_nav_mps": [0.0, 0.0, 0.0],
    }
    output = track_basin_rows({0: [row]}, native_imu_fgo=True)
    assert output[0]["native_imu_fgo_available"] == 0
    assert output[0]["native_imu_motion_used"] == 0


def test_tracker_fails_closed_on_recovered_native_imu_fgo() -> None:
    row = _row(0, 0, 8, 0.0, True)
    row["imu_fgo"] = {
        "available": True,
        "converged": True,
        "recovery_epochs": 1,
        "fault_reason": "ok",
        "position_ecef": [6_378_137.0, 0.0, 0.0],
        "velocity_nav_mps": [0.0, 0.0, 0.0],
    }
    output = track_basin_rows({0: [row]}, native_imu_fgo=True)
    assert output[0]["native_imu_fgo_available"] == 0
    assert output[0]["native_imu_motion_used"] == 0


def test_native_imu_aperture_selects_only_among_holdout_passes() -> None:
    rows = {}
    for epoch in range(4):
        preferred = _row(epoch, 0, 8, 0.0, True)
        alternate = _row(epoch, 1, 9, -0.1, epoch > 0)
        preferred["position_ecef"] = [6_378_137.0, float(epoch), 0.0]
        alternate["position_ecef"] = [6_378_137.0, float(epoch) + 0.20, 0.0]
        payload = {
            "available": True,
            "converged": True,
            "position_ecef": [6_378_137.0, float(epoch) + 0.01, 0.0],
            "velocity_nav_mps": [1.0, 0.0, 0.0],
        }
        preferred["imu_fgo"] = payload
        alternate["imu_fgo"] = payload
        rows[epoch] = [preferred, alternate]
    output = track_basin_rows(
        rows,
        native_imu_fgo=True,
        native_imu_aperture_m=0.5,
        native_imu_aperture_margin_m=0.05,
    )
    assert [row["strict_passing_candidates"] for row in output] == [1, 2, 2, 2]
    assert [row["unique_holdout_pass"] for row in output] == [1, 0, 0, 0]
    assert [row["imu_aperture_selected"] for row in output] == [0, 1, 1, 1]
    assert [row["shadow_fixed"] for row in output] == [0, 0, 1, 1]


def test_native_imu_motion_can_accelerate_but_not_bypass_gnss_streak() -> None:
    rows = {}
    for epoch in range(2):
        row = _row(epoch, 0, 8, 0.0, True)
        row["position_ecef"] = [6_378_137.0, float(epoch), 0.0]
        row["imu_fgo"] = {
            "available": True,
            "converged": True,
            "position_ecef": [6_378_137.0, float(epoch), 0.0],
            "velocity_nav_mps": [1.0, 0.0, 0.0],
        }
        rows[epoch] = [row]
    output = track_basin_rows(
        rows,
        native_imu_fgo=True,
        native_imu_fix_min_streak=2,
        native_imu_motion_gate_m=0.3,
    )
    assert [row["validated_fix_streak"] for row in output] == [1, 2]
    assert [row["native_validation_streak"] for row in output] == [1, 2]
    assert [row["imu_accelerated_fix"] for row in output] == [0, 1]
    assert [row["shadow_fixed"] for row in output] == [0, 1]


def test_causal_imu_motion_consensus_uses_only_prior_fixed_anchors() -> None:
    rows = {}
    for epoch in range(9):
        preferred = _with_native_imu(_row(epoch, 0, 8, 0.0, True), float(epoch))
        preferred["position_ecef"] = [6_378_137.0, float(epoch), 0.0]
        rows[epoch] = [preferred]
    preferred = _with_carrier_partition_costs(
        _with_native_imu(_row(9, 0, 8, 0.0, True), 9.0), 0.1, 0.1
    )
    alternate = _with_carrier_partition_costs(
        _with_native_imu(_row(9, 1, 9, -0.1, True), 9.0), 0.2, 0.2
    )
    preferred["position_ecef"] = [6_378_137.0, 9.0, 0.0]
    alternate["position_ecef"] = [6_378_137.0, 9.25, 0.0]
    rows[9] = [preferred, alternate]

    output = track_basin_rows(
        rows,
        fix_min_streak=2,
        native_imu_fgo=True,
        causal_imu_motion_consensus=True,
        disjoint_holdout_min_carrier_fraction=0.1,
        causal_imu_motion_min_carrier_fraction=0.75,
    )
    assert output[9]["strict_passing_candidates"] == 2
    assert output[9]["causal_imu_motion_anchor_count"] >= 6
    assert output[9]["causal_imu_motion_selected"] == 1
    assert output[9]["selected_rank"] == 0
    assert output[9]["shadow_fixed"] == 1


def test_causal_imu_motion_consensus_rejects_inconsistent_carrier_details() -> None:
    rows = {}
    for epoch in range(9):
        preferred = _with_native_imu(_row(epoch, 0, 8, 0.0, True), float(epoch))
        preferred["position_ecef"] = [6_378_137.0, float(epoch), 0.0]
        rows[epoch] = [preferred]
    preferred = _with_carrier_partition_costs(
        _with_native_imu(_row(9, 0, 8, 0.0, True), 9.0), 0.1, 0.1
    )
    alternate = _with_carrier_partition_costs(
        _with_native_imu(_row(9, 1, 9, -0.1, True), 9.0), 0.2, 0.2
    )
    preferred["position_ecef"] = [6_378_137.0, 9.0, 0.0]
    alternate["position_ecef"] = [6_378_137.0, 9.25, 0.0]
    for residual in preferred["validation_residuals"][2:]:
        residual["pass"] = False
    rows[9] = [preferred, alternate]

    output = track_basin_rows(
        rows,
        fix_min_streak=2,
        native_imu_fgo=True,
        causal_imu_motion_consensus=True,
        disjoint_holdout_min_carrier_fraction=0.1,
        causal_imu_motion_min_carrier_fraction=0.75,
    )
    assert output[9]["causal_imu_motion_selected"] == 0
    assert output[9]["shadow_fixed"] == 0


def test_pf_feedback_contains_only_fixed_selected_holdout_mode(tmp_path) -> None:
    native = _row(3, 0, 8, 0.0, True)
    native["gps_week"] = 2325
    tracker_rows = [
        {
            "epoch_index": 3,
            "shadow_fixed": 1,
            "selected_rank": 0,
            "unique_holdout_pass": 1,
            "disjoint_holdout_selected": 0,
            "causal_imu_motion_selected": 0,
            "imu_aperture_selected": 0,
            "imu_accelerated_fix": 0,
        },
        {"epoch_index": 4, "shadow_fixed": 0, "selected_rank": 0},
    ]
    path = tmp_path / "feedback.csv"
    count = write_validated_pf_feedback(
        path, tracker_rows, {3: [native]}, group_index=0
    )
    assert count == 6
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 7
    assert lines[1].startswith("2325,3.0,3,G02,G01,0,")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["selected_native_holdout_pass"] for row in rows} == {"1"}
    assert {row["disjoint_holdout_selected"] for row in rows} == {"0"}
    assert {row["causal_imu_motion_selected"] for row in rows} == {"0"}
    assert {row["schema"] for row in rows} == {"gnss_gpu_pf_fgo_feedback_v1"}


def test_tracker_emits_delayed_ffbsi_without_changing_current_fix() -> None:
    rows = {epoch: [_row(epoch, 0, 8, 0.0, True)] for epoch in range(4)}
    output = track_basin_rows(
        rows,
        fix_min_streak=2,
        ffbsi_lag_epochs=2,
        ffbsi_backward_samples=64,
        ffbsi_seed=11,
    )
    assert [row["ffbsi_valid"] for row in output] == [0, 0, 1, 1]
    assert output[2]["ffbsi_tow"] == 0.0
    assert output[3]["ffbsi_tow"] == 1.0
    assert output[2]["shadow_fixed"] == 1
