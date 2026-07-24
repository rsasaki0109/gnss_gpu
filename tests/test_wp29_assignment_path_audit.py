from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from run_wp29_tdcp_anchor_smoother import (
    _build_anchor_path_audit,
    _is_reacquisition_proposal,
    _is_snapshot_proposal,
    _resolve_tdcp_fallback,
    _resolve_path_mode,
    _resolve_static_anchor_offset,
    _robust_trusted_fix_velocity_bias,
)


def _row(
    basin: str, assignment: str, integer: int, y: float, weight: float, source: str
) -> dict[str, str]:
    return {
        "basin_id": basin,
        "assignment_id": assignment,
        "assignment_json": f'[["G01","G02",190,0,{integer}]]',
        "ecef_x": "1.0",
        "ecef_y": str(y),
        "ecef_z": "0.0",
        "log_weight": str(weight),
        "proposal_sources": source,
    }


def test_anchor_audit_exposes_assignment_transition_and_truth_oracle() -> None:
    by_epoch = {
        0: [_row("b0", "a", 10, 0.0, -1.0, "0:snapshot:0")],
        5: [
            _row("b1", "a", 10, 0.2, -2.0, "5:history:0"),
            _row("b2", "b", 11, 0.0, -1.0, "5:snapshot:0"),
        ],
    }
    truth = np.asarray([[1.0, 0.0, 0.0]] * 6)

    rows = _build_anchor_path_audit(
        by_epoch,
        {0: 0, 5: 0},
        truth,
        {(0, 5): np.zeros(3)},
        static_path_offset=None,
    )

    assert rows[1]["selected_assignment_id"] == "a"
    assert rows[1]["oracle_assignment_id"] == "b"
    assert rows[1]["matching_integers_from_previous"] == 1
    assert rows[1]["conflicting_integers_from_previous"] == 0
    assert rows[1]["selected_log_weight_rank"] == 2


def test_snapshot_proposal_accepts_explicit_and_legacy_trace_tags() -> None:
    assert _is_snapshot_proposal({"proposal_sources": "20:snapshot:3"}, 20)
    assert _is_snapshot_proposal({"proposal_sources": "20:1|15:4"}, 20)
    assert not _is_snapshot_proposal({"proposal_sources": "20:10|15:1"}, 20)
    assert _is_reacquisition_proposal(
        {"proposal_sources": "20:trusted_float_line:3:7"}, 20
    )


def test_auto_path_mode_is_selected_from_evidence_provenance() -> None:
    assert (
        _resolve_path_mode("auto", "static_stop", has_external_route_seed=True)
        == "assignment-viterbi"
    )
    assert (
        _resolve_path_mode("auto", "static_stop", has_external_route_seed=False)
        == "assignment-reacquisition-greedy"
    )
    assert (
        _resolve_path_mode("auto", "trusted_fix", has_external_route_seed=False)
        == "viterbi"
    )
    assert (
        _resolve_path_mode("greedy", "static_stop", has_external_route_seed=True)
        == "greedy"
    )


def test_static_anchor_offset_auto_defers_to_external_route_geometry() -> None:
    assert _resolve_static_anchor_offset(
        False, True, "static_stop", has_external_route_seed=False
    )
    assert not _resolve_static_anchor_offset(
        False, True, "static_stop", has_external_route_seed=True
    )
    assert not _resolve_static_anchor_offset(
        False, True, "trusted_fix", has_external_route_seed=False
    )
    assert _resolve_static_anchor_offset(
        True, False, "static_stop", has_external_route_seed=True
    )


def test_trusted_fix_velocity_bias_requires_five_samples_and_rejects_outlier() -> None:
    nominal = [
        np.array([0.10, -0.20, 0.05]),
        np.array([0.11, -0.19, 0.04]),
        np.array([0.09, -0.21, 0.06]),
        np.array([0.10, -0.20, 0.05]),
    ]
    assert _robust_trusted_fix_velocity_bias(nominal) is None

    bias = _robust_trusted_fix_velocity_bias(
        [*nominal, np.array([12.0, -8.0, 4.0])]
    )

    np.testing.assert_allclose(bias, [0.10, -0.20, 0.05], atol=0.01)


def test_tdcp_fallback_auto_uses_the_validated_anchor_source() -> None:
    assert (
        _resolve_tdcp_fallback("doppler-calibrated-auto", "trusted_fix")
        == "doppler-calibrated-trusted-fix"
    )
    assert (
        _resolve_tdcp_fallback("doppler-calibrated-auto", "static_stop")
        == "doppler-calibrated-static"
    )
    assert _resolve_tdcp_fallback("zero", "static_stop") == "zero"
