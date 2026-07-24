from __future__ import annotations

from copy import deepcopy

import pytest

from experiments.select_wp44_anchor_boundary_identity_profile import (
    select_boundary_identity_profile,
)


def _documents() -> tuple[dict, dict]:
    identity = {
        "seed_id": 0,
        "seed_offset_ecef_m": [0.0, 0.0, 0.0],
        "offset_ecef_m": [0.1, 0.0, 0.0],
        "block_offsets_ecef_m": [[0.1, 0.0, 0.0]] * 4,
        "integer_arcs": 10,
        "carrier_rows": 100,
        "ddpr_rows": 100,
        "carrier_rms_cycles": 0.2,
        "block_spread_m": 0.05,
        "audit_median_error_m": 999.0,
    }
    source = {
        "production_input_truth": False,
        "segment": [100, 155],
        "stride_phase_mode": "auto",
        "selected_stride_phase": 2,
        "stride_phase_diagnostics": [
            {"phase": phase, "evidence_epochs": 11 if phase == 2 else 0}
            for phase in range(5)
        ],
        "evidence_epochs": 11,
        "hypotheses": [
            identity,
            {
                **identity,
                "seed_id": 1,
                "seed_offset_ecef_m": [2.0, 0.0, 0.0],
                "offset_ecef_m": [2.0, 0.0, 0.0],
            },
        ],
    }
    anchors = {
        "production_promoted": True,
        "development_anchor_used": False,
        "static_anchor_spans": [
            {"start": 0, "end": 100, "candidate_id": 4, "reason": "accepted"}
        ],
    }
    return source, anchors


def test_accepts_small_stable_identity_without_reading_audit() -> None:
    source, anchors = _documents()
    assert (
        select_boundary_identity_profile(source, anchors)["production_promoted"] is True
    )


def test_rejects_unsafe_profile_excursion() -> None:
    source, anchors = _documents()
    source = deepcopy(source)
    source["hypotheses"][0]["block_offsets_ecef_m"][-1] = [0.25, 0.0, 0.0]
    result = select_boundary_identity_profile(source, anchors)
    assert result["production_promoted"] is False
    assert result["gates"]["profile_offset_norm"] is False


def test_requires_exact_accepted_anchor_adjacency() -> None:
    source, anchors = _documents()
    anchors["static_anchor_spans"][0]["end"] = 99
    with pytest.raises(ValueError, match="adjacent"):
        select_boundary_identity_profile(source, anchors)
