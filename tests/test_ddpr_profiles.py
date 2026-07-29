from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.ddpr_profiles import (
    ArcObservation,
    ArcScreenPolicy,
    OffsetMode,
    evidence_aware_weight,
    fit_offset_profile,
    screen_satellite_arcs,
    select_offset_profile,
)
from gnss_gpu.evidence import BasinEvidence, EvidenceBuilder, score_basin


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_affine_profile_recovers_drifting_wp163_type_block() -> None:
    epochs = np.arange(8, dtype=np.float64)
    offsets = np.column_stack((0.4 * epochs, -0.2 * epochs, 0.1 * epochs))
    offsets += np.asarray([0.01, -0.01, 0.005])
    selection = select_offset_profile(epochs, offsets)
    constant = selection.fits["constant"]
    affine = selection.fits["affine"]
    assert selection.accepted is True
    assert selection.selected is affine
    assert affine.profile is not None
    assert affine.profile.mode == OffsetMode.AFFINE
    assert affine.weighted_rms_m == pytest.approx(0.0, abs=1e-12)
    assert constant.weighted_rms_m is not None and constant.weighted_rms_m > 0.5


def test_piecewise_profile_recovers_kink_without_discontinuity() -> None:
    epochs = np.arange(9, dtype=np.float64)
    x = np.where(epochs <= 4, epochs, 8 - epochs)
    offsets = np.column_stack((x, 0.5 * x, -0.25 * x))
    selection = select_offset_profile(
        epochs,
        offsets,
        piecewise_knots=(0.0, 4.0, 8.0),
    )
    assert selection.selected is selection.fits["piecewise_linear"]
    assert selection.selected.profile is not None
    predicted = selection.selected.profile.evaluate(epochs)
    np.testing.assert_allclose(predicted, offsets, atol=1e-12)


def test_constant_normal_block_is_not_overfit() -> None:
    epochs = np.arange(10, dtype=np.float64)
    offsets = np.repeat([[1.0, -2.0, 0.5]], len(epochs), axis=0)
    offsets[::2, 0] += 0.01
    selection = select_offset_profile(
        epochs,
        offsets,
        piecewise_knots=(0.0, 4.0, 9.0),
    )
    assert selection.selected is selection.fits["constant"]
    assert selection.improvement_fraction == 0.0


def test_piecewise_fit_fails_closed_on_thin_evidence() -> None:
    fit = fit_offset_profile(
        [0.0, 1.0, 2.0],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        mode=OffsetMode.PIECEWISE_LINEAR,
        knot_epochs=(0.0, 1.0, 2.0),
    )
    assert fit.accepted is False
    assert fit.reason == "insufficient_evidence"


def test_screen_v2_splits_arcs_and_excludes_only_supported_persistent_bias() -> None:
    observations = []
    for epoch in range(6):
        observations.extend(
            [
                ArcObservation(epoch, "G01", 0.0),
                ArcObservation(epoch, "G02", 0.2),
                ArcObservation(epoch, "G03", -0.2),
                ArcObservation(epoch, "G09", 30.0),
            ]
        )
    observations.append(ArcObservation(0, "G15", 25.0))
    observations.extend(
        [
            ArcObservation(0, "E01", 0.0),
            ArcObservation(5, "E01", 0.0),
        ]
    )
    qualities = screen_satellite_arcs(observations)
    by_id = {quality.arc_id: quality for quality in qualities}
    assert by_id["G09:0"].hard_excluded is True
    assert by_id["G09:0"].quality_weight == 0.0
    assert by_id["G01:0"].hard_excluded is False
    assert by_id["G01:0"].quality_weight > 0.9
    assert by_id["G15:0"].hard_excluded is False
    assert by_id["E01:0"].end_epoch == 0
    assert by_id["E01:1"].start_epoch == 5


def test_equal_cluster_tie_is_ambiguous_not_arbitrarily_excluded() -> None:
    observations = [
        ArcObservation(0, "G01", 0.0),
        ArcObservation(0, "G02", 0.1),
        ArcObservation(0, "G03", 20.0),
        ArcObservation(0, "G04", 20.1),
    ]
    qualities = screen_satellite_arcs(
        observations,
        ArcScreenPolicy(minimum_hard_exclusion_epochs=1),
    )
    assert all(not quality.hard_excluded for quality in qualities)
    assert all(quality.ambiguous_fraction == 1.0 for quality in qualities)
    with pytest.raises(ValueError, match="duplicate arc observation"):
        screen_satellite_arcs(
            [
                ArcObservation(0, "G01", 0.0),
                ArcObservation(0, "G01", 0.1),
            ]
        )


def test_evidence_aware_weight_penalizes_weak_family() -> None:
    strong_evidence: BasinEvidence = (
        EvidenceBuilder("strong")
        .tdcp(1, 0.1, 1.0)
        .tdcp(2, 0.1, 1.0)
        .carrier_continuity(1, 0.1, 1.0)
        .carrier_continuity(2, 0.1, 1.0)
        .road_height(1, 0.1, 1.0)
        .road_height(2, 0.1, 1.0)
        .build()
    )
    weak_evidence: BasinEvidence = (
        EvidenceBuilder("weak")
        .tdcp(1, 3.0, 1.0)
        .tdcp(2, 3.0, 1.0)
        .carrier_continuity(1, 0.1, 1.0)
        .carrier_continuity(2, 0.1, 1.0)
        .road_height(1, 0.1, 1.0)
        .road_height(2, 0.1, 1.0)
        .build()
    )
    strong = evidence_aware_weight(
        10.0,
        arc_quality_weight=0.9,
        basin_score=score_basin(strong_evidence),
    )
    weak = evidence_aware_weight(
        10.0,
        arc_quality_weight=0.9,
        basin_score=score_basin(weak_evidence),
    )
    assert strong > weak
    assert 0 < weak < 10


def test_checked_in_structural_audit_records_recovery_and_non_degradation() -> None:
    payload = json.loads(
        (REPO_ROOT / "internal_docs/phase2_structural_audit_2026_07_29.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["production_input_truth"] is False
    assert payload["wp163"]["recovered_reference_ranks"] >= 2
    assert payload["wp164"]["false_passing_hypotheses"] == 0
    assert payload["wp163"]["screen_v2"]["retained_sparse_satellite"]["sat"] == "G15"
    assert payload["passed"] is True
