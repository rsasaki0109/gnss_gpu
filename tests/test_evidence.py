from __future__ import annotations

from pathlib import Path

import pytest

from experiments.audit_phase1_holdout_detector import audit
from gnss_gpu.evidence import (
    AcceptanceDecision,
    BasinEvidence,
    EvidenceBuilder,
    EvidenceFamily,
    EvidenceSample,
    TemporalEvidenceTracker,
    UnsafeAcceptanceDetector,
    score_basin,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample(
    family: EvidenceFamily,
    epoch: int,
    residual: float = 0.1,
    scale: float = 1.0,
) -> EvidenceSample:
    return EvidenceSample(family, epoch, residual, scale)


def _strong_basin(basin_id: str, residual: float = 0.1) -> BasinEvidence:
    return BasinEvidence(
        basin_id,
        tuple(
            _sample(family, epoch, residual)
            for epoch in (10, 11)
            for family in (
                EvidenceFamily.TDCP,
                EvidenceFamily.CARRIER_CONTINUITY,
                EvidenceFamily.ROAD_HEIGHT,
            )
        ),
    )


def test_truth_and_post_audit_fields_are_forbidden() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        EvidenceSample(
            EvidenceFamily.TDCP,
            1,
            0.1,
            1.0,
            metadata={"ground_truth": "must not enter production"},
        )
    with pytest.raises(ValueError, match="forbidden"):
        EvidenceSample(
            EvidenceFamily.DOPPLER,
            1,
            0.1,
            1.0,
            metadata={"gained_epochs": 10},
        )
    with pytest.raises(ValueError, match="forbidden"):
        EvidenceSample(
            EvidenceFamily.IMU,
            1,
            0.1,
            1.0,
            metadata={"diagnostic": {"audit_error": 0.2}},
        )


def test_typed_builder_exposes_all_seven_truth_free_families() -> None:
    evidence = (
        EvidenceBuilder("candidate", generation="affine-v1")
        .tdcp(1, 0.1, 0.5, source="tdcp")
        .doppler(1, 0.2, 1.0, source="doppler")
        .imu(1, 0.1, 0.5, source="preintegration")
        .carrier_continuity(1, 0.01, 0.2, source="adr")
        .satellite_arc(1, 0.0, 2.0, source="arc")
        .road_height(1, 0.2, 1.0, source="map")
        .los_nlos(1, 0.1, 0.3, source="raycast")
        .build()
    )
    assert {sample.family for sample in evidence.samples} == set(EvidenceFamily)
    assert evidence.generation == "affine-v1"
    with pytest.raises(ValueError, match="mismatch_fraction"):
        EvidenceBuilder("bad").los_nlos(1, 1.1, 0.2)


def test_family_balancing_prevents_sample_rich_channel_from_dominating() -> None:
    carrier_samples = tuple(
        _sample(EvidenceFamily.CARRIER_CONTINUITY, epoch, residual=0.01)
        for epoch in range(100)
    )
    evidence = BasinEvidence(
        "balanced",
        carrier_samples
        + (
            _sample(EvidenceFamily.TDCP, 0, residual=2.0),
            _sample(EvidenceFamily.TDCP, 1, residual=2.0),
            _sample(EvidenceFamily.ROAD_HEIGHT, 0, residual=2.0),
            _sample(EvidenceFamily.ROAD_HEIGHT, 1, residual=2.0),
        ),
    )
    score = score_basin(evidence)
    assert score.family_sample_counts["carrier_continuity"] == 100
    assert score.score < 0.5


def test_detector_accepts_well_supported_separated_basin() -> None:
    winner = _strong_basin("winner", residual=0.1)
    runner = _strong_basin("runner", residual=1.2)
    decision = UnsafeAcceptanceDetector().decide((winner, runner))
    assert decision.accepted is True
    assert decision.selected_basin_id == "winner"
    assert decision.runner_margin is not None
    assert decision.runner_margin > 0.08


def test_detector_rejects_dominant_but_unopposed_thin_basin() -> None:
    thin = BasinEvidence(
        "thin",
        (
            _sample(EvidenceFamily.CARRIER_CONTINUITY, 10),
            _sample(EvidenceFamily.CARRIER_CONTINUITY, 11),
            _sample(EvidenceFamily.SATELLITE_ARC, 10),
            _sample(EvidenceFamily.SATELLITE_ARC, 11),
        ),
    )
    decision = UnsafeAcceptanceDetector().decide((thin,))
    assert decision.accepted is False
    assert "insufficient_independent_families" in decision.unsafe_reasons
    assert "unopposed_basin" in decision.unsafe_reasons


def test_detector_rejects_ambiguous_basin_identity() -> None:
    decision = UnsafeAcceptanceDetector().decide(
        (_strong_basin("a", 0.10), _strong_basin("b", 0.11))
    )
    assert decision.accepted is False
    assert "ambiguous_basin_identity" in decision.unsafe_reasons


def test_temporal_tracker_reports_basin_switches() -> None:
    tracker = TemporalEvidenceTracker(window_size=4)

    def decision(basin_id: str) -> AcceptanceDecision:
        result = UnsafeAcceptanceDetector().decide(
            (_strong_basin(basin_id, 0.1), _strong_basin("runner", 1.2))
        )
        assert result.winner is not None
        return result

    tracker.observe(1, decision("a"))
    tracker.observe(2, decision("a"))
    state = tracker.observe(3, decision("b"))
    assert state.observation_count == 3
    assert state.winner_switches == 1
    assert state.winner_continuity == 0.5
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.observe(3, decision("b"))


def test_historical_truth_free_audit_recovers_at_least_two_rejections() -> None:
    result = audit(REPO_ROOT)
    assert result["production_input_truth"] is False
    assert result["rejected_holdouts"] >= 2
    assert result["passed"] is True
    assert all(not item["accepted"] for item in result["results"])
