from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnss_gpu.ambiguity_basin_pf import AmbiguityBasinParticleFilter, BasinKalmanState
from gnss_gpu.evidence import (
    BasinEvidence,
    EvidenceBuilder,
    UnsafeAcceptanceDetector,
)
from gnss_gpu.multihypothesis_navigation import (
    InertialBiasState,
    MapProposal,
    MultiHypothesisNavigationController,
    NavigationPhase,
    PreintegratedNavigationDelta,
    RecoveryPolicy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _state(x: float) -> BasinKalmanState:
    return BasinKalmanState.from_position(
        np.array([x, 0.0, 0.0]),
        np.eye(3),
        velocity_ecef=np.array([1.0, 0.0, 0.0]),
        velocity_sigma_mps=0.5,
    )


def _pf(*positions: float, max_basins: int = 16) -> AmbiguityBasinParticleFilter:
    pf = AmbiguityBasinParticleFilter(
        max_basins=max_basins,
        min_fixed_ambiguities=0,
        fix_gamma_threshold=0.7,
        fix_min_streak=1,
        dedup_position_radius_m=0.25,
        diversity_reserve_fraction=0.25,
        diversity_radius_m=2.0,
    )
    pf.spawn([{} for _ in positions], [_state(x) for x in positions])
    return pf


def _accepted_evidence():
    def basin(basin_id: str, residual: float) -> BasinEvidence:
        return (
            EvidenceBuilder(basin_id)
            .tdcp(1, residual, 1.0)
            .tdcp(2, residual, 1.0)
            .carrier_continuity(1, residual, 1.0)
            .carrier_continuity(2, residual, 1.0)
            .road_height(1, residual, 1.0)
            .road_height(2, residual, 1.0)
            .build()
        )

    decision = UnsafeAcceptanceDetector().decide((basin("winner", 0.1), basin("runner", 1.2)))
    assert decision.accepted
    return decision


def test_bias_covariance_propagates_through_imu_prediction() -> None:
    pf = _pf(0.0)
    bias = InertialBiasState(
        accel_bias_mps2=np.array([0.1, 0.0, 0.0]),
        gyro_bias_radps=np.zeros(3),
        covariance=np.eye(6) * 0.04,
    )
    controller = MultiHypothesisNavigationController(pf, bias=bias)
    jacobian = np.zeros((6, 6))
    jacobian[0, 0] = 0.5
    jacobian[3, 0] = 1.0
    delta = PreintegratedNavigationDelta(
        dt=1.0,
        cv_position_correction_ecef_m=np.array([0.2, 0.0, 0.0]),
        delta_velocity_ecef_mps=np.array([0.3, 0.0, 0.0]),
        covariance=np.eye(6) * 0.01,
        bias_jacobian=jacobian,
        sample_count=100,
    )
    before_covariance = pf.basins[0].conditional.covariance.copy()
    controller.predict_imu(delta, fallback_dt=1.0)
    state = pf.basins[0].conditional
    assert state.mean[0] == 1.15
    assert state.mean[3] == 1.2
    assert np.trace(state.covariance) > np.trace(before_covariance)
    assert controller.bias.covariance[0, 0] > 0.04


def test_map_proposals_preserve_distinct_road_hypotheses() -> None:
    controller = MultiHypothesisNavigationController(_pf(0.0, 8.0))
    count = controller.propose_from_map(
        [
            MapProposal("left", np.array([0.0, 1.0, 0.0]), np.eye(3) * 0.25),
            MapProposal("right", np.array([8.0, -1.0, 0.0]), np.eye(3) * 0.25),
        ]
    )
    assert count >= 2
    positions = np.asarray([basin.conditional.mean[:3] for basin in controller.pf.basins])
    assert np.ptp(positions[:, 0]) > 5.0
    assert len(controller.pf.basins) >= 2


def test_outage_invalidates_fix_and_requires_safe_reacquisition_streak() -> None:
    controller = MultiHypothesisNavigationController(
        _pf(0.0, 5.0),
        policy=RecoveryPolicy(required_reacquisition_streak=3),
    )
    outage = controller.observe_gnss(available=False)
    assert outage.phase == NavigationPhase.COASTING
    assert outage.safe_to_emit_fix is False

    accepted = _accepted_evidence()
    first = controller.observe_gnss(available=True, evidence_decision=accepted)
    second = controller.observe_gnss(available=True, evidence_decision=accepted)
    third = controller.observe_gnss(available=True, evidence_decision=accepted)
    assert first.phase == NavigationPhase.REACQUIRING
    assert second.safe_to_emit_fix is False
    assert third.phase == NavigationPhase.TRACKING
    assert third.recovery_epochs == 3


def test_unsafe_reacquisition_resets_streak() -> None:
    controller = MultiHypothesisNavigationController(_pf(0.0, 5.0))
    controller.observe_gnss(available=False)
    accepted = _accepted_evidence()
    controller.observe_gnss(available=True, evidence_decision=accepted)
    rejected = UnsafeAcceptanceDetector().decide(())
    status = controller.observe_gnss(available=True, evidence_decision=rejected)
    assert status.phase == NavigationPhase.REACQUIRING
    assert status.reacquisition_streak == 0


def test_premature_collapse_is_reported_during_outage() -> None:
    controller = MultiHypothesisNavigationController(_pf(0.0))
    status = controller.observe_gnss(available=False)
    assert status.premature_collapse is True
    assert status.safe_to_emit_fix is False


def test_status_reads_do_not_advance_pf_fix_streak() -> None:
    controller = MultiHypothesisNavigationController(_pf(0.0, 5.0))
    before = controller.pf.posterior_snapshot().fix_streak
    controller.status()
    controller.status()
    assert controller.pf.posterior_snapshot().fix_streak == before


def test_checked_in_outage_audit_records_safe_fast_recovery() -> None:
    payload = json.loads(
        (REPO_ROOT / "internal_docs/phase3_outage_recovery_audit_2026_07_29.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["production_input_truth"] is False
    assert payload["outage_fix_suppressed"] is True
    assert payload["retained_hypotheses"] >= 2
    assert payload["recovery_epochs"] <= 3
    assert payload["multihypothesis_error_m"] < payload["legacy_greedy_error_m"]
    assert payload["passed"] is True
