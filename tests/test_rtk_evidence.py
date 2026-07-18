import pytest

from gnss_gpu.rtk_evidence import (
    EvidenceAuditError,
    EvidenceLedger,
    RTKEpochTrace,
    TrustedFixCommitPolicy,
    TrustedFixPolicyConfig,
    TrustedFixPolicyInput,
    ambiguity_assignment_id,
    replay_fix_decisions,
)


def test_evidence_ledger_accepts_complete_annealing_and_separate_targets() -> None:
    ledger = EvidenceLedger()
    ledger.record(epoch=2, target="basins", source="ddpr", observation_id="t=1", stage=0, beta=0.4, n_rows=8)
    ledger.record(epoch=2, target="basins", source="ddpr", observation_id="t=1", stage=1, beta=0.6, n_rows=8)
    ledger.record(epoch=2, target="float", source="ddpr", observation_id="t=1", beta=1.0, n_rows=8)
    audit = ledger.audit()
    assert audit.n_records == 3
    assert audit.n_updates == 2


def test_evidence_ledger_rejects_duplicate_and_invalid_beta_total() -> None:
    ledger = EvidenceLedger()
    values = dict(epoch=0, target="basins", source="ddcp", observation_id="t=0", stage=0, beta=0.5)
    ledger.record(**values)
    with pytest.raises(EvidenceAuditError, match="duplicate"):
        ledger.record(**values)
    with pytest.raises(EvidenceAuditError, match="beta totals"):
        ledger.audit()


def _input(epoch: int, assignment: str = "a", gamma: float = 0.995, **overrides):
    values = dict(
        epoch=epoch,
        assignment_id=assignment,
        gamma=gamma,
        n_ambiguities=8,
        map_float_separation_m=0.2,
        map_ddpr_separation_m=1.0,
        last_ddpr_pairs=10,
        ddpr_age_epochs=0,
    )
    values.update(overrides)
    return TrustedFixPolicyInput(**values)


def test_commit_policy_requires_streak_and_resets_on_assignment_change() -> None:
    policy = TrustedFixCommitPolicy(TrustedFixPolicyConfig(min_streak=3))
    assert not policy.evaluate(_input(0)).fixed
    assert not policy.evaluate(_input(1)).fixed
    assert policy.evaluate(_input(2)).fixed
    changed = policy.evaluate(_input(3, assignment="b"))
    assert not changed.fixed
    assert changed.fix_streak == 1


def test_commit_policy_keeps_gamma_streak_but_guard_can_reject() -> None:
    policy = TrustedFixCommitPolicy(TrustedFixPolicyConfig(min_streak=1))
    decision = policy.evaluate(_input(0, map_ddpr_separation_m=2.0))
    assert decision.gamma_eligible
    assert decision.fix_streak == 1
    assert not decision.gate.ddpr_consistent
    assert not decision.fixed


def test_truth_free_trace_replays_online_decisions() -> None:
    config = TrustedFixPolicyConfig(min_streak=2)
    online = TrustedFixCommitPolicy(config)
    traces = []
    for epoch in range(4):
        value = _input(epoch, assignment="a" if epoch < 3 else "b")
        decision = online.evaluate(value)
        traces.append(
            RTKEpochTrace(
                **value.__dict__,
                tow=float(epoch),
                ecef_x=1.0,
                ecef_y=2.0,
                ecef_z=3.0,
                gamma_eligible=decision.gamma_eligible,
                fix_streak=decision.fix_streak,
                fixed=decision.fixed,
            )
        )
    replayed = replay_fix_decisions(traces, config)
    assert [x.fixed for x in replayed] == [x.fixed for x in traces]
    assert [x.fix_streak for x in replayed] == [x.fix_streak for x in traces]
    assert "error" not in traces[0].row()


def test_assignment_identity_is_stable_and_generation_sensitive() -> None:
    assignment = (((("G01", "G02", 190293673), 4), 12),)
    assert ambiguity_assignment_id(assignment) == ambiguity_assignment_id(assignment)
    changed = (((("G01", "G02", 190293673), 5), 12),)
    assert ambiguity_assignment_id(assignment) != ambiguity_assignment_id(changed)
