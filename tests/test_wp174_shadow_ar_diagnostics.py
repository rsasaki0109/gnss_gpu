from __future__ import annotations

from pathlib import Path

from experiments.analyze_wp174_shadow_ar import analyze, trace_declaration_gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate(*, ratio: float = 3.0, satellites: int = 8) -> dict[str, float | int]:
    return {
        "ratio": ratio,
        "num_satellites": satellites,
        "prefit_residual_rms_m": 1.0,
        "update_nis_per_observation": 1.0,
    }


def test_trace_matches_wp173_causal_streak() -> None:
    candidates = {
        1.0: _candidate(),
        1.2: _candidate(),
        1.4: _candidate(),
        1.6: _candidate(),
        1.8: _candidate(),
        2.0: _candidate(),
    }
    trace = trace_declaration_gate(
        candidates,
        minimum_ratio=3.0,
        minimum_satellites=6,
        minimum_contiguous_epochs=5,
        maximum_epoch_gap_s=0.21,
    )

    assert [tow for tow, state in trace.items() if state["declared"]] == [1.8, 2.0]
    assert trace[1.6]["reason"] == "streak_warmup_or_gap"
    assert trace[1.8]["streak"] == 5


def test_trace_reports_exclusive_fail_closed_reasons() -> None:
    candidates = {
        1.0: _candidate(ratio=2.9),
        1.2: _candidate(satellites=5),
        1.4: _candidate(ratio=2.9, satellites=5),
        2.0: _candidate(),
    }
    trace = trace_declaration_gate(
        candidates,
        minimum_ratio=3.0,
        minimum_satellites=6,
        minimum_contiguous_epochs=1,
        maximum_epoch_gap_s=0.21,
    )

    assert trace[1.0]["reason"] == "ratio"
    assert trace[1.2]["reason"] == "satellites"
    assert trace[1.4]["reason"] == "ratio_and_satellites"
    assert trace[2.0]["reason"] == "declared_fix"
    assert trace[2.0]["contiguous"] is False


def test_empty_trace_is_fail_closed() -> None:
    assert (
        trace_declaration_gate(
            {},
            minimum_ratio=3.0,
            minimum_satellites=6,
            minimum_contiguous_epochs=5,
            maximum_epoch_gap_s=0.21,
        )
        == {}
    )


def test_locked_tokyo_shadow_audit_reproduces_wp173_without_mutation() -> None:
    rows, summary = analyze(
        REPO_ROOT
        / "data/tokyo_run1_wp172_pf_seeded_rtk_consensus_trajectory.csv",
        REPO_ROOT / "dist/tokyo-supply/wp160_seeded_demo5.pos",
        REPO_ROOT
        / "experiments/results/libgnss_rtk_pos_v5/tokyo_run1_full.pos",
        REPO_ROOT / "configs/evaluation/wp174_shadow_ar_diagnostics.json",
    )

    assert len(rows) == 11_924
    assert summary["gate_funnel"] == {
        "full_denominator_epochs": 11_924,
        "consensus_candidate_epochs": 3_298,
        "ratio_pass_epochs": 2_680,
        "satellites_pass_epochs": 3_298,
        "eligible_epochs": 2_680,
        "locked_declared_fix_epochs": 1_296,
    }
    assert summary["locked_counterfactual_matches_production"] is True
    assert summary["positions_modified"] is False
    assert summary["fix_declarations_modified"] is False
