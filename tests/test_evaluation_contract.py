from __future__ import annotations

import json
from pathlib import Path

from gnss_gpu.evaluation_contract import (
    EVALUATION_INPUT_SCHEMA,
    MANDATORY_NEGATIVE_HOLDOUTS,
    build_reproducibility_manifest,
    evaluate_campaign,
    verify_locked_contract,
    verify_reproducibility_manifest,
)
from experiments.evaluate_urban_campaign import main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate() -> dict:
    return {
        "id": "candidate-safe",
        "production_input_truth": False,
        "total_epochs": 100,
        "sub50cm_epochs": 50,
        "false_fix_epochs": 0,
        "gained_epochs": 2,
        "lost_epochs": 0,
        "p50_error_m": 0.49,
        "p95_error_m": 1.2,
        "max_contiguous_failure_s": 3.0,
        "latency_p50_ms": 50.0,
        "latency_p95_ms": 80.0,
        "peak_gpu_memory_mb": 1024.0,
    }


def _holdouts() -> dict:
    return {
        spec.holdout_id: {
            "accepted": False,
            "disposition": spec.expected_disposition,
            "evidence_complete": True,
        }
        for spec in MANDATORY_NEGATIVE_HOLDOUTS
    }


def _manifest(tmp_path: Path) -> dict:
    source = tmp_path / "input.json"
    source.write_text('{"stable": true}\n', encoding="utf-8")
    return build_reproducibility_manifest(
        repo_root=REPO_ROOT,
        input_paths=[source],
        config={"selector": "test"},
        command=["evaluate", "--test"],
    )


def test_locked_contract_matches_checked_in_evidence() -> None:
    result = verify_locked_contract(REPO_ROOT)
    assert result["passed"] is True
    assert len(result["mandatory_negative_holdouts"]) == 4


def test_manifest_detects_input_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    manifest = build_reproducibility_manifest(
        repo_root=REPO_ROOT,
        input_paths=[source],
        config={"mode": "constant"},
        command=["evaluate"],
    )
    assert verify_reproducibility_manifest(manifest, REPO_ROOT)["passed"] is True

    source.write_text('{"value": 2}\n', encoding="utf-8")
    result = verify_reproducibility_manifest(manifest, REPO_ROOT)
    assert result["passed"] is False
    assert result["inputs"][0]["passed"] is False


def test_campaign_promotes_only_with_complete_safe_evidence(tmp_path: Path) -> None:
    result = evaluate_campaign(
        {
            "schema": EVALUATION_INPUT_SCHEMA,
            "candidate": _candidate(),
            "holdouts": _holdouts(),
            "failure_counts": {"unsafe_acceptance": 0},
            "reproducibility_manifest": _manifest(tmp_path),
        },
        REPO_ROOT,
    )
    assert result["promoted"] is True
    assert result["metrics"]["sub50cm_rate"] == 0.5


def test_campaign_fails_closed_when_a_holdout_is_missing(tmp_path: Path) -> None:
    holdouts = _holdouts()
    holdouts.pop("tokyo_wp168")
    result = evaluate_campaign(
        {
            "schema": EVALUATION_INPUT_SCHEMA,
            "candidate": _candidate(),
            "holdouts": holdouts,
            "reproducibility_manifest": _manifest(tmp_path),
        },
        REPO_ROOT,
    )
    assert result["promoted"] is False
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert "holdout:tokyo_wp168" in failed


def test_campaign_rejects_truth_or_zero_gain(tmp_path: Path) -> None:
    candidate = _candidate()
    candidate["production_input_truth"] = True
    candidate["gained_epochs"] = 0
    result = evaluate_campaign(
        {
            "schema": EVALUATION_INPUT_SCHEMA,
            "candidate": candidate,
            "holdouts": _holdouts(),
            "reproducibility_manifest": _manifest(tmp_path),
        },
        REPO_ROOT,
    )
    assert result["promoted"] is False
    failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
    assert {"truth_free_production_input", "positive_gain"} <= failed


def test_split_registry_has_disjoint_roles_and_declared_holdouts() -> None:
    path = REPO_ROOT / "configs/evaluation/urban_campaign_splits_v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for campaign in payload["campaigns"]:
        roles = [
            set(campaign["development"]),
            set(campaign["validation"]),
            set(campaign["final_holdout"]),
        ]
        assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    assert set(payload["mandatory_negative_holdouts"]) == {
        spec.holdout_id for spec in MANDATORY_NEGATIVE_HOLDOUTS
    }


def test_common_cli_writes_a_machine_readable_promotion_result(
    tmp_path: Path,
    capsys,
) -> None:
    input_path = tmp_path / "campaign.json"
    output_path = tmp_path / "result.json"
    input_path.write_text(
        json.dumps(
            {
                "schema": EVALUATION_INPUT_SCHEMA,
                "candidate": _candidate(),
                "holdouts": _holdouts(),
                "manifest_config": {"selector": "candidate-safe"},
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--repo-root",
                str(REPO_ROOT),
            ]
        )
        == 0
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["promoted"] is True
    assert result["manifest_verification"]["passed"] is True
    assert "gnss_gpu_campaign_evaluation_result_v1" in capsys.readouterr().out
