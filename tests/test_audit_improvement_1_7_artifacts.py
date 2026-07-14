from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/audit_improvement_1_7_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("audit_improvement_1_7_artifacts", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_main_writes_requested_json_snapshot(tmp_path: Path, monkeypatch):
    payload = {"complete": True, "checks": []}
    monkeypatch.setattr(MODULE, "audit", lambda: payload)
    output = tmp_path / "nested" / "audit.json"

    assert MODULE.main(["--json-out", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_source_contract_check_fails_closed_on_missing_snippet(tmp_path: Path, monkeypatch):
    path = tmp_path / "implementation.py"
    path.write_text("def wrong_name():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        MODULE, "SOURCE_CONTRACTS", {"implementation.py": ("def required_name",)}
    )

    check = MODULE._source_contract_check(tmp_path)

    assert not check["complete"]
    assert check["missing_files"] == []
    assert check["missing_snippets"] == [
        ("implementation.py", "def required_name")
    ]


def test_source_contract_rejects_experimental_production_flag(tmp_path: Path, monkeypatch):
    script = tmp_path / "experiments/scripts_run_phase71_osmroad_production.sh"
    script.parent.mkdir(parents=True)
    script.write_text("run --tight-dd-carrier-experimental\n", encoding="utf-8")
    monkeypatch.setattr(MODULE, "SOURCE_CONTRACTS", {})
    monkeypatch.setattr(
        MODULE, "FORBIDDEN_PRODUCTION_FLAGS", {"--tight-dd-carrier-experimental"}
    )

    check = MODULE._source_contract_check(tmp_path)

    assert not check["complete"]
    assert check["forbidden_production_flags"] == [
        "--tight-dd-carrier-experimental"
    ]


def test_matrix_check_fails_closed_on_missing_pair_and_field(tmp_path: Path):
    path = tmp_path / "matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "variant", "coverage"])
        writer.writeheader()
        writer.writerow({"scope": "a", "variant": "base", "coverage": 1})
    check = MODULE._matrix_check(
        path,
        scope_field="scope",
        expected_scopes={"a", "b"},
        variant_field="variant",
        expected_variants={"base", "new"},
        required_fields={"coverage", "p95"},
    )
    assert not check["complete"]
    assert ("a", "new") in check["missing_pairs"]
    assert ("b", "base") in check["missing_pairs"]
    assert check["missing_fields"] == ["p95"]


def test_matrix_check_rejects_unexpected_and_duplicate_pairs(tmp_path: Path):
    path = tmp_path / "matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "variant", "coverage"])
        writer.writeheader()
        writer.writerow({"scope": "a", "variant": "base", "coverage": 1})
        writer.writerow({"scope": "a", "variant": "base", "coverage": 1})
        writer.writerow({"scope": "extra", "variant": "base", "coverage": 1})

    check = MODULE._matrix_check(
        path,
        scope_field="scope",
        expected_scopes={"a"},
        variant_field="variant",
        expected_variants={"base"},
        required_fields={"coverage"},
    )

    assert not check["complete"]
    assert check["duplicate_pairs"] == [("a", "base")]
    assert check["unexpected_pairs"] == [("extra", "base")]


def test_finite_check_rejects_nan_but_allows_explicit_abstention_blank():
    rows = [
        {"scope": "ok", "variant": "new", "p95": ""},
        {"scope": "bad", "variant": "new", "p95": "nan"},
    ]
    check = {"complete": True}

    MODULE._apply_finite_check(
        check,
        rows,
        scope_field="scope",
        variant_field="variant",
        numeric_fields={"p95"},
    )

    assert not check["complete"]
    assert check["invalid_values"] == [("bad", "new", "p95", "nan")]


def test_full_run_score_check_fails_closed_on_missing_scope_and_value(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "runs.csv"
    fields = [
        "city",
        "run",
        "coverage_pct",
        "honest_ppc_pct",
        "honest_pass_m",
        "honest_total_m",
        "ms_per_epoch",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "city": "tokyo",
                "run": "run1",
                "coverage_pct": "100",
                "honest_ppc_pct": "1.5",
                "honest_pass_m": "",
                "honest_total_m": "100",
                "ms_per_epoch": "2",
            }
        )
    monkeypatch.setattr(
        MODULE, "FULL_SCOPES", {"tokyo_run1_full", "tokyo_run2_full"}
    )

    check = MODULE._full_run_score_check(path)

    assert not check["complete"]
    assert check["missing_scopes"] == ["tokyo_run2_full"]
    assert check["missing_values"] == [
        ("tokyo_run1_full", "honest_pass_m")
    ]


def test_full_run_score_check_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "runs.csv"
    fields = [
        "city",
        "run",
        "coverage_pct",
        "honest_ppc_pct",
        "honest_pass_m",
        "honest_total_m",
        "ms_per_epoch",
    ]
    row = {
        "city": "tokyo",
        "run": "run1",
        "coverage_pct": "100",
        "honest_ppc_pct": "nan",
        "honest_pass_m": "1",
        "honest_total_m": "100",
        "ms_per_epoch": "2",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)
    monkeypatch.setattr(MODULE, "FULL_SCOPES", {"tokyo_run1_full"})

    check = MODULE._full_run_score_check(path)

    assert not check["complete"]
    assert check["duplicate_scopes"] == ["tokyo_run1_full"]
    assert check["invalid_values"] == [
        ("tokyo_run1_full", "honest_ppc_pct", "nan"),
        ("tokyo_run1_full", "honest_ppc_pct", "nan"),
    ]


def test_full_run_score_consistency_rejects_impossible_metrics():
    row = {
        "coverage_pct": "101",
        "honest_ppc_pct": "50",
        "honest_pass_m": "2",
        "honest_total_m": "1",
        "ms_per_epoch": "-1",
    }

    assert MODULE._full_run_score_consistency_mismatches(
        [row], ["scope"]
    ) == [
        ("scope", "coverage outside [0, 100]"),
        ("scope", "pass distance outside total distance"),
        ("scope", "runtime is negative"),
        ("scope", "score inconsistent with pass distance"),
    ]


def test_recurrence_full_check_rejects_wrong_mode_and_nonfinite_value(
    tmp_path: Path, monkeypatch
):
    path = tmp_path / "recurrence.csv"
    fields = [
        "city",
        "run",
        "requested_epochs",
        "evaluated_epochs",
        "coverage",
        "honest_ppc_score_pct",
        "selected_p50_m",
        "selected_p95_m",
        "selected_p99_m",
        "recurrence_abstained_epochs",
        "recurrence_acceptance_rate",
        "runtime_s",
        "recurrence_mode",
        "recurrence_min_selected_probability",
        "recurrence_max_source_error_m",
        "recurrence_allow_boundary",
        "evaluation_role",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "city": "tokyo",
                "run": "run1",
                "requested_epochs": "10",
                "evaluated_epochs": "10",
                "coverage": "1",
                "honest_ppc_score_pct": "0",
                "selected_p50_m": "1",
                "selected_p95_m": "inf",
                "selected_p99_m": "3",
                "recurrence_abstained_epochs": "5",
                "recurrence_acceptance_rate": "0.5",
                "runtime_s": "10",
                "recurrence_mode": "safe_gated",
                "recurrence_min_selected_probability": "0.05",
                "recurrence_max_source_error_m": "20.0",
                "recurrence_allow_boundary": "false",
                "evaluation_role": "holdout",
            }
        )
    monkeypatch.setattr(MODULE, "FULL_SCOPES", {"tokyo_run1_full"})

    check = MODULE._recurrence_full_check(
        path,
        expected_mode="raw_counterfactual",
        expected_min_probability=0.0,
    )

    assert not check["complete"]
    assert check["mode_mismatches"] == [("tokyo_run1_full", "safe_gated")]
    assert check["policy_mismatches"] == [
        ("tokyo_run1_full", "0.05"),
        ("tokyo_run1_full", "20.0"),
        ("tokyo_run1_full", "false"),
    ]
    assert check["role_mismatches"] == [
        ("tokyo_run1_full", "holdout", "development")
    ]
    assert check["invalid_values"] == [
        ("tokyo_run1_full", "selected_p95_m", "inf")
    ]


def test_recurrence_consistency_rejects_conflicting_coverage_and_abstention():
    rows = [
        {
            "requested_epochs": "100",
            "evaluated_epochs": "80",
            "recurrence_abstained_epochs": "20",
            "coverage": "0.9",
            "recurrence_acceptance_rate": "0.5",
        }
    ]

    assert MODULE._recurrence_consistency_mismatches(rows, ["scope"]) == [
        ("scope", "acceptance inconsistent with abstention"),
        ("scope", "coverage inconsistent with epoch counts"),
    ]


def test_tight_summary_consistency_rejects_count_and_baseline_leakage():
    common = {
        "scope_id": "scope",
        "requested_epochs": "10",
        "emitted_epochs": "8",
        "coverage": "0.8",
        "tight_dd_epochs": "5",
        "tight_dd_accepted": "2",
        "tight_dd_rejected": "2",
        "tight_dd_rows": "20",
        "carrier_to_code_fallbacks": "0",
        "partial_ar_epochs": "0",
        "fixed_ambiguities": "0",
        "tight_dd_soft_resets": "0",
    }
    rows = [
        {**common, "variant": "tight_dd_imu"},
        {
            **common,
            "variant": "baseline",
            "tight_dd_epochs": "1",
            "tight_dd_accepted": "0",
            "tight_dd_rejected": "0",
        },
    ]

    assert MODULE._tight_summary_consistency_mismatches(
        rows, scope_field="scope_id"
    ) == [
        ("scope", "baseline", "baseline has tight-DD diagnostics"),
        (
            "scope",
            "tight_dd_imu",
            "accepted plus rejected differs from epochs",
        ),
    ]


def test_tc_consistency_rejects_bad_coverage_and_variant_diagnostics():
    row = {
        "scope_id": "scope",
        "variant": "baseline",
        "requested_epochs": "10",
        "output_epochs": "8",
        "evaluated_epochs": "8",
        "coverage": "0.9",
        "runtime_s": "0.8",
        "runtime_ms_per_output_epoch": "100",
        "pass_0_5m": "0.5",
        "pass_1m": "0.6",
        "pass_3m": "0.7",
        "n_wcp_factors": "1",
        "n_switchable_pseudorange": "2",
        "n_switched_pseudorange": "3",
        "n_switch_integrity_abstained_epochs": "0",
        "n_switch_integrity_abstained_rows": "0",
        "n_switch_shadow_epochs": "0",
    }

    assert MODULE._tc_consistency_mismatches(
        [row], scope_field="scope_id"
    ) == [
        ("scope", "baseline", "coverage inconsistent with epoch counts"),
        ("scope", "baseline", "non-WCP variant has WCP factors"),
        ("scope", "baseline", "non-switch variant has switch diagnostics"),
        ("scope", "baseline", "switched rows exceed switchable rows"),
    ]


def test_tc_phase_init_protocol_pins_only_nagoya_run3_exception():
    rows = [
        {"scope_id": "tokyo_run1_full", "phase_init_static_fixes": "5"},
        {"scope_id": "nagoya_run3_full", "phase_init_static_fixes": "4"},
    ]
    assert not MODULE._tc_phase_init_protocol_mismatches(
        rows, scope_field="scope_id"
    )

    rows[0]["phase_init_static_fixes"] = "4"
    assert MODULE._tc_phase_init_protocol_mismatches(
        rows, scope_field="scope_id"
    ) == [("tokyo_run1_full", "4", "5")]


def test_decision_check_rejects_pending_wrong_phase71_and_missing_evidence(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(MODULE, "RESULTS", tmp_path)
    path = tmp_path / "decisions.csv"
    fields = sorted(MODULE.DECISION_FIELDS)
    row = {field: "value" for field in fields}
    row.update(
        {
            "item_id": "1",
            "evaluation_status": "pending",
            "production_decision": "pending",
            "canonical_phase71_pct": "0",
            "best_honest_ppc_pct": "0",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    check = MODULE._decision_check(path)

    assert not check["complete"]
    assert check["pending_items"] == ["1"]
    assert check["invalid_phase71"] == [("1", "0")]
    assert check["missing_evidence"] == [("1", "value")]
    assert check["missing_ids"] == ["2", "3", "4", "5", "6", "7"]


def test_decision_check_requires_integrated_configuration(tmp_path: Path):
    path = tmp_path / "decisions.csv"
    fields = sorted(MODULE.DECISION_FIELDS - {"integrated_configuration"})
    row = {field: "value" for field in fields}
    row.update(
        {
            "item_id": "1",
            "evaluation_status": "complete",
            "production_decision": "reject",
            "canonical_phase71_pct": "86.205492",
            "best_honest_ppc_pct": "0",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    check = MODULE._decision_check(path)

    assert not check["complete"]
    assert check["missing_fields"] == ["integrated_configuration"]
    assert check["missing_values"] == [("1", "integrated_configuration")]


def test_decision_check_rejects_nonfinite_best_score(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(MODULE, "RESULTS", tmp_path)
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("ok\n", encoding="utf-8")
    path = tmp_path / "decisions.csv"
    fields = sorted(MODULE.DECISION_FIELDS)
    row = {field: "value" for field in fields}
    row.update(
        {
            "item_id": "1",
            "evaluation_status": "complete",
            "production_decision": "reject",
            "canonical_phase71_pct": "86.205492",
            "best_honest_ppc_pct": "nan",
            "evidence_artifact": evidence.name,
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    check = MODULE._decision_check(path)

    assert not check["complete"]
    assert check["invalid_best_scores"] == [("1", "nan")]
    assert check["missing_evidence"] == []


def test_tc_shadow_identity_rejects_position_hash_or_metric_change():
    common = {field: "same" for field in MODULE.TC_EMISSION_IDENTITY_FIELDS}
    rows = [
        {**common, "scope_id": "scope", "variant": "baseline"},
        {
            **common,
            "scope_id": "scope",
            "variant": "switch",
            "position_sha256": "different",
            "error_p95_m": "different",
        },
        {**common, "scope_id": "scope", "variant": "wcp"},
        {**common, "scope_id": "scope", "variant": "wcp_switch"},
    ]
    check = {"complete": True}

    MODULE._apply_tc_shadow_identity(check, rows, scope_field="scope_id")

    assert not check["complete"]
    assert check["shadow_identity_mismatches"] == [
        ("scope", "baseline", "switch", "error_p95_m", "same != different"),
        ("scope", "baseline", "switch", "position_sha256", "same != different"),
    ]
