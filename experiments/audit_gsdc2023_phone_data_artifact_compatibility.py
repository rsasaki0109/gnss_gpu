#!/usr/bin/env python3
"""Audit Python compatibility coverage for MATLAB ``phone_data`` artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.audit_gsdc2023_matlab_equivalence_gate import (  # noqa: E402
    DEFAULT_EQUIVALENCE_TRIPS,
    DEFAULT_WRITER_REGRESSION_MANIFEST,
    cached_summary_mismatches,
)
from experiments.gsdc2023_audit_cli import (  # noqa: E402
    add_output_dir_arg as _add_output_dir_arg,
    resolved_output_root as _resolved_output_root,
)
from experiments.gsdc2023_audit_output import (  # noqa: E402
    print_summary_and_output_dir as _print_summary_and_output_dir,
    timestamped_output_dir as _timestamped_output_dir,
    write_summary_json as _write_summary_json,
)
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402


EXPECTED_RESIDUAL_DIAGNOSTICS_COLUMNS = 44


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"summary must contain a JSON object: {path}")
    return payload


def _gate(payload: dict[str, Any], name: str) -> dict[str, Any]:
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        return {}
    gate = gates.get(name)
    return gate if isinstance(gate, dict) else {}


def _int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(payload.get(key, default) or 0)
    except (TypeError, ValueError):
        return default


def _bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    return bool(value)


def _artifact_row(
    *,
    artifact: str,
    python_writer_available: bool,
    writer_export_checked: bool,
    matlab_equivalence_checked: bool,
    required_for_submit_ready: bool,
    status: str,
    passed: bool,
    decision: str,
    evidence: str,
    notes: str,
    export_count: int | None = None,
    row_count: int | None = None,
    column_count_min: int | None = None,
    column_count_max: int | None = None,
    mismatch_count: int | None = None,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "python_writer_available": bool(python_writer_available),
        "writer_export_checked": bool(writer_export_checked),
        "matlab_equivalence_checked": bool(matlab_equivalence_checked),
        "required_for_submit_ready": bool(required_for_submit_ready),
        "status": status,
        "passed": bool(passed),
        "decision": decision,
        "evidence": evidence,
        "notes": notes,
        "export_count": export_count,
        "row_count": row_count,
        "column_count_min": column_count_min,
        "column_count_max": column_count_max,
        "mismatch_count": mismatch_count,
    }


def _factor_counts_row(
    equivalence_summary: dict[str, Any],
    factor_count_summary: dict[str, Any] | None,
    *,
    require_writer_exports: bool,
) -> dict[str, Any]:
    gate = _gate(equivalence_summary, "raw_bridge_counts")
    gate_passed = (
        _bool(gate, "passed")
        and _int(gate, "count_delta_failure_count") == 0
        and _int(gate, "missing_bridge_count_rows") == 0
    )
    writer_checked = factor_count_summary is not None and _int(
        factor_count_summary,
        "bridge_factor_count_exports_written",
    ) > 0
    writer_passed = True
    if factor_count_summary is not None:
        writer_passed = (
            _int(factor_count_summary, "count_delta_failure_count") == 0
            and _int(factor_count_summary, "matched_abs_delta_total") == 0
            and _int(factor_count_summary, "missing_bridge_count_rows") == 0
            and _int(factor_count_summary, "bridge_factor_count_exports_written") > 0
        )
    passed = gate_passed and writer_passed and (writer_checked or not require_writer_exports)
    status = "matlab_equivalent"
    if require_writer_exports and not writer_checked:
        status = "writer_export_not_checked"
    elif not passed:
        status = "failed"
    return _artifact_row(
        artifact="phone_data_factor_counts.csv",
        python_writer_available=True,
        writer_export_checked=writer_checked,
        matlab_equivalence_checked=bool(gate),
        required_for_submit_ready=True,
        status=status,
        passed=passed,
        decision="keep_csv_sidecar_writer",
        evidence="raw_bridge_counts gate"
        + (" + factor count writer summary" if factor_count_summary is not None else ""),
        notes=(
            "GPS L1/L5 factor counts match MATLAB counts; "
            "missing MATLAB rows are allowed only when bridge rows are not missing."
        ),
        export_count=(
            _int(factor_count_summary, "bridge_factor_count_exports_written")
            if factor_count_summary is not None
            else None
        ),
        mismatch_count=_int(gate, "count_delta_failure_count") if gate else None,
    )


def _factor_mask_row(
    equivalence_summary: dict[str, Any],
    factor_mask_summary: dict[str, Any] | None,
    *,
    require_writer_exports: bool,
) -> dict[str, Any]:
    gate = _gate(equivalence_summary, "factor_mask")
    gate_passed = (
        _bool(gate, "passed")
        and _int(gate, "side_only_failure_count") == 0
        and _int(gate, "total_matlab_only") == 0
        and _int(gate, "total_bridge_only") == 0
    )
    writer_checked = False
    writer_export_count: int | None = None
    writer_failure_count = 0
    if factor_mask_summary is not None:
        writer_export_count = _int(factor_mask_summary, "bridge_factor_mask_export_count")
        if writer_export_count == 0 and factor_mask_summary.get("bridge_factor_mask_export_path"):
            writer_export_count = 1
        writer_checked = bool(writer_export_count)
        writer_failure_count = _int(factor_mask_summary, "bridge_factor_mask_export_failure_count")
    writer_passed = factor_mask_summary is None or (
        writer_checked
        and writer_failure_count == 0
        and _int(factor_mask_summary, "side_only_failure_count") == 0
        and _int(factor_mask_summary, "total_matlab_only") == 0
        and _int(factor_mask_summary, "total_bridge_only") == 0
    )
    passed = gate_passed and writer_passed and (writer_checked or not require_writer_exports)
    status = "matlab_equivalent"
    if require_writer_exports and not writer_checked:
        status = "writer_export_not_checked"
    elif not passed:
        status = "failed"
    return _artifact_row(
        artifact="phone_data_factor_mask.csv",
        python_writer_available=True,
        writer_export_checked=writer_checked,
        matlab_equivalence_checked=bool(gate),
        required_for_submit_ready=True,
        status=status,
        passed=passed,
        decision="keep_csv_sidecar_writer",
        evidence="factor_mask gate"
        + (" + factor mask writer summary" if factor_mask_summary is not None else ""),
        notes="Bridge factor mask key set is side-only zero against MATLAB.",
        export_count=writer_export_count,
        mismatch_count=_int(gate, "side_only_failure_count") if gate else None,
    )


def _residual_diagnostics_row(equivalence_summary: dict[str, Any]) -> dict[str, Any]:
    gate = _gate(equivalence_summary, "residual_diagnostics_writer")
    export_count = _int(gate, "bridge_residual_diagnostics_export_count")
    row_count = _int(gate, "bridge_residual_diagnostics_export_total_rows")
    column_min = _int(gate, "bridge_residual_diagnostics_export_column_count_min")
    column_max = _int(gate, "bridge_residual_diagnostics_export_column_count_max")
    column_mismatch = _int(gate, "bridge_residual_diagnostics_export_column_mismatch_count")
    regression_mismatch = _int(gate, "writer_regression_mismatch_count")
    passed = (
        _bool(gate, "passed")
        and _bool(gate, "bridge_residual_diagnostics_export_enabled")
        and export_count > 0
        and column_min == EXPECTED_RESIDUAL_DIAGNOSTICS_COLUMNS
        and column_max == EXPECTED_RESIDUAL_DIAGNOSTICS_COLUMNS
        and column_mismatch == 0
        and _bool(gate, "writer_regression_checked")
        and _bool(gate, "writer_regression_passed")
        and regression_mismatch == 0
    )
    return _artifact_row(
        artifact="phone_data_residual_diagnostics.csv",
        python_writer_available=True,
        writer_export_checked=bool(export_count),
        matlab_equivalence_checked=bool(gate),
        required_for_submit_ready=True,
        status="schema_and_value_equivalent_regression_locked" if passed else "failed",
        passed=passed,
        decision="keep_csv_sidecar_writer_and_regression_manifest",
        evidence="residual_diagnostics_writer gate + writer regression manifest",
        notes=(
            "Numeric value parity and 44-column schema are gated; byte differences "
            "from MATLAB CSV formatting are informational."
        ),
        export_count=export_count,
        row_count=row_count,
        column_count_min=column_min,
        column_count_max=column_max,
        mismatch_count=column_mismatch + regression_mismatch,
    )


def _phone_data_mat_row(*, require_phone_data_mat: bool) -> dict[str, Any]:
    return _artifact_row(
        artifact="phone_data.mat",
        python_writer_available=False,
        writer_export_checked=False,
        matlab_equivalence_checked=False,
        required_for_submit_ready=False,
        status="deferred" if not require_phone_data_mat else "required_but_not_generated",
        passed=not require_phone_data_mat,
        decision="defer_mat_struct_writer_until_a_downstream_matlab_consumer_requires_it",
        evidence="submit-ready flow consumes Python bridge state and CSV sidecar gates, not phone_data.mat",
        notes="The MAT struct is a legacy MATLAB container; current parity and submit gates are covered by CSV/state checks.",
    )


def phone_data_artifact_compatibility_report(
    matlab_equivalence_summary: Path,
    *,
    factor_count_summary: Path | None = None,
    factor_mask_summary: Path | None = None,
    require_csv_writer_exports: bool = False,
    require_phone_data_mat: bool = False,
    skip_cached_scope_validation: bool = False,
) -> dict[str, Any]:
    equivalence_payload = _load_json(matlab_equivalence_summary)
    assert equivalence_payload is not None
    factor_count_payload = _load_json(factor_count_summary)
    factor_mask_payload = _load_json(factor_mask_summary)

    cached_mismatches: list[str] = []
    if not skip_cached_scope_validation:
        cached_mismatches = cached_summary_mismatches(
            equivalence_payload,
            data_root=DEFAULT_ROOT,
            trips=DEFAULT_EQUIVALENCE_TRIPS,
            max_epochs=0,
            count_max_epochs=0,
            factor_multi_gnss=False,
            residual_multi_gnss=False,
            residual_observation_mask=True,
            residual_include_inactive_observations=True,
            count_multi_gnss=False,
            asset_datasets=("train",),
            quick_assets=True,
            strict_ref_height=False,
            writer_regression_manifest=DEFAULT_WRITER_REGRESSION_MANIFEST,
        )

    rows = [
        _factor_counts_row(
            equivalence_payload,
            factor_count_payload,
            require_writer_exports=require_csv_writer_exports,
        ),
        _factor_mask_row(
            equivalence_payload,
            factor_mask_payload,
            require_writer_exports=require_csv_writer_exports,
        ),
        _residual_diagnostics_row(equivalence_payload),
        _phone_data_mat_row(require_phone_data_mat=require_phone_data_mat),
    ]
    failed_rows = [row for row in rows if not bool(row["passed"])]
    passed = (
        bool(equivalence_payload.get("passed", False))
        and equivalence_payload.get("equivalence_claim") == "matlab_equivalent"
        and not cached_mismatches
        and not failed_rows
    )
    return {
        "matlab_equivalence_summary": str(Path(matlab_equivalence_summary)),
        "factor_count_summary": str(factor_count_summary) if factor_count_summary is not None else None,
        "factor_mask_summary": str(factor_mask_summary) if factor_mask_summary is not None else None,
        "equivalence_claim": equivalence_payload.get("equivalence_claim"),
        "matlab_equivalence_passed": bool(equivalence_payload.get("passed", False)),
        "cached_summary_validation_checked": not skip_cached_scope_validation,
        "cached_summary_validation_passed": not cached_mismatches,
        "cached_summary_validation_mismatch_count": int(len(cached_mismatches)),
        "cached_summary_validation_mismatches": cached_mismatches[:20],
        "require_csv_writer_exports": bool(require_csv_writer_exports),
        "require_phone_data_mat": bool(require_phone_data_mat),
        "artifact_count": int(len(rows)),
        "failed_artifact_count": int(len(failed_rows)),
        "artifacts": rows,
        "phone_data_mat_decision": "defer",
        "passed": passed,
    }


def _write_artifact_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "artifact",
        "python_writer_available",
        "writer_export_checked",
        "matlab_equivalence_checked",
        "required_for_submit_ready",
        "status",
        "passed",
        "decision",
        "evidence",
        "notes",
        "export_count",
        "row_count",
        "column_count_min",
        "column_count_max",
        "mismatch_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab-equivalence-summary", type=Path, required=True)
    parser.add_argument("--factor-count-summary", type=Path)
    parser.add_argument("--factor-mask-summary", type=Path)
    parser.add_argument(
        "--require-csv-writer-exports",
        action="store_true",
        help="fail if factor-count/factor-mask writer-export summaries are not supplied",
    )
    parser.add_argument(
        "--require-phone-data-mat",
        action="store_true",
        help="fail because the Python pipeline intentionally does not generate phone_data.mat yet",
    )
    parser.add_argument(
        "--skip-cached-scope-validation",
        action="store_true",
        help="do not validate the equivalence summary against the default full-window cached gate scope",
    )
    _add_output_dir_arg(parser)
    args = parser.parse_args()

    out_dir = _timestamped_output_dir(_resolved_output_root(args), "gsdc2023_phone_data_artifact_compatibility")
    payload = phone_data_artifact_compatibility_report(
        args.matlab_equivalence_summary,
        factor_count_summary=args.factor_count_summary,
        factor_mask_summary=args.factor_mask_summary,
        require_csv_writer_exports=bool(args.require_csv_writer_exports),
        require_phone_data_mat=bool(args.require_phone_data_mat),
        skip_cached_scope_validation=bool(args.skip_cached_scope_validation),
    )
    _write_summary_json(out_dir, payload)
    _write_artifact_csv(out_dir / "artifact_compatibility.csv", payload["artifacts"])
    _print_summary_and_output_dir(payload, out_dir, label="artifact_compatibility_dir")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
