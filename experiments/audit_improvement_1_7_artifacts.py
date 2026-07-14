#!/usr/bin/env python3
"""Fail-closed artifact audit for improvement items 1--7."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments/results"
FULL_SCOPES = {
    f"{city}_{run}_full"
    for city in ("tokyo", "nagoya")
    for run in ("run1", "run2", "run3")
}
TC_VARIANTS = {"baseline", "wcp", "switch", "wcp_switch"}
REQUIRED_METRICS = {
    "coverage",
    "honest_ppc_score_pct",
    "error_p50_m",
    "error_p95_m",
    "error_p99_m",
    "runtime_s",
}
TC_DIAGNOSTICS = {
    "n_wcp_factors",
    "n_switchable_pseudorange",
    "n_switched_pseudorange",
    "n_switch_integrity_abstained_epochs",
    "n_switch_integrity_abstained_rows",
    "n_switch_shadow_epochs",
}
TC_COUNT_METRICS = {
    "requested_epochs",
    "output_epochs",
    "evaluated_epochs",
    "pass_0_5m",
    "pass_1m",
    "pass_3m",
    "runtime_ms_per_output_epoch",
    "warmup_epochs",
}
TC_EMISSION_IDENTITY_FIELDS = {
    "run_status",
    "failure_reason",
    "requested_epochs",
    "output_epochs",
    "evaluated_epochs",
    "coverage",
    "pass_0_5m",
    "pass_1m",
    "pass_3m",
    "error_p50_m",
    "error_p95_m",
    "error_p99_m",
    "honest_ppc_score_pct",
    "pass_distance_m",
    "total_distance_m",
    "warmup_epochs",
    "n_wcp_factors",
    "position_sha256",
}
DECISION_FIELDS = {
    "item_id",
    "method",
    "evaluation_status",
    "production_decision",
    "canonical_phase71_pct",
    "best_honest_ppc_pct",
    "comparison_scope",
    "evidence_artifact",
    "integrated_configuration",
    "rationale",
}

SOURCE_CONTRACTS = {
    "CMakeLists.txt": (
        "set(CMAKE_CXX_STANDARD 20)",
        "set(CMAKE_CUDA_STANDARD 20)",
        "set(CMAKE_CXX_EXTENSIONS OFF)",
        "set(CMAKE_CUDA_EXTENSIONS OFF)",
    ),
    "third_party/gnssplusplus/CMakeLists.txt": (
        "set(GNSSPP_CXX_STANDARD 20 CACHE STRING",
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
    ),
    "python/gnss_gpu/particle_modes.py": ("def extract_particle_modes",),
    "python/gnss_gpu/particle_fixed_lag.py": ("class FixedLagParticleSmoother",),
    "python/gnss_gpu/particle_ffbsi.py": ("def ffbsi_smooth_sample",),
    "third_party/gnssplusplus/include/libgnss++/fusion/dd_imu_bridge.hpp": (
        "class DDIMUBridge",
    ),
    "third_party/gnssplusplus/src/fusion/dd_imu_bridge.cpp": (
        "DDIMUBridge::",
    ),
    "python/gnss_gpu/wcp_factor.py": ("def left_nullspace_project", "def single_arc_wcp"),
    "python/gnss_gpu/switchable_factor.py": ("def reduce_switchable_factor",),
    "python/gnss_gpu/doppler_signals.py": (
        "def carrier_frequency_hz",
        "def fit_constellation_clock_drifts",
    ),
    "python/gnss_gpu/candidate_3dma.py": ("def recurrence_vector_scores",),
    "experiments/scripts_run_phase71_osmroad_production.sh": (
        "--pf-mode-policy off",
        "--doppler-systems G,E,J",
    ),
}

FORBIDDEN_PRODUCTION_FLAGS = {
    "--enable-pf-ffbsi-smoother",
    "--tight-dd-imu",
    "--tight-dd-carrier-experimental",
    "--wcp",
    "--switchable-pseudorange",
    "--switchable-pseudorange-experimental",
    "--strategy recurrence_vector",
}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _source_contract_check(repo: Path = REPO) -> dict[str, object]:
    missing_files: list[str] = []
    missing_snippets: list[tuple[str, str]] = []
    for relative_path, snippets in SOURCE_CONTRACTS.items():
        path = repo / relative_path
        if not path.is_file():
            missing_files.append(relative_path)
            continue
        content = path.read_text(encoding="utf-8")
        missing_snippets.extend(
            (relative_path, snippet) for snippet in snippets if snippet not in content
        )
    production_path = repo / "experiments/scripts_run_phase71_osmroad_production.sh"
    production_content = (
        production_path.read_text(encoding="utf-8") if production_path.is_file() else ""
    )
    forbidden_production_flags = sorted(
        flag for flag in FORBIDDEN_PRODUCTION_FLAGS if flag in production_content
    )
    return {
        "artifact": "source_and_cpp20_contract",
        "files": len(SOURCE_CONTRACTS),
        "missing_files": missing_files,
        "missing_snippets": missing_snippets,
        "forbidden_production_flags": forbidden_production_flags,
        "complete": not missing_files
        and not missing_snippets
        and not forbidden_production_flags,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _matrix_check(
    path: Path,
    *,
    scope_field: str,
    expected_scopes: set[str],
    variant_field: str | None = None,
    expected_variants: set[str] | None = None,
    required_fields: set[str] | None = None,
    required_value_fields: set[str] | None = None,
) -> dict[str, object]:
    rows = _rows(path)
    required = required_fields or set()
    present_fields = set(rows[0]) if rows else set()
    pairs = {
        (
            row.get(scope_field, ""),
            row.get(variant_field, "") if variant_field else "",
        )
        for row in rows
    }
    pair_counts = Counter(
        (
            row.get(scope_field, ""),
            row.get(variant_field, "") if variant_field else "",
        )
        for row in rows
    )
    expected_pairs = {
        (scope, variant)
        for scope in expected_scopes
        for variant in (expected_variants if variant_field else {""})
    }
    value_fields = required if required_value_fields is None else required_value_fields
    unexpected_pairs = sorted(pairs - expected_pairs)
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count != 1)
    missing_values = sorted(
        (
            row.get(scope_field, ""),
            row.get(variant_field, "") if variant_field else "",
            field,
        )
        for row in rows
        if (
            row.get(scope_field, ""),
            row.get(variant_field, "") if variant_field else "",
        )
        in expected_pairs
        for field in value_fields
        if row.get(field, "").strip() == ""
    )
    return {
        "artifact": _display_path(path),
        "exists": path.exists(),
        "rows": len(rows),
        "missing_pairs": sorted(expected_pairs - pairs),
        "unexpected_pairs": unexpected_pairs,
        "duplicate_pairs": duplicate_pairs,
        "missing_fields": sorted(required - present_fields),
        "missing_values": missing_values,
        "complete": bool(rows)
        and not (expected_pairs - pairs)
        and not unexpected_pairs
        and not duplicate_pairs
        and not (required - present_fields)
        and not missing_values,
    }


def _apply_finite_check(
    check: dict[str, object],
    rows: list[dict[str, str]],
    *,
    scope_field: str,
    numeric_fields: set[str],
    variant_field: str | None = None,
) -> None:
    """Reject non-finite populated metrics while permitting explicit blanks."""
    invalid: list[tuple[str, str, str, str]] = []
    for row in rows:
        scope = row.get(scope_field, "")
        variant = row.get(variant_field, "") if variant_field else ""
        for field in sorted(numeric_fields):
            value = row.get(field, "").strip()
            if not value:
                continue
            try:
                valid = math.isfinite(float(value))
            except ValueError:
                valid = False
            if not valid:
                invalid.append((scope, variant, field, value))
    check["invalid_values"] = invalid
    check["complete"] = bool(check["complete"] and not invalid)


def _full_run_score_check(path: Path) -> dict[str, object]:
    rows = _rows(path)
    fields = {
        "coverage_pct",
        "honest_ppc_pct",
        "honest_pass_m",
        "honest_total_m",
        "ms_per_epoch",
    }
    row_scopes = [
        f"{row.get('city', '')}_{row.get('run', '')}_full" for row in rows
    ]
    scopes = set(row_scopes)
    scope_counts = Counter(row_scopes)
    missing_values = sorted(
        (f"{row.get('city', '')}_{row.get('run', '')}_full", field)
        for row in rows
        for field in fields
        if row.get(field, "").strip() == ""
    )
    invalid_values: list[tuple[str, str, str]] = []
    for row, scope in zip(rows, row_scopes, strict=True):
        for field in fields:
            value = row.get(field, "").strip()
            if not value:
                continue
            try:
                finite = math.isfinite(float(value))
            except ValueError:
                finite = False
            if not finite:
                invalid_values.append((scope, field, value))
    present_fields = set(rows[0]) if rows else set()
    missing_scopes = sorted(FULL_SCOPES - scopes)
    unexpected_scopes = sorted(scopes - FULL_SCOPES)
    duplicate_scopes = sorted(
        scope for scope, count in scope_counts.items() if count != 1
    )
    consistency_mismatches = _full_run_score_consistency_mismatches(
        rows, row_scopes
    )
    return {
        "artifact": _display_path(path),
        "exists": path.exists(),
        "rows": len(rows),
        "missing_scopes": missing_scopes,
        "unexpected_scopes": unexpected_scopes,
        "duplicate_scopes": duplicate_scopes,
        "missing_fields": sorted(fields - present_fields),
        "missing_values": missing_values,
        "invalid_values": invalid_values,
        "consistency_mismatches": consistency_mismatches,
        "complete": bool(rows)
        and len(rows) == len(FULL_SCOPES)
        and not missing_scopes
        and not unexpected_scopes
        and not duplicate_scopes
        and not (fields - present_fields)
        and not missing_values
        and not invalid_values
        and not consistency_mismatches,
    }


def _full_run_score_consistency_mismatches(
    rows: list[dict[str, str]], scopes: list[str]
) -> list[tuple[str, str]]:
    mismatches: list[tuple[str, str]] = []
    for row, scope in zip(rows, scopes, strict=True):
        try:
            coverage = float(row["coverage_pct"])
            score = float(row["honest_ppc_pct"])
            passed = float(row["honest_pass_m"])
            total = float(row["honest_total_m"])
            runtime = float(row["ms_per_epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0.0 <= coverage <= 100.0:
            mismatches.append((scope, "coverage outside [0, 100]"))
        if passed < 0.0 or total < 0.0 or passed > total:
            mismatches.append((scope, "pass distance outside total distance"))
        expected_score = 100.0 * passed / total if total else 0.0
        if not math.isclose(score, expected_score, rel_tol=0.0, abs_tol=1e-9):
            mismatches.append((scope, "score inconsistent with pass distance"))
        if runtime < 0.0:
            mismatches.append((scope, "runtime is negative"))
    return sorted(mismatches)


def _apply_role_check(
    check: dict[str, object],
    rows: list[dict[str, str]],
    *,
    scope_field: str,
    expected_roles: dict[str, str],
) -> None:
    mismatches = sorted(
        (row.get(scope_field, ""), row.get("evaluation_role", ""), expected_roles.get(row.get(scope_field, ""), ""))
        for row in rows
        if row.get(scope_field, "") in expected_roles
        and row.get("evaluation_role", "") != expected_roles[row[scope_field]]
    )
    check["role_mismatches"] = mismatches
    check["complete"] = bool(check["complete"] and not mismatches)


def _decision_check(path: Path) -> dict[str, object]:
    rows = _rows(path)
    present_fields = set(rows[0]) if rows else set()
    ids = [row.get("item_id", "") for row in rows]
    expected_ids = {str(item) for item in range(1, 8)}
    invalid_phase71: list[tuple[str, str]] = []
    invalid_best_scores: list[tuple[str, str]] = []
    missing_evidence: list[tuple[str, str]] = []
    for row in rows:
        value = row.get("canonical_phase71_pct", "")
        try:
            valid = math.isfinite(float(value)) and abs(float(value) - 86.205492) < 1.0e-9
        except ValueError:
            valid = False
        if not valid:
            invalid_phase71.append((row.get("item_id", ""), value))
        score = row.get("best_honest_ppc_pct", "")
        try:
            score_valid = math.isfinite(float(score)) and 0.0 <= float(score) <= 100.0
        except ValueError:
            score_valid = False
        if not score_valid:
            invalid_best_scores.append((row.get("item_id", ""), score))
        evidence = row.get("evidence_artifact", "").strip()
        if evidence and not (RESULTS / evidence).is_file():
            missing_evidence.append((row.get("item_id", ""), evidence))
    pending = sorted(
        row.get("item_id", "")
        for row in rows
        if row.get("evaluation_status", "") != "complete"
        or row.get("production_decision", "") in {"", "pending"}
    )
    missing_values = sorted(
        (row.get("item_id", ""), field)
        for row in rows
        for field in DECISION_FIELDS
        if row.get(field, "").strip() == ""
    )
    return {
        "artifact": _display_path(path),
        "exists": path.exists(),
        "rows": len(rows),
        "missing_ids": sorted(expected_ids - set(ids)),
        "unexpected_ids": sorted(set(ids) - expected_ids),
        "duplicate_ids": sorted(item for item, count in Counter(ids).items() if count != 1),
        "missing_fields": sorted(DECISION_FIELDS - present_fields),
        "missing_values": missing_values,
        "invalid_phase71": invalid_phase71,
        "invalid_best_scores": invalid_best_scores,
        "missing_evidence": missing_evidence,
        "pending_items": pending,
        "complete": len(rows) == 7
        and set(ids) == expected_ids
        and not (DECISION_FIELDS - present_fields)
        and not missing_values
        and not invalid_phase71
        and not invalid_best_scores
        and not missing_evidence
        and not pending,
    }


def _apply_tc_shadow_identity(
    check: dict[str, object], rows: list[dict[str, str]], *, scope_field: str
) -> None:
    """Require safe shadow variants to preserve their source emission exactly."""
    by_pair = {
        (row.get(scope_field, ""), row.get("variant", "")): row for row in rows
    }
    mismatches: list[tuple[str, str, str, str, str]] = []
    for scope in sorted({row.get(scope_field, "") for row in rows}):
        for source_variant, shadow_variant in (
            ("baseline", "switch"),
            ("wcp", "wcp_switch"),
        ):
            source = by_pair.get((scope, source_variant))
            shadow = by_pair.get((scope, shadow_variant))
            if source is None or shadow is None:
                continue
            for field in sorted(TC_EMISSION_IDENTITY_FIELDS):
                source_value = source.get(field, "")
                shadow_value = shadow.get(field, "")
                if source_value != shadow_value:
                    mismatches.append(
                        (scope, source_variant, shadow_variant, field, f"{source_value} != {shadow_value}")
                    )
    check["shadow_identity_mismatches"] = mismatches
    check["complete"] = bool(check["complete"] and not mismatches)


def _recurrence_full_check(
    path: Path, *, expected_mode: str, expected_min_probability: float
) -> dict[str, object]:
    rows = _rows(path)
    row_scopes = [
        f"{row.get('city', '')}_{row.get('run', '')}_full" for row in rows
    ]
    scopes = set(row_scopes)
    counts = Counter(row_scopes)
    fields = {
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
    }
    present_fields = set(rows[0]) if rows else set()
    missing_values = sorted(
        (scope, field)
        for row, scope in zip(rows, row_scopes, strict=True)
        for field in fields
        if row.get(field, "").strip() == ""
    )
    numeric_fields = fields - {
        "recurrence_mode",
        "recurrence_allow_boundary",
        "evaluation_role",
    }
    invalid_values: list[tuple[str, str, str]] = []
    for row, scope in zip(rows, row_scopes, strict=True):
        for field in numeric_fields:
            value = row.get(field, "").strip()
            if not value:
                continue
            try:
                finite = math.isfinite(float(value))
            except ValueError:
                finite = False
            if not finite:
                invalid_values.append((scope, field, value))
    mode_mismatches = sorted(
        (scope, row.get("recurrence_mode", ""))
        for row, scope in zip(rows, row_scopes, strict=True)
        if row.get("recurrence_mode", "") != expected_mode
    )
    expected_roles = {
        scope: (
            "development"
            if scope in {"tokyo_run1_full", "nagoya_run1_full"}
            else "holdout"
        )
        for scope in FULL_SCOPES
    }
    role_mismatches = sorted(
        (scope, row.get("evaluation_role", ""), expected_roles.get(scope, ""))
        for row, scope in zip(rows, row_scopes, strict=True)
        if scope in expected_roles
        and row.get("evaluation_role", "") != expected_roles[scope]
    )
    policy_mismatches: list[tuple[str, str]] = []
    expected_max_source_error = 0.0 if expected_mode == "raw_counterfactual" else 20.0
    expected_allow_boundary = expected_mode == "raw_counterfactual"
    for row, scope in zip(rows, row_scopes, strict=True):
        value = row.get("recurrence_min_selected_probability", "").strip()
        if not value:
            continue
        try:
            actual = float(value)
        except ValueError:
            continue
        if math.isfinite(actual) and actual != float(expected_min_probability):
            policy_mismatches.append((scope, value))
        max_source_value = row.get("recurrence_max_source_error_m", "").strip()
        if max_source_value:
            try:
                max_source = float(max_source_value)
            except ValueError:
                max_source = float("nan")
            if (
                math.isfinite(max_source)
                and max_source != expected_max_source_error
            ):
                policy_mismatches.append((scope, max_source_value))
        allow_boundary = row.get("recurrence_allow_boundary", "").strip().lower()
        expected_boundary_text = str(expected_allow_boundary).lower()
        if allow_boundary and allow_boundary != expected_boundary_text:
            policy_mismatches.append((scope, allow_boundary))
    policy_mismatches.sort()
    consistency_mismatches = _recurrence_consistency_mismatches(rows, row_scopes)
    missing_scopes = sorted(FULL_SCOPES - scopes)
    unexpected_scopes = sorted(scopes - FULL_SCOPES)
    duplicate_scopes = sorted(scope for scope, count in counts.items() if count != 1)
    return {
        "artifact": _display_path(path),
        "exists": path.exists(),
        "rows": len(rows),
        "expected_mode": expected_mode,
        "expected_min_probability": expected_min_probability,
        "missing_scopes": missing_scopes,
        "unexpected_scopes": unexpected_scopes,
        "duplicate_scopes": duplicate_scopes,
        "missing_fields": sorted(fields - present_fields),
        "missing_values": missing_values,
        "invalid_values": invalid_values,
        "mode_mismatches": mode_mismatches,
        "role_mismatches": role_mismatches,
        "policy_mismatches": policy_mismatches,
        "consistency_mismatches": consistency_mismatches,
        "complete": bool(rows)
        and len(rows) == len(FULL_SCOPES)
        and not missing_scopes
        and not unexpected_scopes
        and not duplicate_scopes
        and not (fields - present_fields)
        and not missing_values
        and not invalid_values
        and not mode_mismatches
        and not role_mismatches
        and not policy_mismatches
        and not consistency_mismatches,
    }


def _recurrence_consistency_mismatches(
    rows: list[dict[str, str]], scopes: list[str]
) -> list[tuple[str, str]]:
    mismatches: list[tuple[str, str]] = []
    for row, scope in zip(rows, scopes, strict=True):
        try:
            requested = float(row["requested_epochs"])
            evaluated = float(row["evaluated_epochs"])
            abstained = float(row["recurrence_abstained_epochs"])
            coverage = float(row["coverage"])
            acceptance = float(row["recurrence_acceptance_rate"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(
            not value.is_integer() for value in (requested, evaluated, abstained)
        ):
            mismatches.append((scope, "epoch counts must be integers"))
            continue
        if requested < 0 or evaluated < 0 or evaluated > requested:
            mismatches.append((scope, "evaluated epochs outside requested range"))
        if abstained < 0 or abstained > evaluated:
            mismatches.append((scope, "abstained epochs outside evaluated range"))
        expected_coverage = evaluated / requested if requested else 0.0
        expected_acceptance = (evaluated - abstained) / evaluated if evaluated else 0.0
        if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append((scope, "coverage inconsistent with epoch counts"))
        if not math.isclose(
            acceptance, expected_acceptance, rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches.append((scope, "acceptance inconsistent with abstention"))
    return sorted(mismatches)


def _tight_summary_consistency_mismatches(
    rows: list[dict[str, str]], *, scope_field: str
) -> list[tuple[str, str, str]]:
    mismatches: list[tuple[str, str, str]] = []
    diagnostic_fields = (
        "tight_dd_epochs",
        "tight_dd_accepted",
        "tight_dd_rejected",
        "tight_dd_rows",
        "carrier_to_code_fallbacks",
        "partial_ar_epochs",
        "fixed_ambiguities",
        "tight_dd_soft_resets",
    )
    for row in rows:
        scope = row.get(scope_field, "")
        variant = row.get("variant", "")
        try:
            requested = float(row["requested_epochs"])
            emitted = float(row["emitted_epochs"])
            coverage = float(row["coverage"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(
            not value.is_integer() or value < 0 for value in (requested, emitted)
        ):
            mismatches.append((scope, variant, "counts must be nonnegative integers"))
            continue
        if emitted > requested:
            mismatches.append((scope, variant, "emitted epochs exceed requested"))
        expected_coverage = emitted / requested if requested else 0.0
        if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append((scope, variant, "coverage inconsistent with epoch counts"))
        try:
            diagnostics = {field: float(row[field]) for field in diagnostic_fields}
        except (KeyError, TypeError, ValueError):
            continue
        if any(
            not value.is_integer() or value < 0 for value in diagnostics.values()
        ):
            mismatches.append((scope, variant, "counts must be nonnegative integers"))
            continue
        epochs = diagnostics["tight_dd_epochs"]
        accepted = diagnostics["tight_dd_accepted"]
        rejected = diagnostics["tight_dd_rejected"]
        if variant == "baseline":
            if any(value != 0 for value in diagnostics.values()):
                mismatches.append((scope, variant, "baseline has tight-DD diagnostics"))
        elif accepted + rejected != epochs:
            mismatches.append((scope, variant, "accepted plus rejected differs from epochs"))
        for field in (
            "carrier_to_code_fallbacks",
            "partial_ar_epochs",
            "tight_dd_soft_resets",
        ):
            if diagnostics[field] > epochs:
                mismatches.append((scope, variant, f"{field} exceeds tight-DD epochs"))
    return sorted(mismatches)


def _tc_consistency_mismatches(
    rows: list[dict[str, str]], *, scope_field: str
) -> list[tuple[str, str, str]]:
    mismatches: list[tuple[str, str, str]] = []
    switch_fields = {
        "n_switchable_pseudorange",
        "n_switched_pseudorange",
        "n_switch_integrity_abstained_epochs",
        "n_switch_integrity_abstained_rows",
        "n_switch_shadow_epochs",
    }
    for row in rows:
        scope = row.get(scope_field, "")
        variant = row.get("variant", "")
        try:
            requested = float(row["requested_epochs"])
            output = float(row["output_epochs"])
            evaluated = float(row["evaluated_epochs"])
            coverage = float(row["coverage"])
            runtime_s = float(row["runtime_s"])
            diagnostics = {
                field: float(row[field]) for field in TC_DIAGNOSTICS
            }
        except (KeyError, TypeError, ValueError):
            continue
        counts = (requested, output, evaluated, *diagnostics.values())
        if any(not value.is_integer() or value < 0 for value in counts):
            mismatches.append((scope, variant, "counts must be nonnegative integers"))
            continue
        if evaluated > output or output > requested:
            mismatches.append((scope, variant, "epoch counts are out of order"))
        expected_coverage = evaluated / requested if requested else 0.0
        if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1e-12):
            mismatches.append((scope, variant, "coverage inconsistent with epoch counts"))
        runtime_ms = row.get("runtime_ms_per_output_epoch", "").strip()
        if output and runtime_ms:
            if not math.isclose(
                float(runtime_ms),
                1000.0 * runtime_s / output,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                mismatches.append((scope, variant, "runtime normalization mismatch"))
        for field in ("pass_0_5m", "pass_1m", "pass_3m"):
            value = row.get(field, "").strip()
            if value and not 0.0 <= float(value) <= 1.0:
                mismatches.append((scope, variant, f"{field} outside [0, 1]"))
        if diagnostics["n_switched_pseudorange"] > diagnostics[
            "n_switchable_pseudorange"
        ]:
            mismatches.append((scope, variant, "switched rows exceed switchable rows"))
        if diagnostics["n_switch_integrity_abstained_rows"] > diagnostics[
            "n_switchable_pseudorange"
        ]:
            mismatches.append((scope, variant, "abstained rows exceed switchable rows"))
        if variant in {"baseline", "wcp"} and any(
            diagnostics[field] != 0 for field in switch_fields
        ):
            mismatches.append((scope, variant, "non-switch variant has switch diagnostics"))
        if variant in {"baseline", "switch"} and diagnostics["n_wcp_factors"] != 0:
            mismatches.append((scope, variant, "non-WCP variant has WCP factors"))
    return sorted(mismatches)


def _tc_phase_init_protocol_mismatches(
    rows: list[dict[str, str]], *, scope_field: str
) -> list[tuple[str, str, str]]:
    """Pin the one data-feasibility exception instead of permitting tuning."""
    mismatches: list[tuple[str, str, str]] = []
    for row in rows:
        scope = row.get(scope_field, "")
        expected = "4" if scope == "nagoya_run3_full" else "5"
        actual = row.get("phase_init_static_fixes", "")
        if actual != expected:
            mismatches.append((scope, actual, expected))
    return sorted(mismatches)


def audit(results: Path = RESULTS) -> dict[str, object]:
    manifest = _rows(REPO / "experiments/blocked_span_manifest.csv")
    blocked_scopes = {row["span_id"] for row in manifest}
    full_roles = {
        scope: ("development" if scope == "tokyo_run1_full" else "holdout")
        for scope in FULL_SCOPES
    }
    blocked_roles = {
        row["span_id"]: row.get("tcfgo_evaluation_role", row["evaluation_role"])
        for row in manifest
    }
    recurrence_blocked_roles = {
        row["span_id"]: row.get("recurrence_evaluation_role", row["evaluation_role"])
        for row in manifest
    }
    tight_roles = full_roles | blocked_roles
    checks: list[dict[str, object]] = []
    checks.append(_source_contract_check())
    checks.append(_decision_check(results / "improvement_1_7_phase71_decisions.csv"))
    holdout_scopes = {
        "all_after_development",
        *{
            f"{city}_{run}_after_development"
            for city in ("tokyo", "nagoya")
            for run in ("run1", "run2", "run3")
        },
        *blocked_scopes,
    }
    diagnostic_scopes = holdout_scopes | {
        "all",
        *{
            f"{city}_{run}"
            for city in ("tokyo", "nagoya")
            for run in ("run1", "run2", "run3")
        },
    }
    checks.append(
        _matrix_check(
            results / "pf_mode_full6_p2k_diagnostic_ablation_summary.csv",
            scope_field="scope_id",
            expected_scopes=diagnostic_scopes,
            required_fields={
                "reference_coverage",
                "error_p50_m",
                "error_p95_m",
                "error_p99_m",
                "mode_evaluated",
                "mode_abstention_rate",
                "mode_counterfactual_improved_rate",
                "ms_per_epoch",
            },
        )
    )
    for name in (
        "pf_mode_full6_p2k_diagnostic_runs.csv",
        "pf_ffbsi_full6_p2k_lag10_paths8_runs.csv",
        "rbpf_doppler_gej_full6_p2k_runs.csv",
        "rbpf_doppler_gejcr_full6_p2k_runs.csv",
    ):
        checks.append(_full_run_score_check(results / name))
    checks.append(
        _matrix_check(
            results / "pf_ffbsi_full6_p2k_lag10_paths8_ablation_summary.csv",
            scope_field="scope_id",
            expected_scopes=diagnostic_scopes,
            required_fields={
                "reference_coverage",
                "error_p50_m",
                "error_p95_m",
                "error_p99_m",
                "ffbsi_available",
                "ffbsi_applied",
                "ffbsi_abstention_rate",
                "ffbsi_error_delta_p95_m",
                "ms_per_epoch",
            },
        )
    )
    checks.append(
        _matrix_check(
            results / "rbpf_doppler_gej_vs_gejcr_comparison.csv",
            scope_field="scope_id",
            expected_scopes=diagnostic_scopes,
            required_fields={
                "reference_coverage_baseline",
                "reference_coverage_candidate",
                "error_p50_m_delta",
                "error_p95_m_delta",
                "error_p99_m_delta",
                "doppler_update_rate_candidate",
                "doppler_clock_groups_mean_candidate",
                "doppler_clock_fit_rms_p95_mps_candidate",
                "ms_per_epoch_candidate",
            },
        )
    )
    tc_full_path = results / "tcfgo_structural_full_runs_summary.csv"
    tc_full = _matrix_check(
            tc_full_path,
            scope_field="scope_id",
            expected_scopes=FULL_SCOPES,
            variant_field="variant",
            expected_variants=TC_VARIANTS,
            required_fields=REQUIRED_METRICS
            | TC_DIAGNOSTICS
            | TC_COUNT_METRICS
            | {"evaluation_role", "phase_init_static_fixes", "position_sha256"},
            required_value_fields=REQUIRED_METRICS
            | TC_DIAGNOSTICS
            | {"evaluation_role", "phase_init_static_fixes", "position_sha256"},
        )
    _apply_role_check(
        tc_full, _rows(tc_full_path), scope_field="scope_id", expected_roles=full_roles
    )
    _apply_tc_shadow_identity(tc_full, _rows(tc_full_path), scope_field="scope_id")
    _apply_finite_check(
        tc_full,
        _rows(tc_full_path),
        scope_field="scope_id",
        variant_field="variant",
        numeric_fields=REQUIRED_METRICS | TC_DIAGNOSTICS | TC_COUNT_METRICS,
    )
    tc_full["consistency_mismatches"] = _tc_consistency_mismatches(
        _rows(tc_full_path), scope_field="scope_id"
    )
    tc_full["phase_init_protocol_mismatches"] = _tc_phase_init_protocol_mismatches(
        _rows(tc_full_path), scope_field="scope_id"
    )
    tc_full["complete"] = bool(
        tc_full["complete"]
        and not tc_full["consistency_mismatches"]
        and not tc_full["phase_init_protocol_mismatches"]
    )
    checks.append(tc_full)
    tc_blocked_path = results / "tcfgo_structural_blocked_spans_summary.csv"
    tc_blocked = _matrix_check(
            tc_blocked_path,
            scope_field="span_id",
            expected_scopes=blocked_scopes,
            variant_field="variant",
            expected_variants=TC_VARIANTS,
            required_fields=REQUIRED_METRICS
            | TC_DIAGNOSTICS
            | TC_COUNT_METRICS
            | {"run_status", "failure_reason", "evaluation_role", "position_sha256"},
            required_value_fields={
                "coverage",
                "honest_ppc_score_pct",
                "runtime_s",
                "requested_epochs",
                "output_epochs",
                "evaluated_epochs",
                "run_status",
            }
            | TC_DIAGNOSTICS,
        )
    _apply_role_check(
        tc_blocked,
        _rows(tc_blocked_path),
        scope_field="span_id",
        expected_roles=blocked_roles,
    )
    _apply_tc_shadow_identity(
        tc_blocked, _rows(tc_blocked_path), scope_field="span_id"
    )
    _apply_finite_check(
        tc_blocked,
        _rows(tc_blocked_path),
        scope_field="span_id",
        variant_field="variant",
        numeric_fields=REQUIRED_METRICS
        | TC_DIAGNOSTICS
        | (TC_COUNT_METRICS - {"runtime_ms_per_output_epoch"}),
    )
    tc_blocked["consistency_mismatches"] = _tc_consistency_mismatches(
        _rows(tc_blocked_path), scope_field="span_id"
    )
    tc_blocked["complete"] = bool(
        tc_blocked["complete"] and not tc_blocked["consistency_mismatches"]
    )
    checks.append(tc_blocked)
    tight = _matrix_check(
        results / "tight_dd_imu/tight_dd_imu_ablation_comparison.csv",
        scope_field="scope_id",
        expected_scopes=FULL_SCOPES | blocked_scopes,
        required_fields={
            "evaluation_role",
            "comparison_status",
            "binary_sha256_match",
            "baseline_honest_ppc_score_pct",
            "baseline_coverage",
            "tight_coverage",
            "tight_minus_baseline_coverage",
            "baseline_pass_0_5m",
            "tight_pass_0_5m",
            "tight_minus_baseline_pass_0_5m",
            "baseline_error_p50_m",
            "tight_error_p50_m",
            "tight_minus_baseline_error_p50_m",
            "baseline_error_p95_m",
            "tight_honest_ppc_score_pct",
            "tight_minus_baseline_honest_ppc_score_pct",
            "tight_error_p95_m",
            "tight_minus_baseline_error_p95_m",
            "baseline_error_p99_m",
            "tight_error_p99_m",
            "tight_minus_baseline_error_p99_m",
            "baseline_runtime_ms_per_requested_epoch",
            "tight_runtime_ms_per_requested_epoch",
            "tight_minus_baseline_runtime_ms_per_requested_epoch",
        },
    )
    tight_rows = _rows(results / "tight_dd_imu/tight_dd_imu_ablation_comparison.csv")
    _apply_role_check(
        tight,
        tight_rows,
        scope_field="scope_id",
        expected_roles=tight_roles,
    )
    tight["unmatched_scopes"] = sorted(
        row.get("scope_id", "")
        for row in tight_rows
        if row.get("comparison_status") != "matched"
        or row.get("binary_sha256_match", "").lower() != "true"
    )
    tight["complete"] = bool(tight["complete"] and not tight["unmatched_scopes"])
    tight_metric_fields = {
        f"{prefix}_{metric}"
        for prefix in ("baseline", "tight", "tight_minus_baseline")
        for metric in (
            "honest_ppc_score_pct",
            "coverage",
            "pass_0_5m",
            "error_p50_m",
            "error_p95_m",
            "error_p99_m",
            "runtime_ms_per_requested_epoch",
        )
    }
    _apply_finite_check(
        tight,
        tight_rows,
        scope_field="scope_id",
        numeric_fields=tight_metric_fields,
    )
    checks.append(tight)
    tight_summary_path = results / "tight_dd_imu/tight_dd_imu_ablation_summary.csv"
    tight_summary = _matrix_check(
            tight_summary_path,
            scope_field="scope_id",
            expected_scopes=FULL_SCOPES | blocked_scopes,
            variant_field="variant",
            expected_variants={"baseline", "tight_dd_imu"},
            required_fields={
                "evaluation_role",
                "binary_sha256",
                "requested_epochs",
                "emitted_epochs",
                "coverage",
                "honest_ppc_score_pct",
                "pass_0_5m",
                "error_p50_m",
                "error_p95_m",
                "error_p99_m",
                "runtime_ms_per_requested_epoch",
                "tight_dd_epochs",
                "tight_dd_accepted",
                "tight_dd_rejected",
                "tight_dd_rows",
                "carrier_to_code_fallbacks",
                "partial_ar_epochs",
                "fixed_ambiguities",
                "tight_dd_soft_resets",
            },
            required_value_fields={
                "evaluation_role",
                "binary_sha256",
                "requested_epochs",
                "emitted_epochs",
                "coverage",
                "honest_ppc_score_pct",
                "pass_0_5m",
                "error_p50_m",
                "error_p95_m",
                "error_p99_m",
                "runtime_ms_per_requested_epoch",
            },
        )
    _apply_role_check(
        tight_summary,
        _rows(tight_summary_path),
        scope_field="scope_id",
        expected_roles=tight_roles,
    )
    _apply_finite_check(
        tight_summary,
        _rows(tight_summary_path),
        scope_field="scope_id",
        variant_field="variant",
        numeric_fields={
            "requested_epochs",
            "emitted_epochs",
            "coverage",
            "honest_ppc_score_pct",
            "pass_0_5m",
            "error_p50_m",
            "error_p95_m",
            "error_p99_m",
            "runtime_ms_per_requested_epoch",
            "tight_dd_epochs",
            "tight_dd_accepted",
            "tight_dd_rejected",
            "tight_dd_rows",
            "carrier_to_code_fallbacks",
            "partial_ar_epochs",
            "fixed_ambiguities",
            "tight_dd_soft_resets",
        },
    )
    tight_summary["missing_full_diagnostics"] = sorted(
        (row.get("scope_id", ""), row.get("variant", ""), field)
        for row in _rows(tight_summary_path)
        if row.get("scope_id", "") in FULL_SCOPES
        for field in (
            "tight_dd_epochs",
            "tight_dd_accepted",
            "tight_dd_rejected",
            "tight_dd_rows",
            "carrier_to_code_fallbacks",
            "partial_ar_epochs",
            "fixed_ambiguities",
            "tight_dd_soft_resets",
        )
        if row.get(field, "").strip() == ""
    )
    tight_summary["complete"] = bool(
        tight_summary["complete"] and not tight_summary["missing_full_diagnostics"]
    )
    tight_summary["consistency_mismatches"] = _tight_summary_consistency_mismatches(
        _rows(tight_summary_path), scope_field="scope_id"
    )
    tight_summary["complete"] = bool(
        tight_summary["complete"] and not tight_summary["consistency_mismatches"]
    )
    checks.append(tight_summary)
    for prefix, mode, min_probability in (
        ("candidate_3dma_recurrence_full", "safe_gated", 0.05),
        ("candidate_3dma_recurrence_raw_full", "raw_counterfactual", 0.0),
    ):
        checks.append(
            _recurrence_full_check(
                results / f"{prefix}_runs_summary.csv",
                expected_mode=mode,
                expected_min_probability=min_probability,
            )
        )
    for filename, mode in (
        ("candidate_3dma_recurrence_blocked_spans_summary.csv", "safe_gated"),
        (
            "candidate_3dma_recurrence_raw_blocked_spans_summary.csv",
            "raw_counterfactual",
        ),
    ):
        check = _matrix_check(
                results / filename,
                scope_field="span_id",
                expected_scopes=blocked_scopes,
                required_fields={
                    "evaluation_role",
                    "recurrence_mode",
                    "recurrence_min_selected_probability",
                    "recurrence_max_source_error_m",
                    "recurrence_allow_boundary",
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
                },
            )
        rows = _rows(results / filename)
        _apply_role_check(
            check,
            rows,
            scope_field="span_id",
            expected_roles=recurrence_blocked_roles,
        )
        check["mode_mismatches"] = sorted(
            (row.get("span_id", ""), row.get("recurrence_mode", ""))
            for row in rows
            if row.get("recurrence_mode", "") != mode
        )
        check["complete"] = bool(check["complete"] and not check["mode_mismatches"])
        expected_min_probability = 0.0 if mode == "raw_counterfactual" else 0.05
        expected_max_source_error = 0.0 if mode == "raw_counterfactual" else 20.0
        expected_allow_boundary = mode == "raw_counterfactual"
        check["policy_mismatches"] = sorted(
            (row.get("span_id", ""), field, row.get(field, ""))
            for row in rows
            for field, expected in (
                ("recurrence_min_selected_probability", expected_min_probability),
                ("recurrence_max_source_error_m", expected_max_source_error),
                ("recurrence_allow_boundary", expected_allow_boundary),
            )
            if row.get(field, "").strip().lower() != str(expected).lower()
        )
        check["complete"] = bool(
            check["complete"] and not check["policy_mismatches"]
        )
        check["consistency_mismatches"] = _recurrence_consistency_mismatches(
            rows, [row.get("span_id", "") for row in rows]
        )
        check["complete"] = bool(
            check["complete"] and not check["consistency_mismatches"]
        )
        _apply_finite_check(
            check,
            rows,
            scope_field="span_id",
            numeric_fields={
                "honest_ppc_score_pct",
                "coverage",
                "selected_p50_m",
                "selected_p95_m",
                "selected_p99_m",
                "recurrence_abstained_epochs",
                "recurrence_acceptance_rate",
                "runtime_s",
            },
        )
        checks.append(check)
    return {"complete": all(bool(check["complete"]) for check in checks), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-out",
        type=Path,
        help="also save the complete audit payload as JSON",
    )
    args = parser.parse_args(argv)
    payload = audit()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
