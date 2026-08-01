#!/usr/bin/env python3
"""Truth-only audit of native basin candidate supply and validation recall."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from experiments.run_multisd_fgo_ppc_cv import _quantile, read_reference
except ModuleNotFoundError:  # Direct `python experiments/<script>.py` execution.
    from run_multisd_fgo_ppc_cv import _quantile, read_reference  # type: ignore[no-redef]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stream(path: Path, group_index: int) -> dict[float, list[dict[str, Any]]]:
    epochs: dict[float, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != "gnsspp_multisd_basin_v1":
                raise ValueError(f"invalid basin schema on line {line_number}")
            tow = round(float(row["tow"]), 3)
            epochs[tow]
            if (
                row.get("group_index") == int(group_index)
                and row.get("evaluated") is True
            ):
                epochs[tow].append(row)
    return dict(epochs)


def audit_candidate_supply(
    basin_jsonl: Path,
    reference_csv: Path,
    *,
    group_index: int = 0,
    correct_threshold_m: float = 0.5,
) -> dict[str, Any]:
    if not math.isfinite(correct_threshold_m) or correct_threshold_m <= 0.0:
        raise ValueError("correct threshold must be finite and positive")

    # Estimator artifact is parsed and hashed before truth is opened.
    epochs = _load_stream(basin_jsonl, group_index)
    stream_hash = _sha256(basin_jsonl)
    truth = read_reference(reference_csv)
    evaluated_epochs = 0
    oracle_correct = 0
    passed_correct = 0
    any_pass = 0
    unique_pass = 0
    unique_pass_correct = 0
    missing_truth = 0
    malformed_positions = 0
    best_errors: list[float] = []
    candidate_errors: list[float] = []
    correct_rank: dict[str, int] = defaultdict(int)
    residuals_seen = 0
    malformed_residuals = 0
    residual_groups: dict[
        str, dict[str, list[float] | int]
    ] = defaultdict(
        lambda: {
            "correct_abs_normalized": [],
            "wrong_abs_normalized": [],
            "correct_failed": 0,
            "wrong_failed": 0,
        }
    )
    for tow, rows in epochs.items():
        if not rows:
            continue
        evaluated_epochs += 1
        reference = truth.get(tow)
        if reference is None:
            missing_truth += 1
            continue
        scored: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            try:
                position = tuple(float(value) for value in row["position_ecef"])
            except (KeyError, TypeError, ValueError):
                malformed_positions += 1
                continue
            if len(position) != 3 or not all(math.isfinite(value) for value in position):
                malformed_positions += 1
                continue
            error = math.dist(position, reference)
            candidate_errors.append(error)
            scored.append((row, error))
            candidate_class = "correct" if error < correct_threshold_m else "wrong"
            residual_rows = row.get("validation_residuals", [])
            if not isinstance(residual_rows, list):
                malformed_residuals += 1
                continue
            for residual in residual_rows:
                if not isinstance(residual, dict):
                    malformed_residuals += 1
                    continue
                try:
                    satellite = str(residual["satellite"])
                    reference_satellite = str(residual["reference_satellite"])
                    signal = int(residual["signal"])
                    kind = str(residual["kind"])
                    normalized = abs(float(residual["normalized_residual"]))
                except (KeyError, TypeError, ValueError):
                    malformed_residuals += 1
                    continue
                if (
                    not satellite
                    or not reference_satellite
                    or kind not in {"carrier", "ddpr"}
                    or not math.isfinite(normalized)
                ):
                    malformed_residuals += 1
                    continue
                key = f"{kind}:{satellite}:{reference_satellite}:{signal}"
                group = residual_groups[key]
                values = group[f"{candidate_class}_abs_normalized"]
                assert isinstance(values, list)
                values.append(normalized)
                if residual.get("pass") is not True:
                    failure_key = f"{candidate_class}_failed"
                    group[failure_key] = int(group[failure_key]) + 1
                residuals_seen += 1
        if not scored:
            continue
        best_row, best_error = min(scored, key=lambda value: value[1])
        best_errors.append(best_error)
        is_oracle_correct = best_error < correct_threshold_m
        oracle_correct += int(is_oracle_correct)
        if is_oracle_correct:
            correct_rank[str(int(best_row.get("rank", -1)))] += 1
        passing = [(row, error) for row, error in scored if row.get("pass") is True]
        any_pass += int(bool(passing))
        passed_correct += int(
            any(error < correct_threshold_m for _row, error in passing)
        )
        unique_pass += int(len(passing) == 1)
        unique_pass_correct += int(
            len(passing) == 1 and passing[0][1] < correct_threshold_m
        )

    residual_summary: dict[str, Any] = {}
    for key, group in sorted(residual_groups.items()):
        correct_values = group["correct_abs_normalized"]
        wrong_values = group["wrong_abs_normalized"]
        assert isinstance(correct_values, list)
        assert isinstance(wrong_values, list)
        residual_summary[key] = {
            "correct_rows": len(correct_values),
            "wrong_rows": len(wrong_values),
            "correct_failed_rows": int(group["correct_failed"]),
            "wrong_failed_rows": int(group["wrong_failed"]),
            "correct_abs_normalized_p50": _quantile(correct_values, 0.5),
            "correct_abs_normalized_p95": _quantile(correct_values, 0.95),
            "wrong_abs_normalized_p50": _quantile(wrong_values, 0.5),
            "wrong_abs_normalized_p95": _quantile(wrong_values, 0.95),
        }

    return {
        "schema": "gnss_gpu_ppc_basin_fgo_candidate_supply_audit_v1",
        "truth_usage": "post_estimator_candidate_audit_only",
        "truth_opened_after_estimator_artifact": True,
        "group_index": group_index,
        "correct_threshold_m": correct_threshold_m,
        "stream_epochs": len(epochs),
        "evaluated_epochs": evaluated_epochs,
        "candidate_hypotheses": len(candidate_errors),
        "oracle_correct_epochs": oracle_correct,
        "oracle_correct_rate": oracle_correct / evaluated_epochs if evaluated_epochs else 0.0,
        "passed_correct_epochs": passed_correct,
        "any_pass_epochs": any_pass,
        "unique_pass_epochs": unique_pass,
        "unique_pass_correct_epochs": unique_pass_correct,
        "correct_candidate_rank_histogram": dict(sorted(correct_rank.items())),
        "validation_residual_diagnostics": {
            "rows": residuals_seen,
            "malformed_rows": malformed_residuals,
            "satellite_reference_signal_groups": residual_summary,
        },
        "best_candidate_error": {
            "median_m": _quantile(best_errors, 0.5),
            "p95_m": _quantile(best_errors, 0.95),
            "maximum_m": max(best_errors, default=None),
        },
        "integrity": {
            "missing_truth_epochs": missing_truth,
            "malformed_positions": malformed_positions,
            "passed": (
                missing_truth == 0
                and malformed_positions == 0
                and malformed_residuals == 0
            ),
        },
        "artifacts": {
            "basin_jsonl_sha256": stream_hash,
            "reference_csv_sha256": _sha256(reference_csv),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin-jsonl", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--group-index", type=int, default=0)
    parser.add_argument("--correct-threshold-m", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_candidate_supply(
        args.basin_jsonl,
        args.reference,
        group_index=args.group_index,
        correct_threshold_m=args.correct_threshold_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
