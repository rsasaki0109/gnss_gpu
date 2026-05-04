#!/usr/bin/env python3
"""Build a reproducible pre-submit manifest for GSDC2023 candidate CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.smooth_gsdc2023_submission import gsdc_score_m, haversine_m


DEFAULT_RISKY_TRIPS = (
    "2021-11-05-18-28-us-ca-mtv-m/pixel6pro",
    "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro",
    "2023-05-25-17-32-us-ca-pao-j/pixel6pro",
)
REQUIRED_COLUMNS = ("tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = path.resolve()
    if cwd_path.exists():
        return cwd_path
    return (base_dir / path).resolve()


def _read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(missing)}")
    return frame


def _assert_same_keys(reference: pd.DataFrame, candidate: pd.DataFrame, *, label: str) -> None:
    if len(reference) != len(candidate):
        raise SystemExit(f"{label}: row count mismatch {len(reference)} != {len(candidate)}")
    for column in ("tripId", "UnixTimeMillis"):
        if not reference[column].equals(candidate[column]):
            raise SystemExit(f"{label}: {column} mismatch")


def _empty_delta_summary() -> dict[str, float | int | None]:
    return {
        "rows": 0,
        "changed_rows": 0,
        "score_m": None,
        "p50_m": None,
        "p95_m": None,
        "mean_m": None,
        "max_m": None,
    }


def _delta_summary(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float | int | None]:
    if reference.empty:
        return _empty_delta_summary()
    deltas = haversine_m(
        reference["LatitudeDegrees"].to_numpy(),
        reference["LongitudeDegrees"].to_numpy(),
        candidate["LatitudeDegrees"].to_numpy(),
        candidate["LongitudeDegrees"].to_numpy(),
    )
    score = gsdc_score_m(deltas)
    return {
        "rows": int(len(reference)),
        "changed_rows": int(np.count_nonzero(deltas > 1.0e-9)),
        "score_m": float(score["score_m"]),
        "p50_m": float(score["p50_m"]),
        "p95_m": float(score["p95_m"]),
        "mean_m": float(score["mean_m"]),
        "max_m": float(score["max_m"]),
    }


def _candidate_name(candidate_summary: dict[str, Any]) -> str:
    name = candidate_summary.get("candidate")
    if not isinstance(name, str) or not name:
        raise SystemExit("candidate summary is missing candidate name")
    return name


def _previous_candidate_path(previous_output_dir: Path, candidate_name: str, previous_tag: str) -> Path:
    previous_name = candidate_name.removesuffix("_p6p0")
    return (
        previous_output_dir
        / previous_name
        / f"submission_best_basecorr_posoffset_{previous_name}_plus_pixel5_patch_{previous_tag}.csv"
    )


def _trip_delta_rows(
    *,
    candidate_name: str,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    risky_trips: tuple[str, ...],
    previous: pd.DataFrame | None,
    previous_path: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trip in risky_trips:
        mask = reference["tripId"] == trip
        ref_trip = reference[mask].reset_index(drop=True)
        cand_trip = candidate[mask].reset_index(drop=True)
        input_delta = _delta_summary(ref_trip, cand_trip)

        previous_delta = _empty_delta_summary()
        previous_exists = False
        if previous is not None:
            previous_exists = True
            prev_trip = previous[mask].reset_index(drop=True)
            previous_delta = _delta_summary(prev_trip, cand_trip)

        rows.append(
            {
                "candidate": candidate_name,
                "tripId": trip,
                "rows": int(input_delta["rows"] or 0),
                "input_changed_rows": int(input_delta["changed_rows"] or 0),
                "input_score_m": input_delta["score_m"],
                "input_p50_m": input_delta["p50_m"],
                "input_p95_m": input_delta["p95_m"],
                "input_mean_m": input_delta["mean_m"],
                "input_max_m": input_delta["max_m"],
                "previous_output": str(previous_path) if previous_path is not None else None,
                "previous_exists": previous_exists,
                "previous_changed_rows": int(previous_delta["changed_rows"] or 0),
                "previous_score_m": previous_delta["score_m"],
                "previous_p50_m": previous_delta["p50_m"],
                "previous_p95_m": previous_delta["p95_m"],
                "previous_mean_m": previous_delta["mean_m"],
                "previous_max_m": previous_delta["max_m"],
            },
        )
    return rows


def _candidate_manifest_row(
    candidate_summary: dict[str, Any],
    *,
    candidate_name: str,
    candidate_path: Path,
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    risk_report: dict[str, Any],
) -> dict[str, Any]:
    scales = candidate_summary.get("effective_phone_scales")
    scales = scales if isinstance(scales, dict) else {}
    delta = _delta_summary(reference, candidate)
    return {
        "candidate": candidate_name,
        "output": str(candidate_path),
        "output_sha256": sha256_file(candidate_path),
        "summary_output_sha256": candidate_summary.get("output_sha256"),
        "rows": int(len(candidate)),
        "pixel6pro_scale": scales.get("pixel6pro"),
        "risk_enabled": bool(risk_report.get("enabled", False)),
        "risk_risky_chunks": int(risk_report.get("risky_chunks", 0) or 0),
        "risk_risky_rows": int(risk_report.get("risky_rows", 0) or 0),
        "risk_vd_guard_rows": int(risk_report.get("vd_guard_rows", 0) or 0),
        "risk_candidate_actionable_chunks": int(
            risk_report.get("candidate_actionable_risky_chunks", risk_report.get("risky_chunks", 0)) or 0,
        ),
        "risk_candidate_actionable_rows": int(
            risk_report.get("candidate_actionable_risky_rows", risk_report.get("risky_rows", 0)) or 0,
        ),
        "delta_vs_input_score_m": delta["score_m"],
        "delta_vs_input_p50_m": delta["p50_m"],
        "delta_vs_input_p95_m": delta["p95_m"],
        "delta_vs_input_max_m": delta["max_m"],
        "delta_vs_input_changed_rows": delta["changed_rows"],
    }


def _matlab_equivalence_manifest(summary_path: Path) -> dict[str, Any]:
    summary_path = summary_path.expanduser().resolve()
    payload = _read_json(summary_path)
    gates = payload.get("gates")
    gates = gates if isinstance(gates, dict) else {}
    residual = gates.get("residual_values")
    residual = residual if isinstance(residual, dict) else {}
    return {
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "passed": bool(payload.get("passed", False)),
        "equivalence_claim": payload.get("equivalence_claim"),
        "trip_count": int(payload.get("trip_count", 0) or 0),
        "max_epochs": int(payload.get("max_epochs", 0) or 0),
        "count_max_epochs": int(payload.get("count_max_epochs", 0) or 0),
        "factor_mask_passed": bool(gates.get("factor_mask", {}).get("passed", False))
        if isinstance(gates.get("factor_mask"), dict)
        else False,
        "raw_bridge_counts_passed": bool(gates.get("raw_bridge_counts", {}).get("passed", False))
        if isinstance(gates.get("raw_bridge_counts"), dict)
        else False,
        "residual_values_passed": bool(residual.get("passed", False)),
        "residual_total_matlab_only": int(residual.get("total_matlab_only", 0) or 0),
        "residual_total_bridge_only": int(residual.get("total_bridge_only", 0) or 0),
        "residual_overall_max_abs_delta_m": residual.get("overall_max_abs_delta"),
        "residual_max_abs_delta_threshold_m": residual.get("max_abs_delta_threshold_m"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pre_submit_manifest(
    build_summary_path: Path,
    *,
    output_dir: Path | None = None,
    previous_output_dir: Path | None = None,
    previous_tag: str = "20260501",
    risky_trips: tuple[str, ...] = DEFAULT_RISKY_TRIPS,
    matlab_equivalence_summary: Path | None = None,
) -> dict[str, Any]:
    build_summary_path = build_summary_path.expanduser().resolve()
    build_summary = _read_json(build_summary_path)
    base_dir = build_summary_path.parent
    output_dir = (output_dir or base_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path_raw = build_summary.get("input")
    if not isinstance(input_path_raw, str):
        raise SystemExit(f"{build_summary_path} is missing input")
    input_path = _resolve_path(input_path_raw, base_dir)
    reference = _read_submission(input_path)

    candidates = build_summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SystemExit(f"{build_summary_path} has no candidates")

    risk_report = build_summary.get("pr_proxy_risk_report")
    risk_report = risk_report if isinstance(risk_report, dict) else {"enabled": False}
    matlab_equivalence_gate = (
        _matlab_equivalence_manifest(matlab_equivalence_summary) if matlab_equivalence_summary is not None else None
    )
    candidate_rows: list[dict[str, Any]] = []
    trip_rows: list[dict[str, Any]] = []

    for candidate_summary_raw in candidates:
        if not isinstance(candidate_summary_raw, dict):
            raise SystemExit("candidate summary must be an object")
        candidate_name = _candidate_name(candidate_summary_raw)
        output_raw = candidate_summary_raw.get("output")
        if not isinstance(output_raw, str):
            raise SystemExit(f"{candidate_name} is missing output path")
        candidate_path = _resolve_path(output_raw, base_dir)
        candidate = _read_submission(candidate_path)
        _assert_same_keys(reference, candidate, label=candidate_name)

        previous_path: Path | None = None
        previous_frame: pd.DataFrame | None = None
        if previous_output_dir is not None:
            previous_path = _previous_candidate_path(previous_output_dir.expanduser().resolve(), candidate_name, previous_tag)
            if previous_path.is_file():
                previous_frame = _read_submission(previous_path)
                _assert_same_keys(reference, previous_frame, label=f"previous {candidate_name}")

        candidate_rows.append(
            _candidate_manifest_row(
                candidate_summary_raw,
                candidate_name=candidate_name,
                candidate_path=candidate_path,
                candidate=candidate,
                reference=reference,
                risk_report=risk_report,
            ),
        )
        trip_rows.extend(
            _trip_delta_rows(
                candidate_name=candidate_name,
                reference=reference,
                candidate=candidate,
                risky_trips=risky_trips,
                previous=previous_frame,
                previous_path=previous_path,
            ),
        )

    candidate_csv = output_dir / "pre_submit_candidate_manifest.csv"
    trip_csv = output_dir / "pre_submit_trip_delta_checks.csv"
    manifest_path = output_dir / "pre_submit_manifest.json"
    _write_csv(candidate_csv, candidate_rows)
    _write_csv(trip_csv, trip_rows)

    manifest = {
        "build_summary": str(build_summary_path),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_dir": str(output_dir),
        "previous_output_dir": str(previous_output_dir.expanduser().resolve()) if previous_output_dir else None,
        "previous_tag": previous_tag if previous_output_dir else None,
        "risky_trips": list(risky_trips),
        "candidate_count": len(candidate_rows),
        "risk_report": {
            "enabled": bool(risk_report.get("enabled", False)),
            "risky_chunks": int(risk_report.get("risky_chunks", 0) or 0),
            "risky_rows": int(risk_report.get("risky_rows", 0) or 0),
            "vd_guard_rows": int(risk_report.get("vd_guard_rows", 0) or 0),
            "candidate_actionable_risky_chunks": int(
                risk_report.get("candidate_actionable_risky_chunks", risk_report.get("risky_chunks", 0)) or 0,
            ),
            "candidate_actionable_risky_rows": int(
                risk_report.get("candidate_actionable_risky_rows", risk_report.get("risky_rows", 0)) or 0,
            ),
            "candidate_actionable_by_candidate": risk_report.get("candidate_actionable_by_candidate", {}),
        },
        "matlab_equivalence_gate": matlab_equivalence_gate,
        "candidate_manifest_csv": str(candidate_csv),
        "trip_delta_checks_csv": str(trip_csv),
        "candidates": candidate_rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved: {manifest_path}")
    print(f"saved: {candidate_csv}")
    print(f"saved: {trip_csv}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--previous-output-dir", type=Path, default=None)
    parser.add_argument("--previous-tag", default="20260501")
    parser.add_argument("--risky-trip", action="append", dest="risky_trips")
    parser.add_argument("--matlab-equivalence-summary", type=Path)
    args = parser.parse_args(argv)

    build_pre_submit_manifest(
        args.build_summary,
        output_dir=args.output_dir,
        previous_output_dir=args.previous_output_dir,
        previous_tag=args.previous_tag,
        risky_trips=tuple(args.risky_trips or DEFAULT_RISKY_TRIPS),
        matlab_equivalence_summary=args.matlab_equivalence_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
