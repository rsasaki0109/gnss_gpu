#!/usr/bin/env python3
"""Export source-selection chunk rows from raw-bridge summary JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_chunk_selection import (  # noqa: E402
    GATED_BASELINE_THRESHOLD_DEFAULT,
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    select_gated_chunk_source,
)
from experiments.gsdc2023_solver_selection import (  # noqa: E402
    mi8_gated_baseline_jump_guard_enabled,
    raw_wls_max_gap_guard_m,
)


SOURCE_SCORE_KEYS = {
    "baseline": "kaggle_wls_score_m",
    "raw_wls": "raw_wls_score_m",
    "fgo": "fgo_score_m",
    "selected": "selected_score_m",
}
SOURCE_METRIC_KEYS = {
    "baseline": "kaggle_wls_metrics",
    "raw_wls": "raw_wls_metrics",
    "fgo": "fgo_metrics",
    "selected": "selected_metrics",
}
CANDIDATE_SOURCES = ("baseline", "raw_wls", "fgo", "fgo_no_tdcp")
CANDIDATE_QUALITY_FIELDS = (
    "mse_pr",
    "step_mean_m",
    "step_p95_m",
    "accel_mean_m",
    "accel_p95_m",
    "bridge_jump_m",
    "baseline_gap_mean_m",
    "baseline_gap_p95_m",
    "baseline_gap_max_m",
    "quality_score",
)


def expand_summary_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        paths.extend(matches if matches else [Path(pattern)])
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file():
            unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def raw_bridge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    bridge = payload.get("raw_bridge")
    return bridge if isinstance(bridge, dict) else payload


def _finite_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _oracle_source(row: dict[str, object]) -> str:
    scores = {
        source: _finite_float(row.get(f"{source}_score_m"))
        for source in ("baseline", "raw_wls", "fgo")
    }
    finite = {source: value for source, value in scores.items() if np.isfinite(value)}
    if not finite:
        return ""
    return min(finite, key=finite.get)


def _quality_from_payload(payload: object) -> ChunkCandidateQuality | None:
    if not isinstance(payload, dict):
        return None
    values: dict[str, float] = {}
    for field in CANDIDATE_QUALITY_FIELDS:
        value = _finite_float(payload.get(field))
        if not np.isfinite(value):
            return None
        values[field] = value
    return ChunkCandidateQuality(**values)


def recompute_gated_source(
    record: dict[str, Any],
    *,
    trip: str,
    position_source: str,
    gated_threshold: float,
) -> str | None:
    candidates_payload = record.get("candidates")
    if not isinstance(candidates_payload, dict):
        return None
    candidates: dict[str, ChunkCandidateQuality] = {}
    for source, payload in candidates_payload.items():
        quality = _quality_from_payload(payload)
        if quality is not None:
            candidates[str(source)] = quality
    if "baseline" not in candidates:
        return None
    chunk_record = ChunkSelectionRecord(
        start_epoch=int(record.get("start_epoch", 0)),
        end_epoch=int(record.get("end_epoch", 0)),
        auto_source=str(record.get("auto_source", "")),
        candidates=candidates,
    )
    phone = Path(trip).name
    return select_gated_chunk_source(
        chunk_record,
        gated_threshold,
        allow_raw_wls_on_mi8_baseline_jump=mi8_gated_baseline_jump_guard_enabled(phone, position_source),
        raw_wls_max_gap_m=raw_wls_max_gap_guard_m(phone, position_source),
    )


def summary_chunk_rows(path: Path, *, include_multi_chunk_scores: bool = False) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    bridge = raw_bridge_payload(payload)
    records = bridge.get("chunk_selection_records")
    if not isinstance(records, list):
        return []

    trip = str(bridge.get("trip") or payload.get("trip") or path.parent.name)
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    position_source = str(bridge.get("selected_source_mode") or config.get("position_source") or "")
    gated_threshold = _finite_float(
        config.get("gated_threshold", config.get("gated_baseline_threshold", GATED_BASELINE_THRESHOLD_DEFAULT)),
    )
    if not np.isfinite(gated_threshold):
        gated_threshold = GATED_BASELINE_THRESHOLD_DEFAULT
    n_records = len(records)
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        use_trip_scores = n_records == 1 or include_multi_chunk_scores
        row: dict[str, object] = {
            "metrics_name": path.parent.parent.name if path.parent.parent.name else path.parent.name,
            "trip_slug": trip.replace("/", "__"),
            "summary_path": str(path),
            "start_epoch": int(record.get("start_epoch", 0)),
            "end_epoch": int(record.get("end_epoch", bridge.get("n_epochs", 0))),
            "n_epochs": int(bridge.get("n_epochs", record.get("end_epoch", 0))),
            "auto_source": str(record.get("auto_source", "")),
            "gated_source": str(record.get("gated_source", "")),
        }
        row["current_gated_source"] = recompute_gated_source(
            record,
            trip=trip,
            position_source=position_source,
            gated_threshold=float(gated_threshold),
        )
        for source, key in SOURCE_SCORE_KEYS.items():
            row[f"{source}_score_m"] = bridge.get(key) if use_trip_scores else None
        for source, key in SOURCE_METRIC_KEYS.items():
            metrics = bridge.get(key)
            if isinstance(metrics, dict):
                row[f"{source}_rms2d_m"] = metrics.get("rms_2d_m")
        candidates = record.get("candidates", {})
        if isinstance(candidates, dict):
            for source in CANDIDATE_SOURCES:
                quality = candidates.get(source)
                if not isinstance(quality, dict):
                    continue
                row[f"{source}_candidate_mse_pr"] = quality.get("mse_pr")
                row[f"{source}_candidate_quality_score"] = quality.get("quality_score")
                row[f"{source}_candidate_gap_p95_m"] = quality.get("baseline_gap_p95_m")
                row[f"{source}_candidate_gap_max_m"] = quality.get("baseline_gap_max_m")
        row["oracle_source"] = _oracle_source(row)
        baseline = _finite_float(row.get("baseline_score_m"))
        fgo = _finite_float(row.get("fgo_score_m"))
        row["fgo_minus_baseline_score_m"] = fgo - baseline if np.isfinite(baseline) and np.isfinite(fgo) else None
        rows.append(row)
    return rows


def summary_paths_to_frame(paths: list[Path], *, include_multi_chunk_scores: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(summary_chunk_rows(path, include_multi_chunk_scores=include_multi_chunk_scores))
    return pd.DataFrame(rows)


def dataset_summary(frame: pd.DataFrame, paths: list[Path]) -> dict[str, object]:
    out: dict[str, object] = {
        "inputs": len(paths),
        "rows": int(len(frame)),
    }
    if frame.empty:
        return out
    out["oracle_source_counts"] = frame["oracle_source"].value_counts(dropna=False).astype(int).to_dict()
    if "current_gated_source" in frame and "gated_source" in frame:
        out["current_gated_changes"] = int((frame["current_gated_source"].astype(str) != frame["gated_source"].astype(str)).sum())
        out["current_gated_source_counts"] = frame["current_gated_source"].value_counts(dropna=False).astype(int).to_dict()
    if {"baseline_score_m", "selected_score_m", "fgo_score_m", "raw_wls_score_m"}.issubset(frame.columns):
        baseline = pd.to_numeric(frame["baseline_score_m"], errors="coerce")
        out["source_vs_baseline_gain_m"] = {}
        for source in ("selected", "raw_wls", "fgo"):
            values = pd.to_numeric(frame[f"{source}_score_m"], errors="coerce")
            out["source_vs_baseline_gain_m"][source] = float((baseline - values).sum(skipna=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", action="append", required=True, help="summary.json path or glob")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument(
        "--include-multi-chunk-scores",
        action="store_true",
        help="repeat full-trip scores on every chunk when a summary has multiple chunk records",
    )
    args = parser.parse_args()

    paths = expand_summary_paths(args.summary_json)
    frame = summary_paths_to_frame(paths, include_multi_chunk_scores=args.include_multi_chunk_scores)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)

    payload = dataset_summary(frame, paths)
    payload["output_csv"] = str(args.output_csv)
    print(json.dumps(payload, indent=2))
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
