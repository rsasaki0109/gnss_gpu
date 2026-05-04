#!/usr/bin/env python3
"""Flag chunk candidates where the pseudorange proxy improves but gated output stays baseline."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_LOW_BASELINE_MSE_MAX = 50.0
DEFAULT_MSE_RATIO_MAX = 0.75
GUARD_ROW_COLUMNS = [
    "file",
    "trip",
    "phone",
    "chunk_start_epoch",
    "chunk_end_epoch",
    "segment_start_epoch",
    "segment_end_epoch",
    "segment_epochs",
    "doppler_count",
    "doppler_rms_mps",
    "tdcp_count",
    "tdcp_rms_m",
    "reject_reason",
]


def expand_inputs(values: Iterable[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for value in values:
        matches = [Path(path) for path in glob.glob(value, recursive=True)]
        for path in matches if matches else [Path(value)]:
            if path.is_file() and path.suffix == ".json":
                paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]


def _payload_with_records(payload: dict[str, object]) -> tuple[dict[str, object] | None, str]:
    if isinstance(payload.get("chunk_selection_records"), list):
        return payload, str(payload.get("trip", ""))
    raw_bridge = payload.get("raw_bridge")
    if isinstance(raw_bridge, dict) and isinstance(raw_bridge.get("chunk_selection_records"), list):
        return raw_bridge, str(raw_bridge.get("trip", payload.get("trip", "")))
    return None, ""


def _float_from_mapping(payload: object, field: str) -> float:
    if not isinstance(payload, dict):
        return float("nan")
    try:
        return float(payload.get(field, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _int_from_mapping(payload: object, field: str, default: int = -1) -> int:
    if not isinstance(payload, dict):
        return default
    try:
        return int(payload.get(field, default))
    except (TypeError, ValueError):
        return default


def _phone_from_trip(trip: str) -> str:
    return trip.rstrip("/").split("/")[-1].lower()


def _guard_rows_from_bridge(path: Path, bridge: dict[str, object], trip: str) -> list[dict[str, object]]:
    records = bridge.get("vd_seed_guard_records", [])
    if not isinstance(records, list):
        return []
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        rows.append(
            {
                "file": str(path),
                "trip": trip,
                "phone": _phone_from_trip(trip),
                "chunk_start_epoch": _int_from_mapping(record, "chunk_start_epoch"),
                "chunk_end_epoch": _int_from_mapping(record, "chunk_end_epoch"),
                "segment_start_epoch": _int_from_mapping(record, "segment_start_epoch"),
                "segment_end_epoch": _int_from_mapping(record, "segment_end_epoch"),
                "segment_epochs": _int_from_mapping(record, "segment_epochs", 0),
                "doppler_count": _int_from_mapping(record, "doppler_count", 0),
                "doppler_rms_mps": _float_from_mapping(record, "doppler_rms_mps"),
                "tdcp_count": _int_from_mapping(record, "tdcp_count", 0),
                "tdcp_rms_m": _float_from_mapping(record, "tdcp_rms_m"),
                "reject_reason": str(record.get("reject_reason", "")),
            },
        )
    return rows


def guard_rows_from_payload(path: Path, payload: dict[str, object]) -> list[dict[str, object]]:
    bridge, trip = _payload_with_records(payload)
    if bridge is None:
        return []
    return _guard_rows_from_bridge(path, bridge, trip)


def _guard_overlap_summary(records: object, start_epoch: int, end_epoch: int) -> dict[str, object]:
    if not isinstance(records, list):
        records = []
    overlapping: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        seg_start = _int_from_mapping(record, "segment_start_epoch")
        seg_end = _int_from_mapping(record, "segment_end_epoch")
        chunk_start = _int_from_mapping(record, "chunk_start_epoch")
        chunk_end = _int_from_mapping(record, "chunk_end_epoch")
        same_chunk = chunk_start == start_epoch and chunk_end == end_epoch
        overlaps = seg_start < end_epoch and seg_end > start_epoch
        if same_chunk or overlaps:
            overlapping.append(record)
    reasons = sorted({str(record.get("reject_reason", "")) for record in overlapping if record.get("reject_reason")})
    doppler_values = [
        _float_from_mapping(record, "doppler_rms_mps")
        for record in overlapping
        if math.isfinite(_float_from_mapping(record, "doppler_rms_mps"))
    ]
    tdcp_values = [
        _float_from_mapping(record, "tdcp_rms_m")
        for record in overlapping
        if math.isfinite(_float_from_mapping(record, "tdcp_rms_m"))
    ]
    return {
        "vd_guard_overlap_segments": int(len(overlapping)),
        "vd_guard_overlap_epochs": int(sum(max(_int_from_mapping(record, "segment_epochs", 0), 0) for record in overlapping)),
        "vd_guard_reject_reasons": ";".join(reasons),
        "vd_guard_max_doppler_rms_mps": max(doppler_values) if doppler_values else float("nan"),
        "vd_guard_max_tdcp_rms_m": max(tdcp_values) if tdcp_values else float("nan"),
    }


def _source_row(
    *,
    path: Path,
    trip: str,
    record: dict[str, object],
    source: str,
    guard_records: object,
    low_baseline_mse_max: float,
    mse_ratio_max: float,
) -> dict[str, object] | None:
    candidates = record.get("candidates", {})
    if not isinstance(candidates, dict):
        return None
    baseline = candidates.get("baseline", {})
    candidate = candidates.get(source, {})
    baseline_mse = _float_from_mapping(baseline, "mse_pr")
    candidate_mse = _float_from_mapping(candidate, "mse_pr")
    if not (math.isfinite(baseline_mse) and math.isfinite(candidate_mse)):
        return None
    if baseline_mse <= 0.0 or candidate_mse >= baseline_mse * mse_ratio_max:
        return None
    if baseline_mse > low_baseline_mse_max:
        return None
    gated_source = str(record.get("gated_source", ""))
    if gated_source != "baseline":
        return None
    baseline_step_p95 = _float_from_mapping(baseline, "step_p95_m")
    candidate_gap_p95 = _float_from_mapping(candidate, "baseline_gap_p95_m")
    gap_guard_m = max(baseline_step_p95 if math.isfinite(baseline_step_p95) else 0.0, 12.0)
    risk_reasons: list[str] = ["low_baseline_pr_proxy_gain"]
    if math.isfinite(candidate_gap_p95) and candidate_gap_p95 > gap_guard_m:
        risk_reasons.append("baseline_gap_exceeds_fgo_guard")
    if source == "raw_wls":
        risk_reasons.append("raw_wls_proxy_candidate")
    if source.startswith("fgo"):
        risk_reasons.append("fgo_proxy_candidate")
    row = {
        "file": str(path),
        "trip": trip,
        "phone": _phone_from_trip(trip),
        "start_epoch": int(record.get("start_epoch", -1)),
        "end_epoch": int(record.get("end_epoch", -1)),
        "auto_source": str(record.get("auto_source", "")),
        "gated_source": gated_source,
        "candidate_source": source,
        "baseline_mse_pr": baseline_mse,
        "candidate_mse_pr": candidate_mse,
        "candidate_mse_ratio": candidate_mse / baseline_mse,
        "baseline_step_p95_m": baseline_step_p95,
        "candidate_step_p95_m": _float_from_mapping(candidate, "step_p95_m"),
        "candidate_baseline_gap_p95_m": candidate_gap_p95,
        "candidate_baseline_gap_max_m": _float_from_mapping(candidate, "baseline_gap_max_m"),
        "candidate_quality_score": _float_from_mapping(candidate, "quality_score"),
        "risk_reasons": ";".join(risk_reasons),
    }
    row.update(_guard_overlap_summary(guard_records, int(row["start_epoch"]), int(row["end_epoch"])))
    return row


def risky_rows_from_payload(
    path: Path,
    payload: dict[str, object],
    *,
    low_baseline_mse_max: float = DEFAULT_LOW_BASELINE_MSE_MAX,
    mse_ratio_max: float = DEFAULT_MSE_RATIO_MAX,
) -> list[dict[str, object]]:
    bridge, trip = _payload_with_records(payload)
    if bridge is None:
        return []
    records = bridge.get("chunk_selection_records", [])
    if not isinstance(records, list):
        return []
    guard_records = bridge.get("vd_seed_guard_records", [])
    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        candidates = record.get("candidates", {})
        if not isinstance(candidates, dict):
            continue
        for source in ("raw_wls", "fgo", "fgo_no_tdcp"):
            if source not in candidates:
                continue
            row = _source_row(
                path=path,
                trip=trip,
                record=record,
                source=source,
                guard_records=guard_records,
                low_baseline_mse_max=low_baseline_mse_max,
                mse_ratio_max=mse_ratio_max,
            )
            if row is not None:
                rows.append(row)
    return rows


def load_risky_rows(
    paths: Iterable[Path],
    *,
    low_baseline_mse_max: float = DEFAULT_LOW_BASELINE_MSE_MAX,
    mse_ratio_max: float = DEFAULT_MSE_RATIO_MAX,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows.extend(
            risky_rows_from_payload(
                path,
                payload,
                low_baseline_mse_max=low_baseline_mse_max,
                mse_ratio_max=mse_ratio_max,
            ),
        )
    return pd.DataFrame(rows)


def load_guard_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        rows.extend(guard_rows_from_payload(path, payload))
    return pd.DataFrame(rows, columns=GUARD_ROW_COLUMNS)


def summarize(rows: pd.DataFrame, input_count: int, guard_rows: pd.DataFrame | None = None) -> dict[str, object]:
    guard_summary: dict[str, object] = {
        "vd_guard_rows": 0,
        "vd_guard_by_phone": {},
        "vd_guard_reject_reasons": {},
    }
    if guard_rows is not None and not guard_rows.empty:
        guard_summary = {
            "vd_guard_rows": int(len(guard_rows)),
            "vd_guard_by_phone": guard_rows["phone"].value_counts(dropna=False).to_dict(),
            "vd_guard_reject_reasons": guard_rows["reject_reason"].value_counts(dropna=False).to_dict(),
        }
    if rows.empty:
        return {
            "input_files": int(input_count),
            "risky_rows": 0,
            "risky_chunks": 0,
            "by_phone": {},
            "by_candidate_source": {},
            "by_trip": {},
            **guard_summary,
        }
    chunk_keys = rows[["trip", "start_epoch", "end_epoch"]].drop_duplicates()
    return {
        "input_files": int(input_count),
        "risky_rows": int(len(rows)),
        "risky_chunks": int(len(chunk_keys)),
        "by_phone": rows["phone"].value_counts(dropna=False).to_dict(),
        "by_candidate_source": rows["candidate_source"].value_counts(dropna=False).to_dict(),
        "by_trip": rows["trip"].value_counts(dropna=False).head(50).to_dict(),
        **guard_summary,
    }


def write_outputs(
    output_dir: Path,
    rows: pd.DataFrame,
    summary: dict[str, object],
    guard_rows: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pr_proxy_risk_chunks.csv"
    rows.to_csv(csv_path, index=False)
    payload = dict(summary)
    payload["pr_proxy_risk_chunks_csv"] = str(csv_path)
    if guard_rows is not None:
        guard_csv_path = output_dir / "vd_seed_guard_records.csv"
        guard_rows.to_csv(guard_csv_path, index=False)
        payload["vd_seed_guard_records_csv"] = str(guard_csv_path)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="bridge_metrics.json path or glob; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--low-baseline-mse-max", type=float, default=DEFAULT_LOW_BASELINE_MSE_MAX)
    parser.add_argument("--mse-ratio-max", type=float, default=DEFAULT_MSE_RATIO_MAX)
    parser.add_argument("--fail-on-risk", action="store_true", help="exit non-zero when risky chunks are found")
    args = parser.parse_args(argv)

    paths = expand_inputs(args.input)
    rows = load_risky_rows(
        paths,
        low_baseline_mse_max=args.low_baseline_mse_max,
        mse_ratio_max=args.mse_ratio_max,
    )
    guard_rows = load_guard_rows(paths)
    summary = summarize(rows, len(paths), guard_rows)
    write_outputs(args.output_dir, rows, summary, guard_rows)
    return 2 if args.fail_on_risk and summary["risky_chunks"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
