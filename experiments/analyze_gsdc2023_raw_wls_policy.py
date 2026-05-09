"""Analyze train-backed raw_wls source-selection policy evidence.

The selector has two qualitatively different raw_wls paths:

* low/medium baseline PR-MSE rescue, which can be checked on train windows with
  ground-truth score deltas;
* high-baseline fallback, which often appears on test-only outlier chunks and
  therefore needs to be tracked separately from train-backed evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.gsdc2023_chunk_selection import (
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_FLOOR_M,
    GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_RATIO_MAX,
    GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_FLOOR_M,
    GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_RATIO_MAX,
    GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
    GATED_RAW_WLS_RESCUE_BASELINE_GAP_MAX_M,
    GATED_RAW_WLS_RESCUE_MSE_PR_MAX,
    GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX,
    select_gated_chunk_source,
)
from experiments.gsdc2023_solver_selection import (
    mi8_gated_baseline_jump_guard_enabled,
    raw_wls_max_gap_guard_m,
)


@dataclass(frozen=True)
class RawWlsVariant:
    name: str
    baseline_mse_min: float
    raw_mse_max: float
    raw_mse_ratio_max: float
    gap_max_m: float | None = None
    gap_p95_max_m: float | None = None


DEFAULT_VARIANTS = [
    RawWlsVariant(
        name="current_high_pr",
        baseline_mse_min=GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
        raw_mse_max=GATED_RAW_WLS_RESCUE_MSE_PR_MAX,
        raw_mse_ratio_max=GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX,
        gap_max_m=GATED_RAW_WLS_RESCUE_BASELINE_GAP_MAX_M,
    ),
    RawWlsVariant(
        name="current_high_pr_gap200",
        baseline_mse_min=GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
        raw_mse_max=GATED_RAW_WLS_RESCUE_MSE_PR_MAX,
        raw_mse_ratio_max=GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX,
        gap_max_m=200.0,
    ),
    RawWlsVariant(
        name="strict_raw20_ratio035_gap200",
        baseline_mse_min=GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
        raw_mse_max=20.0,
        raw_mse_ratio_max=0.35,
        gap_max_m=200.0,
    ),
    RawWlsVariant(
        name="strict_raw20_ratio030_gap150",
        baseline_mse_min=GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN,
        raw_mse_max=20.0,
        raw_mse_ratio_max=0.30,
        gap_max_m=150.0,
    ),
]


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else result


def phone_from_trip(trip: str) -> str:
    return str(trip).rstrip("/").split("/")[-1].lower()


def expand_inputs(values: Iterable[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for value in values:
        matches = glob.glob(value, recursive=True)
        for raw_path in matches if matches else [value]:
            path = Path(raw_path)
            if path.is_file():
                paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]


def variant_passes(row: pd.Series, variant: RawWlsVariant) -> bool:
    baseline_mse = finite_float(row.get("baseline_mse_pr"))
    raw_mse = finite_float(row.get("raw_wls_mse_pr"))
    if not (
        math.isfinite(baseline_mse)
        and math.isfinite(raw_mse)
        and baseline_mse >= variant.baseline_mse_min
        and raw_mse <= variant.raw_mse_max
        and raw_mse <= baseline_mse * variant.raw_mse_ratio_max
    ):
        return False
    if variant.gap_max_m is not None:
        gap_max = finite_float(row.get("raw_wls_baseline_gap_max_m"))
        if not math.isfinite(gap_max) or gap_max > variant.gap_max_m:
            return False
    if variant.gap_p95_max_m is not None:
        gap_p95 = finite_float(row.get("raw_wls_baseline_gap_p95_m"))
        if not math.isfinite(gap_p95) or gap_p95 > variant.gap_p95_max_m:
            return False
    return True


def load_train_audit(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["audit_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_train_variants(train: pd.DataFrame, variants: list[RawWlsVariant]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for variant in variants:
        if train.empty:
            selected = train.copy()
        else:
            mask = train.apply(lambda row: variant_passes(row, variant), axis=1)
            selected = train[mask].copy()
        if not selected.empty:
            selected["variant"] = variant.name
            selected_rows.append(selected)
        delta = pd.to_numeric(selected.get("raw_minus_baseline_score_m", pd.Series(dtype=float)), errors="coerce")
        summary_rows.append(
            {
                "variant": variant.name,
                "selected_windows": int(len(selected)),
                "raw_better_count": int((delta < -1e-9).sum()),
                "raw_worse_count": int((delta > 1e-9).sum()),
                "raw_equal_count": int((delta.abs() <= 1e-9).sum()),
                "sum_raw_minus_baseline_score_m": float(delta.sum()) if len(delta) else 0.0,
                "best_delta_m": float(delta.min()) if len(delta) else math.nan,
                "worst_delta_m": float(delta.max()) if len(delta) else math.nan,
                "baseline_mse_min": variant.baseline_mse_min,
                "raw_mse_max": variant.raw_mse_max,
                "raw_mse_ratio_max": variant.raw_mse_ratio_max,
                "gap_max_m": variant.gap_max_m,
                "gap_p95_max_m": variant.gap_p95_max_m,
            }
        )
    selected_frame = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return pd.DataFrame(summary_rows), selected_frame


def payload_with_records(payload: dict[str, object]) -> tuple[dict[str, object] | None, str]:
    if isinstance(payload.get("chunk_selection_records"), list):
        return payload, str(payload.get("trip", ""))
    raw_bridge = payload.get("raw_bridge")
    if isinstance(raw_bridge, dict) and isinstance(raw_bridge.get("chunk_selection_records"), list):
        return raw_bridge, str(raw_bridge.get("trip", payload.get("trip", "")))
    return None, ""


def candidate_value(candidates: dict[str, object], source: str, field: str) -> float:
    payload = candidates.get(source, {})
    if not isinstance(payload, dict):
        return math.nan
    return finite_float(payload.get(field))


def quality_from_payload(payload: dict[str, object]) -> ChunkCandidateQuality:
    return ChunkCandidateQuality(
        mse_pr=finite_float(payload.get("mse_pr")),
        step_mean_m=finite_float(payload.get("step_mean_m"), 0.0),
        step_p95_m=finite_float(payload.get("step_p95_m"), 0.0),
        accel_mean_m=finite_float(payload.get("accel_mean_m"), 0.0),
        accel_p95_m=finite_float(payload.get("accel_p95_m"), 0.0),
        bridge_jump_m=finite_float(payload.get("bridge_jump_m"), 0.0),
        baseline_gap_mean_m=finite_float(payload.get("baseline_gap_mean_m"), 0.0),
        baseline_gap_p95_m=finite_float(payload.get("baseline_gap_p95_m"), 0.0),
        baseline_gap_max_m=finite_float(payload.get("baseline_gap_max_m"), 0.0),
        quality_score=finite_float(payload.get("quality_score"), 0.0),
    )


def record_from_payload(record: dict[str, object]) -> ChunkSelectionRecord | None:
    candidates_raw = record.get("candidates", {})
    if not isinstance(candidates_raw, dict) or "baseline" not in candidates_raw:
        return None
    candidates: dict[str, ChunkCandidateQuality] = {}
    for name, candidate_payload in candidates_raw.items():
        if isinstance(candidate_payload, dict):
            candidates[str(name)] = quality_from_payload(candidate_payload)
    if "baseline" not in candidates:
        return None
    return ChunkSelectionRecord(
        start_epoch=int(record.get("start_epoch", -1)),
        end_epoch=int(record.get("end_epoch", -1)),
        auto_source=str(record.get("auto_source", "")),
        candidates=candidates,
    )


def high_baseline_motion_guard(row: dict[str, object]) -> bool:
    baseline_step_p95 = finite_float(row["baseline_step_p95_m"])
    raw_step_p95 = finite_float(row["raw_wls_step_p95_m"])
    raw_gap_p95 = finite_float(row["raw_wls_baseline_gap_p95_m"])
    return (
        math.isfinite(baseline_step_p95)
        and math.isfinite(raw_step_p95)
        and math.isfinite(raw_gap_p95)
        and raw_step_p95
        <= max(
            baseline_step_p95 * GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_RATIO_MAX,
            GATED_HIGH_BASELINE_CANDIDATE_STEP_P95_FLOOR_M,
        )
        and raw_gap_p95
        <= max(
            baseline_step_p95 * GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_RATIO_MAX,
            GATED_HIGH_BASELINE_CANDIDATE_GAP_P95_FLOOR_M,
        )
    )


def classify_raw_wls_branch(row: dict[str, object]) -> str:
    baseline_mse = finite_float(row["baseline_mse_pr"])
    raw_mse = finite_float(row["raw_wls_mse_pr"])
    if not math.isfinite(baseline_mse) or not math.isfinite(raw_mse):
        return "nonfinite"
    if baseline_mse > GATED_BASELINE_THRESHOLD_DEFAULT:
        if raw_mse < baseline_mse and high_baseline_motion_guard(row):
            return "high_baseline_fallback"
        return "high_baseline_blocked"
    series = pd.Series(row)
    if variant_passes(series, DEFAULT_VARIANTS[0]):
        return "train_backed_high_pr_rescue"
    return "other_raw_wls"


def load_metrics_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        chunk_payload, trip = payload_with_records(payload)
        if chunk_payload is None:
            continue
        records = chunk_payload.get("chunk_selection_records", [])
        if not isinstance(records, list):
            continue
        phone = phone_from_trip(trip)
        for record in records:
            if not isinstance(record, dict):
                continue
            candidates = record.get("candidates", {})
            if not isinstance(candidates, dict):
                candidates = {}
            selection_record = record_from_payload(record)
            current_source = str(record.get("gated_source", ""))
            if selection_record is not None:
                current_source = select_gated_chunk_source(
                    selection_record,
                    GATED_BASELINE_THRESHOLD_DEFAULT,
                    allow_raw_wls_on_mi8_baseline_jump=mi8_gated_baseline_jump_guard_enabled(phone, "gated"),
                    raw_wls_max_gap_m=raw_wls_max_gap_guard_m(phone, "gated"),
                )
            row = {
                "metrics_file": str(path),
                "trip": trip,
                "phone": phone,
                "start_epoch": int(record.get("start_epoch", -1)),
                "end_epoch": int(record.get("end_epoch", -1)),
                "saved_gated_source": str(record.get("gated_source", "")),
                "current_source": current_source,
                "auto_source": str(record.get("auto_source", "")),
                "baseline_mse_pr": candidate_value(candidates, "baseline", "mse_pr"),
                "raw_wls_mse_pr": candidate_value(candidates, "raw_wls", "mse_pr"),
                "raw_wls_mse_ratio": math.nan,
                "baseline_step_p95_m": candidate_value(candidates, "baseline", "step_p95_m"),
                "raw_wls_step_p95_m": candidate_value(candidates, "raw_wls", "step_p95_m"),
                "raw_wls_baseline_gap_p95_m": candidate_value(candidates, "raw_wls", "baseline_gap_p95_m"),
                "raw_wls_baseline_gap_max_m": candidate_value(candidates, "raw_wls", "baseline_gap_max_m"),
                "raw_wls_quality_score": candidate_value(candidates, "raw_wls", "quality_score"),
            }
            if math.isfinite(row["baseline_mse_pr"]) and row["baseline_mse_pr"] > 0:
                row["raw_wls_mse_ratio"] = row["raw_wls_mse_pr"] / row["baseline_mse_pr"]
            row["raw_wls_branch"] = classify_raw_wls_branch(row)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    raw = metrics[metrics["current_source"] == "raw_wls"].copy()
    if raw.empty:
        return pd.DataFrame(
            columns=["raw_wls_branch", "phone", "records", "epochs", "trip_count", "gap_max_max_m", "gap_p95_max_m"]
        )
    raw["epochs"] = raw["end_epoch"] - raw["start_epoch"]
    grouped = raw.groupby(["raw_wls_branch", "phone"], dropna=False)
    rows = []
    for (branch, phone), group in grouped:
        rows.append(
            {
                "raw_wls_branch": branch,
                "phone": phone,
                "records": int(len(group)),
                "epochs": int(group["epochs"].sum()),
                "trip_count": int(group["trip"].nunique()),
                "gap_max_max_m": float(group["raw_wls_baseline_gap_max_m"].max()),
                "gap_p95_max_m": float(group["raw_wls_baseline_gap_p95_m"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["raw_wls_branch", "phone"]).reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    train_summary: pd.DataFrame,
    train_selected: pd.DataFrame,
    metrics: pd.DataFrame,
    metrics_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_summary_path = output_dir / "train_variant_summary.csv"
    train_selected_path = output_dir / "train_variant_selected_windows.csv"
    metrics_path = output_dir / "metrics_chunk_policy_rows.csv"
    raw_metrics_path = output_dir / "metrics_raw_wls_records.csv"
    metrics_summary_path = output_dir / "metrics_raw_wls_branch_summary.csv"

    train_summary.to_csv(train_summary_path, index=False)
    train_selected.to_csv(train_selected_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metrics[metrics["current_source"] == "raw_wls"].to_csv(raw_metrics_path, index=False)
    metrics_summary.to_csv(metrics_summary_path, index=False)

    zero_worse = (
        train_summary[train_summary["raw_worse_count"] == 0]["variant"].tolist()
        if "raw_worse_count" in train_summary
        else []
    )
    summary = {
        "train_variant_summary_csv": str(train_summary_path),
        "train_variant_selected_windows_csv": str(train_selected_path),
        "metrics_chunk_policy_rows_csv": str(metrics_path),
        "metrics_raw_wls_records_csv": str(raw_metrics_path),
        "metrics_raw_wls_branch_summary_csv": str(metrics_summary_path),
        "train_zero_worse_variants": zero_worse,
        "metrics_raw_wls_branch_counts": (
            metrics_summary.to_dict(orient="records") if not metrics_summary.empty else []
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-audit", action="append", default=[], help="raw_wls rescue feature audit CSV.")
    parser.add_argument("--metrics", action="append", default=[], help="bridge_metrics JSON path or glob.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train = load_train_audit(expand_inputs(args.train_audit))
    train_summary, train_selected = evaluate_train_variants(train, DEFAULT_VARIANTS)
    metrics = load_metrics_rows(expand_inputs(args.metrics))
    metrics_summary = summarize_metrics(metrics)
    write_outputs(args.output_dir, train_summary, train_selected, metrics, metrics_summary)


if __name__ == "__main__":
    main()
