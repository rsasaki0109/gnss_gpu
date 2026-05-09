"""Replay hypothetical GSDC2023 source-selection guards over saved chunk payloads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.gsdc2023_chunk_selection import (
    ChunkCandidateQuality,
    ChunkSelectionRecord,
    GATED_BASELINE_THRESHOLD_DEFAULT,
    select_gated_chunk_source,
)
from experiments.gsdc2023_solver_selection import raw_wls_max_gap_guard_m


@dataclass(frozen=True)
class GuardVariant:
    name: str
    phone: str
    max_gap_m: float


def parse_variant(raw: str) -> GuardVariant:
    # Format: "name:phone:max_gap" or "phone:max_gap".
    parts = [part.strip() for part in raw.split(":")]
    if len(parts) == 2:
        phone, max_gap = parts
        name = f"{phone}_raw_gap_max_{max_gap.replace('.', 'p')}"
    elif len(parts) == 3:
        name, phone, max_gap = parts
    else:
        raise ValueError(f"invalid variant {raw!r}; expected phone:max_gap or name:phone:max_gap")
    phone = phone.lower()
    if phone not in {"all", "pixel5", "pixel", "non_mi8"}:
        raise ValueError(f"unsupported phone scope {phone!r}")
    return GuardVariant(name=name, phone=phone, max_gap_m=float(max_gap))


def expand_inputs(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(path) for path in glob.glob(value, recursive=True)]
        paths.extend(matches if matches else [Path(value)])
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file() and path.suffix == ".json":
            unique[str(path)] = path
    return [unique[key] for key in sorted(unique)]


def _payload_with_records(payload: dict[str, object]) -> tuple[dict[str, object] | None, str]:
    if isinstance(payload.get("chunk_selection_records"), list):
        return payload, str(payload.get("trip", ""))
    raw_bridge = payload.get("raw_bridge")
    if isinstance(raw_bridge, dict) and isinstance(raw_bridge.get("chunk_selection_records"), list):
        return raw_bridge, str(raw_bridge.get("trip", payload.get("trip", "")))
    return None, ""


def load_chunk_payload(path: Path) -> tuple[dict[str, object] | None, str, str | None]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, "", "read_error"
    if not isinstance(payload, dict):
        return None, "", "not_object"
    chunk_payload, trip = _payload_with_records(payload)
    if chunk_payload is None:
        return None, "", None
    return chunk_payload, trip, None


def phone_from_trip(trip: str) -> str:
    return str(trip).rstrip("/").split("/")[-1].lower()


def quality_from_payload(payload: dict[str, object]) -> ChunkCandidateQuality:
    def value(name: str) -> float:
        try:
            return float(payload.get(name, 0.0))
        except (TypeError, ValueError):
            return float("nan")

    return ChunkCandidateQuality(
        mse_pr=value("mse_pr"),
        step_mean_m=value("step_mean_m"),
        step_p95_m=value("step_p95_m"),
        accel_mean_m=value("accel_mean_m"),
        accel_p95_m=value("accel_p95_m"),
        bridge_jump_m=value("bridge_jump_m"),
        baseline_gap_mean_m=value("baseline_gap_mean_m"),
        baseline_gap_p95_m=value("baseline_gap_p95_m"),
        baseline_gap_max_m=value("baseline_gap_max_m"),
        quality_score=value("quality_score"),
    )


def record_from_payload(record: dict[str, object]) -> ChunkSelectionRecord | None:
    candidates_raw = record.get("candidates", {})
    if not isinstance(candidates_raw, dict) or "baseline" not in candidates_raw:
        return None
    candidates: dict[str, ChunkCandidateQuality] = {}
    for name, quality_payload in candidates_raw.items():
        if isinstance(quality_payload, dict):
            candidates[str(name)] = quality_from_payload(quality_payload)
    if "baseline" not in candidates:
        return None
    return ChunkSelectionRecord(
        start_epoch=int(record.get("start_epoch", -1)),
        end_epoch=int(record.get("end_epoch", -1)),
        auto_source=str(record.get("auto_source", "")),
        candidates=candidates,
    )


def candidate_metric(record: ChunkSelectionRecord, source: str, metric: str) -> float:
    quality = record.candidates.get(source)
    if quality is None:
        return float("nan")
    return float(getattr(quality, metric))


def variant_applies(variant: GuardVariant, phone: str) -> bool:
    if variant.phone == "all":
        return True
    if variant.phone == "pixel":
        return phone.startswith("pixel")
    if variant.phone == "pixel5":
        return phone == "pixel5"
    if variant.phone == "non_mi8":
        return phone not in {"mi8", "xiaomimi8"}
    return False


def guarded_source(current_source: str, record: ChunkSelectionRecord, phone: str, variant: GuardVariant) -> str:
    if current_source != "raw_wls" or not variant_applies(variant, phone):
        return current_source
    raw_wls = record.candidates.get("raw_wls")
    if raw_wls is None:
        return current_source
    if raw_wls.baseline_gap_max_m > variant.max_gap_m:
        return "baseline"
    return current_source


def replay_file(path: Path, variants: list[GuardVariant], baseline_threshold: float) -> list[dict[str, object]]:
    payload, trip, error = load_chunk_payload(path)
    if error is not None or payload is None:
        return []
    records = payload.get("chunk_selection_records", [])
    if not isinstance(records, list):
        return []
    phone = phone_from_trip(trip)
    allow_mi8 = phone in {"mi8", "xiaomimi8"}
    current_raw_wls_max_gap_m = raw_wls_max_gap_guard_m(phone, "gated")
    rows: list[dict[str, object]] = []
    for record_payload in records:
        if not isinstance(record_payload, dict):
            continue
        record = record_from_payload(record_payload)
        if record is None:
            continue
        current = select_gated_chunk_source(
            record,
            baseline_threshold,
            allow_raw_wls_on_mi8_baseline_jump=allow_mi8,
            raw_wls_max_gap_m=current_raw_wls_max_gap_m,
        )
        row: dict[str, object] = {
            "file": str(path),
            "trip": trip,
            "phone": phone,
            "start_epoch": record.start_epoch,
            "end_epoch": record.end_epoch,
            "old_gated_source": str(record_payload.get("gated_source", "")),
            "current_source": current,
            "auto_source": record.auto_source,
            "baseline_mse_pr": candidate_metric(record, "baseline", "mse_pr"),
            "raw_wls_mse_pr": candidate_metric(record, "raw_wls", "mse_pr"),
            "fgo_mse_pr": candidate_metric(record, "fgo", "mse_pr"),
            "fgo_no_tdcp_mse_pr": candidate_metric(record, "fgo_no_tdcp", "mse_pr"),
            "baseline_step_p95_m": candidate_metric(record, "baseline", "step_p95_m"),
            "raw_wls_step_p95_m": candidate_metric(record, "raw_wls", "step_p95_m"),
            "raw_wls_baseline_gap_p95_m": candidate_metric(record, "raw_wls", "baseline_gap_p95_m"),
            "raw_wls_baseline_gap_max_m": candidate_metric(record, "raw_wls", "baseline_gap_max_m"),
            "raw_wls_quality_score": candidate_metric(record, "raw_wls", "quality_score"),
        }
        for variant in variants:
            source = guarded_source(current, record, phone, variant)
            row[f"{variant.name}_source"] = source
            row[f"{variant.name}_changed"] = source != current
        rows.append(row)
    return rows


def summarize(rows: pd.DataFrame, variants: list[GuardVariant]) -> dict[str, object]:
    summary: dict[str, object] = {
        "files": int(rows["file"].nunique()) if not rows.empty else 0,
        "records": int(len(rows)),
        "current_source_counts": rows["current_source"].value_counts(dropna=False).to_dict() if not rows.empty else {},
        "variants": {},
    }
    for variant in variants:
        source_column = f"{variant.name}_source"
        changed_column = f"{variant.name}_changed"
        changed = rows[rows[changed_column]] if changed_column in rows else rows.iloc[0:0]
        variant_summary = {
            "scope": variant.phone,
            "max_gap_m": variant.max_gap_m,
            "changed_records": int(len(changed)),
            "source_counts": rows[source_column].value_counts(dropna=False).to_dict() if source_column in rows else {},
            "changed_by_phone": changed["phone"].value_counts(dropna=False).to_dict() if not changed.empty else {},
            "changed_by_trip": changed["trip"].value_counts(dropna=False).head(20).to_dict() if not changed.empty else {},
        }
        summary["variants"][variant.name] = variant_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[], help="JSON path or glob. May be repeated.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[
            "all_gap200:all:200",
            "non_mi8_gap200:non_mi8:200",
            "pixel_gap200:pixel:200",
            "pixel5_gap200:pixel5:200",
            "pixel5_gap300:pixel5:300",
        ],
        help="guard variant as name:phone:max_gap. phone is all, non_mi8, pixel, or pixel5.",
    )
    parser.add_argument("--baseline-threshold", type=float, default=GATED_BASELINE_THRESHOLD_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants = [parse_variant(raw) for raw in args.variant]
    paths = expand_inputs(args.input)
    rows: list[dict[str, object]] = []
    for path in paths:
        rows.extend(replay_file(path, variants, args.baseline_threshold))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    replay_csv = args.output_dir / "guard_replay.csv"
    frame.to_csv(replay_csv, index=False)
    summary = summarize(frame, variants)
    summary["input_files"] = len(paths)
    summary["guard_replay_csv"] = str(replay_csv)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
