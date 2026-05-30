#!/usr/bin/env python3
"""Summarize why Phase59 consensus guard misses Phase58 oracle spans."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase43_span_oracle import _load_pool, _split_csv_values  # noqa: E402
from sim_phase43_consensus_guard import _read_csv, prepare_records  # noqa: E402


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def _int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def _fmt_counts(counts: Counter[str]) -> str:
    return ",".join(f"{label}:{count}" for label, count in counts.most_common())


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def summarize_span(
    span: dict[str, str],
    records_by_epoch: dict[int, dict[str, Any]],
    *,
    selected_labels: set[str],
    family_span_min_m: float,
    selected_agreement_max: float,
    min_agreement: int,
    pass_m: float,
) -> dict[str, Any]:
    start_epoch = _int(span, "start_epoch")
    end_epoch = _int(span, "end_epoch")
    records = [records_by_epoch[e] for e in range(start_epoch, end_epoch + 1) if e in records_by_epoch]

    weight_total = sum(float(rec["weight"]) for rec in records)
    base_pass = sum(float(rec["weight"]) for rec in records if bool(rec["current_pass"]))
    consensus_pass = 0.0
    consensus_override_epochs = 0
    consensus_good_epochs = 0
    consensus_bad_epochs = 0
    consensus_labels: Counter[str] = Counter()
    selected_counter: Counter[str] = Counter(str(rec["selected"]) for rec in records)
    chosen_errors: list[float] = []
    family_spans = _finite([float(rec["family_span"]) for rec in records])
    selected_agreements = _finite([float(rec["selected_agreement"]) for rec in records])
    eligible_epochs = 0
    chosen_available_epochs = 0

    for rec in records:
        current_pass = bool(rec["current_pass"])
        chosen_error = float(rec["current_error"])
        selected = str(rec["selected"])
        family_span = float(rec["family_span"])
        selected_agreement = float(rec["selected_agreement"])
        eligible = (
            selected in selected_labels
            and math.isfinite(family_span)
            and family_span >= family_span_min_m
            and (not math.isfinite(selected_agreement) or selected_agreement <= selected_agreement_max)
        )
        if eligible:
            eligible_epochs += 1
            chosen = rec["best_by_min"].get(min_agreement)
            if chosen is not None:
                chosen_available_epochs += 1
                chosen_error = float(chosen["error_m"])
                chosen_errors.append(chosen_error)
                consensus_labels[str(chosen["label"])] += 1
                consensus_override_epochs += 1
                chosen_pass = chosen_error <= pass_m
                if chosen_pass and not current_pass:
                    consensus_good_epochs += 1
                elif current_pass and not chosen_pass:
                    consensus_bad_epochs += 1
        if math.isfinite(chosen_error) and chosen_error <= pass_m:
            consensus_pass += float(rec["weight"])

    current_fail = weight_total - base_pass
    consensus_gain = consensus_pass - base_pass
    best_oracle_gain = _float(span, "gated_oracle_gain_m")
    capture_ratio = consensus_gain / best_oracle_gain if best_oracle_gain > 0.0 else float("nan")
    return {
        "label": span.get("label", ""),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "n_epochs": len(records),
        "current_fail_m": current_fail,
        "gated_oracle_gain_m": best_oracle_gain,
        "consensus_gain_m": consensus_gain,
        "consensus_capture_ratio": capture_ratio,
        "eligible_epochs": eligible_epochs,
        "chosen_available_epochs": chosen_available_epochs,
        "consensus_override_epochs": consensus_override_epochs,
        "consensus_good_epochs": consensus_good_epochs,
        "consensus_bad_epochs": consensus_bad_epochs,
        "median_family_span_m": median(family_spans) if family_spans else "",
        "median_selected_agreement_1m": median(selected_agreements) if selected_agreements else "",
        "median_chosen_error_m": median(chosen_errors) if chosen_errors else "",
        "selected_labels": _fmt_counts(selected_counter),
        "consensus_labels": _fmt_counts(consensus_labels),
        "gated_best_labels": span.get("gated_best_labels", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("span_oracle_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument(
        "--selected-labels",
        default="xd_gici_w5,xd_gici_ir,xd_gici_z,xd_gici_zr,xd_gici_mb,xd_gici_r4,xd_gici_combo,xd_gici_he,xd_gici_la",
    )
    parser.add_argument("--family-span-min-m", type=float, default=100.0)
    parser.add_argument("--selected-agreement-max", type=float, default=20.0)
    parser.add_argument("--radius-m", type=float, default=1.0)
    parser.add_argument("--min-agreement", type=int, default=15)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--ratio-min", type=float, default=1.0)
    parser.add_argument("--residual-rms-max", type=float, default=50.0)
    parser.add_argument("--status5-residual-rms-max", type=float, default=0.3)
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/results/phase59_consensus_span_signals.csv"))
    args = parser.parse_args(argv)

    rows = _read_csv(args.internal_epochs_csv)
    spans = _read_csv(args.span_oracle_csv)
    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    records = prepare_records(
        rows=rows,
        candidates=candidates,
        min_agreement_values=[int(args.min_agreement)],
        radius_m=float(args.radius_m),
        exclude_current=True,
        pass_m=float(args.pass_m),
        ratio_min=float(args.ratio_min),
        residual_rms_max=float(args.residual_rms_max),
        status5_residual_rms_max=float(args.status5_residual_rms_max),
    )
    records_by_epoch = {int(float(row["epoch"])): rec for row, rec in zip(rows, records)}
    selected_labels = set(_split_csv_values(args.selected_labels))

    out = [
        summarize_span(
            span,
            records_by_epoch,
            selected_labels=selected_labels,
            family_span_min_m=float(args.family_span_min_m),
            selected_agreement_max=float(args.selected_agreement_max),
            min_agreement=int(args.min_agreement),
            pass_m=float(args.pass_m),
        )
        for span in spans
    ]
    _write_csv(args.out_csv, out)
    print(f"loaded candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
