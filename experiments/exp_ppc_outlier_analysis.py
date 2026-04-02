#!/usr/bin/env python3
"""Analyze per-epoch PPC WLS errors for selected run/config pairs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from evaluate import compute_metrics, save_results
from exp_urbannav_baseline import _parse_weight_scale, load_real_data, run_wls

RESULTS_DIR = _SCRIPT_DIR / "results"


@dataclass(frozen=True)
class Selection:
    city: str
    run: str
    config: str
    systems: tuple[str, ...]
    weight_scale: str

    @property
    def run_dir(self) -> Path:
        return Path(self.city) / self.run


def load_selections(selection_csv: Path) -> list[Selection]:
    with open(selection_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    selections: list[Selection] = []
    for row in rows:
        systems = tuple(part.strip() for part in row["systems"].split(",") if part.strip())
        weight_scale = row.get("weight_scale", "").strip()
        if weight_scale == "-":
            weight_scale = ""
        selections.append(
            Selection(
                city=row["city"].strip(),
                run=row["run"].strip(),
                config=row["config"].strip(),
                systems=systems,
                weight_scale=weight_scale,
            )
        )
    return selections


def rows_to_columns(rows: list[dict]) -> dict[str, list]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {key: [row[key] for row in rows] for key in keys}


def build_segments(
    times: np.ndarray,
    errors_2d: np.ndarray,
    threshold: float,
    gap_limit: float,
) -> list[dict]:
    mask = np.asarray(errors_2d) >= threshold
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    segments: list[dict] = []
    start_pos = 0
    prev = idx[0]
    segment_id = 0
    for pos in range(1, len(idx)):
        current = idx[pos]
        if current != prev + 1 or float(times[current] - times[prev]) > gap_limit:
            segment_idx = idx[start_pos:pos]
            segment_errors = errors_2d[segment_idx]
            segments.append(
                {
                    "segment_id": segment_id,
                    "start_index": int(segment_idx[0]),
                    "end_index": int(prev),
                    "start_time": float(times[segment_idx[0]]),
                    "end_time": float(times[prev]),
                    "n_epochs": int(len(segment_idx)),
                    "duration_s": float(times[prev] - times[segment_idx[0]]),
                    "mean_error_2d": float(np.mean(segment_errors)),
                    "max_error_2d": float(np.max(segment_errors)),
                }
            )
            segment_id += 1
            start_pos = pos
        prev = current

    segment_idx = idx[start_pos:]
    segment_errors = errors_2d[segment_idx]
    segments.append(
        {
            "segment_id": segment_id,
            "start_index": int(segment_idx[0]),
            "end_index": int(prev),
            "start_time": float(times[segment_idx[0]]),
            "end_time": float(times[prev]),
            "n_epochs": int(len(segment_idx)),
            "duration_s": float(times[prev] - times[segment_idx[0]]),
            "mean_error_2d": float(np.mean(segment_errors)),
            "max_error_2d": float(np.max(segment_errors)),
        }
    )
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PPC outlier epochs and segments")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="PPC-Dataset root directory",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        required=True,
        help="CSV with city/run/config/systems/weight_scale columns",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Optional epoch cap. Use 0 or negative for full runs.",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=100.0,
        help="2D error threshold [m] used to define outlier epochs",
    )
    parser.add_argument(
        "--gap-factor",
        type=float,
        default=2.0,
        help="Allowed multiple of median dt within one outlier segment",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_outlier_analysis",
        help="Prefix for CSV outputs under experiments/results",
    )
    args = parser.parse_args()

    selections = load_selections(args.selection_csv.resolve())
    max_epochs = args.max_epochs if args.max_epochs and args.max_epochs > 0 else None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    epoch_rows: list[dict] = []
    outlier_rows: list[dict] = []
    segment_rows: list[dict] = []
    summary_rows: list[dict] = []

    print("=" * 72)
    print("  PPC Outlier Analysis")
    print("=" * 72)
    print(f"  Data root          : {args.data_root.resolve()}")
    print(f"  Selection CSV      : {args.selection_csv.resolve()}")
    print(f"  Runs               : {len(selections)}")
    print(f"  Max epochs         : {max_epochs if max_epochs is not None else 'full'}")
    print(f"  Outlier threshold  : {args.outlier_threshold:.1f} m")
    print()

    for selection in selections:
        run_dir = args.data_root / selection.city / selection.run
        print(f"[run] {selection.city}/{selection.run} ({selection.config})")
        data = load_real_data(run_dir, max_epochs=max_epochs, systems=selection.systems)
        if data is None:
            raise RuntimeError(f"failed to load PPC data: {run_dir}")

        positions, wls_ms = run_wls(
            data,
            weight_scale_by_system=_parse_weight_scale(selection.weight_scale),
        )
        metrics = compute_metrics(positions[:, :3], data["ground_truth"])
        errors_2d = np.asarray(metrics["errors_2d"], dtype=np.float64)
        errors_3d = np.asarray(metrics["errors_3d"], dtype=np.float64)
        times = np.asarray(data["times"], dtype=np.float64)
        sat_counts = np.asarray(data["satellite_counts"], dtype=np.int32)
        dt = float(np.median(np.diff(times))) if len(times) > 1 else float(data.get("dt", 1.0))
        gap_limit = max(args.gap_factor * dt, dt + 1e-6)

        segments = build_segments(times, errors_2d, args.outlier_threshold, gap_limit)
        worst_idx = int(np.argmax(errors_2d))
        outlier_mask = errors_2d >= args.outlier_threshold
        severe_mask = errors_2d >= 200.0
        catastrophic_mask = errors_2d >= 500.0

        for i in range(len(times)):
            epoch_row = {
                "city": selection.city,
                "run": selection.run,
                "config": selection.config,
                "systems": ",".join(selection.systems),
                "weight_scale": selection.weight_scale or "-",
                "epoch_index": int(i),
                "gps_tow": float(times[i]),
                "n_sat": int(sat_counts[i]),
                "error_2d": float(errors_2d[i]),
                "error_3d": float(errors_3d[i]),
                "est_x": float(positions[i, 0]),
                "est_y": float(positions[i, 1]),
                "est_z": float(positions[i, 2]),
            }
            epoch_rows.append(epoch_row)
            if outlier_mask[i]:
                outlier_rows.append(epoch_row.copy())

        for segment in segments:
            segment_rows.append(
                {
                    "city": selection.city,
                    "run": selection.run,
                    "config": selection.config,
                    "systems": ",".join(selection.systems),
                    "weight_scale": selection.weight_scale or "-",
                    **segment,
                }
            )

        longest_segment = max(segments, key=lambda row: row["n_epochs"], default=None)
        worst_segment = max(segments, key=lambda row: row["max_error_2d"], default=None)
        summary_rows.append(
            {
                "city": selection.city,
                "run": selection.run,
                "config": selection.config,
                "systems": ",".join(selection.systems),
                "weight_scale": selection.weight_scale or "-",
                "n_epochs": int(len(times)),
                "rms_2d": float(metrics["rms_2d"]),
                "p95": float(metrics["p95"]),
                "max_2d": float(metrics["max_2d"]),
                "wls_ms": float(wls_ms),
                "outlier_threshold": float(args.outlier_threshold),
                "outlier_epochs": int(np.count_nonzero(outlier_mask)),
                "outlier_rate_pct": float(100.0 * np.mean(outlier_mask)),
                "severe_epochs_200m": int(np.count_nonzero(severe_mask)),
                "catastrophic_epochs_500m": int(np.count_nonzero(catastrophic_mask)),
                "worst_epoch_index": int(worst_idx),
                "worst_epoch_time": float(times[worst_idx]),
                "worst_epoch_error_2d": float(errors_2d[worst_idx]),
                "n_outlier_segments": int(len(segments)),
                "longest_segment_epochs": int(longest_segment["n_epochs"]) if longest_segment else 0,
                "longest_segment_duration_s": float(longest_segment["duration_s"]) if longest_segment else 0.0,
                "worst_segment_max_error_2d": float(worst_segment["max_error_2d"]) if worst_segment else 0.0,
                "worst_segment_epochs": int(worst_segment["n_epochs"]) if worst_segment else 0,
            }
        )
        print(
            f"    rms_2d={metrics['rms_2d']:.2f} m"
            f"  p95={metrics['p95']:.2f} m"
            f"  outliers>{args.outlier_threshold:.0f}m={int(np.count_nonzero(outlier_mask))}"
            f"  segments={len(segments)}"
        )
        print()

    save_results(rows_to_columns(summary_rows), RESULTS_DIR / f"{args.results_prefix}_summary.csv")
    save_results(rows_to_columns(epoch_rows), RESULTS_DIR / f"{args.results_prefix}_epochs.csv")
    save_results(rows_to_columns(outlier_rows), RESULTS_DIR / f"{args.results_prefix}_outliers.csv")
    save_results(rows_to_columns(segment_rows), RESULTS_DIR / f"{args.results_prefix}_segments.csv")

    print("=" * 72)
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_summary.csv'}")
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_epochs.csv'}")
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_outliers.csv'}")
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_segments.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
