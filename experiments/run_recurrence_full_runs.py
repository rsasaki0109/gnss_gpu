#!/usr/bin/env python3
"""Chunked, resumable Recurrence Vector replay on all six official runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ppc_distance_score import honest_ppc_distance_score  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
OFFICIAL_RUNS = (
    ("tokyo", "run1", 11_951),
    ("tokyo", "run2", 9_151),
    ("tokyo", "run3", 15_301),
    ("nagoya", "run1", 7_651),
    ("nagoya", "run2", 9_451),
    ("nagoya", "run3", 5_201),
)
SAFE_MIN_SELECTED_PROBABILITY = 0.05
SAFE_MAX_SOURCE_ERROR_M = 20.0


def _recurrence_mode_flags(raw: bool) -> list[str]:
    if raw:
        return [
            "--recurrence-max-source-error-m",
            "0",
            "--recurrence-min-selected-probability",
            "0",
            "--recurrence-allow-boundary",
        ]
    return [
        "--recurrence-max-source-error-m",
        str(SAFE_MAX_SOURCE_ERROR_M),
        "--recurrence-min-selected-probability",
        str(SAFE_MIN_SELECTED_PROBABILITY),
    ]


def _chunk_is_complete(
    summary_path: Path,
    epoch_path: Path,
    *,
    start: int,
    count: int,
    raw: bool,
) -> bool:
    if not summary_path.exists() or not epoch_path.exists():
        return False
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_probability = 0.0 if raw else SAFE_MIN_SELECTED_PROBABILITY
    try:
        probability_matches = (
            float(payload["recurrence_min_selected_probability"])
            == expected_probability
        )
    except (KeyError, TypeError, ValueError):
        probability_matches = False
    try:
        raw_policy_matches = not raw or (
            float(payload["recurrence_max_source_error_m"]) == 0.0
            and payload.get("recurrence_allow_boundary") is True
        )
    except (KeyError, TypeError, ValueError):
        raw_policy_matches = False
    try:
        evaluated = int(payload["evaluated_epochs"])
        skipped = int(payload["skipped_epochs"])
        with epoch_path.open(newline="", encoding="utf-8") as handle:
            epoch_rows = max(0, sum(1 for _ in csv.reader(handle)) - 1)
        population_matches = (
            evaluated >= 0
            and skipped >= 0
            and evaluated + skipped <= int(count)
            and epoch_rows == evaluated
        )
    except (KeyError, OSError, TypeError, ValueError):
        population_matches = False
    return (
        int(payload.get("start_epoch", -1)) == int(start)
        and int(payload.get("requested_epochs", -1)) == int(count)
        and population_matches
        and probability_matches
        and raw_policy_matches
    )


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else float("nan")


def _summarize_epoch_files(
    paths: list[Path],
    *,
    requested_epochs: int,
    runtime_s: float,
    reference_path: Path | None = None,
    start_epoch: int = 0,
    end_epoch: int | None = None,
) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(csv.DictReader(path.open(newline="", encoding="utf-8")))
    raw_baseline = np.asarray([float(row["baseline_error_m"]) for row in rows])
    raw_selected = np.asarray([float(row["selected_error_m"]) for row in rows])
    finite = np.isfinite(raw_baseline) & np.isfinite(raw_selected)
    valid_rows = [row for row, valid in zip(rows, finite, strict=True) if valid]
    baseline = raw_baseline[finite]
    selected = raw_selected[finite]
    abstained = sum(
        row["recurrence_abstained"].strip().lower() == "true"
        for row in valid_rows
    )
    delta = baseline - selected
    result: dict[str, object] = {
        "requested_epochs": int(requested_epochs),
        "evaluated_epochs": int(selected.size),
        "coverage": float(selected.size / requested_epochs) if requested_epochs else 0.0,
        "recurrence_abstained_epochs": int(abstained),
        "recurrence_acceptance_rate": (
            float((selected.size - abstained) / selected.size)
            if selected.size
            else 0.0
        ),
        "baseline_p50_m": _quantile(baseline, 0.50),
        "baseline_p95_m": _quantile(baseline, 0.95),
        "baseline_p99_m": _quantile(baseline, 0.99),
        "selected_p50_m": _quantile(selected, 0.50),
        "selected_p95_m": _quantile(selected, 0.95),
        "selected_p99_m": _quantile(selected, 0.99),
        "selected_pass_0_5m": float(np.mean(selected <= 0.5)) if selected.size else 0.0,
        "selected_pass_1m": float(np.mean(selected <= 1.0)) if selected.size else 0.0,
        "selected_pass_3m": float(np.mean(selected <= 3.0)) if selected.size else 0.0,
        "improved_epochs": int(np.sum(delta > 1.0e-9)),
        "worsened_epochs": int(np.sum(delta < -1.0e-9)),
        "runtime_s": float(runtime_s),
        "runtime_ms_per_evaluated_epoch": (
            float(1000.0 * runtime_s / selected.size) if selected.size else float("nan")
        ),
    }
    if reference_path is not None:
        selected_score = honest_ppc_distance_score(
            {
                int(row["epoch"]): float(row["selected_error_m"])
                for row in valid_rows
            },
            reference_path,
            start_epoch=start_epoch,
            end_epoch=(start_epoch + requested_epochs if end_epoch is None else end_epoch),
        )
        baseline_score = honest_ppc_distance_score(
            {
                int(row["epoch"]): float(row["baseline_error_m"])
                for row in valid_rows
            },
            reference_path,
            start_epoch=start_epoch,
            end_epoch=(start_epoch + requested_epochs if end_epoch is None else end_epoch),
        )
        result.update(selected_score)
        result["baseline_honest_ppc_score_pct"] = baseline_score[
            "honest_ppc_score_pct"
        ]
        result["baseline_pass_distance_m"] = baseline_score["pass_distance_m"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO / "datasets/PPC-Dataset-data")
    parser.add_argument(
        "--source-pos-dir",
        type=Path,
        default=REPO / "experiments/results/libgnss_rtk_pos_v5",
    )
    parser.add_argument("--triangle-cache-dir", type=Path, default=Path("E:/datasets/plateau_cache"))
    parser.add_argument("--out-dir", type=Path, default=REPO / "experiments/results")
    parser.add_argument("--chunk-epochs", type=int, default=500)
    parser.add_argument("--radius-m", type=float, default=3.0)
    parser.add_argument("--spacing-m", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-scope",
        choices=[f"{city}_{run}_full" for city, run, _ in OFFICIAL_RUNS],
        help="process one official run only (for independent parallel workers)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="evaluate the ungated paper-method argmax as a counterfactual",
    )
    args = parser.parse_args()
    if args.chunk_epochs <= 0:
        raise ValueError("--chunk-epochs must be positive")
    artifact_prefix = (
        "candidate_3dma_recurrence_raw_full"
        if args.raw
        else "candidate_3dma_recurrence_full"
    )

    run_summaries: list[dict[str, object]] = []
    for city, run, n_ref_epochs in OFFICIAL_RUNS:
        if args.only_scope and f"{city}_{run}_full" != args.only_scope:
            continue
        epoch_paths: list[Path] = []
        runtime_s = 0.0
        for start in range(0, n_ref_epochs, args.chunk_epochs):
            count = min(args.chunk_epochs, n_ref_epochs - start)
            stem = f"{artifact_prefix}_{city}_{run}_chunk{start:05d}"
            prefix = args.out_dir / stem
            summary_path = args.out_dir / f"{stem}_summary.json"
            epoch_path = args.out_dir / f"{stem}_epochs.csv"
            if args.force or not _chunk_is_complete(
                summary_path, epoch_path, start=start, count=count, raw=args.raw
            ):
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("eval_candidate_3dma_ppc.py")),
                    "--data-dir",
                    str(args.data_root / city / run),
                    "--source-pos",
                    str(args.source_pos_dir / f"{city}_{run}_full.pos"),
                    "--triangle-cache-npz",
                    str(args.triangle_cache_dir / f"{city}_{run}_triangles.npz"),
                    "--out-prefix",
                    str(prefix),
                    "--start-epoch",
                    str(start),
                    "--max-epochs",
                    str(count),
                    "--strategy",
                    "recurrence_vector",
                    "--radius-m",
                    str(args.radius_m),
                    "--spacing-m",
                    str(args.spacing_m),
                ]
                command.extend(_recurrence_mode_flags(args.raw))
                print(f"[{city}/{run} {start}:{start + count}]", flush=True)
                subprocess.run(command, cwd=REPO, check=True)
            chunk = json.loads(summary_path.read_text(encoding="utf-8"))
            runtime_s += float(chunk.get("runtime_s", 0.0))
            epoch_paths.append(epoch_path)
        payload = _summarize_epoch_files(
            epoch_paths,
            requested_epochs=n_ref_epochs,
            runtime_s=runtime_s,
            reference_path=args.data_root / city / run / "reference.csv",
        )
        payload.update(
            {
                "city": city,
                "run": run,
                "evaluation_role": (
                    "development"
                    if (city, run) in {("tokyo", "run1"), ("nagoya", "run1")}
                    else "holdout"
                ),
                "recurrence_mode": "raw_counterfactual" if args.raw else "safe_gated",
                "recurrence_min_selected_probability": (
                    0.0 if args.raw else SAFE_MIN_SELECTED_PROBABILITY
                ),
                "recurrence_max_source_error_m": (
                    0.0 if args.raw else SAFE_MAX_SOURCE_ERROR_M
                ),
                "recurrence_allow_boundary": bool(args.raw),
            }
        )
        run_summaries.append(payload)
        (args.out_dir / f"{artifact_prefix}_{city}_{run}_summary.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    fields = ["city", "run", *[key for key in run_summaries[0] if key not in {"city", "run"}]]
    output = args.out_dir / f"{artifact_prefix}_runs_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in run_summaries)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
