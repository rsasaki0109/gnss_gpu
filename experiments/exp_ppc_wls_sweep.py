#!/usr/bin/env python3
"""Sweep PPC-Dataset runs for a fixed set of WLS constellation configurations.

This script is meant for quick cross-run comparison after tuning multi-GNSS
weighting on a single PPC run.
"""

from __future__ import annotations

import argparse
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
from gnss_gpu.io.ppc import PPCDatasetLoader

RESULTS_DIR = _SCRIPT_DIR / "results"


@dataclass(frozen=True)
class SweepConfig:
    label: str
    systems: tuple[str, ...]
    weight_scale: str = ""


DEFAULT_CONFIGS = (
    SweepConfig(label="G", systems=("G",)),
    SweepConfig(label="G,J", systems=("G", "J")),
    SweepConfig(label="G,E", systems=("G", "E")),
    SweepConfig(label="G,E,J", systems=("G", "E", "J")),
    SweepConfig(label="G,E,J scaled", systems=("G", "E", "J"), weight_scale="E=0.1,J=2.0"),
)


def select_configs(labels: tuple[str, ...] | None) -> tuple[SweepConfig, ...]:
    if not labels:
        return DEFAULT_CONFIGS

    selected: list[SweepConfig] = []
    available = {config.label: config for config in DEFAULT_CONFIGS}
    for label in labels:
        key = label.strip()
        if not key:
            continue
        if key not in available:
            raise ValueError(f"unknown config label: {key}")
        selected.append(available[key])
    return tuple(selected)


def discover_run_dirs(data_root: Path) -> list[Path]:
    run_dirs = [
        path
        for path in sorted(data_root.rglob("run*"))
        if PPCDatasetLoader.is_run_directory(path)
    ]
    return run_dirs


def summarize_configs(rows: list[dict]) -> list[dict]:
    labels = sorted({row["config"] for row in rows})
    summary_rows: list[dict] = []
    for label in labels:
        selected = [row for row in rows if row["config"] == label]
        rms = np.array([row["rms_2d"] for row in selected], dtype=np.float64)
        p95 = np.array([row["p95"] for row in selected], dtype=np.float64)
        summary_rows.append(
            {
                "config": label,
                "systems": selected[0]["systems"],
                "weight_scale": selected[0]["weight_scale"],
                "n_runs": len(selected),
                "mean_rms_2d": float(np.mean(rms)),
                "median_rms_2d": float(np.median(rms)),
                "max_rms_2d": float(np.max(rms)),
                "mean_p95": float(np.mean(p95)),
            }
        )
    return summary_rows


def summarize_best(rows: list[dict]) -> list[dict]:
    run_keys = sorted({(row["city"], row["run"]) for row in rows})
    best_rows: list[dict] = []
    for city, run in run_keys:
        selected = [row for row in rows if row["city"] == city and row["run"] == run]
        best = min(selected, key=lambda row: row["rms_2d"])
        best_rows.append(best)
    return best_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep PPC WLS multi-GNSS configurations")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to PPC-Dataset root, city directory, or single run directory",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=300,
        help="Epoch cap per run. Use 0 or a negative value for full runs.",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="ppc_wls_sweep",
        help="Prefix for CSV outputs under experiments/results",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Optional config labels separated by ';', e.g. 'G,E;G,E,J'",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    run_dirs = [data_root] if PPCDatasetLoader.is_run_directory(data_root) else discover_run_dirs(data_root)
    if not run_dirs:
        raise FileNotFoundError(f"no PPC run directories found under: {data_root}")

    max_epochs = args.max_epochs if args.max_epochs and args.max_epochs > 0 else None
    config_labels = tuple(part.strip() for part in args.configs.split(";") if part.strip())
    configs = select_configs(config_labels)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  PPC WLS Sweep")
    print("=" * 72)
    print(f"  Data root : {data_root}")
    print(f"  Runs      : {len(run_dirs)}")
    print(f"  Max epochs: {max_epochs if max_epochs is not None else 'full'}")
    print(f"  Configs   : {', '.join(config.label for config in configs)}")
    print()

    cache: dict[tuple[Path, tuple[str, ...]], dict] = {}
    rows: list[dict] = []

    for run_dir in run_dirs:
        print(f"[run] {run_dir.parent.name}/{run_dir.name}")
        for config in configs:
            cache_key = (run_dir, config.systems)
            if cache_key not in cache:
                data = load_real_data(run_dir, max_epochs=max_epochs, systems=config.systems)
                if data is None:
                    raise RuntimeError(f"failed to load PPC data: {run_dir}")
                cache[cache_key] = data
            data = cache[cache_key]

            pos, wls_ms = run_wls(
                data,
                weight_scale_by_system=_parse_weight_scale(config.weight_scale),
            )
            metrics = compute_metrics(pos[:, :3], data["ground_truth"])
            row = {
                "city": run_dir.parent.name,
                "run": run_dir.name,
                "config": config.label,
                "systems": ",".join(config.systems),
                "weight_scale": config.weight_scale or "-",
                "n_epochs": int(data["n_epochs"]),
                "median_satellites": int(data["n_satellites"]),
                "rms_2d": float(metrics["rms_2d"]),
                "p95": float(metrics["p95"]),
                "time_ms": float(wls_ms),
            }
            rows.append(row)
            print(
                f"    {config.label:<12} rms_2d={row['rms_2d']:.2f} m"
                f"  p95={row['p95']:.2f} m  sats={row['median_satellites']}"
            )
        print()

    per_run = {key: [row[key] for row in rows] for key in rows[0]}
    config_summary_rows = summarize_configs(rows)
    config_summary = {key: [row[key] for row in config_summary_rows] for key in config_summary_rows[0]}
    best_rows = summarize_best(rows)
    best_summary = {key: [row[key] for row in best_rows] for key in best_rows[0]}

    save_results(per_run, RESULTS_DIR / f"{args.results_prefix}_runs.csv")
    save_results(config_summary, RESULTS_DIR / f"{args.results_prefix}_configs.csv")
    save_results(best_summary, RESULTS_DIR / f"{args.results_prefix}_best.csv")

    print("=" * 72)
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_runs.csv'}")
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_configs.csv'}")
    print(f"  Saved: {RESULTS_DIR / f'{args.results_prefix}_best.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
