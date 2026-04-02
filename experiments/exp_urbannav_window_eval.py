#!/usr/bin/env python3
"""Evaluate fixed-size external UrbanNav windows from saved epoch errors."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _window_metrics(errors: np.ndarray) -> dict[str, float]:
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "mean_2d": float(np.mean(errors)),
        "rms_2d": float(np.sqrt(np.mean(errors * errors))),
        "p50": float(np.percentile(errors, 50.0)),
        "p95": float(np.percentile(errors, 95.0)),
        "max_2d": float(np.max(errors)),
        "outlier_rate_pct": float(np.mean(errors > 100.0) * 100.0),
        "catastrophic_rate_pct": float(np.mean(errors > 500.0) * 100.0),
    }


def _plot_summary(
    output_path: Path,
    summary_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    baseline: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [str(row["method"]) for row in summary_rows if str(row["method"]) != baseline]
    if not methods:
        return

    win_metrics = ("win_rate_rms_pct", "win_rate_p95_pct", "win_rate_outlier_pct", "win_rate_catastrophic_pct")
    win_labels = ("RMS", "P95", ">100 m", ">500 m")
    colors = {
        "PF-10K": "#f97316",
        "PF+RobustClear-10K": "#059669",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax_win, ax_delta = axes

    x = np.arange(len(win_metrics))
    width = 0.35 if len(methods) > 1 else 0.55
    offsets = np.linspace(-(len(methods) - 1) * width / 2.0, (len(methods) - 1) * width / 2.0, len(methods))
    for offset, method in zip(offsets, methods, strict=True):
        row = next(row for row in summary_rows if row["method"] == method)
        vals = [float(row[key]) for key in win_metrics]
        ax_win.bar(x + offset, vals, width=width, label=method, color=colors.get(method))
    ax_win.set_xticks(x, win_labels)
    ax_win.set_ylim(0.0, 105.0)
    ax_win.set_ylabel("win rate [%]")
    ax_win.set_title(f"Window wins vs {baseline}")
    ax_win.grid(True, axis="y", alpha=0.25)
    ax_win.legend()

    delta_data = [
        [float(row["delta_rms_2d_m"]) for row in comparison_rows if row["method"] == method]
        for method in methods
    ]
    box = ax_delta.boxplot(delta_data, tick_labels=methods, patch_artist=True, showfliers=False)
    for patch, method in zip(box["boxes"], methods, strict=True):
        patch.set_facecolor(colors.get(method, "#94a3b8"))
    ax_delta.axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    ax_delta.set_ylabel(r"$\Delta$RMS vs baseline [m]")
    ax_delta.set_title("Per-window RMS delta")
    ax_delta.grid(True, axis="y", alpha=0.25)

    fig.suptitle("UrbanNav external window analysis", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Windowed external evaluation from UrbanNav epoch dumps")
    parser.add_argument(
        "--epoch-csv",
        type=Path,
        default=RESULTS_DIR / "urbannav_fixed_eval_external_gej_trimble_qualityveto_epochs_epochs.csv",
        help="Combined epoch-error CSV from exp_urbannav_fixed_eval.py",
    )
    parser.add_argument("--window-size", type=int, default=500, help="Window size in epochs")
    parser.add_argument("--stride", type=int, default=250, help="Window stride in epochs")
    parser.add_argument("--baseline", type=str, default="EKF", help="Baseline method name")
    parser.add_argument(
        "--methods",
        type=str,
        default="EKF,PF-10K,PF+RobustClear-10K",
        help="Comma-separated methods to include",
    )
    parser.add_argument(
        "--results-prefix",
        type=str,
        default="urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250",
        help="Prefix for output files under experiments/results",
    )
    args = parser.parse_args()

    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    if args.baseline not in methods:
        raise ValueError("baseline must be included in --methods")
    if args.window_size <= 0 or args.stride <= 0:
        raise ValueError("window-size and stride must be positive")

    rows = _read_rows(args.epoch_csv)
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = row["method"]
        if method not in methods:
            continue
        grouped[row["run"]][method].append((int(row["epoch_index"]), float(row["error_2d"])))

    run_windows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for run_name, method_map in sorted(grouped.items()):
        missing = [method for method in methods if method not in method_map]
        if missing:
            raise ValueError(f"run {run_name} is missing methods: {', '.join(missing)}")

        series = {}
        for method, entries in method_map.items():
            entries_sorted = sorted(entries)
            series[method] = np.array([error for _, error in entries_sorted], dtype=np.float64)

        n_epochs = min(len(series[method]) for method in methods)
        if n_epochs < args.window_size:
            continue

        for start in range(0, n_epochs - args.window_size + 1, args.stride):
            end = start + args.window_size
            baseline_metrics = _window_metrics(series[args.baseline][start:end])
            baseline_metrics["method"] = args.baseline

            for method in methods:
                metrics = _window_metrics(series[method][start:end])
                run_windows.append(
                    {
                        "run": run_name,
                        "window_start_epoch": start,
                        "window_end_epoch": end - 1,
                        "window_size_epochs": args.window_size,
                        "method": method,
                        **metrics,
                    }
                )
                if method == args.baseline:
                    continue
                comparison_rows.append(
                    {
                        "run": run_name,
                        "window_start_epoch": start,
                        "window_end_epoch": end - 1,
                        "window_size_epochs": args.window_size,
                        "baseline": args.baseline,
                        "method": method,
                        "delta_rms_2d_m": float(metrics["rms_2d"] - baseline_metrics["rms_2d"]),
                        "delta_p95_m": float(metrics["p95"] - baseline_metrics["p95"]),
                        "delta_outlier_rate_pct": float(
                            metrics["outlier_rate_pct"] - baseline_metrics["outlier_rate_pct"]
                        ),
                        "delta_catastrophic_rate_pct": float(
                            metrics["catastrophic_rate_pct"] - baseline_metrics["catastrophic_rate_pct"]
                        ),
                        "win_rms": int(metrics["rms_2d"] < baseline_metrics["rms_2d"]),
                        "win_p95": int(metrics["p95"] < baseline_metrics["p95"]),
                        "win_outlier": int(metrics["outlier_rate_pct"] < baseline_metrics["outlier_rate_pct"]),
                        "win_catastrophic": int(
                            metrics["catastrophic_rate_pct"] <= baseline_metrics["catastrophic_rate_pct"]
                        ),
                    }
                )

    summary_rows: list[dict[str, object]] = []
    for method in methods:
        method_windows = [row for row in run_windows if row["method"] == method]
        summary_row: dict[str, object] = {
            "method": method,
            "n_windows": len(method_windows),
            "mean_rms_2d": float(np.mean([float(row["rms_2d"]) for row in method_windows])),
            "mean_p95": float(np.mean([float(row["p95"]) for row in method_windows])),
            "mean_outlier_rate_pct": float(
                np.mean([float(row["outlier_rate_pct"]) for row in method_windows])
            ),
            "mean_catastrophic_rate_pct": float(
                np.mean([float(row["catastrophic_rate_pct"]) for row in method_windows])
            ),
        }
        if method != args.baseline:
            method_comparisons = [row for row in comparison_rows if row["method"] == method]
            n_comp = len(method_comparisons)
            summary_row.update(
                {
                    "baseline": args.baseline,
                    "mean_delta_rms_2d_m": float(
                        np.mean([float(row["delta_rms_2d_m"]) for row in method_comparisons])
                    ),
                    "mean_delta_p95_m": float(
                        np.mean([float(row["delta_p95_m"]) for row in method_comparisons])
                    ),
                    "win_rate_rms_pct": float(np.mean([int(row["win_rms"]) for row in method_comparisons]) * 100.0),
                    "win_rate_p95_pct": float(np.mean([int(row["win_p95"]) for row in method_comparisons]) * 100.0),
                    "win_rate_outlier_pct": float(
                        np.mean([int(row["win_outlier"]) for row in method_comparisons]) * 100.0
                    ),
                    "win_rate_catastrophic_pct": float(
                        np.mean([int(row["win_catastrophic"]) for row in method_comparisons]) * 100.0
                    ),
                    "wins_rms": int(sum(int(row["win_rms"]) for row in method_comparisons)),
                    "wins_p95": int(sum(int(row["win_p95"]) for row in method_comparisons)),
                    "wins_outlier": int(sum(int(row["win_outlier"]) for row in method_comparisons)),
                    "wins_catastrophic": int(
                        sum(int(row["win_catastrophic"]) for row in method_comparisons)
                    ),
                    "n_comparisons": n_comp,
                }
            )
        summary_rows.append(summary_row)

    prefix = RESULTS_DIR / args.results_prefix
    _write_rows(run_windows, prefix.with_name(prefix.name + "_windows.csv"))
    _write_rows(comparison_rows, prefix.with_name(prefix.name + "_comparisons.csv"))
    _write_rows(summary_rows, prefix.with_name(prefix.name + "_summary.csv"))
    _plot_summary(prefix.with_name(prefix.name + "_wins.png"), summary_rows, comparison_rows, args.baseline)
    print(f"wrote {prefix}_*.csv and {prefix}_wins.png")


if __name__ == "__main__":
    main()
