#!/usr/bin/env python3
"""Build paper-ready tables and figures from fixed experiment CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluate import plot_cdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR = RESULTS_DIR / "paper_assets"

PPC_HOLDOUT_RUNS_CSV = "pf_strategy_lab_holdout6_r200_s200_runs.csv"
PPC_HOLDOUT_SUMMARY_CSV = "pf_strategy_lab_holdout6_r200_s200_summary.csv"
URBANNAV_RUNS_CSV = "urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv"
URBANNAV_SUMMARY_CSV = "urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv"
URBANNAV_EPOCHS_PREFIX = "urbannav_fixed_eval_external_gej_trimble_qualityveto_epochs"
BVH_RUNTIME_CSV = "ppc_pf3d_tokyo_run1_g_100_plateau_summary.csv"
HK_ADAPTIVE_SUMMARY_CSV = "urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv"

PPC_SAFE = "always_robust"
PPC_EXPLORATORY = (
    "entry_veto_negative_exit_rescue_branch_aware_"
    "hysteresis_quality_veto_regime_gate"
)


def _read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS_DIR / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"row not found: {key}={value}")


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _aggregate_strategy_runs(rows: list[dict[str, str]], strategy: str) -> dict[str, float]:
    selected = [row for row in rows if row["strategy"] == strategy]
    return {
        "mean_rms_2d": float(np.mean([_f(row, "rms_2d") for row in selected])),
        "mean_p95": float(np.mean([_f(row, "p95") for row in selected])),
        "mean_outlier_rate_pct": float(np.mean([_f(row, "outlier_rate_pct") for row in selected])),
        "mean_catastrophic_rate_pct": float(
            np.mean([_f(row, "catastrophic_rate_pct") for row in selected])
        ),
        "mean_blocked_epoch_frac": float(
            np.mean([_f(row, "blocked_epoch_frac") for row in selected])
        ),
    }


def _paper_main_table() -> list[dict[str, object]]:
    ppc_holdout_runs = _read_csv(PPC_HOLDOUT_RUNS_CSV)
    urbannav_summary = _read_csv(URBANNAV_SUMMARY_CSV)
    bvh_rows = _read_csv(BVH_RUNTIME_CSV)

    ppc_safe = _aggregate_strategy_runs(ppc_holdout_runs, PPC_SAFE)
    ppc_exploratory = _aggregate_strategy_runs(ppc_holdout_runs, PPC_EXPLORATORY)

    ekf = _find_row(urbannav_summary, "method", "EKF")
    pf = _find_row(urbannav_summary, "method", "PF-10K")
    robust = _find_row(urbannav_summary, "method", "PF+RobustClear-10K")
    wls_qv = _find_row(urbannav_summary, "method", "WLS+QualityVeto")

    hk_summary = _read_csv(HK_ADAPTIVE_SUMMARY_CSV)
    hk_ekf = _find_row(hk_summary, "method", "EKF")
    hk_adaptive = _find_row(hk_summary, "method", "PF+AdaptiveGuide-10K")

    pf3d = _find_row(bvh_rows, "method", "PF3D-10K")
    pf3d_bvh = _find_row(bvh_rows, "method", "PF3D-BVH-10K")

    return [
        {
            "section": "PPC holdout",
            "method": "Safe baseline",
            "rms_2d_m": _round(ppc_safe["mean_rms_2d"]),
            "p95_m": _round(ppc_safe["mean_p95"]),
            "outlier_rate_pct": _round(ppc_safe["mean_outlier_rate_pct"]),
            "catastrophic_rate_pct": _round(ppc_safe["mean_catastrophic_rate_pct"], 3),
            "time_ms_per_epoch": "",
            "note": "always_robust",
        },
        {
            "section": "PPC holdout",
            "method": "Exploratory gate",
            "rms_2d_m": _round(ppc_exploratory["mean_rms_2d"]),
            "p95_m": _round(ppc_exploratory["mean_p95"]),
            "outlier_rate_pct": _round(ppc_exploratory["mean_outlier_rate_pct"]),
            "catastrophic_rate_pct": _round(ppc_exploratory["mean_catastrophic_rate_pct"], 3),
            "time_ms_per_epoch": "",
            "note": "entry_veto_negative_exit_rescue...",
        },
        {
            "section": "UrbanNav external",
            "method": "EKF",
            "rms_2d_m": _round(_f(ekf, "mean_rms_2d")),
            "p95_m": _round(_f(ekf, "mean_p95")),
            "outlier_rate_pct": _round(_f(ekf, "mean_outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(ekf, "mean_catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(ekf, "mean_time_ms_per_epoch"), 3),
            "note": "trimble + G,E,J",
        },
        {
            "section": "UrbanNav external",
            "method": "PF-10K",
            "rms_2d_m": _round(_f(pf, "mean_rms_2d")),
            "p95_m": _round(_f(pf, "mean_p95")),
            "outlier_rate_pct": _round(_f(pf, "mean_outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(pf, "mean_catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(pf, "mean_time_ms_per_epoch"), 3),
            "note": "trimble + G,E,J",
        },
        {
            "section": "UrbanNav external",
            "method": "PF+RobustClear-10K",
            "rms_2d_m": _round(_f(robust, "mean_rms_2d")),
            "p95_m": _round(_f(robust, "mean_p95")),
            "outlier_rate_pct": _round(_f(robust, "mean_outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(robust, "mean_catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(robust, "mean_time_ms_per_epoch"), 3),
            "note": "trimble + G,E,J",
        },
        {
            "section": "UrbanNav external",
            "method": "WLS+QualityVeto",
            "rms_2d_m": _round(_f(wls_qv, "mean_rms_2d")),
            "p95_m": _round(_f(wls_qv, "mean_p95")),
            "outlier_rate_pct": _round(_f(wls_qv, "mean_outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(wls_qv, "mean_catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(wls_qv, "mean_time_ms_per_epoch"), 3),
            "note": "promoted core hook",
        },
        {
            "section": "HK supplemental",
            "method": "EKF",
            "rms_2d_m": _round(_f(hk_ekf, "mean_rms_2d")),
            "p95_m": _round(_f(hk_ekf, "mean_p95")),
            "outlier_rate_pct": _round(_f(hk_ekf, "mean_outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(hk_ekf, "mean_catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(hk_ekf, "mean_time_ms_per_epoch"), 3),
            "note": "ublox + G (GPS-only)",
        },
        {
            "section": "HK supplemental",
            "method": "PF+AdaptiveGuide-10K",
            "rms_2d_m": _round(_f(hk_adaptive, "mean_rms_2d")),
            "p95_m": _round(_f(hk_adaptive, "mean_p95")),
            "outlier_rate_pct": _round(_f(hk_adaptive, "mean_outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(hk_adaptive, "mean_catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(hk_adaptive, "mean_time_ms_per_epoch"), 3),
            "note": "ublox + G,C (adaptive guide)",
        },
        {
            "section": "BVH systems",
            "method": "PF3D-10K",
            "rms_2d_m": _round(_f(pf3d, "rms_2d")),
            "p95_m": _round(_f(pf3d, "p95")),
            "outlier_rate_pct": _round(_f(pf3d, "outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(pf3d, "catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(pf3d, "time_ms"), 2),
            "note": "real PLATEAU subset",
        },
        {
            "section": "BVH systems",
            "method": "PF3D-BVH-10K",
            "rms_2d_m": _round(_f(pf3d_bvh, "rms_2d")),
            "p95_m": _round(_f(pf3d_bvh, "p95")),
            "outlier_rate_pct": _round(_f(pf3d_bvh, "outlier_rate_pct")),
            "catastrophic_rate_pct": _round(_f(pf3d_bvh, "catastrophic_rate_pct"), 3),
            "time_ms_per_epoch": _round(_f(pf3d_bvh, "time_ms"), 2),
            "note": "57.8x faster",
        },
    ]


def _write_captions(path: Path) -> None:
    ppc_holdout_runs = _read_csv(PPC_HOLDOUT_RUNS_CSV)
    urbannav_summary = _read_csv(URBANNAV_SUMMARY_CSV)
    bvh_rows = _read_csv(BVH_RUNTIME_CSV)

    ppc_safe = _aggregate_strategy_runs(ppc_holdout_runs, PPC_SAFE)
    ppc_exploratory = _aggregate_strategy_runs(ppc_holdout_runs, PPC_EXPLORATORY)

    ekf = _find_row(urbannav_summary, "method", "EKF")
    pf = _find_row(urbannav_summary, "method", "PF-10K")
    robust = _find_row(urbannav_summary, "method", "PF+RobustClear-10K")
    wls_qv = _find_row(urbannav_summary, "method", "WLS+QualityVeto")

    pf3d = _find_row(bvh_rows, "method", "PF3D-10K")
    pf3d_bvh = _find_row(bvh_rows, "method", "PF3D-BVH-10K")
    bvh_speedup = _f(pf3d, "time_ms") / _f(pf3d_bvh, "time_ms")

    lines = [
        "# Paper Captions",
        "",
        "## Table 1",
        (
            "Main quantitative summary used in the paper. PPC holdout is reported as a"
            " design-discipline result rather than a headline accuracy claim. UrbanNav"
            " external uses fixed `trimble + G,E,J` settings without UrbanNav-specific"
            " retuning. `PF+RobustClear-10K` is the strongest external method, improving"
            f" mean RMS horizontal error from {_f(ekf, 'mean_rms_2d'):.2f} m (`EKF`) to"
            f" {_f(robust, 'mean_rms_2d'):.2f} m and mean p95 from"
            f" {_f(ekf, 'mean_p95'):.2f} m to {_f(robust, 'mean_p95'):.2f} m while reducing"
            f" the >100 m rate from {_f(ekf, 'mean_outlier_rate_pct'):.2f}% to"
            f" {_f(robust, 'mean_outlier_rate_pct'):.2f}% and the >500 m rate from"
            f" {_f(ekf, 'mean_catastrophic_rate_pct'):.3f}% to"
            f" {_f(robust, 'mean_catastrophic_rate_pct'):.3f}%. `WLS+QualityVeto` is shown"
            " as a promoted core utility, not as the main external method. BVH systems"
            " rows isolate runtime on a real PLATEAU subset and show unchanged PF3D"
            " accuracy with large acceleration."
        ),
        "",
        "## Figure 1",
        (
            "Segment-wise PPC holdout comparison between the safe baseline `always_robust`"
            " and the best exploratory gate"
            " `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`."
            f" The gain survives holdout but remains modest: mean RMS decreases from"
            f" {ppc_safe['mean_rms_2d']:.2f} m to {ppc_exploratory['mean_rms_2d']:.2f} m"
            f" and mean p95 decreases from {ppc_safe['mean_p95']:.2f} m to"
            f" {ppc_exploratory['mean_p95']:.2f} m. This figure should be used to support"
            " the paper's experiment-discipline claim, not as the main accuracy result."
        ),
        "",
        "## Figure 2",
        (
            "UrbanNav external validation on Odaiba and Shinjuku using trimble observations"
            " and `G,E,J` measurements. Left: empirical CDF of 2D error. Right: rates of"
            " large failures above 100 m and catastrophic failures above 500 m."
            f" `PF+RobustClear-10K` achieves {_f(robust, 'mean_rms_2d'):.2f} m mean RMS and"
            f" {_f(robust, 'mean_p95'):.2f} m mean p95, outperforming `EKF` at"
            f" {_f(ekf, 'mean_rms_2d'):.2f} m and {_f(ekf, 'mean_p95'):.2f} m."
            f" Relative to `EKF`, the >100 m rate falls from"
            f" {_f(ekf, 'mean_outlier_rate_pct'):.2f}% to"
            f" {_f(robust, 'mean_outlier_rate_pct'):.2f}% and the >500 m rate falls from"
            f" {_f(ekf, 'mean_catastrophic_rate_pct'):.3f}% to"
            f" {_f(robust, 'mean_catastrophic_rate_pct'):.3f}%. `PF-10K` follows closely,"
            " while `WLS+QualityVeto` improves raw multi-GNSS WLS tails but remains far"
            " worse in RMS."
        ),
        "",
        "## Figure 3",
        (
            "Runtime and accuracy on the real PLATEAU PF3D subset. `PF3D-BVH-10K` preserves"
            f" the same accuracy as `PF3D-10K` ({_f(pf3d, 'rms_2d'):.2f} m RMS,"
            f" {_f(pf3d, 'p95'):.2f} m p95) while reducing runtime from"
            f" {_f(pf3d, 'time_ms'):.2f} ms/epoch to {_f(pf3d_bvh, 'time_ms'):.2f} ms/epoch,"
            f" a {_round(bvh_speedup, 1):.1f}x speedup. This figure should be framed as a"
            " systems contribution rather than a real-data accuracy gain from explicit"
            " 3D reasoning."
        ),
        "",
        "## In-Text Placement",
        "",
        "- Table 1: first paragraph of the Results section as the paper-wide summary.",
        "- Figure 1: PPC holdout ablation subsection.",
        "- Figure 2: UrbanNav external validation subsection as the main accuracy figure.",
        "- Figure 3: systems / implementation subsection for the BVH result.",
        "",
        "## Supporting Numbers",
        "",
        f"- `PF-10K` UrbanNav external mean RMS/p95: {_f(pf, 'mean_rms_2d'):.2f} / {_f(pf, 'mean_p95'):.2f} m.",
        f"- `WLS+QualityVeto` UrbanNav external mean RMS/p95: {_f(wls_qv, 'mean_rms_2d'):.2f} / {_f(wls_qv, 'mean_p95'):.2f} m.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_ppc_holdout(output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(PPC_HOLDOUT_RUNS_CSV)
    safe = {row["segment_label"]: row for row in rows if row["strategy"] == PPC_SAFE}
    best = {row["segment_label"]: row for row in rows if row["strategy"] == PPC_EXPLORATORY}
    segments = sorted(set(safe) & set(best))
    x = np.arange(len(segments))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, metric, title in zip(
        axes,
        ("rms_2d", "p95"),
        ("PPC Holdout RMS 2D", "PPC Holdout P95"),
        strict=True,
    ):
        safe_vals = np.array([_f(safe[seg], metric) for seg in segments], dtype=np.float64)
        best_vals = np.array([_f(best[seg], metric) for seg in segments], dtype=np.float64)
        for idx, seg in enumerate(segments):
            color = "#147a73" if best_vals[idx] <= safe_vals[idx] else "#b84949"
            ax.plot([0, 1], [safe_vals[idx], best_vals[idx]], color=color, alpha=0.6, linewidth=1.4)
            ax.scatter([0, 1], [safe_vals[idx], best_vals[idx]], color=color, s=18)
        ax.scatter(
            [0, 1],
            [float(np.mean(safe_vals)), float(np.mean(best_vals))],
            color=["#1f4e79", "#d97706"],
            s=90,
            marker="D",
            zorder=4,
        )
        ax.set_xticks([0, 1], ["always_robust", "exploratory"])
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("meters")
    fig.suptitle("PPC Holdout: segment-wise comparison", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _collect_urbannav_epoch_rows(prefix: str) -> list[dict[str, str]]:
    combined = RESULTS_DIR / f"{prefix}_epochs.csv"
    if combined.exists():
        with combined.open(newline="") as fh:
            return list(csv.DictReader(fh))

    rows: list[dict[str, str]] = []
    for path in sorted(RESULTS_DIR.glob(f"{prefix}__*__*_epochs.csv")):
        with path.open(newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _plot_urbannav_external(output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_rows = _read_csv(URBANNAV_SUMMARY_CSV)
    methods = ["WLS+QualityVeto", "EKF", "PF-10K", "PF+RobustClear-10K"]
    labels = {
        "WLS+QualityVeto": "WLS+QV",
        "EKF": "EKF",
        "PF-10K": "PF-10K",
        "PF+RobustClear-10K": "PF+RobustClear",
    }
    colors = {
        "WLS+QualityVeto": "#9f7aea",
        "EKF": "#3b82f6",
        "PF-10K": "#f97316",
        "PF+RobustClear-10K": "#059669",
    }
    epoch_rows = _collect_urbannav_epoch_rows(URBANNAV_EPOCHS_PREFIX)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    cdf_ax, tail_ax = axes

    if epoch_rows:
        errors_dict = {}
        for method in methods:
            errors = [
                float(row["error_2d"])
                for row in epoch_rows
                if row["method"] == method
            ]
            if errors:
                errors_dict[f"{labels[method]} ({_round(_f(_find_row(summary_rows, 'method', method), 'mean_rms_2d'))} m)"] = np.array(
                    errors,
                    dtype=np.float64,
                )
        plot_cdf(errors_dict, output_path.with_suffix(".tmp.png"), title="tmp")
        tmp_path = output_path.with_suffix(".tmp.png")
        if tmp_path.exists():
            tmp_path.unlink()
        for label, errors in errors_dict.items():
            errs = np.sort(errors)
            cdf = np.arange(1, len(errs) + 1) / len(errs)
            method = next(k for k, v in labels.items() if label.startswith(v))
            cdf_ax.plot(errs, cdf * 100.0, linewidth=1.8, label=label, color=colors[method])
        cdf_ax.set_xlabel("2D error [m]")
        cdf_ax.set_ylabel("CDF [%]")
        cdf_ax.set_title("UrbanNav external CDF")
        cdf_ax.set_xlim(left=0.0, right=300.0)
        cdf_ax.set_ylim(0.0, 100.0)
        cdf_ax.grid(True, alpha=0.25)
        cdf_ax.legend(fontsize=8)
    else:
        cdf_ax.text(0.5, 0.5, "epoch dump not found", ha="center", va="center")
        cdf_ax.set_axis_off()

    x = np.arange(len(methods))
    width = 0.38
    out100 = [_f(_find_row(summary_rows, "method", method), "mean_outlier_rate_pct") for method in methods]
    out500 = [_f(_find_row(summary_rows, "method", method), "mean_catastrophic_rate_pct") for method in methods]
    tail_ax.bar(x - width / 2, out100, width, label=">100 m", color="#ef4444")
    tail_ax.bar(x + width / 2, out500, width, label=">500 m", color="#111827")
    tail_ax.set_xticks(x, [labels[m] for m in methods], rotation=15)
    tail_ax.set_ylabel("rate [%]")
    tail_ax.set_title("UrbanNav external tail")
    tail_ax.grid(True, axis="y", alpha=0.25)
    tail_ax.legend()

    fig.suptitle("UrbanNav external validation: trimble + G,E,J", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_bvh_runtime(output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(BVH_RUNTIME_CSV)
    methods = ["PF3D-10K", "PF3D-BVH-10K"]
    labels = ["PF3D", "PF3D-BVH"]
    runtime = [_f(_find_row(rows, "method", method), "time_ms") for method in methods]
    rms = [_f(_find_row(rows, "method", method), "rms_2d") for method in methods]
    p95 = [_f(_find_row(rows, "method", method), "p95") for method in methods]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax_rt, ax_acc = axes

    ax_rt.bar(labels, runtime, color=["#d97706", "#0f766e"])
    ax_rt.set_yscale("log")
    ax_rt.set_ylabel("time [ms/epoch]")
    ax_rt.set_title("Real-PLATEAU runtime")
    ax_rt.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(runtime):
        ax_rt.text(idx, value * 1.1, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    x = np.arange(len(labels))
    width = 0.36
    ax_acc.bar(x - width / 2, rms, width, label="RMS 2D", color="#60a5fa")
    ax_acc.bar(x + width / 2, p95, width, label="P95", color="#34d399")
    ax_acc.set_xticks(x, labels)
    ax_acc.set_ylabel("meters")
    ax_acc.set_title("Accuracy stays unchanged")
    ax_acc.grid(True, axis="y", alpha=0.25)
    ax_acc.legend()

    fig.suptitle("BVH acceleration on real PLATEAU subset", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_particle_scaling(output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Scaling experiment results: trimble + G,E,J
    # (N, RMS 2D, P95, >100m rate)
    odaiba_data = [
        (100, 135.88, 264.19, 46.01),
        (500, 82.27, 153.98, 17.00),
        (1_000, 70.59, 115.92, 11.64),
        (5_000, 62.48, 96.50, 3.51),
        (10_000, 61.86, 95.72, 3.31),
        (50_000, 62.19, 90.49, 2.73),
        (100_000, 60.87, 86.58, 2.10),
        (500_000, 60.65, 84.84, 2.00),
        (1_000_000, 60.40, 84.47, 1.97),
    ]
    shinjuku_data = [
        (100, 120.17, 242.63, 36.00),
        (500, 82.41, 141.52, 14.50),
        (1_000, 78.46, 124.97, 10.49),
        (5_000, 70.82, 110.24, 7.69),
        (10_000, 71.72, 107.39, 7.46),
        (50_000, 71.11, 101.71, 5.47),
        (100_000, 71.30, 98.75, 4.49),
        (1_000_000, 73.26, 98.81, 4.49),
    ]
    ekf_odaiba = (89.42, 151.43, 14.23)
    ekf_shinjuku = (97.07, 155.93, None)  # P95/outlier from per-run data

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    metrics = ("RMS 2D [m]", "P95 [m]", ">100 m rate [%]")
    titles = ("Mean RMS 2D", "Mean P95", "Failure rate >100 m")

    for col, (ax, ylabel, title) in enumerate(zip(axes, metrics, titles, strict=True)):
        for data, ekf, label, color, marker in [
            (odaiba_data, ekf_odaiba, "Odaiba", "#059669", "o"),
            (shinjuku_data, ekf_shinjuku, "Shinjuku", "#7c3aed", "s"),
        ]:
            ns = [d[0] for d in data]
            vals = [d[col + 1] for d in data]
            ax.plot(ns, vals, f"{marker}-", color=color, linewidth=1.8,
                    markersize=5, label=f"PF {label}")
            if ekf[col] is not None:
                ax.axhline(ekf[col], color=color, linestyle="--",
                           linewidth=1.2, alpha=0.5, label=f"EKF {label}")
        ax.set_xscale("log")
        ax.set_xlabel("N particles")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        ax.axvspan(500, 2000, alpha=0.06, color="#f59e0b")

    fig.suptitle(
        "Particle count scaling on UrbanNav Tokyo (trimble + G,E,J)",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_rows = _paper_main_table()
    _write_csv(table_rows, OUTPUT_DIR / "paper_main_table.csv")
    _write_markdown(table_rows, OUTPUT_DIR / "paper_main_table.md")
    _write_captions(OUTPUT_DIR / "paper_captions.md")
    _plot_ppc_holdout(OUTPUT_DIR / "paper_ppc_holdout.png")
    _plot_urbannav_external(OUTPUT_DIR / "paper_urbannav_external.png")
    _plot_bvh_runtime(OUTPUT_DIR / "paper_bvh_runtime.png")
    _plot_particle_scaling(OUTPUT_DIR / "paper_particle_scaling.png")
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
