#!/usr/bin/env python3
"""Build a GitHub Pages-friendly summary snapshot from experiment CSVs."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PAPER_ASSETS_DIR = RESULTS_DIR / "paper_assets"
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
DATA_DIR = ASSETS_DIR / "data"
FIGURES_DIR = ASSETS_DIR / "figures"
MEDIA_DIR = ASSETS_DIR / "media"
SNAPSHOT_PATH = ASSETS_DIR / "results_snapshot.json"
SNAPSHOT_JS_PATH = ASSETS_DIR / "results_snapshot.js"

PPC_TUNED_CSV = "pf_strategy_lab_positive6_summary.csv"
PPC_HOLDOUT_CSV = "pf_strategy_lab_holdout6_r200_s200_summary.csv"
URBANNAV_SUMMARY_CSV = "urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv"
URBANNAV_RUNS_CSV = "urbannav_fixed_eval_external_gej_trimble_qualityveto_runs.csv"
URBANNAV_WINDOW_SUMMARY_CSV = (
    "urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv"
)
URBANNAV_HK_ADAPTIVE_CSV = "urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv"
BVH_RUNTIME_CSV = "ppc_pf3d_tokyo_run1_g_100_plateau_summary.csv"

PAPER_MAIN_TABLE_CSV = "paper_main_table.csv"
VALIDATION_SUMMARY_JSON = RESULTS_DIR / "freeze_validation_summary.json"
SITE_MEDIA = {}
SITE_VIDEO = None
SITE_CHARTS = {
    "site_urbannav_runs.png": {
        "title": "UrbanNav Per-Run Comparison",
        "caption": "Odaiba and Shinjuku side-by-side for EKF, PF-10K, and PF+RobustClear-10K.",
    },
    "site_window_wins.png": {
        "title": "Window Win Rates",
        "caption": "Fixed-window win rates against EKF on UrbanNav Tokyo.",
    },
    "site_hk_control.png": {
        "title": "Hong Kong Control",
        "caption": "Cross-geometry control check showing why adaptive guidance stays supplemental.",
    },
    "site_urbannav_timeline.png": {
        "title": "Epoch Error Timeline",
        "caption": "Smoothed Odaiba and Shinjuku traces make the PF-vs-EKF gap visible over time.",
    },
    "site_error_bands.png": {
        "title": "Error-Band Composition",
        "caption": "Epoch share in <25 m, 25-50 m, 50-100 m, 100-500 m, and >500 m bands.",
    },
}
PAPER_FIGURES = {
    "paper_urbannav_external.png": {
        "title": "UrbanNav External",
        "caption": (
            "Main accuracy figure. `PF+RobustClear-10K` is the frozen external winner "
            "on trimble + G,E,J without UrbanNav-specific retuning."
        ),
    },
    "paper_particle_scaling.png": {
        "title": "Particle Scaling",
        "caption": (
            "Phase transition at N~1,000: PF crosses EKF. RMS saturates near N=5K, "
            "but >100m failure rate continues to improve up to 1M particles."
        ),
    },
    "paper_bvh_runtime.png": {
        "title": "BVH Runtime",
        "caption": (
            "Systems figure. `PF3D-BVH-10K` preserves PF3D accuracy on the real "
            "PLATEAU subset while cutting runtime by 57.8x."
        ),
    },
    "paper_ppc_holdout.png": {
        "title": "PPC Holdout",
        "caption": (
            "Holdout gain survives but stays modest. This is the design-discipline "
            "figure, not the headline accuracy claim."
        ),
    },
}

PPC_SAFE = "always_robust"
PPC_EXPLORATORY = (
    "entry_veto_negative_exit_rescue_branch_aware_"
    "hysteresis_quality_veto_regime_gate"
)


def _read_csv(name: str) -> list[dict[str, str]]:
    return _read_csv_path(RESULTS_DIR / name)


def _read_csv_path(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _copy_file(src: Path, dst_dir: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst.relative_to(DOCS_DIR)).replace("\\", "/")


def _copy_data_file(name: str) -> str:
    return _copy_file(RESULTS_DIR / name, DATA_DIR)


def _copy_paper_data_file(name: str) -> str:
    return _copy_file(PAPER_ASSETS_DIR / name, DATA_DIR)


def _copy_figure(name: str) -> str:
    return _copy_file(PAPER_ASSETS_DIR / name, FIGURES_DIR)


def _media_href(name: str) -> str:
    return f"assets/media/{name}"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"row not found: {key}={value}")


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _round(v: float, digits: int = 2) -> float:
    return round(float(v), digits)


def _card(label: str, value: str, detail: str) -> dict[str, str]:
    return {"label": label, "value": value, "detail": detail}


def _ensure_paper_assets() -> None:
    required = [
        PAPER_ASSETS_DIR / PAPER_MAIN_TABLE_CSV,
        *(PAPER_ASSETS_DIR / name for name in PAPER_FIGURES),
    ]
    if all(path.exists() for path in required):
        latest_input = max([
            (RESULTS_DIR / name).stat().st_mtime
            for name in (
                PPC_TUNED_CSV,
                PPC_HOLDOUT_CSV,
                URBANNAV_SUMMARY_CSV,
                URBANNAV_RUNS_CSV,
                BVH_RUNTIME_CSV,
            )
            if (RESULTS_DIR / name).exists()
        ] + [(PROJECT_ROOT / "experiments" / "build_paper_assets.py").stat().st_mtime])
        oldest_output = min(path.stat().st_mtime for path in required)
        if oldest_output >= latest_input:
            return

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "build_paper_assets.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _ensure_site_media() -> None:
    required = [
        MEDIA_DIR / name
        for name in (
            *SITE_MEDIA,
            *SITE_CHARTS,
        )
    ]
    if all(path.exists() for path in required):
        latest_input = max([
            (PAPER_ASSETS_DIR / name).stat().st_mtime
            for name in PAPER_FIGURES
            if (PAPER_ASSETS_DIR / name).exists()
        ] + [(PROJECT_ROOT / "experiments" / "build_site_media.py").stat().st_mtime])
        oldest_output = min(path.stat().st_mtime for path in required)
        if oldest_output >= latest_input:
            return

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments" / "build_site_media.py")],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _build_snapshot() -> dict:
    _ensure_paper_assets()
    _ensure_site_media()

    ppc_tuned_rows = _read_csv(PPC_TUNED_CSV)
    ppc_holdout_rows = _read_csv(PPC_HOLDOUT_CSV)
    urbannav_summary_rows = _read_csv(URBANNAV_SUMMARY_CSV)
    urbannav_run_rows = _read_csv(URBANNAV_RUNS_CSV)
    urbannav_window_rows = _read_csv(URBANNAV_WINDOW_SUMMARY_CSV)
    hk_adaptive_rows = _read_csv(URBANNAV_HK_ADAPTIVE_CSV)
    bvh_rows = _read_csv(BVH_RUNTIME_CSV)
    paper_main_rows = _read_csv_path(PAPER_ASSETS_DIR / PAPER_MAIN_TABLE_CSV)
    validation = _read_json(VALIDATION_SUMMARY_JSON)

    tuned_safe = _find_row(ppc_tuned_rows, "strategy", PPC_SAFE)
    tuned_best = _find_row(ppc_tuned_rows, "strategy", PPC_EXPLORATORY)
    holdout_safe = _find_row(ppc_holdout_rows, "strategy", PPC_SAFE)
    holdout_best = _find_row(ppc_holdout_rows, "strategy", PPC_EXPLORATORY)

    ekf = _find_row(urbannav_summary_rows, "method", "EKF")
    wls = _find_row(urbannav_summary_rows, "method", "WLS")
    wls_qv = _find_row(urbannav_summary_rows, "method", "WLS+QualityVeto")
    pf = _find_row(urbannav_summary_rows, "method", "PF-10K")
    robust = _find_row(urbannav_summary_rows, "method", "PF+RobustClear-10K")

    window_ekf = _find_row(urbannav_window_rows, "method", "EKF")
    window_pf = _find_row(urbannav_window_rows, "method", "PF+RobustClear-10K")
    hk_ekf = _find_row(hk_adaptive_rows, "method", "EKF")
    hk_adaptive = _find_row(hk_adaptive_rows, "method", "PF+AdaptiveGuide-10K")

    pf3d = _find_row(bvh_rows, "method", "PF3D-10K")
    pf3d_bvh = _find_row(bvh_rows, "method", "PF3D-BVH-10K")
    bvh_speedup = _f(pf3d, "time_ms") / _f(pf3d_bvh, "time_ms")

    figure_cards = []
    figure_links = {}
    for name, meta in PAPER_FIGURES.items():
        rel_path = _copy_figure(name)
        figure_links[name] = rel_path
        figure_cards.append(
            {
                "title": meta["title"],
                "image": rel_path,
                "href": rel_path,
                "caption": meta["caption"],
                "alt": meta["title"],
            }
        )

    media_cards = []
    for viz_name, viz_title, viz_caption in [
        ("particle_viz_odaiba.mp4", "Odaiba Particle Cloud",
         "100K particles on OpenStreetMap (full + zoom). Orange: particles. Red: PF estimate. Blue: ground truth."),
        ("particle_viz_shinjuku.mp4", "Shinjuku Particle Cloud",
         "100K particles in deep urban Shinjuku (full + zoom). Watch the particle spread in canyon sections."),
    ]:
        if (MEDIA_DIR / viz_name).exists():
            media_cards.append({
                "kind": "video",
                "title": viz_title,
                "href": _media_href(viz_name),
                "caption": viz_caption,
                "poster": "",
                "sources": [{"src": _media_href(viz_name), "type": "video/mp4"}],
            })

    chart_cards = []
    for name, meta in SITE_CHARTS.items():
        rel_path = _media_href(name)
        chart_cards.append(
            {
                "kind": "image",
                "title": meta["title"],
                "image": rel_path,
                "href": rel_path,
                "caption": meta["caption"],
                "alt": meta["title"],
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": "gnss_gpu Artifact Snapshot",
        "subtitle": (
            "Mainline frozen on `PF+RobustClear-10K`. PF beats EKF across "
            "5 sequences in 2 cities. Particle scaling reveals a phase transition "
            "at N~1,000 with tail improvement up to 1M."
        ),
        "status": {
            "label": "Current Read",
            "value": "PF beats RTKLIB demo5 — RMS 6.72m vs 13.08m",
            "detail": (
                "With gnssplusplus-corrected pseudoranges, PF (1M particles) achieves "
                "RMS 6.72m on Odaiba — 49% better than RTKLIB demo5 (13.08m), "
                "59% better P95, zero catastrophic failures. "
                "PF family outperforms EKF on all 5 evaluated sequences (Tokyo + Hong Kong). "
                "24 cited references, gnssplusplus as submodule for GNSS corrections."
            ),
        },
        "hero_cards": [
            _card(
                "Frozen Mainline",
                "PF+RobustClear-10K",
                "External winner on UrbanNav trimble + G,E,J.",
            ),
            _card(
                "UrbanNav vs EKF",
                (
                    f"{_round(_f(ekf, 'mean_rms_2d'))} -> "
                    f"{_round(_f(robust, 'mean_rms_2d'))} m"
                ),
                "Mean RMS 2D across Odaiba + Shinjuku.",
            ),
            _card(
                "Window Wins",
                (
                    f"{int(_f(window_pf, 'wins_rms'))}/"
                    f"{int(_f(window_pf, 'n_windows'))} RMS wins"
                ),
                "Fixed 500-epoch windows against EKF on UrbanNav Tokyo.",
            ),
            _card(
                "HK Breadth",
                (
                    f"{_round(_f(hk_ekf, 'mean_rms_2d'))} -> "
                    f"{_round(_f(hk_adaptive, 'mean_rms_2d'))} m"
                ),
                "PF+AdaptiveGuide beats EKF on all 3 Hong Kong sequences (G+C).",
            ),
            _card(
                "Scaling Phase Transition",
                "N ~ 1,000",
                "PF crosses EKF at ~1K particles. Tail improves to 1M.",
            ),
            _card(
                "PF vs RTKLIB demo5",
                "RMS 6.72 vs 13.08 m",
                "PF (1M particles + gnssplusplus corrections) beats RTKLIB demo5 by 49% in RMS, 59% in P95, with zero >100m failures.",
            ),
            _card(
                "BVH Speedup",
                f"{_round(bvh_speedup, 1)}x",
                "PF3D vs PF3D-BVH on the real PLATEAU subset.",
            ),
        ],
        "repo_summary": [
            "This repo is not presenting a single heroic algorithm. It is an experiment-first GNSS package where comparable variants are built, measured, and either frozen or discarded.",
            "The accuracy headline is the UrbanNav external result with trimble + G,E,J. The PPC gate family remains useful as a design-discipline story, but not as the main empirical claim.",
            "The 3D PF path is currently a systems contribution: BVH preserves PF3D accuracy on a real PLATEAU subset while making runtime practical.",
        ],
        "quick_links": [
            {
                "label": "Paper Main Table",
                "href": _copy_paper_data_file(PAPER_MAIN_TABLE_CSV),
                "detail": "Paper-ready CSV summary copied from `experiments/results/paper_assets/`.",
            },
            {
                "label": "UrbanNav Summary CSV",
                "href": _copy_data_file(URBANNAV_SUMMARY_CSV),
                "detail": "Frozen external result for trimble + G,E,J.",
            },
            {
                "label": "UrbanNav Runs CSV",
                "href": _copy_data_file(URBANNAV_RUNS_CSV),
                "detail": "Per-run Odaiba and Shinjuku metrics.",
            },
            {
                "label": "PPC Holdout Figure",
                "href": figure_links["paper_ppc_holdout.png"],
                "detail": "Holdout comparison between `always_robust` and the exploratory gate.",
            },
            {
                "label": "UrbanNav Figure",
                "href": figure_links["paper_urbannav_external.png"],
                "detail": "CDF and tail summary for the main external result.",
            },
            {
                "label": "BVH Figure",
                "href": figure_links["paper_bvh_runtime.png"],
                "detail": "Runtime and accuracy on the real PLATEAU subset.",
            },
            {
                "label": "Snapshot JSON",
                "href": "assets/results_snapshot.json",
                "detail": "Machine-readable version of this page.",
            },
        ],
        "method_freeze": [
            _card(
                "Mainline",
                "PF+RobustClear-10K",
                (
                    "Best full-run external method on UrbanNav Tokyo. "
                    "This is the number that drives the README, Pages, and paper assets."
                ),
            ),
            _card(
                "Exploratory PPC Gate",
                "entry_veto_negative_exit_...",
                (
                    f"Holdout RMS improves from {_round(_f(holdout_safe, 'mean_rms_2d'))} "
                    f"to {_round(_f(holdout_best, 'mean_rms_2d'))} m, but the gain stays small."
                ),
            ),
            _card(
                "HK Supplemental Winner",
                "PF+AdaptiveGuide-10K",
                (
                    "Beats EKF on all 3 Hong Kong sequences with GPS+BeiDou. "
                    "Different config from Tokyo mainline, but PF family wins in both cities."
                ),
            ),
            _card(
                "Promoted Core Hook",
                "WLS+QualityVeto",
                (
                    "A reusable multi-GNSS stabilization hook in core code. "
                    "It improves raw multi-WLS tails but is not the final best method."
                ),
            ),
        ],
        "showcase_media": media_cards,
        "analysis_charts": chart_cards,
        "featured_figures": figure_cards,
        "supplemental_checks": [
            (
                "UrbanNav fixed-window check: `PF+RobustClear-10K` beats `EKF` in "
                f"{int(_f(window_pf, 'wins_rms'))}/{int(_f(window_pf, 'n_windows'))} "
                "RMS windows and "
                f"{int(_f(window_pf, 'wins_p95'))}/{int(_f(window_pf, 'n_windows'))} "
                "P95 windows."
            ),
            (
                "Hong Kong supplemental: `PF+AdaptiveGuide-10K` with GPS+BeiDou achieves "
                f"{_round(_f(hk_adaptive, 'mean_rms_2d'))} m RMS, beating GPS-only `EKF` "
                f"at {_round(_f(hk_ekf, 'mean_rms_2d'))} m — the PF family wins in a "
                "second urban geometry."
            ),
            (
                "Adaptive guide helps cross-geometry robustness and Hong Kong recovery, "
                "but full Tokyo runs still leave it behind `PF+RobustClear-10K`, so it "
                "stays supplemental rather than mainline."
            ),
        ],
        "repo_map": [
            {
                "path": "python/gnss_gpu/",
                "detail": (
                    "Core library code, pybind wrappers, reusable hooks like "
                    "`multi_gnss_quality.py`, and the main Python API."
                ),
            },
            {
                "path": "experiments/",
                "detail": (
                    "Discardable experiment code, evaluators, dataset adapters, and "
                    "artifact builders such as `build_paper_assets.py` and "
                    "`build_githubio_summary.py`."
                ),
            },
            {
                "path": "docs/experiments.md",
                "detail": "What was run, on which split, and what the result actually was.",
            },
            {
                "path": "docs/decisions.md",
                "detail": "Why each variant was adopted, kept as supplemental, or rejected.",
            },
            {
                "path": "docs/interfaces.md",
                "detail": "The minimal interface kept after comparison and pruning.",
            },
            {
                "path": "docs/index.html",
                "detail": "This GitHub Pages front door for artifact readers.",
            },
        ],
        "reproduce_commands": [
            "python3 experiments/build_paper_assets.py",
            "python3 experiments/build_githubio_summary.py",
            "PYTHONPATH=python python3 -m pytest tests/ -q",
        ],
        "takeaways": [
            (
                "PPC tuned split: the exploratory gate trims mean RMS from "
                f"{_round(_f(tuned_safe, 'mean_rms_2d'))} m to "
                f"{_round(_f(tuned_best, 'mean_rms_2d'))} m, but mean P95 stays "
                f"flat at {_round(_f(tuned_best, 'mean_p95'))} m."
            ),
            (
                "PPC holdout split: the same gate survives holdout with "
                f"{_round(_f(holdout_safe, 'mean_rms_2d'))} -> "
                f"{_round(_f(holdout_best, 'mean_rms_2d'))} m RMS and "
                f"{_round(_f(holdout_safe, 'mean_p95'))} -> "
                f"{_round(_f(holdout_best, 'mean_p95'))} m P95."
            ),
            (
                "UrbanNav external validation: `PF+RobustClear-10K` is the frozen winner "
                f"at {_round(_f(robust, 'mean_rms_2d'))} m RMS and "
                f"{_round(_f(robust, 'mean_p95'))} m P95, ahead of `EKF` at "
                f"{_round(_f(ekf, 'mean_rms_2d'))} m and {_round(_f(ekf, 'mean_p95'))} m."
            ),
            (
                "`PF-10K` remains a close ablation at "
                f"{_round(_f(pf, 'mean_rms_2d'))} m RMS, which keeps the robust-clear "
                "story honest: the gain is real but not huge."
            ),
            (
                "`WLS+QualityVeto` is useful because it moved a stabilization policy into "
                "reusable core code, not because it became the best external method."
            ),
            (
                "Systems result: `PF3D-BVH-10K` keeps PF3D accuracy while cutting runtime "
                f"from {_round(_f(pf3d, 'time_ms'))} to {_round(_f(pf3d_bvh, 'time_ms'))} "
                "ms/epoch."
            ),
            (
                "Hong Kong supplemental: `PF+AdaptiveGuide-10K` with GPS+BeiDou achieves "
                f"{_round(_f(hk_adaptive, 'mean_rms_2d'))} m RMS, outperforming GPS-only "
                f"`EKF` at {_round(_f(hk_ekf, 'mean_rms_2d'))} m. The PF framework "
                "generalizes to a second urban geometry when appropriately configured."
            ),
            (
                "PF vs RTKLIB demo5 (Odaiba, gnssplusplus corrections): "
                "PF 1M achieves P50=3.64m, RMS=6.72m, >100m=0% vs "
                "RTKLIB P50=2.67m, RMS=13.08m. PF wins RMS by 49%, P95 by 59%, "
                "with zero catastrophic failures. RTKLIB wins P50 by 27%."
            ),
            (
                "Urban canyon simulation: PF advantage increases with NLOS severity. "
                "At 91% NLOS (80m buildings), PF achieves 7.88m vs WLS 51.72m (85% gain). "
                "Map prior (Oh et al. 2004) adds 14-18% further improvement (PF+Map: 6.47m)."
            ),
        ],
        "tables": {
            "paper_main": {
                "title": "Frozen Paper Main Table",
                "source_csv": _copy_paper_data_file(PAPER_MAIN_TABLE_CSV),
                "columns": list(paper_main_rows[0].keys()),
                "rows": paper_main_rows,
            },
            "ppc_tuned": {
                "title": "PPC Tuned Split",
                "source_csv": _copy_data_file(PPC_TUNED_CSV),
                "columns": [
                    "Variant",
                    "Mean RMS 2D (m)",
                    "Mean P95 (m)",
                    "PF Wins",
                    "Blocked Epoch Fraction",
                ],
                "rows": [
                    {
                        "Variant": "Safe baseline: always_robust",
                        "Mean RMS 2D (m)": _round(_f(tuned_safe, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(tuned_safe, "mean_p95")),
                        "PF Wins": f"{int(_f(tuned_safe, 'pf_rms_wins'))}/6",
                        "Blocked Epoch Fraction": _round(
                            _f(tuned_safe, "mean_blocked_epoch_frac"),
                            3,
                        ),
                    },
                    {
                        "Variant": "Exploratory best: entry_veto_negative_exit_...",
                        "Mean RMS 2D (m)": _round(_f(tuned_best, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(tuned_best, "mean_p95")),
                        "PF Wins": f"{int(_f(tuned_best, 'pf_rms_wins'))}/6",
                        "Blocked Epoch Fraction": _round(
                            _f(tuned_best, "mean_blocked_epoch_frac"),
                            3,
                        ),
                    },
                ],
            },
            "ppc_holdout": {
                "title": "PPC Holdout Split",
                "source_csv": _copy_data_file(PPC_HOLDOUT_CSV),
                "columns": [
                    "Variant",
                    "Mean RMS 2D (m)",
                    "Mean P95 (m)",
                    "PF Wins",
                    "Blocked Epoch Fraction",
                ],
                "rows": [
                    {
                        "Variant": "Safe baseline: always_robust",
                        "Mean RMS 2D (m)": _round(_f(holdout_safe, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(holdout_safe, "mean_p95")),
                        "PF Wins": f"{int(_f(holdout_safe, 'pf_rms_wins'))}/6",
                        "Blocked Epoch Fraction": _round(
                            _f(holdout_safe, "mean_blocked_epoch_frac"),
                            3,
                        ),
                    },
                    {
                        "Variant": "Exploratory best: entry_veto_negative_exit_...",
                        "Mean RMS 2D (m)": _round(_f(holdout_best, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(holdout_best, "mean_p95")),
                        "PF Wins": f"{int(_f(holdout_best, 'pf_rms_wins'))}/6",
                        "Blocked Epoch Fraction": _round(
                            _f(holdout_best, "mean_blocked_epoch_frac"),
                            3,
                        ),
                    },
                ],
            },
            "urbannav_external": {
                "title": "UrbanNav External Validation",
                "source_csv": _copy_data_file(URBANNAV_SUMMARY_CSV),
                "columns": [
                    "Method",
                    "Mean RMS 2D (m)",
                    "Mean P95 (m)",
                    ">100 m Rate (%)",
                    ">500 m Rate (%)",
                    "Wins vs WLS RMS",
                ],
                "rows": [
                    {
                        "Method": "WLS",
                        "Mean RMS 2D (m)": _round(_f(wls, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(wls, "mean_p95")),
                        ">100 m Rate (%)": _round(_f(wls, "mean_outlier_rate_pct")),
                        ">500 m Rate (%)": _round(
                            _f(wls, "mean_catastrophic_rate_pct"),
                            3,
                        ),
                        "Wins vs WLS RMS": int(_f(wls, "wins_vs_wls_rms")),
                    },
                    {
                        "Method": "WLS+QualityVeto",
                        "Mean RMS 2D (m)": _round(_f(wls_qv, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(wls_qv, "mean_p95")),
                        ">100 m Rate (%)": _round(
                            _f(wls_qv, "mean_outlier_rate_pct")
                        ),
                        ">500 m Rate (%)": _round(
                            _f(wls_qv, "mean_catastrophic_rate_pct"),
                            3,
                        ),
                        "Wins vs WLS RMS": int(_f(wls_qv, "wins_vs_wls_rms")),
                    },
                    {
                        "Method": "EKF",
                        "Mean RMS 2D (m)": _round(_f(ekf, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(ekf, "mean_p95")),
                        ">100 m Rate (%)": _round(_f(ekf, "mean_outlier_rate_pct")),
                        ">500 m Rate (%)": _round(
                            _f(ekf, "mean_catastrophic_rate_pct"),
                            3,
                        ),
                        "Wins vs WLS RMS": int(_f(ekf, "wins_vs_wls_rms")),
                    },
                    {
                        "Method": "PF-10K",
                        "Mean RMS 2D (m)": _round(_f(pf, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(pf, "mean_p95")),
                        ">100 m Rate (%)": _round(_f(pf, "mean_outlier_rate_pct")),
                        ">500 m Rate (%)": _round(
                            _f(pf, "mean_catastrophic_rate_pct"),
                            3,
                        ),
                        "Wins vs WLS RMS": int(_f(pf, "wins_vs_wls_rms")),
                    },
                    {
                        "Method": "PF+RobustClear-10K",
                        "Mean RMS 2D (m)": _round(_f(robust, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(robust, "mean_p95")),
                        ">100 m Rate (%)": _round(
                            _f(robust, "mean_outlier_rate_pct")
                        ),
                        ">500 m Rate (%)": _round(
                            _f(robust, "mean_catastrophic_rate_pct"),
                            3,
                        ),
                        "Wins vs WLS RMS": int(_f(robust, "wins_vs_wls_rms")),
                    },
                ],
            },
            "urbannav_windows": {
                "title": "UrbanNav Fixed-Window Check",
                "source_csv": _copy_data_file(URBANNAV_WINDOW_SUMMARY_CSV),
                "columns": [
                    "Method",
                    "Mean RMS 2D (m)",
                    "Mean P95 (m)",
                    "RMS Wins vs EKF",
                    "P95 Wins vs EKF",
                    "Catastrophic Wins vs EKF",
                ],
                "rows": [
                    {
                        "Method": "EKF",
                        "Mean RMS 2D (m)": _round(_f(window_ekf, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(window_ekf, "mean_p95")),
                        "RMS Wins vs EKF": "-",
                        "P95 Wins vs EKF": "-",
                        "Catastrophic Wins vs EKF": "-",
                    },
                    {
                        "Method": "PF-10K",
                        "Mean RMS 2D (m)": _round(
                            _f(_find_row(urbannav_window_rows, "method", "PF-10K"), "mean_rms_2d")
                        ),
                        "Mean P95 (m)": _round(
                            _f(_find_row(urbannav_window_rows, "method", "PF-10K"), "mean_p95")
                        ),
                        "RMS Wins vs EKF": (
                            f"{int(_f(_find_row(urbannav_window_rows, 'method', 'PF-10K'), 'wins_rms'))}/"
                            f"{int(_f(_find_row(urbannav_window_rows, 'method', 'PF-10K'), 'n_windows'))}"
                        ),
                        "P95 Wins vs EKF": (
                            f"{int(_f(_find_row(urbannav_window_rows, 'method', 'PF-10K'), 'wins_p95'))}/"
                            f"{int(_f(_find_row(urbannav_window_rows, 'method', 'PF-10K'), 'n_windows'))}"
                        ),
                        "Catastrophic Wins vs EKF": (
                            f"{int(_f(_find_row(urbannav_window_rows, 'method', 'PF-10K'), 'wins_catastrophic'))}/"
                            f"{int(_f(_find_row(urbannav_window_rows, 'method', 'PF-10K'), 'n_windows'))}"
                        ),
                    },
                    {
                        "Method": "PF+RobustClear-10K",
                        "Mean RMS 2D (m)": _round(_f(window_pf, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(window_pf, "mean_p95")),
                        "RMS Wins vs EKF": (
                            f"{int(_f(window_pf, 'wins_rms'))}/"
                            f"{int(_f(window_pf, 'n_windows'))}"
                        ),
                        "P95 Wins vs EKF": (
                            f"{int(_f(window_pf, 'wins_p95'))}/"
                            f"{int(_f(window_pf, 'n_windows'))}"
                        ),
                        "Catastrophic Wins vs EKF": (
                            f"{int(_f(window_pf, 'wins_catastrophic'))}/"
                            f"{int(_f(window_pf, 'n_windows'))}"
                        ),
                    },
                ],
            },
            "hong_kong_control": {
                "title": "Hong Kong Control Check",
                "source_csv": _copy_data_file(URBANNAV_HK_ADAPTIVE_CSV),
                "columns": [
                    "Method",
                    "Mean RMS 2D (m)",
                    "Mean P95 (m)",
                    ">100 m Rate (%)",
                    "Longest >100 m Segment (s)",
                ],
                "rows": [
                    {
                        "Method": row["method"],
                        "Mean RMS 2D (m)": _round(_f(row, "mean_rms_2d")),
                        "Mean P95 (m)": _round(_f(row, "mean_p95")),
                        ">100 m Rate (%)": _round(_f(row, "mean_outlier_rate_pct")),
                        "Longest >100 m Segment (s)": _round(
                            _f(row, "mean_longest_outlier_segment_s"),
                            2,
                        ),
                    }
                    for row in hk_adaptive_rows
                ],
            },
            "bvh_runtime": {
                "title": "Real-PLATEAU Runtime",
                "source_csv": _copy_data_file(BVH_RUNTIME_CSV),
                "columns": ["Method", "RMS 2D (m)", "P95 (m)", "Time (ms/epoch)"],
                "rows": [
                    {
                        "Method": "PF3D-10K",
                        "RMS 2D (m)": _round(_f(pf3d, "rms_2d")),
                        "P95 (m)": _round(_f(pf3d, "p95")),
                        "Time (ms/epoch)": _round(_f(pf3d, "time_ms"), 2),
                    },
                    {
                        "Method": "PF3D-BVH-10K",
                        "RMS 2D (m)": _round(_f(pf3d_bvh, "rms_2d")),
                        "P95 (m)": _round(_f(pf3d_bvh, "p95")),
                        "Time (ms/epoch)": _round(_f(pf3d_bvh, "time_ms"), 2),
                    },
                ],
            },
            "urbannav_runs": {
                "title": "UrbanNav Per-Run Results",
                "source_csv": _copy_data_file(URBANNAV_RUNS_CSV),
                "columns": [
                    "Run",
                    "Method",
                    "RMS 2D (m)",
                    "P95 (m)",
                    ">100 m Rate (%)",
                    "Time (ms/epoch)",
                ],
                "rows": [
                    {
                        "Run": row["run"],
                        "Method": row["method"],
                        "RMS 2D (m)": _round(_f(row, "rms_2d")),
                        "P95 (m)": _round(_f(row, "p95")),
                        ">100 m Rate (%)": _round(_f(row, "outlier_rate_pct")),
                        "Time (ms/epoch)": _round(_f(row, "time_ms_per_epoch"), 2),
                    }
                    for row in urbannav_run_rows
                ],
            },
        },
        "limitations": [
            "The exploratory PPC gate generalizes on holdout, but the gain is still small enough that it should not carry the paper by itself.",
            "The strongest current story is not explicit 3D-map accuracy. The real-data 3D path is still best framed as a BVH systems contribution.",
            "UrbanNav external gains depend on the repaired multi-GNSS measurement path. The older GPS-only results were limited by loader and measurement issues.",
            "Hong Kong is now a supplemental positive result across 3 sequences (PF+AdaptiveGuide beats EKF on all), but the winning method differs from the frozen Tokyo mainline.",
            "AdaptiveGuide and EKFRescue help robustness in the right regimes, but they do not beat the frozen Tokyo mainline `PF+RobustClear-10K` on full runs.",
        ],
        "validation": {
            "tests": validation["summary"],
            "note": validation["note"],
            "command": validation["command"],
            "generated_at_utc": validation["generated_at_utc"],
        },
    }


def main() -> None:
    snapshot = _build_snapshot()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    SNAPSHOT_JS_PATH.write_text(
        "window.__GNSS_GPU_SNAPSHOT__ = "
        + json.dumps(snapshot, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
