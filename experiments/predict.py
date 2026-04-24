#!/usr/bin/env python3
"""Product-mode pipeline runner for the adopted §7.16 FIX-rate predictor.

This wrapper chains the four pipeline steps that produce the adopted
`transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45`
predictions.  The default behaviour reproduces the adopted outputs on
the bundled 6-run test set.  For new data, pass `--epochs-csv` and
`--window-csv` pointing at your preprocessed inputs; the pipeline will
include them as additional LORO outer folds.

Caveat: the nested stack uses strict leave-one-run-out validation, so a
"prediction" for a new run means that run is held out while the rest of
the dataset trains the stack.  There is no single pretrained model
artefact saved to disk; the stack is retrained end-to-end each call.
Total runtime is typically 2-4 minutes on the bundled 6-run dataset.

Usage (default: reproduce bundled outputs):

    python3 experiments/predict.py

Usage (with new data):

    python3 experiments/predict.py \\
        --epochs-csv path/to/new_epochs.csv \\
        --window-csv path/to/new_window_predictions.csv \\
        --results-prefix ppc_..._my_run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


_stage_counter = {"current": 0, "total": 5, "started": None}


EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"

DEFAULT_EPOCHS_CSV = (
    RESULTS_DIR
    / "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_epochs.csv"
)
DEFAULT_WINDOW_CSV = (
    RESULTS_DIR
    / "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_window_predictions.csv"
)
DEFAULT_BASE_PREFIX = (
    "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_baseerror15_refinedgrid"
)
DEFAULT_PRESET = "current_tight_hold"
DEFAULT_RESULTS_PREFIX = (
    "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45"
)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def stage(label: str) -> None:
    _stage_counter["current"] += 1
    _stage_counter["started"] = time.monotonic()
    elapsed = ""
    print(f"\n[{_timestamp()}] [{_stage_counter['current']}/{_stage_counter['total']}] {label}{elapsed}")


def stage_done() -> None:
    if _stage_counter["started"] is None:
        return
    dt = time.monotonic() - _stage_counter["started"]
    print(f"[{_timestamp()}] [{_stage_counter['current']}/{_stage_counter['total']}] done in {dt:.1f}s")


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _require(path: Path, producer: str) -> None:
    if not path.exists():
        sys.stderr.write(
            f"\nERROR: expected output missing after running {producer}:\n  {path}\n"
            f"The pipeline cannot continue.  Check the preceding command's output for\n"
            f"silent failures (e.g. missing input columns, empty epoch CSV).\n"
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adopted §7.16 FIX-rate prediction pipeline")
    parser.add_argument("--epochs-csv", type=Path, default=DEFAULT_EPOCHS_CSV)
    parser.add_argument("--window-csv", type=Path, default=DEFAULT_WINDOW_CSV)
    parser.add_argument("--base-prefix", default=DEFAULT_BASE_PREFIX)
    parser.add_argument("--preset", default=DEFAULT_PRESET)
    parser.add_argument("--residual-alpha", type=float, default=0.75)
    parser.add_argument("--residual-clip-pp", type=float, default=50.0)
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--skip-epoch-augment", action="store_true")
    parser.add_argument("--skip-window-aggregate", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-deliverable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_csv = RESULTS_DIR / "ppc_validationhold_window_summary.csv"
    preset_summary_csv = RESULTS_DIR / f"ppc_validationhold_window_summary_{args.preset}.csv"
    augmented_window_csv = RESULTS_DIR / (
        f"ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_{args.preset}_carry_window_predictions.csv"
    )

    epochs_out_csv = RESULTS_DIR / "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_validationhold_epochs.csv"

    overall_start = time.monotonic()

    # 1. Epoch-level validationhold surrogate
    if not args.skip_epoch_augment:
        stage("epoch-level validationhold surrogate")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "augment_ppc_epochs_with_validation_hold_surrogate.py"),
            "--epochs-csv", str(args.epochs_csv),
        ])
        _require(epochs_out_csv, "augment_ppc_epochs_with_validation_hold_surrogate.py")
        stage_done()

    # 2. Window-level aggregation
    if not args.skip_window_aggregate:
        stage("window-level aggregation")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "analyze_ppc_validation_hold_surrogate_windows.py"),
            "--window-csv", str(args.window_csv),
        ])
        _require(summary_csv, "analyze_ppc_validation_hold_surrogate_windows.py")
        stage_done()

    # 3. Apply threshold preset and augment window CSV
    stage(f"apply threshold preset '{args.preset}' and augment window CSV")
    run([
        sys.executable,
        str(EXPERIMENTS_DIR / "rebuild_validationhold_flag_thresholds.py"),
        "--input-csv", str(summary_csv),
        "--preset", args.preset,
        "--output-csv", str(preset_summary_csv),
    ])
    _require(preset_summary_csv, "rebuild_validationhold_flag_thresholds.py")
    run([
        sys.executable,
        str(EXPERIMENTS_DIR / "augment_ppc_windows_with_validationhold_features.py"),
        "--window-csv", str(args.window_csv),
        "--validationhold-csv", str(preset_summary_csv),
        "--output-csv", str(augmented_window_csv),
    ])
    _require(augmented_window_csv, "augment_ppc_windows_with_validationhold_features.py")
    stage_done()

    # 4. Train nested stack and produce window predictions
    if not args.skip_training:
        stage(f"train nested stack (α={args.residual_alpha}, clip={args.residual_clip_pp} pp)")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "train_ppc_solver_transition_surrogate_nested_stack.py"),
            "--window-csv", str(augmented_window_csv),
            "--base-prefix", args.base_prefix,
            "--classifier-include-run-position",
            "--alphas", str(args.residual_alpha),
            "--residual-clip-pp", str(args.residual_clip_pp),
            "--max-run-mae-pp", "4.5",
            "--max-abs-aggregate-error-pp", "2.0",
            "--results-prefix", args.results_prefix,
        ])
        pred_csv = RESULTS_DIR / f"{args.results_prefix}_window_predictions.csv"
        _require(pred_csv, "train_ppc_solver_transition_surrogate_nested_stack.py")
        stage_done()

    # 5. Build product deliverable CSVs
    if not args.skip_deliverable:
        stage("build product deliverable CSVs + dashboard")
        pred_csv = RESULTS_DIR / f"{args.results_prefix}_window_predictions.csv"
        _require(pred_csv, "upstream training (did it run?)")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "build_product_deliverable.py"),
            "--prediction-csv", str(pred_csv),
        ])
        stage_done()

    total_dt = time.monotonic() - overall_start
    print(f"\n[{_timestamp()}] pipeline finished in {total_dt:.1f}s")


if __name__ == "__main__":
    main()
