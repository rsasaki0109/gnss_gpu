#!/usr/bin/env python3
"""Product-mode runner for the adopted §7.16 FIX-rate predictor.

The default mode is the frozen product inference flow: it rebuilds the
route/window deliverable CSVs and dashboard from the committed adopted
§7.16 window predictions.  This works from a clean checkout and is the
operator-facing one-command path.

`--inference` scores a new pre-augmented window CSV with a saved
single-model artifact and does not retrain.  This is the product path for
fresh PPC/taroz data once upstream window/base features exist.

`--retrain` runs the research/training pipeline that created the adopted
predictions.  It requires the large preprocessed epoch/window CSVs and a
refined-grid base prediction CSV.  The nested stack uses strict
leave-one-run-out validation and is not the operator path for new runs.

Usage (default: refresh product deliverables from frozen predictions):

    python3 experiments/predict.py

Usage (full LORO retrain with new preprocessed data):

    python3 experiments/predict.py \\
        --retrain \\
        --epochs-csv path/to/new_epochs.csv \\
        --window-csv path/to/new_window_predictions.csv \\
        --base-prefix path/to/refinedgrid_prefix \\
        --results-prefix ppc_..._my_run

Usage (fresh-data inference without retraining):

    python3 experiments/predict.py \\
        --inference \\
        --window-csv path/to/pre_augmented_window_predictions.csv \\
        --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \\
        --inference-output-prefix experiments/results/my_run_product
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from csv import DictReader
from datetime import datetime
from pathlib import Path


_stage_counter = {"current": 0, "total": 1, "started": None}


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
DEFAULT_PREDICTION_CSV = RESULTS_DIR / f"{DEFAULT_RESULTS_PREFIX}_window_predictions.csv"
DEFAULT_INFERENCE_MODEL = RESULTS_DIR / f"{DEFAULT_RESULTS_PREFIX}_product_model.pkl.gz"
DEFAULT_INFERENCE_OUTPUT_PREFIX = RESULTS_DIR / "ppc_product_inference"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_DIR.parent / "internal_docs" / "product_deliverable"

PREDICTION_REQUIRED_COLUMNS = {
    "city",
    "run",
    "window_index",
    "sim_matched_epochs",
    "actual_fix_rate_pct",
    "base_pred_fix_rate_pct",
    "corrected_pred_fix_rate_pct",
}
BASE_PREDICTION_REQUIRED_COLUMNS = {
    "city",
    "run",
    "window_index",
}
BASE_PREDICTION_VALUE_COLUMNS = {
    "corrected_pred_fix_rate_pct",
    "pred_fix_rate_pct",
}

EPOCH_REQUIRED_COLUMNS = {
    "city",
    "run",
    "gps_tow",
}

WINDOW_REQUIRED_COLUMNS = {
    "city",
    "run",
    "window_index",
    "window_start_tow",
    "window_end_tow",
    "actual_fix_rate_pct",
    "base_pred_fix_rate_pct",
}
INFERENCE_WINDOW_REQUIRED_COLUMNS = {
    "city",
    "run",
    "window_index",
    "sim_matched_epochs",
}
FIT_INFERENCE_MODEL_REQUIRED_COLUMNS = INFERENCE_WINDOW_REQUIRED_COLUMNS | {
    "actual_fix_rate_pct",
    "solver_demo5_ratio_mean",
    "solver_demo5_ratio_p90",
    "solver_demo5_ratio_p95",
    "solver_demo5_ratio_mean_past_delta",
    "rtk_lock_p90_p50",
    "rtk_lock_p90_p50_past_delta",
}


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def stage(label: str) -> None:
    _stage_counter["current"] += 1
    _stage_counter["started"] = time.monotonic()
    elapsed = ""
    print(f"\n[{_timestamp()}] [{_stage_counter['current']}/{_stage_counter['total']}] {label}{elapsed}", flush=True)


def stage_done() -> None:
    if _stage_counter["started"] is None:
        return
    dt = time.monotonic() - _stage_counter["started"]
    print(f"[{_timestamp()}] [{_stage_counter['current']}/{_stage_counter['total']}] done in {dt:.1f}s", flush=True)


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}", flush=True)
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


def _read_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = DictReader(fh)
        return set(reader.fieldnames or [])


def _require_input(path: Path, label: str, required_columns: set[str] | None = None) -> None:
    if not path.exists():
        sys.stderr.write(f"\nERROR: missing {label}:\n  {path}\n")
        sys.exit(1)
    if required_columns is None:
        return
    columns = _read_columns(path)
    missing = sorted(required_columns - columns)
    if missing:
        sys.stderr.write(
            f"\nERROR: {label} is missing required columns:\n"
            f"  {path}\n"
            f"  missing: {', '.join(missing)}\n"
        )
        sys.exit(1)


def _require_base_prediction(path: Path) -> None:
    _require_input(path, "refined-grid base prediction CSV", BASE_PREDICTION_REQUIRED_COLUMNS)
    columns = _read_columns(path)
    if not (BASE_PREDICTION_VALUE_COLUMNS & columns):
        sys.stderr.write(
            "\nERROR: refined-grid base prediction CSV needs one prediction column:\n"
            f"  {path}\n"
            f"  expected one of: {', '.join(sorted(BASE_PREDICTION_VALUE_COLUMNS))}\n"
        )
        sys.exit(1)


def _base_prediction_path(base_prefix: str) -> Path:
    """Resolve a refined-grid base prediction prefix or CSV path."""
    path = Path(base_prefix)
    if path.suffix == ".csv":
        return path
    if path.is_absolute() or path.parent != Path("."):
        return path.with_name(f"{path.name}_window_predictions.csv")
    return RESULTS_DIR / f"{base_prefix}_window_predictions.csv"


def _planned_stage_count(args: argparse.Namespace) -> int:
    if args.inference or args.fit_inference_model:
        return 1
    if not args.retrain:
        return 1
    total = 1  # threshold preset + window augmentation always runs in retrain mode
    if not args.skip_epoch_augment:
        total += 1
    if not args.skip_window_aggregate:
        total += 1
    if not args.skip_training:
        total += 1
    if not args.skip_deliverable:
        total += 1
    return total


def _preflight_frozen(args: argparse.Namespace) -> None:
    _require_input(args.prediction_csv, "frozen adopted prediction CSV", PREDICTION_REQUIRED_COLUMNS)


def _preflight_retrain(args: argparse.Namespace, summary_csv: Path, epochs_out_csv: Path) -> None:
    if not args.skip_epoch_augment:
        _require_input(args.epochs_csv, "epoch input CSV", EPOCH_REQUIRED_COLUMNS)
    elif not args.skip_window_aggregate:
        _require_input(epochs_out_csv, "existing validationhold epoch CSV")
    if not args.skip_window_aggregate or not args.skip_training:
        _require_input(args.window_csv, "window input CSV", WINDOW_REQUIRED_COLUMNS)
    if args.skip_window_aggregate:
        _require_input(summary_csv, "existing validationhold window summary CSV")
    if not args.skip_training:
        _require_base_prediction(_base_prediction_path(args.base_prefix))


def _preflight_inference(args: argparse.Namespace) -> None:
    _require_input(args.inference_model, "saved product inference model")
    required = set(INFERENCE_WINDOW_REQUIRED_COLUMNS)
    if args.use_window_base_prediction:
        required.add("base_pred_fix_rate_pct")
    _require_input(args.window_csv, "inference window CSV", required)
    if not args.use_window_base_prediction:
        _require_base_prediction(_base_prediction_path(args.base_prefix))


def _preflight_fit_inference_model(args: argparse.Namespace) -> None:
    _require_input(args.window_csv, "training window CSV", FIT_INFERENCE_MODEL_REQUIRED_COLUMNS)
    _require_base_prediction(_base_prediction_path(args.base_prefix))
    if args.calibration_prediction_csv is not None:
        _require_input(args.calibration_prediction_csv, "calibration prediction CSV", PREDICTION_REQUIRED_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adopted §7.16 FIX-rate prediction pipeline")
    parser.add_argument("--retrain", action="store_true",
                        help="run the full LORO training pipeline instead of the frozen product flow")
    parser.add_argument("--inference", action="store_true",
                        help="score --window-csv with a saved single-model artifact; no retraining")
    parser.add_argument("--fit-inference-model", action="store_true",
                        help="fit and save the single-model product inference artifact")
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PREDICTION_CSV,
                        help="adopted window prediction CSV used by the frozen product flow")
    parser.add_argument("--inference-model", type=Path, default=DEFAULT_INFERENCE_MODEL,
                        help="saved product inference model artifact")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_INFERENCE_MODEL,
                        help="output path for --fit-inference-model")
    parser.add_argument("--inference-output-prefix", type=Path, default=DEFAULT_INFERENCE_OUTPUT_PREFIX,
                        help="prefix for --inference route/window prediction CSVs")
    parser.add_argument("--calibration-prediction-csv", type=Path,
                        help="optional LORO prediction CSV used to calibrate the saved residual corrector")
    parser.add_argument("--use-window-base-prediction", action="store_true",
                        help="in --inference mode, use base_pred_fix_rate_pct from --window-csv instead of --base-prefix")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory for route/window deliverable CSVs and dashboard")
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
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="do not regenerate dashboard.html when building deliverables")
    parser.add_argument("--check-only", action="store_true",
                        help="validate required inputs and exit without writing outputs")
    args = parser.parse_args()
    active_modes = sum(bool(v) for v in (args.retrain, args.inference, args.fit_inference_model))
    if active_modes > 1:
        parser.error("--retrain, --inference, and --fit-inference-model are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()

    summary_csv = RESULTS_DIR / "ppc_validationhold_window_summary.csv"
    preset_summary_csv = RESULTS_DIR / f"ppc_validationhold_window_summary_{args.preset}.csv"
    augmented_window_csv = RESULTS_DIR / (
        f"ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_windowopt_validationhold_{args.preset}_carry_window_predictions.csv"
    )

    epochs_out_csv = RESULTS_DIR / "ppc_demo5_fix_rate_compare_fullsim_stride1_phase_proxy_stat_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_validationhold_epochs.csv"

    _stage_counter["current"] = 0
    _stage_counter["total"] = _planned_stage_count(args)
    _stage_counter["started"] = None
    overall_start = time.monotonic()

    if args.inference:
        _preflight_inference(args)
        if args.check_only:
            print(f"preflight ok: inference can use {args.inference_model}")
            return
        stage("score fresh window CSV with saved product inference model")
        cmd = [
            sys.executable,
            str(EXPERIMENTS_DIR / "product_inference_model.py"),
            "infer",
            "--window-csv", str(args.window_csv),
            "--model", str(args.inference_model),
            "--output-prefix", str(args.inference_output_prefix),
            "--base-prediction-column", "corrected_pred_fix_rate_pct",
        ]
        if not args.use_window_base_prediction:
            cmd.extend(["--base-prefix", args.base_prefix])
        run(cmd)
        stage_done()
        total_dt = time.monotonic() - overall_start
        print(f"\n[{_timestamp()}] fresh-data inference finished in {total_dt:.1f}s")
        return

    if args.fit_inference_model:
        _preflight_fit_inference_model(args)
        if args.check_only:
            print(f"preflight ok: inference model can be fit to {args.window_csv}")
            return
        stage("fit saved product inference model")
        cmd = [
            sys.executable,
            str(EXPERIMENTS_DIR / "product_inference_model.py"),
            "fit",
            "--window-csv", str(args.window_csv),
            "--base-prefix", args.base_prefix,
            "--base-prediction-column", "corrected_pred_fix_rate_pct",
            "--model-output", str(args.model_output),
            "--classifier-include-run-position",
            "--residual-model", "ridge",
            "--residual-alpha", str(args.residual_alpha),
            "--residual-clip-pp", str(args.residual_clip_pp),
        ]
        if args.calibration_prediction_csv is not None:
            cmd.extend(["--calibration-prediction-csv", str(args.calibration_prediction_csv)])
        run(cmd)
        stage_done()
        total_dt = time.monotonic() - overall_start
        print(f"\n[{_timestamp()}] inference model fit finished in {total_dt:.1f}s")
        return

    if not args.retrain:
        _preflight_frozen(args)
        if args.check_only:
            print(f"preflight ok: frozen product flow can use {args.prediction_csv}")
            return
        stage("build product deliverable CSVs + dashboard from frozen predictions")
        cmd = [
            sys.executable,
            str(EXPERIMENTS_DIR / "build_product_deliverable.py"),
            "--prediction-csv", str(args.prediction_csv),
            "--output-dir", str(args.output_dir),
        ]
        if args.skip_dashboard:
            cmd.append("--skip-dashboard")
        run(cmd)
        stage_done()
        total_dt = time.monotonic() - overall_start
        print(f"\n[{_timestamp()}] frozen product flow finished in {total_dt:.1f}s")
        return

    _preflight_retrain(args, summary_csv, epochs_out_csv)
    if args.check_only:
        print("preflight ok: retrain inputs are present")
        return

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
            "--output-dir", str(args.output_dir),
            *(["--skip-dashboard"] if args.skip_dashboard else []),
        ])
        stage_done()

    total_dt = time.monotonic() - overall_start
    print(f"\n[{_timestamp()}] pipeline finished in {total_dt:.1f}s")


if __name__ == "__main__":
    main()
