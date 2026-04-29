#!/usr/bin/env python3
"""Product-mode runner for the adopted §7.16 FIX-rate predictor.

The default mode is the frozen product inference flow: it rebuilds the
route/window deliverable CSVs and dashboard from the committed adopted
§7.16 window predictions.  This works from a clean checkout and is the
operator-facing one-command path.

`--batch-inference` builds the product-ready window CSV from
preprocessed epoch/window/base CSVs and scores it with the saved
single-model artifact in one operator command.  `--prepare-inference`
and `--inference` keep the same stages split for debugging and
intermediate inspection.

`--source-bundle-prepare` starts from raw PPC RINEX/reference run
directories and writes bootstrap, label-free epoch/window/base CSVs plus
a derived source manifest for the product flow.  `--source-bundle-inference`
validates that derived manifest and runs the same batch inference path.

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

Usage (prepare fresh-data inference input without retraining):

    python3 experiments/predict.py \\
        --prepare-inference \\
        --epochs-csv path/to/preprocessed_epochs.csv \\
        --window-csv path/to/window_features.csv \\
        --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \\
        --prepared-window-csv experiments/results/my_run_prepared_window_predictions.csv

Usage (one-shot fresh-data preparation + inference without retraining):

    python3 experiments/predict.py \\
        --batch-inference \\
        --epochs-csv path/to/preprocessed_epochs.csv \\
        --window-csv path/to/window_features.csv \\
        --base-prefix path/to/refinedgrid_prefix_or_predictions.csv \\
        --prepare-prefix experiments/results/my_run_prepare \\
        --prepared-window-csv experiments/results/my_run_prepared_window_predictions.csv \\
        --inference-output-prefix experiments/results/my_run_product

Usage (source-manifest validation + one-shot inference):

    python3 experiments/predict.py \\
        --source-bundle-inference \\
        --source-manifest path/to/source_manifest.json

Usage (raw PPC source preparation + derived manifest):

    python3 experiments/predict.py \\
        --source-bundle-prepare \\
        --source-manifest path/to/raw_source_manifest.json \\
        --source-output-prefix experiments/results/my_raw_source

Usage (score prepared fresh-data window CSV without retraining):

    python3 experiments/predict.py \\
        --inference \\
        --window-csv experiments/results/my_run_prepared_window_predictions.csv \\
        --use-window-base-prediction \\
        --inference-output-prefix experiments/results/my_run_product

Usage (causal online-compatible scoring on prepared windows):

    python3 experiments/predict.py \\
        --online-inference \\
        --window-csv experiments/results/my_run_prepared_window_predictions.csv \\
        --use-window-base-prediction \\
        --planned-window-count 32 \\
        --inference-output-prefix experiments/results/my_run_online_product
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from csv import DictReader, DictWriter
from datetime import datetime
from pathlib import Path

from product_source_bundle import load_source_bundle, validate_source_bundle


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
    "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_isotonic_meta_run45"
)
DEFAULT_PREDICTION_CSV = RESULTS_DIR / f"{DEFAULT_RESULTS_PREFIX}_window_predictions.csv"
DEFAULT_INFERENCE_MODEL = RESULTS_DIR / f"{DEFAULT_RESULTS_PREFIX}_product_model.pkl.gz"
DEFAULT_CALIBRATION_PREDICTION_CSV = RESULTS_DIR / (
    "ppc_window_fix_rate_model_stride1_stat_sim_rinex_phasejump_t0p25_gf0p2_simloscont_focused_simadop_nowt_solver_transition_surrogate_nested_et80_validationhold_current_tight_hold_carry_alpha75_meta_run45_window_predictions.csv"
)
DEFAULT_INFERENCE_OUTPUT_PREFIX = RESULTS_DIR / "ppc_product_inference"
DEFAULT_PREPARE_PREFIX = "ppc_product_inference_prepare"
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
PREPARE_WINDOW_REQUIRED_COLUMNS = {
    "city",
    "run",
    "window_index",
    "window_start_tow",
    "window_end_tow",
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


def _prefix_output_path(prefix: str, suffix: str) -> Path:
    path = Path(prefix)
    if path.is_absolute() or path.parent != Path("."):
        return path.with_name(f"{path.name}{suffix}")
    return RESULTS_DIR / f"{prefix}{suffix}"


def _inference_window_output_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(output_prefix.name + "_window_predictions.csv")


def _inference_route_output_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(output_prefix.name + "_route_predictions.csv")


def _require_distinct_batch_outputs(*, prepared_window_csv: Path, inference_output_prefix: Path) -> None:
    final_window_csv = _inference_window_output_path(inference_output_prefix)
    if prepared_window_csv.expanduser().resolve() != final_window_csv.expanduser().resolve():
        return
    sys.stderr.write(
        "\nERROR: --prepared-window-csv must not equal the final inference window output:\n"
        f"  prepared/window output: {prepared_window_csv}\n"
        f"  final window output:    {final_window_csv}\n"
        "Use a distinct prepared path, e.g. *_prepared_window_predictions.csv,\n"
        "or change --inference-output-prefix.\n"
    )
    sys.exit(1)


def _merge_base_prediction_column(
    *,
    window_csv: Path,
    base_prediction_csv: Path,
    output_csv: Path,
    prediction_column: str = "corrected_pred_fix_rate_pct",
) -> None:
    """Write a window CSV with `base_pred_fix_rate_pct` merged by window key."""
    with base_prediction_csv.open(newline="", encoding="utf-8") as fh:
        reader = DictReader(fh)
        base_fields = set(reader.fieldnames or [])
        pred_col = prediction_column
        if pred_col not in base_fields:
            for candidate in ("corrected_pred_fix_rate_pct", "pred_fix_rate_pct", "base_pred_fix_rate_pct"):
                if candidate in base_fields:
                    pred_col = candidate
                    break
            else:
                sys.stderr.write(
                    "\nERROR: base prediction CSV needs one prediction column:\n"
                    f"  {base_prediction_csv}\n"
                    "  expected one of: corrected_pred_fix_rate_pct, pred_fix_rate_pct, base_pred_fix_rate_pct\n"
                )
                sys.exit(1)
        base_by_key = {
            (row["city"], row["run"], str(int(float(row["window_index"])))): row[pred_col]
            for row in reader
        }

    with window_csv.open(newline="", encoding="utf-8") as fh:
        reader = DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        if "base_pred_fix_rate_pct" not in fieldnames:
            fieldnames.append("base_pred_fix_rate_pct")
        rows: list[dict[str, str]] = []
        missing: list[str] = []
        for row in reader:
            key = (row["city"], row["run"], str(int(float(row["window_index"]))))
            if key not in base_by_key:
                missing.append("\t".join(key))
                continue
            row["base_pred_fix_rate_pct"] = base_by_key[key]
            rows.append(row)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        sys.stderr.write(
            f"\nERROR: {len(missing)} window rows are missing from base prediction CSV:\n"
            f"  {preview}{suffix}\n"
        )
        sys.exit(1)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output_csv}")


def _planned_stage_count(args: argparse.Namespace) -> int:
    if args.source_bundle_prepare:
        return 1
    if args.source_bundle_inference:
        return 6
    if args.source_bundle_check:
        return 1
    if args.batch_inference:
        return 5
    if args.prepare_inference:
        return 4
    if args.inference or args.online_inference or args.fit_inference_model:
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


def _preflight_online_inference(args: argparse.Namespace) -> None:
    _preflight_inference(args)
    if args.online_use_input_run_length or args.planned_window_count is not None:
        return
    columns = _read_columns(args.window_csv)
    if args.planned_window_count_column not in columns:
        sys.stderr.write(
            "\nERROR: --online-inference needs a planned route length:\n"
            f"  missing column: {args.planned_window_count_column}\n"
            "Pass --planned-window-count for a single route, add a per-route\n"
            "planned_window_count column, or use --online-use-input-run-length\n"
            "for offline parity/smoke checks.\n"
        )
        sys.exit(1)


def _preflight_fit_inference_model(args: argparse.Namespace) -> None:
    _require_input(args.window_csv, "training window CSV", FIT_INFERENCE_MODEL_REQUIRED_COLUMNS)
    _require_base_prediction(_base_prediction_path(args.base_prefix))
    if args.calibration_prediction_csv is not None:
        _require_input(args.calibration_prediction_csv, "calibration prediction CSV", PREDICTION_REQUIRED_COLUMNS)


def _preflight_prepare_inference(args: argparse.Namespace) -> None:
    _require_input(args.epochs_csv, "epoch input CSV", EPOCH_REQUIRED_COLUMNS)
    _require_input(args.window_csv, "window input CSV", PREPARE_WINDOW_REQUIRED_COLUMNS)
    _require_base_prediction(_base_prediction_path(args.base_prefix))


def _preflight_batch_inference(args: argparse.Namespace) -> None:
    _preflight_prepare_inference(args)
    _require_input(args.inference_model, "saved product inference model")


def _apply_source_bundle(args: argparse.Namespace) -> None:
    validated = validate_source_bundle(load_source_bundle(args.source_manifest))
    args.epochs_csv = validated.epochs_csv
    args.window_csv = validated.window_csv
    args.base_prefix = str(validated.base_prediction_csv)
    if validated.prepare_prefix is not None:
        args.prepare_prefix = validated.prepare_prefix
    if validated.prepared_window_csv is not None:
        args.prepared_window_csv = validated.prepared_window_csv
    if validated.inference_output_prefix is not None:
        args.inference_output_prefix = validated.inference_output_prefix
    print(
        "source bundle ok: "
        f"{validated.run_count} run(s), "
        f"epochs={validated.epochs_csv}, "
        f"windows={validated.window_csv}, "
        f"base={validated.base_prediction_csv}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the adopted §7.16 FIX-rate prediction pipeline")
    parser.add_argument("--retrain", action="store_true",
                        help="run the full LORO training pipeline instead of the frozen product flow")
    parser.add_argument("--batch-inference", action="store_true",
                        help="prepare fresh-data inference inputs and score them in one command; no retraining")
    parser.add_argument("--source-bundle-prepare", action="store_true",
                        help="derive bootstrap product CSVs from raw PPC source runs")
    parser.add_argument("--source-bundle-check", action="store_true",
                        help="validate a derived PPC source manifest and product CSV contract")
    parser.add_argument("--source-bundle-inference", action="store_true",
                        help="validate derived --source-manifest, then run one-shot batch inference; no retraining")
    parser.add_argument("--inference", action="store_true",
                        help="score --window-csv with a saved single-model artifact; no retraining")
    parser.add_argument("--online-inference", action="store_true",
                        help="score --window-csv in causal online-compatible mode; no retraining")
    parser.add_argument("--prepare-inference", action="store_true",
                        help="build a pre-augmented inference window CSV from epoch/window/base inputs")
    parser.add_argument("--fit-inference-model", action="store_true",
                        help="fit and save the single-model product inference artifact")
    parser.add_argument("--prediction-csv", type=Path, default=DEFAULT_PREDICTION_CSV,
                        help="adopted window prediction CSV used by the frozen product flow")
    parser.add_argument("--inference-model", type=Path, default=DEFAULT_INFERENCE_MODEL,
                        help="saved product inference model artifact")
    parser.add_argument("--model-output", type=Path, default=DEFAULT_INFERENCE_MODEL,
                        help="output path for --fit-inference-model")
    parser.add_argument("--inference-output-prefix", type=Path, default=DEFAULT_INFERENCE_OUTPUT_PREFIX,
                        help="prefix for --batch-inference/--inference route/window prediction CSVs")
    parser.add_argument("--prepare-prefix", default=DEFAULT_PREPARE_PREFIX,
                        help="prefix for intermediate --batch-inference/--prepare-inference artifacts")
    parser.add_argument("--prepared-window-csv", type=Path,
                        help="prepared window CSV output; default is <prepare-prefix>_window_predictions.csv")
    parser.add_argument("--source-manifest", type=Path,
                        help="JSON manifest for raw source preparation or derived source-bundle inference")
    parser.add_argument("--source-output-prefix", type=Path,
                        help="prefix for --source-bundle-prepare epoch/window/base CSV outputs")
    parser.add_argument("--source-derived-manifest", type=Path,
                        help="output manifest for --source-bundle-prepare; defaults next to --source-output-prefix")
    parser.add_argument("--raw-source-systems", default="G,E,J",
                        help="constellation list for --source-bundle-prepare, e.g. G,E,J")
    parser.add_argument("--raw-source-window-duration-s", type=float, default=30.0,
                        help="window duration for bootstrap raw-source preparation")
    parser.add_argument("--raw-source-max-epochs-per-run", type=int,
                        help="debug cap per run for --source-bundle-prepare")
    parser.add_argument("--calibration-prediction-csv", type=Path, default=DEFAULT_CALIBRATION_PREDICTION_CSV,
                        help="optional LORO prediction CSV used to calibrate the saved residual corrector")
    parser.add_argument("--final-calibrator", choices=("none", "isotonic"), default="isotonic",
                        help="optional final prediction calibrator for --fit-inference-model")
    parser.add_argument("--use-window-base-prediction", action="store_true",
                        help="in inference modes, use base_pred_fix_rate_pct from --window-csv instead of --base-prefix")
    parser.add_argument("--planned-window-count", type=int,
                        help="planned route window count for --online-inference on a single route")
    parser.add_argument("--planned-window-count-column", default="planned_window_count",
                        help="per-route planned window count column for --online-inference")
    parser.add_argument("--online-use-input-run-length", action="store_true",
                        help="for offline parity/smoke checks, use input row count as the planned window count")
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
    active_modes = sum(
        bool(v)
        for v in (
            args.retrain,
            args.batch_inference,
            args.source_bundle_prepare,
            args.source_bundle_check,
            args.source_bundle_inference,
            args.inference,
            args.online_inference,
            args.prepare_inference,
            args.fit_inference_model,
        )
    )
    if active_modes > 1:
        parser.error(
            "--retrain, --batch-inference, --inference, --prepare-inference, "
            "--online-inference, --source-bundle-prepare, --source-bundle-check, "
            "--source-bundle-inference, and --fit-inference-model are mutually exclusive"
        )
    if (
        args.source_bundle_prepare
        or args.source_bundle_check
        or args.source_bundle_inference
    ) and args.source_manifest is None:
        parser.error("--source-bundle-prepare/check/inference require --source-manifest")
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

    if args.source_bundle_prepare:
        stage("derive bootstrap product CSVs from raw PPC source bundle")
        from product_raw_source_prepare import (
            _parse_systems,
            prepare_raw_source_bundle,
            preflight_raw_source_bundle,
        )

        max_epochs = (
            args.raw_source_max_epochs_per_run
            if args.raw_source_max_epochs_per_run and args.raw_source_max_epochs_per_run > 0
            else None
        )
        if args.check_only:
            _parse_systems(args.raw_source_systems)
            run_count, feature_count = preflight_raw_source_bundle(
                manifest_path=args.source_manifest,
                model_path=args.inference_model,
            )
            stage_done()
            print(
                "preflight ok: raw source preparation can read "
                f"{run_count} run(s) and {feature_count} model features"
            )
            return
        outputs = prepare_raw_source_bundle(
            manifest_path=args.source_manifest,
            output_prefix=args.source_output_prefix,
            output_manifest=args.source_derived_manifest,
            model_path=args.inference_model,
            systems=_parse_systems(args.raw_source_systems),
            window_duration_s=args.raw_source_window_duration_s,
            max_epochs_per_run=max_epochs,
        )
        stage_done()
        total_dt = time.monotonic() - overall_start
        print(f"\n[{_timestamp()}] raw source bundle preparation finished in {total_dt:.1f}s")
        print(f"epochs CSV: {outputs.epochs_csv}")
        print(f"window CSV: {outputs.window_csv}")
        print(f"base CSV: {outputs.base_prediction_csv}")
        print(f"derived source manifest: {outputs.source_manifest}")
        print("\nnext command:")
        print(
            "  python3 experiments/predict.py "
            f"--source-bundle-inference --source-manifest {outputs.source_manifest}"
        )
        return

    if args.source_bundle_check or args.source_bundle_inference:
        stage("validate PPC source manifest and derived product inputs")
        _apply_source_bundle(args)
        stage_done()
        if args.source_bundle_check:
            total_dt = time.monotonic() - overall_start
            print(f"\n[{_timestamp()}] source bundle validation finished in {total_dt:.1f}s")
            return
        args.batch_inference = True

    if args.prepare_inference or args.batch_inference:
        if args.batch_inference:
            _preflight_batch_inference(args)
        else:
            _preflight_prepare_inference(args)
        base_csv = _base_prediction_path(args.base_prefix)
        base_window_csv = _prefix_output_path(args.prepare_prefix, "_base_window_predictions.csv")
        epoch_prefix = f"{args.prepare_prefix}_validationhold"
        vh_epochs_csv = _prefix_output_path(epoch_prefix, "_epochs.csv")
        summary_prefix = f"{args.prepare_prefix}_validationhold"
        summary_csv = _prefix_output_path(summary_prefix, "_window_summary.csv")
        preset_summary_csv = _prefix_output_path(f"{args.prepare_prefix}_validationhold_{args.preset}", "_window_summary.csv")
        prepared_window_csv = args.prepared_window_csv or _prefix_output_path(args.prepare_prefix, "_window_predictions.csv")
        if args.batch_inference:
            _require_distinct_batch_outputs(
                prepared_window_csv=prepared_window_csv,
                inference_output_prefix=args.inference_output_prefix,
            )
        if args.check_only:
            if args.batch_inference:
                print(
                    "preflight ok: batch inference can write "
                    f"{prepared_window_csv} and score with {args.inference_model}"
                )
            else:
                print(f"preflight ok: inference input preparation can write {prepared_window_csv}")
            return

        stage("merge base prediction into window CSV")
        _merge_base_prediction_column(
            window_csv=args.window_csv,
            base_prediction_csv=base_csv,
            output_csv=base_window_csv,
        )
        _require(base_window_csv, "base prediction merge")
        stage_done()

        stage("epoch-level validationhold surrogate")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "augment_ppc_epochs_with_validation_hold_surrogate.py"),
            "--epochs-csv", str(args.epochs_csv),
            "--results-prefix", epoch_prefix,
        ])
        _require(vh_epochs_csv, "augment_ppc_epochs_with_validation_hold_surrogate.py")
        stage_done()

        stage("window-level validationhold aggregation")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "analyze_ppc_validation_hold_surrogate_windows.py"),
            "--window-csv", str(base_window_csv),
            "--epoch-csv", str(vh_epochs_csv),
            "--prediction-column", "base_pred_fix_rate_pct",
            "--results-prefix", summary_prefix,
        ])
        _require(summary_csv, "analyze_ppc_validation_hold_surrogate_windows.py")
        stage_done()

        stage(f"apply threshold preset '{args.preset}' and prepare inference window CSV")
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
            "--window-csv", str(base_window_csv),
            "--validationhold-csv", str(preset_summary_csv),
            "--output-csv", str(prepared_window_csv),
        ])
        _require(prepared_window_csv, "augment_ppc_windows_with_validationhold_features.py")
        stage_done()

        if args.prepare_inference:
            total_dt = time.monotonic() - overall_start
            print(f"\n[{_timestamp()}] inference input preparation finished in {total_dt:.1f}s")
            print(f"prepared window CSV: {prepared_window_csv}")
            return

        stage("score prepared window CSV with saved product inference model")
        run([
            sys.executable,
            str(EXPERIMENTS_DIR / "product_inference_model.py"),
            "infer",
            "--window-csv", str(prepared_window_csv),
            "--model", str(args.inference_model),
            "--output-prefix", str(args.inference_output_prefix),
            "--base-prediction-column", "corrected_pred_fix_rate_pct",
        ])
        stage_done()

        route_csv = _inference_route_output_path(args.inference_output_prefix)
        window_csv = _inference_window_output_path(args.inference_output_prefix)
        total_dt = time.monotonic() - overall_start
        print(f"\n[{_timestamp()}] one-shot fresh-data inference finished in {total_dt:.1f}s")
        print(f"prepared window CSV: {prepared_window_csv}")
        print(f"route predictions: {route_csv}")
        print(f"window predictions: {window_csv}")
        return

    if args.inference or args.online_inference:
        if args.online_inference:
            _preflight_online_inference(args)
        else:
            _preflight_inference(args)
        if args.check_only:
            if args.online_inference:
                print(f"preflight ok: online inference can use {args.inference_model}")
            else:
                print(f"preflight ok: inference can use {args.inference_model}")
            return
        stage_label = (
            "score prepared window CSV with saved product model in online mode"
            if args.online_inference
            else "score fresh window CSV with saved product inference model"
        )
        stage(stage_label)
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
        if args.online_inference:
            cmd.append("--online")
            if args.planned_window_count is not None:
                cmd.extend(["--planned-window-count", str(args.planned_window_count)])
            if args.planned_window_count_column:
                cmd.extend(["--planned-window-count-column", args.planned_window_count_column])
            if args.online_use_input_run_length:
                cmd.append("--online-use-input-run-length")
        run(cmd)
        stage_done()
        total_dt = time.monotonic() - overall_start
        label = "online inference" if args.online_inference else "fresh-data inference"
        print(f"\n[{_timestamp()}] {label} finished in {total_dt:.1f}s")
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
            "--final-calibrator", args.final_calibrator,
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
