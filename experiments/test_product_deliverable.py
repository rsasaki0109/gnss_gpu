#!/usr/bin/env python3
"""Smoke tests for the PPC FIX-rate predictor deliverable helpers.

Run directly: `python3 experiments/test_product_deliverable.py`
Zero external dependencies beyond `pandas` (already used by the
helpers); no pytest required.  Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import tempfile
import json
from argparse import Namespace
from contextlib import redirect_stderr
from csv import DictReader
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

# Make the experiments directory importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import _is_metadata_or_label
from build_product_deliverable import (
    REQUIRED_PREDICTION_COLUMNS,
    _classify_window,
    _confidence_tier,
    _require_prediction_columns,
    _route_action,
    _window_action,
)
from analyze_ppc_validation_hold_surrogate_windows import _window_rows as _validationhold_window_rows
from predict import (
    DEFAULT_PREDICTION_CSV,
    RESULTS_DIR,
    _base_prediction_path,
    _inference_route_output_path,
    _inference_window_output_path,
    _merge_base_prediction_column,
    _planned_stage_count,
    _prefix_output_path,
    _read_columns,
    _require,
    _require_distinct_batch_outputs,
)
import product_raw_source_prepare as raw_source_prepare
from product_inference_model import (
    _append_classifier_meta_online,
    _apply_final_calibrator,
    _apply_prediction_guards,
    _base_from_frame_or_reference,
    _prediction_rows as _product_prediction_rows,
    _planned_counts_from_column,
    _route_rows,
)
from product_raw_source_prepare import build_window_rows
from product_raw_source_prepare import _epoch_rows_for_run as _raw_epoch_rows_for_run
from product_source_bundle import SourceRun, load_source_bundle, validate_source_bundle
from gnss_gpu.io.nav_rinex import _datetime_to_gps_seconds_of_week
from train_ppc_solver_transition_surrogate_stack import (
    _append_classifier_meta as _append_classifier_meta_batch,
    _reference_prediction_path,
)


FAILURES: list[str] = []


def check(name: str, expected, actual) -> None:
    if expected == actual:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}: expected {expected!r} got {actual!r}")
        FAILURES.append(name)


def _row(actual: float, base: float, corrected: float) -> pd.Series:
    return pd.Series({
        "actual_fix_rate_pct": actual,
        "base_pred_fix_rate_pct": base,
        "corrected_pred_fix_rate_pct": corrected,
    })


def test_classify_window() -> None:
    print("test_classify_window")
    # false_high: actual near zero, corrected still inflated (Tokyo run2 w7 archetype)
    tag, _ = _classify_window(_row(0.0, 63.5, 39.5))
    check("false_high: actual 0 corrected 39.5", "false_high", tag)
    # hidden_high: actual 100, corrected under-predicts by >=40 pp (Tokyo run2 w23)
    tag, _ = _classify_window(_row(100.0, 16.3, 25.2))
    check("hidden_high: actual 100 corrected 25 gap 75", "hidden_high", tag)
    # Tokyo run2 w27 case (gap 36 pp < 40 pp threshold, should NOT tag)
    tag, _ = _classify_window(_row(75.3, 23.2, 39.2))
    check("no tag: actual 75 corrected 39 gap 36 (below hidden_high threshold)", "", tag)
    # false_lift: actual 0, delta +19, corrected 19 (Tokyo run3 w17)
    tag, _ = _classify_window(_row(0.0, 0.0, 19.0))
    check("false_lift: actual 0 base 0 corrected 19", "false_lift", tag)
    # false_lift_resolved: Nagoya run2 w27 archetype (base high, corrected pushed down)
    tag, _ = _classify_window(_row(0.0, 27.3, 8.9))
    check("false_lift_resolved: actual 0 base 27 corrected 9", "false_lift_resolved", tag)
    # Normal mid-range window: nothing tagged
    tag, _ = _classify_window(_row(50.0, 45.0, 48.0))
    check("no tag: actual 50 corrected 48 mid-range", "", tag)
    # Edge: actual exactly at 5.0 boundary with high corrected should still be false_high
    tag, _ = _classify_window(_row(5.0, 40.0, 35.0))
    check("boundary: actual=5.0 corrected=35 -> false_high", "false_high", tag)
    # Edge: actual just above the actual_low boundary should NOT trigger false_high via corrected>=35
    tag, _ = _classify_window(_row(5.01, 40.0, 35.0))
    check("boundary: actual=5.01 corrected=35 -> no tag", "", tag)
    # Edge: actual >= 75 but gap just under 40 -> no tag
    tag, _ = _classify_window(_row(75.0, 20.0, 35.5))
    check("boundary: actual=75 gap=39.5 -> no tag", "", tag)
    # Edge: actual >= 75 with gap >= 40 -> hidden_high
    tag, _ = _classify_window(_row(75.0, 20.0, 34.9))
    check("boundary: actual=75 gap=40.1 -> hidden_high", "hidden_high", tag)


def test_is_metadata_or_label() -> None:
    print("test_is_metadata_or_label")
    check("metadata: city", True, _is_metadata_or_label("city"))
    check("metadata: window_index", True, _is_metadata_or_label("window_index"))
    check("metadata: actual_fix_rate_pct", True, _is_metadata_or_label("actual_fix_rate_pct"))
    check("label prefix: rtk_fix3_count", True, _is_metadata_or_label("rtk_fix3_count"))
    check("label prefix: solver_demo5_ratio_mean", True, _is_metadata_or_label("solver_demo5_ratio_mean"))
    check("pred suffix: extra_trees_pred_fix_rate_pct", True, _is_metadata_or_label("extra_trees_pred_fix_rate_pct"))
    check("error suffix: base_error_pp", True, _is_metadata_or_label("base_error_pp"))
    check("constant prefix: constant_global_rate_pred_fix_rate_pct", True, _is_metadata_or_label("constant_global_rate_pred_fix_rate_pct"))
    check("deployable feature: sim_adop_cont_ge60p0s_count_min", False, _is_metadata_or_label("sim_adop_cont_ge60p0s_count_min"))
    check("deployable feature: rinex_phase_streak_ge60p0s_fraction_p10", False, _is_metadata_or_label("rinex_phase_streak_ge60p0s_fraction_p10"))
    check("deployable feature: hold_ready_frac", False, _is_metadata_or_label("hold_ready_frac"))


def test_confidence_tier() -> None:
    print("test_confidence_tier")
    tier, _ = _confidence_tier(["false_high"])
    check("any false_high -> low", "low", tier)
    tier, _ = _confidence_tier(["false_lift"])
    check("any false_lift -> low", "low", tier)
    tier, _ = _confidence_tier(["hidden_high"])
    check("only hidden_high -> medium", "medium", tier)
    tier, _ = _confidence_tier(["false_lift_resolved"])
    check("only false_lift_resolved -> high (positive outcome)", "high", tier)
    tier, _ = _confidence_tier([])
    check("no tags -> high", "high", tier)
    tier, _ = _confidence_tier(["hidden_high", "false_high"])
    check("mixed with false_high -> low (takes precedence)", "low", tier)
    tier, _ = _confidence_tier(["hidden_high", "false_lift_resolved"])
    check("mixed without low-tier trigger -> medium", "medium", tier)


def test_actionability_labels() -> None:
    print("test_actionability_labels")
    action, _ = _window_action("")
    check("normal window action -> use", "use", action)
    action, _ = _window_action("false_lift_resolved")
    check("resolved false lift action -> use", "use", action)
    action, _ = _window_action("hidden_high")
    check("hidden high action -> review", "review", action)
    action, _ = _window_action("false_high")
    check("false high action -> abstain", "abstain", action)
    action, _ = _window_action("false_lift")
    check("false lift action -> abstain", "abstain", action)

    route_action, _ = _route_action(["use", "use"])
    check("all use route action -> ok", "ok", route_action)
    route_action, _ = _route_action(["use", "review"])
    check("review route action -> review", "review", route_action)
    route_action, _ = _route_action(["use", "review", "abstain"])
    check("abstain route action -> review_required", "review_required", route_action)


def test_require() -> None:
    print("test_require")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        existing = tmpdir / "exists.csv"
        existing.write_text("ok\n")
        # _require on existing path should not exit
        try:
            _require(existing, "test_producer")
            check("_require passes on existing file", True, True)
        except SystemExit:
            check("_require passes on existing file", True, False)
        # _require on missing path should SystemExit with nonzero code
        missing = tmpdir / "missing.csv"
        try:
            with redirect_stderr(StringIO()):
                _require(missing, "test_producer")
            check("_require exits on missing file", True, False)
        except SystemExit as exc:
            check("_require exits on missing file", True, exc.code != 0)


def test_prediction_contract() -> None:
    print("test_prediction_contract")
    columns = _read_columns(DEFAULT_PREDICTION_CSV)
    check("default adopted prediction CSV exists", True, DEFAULT_PREDICTION_CSV.exists())
    check("default adopted prediction CSV has required columns", True, REQUIRED_PREDICTION_COLUMNS.issubset(columns))
    try:
        _require_prediction_columns(pd.DataFrame(columns=sorted(REQUIRED_PREDICTION_COLUMNS)), DEFAULT_PREDICTION_CSV)
        check("_require_prediction_columns passes complete schema", True, True)
    except SystemExit:
        check("_require_prediction_columns passes complete schema", True, False)
    try:
        _require_prediction_columns(pd.DataFrame(columns=["city"]), DEFAULT_PREDICTION_CSV)
        check("_require_prediction_columns exits on missing schema", True, False)
    except SystemExit:
        check("_require_prediction_columns exits on missing schema", True, True)


def test_base_prediction_path() -> None:
    print("test_base_prediction_path")
    check(
        "bare base prefix resolves under results",
        RESULTS_DIR / "foo_window_predictions.csv",
        _base_prediction_path("foo"),
    )
    check(
        "relative base prefix path appends window suffix",
        Path("tmp/foo_window_predictions.csv"),
        _base_prediction_path("tmp/foo"),
    )
    check(
        "csv base path passes through",
        Path("tmp/foo.csv"),
        _base_prediction_path("tmp/foo.csv"),
    )
    check(
        "trainer uses same bare-prefix resolution",
        RESULTS_DIR / "foo_window_predictions.csv",
        _reference_prediction_path("foo"),
    )
    check(
        "trainer accepts explicit CSV path",
        Path("tmp/foo.csv"),
        _reference_prediction_path("tmp/foo.csv"),
    )
    check(
        "bare prepare prefix resolves under results",
        RESULTS_DIR / "foo_window_predictions.csv",
        _prefix_output_path("foo", "_window_predictions.csv"),
    )
    check(
        "absolute prepare prefix resolves next to prefix",
        Path("/tmp/foo_window_predictions.csv"),
        _prefix_output_path("/tmp/foo", "_window_predictions.csv"),
    )


def test_prediction_mode_stage_counts() -> None:
    print("test_prediction_mode_stage_counts")
    base_args = {
        "batch_inference": False,
        "source_bundle_prepare": False,
        "source_bundle_check": False,
        "source_bundle_inference": False,
        "prepare_inference": False,
        "inference": False,
        "online_inference": False,
        "fit_inference_model": False,
        "retrain": False,
    }
    check("frozen product flow stage count", 1, _planned_stage_count(Namespace(**base_args)))
    check(
        "batch inference stage count",
        5,
        _planned_stage_count(Namespace(**{**base_args, "batch_inference": True})),
    )
    check(
        "prepare inference stage count",
        4,
        _planned_stage_count(Namespace(**{**base_args, "prepare_inference": True})),
    )
    check(
        "split inference stage count",
        1,
        _planned_stage_count(Namespace(**{**base_args, "inference": True})),
    )
    check(
        "online inference stage count",
        1,
        _planned_stage_count(Namespace(**{**base_args, "online_inference": True})),
    )
    check(
        "source bundle prepare stage count",
        1,
        _planned_stage_count(Namespace(**{**base_args, "source_bundle_prepare": True})),
    )
    check(
        "source bundle check stage count",
        1,
        _planned_stage_count(Namespace(**{**base_args, "source_bundle_check": True})),
    )
    check(
        "source bundle inference stage count",
        6,
        _planned_stage_count(Namespace(**{**base_args, "source_bundle_inference": True})),
    )


def test_online_classifier_meta_matches_batch_run_position() -> None:
    print("test_online_classifier_meta_matches_batch_run_position")
    frame = pd.DataFrame(
        [
            {"city": "tokyo", "run": "run1", "window_index": 0, "planned_window_count": 3},
            {"city": "tokyo", "run": "run1", "window_index": 1, "planned_window_count": 3},
            {"city": "tokyo", "run": "run1", "window_index": 2, "planned_window_count": 3},
        ]
    )
    x = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    base = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    batch_x, batch_names = _append_classifier_meta_batch(
        df=frame,
        x=x,
        names=["feature"],
        base=base,
        include_base=False,
        include_city=False,
        include_run_position=True,
    )
    online_x, online_names = _append_classifier_meta_online(
        df=frame,
        x=x,
        names=["feature"],
        base=base,
        include_base=False,
        include_city=False,
        include_run_position=True,
        planned_window_counts=_planned_counts_from_column(frame, "planned_window_count"),
    )
    check("online meta names match batch", batch_names, online_names)
    check("online meta values match batch", True, bool(np.allclose(batch_x, online_x)))
    try:
        with redirect_stderr(StringIO()):
            _append_classifier_meta_online(
                df=frame,
                x=x,
                names=["feature"],
                base=base,
                include_base=False,
                include_city=False,
                include_run_position=True,
                planned_window_counts={("tokyo", "run1"): 2},
            )
        check("online meta rejects too-short planned count", True, False)
    except SystemExit as exc:
        check("online meta rejects too-short planned count", True, exc.code != 0)


def test_batch_output_paths() -> None:
    print("test_batch_output_paths")
    prefix = Path("/tmp/product_run")
    check("batch route output path", Path("/tmp/product_run_route_predictions.csv"), _inference_route_output_path(prefix))
    check(
        "batch window output path",
        Path("/tmp/product_run_window_predictions.csv"),
        _inference_window_output_path(prefix),
    )
    try:
        with redirect_stderr(StringIO()):
            _require_distinct_batch_outputs(
                prepared_window_csv=Path("/tmp/product_run_prepared_window_predictions.csv"),
                inference_output_prefix=prefix,
            )
        check("batch output path accepts distinct files", True, True)
    except SystemExit:
        check("batch output path accepts distinct files", True, False)
    try:
        with redirect_stderr(StringIO()):
            _require_distinct_batch_outputs(
                prepared_window_csv=Path("/tmp/product_run_window_predictions.csv"),
                inference_output_prefix=prefix,
            )
        check("batch output path rejects overwrite", True, False)
    except SystemExit as exc:
        check("batch output path rejects overwrite", True, exc.code != 0)


def test_product_source_bundle_validation() -> None:
    print("test_product_source_bundle_validation")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        run_dir = tmpdir / "data" / "nagoya" / "run1"
        run_dir.mkdir(parents=True)
        for name in ("rover.obs", "base.obs", "base.nav", "reference.csv"):
            (run_dir / name).write_text("fixture\n", encoding="utf-8")

        epochs = tmpdir / "epochs.csv"
        windows = tmpdir / "windows.csv"
        base = tmpdir / "base_predictions.csv"
        prepared = tmpdir / "product_prepared_window_predictions.csv"
        epochs.write_text(
            "city,run,gps_tow\n"
            "nagoya,run1,1.0\n",
            encoding="utf-8",
        )
        windows.write_text(
            "city,run,window_index,window_start_tow,window_end_tow,sim_matched_epochs\n"
            "nagoya,run1,0,1.0,2.0,1\n",
            encoding="utf-8",
        )
        base.write_text(
            "city,run,window_index,corrected_pred_fix_rate_pct\n"
            "nagoya,run1,0,12.5\n",
            encoding="utf-8",
        )
        manifest = tmpdir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "runs": [{"city": "nagoya", "run": "run1", "run_dir": str(run_dir)}],
                    "derived_inputs": {
                        "epochs_csv": str(epochs),
                        "window_csv": str(windows),
                        "base_prediction_csv": str(base),
                    },
                    "outputs": {
                        "prepare_prefix": str(tmpdir / "product_prepare"),
                        "prepared_window_csv": str(prepared),
                        "inference_output_prefix": str(tmpdir / "product"),
                    },
                }
            ),
            encoding="utf-8",
        )
        validated = validate_source_bundle(load_source_bundle(manifest))
        check("source bundle run count", 1, validated.run_count)
        check("source bundle key", (("nagoya", "run1"),), validated.run_keys)
        check("source bundle epochs path", epochs, validated.epochs_csv)
        check("source bundle prepare prefix", str(tmpdir / "product_prepare"), validated.prepare_prefix)
        check("source bundle prepared path", prepared, validated.prepared_window_csv)

        extra_windows = tmpdir / "extra_windows.csv"
        extra_windows.write_text(
            "city,run,window_index,window_start_tow,window_end_tow,sim_matched_epochs\n"
            "nagoya,run1,0,1.0,2.0,1\n"
            "tokyo,run9,0,1.0,2.0,1\n",
            encoding="utf-8",
        )
        bad_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        bad_manifest["derived_inputs"]["window_csv"] = str(extra_windows)
        manifest.write_text(json.dumps(bad_manifest), encoding="utf-8")
        try:
            with redirect_stderr(StringIO()):
                validate_source_bundle(load_source_bundle(manifest))
            check("source bundle rejects undeclared window run", True, False)
        except SystemExit as exc:
            check("source bundle rejects undeclared window run", True, exc.code != 0)

        bad_base = tmpdir / "bad_base_predictions.csv"
        bad_base.write_text(
            "city,run,window_index,corrected_pred_fix_rate_pct\n"
            "nagoya,run1,1,12.5\n",
            encoding="utf-8",
        )
        bad_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        bad_manifest["derived_inputs"]["window_csv"] = str(windows)
        bad_manifest["derived_inputs"]["base_prediction_csv"] = str(bad_base)
        manifest.write_text(json.dumps(bad_manifest), encoding="utf-8")
        try:
            with redirect_stderr(StringIO()):
                validate_source_bundle(load_source_bundle(manifest))
            check("source bundle rejects missing base window", True, False)
        except SystemExit as exc:
            check("source bundle rejects missing base window", True, exc.code != 0)


def test_raw_source_prepare_manifest_metadata_counts() -> None:
    print("test_raw_source_prepare_manifest_metadata_counts")
    rows_by_run = {
        ("nagoya", "run1"): [
            {"gps_tow": 100.0, "epoch": 0, "sat": 6.0, "phase_fraction": 0.5, "los_fraction": 0.75},
            {"gps_tow": 110.0, "epoch": 1, "sat": 8.0, "phase_fraction": 0.75, "los_fraction": 0.875},
            {"gps_tow": 135.0, "epoch": 2, "sat": 7.0, "phase_fraction": 1.0, "los_fraction": 1.0},
        ],
        ("tokyo", "run2"): [
            {"gps_tow": 200.0, "epoch": 0, "sat": 5.0, "phase_fraction": 0.25, "los_fraction": 0.5},
            {"gps_tow": 210.0, "epoch": 1, "sat": 9.0, "phase_fraction": 0.5, "los_fraction": 0.75},
        ],
    }

    def fake_epoch_rows_for_run(run: SourceRun, *, systems: tuple[str, ...], max_epochs: int | None):
        key = (run.city, run.run)
        rows = []
        for row in rows_by_run[key][:max_epochs]:
            rows.append({"city": run.city, "run": run.run, **row})
        return rows

    original_load_model_feature_names = raw_source_prepare._load_model_feature_names
    original_epoch_rows_for_run = raw_source_prepare._epoch_rows_for_run
    raw_source_prepare._load_model_feature_names = lambda _path: [
        "sat_mean",
        "phase_fraction_mean",
        "los_fraction_mean",
    ]
    raw_source_prepare._epoch_rows_for_run = fake_epoch_rows_for_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            runs = []
            for city, run in rows_by_run:
                run_dir = tmpdir / "data" / city / run
                run_dir.mkdir(parents=True)
                for name in ("rover.obs", "base.obs", "base.nav", "reference.csv"):
                    (run_dir / name).write_text("fixture\n", encoding="utf-8")
                runs.append({"city": city, "run": run, "run_dir": str(run_dir)})
            manifest = tmpdir / "raw_manifest.json"
            manifest.write_text(json.dumps({"runs": runs}), encoding="utf-8")

            outputs = raw_source_prepare.prepare_raw_source_bundle(
                manifest_path=manifest,
                output_prefix=tmpdir / "raw_out",
                output_manifest=None,
                model_path=tmpdir / "model.pkl.gz",
                systems=("G",),
                window_duration_s=30.0,
                max_epochs_per_run=None,
            )
            payload = json.loads(outputs.source_manifest.read_text(encoding="utf-8"))
            metadata = payload["raw_source_prepare"]
            epoch_rows = list(DictReader(outputs.epochs_csv.open(newline="", encoding="utf-8")))
            window_rows = list(DictReader(outputs.window_csv.open(newline="", encoding="utf-8")))
            base_rows = list(DictReader(outputs.base_prediction_csv.open(newline="", encoding="utf-8")))

            check("raw manifest epoch total", len(epoch_rows), metadata["epoch_count"])
            check("raw manifest window total", len(window_rows), metadata["window_count"])
            check("raw manifest base total", len(base_rows), metadata["base_prediction_count"])
            check("raw manifest model feature count", 3, metadata["model_feature_count"])
            check("raw manifest systems", ["G"], metadata["systems"])
            check("raw manifest mode", "bootstrap_neutral_fill", metadata["mode"])

            summaries = {(row["city"], row["run"]): row for row in metadata["runs"]}
            check("raw manifest per-run summary count", 2, len(summaries))
            check("raw manifest nagoya epoch count", 3, summaries[("nagoya", "run1")]["epoch_count"])
            check("raw manifest nagoya window count", 2, summaries[("nagoya", "run1")]["window_count"])
            check("raw manifest tokyo epoch count", 2, summaries[("tokyo", "run2")]["epoch_count"])
            check("raw manifest tokyo window count", 1, summaries[("tokyo", "run2")]["window_count"])
            check(
                "raw manifest elapsed finite",
                True,
                all(np.isfinite(float(row["elapsed_s"])) and float(row["elapsed_s"]) >= 0.0 for row in summaries.values()),
            )

            validated = validate_source_bundle(load_source_bundle(outputs.source_manifest))
            check("raw manifest validates as source bundle", 2, validated.run_count)

            bad_payload = json.loads(json.dumps(payload))
            bad_payload["raw_source_prepare"]["epoch_count"] += 1
            outputs.source_manifest.write_text(json.dumps(bad_payload), encoding="utf-8")
            try:
                with redirect_stderr(StringIO()):
                    validate_source_bundle(load_source_bundle(outputs.source_manifest))
                check("raw manifest rejects bad total count", True, False)
            except SystemExit as exc:
                check("raw manifest rejects bad total count", True, exc.code != 0)

            bad_payload = json.loads(json.dumps(payload))
            bad_payload["raw_source_prepare"]["runs"][0]["window_count"] += 1
            outputs.source_manifest.write_text(json.dumps(bad_payload), encoding="utf-8")
            try:
                with redirect_stderr(StringIO()):
                    validate_source_bundle(load_source_bundle(outputs.source_manifest))
                check("raw manifest rejects bad per-run count", True, False)
            except SystemExit as exc:
                check("raw manifest rejects bad per-run count", True, exc.code != 0)
    finally:
        raw_source_prepare._load_model_feature_names = original_load_model_feature_names
        raw_source_prepare._epoch_rows_for_run = original_epoch_rows_for_run


def test_raw_source_prepare_window_rows() -> None:
    print("test_raw_source_prepare_window_rows")
    epoch_rows = [
        {
            "city": "nagoya",
            "run": "run1",
            "epoch": 0,
            "gps_tow": 100.0,
            "sat": 6.0,
            "phase_fraction": 0.5,
            "los_fraction": 0.75,
            "rinex_phase_jump_ge0p5cy_count": 0.0,
            "snr_p50": 32.0,
        },
        {
            "city": "nagoya",
            "run": "run1",
            "epoch": 1,
            "gps_tow": 110.0,
            "sat": 8.0,
            "phase_fraction": 0.75,
            "los_fraction": 0.875,
            "rinex_phase_jump_ge0p5cy_count": 1.0,
            "snr_p50": 34.0,
        },
        {
            "city": "nagoya",
            "run": "run1",
            "epoch": 2,
            "gps_tow": 135.0,
            "sat": 7.0,
            "phase_fraction": 1.0,
            "los_fraction": 1.0,
            "rinex_phase_jump_ge0p5cy_count": 0.0,
            "snr_p50": 36.0,
        },
    ]
    model_features = [
        "sat_mean",
        "phase_fraction_mean",
        "los_fraction_mean",
        "rinex_phase_jump_ge0p5cy_count_max",
        "sat_delta_pos_count",
        "validation_pass_frac",
        "unsupported_model_feature",
    ]
    window_rows, base_rows = build_window_rows(
        epoch_rows,
        model_feature_names=model_features,
        window_duration_s=30.0,
    )
    check("raw prepare window count", 2, len(window_rows))
    check("raw prepare base count", 2, len(base_rows))
    first = window_rows[0]
    check("raw prepare first key", ("nagoya", "run1", 0), (first["city"], first["run"], first["window_index"]))
    check("raw prepare sat mean", 7.0, first["sat_mean"])
    check("raw prepare phase fraction mean", 0.625, first["phase_fraction_mean"])
    check("raw prepare jump max", 1.0, first["rinex_phase_jump_ge0p5cy_count_max"])
    check("raw prepare delta pos count", 1, first["sat_delta_pos_count"])
    check("raw prepare neutral fill", 0.0, first["unsupported_model_feature"])
    check("raw prepare defers validationhold", False, "validation_pass_frac" in first)
    check("raw prepare base prediction column", True, "corrected_pred_fix_rate_pct" in base_rows[0])
    check("raw prepare no labels", False, "actual_fix_rate_pct" in first)


def test_raw_source_prepare_streaming_epoch_rows() -> None:
    print("test_raw_source_prepare_streaming_epoch_rows")

    def rinex_line(prefix: str, label: str) -> str:
        return f"{prefix:<60}{label}\n"

    def obs_line(sat_id: str, pseudorange: float, phase: float, doppler: float, snr: float) -> str:
        return (
            f"{sat_id:<3}"
            f"{pseudorange:14.3f}  "
            f"{phase:14.3f}  "
            f"{doppler:14.3f}  "
            f"{snr:14.3f}  "
            "\n"
        )

    t0 = datetime(2024, 1, 7, 0, 0, 10)
    t1 = datetime(2024, 1, 7, 0, 0, 10, 200000)
    tow0 = _datetime_to_gps_seconds_of_week(t0)
    tow1 = _datetime_to_gps_seconds_of_week(t1)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "nagoya" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "base.obs").write_text("fixture\n", encoding="utf-8")
        (run_dir / "base.nav").write_text("fixture\n", encoding="utf-8")
        (run_dir / "reference.csv").write_text(
            "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
            f"{tow0},1,2,3\n"
            f"{tow1},1,2,3\n",
            encoding="utf-8",
        )
        obs_header = (
            rinex_line("     3.04           OBSERVATION DATA    M", "RINEX VERSION / TYPE")
            + rinex_line("TEST", "MARKER NAME")
            + rinex_line("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES")
            + rinex_line("", "END OF HEADER")
        )
        epoch0 = (
            "> 2024 01 07 00 00 10.0000000  0  4\n"
            + obs_line("G01", 22_000_001.0, 100.0, -0.1, 35.0)
            + obs_line("G02", 22_000_002.0, 101.0, -0.1, 36.0)
            + obs_line("G03", 22_000_003.0, 102.0, -0.1, 37.0)
            + obs_line("G04", 22_000_004.0, 103.0, -0.1, 38.0)
        )
        epoch1 = (
            "> 2024 01 07 00 00 10.2000000  0  4\n"
            + obs_line("G01", 22_000_001.5, 100.1, -0.5, 35.0)
            + obs_line("G02", 22_000_002.5, 101.1, -0.5, 36.0)
            + obs_line("G03", 22_000_003.5, 102.1, -0.5, 37.0)
            + obs_line("G04", 22_000_004.5, 103.1, -0.5, 38.0)
        )
        (run_dir / "rover.obs").write_text(obs_header + epoch0 + epoch1, encoding="utf-8")

        rows = _raw_epoch_rows_for_run(
            SourceRun(city="nagoya", run="run1", run_dir=run_dir),
            systems=("G",),
            max_epochs=None,
        )
    check("raw streaming epoch row count", 2, len(rows))
    check("raw streaming sat count", 4.0, rows[0]["sat"])
    check("raw streaming phase count", 4.0, rows[0]["rinex_phase_present_count"])
    check("raw streaming snr p50", 36.5, rows[0]["snr_p50"])
    check("raw streaming phase delta second row", 0.1, round(rows[1]["rinex_phase_raw_delta_cycles_p50"], 1))


def test_merge_base_prediction_column() -> None:
    print("test_merge_base_prediction_column")
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        window = tmpdir / "window.csv"
        base = tmpdir / "base.csv"
        out = tmpdir / "out.csv"
        window.write_text(
            "city,run,window_index,window_start_tow,window_end_tow,sim_matched_epochs,feature\n"
            "tokyo,run1,0,1,2,2,7\n",
            encoding="utf-8",
        )
        base.write_text(
            "city,run,window_index,corrected_pred_fix_rate_pct\n"
            "tokyo,run1,0,12.5\n",
            encoding="utf-8",
        )
        _merge_base_prediction_column(window_csv=window, base_prediction_csv=base, output_csv=out)
        rows = list(DictReader(out.open(newline="", encoding="utf-8")))
        check("base merge rows", 1, len(rows))
        check("base merge value", "12.5", rows[0]["base_pred_fix_rate_pct"])


def test_validationhold_window_rows_label_free() -> None:
    print("test_validationhold_window_rows_label_free")
    windows = pd.DataFrame(
        [
            {
                "city": "tokyo",
                "run": "run1",
                "window_index": 0,
                "window_start_tow": 10.0,
                "window_end_tow": 11.0,
                "base_pred_fix_rate_pct": 12.0,
            }
        ]
    )
    epochs = pd.DataFrame(
        [
            {
                "city": "tokyo",
                "run": "run1",
                "gps_tow": 10.5,
                "validation_pass": 1.0,
                "validation_soft_pass": 1.0,
                "validation_hard_block": 0.0,
                "validation_severe_block": 0.0,
                "validation_reject_block": 0.0,
                "validation_block_spike": 0.0,
                "validation_block_score": 0.0,
                "validation_block_ewma_30s": 0.0,
                "validation_block_cooldown_s": 0.0,
                "validation_reject_recent_s": 0.0,
                "validation_quality_score": 7.0,
                "hold_state": 1.0,
                "hold_ready": 1.0,
                "hold_strict_ready": 1.0,
                "hold_carry_score": 3.0,
                "hold_age_s": 20.0,
                "hold_since_reset_s": 20.0,
                "clean_streak_s": 20.0,
                "strict_clean_streak_s": 20.0,
            }
        ]
    )
    rows = _validationhold_window_rows(windows, epochs, "base_pred_fix_rate_pct")
    check("label-free validationhold rows", 1, len(rows))
    check("label-free validationhold omits actual", False, "actual_fix_rate_pct" in rows[0])
    check("label-free validationhold omits demo5", False, "demo5_fix_rate_pct" in rows[0])
    check("label-free validationhold has lift flag", True, "validationhold_low_pred_lift_signal" in rows[0])


def test_product_inference_label_free_output() -> None:
    print("test_product_inference_label_free_output")
    frame = pd.DataFrame(
        [
            {
                "city": "tokyo",
                "run": "run1",
                "window_index": 0,
                "sim_matched_epochs": 2,
                "base_pred_fix_rate_pct": 10.0,
            },
            {
                "city": "tokyo",
                "run": "run1",
                "window_index": 1,
                "sim_matched_epochs": 1,
                "base_pred_fix_rate_pct": 40.0,
            },
        ]
    )
    base = _base_from_frame_or_reference(frame, base_prefix=None, base_prediction_column="corrected_pred_fix_rate_pct")
    rows = _product_prediction_rows(
        df=frame,
        base=base,
        residual=pd.Series([0.01, -0.02]).to_numpy(),
        corrected=pd.Series([0.11, 0.38]).to_numpy(),
        probabilities={"ratio_mean_ge3": pd.Series([0.2, 0.8]).to_numpy()},
    )
    route = _route_rows(rows)
    check("label-free window omits actual", False, "actual_fix_rate_pct" in rows[0])
    check("label-free window omits error", False, "corrected_error_pp" in rows[0])
    check("label-free route count", 1, len(route))
    check("label-free route omits actual", False, "actual_fix_rate_pct" in route[0])
    check("label-free weighted route pred", 20.0, float(route[0]["adopted_pred_fix_rate_pct"]))


def test_product_inference_final_calibrator() -> None:
    print("test_product_inference_final_calibrator")
    corrected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    unchanged = _apply_final_calibrator(corrected, {})
    check("final calibrator absent leaves prediction unchanged", [0.0, 0.5, 1.0], unchanged.tolist())

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.25, 1.0]))
    calibrated = _apply_final_calibrator(
        corrected,
        {"final_calibrator_name": "isotonic", "final_calibrator": model},
    )
    check("final isotonic calibrator applies mapping", [0.0, 0.25, 1.0], calibrated.tolist())
    blended = _apply_final_calibrator(
        corrected,
        {"final_calibrator_name": "isotonic", "final_calibrator": model, "final_calibrator_blend": 0.5},
    )
    check("final isotonic calibrator supports blending", [0.0, 0.375, 1.0], blended.tolist())


def test_product_inference_prediction_guards() -> None:
    print("test_product_inference_prediction_guards")
    frame = pd.DataFrame(
        {
            "rinex_phase_raw_delta_cycles_p50_p75": [500.0, 100.0, 500.0],
        }
    )
    corrected = np.array([0.55, 0.55, 0.10], dtype=np.float64)
    guarded = _apply_prediction_guards(
        frame,
        corrected,
        {
            "prediction_guards": [
                {
                    "name": "phase_delta_cap20",
                    "feature": "rinex_phase_raw_delta_cycles_p50_p75",
                    "operator": ">=",
                    "threshold": 426.419,
                    "cap": 0.20,
                }
            ]
        },
    )
    check("prediction guard caps only active high predictions", [0.2, 0.55, 0.1], guarded.tolist())


def main() -> None:
    test_classify_window()
    test_is_metadata_or_label()
    test_confidence_tier()
    test_actionability_labels()
    test_require()
    test_prediction_contract()
    test_base_prediction_path()
    test_prediction_mode_stage_counts()
    test_online_classifier_meta_matches_batch_run_position()
    test_batch_output_paths()
    test_product_source_bundle_validation()
    test_raw_source_prepare_manifest_metadata_counts()
    test_raw_source_prepare_window_rows()
    test_raw_source_prepare_streaming_epoch_rows()
    test_merge_base_prediction_column()
    test_validationhold_window_rows_label_free()
    test_product_inference_label_free_output()
    test_product_inference_final_calibrator()
    test_product_inference_prediction_guards()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("\nall tests pass")


if __name__ == "__main__":
    main()
