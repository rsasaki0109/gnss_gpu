#!/usr/bin/env python3
"""Smoke tests for the PPC FIX-rate predictor deliverable helpers.

Run directly: `python3 experiments/test_product_deliverable.py`
Zero external dependencies beyond `pandas` (already used by the
helpers); no pytest required.  Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import redirect_stderr
from csv import DictReader
from io import StringIO
from pathlib import Path

import pandas as pd

# Make the experiments directory importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import _is_metadata_or_label
from build_product_deliverable import (
    REQUIRED_PREDICTION_COLUMNS,
    _classify_window,
    _confidence_tier,
    _require_prediction_columns,
)
from analyze_ppc_validation_hold_surrogate_windows import _window_rows as _validationhold_window_rows
from predict import (
    DEFAULT_PREDICTION_CSV,
    RESULTS_DIR,
    _base_prediction_path,
    _merge_base_prediction_column,
    _prefix_output_path,
    _read_columns,
    _require,
)
from product_inference_model import (
    _base_from_frame_or_reference,
    _prediction_rows as _product_prediction_rows,
    _route_rows,
)
from train_ppc_solver_transition_surrogate_stack import _reference_prediction_path


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


def main() -> None:
    test_classify_window()
    test_is_metadata_or_label()
    test_confidence_tier()
    test_require()
    test_prediction_contract()
    test_base_prediction_path()
    test_merge_base_prediction_column()
    test_validationhold_window_rows_label_free()
    test_product_inference_label_free_output()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("\nall tests pass")


if __name__ == "__main__":
    main()
