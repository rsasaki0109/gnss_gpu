#!/usr/bin/env python3
"""Smoke tests for the PPC FIX-rate predictor deliverable helpers.

Run directly: `python3 experiments/test_product_deliverable.py`
Zero external dependencies beyond `pandas` (already used by the
helpers); no pytest required.  Exits non-zero on the first failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

# Make the experiments directory importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import _is_metadata_or_label
from build_product_deliverable import _classify_window, _confidence_tier
from predict import _require


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
            _require(missing, "test_producer")
            check("_require exits on missing file", True, False)
        except SystemExit as exc:
            check("_require exits on missing file", True, exc.code != 0)


def main() -> None:
    test_classify_window()
    test_is_metadata_or_label()
    test_confidence_tier()
    test_require()
    if FAILURES:
        print(f"\n{len(FAILURES)} FAILURE(S):")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("\nall tests pass")


if __name__ == "__main__":
    main()
