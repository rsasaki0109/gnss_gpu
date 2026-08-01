from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments/evaluate_ppc_official_score.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ppc_official_score", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_evaluator_aligns_tow_and_penalizes_missing_epoch(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    with reference.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["GPS TOW (s)", "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"])
        writer.writerows([[1.0, 0, 0, 0], [1.2, 3, 0, 0], [1.4, 7, 0, 0]])
    estimate = tmp_path / "safe.csv"
    with estimate.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tow", "status", "x", "y", "z"])
        writer.writerows([[1.0, 3, 0, 0, 0], [1.2, 4, 3, 0, 0]])

    result = MODULE.evaluate_route(estimate, reference)

    assert result["matched_finite_epochs"] == 2
    assert result["missing_or_nonfinite_epochs"] == 1
    assert result["pass_distance_m"] == pytest.approx(3.0)
    assert result["total_distance_m"] == pytest.approx(7.0)
    assert result["ppc_score_pct"] == pytest.approx(300.0 / 7.0)
    assert result["fixed_epochs"] == 1
    assert result["correct_fix_epochs"] == 1
    assert result["false_fix_epochs"] == 0


def test_evaluator_reports_severe_false_fix(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "tow,x,y,z\n1.0,0,0,0\n1.2,1,0,0\n", encoding="utf-8"
    )
    estimate = tmp_path / "safe.csv"
    estimate.write_text(
        "tow,shadow_fixed,x,y,z\n1.0,0,0,0,0\n1.2,1,3,0,0\n", encoding="utf-8"
    )

    result = MODULE.evaluate_route(estimate, reference)

    assert result["false_fix_epochs"] == 1
    assert result["false_fix_above_1m_epochs"] == 1


def test_evaluator_reads_libgnss_pos_status(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_text("tow,x,y,z\n1.0,0,0,0\n1.2,1,0,0\n", encoding="utf-8")
    estimate = tmp_path / "solution.pos"
    estimate.write_text(
        "% header\n2324 1.0 0 0 0 0 0 0 3\n2324 1.2 3 0 0 0 0 0 4\n",
        encoding="utf-8",
    )

    result = MODULE.evaluate_route(estimate, reference)

    assert result["fixed_epochs"] == 1
    assert result["false_fix_epochs"] == 1
    assert result["false_fix_above_1m_epochs"] == 1


def test_evaluator_scores_legacy_pos_with_float_sigma_column(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_text("tow,x,y,z\n1.0,0,0,0\n1.2,1,0,0\n", encoding="utf-8")
    estimate = tmp_path / "legacy.pos"
    estimate.write_text(
        "% GPST tow x y z Q ns sdx sdy\n"
        "0 1.0 0 0 0 1 0 0.000 0.000\n"
        "0 1.2 1 0 0 1 0 0.000 0.000\n",
        encoding="utf-8",
    )

    result = MODULE.evaluate_route(estimate, reference)

    assert result["ppc_score_pct"] == 100.0
    assert result["fixed_epochs"] == 0


def test_evaluator_reads_native_fgo_dump_and_string_status(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    reference.write_text("tow,x,y,z\n1.0,0,0,0\n1.2,1,0,0\n", encoding="utf-8")
    estimate = tmp_path / "native.csv"
    estimate.write_text(
        "tow,status,x_ecef_m,y_ecef_m,z_ecef_m\n"
        "1.0,FIXED,0,0,0\n"
        "1.2,FLOAT,1,0,0\n",
        encoding="utf-8",
    )

    result = MODULE.evaluate_route(estimate, reference)

    assert result["ppc_score_pct"] == 100.0
    assert result["fixed_epochs"] == 1
    assert result["correct_fix_epochs"] == 1
