"""Unit tests for experiments/score_vs_inuex35.py (synthetic, no dataset/GPU)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from score_vs_inuex35 import (  # noqa: E402
    ScoreResult,
    load_csv_trajectory,
    load_npz_trajectory,
    load_pos_trajectory,
    score_trajectory,
)


def _make_reference() -> dict[float, np.ndarray]:
    return {
        100.0: np.array([0.0, 0.0, 0.0], dtype=np.float64),
        100.2: np.array([1.0, 0.0, 0.0], dtype=np.float64),
        100.4: np.array([2.0, 0.0, 0.0], dtype=np.float64),
        100.6: np.array([3.0, 0.0, 0.0], dtype=np.float64),
        100.8: np.array([4.0, 0.0, 0.0], dtype=np.float64),
    }


def _expected_metrics() -> dict[str, float | int | None]:
    # Four scored epochs; rover denominator is five (one gap).
    errors = np.array([0.3, 0.4, 1.0, 0.6], dtype=np.float64)
    fix_errors = np.array([0.3, 1.0], dtype=np.float64)
    return {
        "n_scored": 4,
        "n_rover_epochs": 5,
        "coverage_pct": 80.0,
        "n_fix": 2,
        "all_rms_m": float(np.sqrt(np.mean(errors**2))),
        "fix_rms_m": float(np.sqrt(np.mean(fix_errors**2))),
        "fix_pct": 50.0,
        "lt50cm_pct": 50.0,
        "lt50cm_full_pct": 40.0,
    }


def _write_pos(path: Path) -> None:
    # tow, ECEF offset from reference, Status 4=FIX (libgnss++ default).
    rows = [
        (100.0, 0.3, 0.0, 0.0, 4),
        (100.2, 0.4, 0.0, 0.0, 3),
        (100.4, 1.0, 0.0, 0.0, 4),
        (100.6, 0.6, 0.0, 0.0, 3),
    ]
    lines = ["% RTKLIB test pos\n"]
    for tow, dx, dy, dz, q in rows:
        x = {"100.0": 0.0, "100.2": 1.0, "100.4": 2.0, "100.6": 3.0}[f"{tow:.1f}"]
        lines.append(
            f"2324 {tow:.3f} {x + dx:.6f} {dy:.6f} {dz:.6f} "
            f"35.0 139.0 40.0 {q} 12 1.5\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["tow", "ecef_x", "ecef_y", "ecef_z", "fix"],
        )
        writer.writeheader()
        writer.writerow(
            {"tow": 100.0, "ecef_x": 0.3, "ecef_y": 0.0, "ecef_z": 0.0, "fix": "1"}
        )
        writer.writerow(
            {"tow": 100.2, "ecef_x": 1.4, "ecef_y": 0.0, "ecef_z": 0.0, "fix": "0"}
        )
        writer.writerow(
            {"tow": 100.4, "ecef_x": 3.0, "ecef_y": 0.0, "ecef_z": 0.0, "fix": "1"}
        )
        writer.writerow(
            {"tow": 100.6, "ecef_x": 3.6, "ecef_y": 0.0, "ecef_z": 0.0, "fix": "0"}
        )


def _write_npz(path: Path) -> None:
    np.savez(
        path,
        tow=np.array([100.0, 100.2, 100.4, 100.6], dtype=np.float64),
        sol_xyz=np.array(
            [
                [0.3, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.6, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        smode=np.array([4, 3, 4, 3], dtype=np.int32),
    )


def _assert_result(result: ScoreResult, expected: dict[str, float | int | None]) -> None:
    assert result.n_scored == expected["n_scored"]
    assert result.n_rover_epochs == expected["n_rover_epochs"]
    assert result.coverage_pct == pytest.approx(expected["coverage_pct"])
    assert result.n_fix == expected["n_fix"]
    assert result.all_rms_m == pytest.approx(expected["all_rms_m"])
    assert result.fix_rms_m == pytest.approx(expected["fix_rms_m"])
    assert result.fix_pct == pytest.approx(expected["fix_pct"])
    assert result.lt50cm_pct == pytest.approx(expected["lt50cm_pct"])
    assert result.lt50cm_full_pct == pytest.approx(expected["lt50cm_full_pct"])
    assert np.isfinite(result.ppc_official_pct)


@pytest.mark.parametrize(
    ("writer", "loader", "suffix", "fmt"),
    [
        (_write_pos, load_pos_trajectory, ".pos", "pos"),
        (_write_csv, load_csv_trajectory, ".csv", "csv"),
        (_write_npz, load_npz_trajectory, ".npz", "npz"),
    ],
)
def test_score_all_formats(
    tmp_path: Path,
    writer,
    loader,
    suffix: str,
    fmt: str,
) -> None:
    reference = _make_reference()
    traj_path = tmp_path / f"traj{suffix}"
    writer(traj_path)
    epochs = loader(traj_path)
    result = score_trajectory(
        epochs,
        reference,
        city="synthetic",
        run="run1",
        traj_path=traj_path,
        fmt=fmt,
    )
    _assert_result(result, _expected_metrics())


def test_unmatched_epoch_reduces_coverage(tmp_path: Path) -> None:
    reference = _make_reference()
    traj_path = tmp_path / "sparse.pos"
    traj_path.write_text(
        "% pos\n2324 100.000 0.300000 0.000000 0.000000 35.0 139.0 40.0 4 12 1.5\n",
        encoding="utf-8",
    )
    epochs = load_pos_trajectory(traj_path)
    result = score_trajectory(
        epochs,
        reference,
        city="synthetic",
        run="run1",
        traj_path=traj_path,
        fmt="pos",
    )
    assert result.n_scored == 1
    assert result.coverage_pct == pytest.approx(20.0)
    assert result.lt50cm_full_pct == pytest.approx(20.0)


def test_pos_default_counts_status_4_as_fix(tmp_path: Path) -> None:
    traj_path = tmp_path / "libgnss.pos"
    traj_path.write_text(
        "% libgnss\n"
        "2324 100.000 0.0 0.0 0.0 35.0 139.0 40.0 4 12 1.5\n"
        "2324 100.200 0.0 0.0 0.0 35.0 139.0 40.0 3 12 1.5\n"
        "2324 100.400 0.0 0.0 0.0 35.0 139.0 40.0 1 12 1.5\n",
        encoding="utf-8",
    )
    epochs = load_pos_trajectory(traj_path)
    assert [epoch.is_fix for epoch in epochs] == [True, False, False]


def test_pos_rtklib_fix_statuses_override(tmp_path: Path) -> None:
    traj_path = tmp_path / "rtklib.pos"
    traj_path.write_text(
        "% rtklib\n"
        "2324 100.000 0.0 0.0 0.0 35.0 139.0 40.0 1 12 1.5\n"
        "2324 100.200 0.0 0.0 0.0 35.0 139.0 40.0 4 12 1.5\n",
        encoding="utf-8",
    )
    epochs = load_pos_trajectory(traj_path, fix_statuses=frozenset({1}))
    assert [epoch.is_fix for epoch in epochs] == [True, False]


def test_lt50cm_full_denominator_uses_rover_epoch_count(tmp_path: Path) -> None:
    reference = _make_reference()
    traj_path = tmp_path / "gap.pos"
    _write_pos(traj_path)
    epochs = load_pos_trajectory(traj_path)
    result = score_trajectory(
        epochs,
        reference,
        city="tokyo",
        run="run1",
        traj_path=traj_path,
        fmt="pos",
    )
    assert result.n_rover_epochs == 11928
    assert result.lt50cm_pct == pytest.approx(50.0)
    assert result.lt50cm_full_pct == pytest.approx(100.0 * 2 / 11928)
