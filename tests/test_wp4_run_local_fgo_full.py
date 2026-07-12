"""Unit tests for experiments/wp4_run_local_fgo_full.py (synthetic, no dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from wp4_run_local_fgo_full import (  # noqa: E402
    ecef_rows_to_llh_rows,
    fill_seed_gaps,
    load_seed_ecef_csv,
    make_windows,
    parse_rover_tows_from_obs,
    read_pos_ecef,
    recover_fix_mask,
    write_pos_file,
    write_trajectory_csv,
)


_ROVER_OBS_HEADER = """     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
G    4 C1C L1C D1C S1C                                    SYS / # / OBS TYPES
                                                            END OF HEADER
"""


def _write_rover_obs(path: Path, dates: list[tuple[int, int, int, int, int, float]]) -> None:
    lines = [_ROVER_OBS_HEADER]
    for y, mo, d, h, mi, s in dates:
        lines.append(f"> {y:04d} {mo:02d} {d:02d} {h:02d} {mi:02d} {s:11.7f}  0  1\n")
        lines.append("G01  20000000.000 8  105000000.000 8       -100.000 8     45.000\n")
    path.write_text("".join(lines), encoding="ascii")


def test_parse_rover_tows_from_obs_reads_epoch_headers_only(tmp_path: Path) -> None:
    path = tmp_path / "rover.obs"
    # Three consecutive 0.2 s epochs on 2024-07-23 (a Tuesday -> GPS DOW=2).
    _write_rover_obs(
        path,
        [
            (2024, 7, 23, 4, 4, 30.0),
            (2024, 7, 23, 4, 4, 30.2),
            (2024, 7, 23, 4, 4, 30.4),
        ],
    )
    tows = parse_rover_tows_from_obs(path)
    assert tows.shape == (3,)
    assert np.allclose(np.diff(tows), 0.2)


def test_load_seed_ecef_csv_rounds_tow_and_reads_ecef(tmp_path: Path) -> None:
    path = tmp_path / "backbone.csv"
    path.write_text(
        "tow,lat_deg,lon_deg,height_m,ecef_x,ecef_y,ecef_z,fix\n"
        "100.00,0,0,0,1.0,2.0,3.0,0\n"
        "100.20,0,0,0,4.0,5.0,6.0,0\n",
        encoding="utf-8",
    )
    seed = load_seed_ecef_csv(path)
    assert set(seed.keys()) == {100.0, 100.2}
    assert np.allclose(seed[100.0], [1.0, 2.0, 3.0])


def test_fill_seed_gaps_interpolates_missing_midpoint() -> None:
    tows = np.array([100.0, 100.2, 100.4, 100.6, 100.8])
    seed = {
        100.0: np.array([0.0, 0.0, 0.0]),
        100.4: np.array([2.0, 0.0, 0.0]),
        100.8: np.array([4.0, 0.0, 0.0]),
    }
    positions, is_interp = fill_seed_gaps(tows, seed)
    assert list(is_interp) == [False, True, False, True, False]
    np.testing.assert_allclose(positions[1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[3], [3.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[4], [4.0, 0.0, 0.0])


def test_fill_seed_gaps_extrapolates_edges_as_constant() -> None:
    tows = np.array([99.8, 100.0, 100.2, 100.4])
    seed = {100.0: np.array([0.0, 0.0, 0.0]), 100.2: np.array([1.0, 0.0, 0.0])}
    positions, is_interp = fill_seed_gaps(tows, seed)
    assert list(is_interp) == [True, False, False, True]
    np.testing.assert_allclose(positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[3], [1.0, 0.0, 0.0])


def test_make_windows_partitions_contiguously_with_short_tail() -> None:
    tows = np.arange(0.0, 1.0, 0.2)  # 5 epochs: 0.0, 0.2, 0.4, 0.6, 0.8
    windows = make_windows(tows, window_epochs=2)
    assert windows == [
        (pytest.approx(0.0), pytest.approx(0.2)),
        (pytest.approx(0.4), pytest.approx(0.6)),
        (pytest.approx(0.8), pytest.approx(0.8)),
    ]


def test_recover_fix_mask_detects_changed_rows() -> None:
    seed = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    fixed_only = np.array([[1.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    mask = recover_fix_mask(seed, fixed_only)
    assert list(mask) == [False, True, False]


def test_recover_fix_mask_requires_matching_shapes() -> None:
    with pytest.raises(ValueError):
        recover_fix_mask(np.zeros((2, 3)), np.zeros((3, 3)))


def test_ecef_rows_to_llh_rows_round_trips_known_point() -> None:
    # Approximate ECEF for a point near the equator/prime meridian at ~0m height.
    tows = np.array([1.0])
    ecef = np.array([[6378137.0, 0.0, 0.0]])
    fix_mask = np.array([True])
    rows = ecef_rows_to_llh_rows(tows, ecef, fix_mask)
    assert len(rows) == 1
    row = rows[0]
    assert row["fix"] == 1
    assert abs(row["lat_deg"]) < 1e-6
    assert abs(row["lon_deg"]) < 1e-6
    assert abs(row["height_m"]) < 1.0


def test_write_pos_file_roundtrips_through_hybrid_loader(tmp_path: Path) -> None:
    # solve_ppc_segment_multifamily_fgo.py loads --seed-pos via
    # exp_ppc_ctrbpf_fgo._load_hybrid_pos_file, which requires a nonzero
    # status at parts[8] (real RTKLIB Q column) to keep a row; verify our
    # writer's column order satisfies that (see write_pos_file docstring for
    # why exp_ppc_ctrbpf_fgo's own _write_pos_file does not).
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from exp_ppc_ctrbpf_fgo import _load_hybrid_pos_file

    path = tmp_path / "seed.pos"
    tows = np.array([100.0, 100.2])
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    write_pos_file(path, tows, positions, status=5)
    by_tow, statuses = _load_hybrid_pos_file(path)
    assert set(by_tow.keys()) == {100.0, 100.2}
    np.testing.assert_allclose(by_tow[100.0], [1.0, 2.0, 3.0])
    assert statuses[100.0] == 5


def test_write_pos_file_zero_status_rows_are_dropped_by_hybrid_loader(tmp_path: Path) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from exp_ppc_ctrbpf_fgo import _load_hybrid_pos_file

    path = tmp_path / "seed_zero.pos"
    write_pos_file(path, np.array([100.0]), np.array([[1.0, 2.0, 3.0]]), status=0)
    by_tow, _ = _load_hybrid_pos_file(path)
    assert by_tow == {}


def test_read_pos_ecef_is_status_independent(tmp_path: Path) -> None:
    # Round-trip through the solver's own writer (exp_ppc_ctrbpf_fgo._write_pos_file)
    # to confirm read_pos_ecef recovers positions regardless of that writer's
    # status-column mismatch.
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from exp_ppc_ctrbpf_fgo import _write_pos_file

    path = tmp_path / "solver_out.pos"
    tows = np.array([100.0, 100.2])
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _write_pos_file(path, tows, positions, status=5)
    by_tow = read_pos_ecef(path)
    assert set(by_tow.keys()) == {100.0, 100.2}
    np.testing.assert_allclose(by_tow[100.2], [4.0, 5.0, 6.0])


def test_write_trajectory_csv_writes_expected_header(tmp_path: Path) -> None:
    path = tmp_path / "traj.csv"
    rows = ecef_rows_to_llh_rows(
        np.array([1.0]),
        np.array([[6378137.0, 0.0, 0.0]]),
        np.array([False]),
    )
    write_trajectory_csv(path, rows)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "tow,lat_deg,lon_deg,height_m,ecef_x,ecef_y,ecef_z,fix"
