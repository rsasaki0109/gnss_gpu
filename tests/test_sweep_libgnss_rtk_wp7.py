"""Unit tests for experiments/sweep_libgnss_rtk_wp7.py's pure helpers.

Mirrors tests/test_sweep_libgnss_rtk_wp6.py's scope: only argv-building and
segment-scoring logic that doesn't require WSL/gnss_solve is covered here.
The engine-level NLOS weighting and dead-knob wiring is validated by the C++
unit tests (test_nlos_weights.cpp, test_rtk_smoke.cpp) and by score deltas on
real PPC runs, not by mocking the engine here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from score_vs_inuex35 import TrajectoryEpoch  # noqa: E402
from sweep_libgnss_rtk_wp7 import (  # noqa: E402
    CANYON_TOW_HI,
    CANYON_TOW_LO,
    Candidate,
    WP6_WINNER_ARGS,
    build_full_argv,
    nlos_mask_path,
    score_segment,
)


def _epoch(tow: float, is_fix: bool = False) -> TrajectoryEpoch:
    return TrajectoryEpoch(tow=tow, ecef=np.array([1.0e6, 1.0e6, 1.0e6]), is_fix=is_fix)


def test_candidate_defaults() -> None:
    c = Candidate("name_only")
    assert c.extra_args == []
    assert c.note == ""
    assert c.needs_nlos_csv is False


def test_nlos_mask_path_matches_phase33_naming_contract() -> None:
    path = nlos_mask_path("tokyo", "run1")
    assert path.name == "tokyo_run1_per_epoch_nlos.csv"
    assert path.parent.name == "plateau_nlos_phase33"


def test_build_full_argv_includes_wp6_winner_base_before_candidate_args() -> None:
    argv = build_full_argv(
        gnss_solve_path=Path("C:/gnss/gnss_solve"),
        rover=Path("E:/data/rover.obs"),
        base=Path("E:/data/base.obs"),
        nav=Path("E:/data/base.nav"),
        out_pos=Path("C:/out/run.pos"),
        candidate_extra_args=["--nlos-weight-mode", "two-tier"],
        nlos_csv=None,
    )
    winner_idx = argv.index(WP6_WINNER_ARGS[0])
    mode_idx = argv.index("--nlos-weight-mode")
    assert mode_idx > winner_idx
    assert argv[-2:] == ["--nlos-weight-mode", "two-tier"]


def test_build_full_argv_inserts_nlos_weights_before_candidate_args() -> None:
    argv = build_full_argv(
        gnss_solve_path=Path("C:/gnss/gnss_solve"),
        rover=Path("E:/data/rover.obs"),
        base=Path("E:/data/base.obs"),
        nav=Path("E:/data/base.nav"),
        out_pos=Path("C:/out/run.pos"),
        candidate_extra_args=["--nlos-weight-mode", "continuous"],
        nlos_csv=Path("E:/nlos/tokyo_run1_per_epoch_nlos.csv"),
    )
    assert "--nlos-weights" in argv
    weights_idx = argv.index("--nlos-weights")
    assert argv[weights_idx + 1] == "/mnt/e/nlos/tokyo_run1_per_epoch_nlos.csv"
    # candidate's own extra_args must still come after --nlos-weights so a
    # candidate could in principle override it (argv-order-wins parsing).
    mode_idx = argv.index("--nlos-weight-mode")
    assert mode_idx > weights_idx


def test_build_full_argv_omits_nlos_weights_when_csv_is_none() -> None:
    argv = build_full_argv(
        gnss_solve_path=Path("C:/gnss/gnss_solve"),
        rover=Path("E:/data/rover.obs"),
        base=Path("E:/data/base.obs"),
        nav=Path("E:/data/base.nav"),
        out_pos=Path("C:/out/run.pos"),
        candidate_extra_args=[],
        nlos_csv=None,
    )
    assert "--nlos-weights" not in argv


def test_canyon_segment_bounds_match_task_spec() -> None:
    # WP7 task spec's urban-canyon segment on tokyo/run1.
    assert CANYON_TOW_LO == 188990.0
    assert CANYON_TOW_HI == 189070.0


def test_score_segment_filters_to_tow_window() -> None:
    epochs = [
        _epoch(188000.0, True),  # before window
        _epoch(189000.0, True),  # inside window
        _epoch(189050.0, False),  # inside window
        _epoch(190000.0, True),  # after window
    ]
    reference = {
        188000.0: np.array([1.0e6, 1.0e6, 1.0e6]),
        189000.0: np.array([1.0e6, 1.0e6, 1.0e6]),
        189050.0: np.array([1.0e6, 1.0e6, 1.0e6]),
        190000.0: np.array([1.0e6, 1.0e6, 1.0e6]),
    }
    result = score_segment(
        epochs, reference, city="tokyo", run="run1", traj_path=Path("dummy.pos"),
        tow_lo=CANYON_TOW_LO, tow_hi=CANYON_TOW_HI,
    )
    assert result is not None
    assert result.n_scored == 2


def test_score_segment_returns_none_when_window_empty() -> None:
    epochs = [_epoch(1000.0, True), _epoch(2000.0, True)]
    result = score_segment(
        epochs, {}, city="tokyo", run="run1", traj_path=Path("dummy.pos"),
        tow_lo=CANYON_TOW_LO, tow_hi=CANYON_TOW_HI,
    )
    assert result is None
