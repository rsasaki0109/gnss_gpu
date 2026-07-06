"""Unit tests for experiments/sweep_libgnss_rtk_wp6.py's pure helpers.

Only the argv-building, stdout-parsing, and fix-time-distribution logic is
covered here (no WSL/gnss_solve invocation, no filesystem RTK runs) --
per TASK_H's constraint that "C++/config changes are validated by the
score, not unit tests", this driver's own orchestration logic is what gets
tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from score_vs_inuex35 import TrajectoryEpoch  # noqa: E402
from sweep_libgnss_rtk_wp6 import (  # noqa: E402
    Candidate,
    build_gnss_solve_argv,
    compute_fix_time_distribution,
    parse_engine_summary,
    to_wsl_path,
)


def test_to_wsl_path_maps_drive_letter() -> None:
    result = to_wsl_path(Path("C:/Users/rsasa/foo/bar.pos"))
    assert result == "/mnt/c/Users/rsasa/foo/bar.pos"


def test_to_wsl_path_lowercases_drive() -> None:
    result = to_wsl_path(Path("E:/datasets/PPC-Dataset-data/tokyo/run1"))
    assert result.startswith("/mnt/e/")


def test_build_gnss_solve_argv_basic_shape() -> None:
    argv = build_gnss_solve_argv(
        gnss_solve_path=Path("C:/gnss/gnss_solve"),
        rover=Path("E:/data/rover.obs"),
        base=Path("E:/data/base.obs"),
        nav=Path("E:/data/base.nav"),
        out_pos=Path("C:/out/run.pos"),
        extra_args=["--ratio", "2.4"],
    )
    assert argv[0] == "wsl"
    assert argv[1] == "/mnt/c/gnss/gnss_solve"
    assert "--rover" in argv and "/mnt/e/data/rover.obs" in argv
    assert "--preset" in argv and "low-cost" in argv
    assert argv[-2:] == ["--ratio", "2.4"]
    # skip-epochs always present, max-epochs omitted when zero (default)
    assert "--skip-epochs" in argv
    assert "--max-epochs" not in argv


def test_build_gnss_solve_argv_includes_skip_and_max_epochs() -> None:
    argv = build_gnss_solve_argv(
        gnss_solve_path=Path("C:/gnss/gnss_solve"),
        rover=Path("E:/data/rover.obs"),
        base=Path("E:/data/base.obs"),
        nav=Path("E:/data/base.nav"),
        out_pos=Path("C:/out/run.pos"),
        extra_args=[],
        skip_epochs=4000,
        max_epochs=4000,
    )
    skip_idx = argv.index("--skip-epochs")
    assert argv[skip_idx + 1] == "4000"
    max_idx = argv.index("--max-epochs")
    assert argv[max_idx + 1] == "4000"


def test_build_gnss_solve_argv_extra_args_appended_last() -> None:
    """Later CLI flags win in gnss_solve's real argv parser; extra_args must
    come after the base --preset low-cost so overrides (e.g. --no-arfilter)
    take effect."""
    argv = build_gnss_solve_argv(
        gnss_solve_path=Path("C:/gnss/gnss_solve"),
        rover=Path("E:/data/rover.obs"),
        base=Path("E:/data/base.obs"),
        nav=Path("E:/data/base.nav"),
        out_pos=Path("C:/out/run.pos"),
        extra_args=["--no-arfilter"],
    )
    preset_idx = argv.index("--preset")
    noaf_idx = argv.index("--no-arfilter")
    assert noaf_idx > preset_idx


_SAMPLE_STDOUT = """libgnss++ post-process solver
  rover: /mnt/e/datasets/PPC-Dataset-data/tokyo/run1/rover.obs
  mode: kinematic (requested auto)

Summary
  total solutions: 7397
  valid solutions: 7397
  fixed solutions: 775
  fix rate: 10.48%
  exact base epochs: 2388
  interpolated base epochs: 9540
  skipped rover epochs: 0
  non-FIX drift guard: enabled inspected_segments=0 rejected_segments=0 rejected_epochs=0
"""


def test_parse_engine_summary_extracts_counts() -> None:
    summary = parse_engine_summary(_SAMPLE_STDOUT)
    assert summary["total_solutions"] == 7397.0
    assert summary["valid_solutions"] == 7397.0
    assert summary["fixed_solutions"] == 775.0
    assert summary["exact_base_epochs"] == 2388.0
    assert summary["interpolated_base_epochs"] == 9540.0
    assert summary["skipped_rover_epochs"] == 0.0


def test_parse_engine_summary_extracts_fix_rate_pct() -> None:
    summary = parse_engine_summary(_SAMPLE_STDOUT)
    assert abs(summary["engine_fix_rate_pct"] - 10.48) < 1e-9


def test_parse_engine_summary_empty_on_garbage() -> None:
    summary = parse_engine_summary("no useful output here\nerror: something broke\n")
    assert summary == {}


def _epoch(tow: float, is_fix: bool) -> TrajectoryEpoch:
    import numpy as np

    return TrajectoryEpoch(tow=tow, ecef=np.zeros(3), is_fix=is_fix)


def test_compute_fix_time_distribution_all_fixes_early() -> None:
    epochs = [_epoch(1000.0 + i, True) for i in range(10)]
    dist = compute_fix_time_distribution(epochs, run_start_tow=1000.0, warmup_s=300.0)
    assert dist["n_fix"] == 10
    assert dist["n_fix_after_warmup"] == 0
    assert dist["frac_fix_after_warmup"] == 0.0


def test_compute_fix_time_distribution_all_fixes_late() -> None:
    epochs = [_epoch(1000.0 + 400.0 + i, True) for i in range(10)]
    dist = compute_fix_time_distribution(epochs, run_start_tow=1000.0, warmup_s=300.0)
    assert dist["n_fix_after_warmup"] == 10
    assert dist["frac_fix_after_warmup"] == 1.0


def test_compute_fix_time_distribution_mixed_and_non_fix_ignored() -> None:
    epochs = [
        _epoch(1000.0, True),  # before warmup
        _epoch(1000.0, False),  # non-fix, should never be counted
        _epoch(1500.0, True),  # after warmup (500s in)
        _epoch(1500.0, False),
    ]
    dist = compute_fix_time_distribution(epochs, run_start_tow=1000.0, warmup_s=300.0)
    assert dist["n_fix"] == 2
    assert dist["n_fix_after_warmup"] == 1
    assert dist["frac_fix_after_warmup"] == 0.5


def test_compute_fix_time_distribution_no_fix_epochs() -> None:
    epochs = [_epoch(1000.0, False), _epoch(1100.0, False)]
    dist = compute_fix_time_distribution(epochs, run_start_tow=1000.0)
    assert dist["n_fix"] == 0
    assert dist["frac_fix_after_warmup"] == 0.0


def test_candidate_dataclass_defaults() -> None:
    c = Candidate("name_only")
    assert c.extra_args == []
    assert c.note == ""
