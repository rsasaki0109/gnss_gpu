"""Unit tests for experiments/sweep_libgnss_rtk_wp9.py's pure helpers.

Mirrors tests/test_sweep_libgnss_rtk_wp8.py's scope: only candidate-building
logic that doesn't require WSL/gnss_solve is covered here. The engine-level
float-trust-policy behavior is validated by the C++ unit tests
(test_float_trust_policy.cpp, test_rtk_smoke.cpp) and by score deltas on
real PPC runs, not by mocking the engine here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_libgnss_rtk_wp9 import (  # noqa: E402
    QPOS_GRID,
    STAGE_CVPREDICT_COARSE,
    STAGE_SCALEDRESET_COARSE,
    combination_candidates,
    nlos_relax_candidates,
    policy_flag_args,
    regression_candidates,
    stretch_candidates,
)


def test_qpos_grid_spans_roughly_three_decades() -> None:
    values = sorted(float(v) for v in QPOS_GRID)
    assert values == [0.1, 1.0, 10.0, 100.0]


def test_cvpredict_coarse_grid_has_one_candidate_per_qpos() -> None:
    assert len(STAGE_CVPREDICT_COARSE) == len(QPOS_GRID)
    for c in STAGE_CVPREDICT_COARSE:
        assert "--float-trust-policy" in c.extra_args
        idx = c.extra_args.index("--float-trust-policy")
        assert c.extra_args[idx + 1] == "cv-predict"
        assert not c.needs_nlos_csv


def test_scaledreset_coarse_grid_has_one_candidate_per_qpos() -> None:
    assert len(STAGE_SCALEDRESET_COARSE) == len(QPOS_GRID)
    for c in STAGE_SCALEDRESET_COARSE:
        idx = c.extra_args.index("--float-trust-policy")
        assert c.extra_args[idx + 1] == "scaled-reset"


def test_coarse_grids_cover_every_qpos_value_exactly_once() -> None:
    for stage in (STAGE_CVPREDICT_COARSE, STAGE_SCALEDRESET_COARSE):
        qpos_values = [c.extra_args[c.extra_args.index("--trust-lapse-qpos") + 1] for c in stage]
        assert sorted(qpos_values, key=float) == sorted(QPOS_GRID, key=float)


def test_policy_flag_args_shape() -> None:
    args = policy_flag_args("cv-predict", "10")
    assert args == ["--float-trust-policy", "cv-predict", "--trust-lapse-qpos", "10"]


def test_regression_candidates_include_baseline_and_winner_verbatim() -> None:
    candidates = regression_candidates("cv-predict", "10")
    assert len(candidates) == 2
    assert candidates[0].extra_args == []
    assert candidates[0].needs_nlos_csv is False
    assert "cv-predict" in candidates[1].extra_args
    assert "10" in candidates[1].extra_args


def test_combination_candidates_include_all_four_configs() -> None:
    candidates = combination_candidates("scaled-reset", "1")
    assert len(candidates) == 4
    baseline = next(c for c in candidates if "baseline" in c.name)
    assert baseline.extra_args == []

    winner_only = next(c for c in candidates if c.name.startswith("c1_"))
    assert "--hold-ratio-threshold" not in winner_only.extra_args
    assert "scaled-reset" in winner_only.extra_args

    hold_only = next(c for c in candidates if c.name.startswith("c2_"))
    assert hold_only.extra_args == ["--hold-ratio-threshold", "2.0"]

    combined = next(c for c in candidates if c.name.startswith("c3_"))
    assert "--hold-ratio-threshold" in combined.extra_args
    assert "scaled-reset" in combined.extra_args
    assert "2.0" in combined.extra_args


def test_stretch_candidates_layer_min_los_sats_on_winner_plus_continuous_weighting() -> None:
    candidates = stretch_candidates("cv-predict", "10", 4)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.needs_nlos_csv is True
    assert "--nlos-min-los-sats" in c.extra_args
    idx = c.extra_args.index("--nlos-min-los-sats")
    assert c.extra_args[idx + 1] == "4"
    assert "--nlos-weight-mode" in c.extra_args
    assert "cv-predict" in c.extra_args


def test_nlos_relax_candidates_layer_flag_on_winner() -> None:
    candidates = nlos_relax_candidates("cv-predict", "10")
    assert len(candidates) == 1
    c = candidates[0]
    assert "--trust-gate-nlos-relax" in c.extra_args
    assert c.needs_nlos_csv is True
    assert "cv-predict" in c.extra_args
