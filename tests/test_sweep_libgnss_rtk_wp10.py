"""Unit tests for experiments/sweep_libgnss_rtk_wp10.py's pure helpers.

Mirrors tests/test_sweep_libgnss_rtk_wp9.py's scope: only candidate-building
logic that doesn't require WSL/gnss_solve is covered here. The engine-level
lapse-gated-policy / min-los-sats behavior is validated by the C++ unit
tests (test_float_trust_policy.cpp, test_nlos_weights.cpp, test_rtk_smoke.cpp)
and by score deltas on real PPC runs, not by mocking the engine here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_libgnss_rtk_wp10 import (  # noqa: E402
    GATE_GRID,
    LAPSE_GATE_UNREACHABLE_S,
    LAPSE_GATED_QPOS,
    MIN_LOS_SATS_GRID,
    NLOS_FRAC_GRID,
    STAGE_BITIDENTITY_CHECK,
    STAGE_GATE_SWEEP,
    STAGE_MIN_LOS_SATS,
    STAGE_NLOS_FRAC_SWEEP,
    combination_candidates,
    lapse_gated_args,
    lapse_gated_nlos_frac_args,
    min_los_sats_args,
    regression_candidates,
)


def test_gate_grid_matches_task_spec() -> None:
    assert GATE_GRID == ("2", "5", "10", "20")


def test_lapse_gated_args_shape() -> None:
    args = lapse_gated_args("10")
    assert args == [
        "--float-trust-policy", "lapse-gated",
        "--trust-lapse-gate-s", "10",
        "--trust-lapse-qpos", LAPSE_GATED_QPOS,
    ]


def test_lapse_gated_args_honors_explicit_qpos_override() -> None:
    args = lapse_gated_args("10", qpos="1.0")
    idx = args.index("--trust-lapse-qpos")
    assert args[idx + 1] == "1.0"


def test_gate_sweep_stage_has_baseline_plus_one_candidate_per_gate() -> None:
    assert len(STAGE_GATE_SWEEP) == len(GATE_GRID) + 1
    baseline = STAGE_GATE_SWEEP[0]
    assert baseline.extra_args == []
    gate_values = []
    for c in STAGE_GATE_SWEEP[1:]:
        assert "--float-trust-policy" in c.extra_args
        idx = c.extra_args.index("--float-trust-policy")
        assert c.extra_args[idx + 1] == "lapse-gated"
        gate_idx = c.extra_args.index("--trust-lapse-gate-s")
        gate_values.append(c.extra_args[gate_idx + 1])
    assert sorted(gate_values, key=float) == sorted(GATE_GRID, key=float)


def test_bitidentity_check_stage_uses_huge_gate() -> None:
    assert len(STAGE_BITIDENTITY_CHECK) == 2
    baseline, huge_gate = STAGE_BITIDENTITY_CHECK
    assert baseline.extra_args == []
    idx = huge_gate.extra_args.index("--trust-lapse-gate-s")
    assert float(huge_gate.extra_args[idx + 1]) >= 1.0e6


def test_regression_candidates_include_baseline_and_winner_gate_verbatim() -> None:
    candidates = regression_candidates("5")
    assert len(candidates) == 2
    assert candidates[0].extra_args == []
    assert candidates[0].needs_nlos_csv is False
    idx = candidates[1].extra_args.index("--trust-lapse-gate-s")
    assert candidates[1].extra_args[idx + 1] == "5"


def test_min_los_sats_args_shape() -> None:
    args = min_los_sats_args("4")
    assert args == [
        "--nlos-weight-mode", "continuous",
        "--nlos-continuous-floor", "0.5",
        "--nlos-min-los-sats", "4",
    ]


def test_min_los_sats_stage_has_baseline_plus_one_candidate_per_n() -> None:
    assert len(STAGE_MIN_LOS_SATS) == len(MIN_LOS_SATS_GRID) + 1
    baseline = STAGE_MIN_LOS_SATS[0]
    assert baseline.extra_args == []
    assert baseline.needs_nlos_csv is False
    for c in STAGE_MIN_LOS_SATS[1:]:
        assert c.needs_nlos_csv is True
        assert "--nlos-min-los-sats" in c.extra_args


def test_lapse_gated_nlos_frac_args_shape() -> None:
    args = lapse_gated_nlos_frac_args("0.5")
    assert args == [
        "--float-trust-policy", "lapse-gated",
        "--trust-lapse-gate-s", LAPSE_GATE_UNREACHABLE_S,
        "--trust-lapse-gate-nlos-frac", "0.5",
        "--trust-lapse-qpos", LAPSE_GATED_QPOS,
    ]
    assert float(LAPSE_GATE_UNREACHABLE_S) >= 1.0e6


def test_nlos_frac_sweep_stage_has_baseline_plus_one_candidate_per_frac() -> None:
    assert len(STAGE_NLOS_FRAC_SWEEP) == len(NLOS_FRAC_GRID) + 1
    baseline = STAGE_NLOS_FRAC_SWEEP[0]
    assert baseline.extra_args == []
    assert baseline.needs_nlos_csv is False
    for c in STAGE_NLOS_FRAC_SWEEP[1:]:
        assert c.needs_nlos_csv is True
        assert "--trust-lapse-gate-nlos-frac" in c.extra_args
        gate_idx = c.extra_args.index("--trust-lapse-gate-s")
        assert c.extra_args[gate_idx + 1] == LAPSE_GATE_UNREACHABLE_S


def test_combination_candidates_include_all_four_configs() -> None:
    candidates = combination_candidates("5", "4")
    assert len(candidates) == 4
    baseline = next(c for c in candidates if "baseline" in c.name)
    assert baseline.extra_args == []

    lapse_only = next(c for c in candidates if c.name.startswith("d1_"))
    assert "--nlos-min-los-sats" not in lapse_only.extra_args
    assert "lapse-gated" in lapse_only.extra_args

    minlos_only = next(c for c in candidates if c.name.startswith("d2_"))
    assert "lapse-gated" not in minlos_only.extra_args
    assert "--nlos-min-los-sats" in minlos_only.extra_args
    assert minlos_only.needs_nlos_csv is True

    combined = next(c for c in candidates if c.name.startswith("d3_"))
    assert "lapse-gated" in combined.extra_args
    assert "--nlos-min-los-sats" in combined.extra_args
    assert combined.needs_nlos_csv is True
