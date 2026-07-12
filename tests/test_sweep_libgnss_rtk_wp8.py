"""Unit tests for experiments/sweep_libgnss_rtk_wp8.py's pure helpers.

Mirrors tests/test_sweep_libgnss_rtk_wp7.py's scope: only candidate-building
logic that doesn't require WSL/gnss_solve is covered here. The engine-level
hard-exclusion behavior is validated by the C++ unit tests
(test_nlos_weights.cpp, test_rtk_smoke.cpp) and by score deltas on real PPC
runs, not by mocking the engine here.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (_PROJECT_ROOT / "experiments",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sweep_libgnss_rtk_wp8 import (  # noqa: E402
    STAGE_EXCLUDE_COARSE,
    STAGE_RETUNE,
    combined_candidates,
    exclude_generalize_candidates,
    exclude_refine_candidates,
    retune_generalize_candidates,
)


def test_exclude_coarse_grid_has_six_candidates_matching_task_spec() -> None:
    # threshold {0.3, 0.5} x min-sats {4, 5, 6} = 6.
    assert len(STAGE_EXCLUDE_COARSE) == 6
    thresholds = {c.extra_args[c.extra_args.index("--nlos-exclude-threshold") + 1] for c in STAGE_EXCLUDE_COARSE}
    assert thresholds == {"0.3", "0.5"}
    min_sats = {c.extra_args[c.extra_args.index("--nlos-min-sats") + 1] for c in STAGE_EXCLUDE_COARSE}
    assert min_sats == {"4", "5", "6"}


def test_exclude_coarse_candidates_all_flag_needs_nlos_csv() -> None:
    assert all(c.needs_nlos_csv for c in STAGE_EXCLUDE_COARSE)


def test_exclude_coarse_candidates_use_exclude_mode() -> None:
    for c in STAGE_EXCLUDE_COARSE:
        idx = c.extra_args.index("--nlos-weight-mode")
        assert c.extra_args[idx + 1] == "exclude"


def test_exclude_refine_brackets_best_min_sats_by_one() -> None:
    candidates = exclude_refine_candidates("0.5", 5)
    min_sats_values = sorted(
        int(c.extra_args[c.extra_args.index("--nlos-min-sats") + 1]) for c in candidates
    )
    assert min_sats_values == [4, 6]
    for c in candidates:
        thr_idx = c.extra_args.index("--nlos-exclude-threshold")
        assert c.extra_args[thr_idx + 1] == "0.5"


def test_exclude_refine_skips_negative_min_sats() -> None:
    candidates = exclude_refine_candidates("0.3", 0)
    min_sats_values = [int(c.extra_args[c.extra_args.index("--nlos-min-sats") + 1]) for c in candidates]
    assert min_sats_values == [1]


def test_exclude_generalize_candidates_include_baseline_and_winner() -> None:
    candidates = exclude_generalize_candidates("0.5", 6)
    assert len(candidates) == 2
    assert candidates[0].extra_args == []
    assert candidates[0].needs_nlos_csv is False
    assert candidates[1].needs_nlos_csv is True
    assert "6" in candidates[1].extra_args


def test_retune_grid_has_twelve_candidates_matching_task_spec() -> None:
    # margin {0.0, 0.2, 0.35, 0.5} x hold-ratio {2.0, 2.5, 3.0} = 12.
    assert len(STAGE_RETUNE) == 12
    margins = {c.extra_args[c.extra_args.index("--arfilter-margin") + 1] for c in STAGE_RETUNE}
    assert margins == {"0.0", "0.2", "0.35", "0.5"}
    holds = {c.extra_args[c.extra_args.index("--hold-ratio-threshold") + 1] for c in STAGE_RETUNE}
    assert holds == {"2.0", "2.5", "3.0"}


def test_retune_candidates_are_orthogonal_to_nlos_flags() -> None:
    # Work item 4's explicit "keep this orthogonal" requirement: no
    # --nlos-* flags anywhere in the retune grid, and needs_nlos_csv=False.
    for c in STAGE_RETUNE:
        assert not any(arg.startswith("--nlos") for arg in c.extra_args)
        assert c.needs_nlos_csv is False


def test_retune_generalize_candidates_include_baseline_and_winner() -> None:
    candidates = retune_generalize_candidates("0.2", "3.0")
    assert len(candidates) == 2
    assert candidates[0].extra_args == []
    assert "0.2" in candidates[1].extra_args
    assert "3.0" in candidates[1].extra_args


def test_combined_candidates_layer_both_levers() -> None:
    candidates = combined_candidates("0.5", 6, "0.2", "3.0")
    assert len(candidates) == 4
    by_name = {c.name: c for c in candidates}
    baseline = next(c for c in candidates if "baseline" in c.name)
    assert baseline.extra_args == []

    exclude_only = [c for c in candidates if "exclude" in c.name and "combined" not in c.name][0]
    assert "--nlos-weight-mode" in exclude_only.extra_args
    assert "--arfilter-margin" not in exclude_only.extra_args

    retune_only = [c for c in candidates if "margin" in c.name and "combined" not in c.name][0]
    assert "--arfilter-margin" in retune_only.extra_args
    assert "--nlos-weight-mode" not in retune_only.extra_args

    combined = next(c for c in candidates if "combined" in c.name)
    assert "--nlos-weight-mode" in combined.extra_args
    assert "--arfilter-margin" in combined.extra_args
    assert combined.needs_nlos_csv is True
    assert by_name  # keep by_name referenced for lint-cleanliness
