from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from select_wp31_static_parent_marginal import (
    require_global_comparison_for_conditioned_parent,
    select_parent_marginal,
)


def _rows(spacing: float = 0.8, wl: tuple[float, float] = (0.35, 0.36)):
    candidates = [
        {"candidate_id": 1, "position_ecef": [0.0, 0.0, 0.0]},
        {"candidate_id": 2, "position_ecef": [spacing, 0.0, 0.0]},
    ]
    widelane = [
        {"candidate_id": 1, "widelane_median_abs_m": wl[0]},
        {"candidate_id": 2, "widelane_median_abs_m": wl[1]},
    ]
    return candidates, widelane


def test_accepts_compact_tied_children_as_parent_mean() -> None:
    candidates, widelane = _rows()
    result = select_parent_marginal(candidates, widelane, evidence_epochs=64)
    assert result["reason"] == "compact_widelane_parent_marginal"
    np.testing.assert_allclose(result["position_ecef"], [0.4, 0.0, 0.0])


def test_rejects_weak_absolute_widelane_even_when_compact() -> None:
    candidates, widelane = _rows(spacing=0.2, wl=(0.95, 1.0))
    result = select_parent_marginal(candidates, widelane, evidence_epochs=64)
    assert result["selected_candidate_id"] is None
    assert result["reason"] == "weak_absolute_widelane"


def test_rejects_spatially_split_children() -> None:
    candidates, widelane = _rows(spacing=2.0)
    result = select_parent_marginal(candidates, widelane, evidence_epochs=64)
    assert result["selected_candidate_id"] is None
    assert result["reason"] == "noncompact_widelane_children"


def test_accepts_small_absolute_tie_when_near_zero_ratio_is_unstable() -> None:
    candidates, widelane = _rows(spacing=0.08, wl=(0.086, 0.099))
    result = select_parent_marginal(candidates, widelane, evidence_epochs=30)
    assert result["reason"] == "compact_widelane_parent_marginal"
    assert result["widelane_ratio"] > 1.1
    assert result["widelane_gap_m"] < 0.05


def test_rejects_ratio_and_absolute_gap_separation() -> None:
    candidates, widelane = _rows(spacing=0.08, wl=(0.1, 0.2))
    result = select_parent_marginal(candidates, widelane, evidence_epochs=30)
    assert result["reason"] == "separated_widelane_children"


def test_conditioned_parent_cannot_self_accept_without_global_comparison() -> None:
    candidates, widelane = _rows()
    local = select_parent_marginal(candidates, widelane, evidence_epochs=64)
    guarded = require_global_comparison_for_conditioned_parent(
        local, {"seed_parent_candidate_id": 19}
    )
    assert guarded["selected_candidate_id"] is None
    assert guarded["reason"] == "parent_conditioned_requires_global_comparison"
