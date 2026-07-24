from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analyze_wp29_moving_offset_shadow import (
    _lookup_assignment_integer,
    recurring_offset_candidates,
)


def test_assignment_lookup_matches_family_qualified_observation_ids() -> None:
    assignments = {("G01", "G02", 190293673): 17}

    assert (
        _lookup_assignment_integer(
            assignments, "G01@L1_E1_B1", "G02@L1_E1_B1", 0.190293673
        )
        == 17
    )


def test_recurring_offsets_are_ranked_by_epoch_coverage() -> None:
    trajectory = {epoch: np.array([float(epoch), 0.0, 0.0]) for epoch in range(3)}
    basins = {
        epoch: np.array(
            [trajectory[epoch] + [0.0, 1.0, 0.0], trajectory[epoch] + [0.0, 3.0, 0.0]]
            if epoch < 2
            else [trajectory[epoch] + [0.0, 1.0, 0.0]]
        )
        for epoch in range(3)
    }

    candidates = recurring_offset_candidates(
        basins,
        trajectory,
        0,
        3,
        sample_stride_epochs=1,
        radius_m=0.1,
        dedup_radius_m=0.1,
        max_candidates=2,
    )

    np.testing.assert_allclose(candidates[0]["offset_ecef_m"], [0.0, 1.0, 0.0])
    assert candidates[0]["coverage_epochs"] == 3
    assert candidates[1]["coverage_epochs"] == 2
