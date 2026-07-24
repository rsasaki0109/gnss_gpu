from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from analyze_wp29_plateau_basin_shadow import _select  # noqa: E402


def test_visibility_evidence_can_override_small_base_weight_gap() -> None:
    selected, gamma = _select(
        np.array([0.0, -0.1]),
        np.zeros(2),
        np.array([-3.0, 0.0]),
        pseudorange_weight=0.0,
        visibility_weight=1.0,
    )

    assert selected == 1
    assert gamma == pytest.approx(0.947846, abs=1.0e-6)


def test_zero_shadow_weights_replay_base_posterior() -> None:
    selected, gamma = _select(
        np.array([0.0, -1.0]),
        np.array([-100.0, 100.0]),
        np.array([-100.0, 100.0]),
        pseudorange_weight=0.0,
        visibility_weight=0.0,
    )

    assert selected == 0
    assert gamma == pytest.approx(0.7310585786)
