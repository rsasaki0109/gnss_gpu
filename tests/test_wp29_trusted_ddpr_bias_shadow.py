from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from analyze_wp29_trusted_ddpr_bias_shadow import (  # noqa: E402
    _bias_corrected_costs,
    _update_satellite_biases,
)


def test_trusted_update_learns_satellite_difference_and_removes_it() -> None:
    biases: dict[str, tuple[float, int]] = {}
    _update_satellite_biases(
        biases,
        np.array([3.0, -2.0]),
        ("G01", "G01"),
        ("G02", "G03"),
        epoch=10,
        blend_alpha=1.0,
    )

    costs, n_rows = _bias_corrected_costs(
        np.array([[3.0, -2.0], [4.0, -2.0]]),
        ("G01", "G01"),
        ("G02", "G03"),
        biases,
        epoch=11,
        max_age_epochs=5,
        scale_m=1.0,
    )

    assert n_rows == 2
    assert costs[0] == 0.0
    assert costs[1] > 0.0


def test_bias_score_abstains_after_memory_expires() -> None:
    costs, n_rows = _bias_corrected_costs(
        np.array([[3.0]]),
        ("G01",),
        ("G02",),
        {"G01": (0.0, 1), "G02": (3.0, 1)},
        epoch=10,
        max_age_epochs=5,
        scale_m=1.0,
    )

    assert n_rows == 0
    np.testing.assert_allclose(costs, [0.0])
