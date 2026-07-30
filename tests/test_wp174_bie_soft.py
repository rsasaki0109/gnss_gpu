from __future__ import annotations

import math

import pytest

from experiments.analyze_wp174_bie_soft import bie_top2


def test_bie_top2_returns_weighted_position_and_spread() -> None:
    position, position_std = bie_top2(
        1.0,
        1.0,
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        1.0,
    )

    assert position == (1.0, 0.0, 0.0)
    assert position_std == pytest.approx(1.0)


def test_bie_top2_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        bie_top2(
            1.0,
            2.0,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            math.nan,
        )
