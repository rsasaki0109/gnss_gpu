from __future__ import annotations

from experiments.analyze_wp174_safe_acquisition_negative import analyze
from tests.test_wp174_strong_instant_policy import _row


def test_all_negative_controls_fail_closed() -> None:
    rows = [
        _row(1.0, 0.0, ratio=2.0),
        _row(1.2, 0.5, ratio=2.0),
        _row(1.4, 0.505, ratio=2.0),
        _row(1.6, 0.510, ratio=2.0),
    ]

    result = analyze(rows, "synthetic")

    assert result["all_pass"]
    assert result["total_negative_fix_epochs"] == 0
