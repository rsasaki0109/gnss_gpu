from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from analyze_wp175_student_t_front_end import _percentile


def test_percentile_interpolates_endpoints_and_middle() -> None:
    assert _percentile([0.0, 10.0], 0.0) == 0.0
    assert _percentile([0.0, 10.0], 50.0) == 5.0
    assert _percentile([0.0, 10.0], 95.0) == 9.5
    assert _percentile([0.0, 10.0], 100.0) == 10.0
