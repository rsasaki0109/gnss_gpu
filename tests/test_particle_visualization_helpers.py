from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from experiments.exp_particle_visualization import _axis_limits, ecef_to_local_enu


def test_ecef_to_local_enu_at_equator_origin():
    origin = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    points = np.array(
        [
            [6378137.0, 10.0, 0.0],
            [6378137.0, 0.0, 5.0],
        ],
        dtype=np.float64,
    )

    enu = ecef_to_local_enu(points, origin)

    np.testing.assert_allclose(enu, [[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]], atol=1e-9)


def test_axis_limits_are_square_and_handle_nonfinite_values():
    xlim, ylim = _axis_limits(np.array([[0.0, 0.0], [10.0, 5.0], [np.nan, 1.0]]), min_span_m=20.0)

    np.testing.assert_allclose(xlim, (-6.6, 16.6))
    np.testing.assert_allclose(ylim, (-9.1, 14.1))


def test_axis_limits_fallback_when_no_finite_points():
    assert _axis_limits(np.array([[np.nan, np.nan]])) == ((-10.0, 10.0), (-10.0, 10.0))
