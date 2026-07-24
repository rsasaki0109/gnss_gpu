from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from build_wp31_static_shell_candidates import build_shell_candidates


def test_cube26_shell_includes_center_and_all_directions() -> None:
    center = np.array([10.0, 20.0, 30.0])
    rows = build_shell_candidates(center, (2.0,), directions="cube26")
    assert len(rows) == 27
    np.testing.assert_allclose(rows[0]["position_ecef"], center)
    distances = [np.linalg.norm(np.asarray(row["position_ecef"]) - center) for row in rows[1:]]
    np.testing.assert_allclose(distances, np.full(26, 2.0))


def test_shell_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError):
        build_shell_candidates(np.zeros(3), (0.0,))
