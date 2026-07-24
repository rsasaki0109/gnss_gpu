from __future__ import annotations

import numpy as np
import pytest

from experiments.build_wp29_imu_heading_route_seed_trace import local_enu_basis
from experiments.build_wp31_horizontal_shell_candidates import build_horizontal_shell


def test_horizontal_shell_preserves_up_component() -> None:
    center = np.array([-3960000.0, 3350000.0, 3700000.0])
    rows = build_horizontal_shell(center, (2.0, 4.0), directions=8)
    basis = local_enu_basis(center)
    assert len(rows) == 17
    for row in rows[1:]:
        delta = np.asarray(row["position_ecef"]) - center
        assert np.linalg.norm(delta) == pytest.approx(row["radius_m"])
        assert float(delta @ basis[2]) == pytest.approx(0.0, abs=1e-9)


def test_horizontal_shell_rejects_bad_direction_count() -> None:
    with pytest.raises(ValueError, match="multiple"):
        build_horizontal_shell(np.ones(3), (1.0,), directions=6)
