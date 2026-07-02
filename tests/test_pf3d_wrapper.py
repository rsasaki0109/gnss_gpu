"""CPU-side validation tests for ParticleFilter3D wrappers."""

import numpy as np
import pytest

from gnss_gpu.particle_filter_3d import ParticleFilter3D
from gnss_gpu.raytrace import BuildingModel


def _make_box_building():
    return BuildingModel.create_box(
        center=[100.0, 0.0, 25.0], width=20.0, depth=20.0, height=50.0)


def _generate_satellites(n_sat=4):
    return np.array([
        [0.0, 0.0, 20000.0],
        [200.0, 0.0, 25.0],
        [0.0, 200.0, 5000.0],
        [-200.0, 0.0, 5000.0],
    ], dtype=np.float64)[:n_sat]


def test_pf3d_init_rejects_nonpositive_sigmas():
    building = _make_box_building()
    with pytest.raises(ValueError, match="sigma_los must be positive"):
        ParticleFilter3D(building_model=building, sigma_los=0.0, n_particles=100)
    with pytest.raises(ValueError, match="sigma_nlos must be finite"):
        ParticleFilter3D(building_model=building, sigma_nlos=np.inf, n_particles=100)


def test_pf3d_update_rejects_invalid_satellites_before_native_call():
    building = _make_box_building()
    pf = ParticleFilter3D(building_model=building, n_particles=100)
    pf.initialize(position_ecef=[0.0, 0.0, 0.0], clock_bias=100.0)

    sat = _generate_satellites(4)
    pr = np.ones(4)

    with pytest.raises(ValueError, match="pseudoranges must contain at least"):
        pf.update(sat, [])
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        pf.update(sat[:2], pr)
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        bad_sat = sat.copy()
        bad_sat[0, 0] = np.nan
        pf.update(bad_sat, pr)
    with pytest.raises(ValueError, match="pseudoranges must be finite"):
        pf.update(sat, [1.0, np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights length must match"):
        pf.update(sat, pr, weights=np.ones(3))
    with pytest.raises(ValueError, match="weights must be non-negative"):
        pf.update(sat, pr, weights=[1.0, -1.0, 1.0, 1.0])


def test_pf3d_update_requires_initialization():
    building = _make_box_building()
    pf = ParticleFilter3D(building_model=building, n_particles=100)
    sat = _generate_satellites(1)
    with pytest.raises(RuntimeError, match="not initialized"):
        pf.update(sat, np.array([20000.0 + 100.0]))
