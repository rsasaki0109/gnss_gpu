"""CPU-side validation tests for Particle Filter wrappers."""

import numpy as np
import pytest

from gnss_gpu.particle_filter import ParticleFilter


def _generate_satellites(n_sat=4, seed=42):
    rng = np.random.RandomState(seed)
    R_orbit = 26_571_000.0
    theta = rng.uniform(0, 2 * np.pi, n_sat)
    phi = rng.uniform(-np.pi / 3, np.pi / 3, n_sat)
    sat = np.zeros((n_sat, 3))
    sat[:, 0] = R_orbit * np.cos(phi) * np.cos(theta)
    sat[:, 1] = R_orbit * np.cos(phi) * np.sin(theta)
    sat[:, 2] = R_orbit * np.sin(phi)
    return sat


def test_pf_init_rejects_invalid_config_before_native_call():
    with pytest.raises(ValueError, match="n_particles must be a positive integer"):
        ParticleFilter(n_particles=0)
    with pytest.raises(ValueError, match="sigma_pos must be positive"):
        ParticleFilter(sigma_pos=0.0)
    with pytest.raises(ValueError, match="sigma_cb must be finite"):
        ParticleFilter(sigma_cb=np.inf)
    with pytest.raises(ValueError, match="sigma_pr must be positive"):
        ParticleFilter(sigma_pr=-1.0)
    with pytest.raises(ValueError, match="ess_threshold must be in"):
        ParticleFilter(ess_threshold=1.5)
    with pytest.raises(ValueError, match='resampling must be "megopolis" or "systematic"'):
        ParticleFilter(resampling="invalid")


def test_pf_initialize_rejects_invalid_position_before_native_call():
    pf = ParticleFilter(n_particles=100)

    with pytest.raises(ValueError, match="position_ecef must have shape"):
        pf.initialize([0.0, 0.0])
    with pytest.raises(ValueError, match="position_ecef must have shape"):
        pf.initialize([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError, match="position_ecef must be finite"):
        pf.initialize([0.0, np.nan, 0.0])


def test_pf_initialize_rejects_invalid_spread_before_native_call():
    pf = ParticleFilter(n_particles=100)
    pos = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="spread_pos must be positive"):
        pf.initialize(pos, spread_pos=0.0)
    with pytest.raises(ValueError, match="spread_cb must be finite"):
        pf.initialize(pos, spread_cb=np.inf)


def test_pf_predict_rejects_invalid_dt_before_native_call():
    pf = ParticleFilter(n_particles=100)
    pf.initialize(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="dt must be non-negative"):
        pf.predict(dt=-1.0)
    with pytest.raises(ValueError, match="dt must be finite"):
        pf.predict(dt=np.nan)


def test_pf_predict_rejects_invalid_velocity_before_native_call():
    pf = ParticleFilter(n_particles=100)
    pf.initialize(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="velocity must contain at least"):
        pf.predict(velocity=[1.0, 2.0])
    with pytest.raises(ValueError, match="velocity must be finite"):
        pf.predict(velocity=[1.0, np.nan, 3.0])


def test_pf_update_rejects_invalid_inputs_before_native_call():
    pf = ParticleFilter(n_particles=100)
    pf.initialize(np.array([1.0, 2.0, 3.0]))
    sat = _generate_satellites(4)
    pr = np.ones(4)

    with pytest.raises(ValueError, match="pseudoranges must contain at least"):
        pf.update(sat, [])
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        pf.update(sat[:2], pr)
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        bad_sat = sat.copy()
        bad_sat[0, 0] = np.inf
        pf.update(bad_sat, pr)
    with pytest.raises(ValueError, match="pseudoranges must be finite"):
        pf.update(sat, [1.0, np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights length must match"):
        pf.update(sat, pr, weights=np.ones(3))
    with pytest.raises(ValueError, match="weights must be non-negative"):
        pf.update(sat, pr, weights=[1.0, -1.0, 1.0, 1.0])
