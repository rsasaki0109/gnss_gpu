"""CPU-side validation tests for ParticleFilterDevice wrappers."""

import numpy as np
import pytest

from gnss_gpu.particle_filter_device import ParticleFilterDevice

try:
    from gnss_gpu._gnss_gpu_pf_device import (
        pf_device_create,
        pf_device_initialize,
        pf_device_predict,
        pf_device_weight,
    )
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def _generate_satellites(n_sat=4):
    return np.array([
        [0.0, 0.0, 20000.0],
        [200.0, 0.0, 25.0],
        [0.0, 200.0, 5000.0],
        [-200.0, 0.0, 5000.0],
    ], dtype=np.float64)[:n_sat]


def test_pf_device_init_rejects_invalid_particle_count():
    with pytest.raises(ValueError, match="n_particles must be >= 1"):
        ParticleFilterDevice(n_particles=0)


def test_pf_device_init_rejects_nonpositive_sigmas():
    with pytest.raises(ValueError, match="sigma_pos must be positive"):
        ParticleFilterDevice(n_particles=100, sigma_pos=0.0)
    with pytest.raises(ValueError, match="sigma_cb must be finite"):
        ParticleFilterDevice(n_particles=100, sigma_cb=np.inf)


def test_pf_device_initialize_rejects_invalid_position_before_native_call():
    pf = ParticleFilterDevice(n_particles=100)

    with pytest.raises(ValueError, match="position_ecef must have shape"):
        pf.initialize([0.0, 0.0])
    with pytest.raises(ValueError, match="position_ecef must be finite"):
        pf.initialize([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="spread_pos must be positive"):
        pf.initialize([0.0, 0.0, 0.0], spread_pos=0.0)


def test_pf_device_predict_rejects_invalid_dt_before_native_call():
    pf = ParticleFilterDevice(n_particles=100)
    pf.initialize(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="dt must be non-negative"):
        pf.predict(dt=-1.0)
    with pytest.raises(ValueError, match="dt must be finite"):
        pf.predict(dt=np.nan)


def test_pf_device_update_rejects_invalid_inputs_before_native_call():
    pf = ParticleFilterDevice(n_particles=100)
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
    with pytest.raises(ValueError, match="sigma_pr must be positive"):
        pf.update(sat, pr, sigma_pr=0.0)


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_pf_device_binding_rejects_invalid_create():
    with pytest.raises(RuntimeError, match="n_particles must be >= 1"):
        pf_device_create(0)


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_pf_device_binding_rejects_invalid_initialize():
    state = pf_device_create(100)
    with pytest.raises(RuntimeError, match="spread_pos must be positive"):
        pf_device_initialize(
            state, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 42)


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_pf_device_binding_rejects_invalid_predict():
    state = pf_device_create(100)
    pf_device_initialize(state, 0.0, 0.0, 0.0, 0.0, 10.0, 100.0, 42)
    with pytest.raises(RuntimeError, match="dt must be non-negative"):
        pf_device_predict(state, 0.0, 0.0, 0.0, -1.0, 1.0, 100.0, 42, 1)
    with pytest.raises(RuntimeError, match="sigma_pos must be positive"):
        pf_device_predict(state, 0.0, 0.0, 0.0, 1.0, 0.0, 100.0, 42, 1)


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_pf_device_binding_rejects_invalid_weight():
    state = pf_device_create(100)
    pf_device_initialize(state, 0.0, 0.0, 0.0, 0.0, 10.0, 100.0, 42)
    sat = np.array([0.0, 0.0, 20000.0], dtype=np.float64)
    pr = np.array([20000.0 + 100.0], dtype=np.float64)
    ws = np.ones(1, dtype=np.float64)

    with pytest.raises(RuntimeError, match="n_sat must be >= 1"):
        pf_device_weight(state, sat[:0], np.array([]), np.array([]), 0, 5.0)
    with pytest.raises(RuntimeError, match="sat_ecef shape must match"):
        pf_device_weight(state, sat, pr, ws, 2, 5.0)
    with pytest.raises(RuntimeError, match="sigma_pr must be positive"):
        pf_device_weight(state, sat, pr, ws, 1, 0.0)


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
def test_pf_device_binding_valid_smoke_shapes():
    state = pf_device_create(64)
    pf_device_initialize(state, 1.0, 2.0, 3.0, 100.0, 10.0, 100.0, 42)
    pf_device_predict(state, 0.0, 0.0, 0.0, 1.0, 1.0, 100.0, 42, 1)
    sat = np.array([0.0, 0.0, 20000.0], dtype=np.float64)
    pr = np.array([20000.0 + 100.0], dtype=np.float64)
    ws = np.ones(1, dtype=np.float64)
    pf_device_weight(state, sat, pr, ws, 1, 5.0)
