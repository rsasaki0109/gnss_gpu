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
        pf_device_weight_dd_pseudorange,
        pf_device_weight_dd_carrier_afv,
        pf_device_weight_dd_joint,
        pf_device_get_log_weights,
        pf_device_sync,
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


@pytest.mark.skipif(not HAS_GPU, reason="CUDA module not available")
@pytest.mark.parametrize(
    "pr_gate,cp_gate,huber,pr_huber_k,cp_huber_k",
    [(0.0, 0.0, False, 1.5, 1.5), (2.0, 0.2, True, 1.2, 1.8)],
)
def test_pf_device_joint_dd_update_matches_sequential_updates(
        pr_gate, cp_gate, huber, pr_huber_k, cp_huber_k):
    n_particles = 256
    n_dd = 3
    receiver = np.array([1.0e6, 2.0e6, 3.0e6], dtype=np.float64)
    sat_k = np.array([
        [2.1e7, 1.2e7, 1.6e7],
        [1.4e7, 2.2e7, 1.1e7],
        [2.3e7, 0.8e7, 1.5e7],
    ], dtype=np.float64)
    sat_ref = np.tile([1.8e7, 1.9e7, 1.3e7], (n_dd, 1)).astype(np.float64)
    base_range_k = np.linalg.norm(sat_k - receiver, axis=1)
    base_range_ref = np.linalg.norm(sat_ref - receiver, axis=1)
    dd_pr = np.array([0.4, -0.7, 1.1], dtype=np.float64)
    dd_cp = np.array([0.04, -0.02, 0.07], dtype=np.float64)
    weights = np.array([1.0, 0.8, 0.6], dtype=np.float64)
    wavelengths = np.full(n_dd, 0.190293673, dtype=np.float64)

    sequential = pf_device_create(n_particles)
    joint = pf_device_create(n_particles)
    for state in (sequential, joint):
        pf_device_initialize(state, *receiver, 0.0, 10.0, 100.0, 42)

    pf_device_weight_dd_pseudorange(
        sequential, sat_k.ravel(), sat_ref.ravel(), dd_pr,
        base_range_k, base_range_ref, weights, n_dd, 0.75,
        pr_gate, huber, pr_huber_k)
    pf_device_weight_dd_carrier_afv(
        sequential, sat_k.ravel(), sat_ref.ravel(), dd_cp,
        base_range_k, base_range_ref, weights, wavelengths, n_dd, 0.05,
        cp_gate, huber, cp_huber_k)
    pf_device_weight_dd_joint(
        joint, sat_k.ravel(), sat_ref.ravel(), dd_pr, dd_cp,
        base_range_k, base_range_ref, weights, wavelengths, n_dd,
        0.75, 0.05, pr_gate, cp_gate, huber, pr_huber_k, cp_huber_k)
    pf_device_sync(sequential)
    pf_device_sync(joint)

    sequential_weights = np.empty(n_particles, dtype=np.float64)
    joint_weights = np.empty(n_particles, dtype=np.float64)
    pf_device_get_log_weights(sequential, sequential_weights)
    pf_device_get_log_weights(joint, joint_weights)
    np.testing.assert_allclose(
        joint_weights, sequential_weights, rtol=1e-14, atol=1e-12)
