"""Exact native Gaussian nodes, fallback and inactive-covariance scratch."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_mixture_geometry_cases import COMPONENTS, COUNTS, check_reuse, check_table
from gnss_gpu.particle_filter_device import ParticleFilterDevice

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
pytestmark = pytest.mark.skipif(not hasattr(native, 'pf_device_position_mixture_components'),
                               reason='New optional native geometry entrypoint')


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('components', COMPONENTS)
@pytest.mark.parametrize('variant', range(8))
def test_exact_numpy_components_and_all_state_preservation(count, components, variant):
    check_table(native, ParticleFilterDevice, count, components, variant)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('components', COMPONENTS)
def test_inactive_scratch_is_safe_after_queued_prediction_kf_and_resampling(count, components):
    check_reuse(native, ParticleFilterDevice, count, components)


@pytest.mark.parametrize('sigma2', [0., -1., np.nan, np.inf, -np.inf])
def test_reject_invalid_variance_without_state_changes(sigma2):
    pf = ParticleFilterDevice(n_particles=33)
    pf.initialize([1., 2., 3.])
    try:
        before = pf.get_particle_states()
        with pytest.raises(RuntimeError, match='sigma_squared'):
            native.pf_device_position_mixture_components(pf._state, np.zeros((2, 3)), np.zeros(2), sigma2)
        np.testing.assert_array_equal(before.view(np.uint64), pf.get_particle_states().view(np.uint64))
    finally:
        native.pf_device_destroy(pf._state)
        pf._state = None


@pytest.mark.parametrize('case', range(12))
def test_reject_bad_reference_and_log_mixture_buffers(case):
    pf = ParticleFilterDevice(n_particles=33)
    pf.initialize([1., 2., 3.])
    refs, mix = np.zeros((2, 3)), np.zeros(2)
    if case < 5:
        refs = np.zeros(((1, 3), (5, 3), (2, 2), (2, 3, 1), (0, 3))[case])
    elif case < 7:
        refs[:] = np.nan if case == 5 else np.inf
    elif case < 10:
        mix = np.zeros((1,) if case == 7 else (3,) if case == 8 else (1, 2))
    else:
        mix[0] = np.nan if case == 10 else np.inf
    try:
        before = pf.get_particle_states()
        with pytest.raises(RuntimeError):
            native.pf_device_position_mixture_components(pf._state, refs, mix, 6.25)
        np.testing.assert_array_equal(before.view(np.uint64), pf.get_particle_states().view(np.uint64))
    finally:
        native.pf_device_destroy(pf._state)
        pf._state = None


@pytest.mark.parametrize('representation', ('C', 'F', 'strided'))
def test_input_representation_and_immediate_input_mutation(representation):
    pf = ParticleFilterDevice(n_particles=257)
    pf.initialize([1., 2., 3.])
    matrix = np.arange(9., dtype=np.float64).reshape(3, 3)
    mix = np.array([0., -.7, -np.inf])
    refs = matrix.copy(order='F' if representation == 'F' else 'C')
    if representation == 'strided':
        storage = np.zeros((6, 6))
        storage[::2, ::2] = matrix
        refs = storage[::2, ::2]
        weights = np.zeros(6)
        weights[::2] = mix
        mix = weights[::2]
    try:
        states = pf.get_particle_states()
        diff = states[:, None, :3]-matrix[None, :, :]
        expected = np.array([0., -.7, -np.inf])[None, :]-.5*np.sum(diff*diff, axis=2)/6.25
        result = native.pf_device_position_mixture_components(pf._state, refs, mix, 6.25)
        refs.fill(-999.)
        mix.fill(-999.)
        np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
        result.fill(-999.)
        np.testing.assert_array_equal(states.view(np.uint64), pf.get_particle_states().view(np.uint64))
    finally:
        native.pf_device_destroy(pf._state)
        pf._state = None
