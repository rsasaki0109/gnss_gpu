"""Shared DD coefficients and fused gates retain the original GPU arithmetic."""

import numpy as np
import pytest

from test_pf_dd_reference_range import _check_against_mixed, _observations
from test_pf_dd_wavelength_cache import native

pytestmark = [pytest.mark.gpu, pytest.mark.cuda]


@pytest.mark.parametrize('kind', ['PR', 'CP', 'joint'])
@pytest.mark.parametrize('pairs', [2, 33, 511, 512])
@pytest.mark.parametrize('sigma', [.05, np.nextafter(.05, np.inf), 1e100])
def test_sigma_cache_preserves_original_bits_and_shared_memory_boundary(kind, pairs, sigma):
    def update(state, data, gate, huber):
        args = [state, data.sat_ecef_k, data.sat_ecef_ref]
        if kind == 'joint':
            native.pf_device_weight_dd_joint(*args, data.dd_pseudorange_m, data.dd_carrier_cycles,
                data.base_range_k, data.base_range_ref, data.dd_weights, data.wavelengths_m,
                data.n_dd, sigma, sigma, 10., .25, huber, 1.5, 1.5)
        else:
            args += [data.dd_pseudorange_m if kind == 'PR' else data.dd_carrier_cycles,
                     data.base_range_k, data.base_range_ref, data.dd_weights]
            if kind == 'CP':
                args.append(data.wavelengths_m)
            operation = native.pf_device_weight_dd_pseudorange if kind == 'PR' else native.pf_device_weight_dd_carrier_afv
            operation(*args, data.n_dd, sigma, gate, huber, 1.5)
    state = native.pf_device_create(257)
    try:
        _check_against_mixed(state, 257, _observations(pairs), update, .25, True)
    finally:
        native.pf_device_destroy(state)


@pytest.mark.parametrize('particles', [1, 31, 257])
@pytest.mark.parametrize('gates', [(10., 0.), (0., .25), (10., .25), (1e-100, 1e-100)])
@pytest.mark.parametrize('variant', ['ordinary', 'nan_observation', 'invalid_wavelength'])
def test_gate_kept_counts_fallback_and_independent_pr_cp_gates(particles, gates, variant):
    data = _observations(33)
    if variant == 'nan_observation':
        data.dd_pseudorange_m[1] = np.nan
    elif variant == 'invalid_wavelength':
        data.wavelengths_m[1] = -0.
    def update(state, obs, gate, huber):
        native.pf_device_weight_dd_joint(state, obs.sat_ecef_k, obs.sat_ecef_ref,
            obs.dd_pseudorange_m, obs.dd_carrier_cycles, obs.base_range_k, obs.base_range_ref,
            obs.dd_weights, obs.wavelengths_m, obs.n_dd, .75, .05, *gates)
    state = native.pf_device_create(particles)
    try:
        _check_against_mixed(state, particles, data, update, 0., False)
    finally:
        native.pf_device_destroy(state)
