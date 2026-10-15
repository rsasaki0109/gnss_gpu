"""Bit-exact uniform CDF construction, selected states and optional fallback."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_uniform_cdf_cases import COUNTS, TARGETS, MODES, check_uniform, check_dispatch, check_rounding


@pytest.fixture(scope='module')
def backend():
    native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    return native, ParticleFilterDevice


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('target', TARGETS)
@pytest.mark.parametrize('variant', range(3))
def test_full_sequential_CDF_and_all16_selected_bits(backend, count, target, variant):
    check_uniform(*backend, count, target, variant)


@pytest.mark.parametrize('count', (65536, 100000, 1000000))
@pytest.mark.parametrize('mode', MODES)
def test_optional_hook_pair_and_count_fallback(backend, count, mode):
    check_dispatch(*backend, count, mode)


@pytest.mark.parametrize('mode', (0, 0x100, 0x200, 0x300))
def test_nearest_only_and_original_other_rounding(backend, mode):
    check_rounding(*backend, mode)


@pytest.mark.parametrize('count', __import__('pf_uniform_cdf_cases').EXTRA_COUNTS)
def test_more_full_GPU_CDF_bits(backend, count):
    from pf_uniform_cdf_cases import check_extra_cdf
    check_extra_cdf(*backend, count)


@pytest.mark.parametrize('count', __import__('pf_uniform_cdf_cases').QUEUE_COUNTS)
def test_uniform_queued_producers_CDF_and_ancestor_B_reuse(backend, count):
    from pf_uniform_cdf_cases import check_uniform_queue
    check_uniform_queue(*backend, count)


@pytest.mark.parametrize('form', ('dtype', 'list', 'readonly', 'stride', 'short', 'long', 'matrix', 'null_target', 'null_source', 'destroyed_target', 'destroyed_source', 'inplace', 'unaligned'))
def test_uniform_native_contract_before_mutation(backend, form):
    import numpy as np
    from bench_pf_covariance_broadcast_compare import _new_pf
    from bench_pf_covariance_compare import _snapshot
    from pf_particle_readout_cases import bits_equal
    native, klass = backend
    method = getattr(native, 'pf_device_gather_from_uniform_cdf', None)
    if method is None:
        assert not hasattr(native, 'pf_device_gather_from_uniform_cdf')
        return  # The prior backend has no new API; runtime fallback is covered.
    count = 65536
    source, target = _new_pf(klass, count), _new_pf(klass, 257)
    try:
        output = np.full(count, -42.)
        before = _snapshot(native, target._state)
        if form == 'dtype':
            output = output.astype(np.float32)
        elif form == 'list':
            output = output.tolist()
        elif form == 'readonly':
            output.flags.writeable = False
        elif form == 'stride':
            output = np.full(count*2, -42.)[::2]
        elif form in ('short', 'long'):
            output = np.full(count+(-1 if form == 'short' else 1), -42.)
        elif form == 'matrix':
            output = output.reshape(256, 256)
        elif form == 'unaligned':
            output = np.ndarray((count,), dtype=np.float64, buffer=bytearray(count*8+1), offset=1)
            assert method(target._state, source._state, output)
            bits_equal(output, np.cumsum(np.full(count, 1./count)))
            return
        a, b = target._state, source._state
        if form.startswith('destroyed'):
            pf = target if form.endswith('target') else source
            native.pf_device_destroy(pf._state)
        if form == 'null_target':
            a = None
        elif form == 'null_source':
            b = None
        if form == 'inplace':
            assert not method(source._state, source._state, output)
        else:
            with pytest.raises(RuntimeError):
                method(a, b, output)
        if form != 'destroyed_target':
            for x, y in zip(_snapshot(native, target._state), before, strict=True):
                bits_equal(x, y)
        bits_equal(np.asarray(output), np.full(np.asarray(output).shape, -42., dtype=np.asarray(output).dtype))
    finally:
        for pf in (source, target):
            if not (form.startswith('destroyed') and pf is (target if form.endswith('target') else source)):
                native.pf_device_destroy(pf._state)
            pf._state = None
