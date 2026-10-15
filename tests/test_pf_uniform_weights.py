"""Exact scalar weight storage and compatible public setter dispatch."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_uniform_weights_cases import BITS,COUNTS,MODES,check_fill,check_public,check_queue,uniform
from gnss_gpu.particle_filter_device import ParticleFilterDevice

native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
new_api=pytest.mark.skipif(not hasattr(native,'pf_device_fill_log_weights'),reason='Optional scalar fill API')


@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('variant',range(len(BITS)+1))
def test_native_scalar_bits_fields_and_cached_ESS_invalidation(count,variant):
    check_fill(native,ParticleFilterDevice,count,variant)


@pytest.mark.parametrize('count',(33,65536,100000))
def test_queued_producer_completion_and_future_resampling(count):
    check_queue(native,ParticleFilterDevice,count)


@pytest.mark.parametrize('count',(33,65536,100000))
@pytest.mark.parametrize('mode',MODES)
def test_public_pool_dispatch_preserves_injected_or_overridden_setters(count,mode):
    check_public(native,ParticleFilterDevice,count,mode)


@new_api
@pytest.mark.parametrize('which',('null','destroyed'))
def test_native_invalid_handle_rejected_before_mutation(which):
    state=native.pf_device_create(33)
    try:
        if which=='destroyed':
            native.pf_device_destroy(state)
        with pytest.raises(RuntimeError,match='state'):
            native.pf_device_fill_log_weights(None if which=='null' else state,1.)
    finally:
        if state.allocated:
            native.pf_device_destroy(state)


@new_api
def test_public_uninitialized_rejection_keeps_runtime_cold():
    pf=ParticleFilterDevice(n_particles=65536)
    try:
        with pytest.raises(RuntimeError,match='not initialized'):
            uniform(pf,0.)
        assert not pf._initialized
    finally:
        native.pf_device_destroy(pf._state)
        pf._state=None


@new_api
def test_original_native_setter_alias_survives_public_module_override(monkeypatch):
    from bench_pf_covariance_broadcast_compare import _new_pf
    from pf_particle_readout_cases import bits_equal
    original=native.pf_device_set_log_weights
    calls=[]
    def spy(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(native,'pf_device_set_log_weights',spy)
    pf=_new_pf(ParticleFilterDevice,65536)
    try:
        calls.clear()
        uniform(pf,-3.)
        assert len(calls)==1
        bits_equal(pf.get_log_weights(),np.full(65536,-3.))
    finally:
        native.pf_device_destroy(pf._state)
        pf._state=None
