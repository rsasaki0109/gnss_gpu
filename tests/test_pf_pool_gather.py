"""Exact cross-state gather and unchanged CPU pool selection/fallback."""

from pathlib import Path
import sys
import pytest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_pool_gather_cases import SIZES,check_gather,check_queue,check_pool
from gnss_gpu.particle_filter_device import ParticleFilterDevice

native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
new_api=pytest.mark.skipif(not hasattr(native,'pf_device_gather_from'),reason='Optional resident gather API')


@pytest.mark.parametrize('sizes',SIZES)
@pytest.mark.parametrize('variant',range(8))
@pytest.mark.parametrize('pattern',range(4))
def test_exact_source_ancestors_all16_fields_and_weight_cache(sizes,variant,pattern):
    check_gather(native,ParticleFilterDevice,*sizes,variant,pattern)


@pytest.mark.parametrize('sizes',SIZES)
def test_target_pending_gather_then_KF_and_next_resample(sizes):
    check_queue(native,ParticleFilterDevice,*sizes)


@pytest.mark.parametrize('sizes',SIZES)
@pytest.mark.parametrize('variant',range(8))
def test_warm_pool_matches_host_CDF_without_state_host_roundtrip(sizes,variant):
    check_pool(native,ParticleFilterDevice,*sizes,variant)


@pytest.mark.parametrize('sizes',SIZES)
@pytest.mark.parametrize('mode',('cold','refuse'))
def test_cold_target_and_unsupported_copy_fallback(sizes,mode):
    check_pool(native,ParticleFilterDevice,*sizes,0,**{mode:True})


@new_api
@pytest.mark.parametrize('case',range(7))
def test_reject_invalid_ancestor_shape_or_range_without_mutation(case):
    from bench_pf_covariance_broadcast_compare import _new_pf
    from bench_pf_covariance_compare import _snapshot
    from pf_particle_readout_cases import bits_equal
    source,target=_new_pf(ParticleFilterDevice,31),_new_pf(ParticleFilterDevice,33)
    try:
        source.set_log_weights(np.arange(31,dtype=np.float64))
        target.set_log_weights(np.arange(33,dtype=np.float64))
        before=[_snapshot(native,pf._state) for pf in (source,target)]
        cached=[pf.get_ess() for pf in (source,target)]
        index=np.arange(33,dtype=np.int64)%31
        if case<3:
            index=np.zeros((0,) if case==0 else (32,) if case==1 else (33,1),dtype=np.int64)
        else:
            index[-1]=(-1,31,np.iinfo(np.int64).max,np.iinfo(np.int64).min)[case-3]
        with pytest.raises(RuntimeError,match='ancestor'):
            native.pf_device_gather_from(target._state,source._state,index)
        for pf,expected,ess in zip((source,target),before,cached,strict=True):
            for a,b in zip(_snapshot(native,pf._state),expected,strict=True):
                bits_equal(a,b)
            bits_equal(pf.get_ess(),ess)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


@new_api
@pytest.mark.parametrize('which',('null_source','null_target','dead_source','dead_target'))
def test_reject_null_or_destroyed_native_handle(which):
    source=native.pf_device_create(31)
    target=native.pf_device_create(33)
    try:
        if which.startswith('dead'):
            native.pf_device_destroy(source if which.endswith('source') else target)
        with pytest.raises(RuntimeError,match='state'):
            native.pf_device_gather_from(None if which=='null_target' else target,
                None if which=='null_source' else source,np.zeros(33,dtype=np.int64))
    finally:
        if source.allocated:
            native.pf_device_destroy(source)
        if target.allocated:
            native.pf_device_destroy(target)


@new_api
def test_in_place_copy_declines_without_state_or_weight_mutation():
    from bench_pf_covariance_broadcast_compare import _new_pf
    from bench_pf_covariance_compare import _snapshot
    from pf_particle_readout_cases import bits_equal
    pf=_new_pf(ParticleFilterDevice,33)
    try:
        pf.set_log_weights(np.arange(33,dtype=np.float64))
        before=_snapshot(native,pf._state)
        cached=pf.get_ess()
        assert not native.pf_device_gather_from(pf._state,pf._state,np.arange(33,dtype=np.int64)[::-1])
        for a,b in zip(_snapshot(native,pf._state),before,strict=True):
            bits_equal(a,b)
        bits_equal(pf.get_ess(),cached)
    finally:
        native.pf_device_destroy(pf._state)
        pf._state=None


@new_api
@pytest.mark.parametrize('form',('int64','int32','strided'))
def test_ancestor_coercion_and_completed_input_ownership(form):
    from bench_pf_covariance_broadcast_compare import _new_pf
    from pf_particle_readout_cases import bits_equal
    source,target=_new_pf(ParticleFilterDevice,31),_new_pf(ParticleFilterDevice,33)
    try:
        expected_indices=np.arange(33,dtype=np.int64)%31
        index=expected_indices.astype(np.int32) if form=='int32' else expected_indices.copy()
        if form=='strided':
            storage=np.zeros(66,dtype=np.int64)
            storage[::2]=expected_indices
            index=storage[::2]
        expected=source.get_particle_states()[expected_indices].copy()
        assert native.pf_device_gather_from(target._state,source._state,index)
        index.fill(0)
        bits_equal(target.get_particle_states(),expected)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None
