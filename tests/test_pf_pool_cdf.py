"""Exact CDF gather against original CPU search and resident gather."""

from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_pool_cdf_cases import GATE_SIZES,SIZES,check_cdf,check_gate,check_queue,select
from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _snapshot
from pf_particle_readout_cases import bits_equal
from gnss_gpu.particle_filter_device import ParticleFilterDevice

native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
new_api=pytest.mark.skipif(not hasattr(native,'pf_device_gather_from_cdf'),reason='Optional CDF gather API')


@pytest.mark.parametrize('sizes',SIZES)
@pytest.mark.parametrize('variant',range(8))
@pytest.mark.parametrize('pattern',range(8))
def test_CPU_left_boundaries_selected_all16_state_bits(sizes,variant,pattern):
    check_cdf(native,ParticleFilterDevice,*sizes,variant,pattern)


@pytest.mark.parametrize('sizes',SIZES)
def test_source_CDF_scratch_reuse_and_queued_target_work(sizes):
    check_queue(native,ParticleFilterDevice,*sizes)


@pytest.mark.parametrize('sizes',GATE_SIZES)
@pytest.mark.parametrize('mode',('active','decline','missing','mismatch'))
def test_size_gate_strict_numpy_and_optional_backend_fallback(sizes,mode):
    check_gate(native,ParticleFilterDevice,*sizes,mode)


@new_api
@pytest.mark.parametrize('case',('empty','short','2D','nan','positive_inf','negative_inf','descending','inplace'))
def test_reject_or_decline_before_any_source_target_mutation(case):
    source,target=_new_pf(ParticleFilterDevice,31),_new_pf(ParticleFilterDevice,33)
    try:
        for pf in (source,target):
            pf.set_log_weights(np.arange(pf.n_particles,dtype=np.float64))
        before=[_snapshot(native,pf._state) for pf in (source,target)]
        ess=[pf.get_ess() for pf in (source,target)]
        cdf=np.linspace(0.,1.,31)
        if case in ('empty','short','2D'):
            cdf=np.zeros((0,) if case=='empty' else (30,) if case=='short' else (31,1))
            with pytest.raises(RuntimeError,match='CDF'):
                native.pf_device_gather_from_cdf(target._state,source._state,cdf)
        else:
            if case in ('nan','positive_inf','negative_inf'):
                cdf[15]={'nan':np.nan,'positive_inf':np.inf,'negative_inf':-np.inf}[case]
            elif case=='descending':
                cdf[15]=cdf[14]-1.
            assert not native.pf_device_gather_from_cdf(source._state if case=='inplace' else target._state,source._state,cdf)
        for pf,expected,cached in zip((source,target),before,ess,strict=True):
            for a,b in zip(_snapshot(native,pf._state),expected,strict=True):
                bits_equal(a,b)
            bits_equal(pf.get_ess(),cached)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


@new_api
@pytest.mark.parametrize('which',('null_source','null_target','dead_source','dead_target'))
def test_null_and_destroyed_state_handles(which):
    source=native.pf_device_create(31)
    target=native.pf_device_create(33)
    try:
        if which.startswith('dead'):
            native.pf_device_destroy(source if which.endswith('source') else target)
        with pytest.raises(RuntimeError,match='state'):
            native.pf_device_gather_from_cdf(None if which=='null_target' else target,
                None if which=='null_source' else source,np.linspace(0.,1.,31))
    finally:
        if source.allocated:
            native.pf_device_destroy(source)
        if target.allocated:
            native.pf_device_destroy(target)


@new_api
@pytest.mark.parametrize('form',('float64','float32','strided','negative','signed_zero'))
def test_CDF_coercion_and_host_input_reuse_after_return(form):
    source,target=_new_pf(ParticleFilterDevice,31),_new_pf(ParticleFilterDevice,33)
    try:
        cdf=np.linspace(0.,1.,31)
        if form=='float32':
            cdf=cdf.astype(np.float32)
        elif form=='strided':
            storage=np.zeros(62)
            storage[::2]=cdf
            cdf=storage[::2]
        elif form=='negative':
            cdf-=2.
        elif form=='signed_zero':
            cdf[:]=np.resize([-0.,0.],31)
        expected=source.get_particle_states()[select(cdf.astype(np.float64),33)].copy()
        assert native.pf_device_gather_from_cdf(target._state,source._state,cdf)
        cdf.fill(0.)
        bits_equal(target.get_particle_states(),expected)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


@pytest.mark.parametrize('sizes',((65536,32768),(32768,65536)))
@pytest.mark.parametrize('form',('single','short','long','matrix'))
def test_overridden_weight_length_uses_original_resident_ancestor_fallback(sizes,form):
    from pf_pool_cdf_cases import check_weight_size
    check_weight_size(native,ParticleFilterDevice,*sizes,form)
