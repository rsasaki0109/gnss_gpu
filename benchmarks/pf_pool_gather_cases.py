"""Exact resident migration against the retained host gather and CPU CDF."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from pf_particle_readout_cases import bits_equal
from pf_state_chunks_cases import validate as validate_previous

SIZES=((1,1),(1,33),(33,1),(31,33),(33,31),(255,257),(257,255),(8193,4097),(65537,32769))


def state_values(states,variant):
    count=len(states)
    states=states.copy()
    if variant==0:
        states[:,4:7]=np.random.default_rng(count+1931).normal(size=(count,3))
        matrix=np.eye(3)[None,:,:]*(1.+np.arange(count)[:,None,None]/count)
        states[:,7:]=matrix.reshape(count,9)
    elif variant in (1,2,3,6):
        values={1:[-0.,0.],2:[5e-324,-5e-324,1e-154],3:[1e308,-1e308,1e154],6:[np.inf,-np.inf]}
        states[:]=np.resize(values[variant],states.shape)
    elif variant in (4,5):
        values=[0x7ff8000000000051,0xfff8000000000052] if variant==4 else [0x7ff0000000000053,0xfff0000000000054]
        states.view(np.uint64)[:]=np.resize(np.array(values,dtype=np.uint64),states.shape)
    else:
        states.view(np.uint64)[:]=np.random.default_rng(count+1932).integers(0,2**64-1,states.shape,dtype=np.uint64)
    return states


def ancestors(source_count,target_count,pattern):
    values=np.arange(target_count,dtype=np.int64)%source_count
    if pattern==1:
        values=source_count-1-values
    elif pattern==2:
        values.fill(source_count-1)
    elif pattern==3:
        values=np.random.default_rng(source_count+target_count).integers(0,source_count,target_count,dtype=np.int64)
    return values


def check_gather(native,klass,source_count,target_count,variant,pattern):
    source,target=_new_pf(klass,source_count),_new_pf(klass,target_count)
    try:
        states=state_values(source.get_particle_states(),variant)
        weights=np.random.default_rng(source_count+variant+1933).normal(0.,2.,source_count)
        source.set_particle_states(states)
        source.set_log_weights(weights)
        source_ess=source.get_ess()
        target.set_log_weights(np.random.default_rng(target_count+1934).normal(0.,3.,target_count))
        target.get_ess()
        indices=ancestors(source_count,target_count,pattern)
        expected=states[indices].copy()
        gather=getattr(native,'pf_device_gather_from',None)
        if gather is None:
            target.set_particle_states(expected)
        else:
            assert gather(target._state,source._state,indices)
        indices.fill(0)
        bits_equal(target.get_particle_states(),expected)
        bits_equal(target.get_log_weights(),np.zeros(target_count))
        uniform=target.get_ess()
        target.set_log_weights(np.zeros(target_count))
        bits_equal(target.get_ess(),uniform)
        after=_snapshot(native,source._state)
        bits_equal(after[0],states)
        bits_equal(after[1],weights)
        bits_equal(source.get_ess(),source_ess)
        source.set_particle_states(np.zeros_like(states))
        bits_equal(target.get_particle_states(),expected)
        return dict(input_sha256=_digest(states,weights,ancestors(source_count,target_count,pattern)),
                    output_sha256=_digest(expected,np.asarray(uniform),*after,np.asarray(source_ess)),
                    all16_source_bits_preserved=True,target_reset_cache_owned_and_input_complete=True)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_queue(native,klass,source_count,target_count):
    source,target,oracle,source_oracle=_new_pf(klass,source_count),_new_pf(klass,target_count),_new_pf(klass,target_count),_new_pf(klass,source_count)
    history=[]
    try:
        initial=state_values(source.get_particle_states(),0)
        source.set_particle_states(initial)
        source_oracle.set_particle_states(initial)
        for step in range(1,4):
            indices=ancestors(source_count,target_count,step)
            source_oracle.predict([1.,2.,3.],dt=.17,rbpf_velocity_kf=True,velocity_process_noise=.13)
            native.pf_device_resample_megopolis_coalesced(source_oracle._state,4,193+step)
            expected=source_oracle.get_particle_states()[indices].copy()
            # Queue actual source/target after the independent oracle's host
            # read, so that allocation cannot synchronize their producers.
            source.predict([1.,2.,3.],dt=.17,rbpf_velocity_kf=True,velocity_process_noise=.13)
            native.pf_device_resample_megopolis_coalesced(source._state,4,193+step)
            for pf in (target,oracle):
                pf.predict([3.,2.,1.],dt=.1,rbpf_velocity_kf=True,velocity_process_noise=.1)
                native.pf_device_resample_megopolis_coalesced(pf._state,4,196+step)
            gather=getattr(native,'pf_device_gather_from',None)
            if gather is None:
                target.set_particle_states(expected)
            else:
                assert gather(target._state,source._state,indices)
            oracle.set_particle_states(expected)
            for pf in (target,oracle):
                _kf(native,pf._state,_observations(12))
                pf.predict([1.,2.,3.],dt=.1,rbpf_velocity_kf=True,velocity_process_noise=.1)
                native.pf_device_resample_megopolis_coalesced(pf._state,60,199+step)
            a,b=_snapshot(native,target._state),_snapshot(native,oracle._state)
            for left,right in zip(a,b,strict=True):
                bits_equal(left,right)
            history.extend(a)
        return dict(output_sha256=_digest(*history),queued_target_gather_KF_and_next_resample_exact=True)
    finally:
        for pf in (source,target,oracle,source_oracle):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_pool(native,klass,source_count,target_count,variant,cold=False,refuse=False):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    source,oracle_source=_new_pf(klass,source_count),_new_pf(klass,source_count)
    target=klass(n_particles=target_count,seed=42) if cold else _new_pf(klass,target_count)
    oracle_target=klass(n_particles=target_count,seed=42) if cold else _new_pf(klass,target_count)
    try:
        states=state_values(source.get_particle_states(),0 if cold else variant)
        weights=np.random.default_rng(source_count+1935).normal(0.,3.,source_count)
        if variant==1:
            weights.fill(0.)
        elif variant==2:
            weights.fill(-np.inf)
        elif variant==3:
            weights.view(np.uint64)[:]=0x7ff8000000000097
        elif variant==4:
            weights[:]=np.resize([1e308,-1e308],source_count)
        elif variant==5:
            weights[:]=np.resize([np.nan,np.inf,-np.inf,0.,-1e308],source_count)
        for pf in (source,oracle_source):
            pf.set_particle_states(states)
            pf.set_log_weights(weights)
        oracle_target._pf_device_gather_from=None
        can_resident=callable(getattr(target,'_pf_device_gather_from',None)) and not cold and not refuse
        if refuse:
            def fallback(*args):
                return False
            source._pf_device_gather_from=target._pf_device_gather_from=fallback
        if can_resident:
            def forbidden(*args):
                raise AssertionError('Warm migration must avoid full-state host get/set')
            source.get_particle_states=forbidden
            target.set_particle_states=forbidden
        with np.errstate(all='ignore'):
            PersistentParticleRuntimePool._migrate(source,target)
            PersistentParticleRuntimePool._migrate(oracle_source,oracle_target)
        a,b=_snapshot(native,target._state),_snapshot(native,oracle_target._state)
        for left,right in zip(a,b,strict=True):
            bits_equal(left,right)
        src_a,src_b=_snapshot(native,source._state),_snapshot(native,oracle_source._state)
        for left,right in zip(src_a,src_b,strict=True):
            bits_equal(left,right)
        return dict(input_sha256=_digest(states,weights,np.array([target_count])),output_sha256=_digest(*a,*src_a),
                    original_CPU_CDF_and_selected_all16_states_exact=True,cold=cold,refused_fallback=refuse)
    finally:
        for pf in (source,target,oracle_source,oracle_target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def validate(native,klass):
    records=validate_previous(native,klass)
    for a,b in SIZES:
        for variant in range(8):
            for pattern in range(4):
                records[f'resident/{a}->{b}/v{variant}/p{pattern}']=check_gather(native,klass,a,b,variant,pattern)
        records[f'resident/{a}->{b}/queue']=check_queue(native,klass,a,b)
        for variant in range(8):
            records[f'pool/{a}->{b}/v{variant}']=check_pool(native,klass,a,b,variant)
        records[f'pool/{a}->{b}/cold']=check_pool(native,klass,a,b,0,cold=True)
        records[f'pool/{a}->{b}/refuse']=check_pool(native,klass,a,b,0,refuse=True)
    return records
