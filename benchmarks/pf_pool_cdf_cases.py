"""Exact CPU CDF search boundaries, state ownership, and scratch reuse."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from pf_particle_readout_cases import bits_equal
from pf_pool_gather_cases import SIZES, state_values, validate as validate_previous

GATE_SIZES=((65535,32767),(65536,32768),(32768,65536),(65536,65535))


def cdf_values(source_count,target_count,pattern):
    if pattern==0:
        return np.cumsum(np.full(source_count,1./source_count))
    rng=np.random.default_rng(source_count+target_count+1940)
    if pattern in (1,7):
        weights=rng.random(source_count)
        weights/=np.sum(weights)
        if pattern==7:
            weights*=.25
        return np.cumsum(weights)
    if pattern==2:
        weights=np.zeros(source_count)
        weights[::max(1,source_count//7)]=1.
        weights/=np.sum(weights)
        return np.cumsum(weights)
    if pattern in (3,4,5):
        thresholds=(np.arange(target_count,dtype=np.float64)+.5)/target_count
        cdf=thresholds[np.linspace(0,target_count-1,source_count,dtype=np.int64)].copy()
        if pattern!=3:
            cdf=np.nextafter(cdf,-np.inf if pattern==4 else np.inf)
        return cdf
    return np.zeros(source_count)


def select(cdf,count):
    return np.minimum(np.searchsorted(cdf,(np.arange(count,dtype=np.float64)+.5)/count,side='left'),len(cdf)-1)


def check_cdf(native,klass,source_count,target_count,variant,pattern):
    source,target=_new_pf(klass,source_count),_new_pf(klass,target_count)
    try:
        states=state_values(source.get_particle_states(),variant)
        weights=np.random.default_rng(source_count+1941).normal(size=source_count)
        source.set_particle_states(states)
        source.set_log_weights(weights)
        source_ess=source.get_ess()
        target.set_log_weights(np.arange(target_count,dtype=np.float64))
        target.get_ess()
        cdf=cdf_values(source_count,target_count,pattern)
        indices=select(cdf,target_count)
        expected=states[indices].copy()
        gather=getattr(native,'pf_device_gather_from_cdf',None)
        if gather is None:
            assert native.pf_device_gather_from(target._state,source._state,indices)
        else:
            assert gather(target._state,source._state,cdf)
        cdf.fill(0.)
        bits_equal(target.get_particle_states(),expected)
        bits_equal(target.get_log_weights(),np.zeros(target_count))
        target_ess=target.get_ess()
        target.set_log_weights(np.zeros(target_count))
        bits_equal(target.get_ess(),target_ess)
        after=_snapshot(native,source._state)
        bits_equal(after[0],states)
        bits_equal(after[1],weights)
        bits_equal(source.get_ess(),source_ess)
        source.set_particle_states(np.zeros_like(states))
        bits_equal(target.get_particle_states(),expected)
        return dict(input_sha256=_digest(states,weights,cdf_values(source_count,target_count,pattern)),
            ancestors_sha256=_digest(indices),output_sha256=_digest(expected,*after,np.asarray(source_ess),np.asarray(target_ess)),
            all16_fields_left_ties_clamp_and_input_ownership_exact=True)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_queue(native,klass,source_count,target_count):
    source,target,oracle,source_oracle=[_new_pf(klass,n) for n in (source_count,target_count,target_count,source_count)]
    history=[]
    try:
        initial=state_values(source.get_particle_states(),0)
        source.set_particle_states(initial)
        source_oracle.set_particle_states(initial)
        for step in range(3):
            cdf=cdf_values(source_count,target_count,step+2)
            indices=select(cdf,target_count)
            def produce(pf):
                pf.predict([1.,2.,3.],dt=.17,rbpf_velocity_kf=True,velocity_process_noise=.13)
                if source_count<=257:
                    native.pf_device_resample_systematic(pf._state,194+step)
                else:
                    native.pf_device_resample_megopolis_coalesced(pf._state,4,194+step)
            produce(source_oracle)
            expected=source_oracle.get_particle_states()[indices].copy()
            produce(source)
            for pf in (target,oracle):
                pf.predict([3.,2.,1.],dt=.1,rbpf_velocity_kf=True,velocity_process_noise=.1)
                native.pf_device_resample_megopolis_coalesced(pf._state,4,197+step)
            gather=getattr(native,'pf_device_gather_from_cdf',None)
            if gather is None:
                assert native.pf_device_gather_from(target._state,source._state,indices)
            else:
                assert gather(target._state,source._state,cdf)
            oracle.set_particle_states(expected)
            for pf in (target,oracle):
                _kf(native,pf._state,_observations(12))
                pf.predict([1.,2.,3.],dt=.1,rbpf_velocity_kf=True,velocity_process_noise=.1)
                native.pf_device_resample_megopolis_coalesced(pf._state,60,200+step)
            a,b=_snapshot(native,target._state),_snapshot(native,oracle._state)
            for left,right in zip(a,b,strict=True):
                bits_equal(left,right)
            history.extend(a)
            # Source's next producer must overwrite scratch as before.
            for left,right in zip(_snapshot(native,source._state),_snapshot(native,source_oracle._state),strict=True):
                bits_equal(left,right)
        return dict(output_sha256=_digest(*history),source_CDF_scratch_systematic_reuse_and_queued_target_exact=True)
    finally:
        for pf in (source,target,oracle,source_oracle):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_gate(native,klass,source_count,target_count,mode):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    source,target,oracle_source,oracle_target=[_new_pf(klass,n) for n in (source_count,target_count,source_count,target_count)]
    calls=[]
    try:
        weights=np.random.default_rng(source_count+1942).normal(0.,2.,source_count)
        for pf in (source,oracle_source):
            pf.set_log_weights(weights)
        api=getattr(native,'pf_device_gather_from_cdf',None)
        if api is not None:
            def wrapped(*args):
                calls.append(1)
                return False if mode=='decline' else api(*args)
            source._pf_device_gather_from_cdf=target._pf_device_gather_from_cdf=wrapped
            if mode=='missing':
                target._pf_device_gather_from_cdf=None
            elif mode=='mismatch':
                source._pf_device_gather_from_cdf=lambda *args:api(*args)
        oracle_target._pf_device_gather_from_cdf=None
        with np.errstate(all='raise'):
            PersistentParticleRuntimePool._migrate(source,target)
            PersistentParticleRuntimePool._migrate(oracle_source,oracle_target)
        if api is not None:
            expected=max(source_count,target_count)>=65536 and mode in ('active','decline')
            assert len(calls)==int(expected)
        a,b=_snapshot(native,target._state),_snapshot(native,oracle_target._state)
        for left,right in zip(a,b,strict=True):
            bits_equal(left,right)
        for left,right in zip(_snapshot(native,source._state),_snapshot(native,oracle_source._state),strict=True):
            bits_equal(left,right)
        return dict(output_sha256=_digest(*a),strict_numpy_policy_selection_and_optional_dispatch_exact=True)
    finally:
        for pf in (source,target,oracle_source,oracle_target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_weight_size(native,klass,source_count,target_count,form):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    source,target,oracle_source,oracle_target=[_new_pf(klass,n) for n in (source_count,target_count,source_count,target_count)]
    calls=[]
    try:
        size=1 if form=='single' else 31 if form=='short' else source_count+1 if form=='long' else source_count
        weights=np.linspace(-2.,2.,size)
        if form=='matrix':
            weights=weights.reshape(256,-1)
        for pf in (source,oracle_source):
            pf.get_log_weights=lambda:weights.copy()
        api=getattr(native,'pf_device_gather_from_cdf',None)
        if api is not None:
            def wrapped(*args):
                calls.append(1)
                return api(*args)
            source._pf_device_gather_from_cdf=target._pf_device_gather_from_cdf=wrapped
        oracle_target._pf_device_gather_from_cdf=None
        def forbidden(*args):
            raise AssertionError('Warm migration must retain resident ancestor fallback')
        source.get_particle_states=forbidden
        target.set_particle_states=forbidden
        with np.errstate(all='raise'):
            PersistentParticleRuntimePool._migrate(oracle_source,oracle_target)
            PersistentParticleRuntimePool._migrate(source,target)
        if api is not None:
            assert len(calls)==int(form=='matrix')
        a,b=_snapshot(native,target._state),_snapshot(native,oracle_target._state)
        for left,right in zip(a,b,strict=True):
            bits_equal(left,right)
        for left,right in zip(_snapshot(native,source._state),_snapshot(native,oracle_source._state),strict=True):
            bits_equal(left,right)
        return dict(input_sha256=_digest(weights),output_sha256=_digest(*a),
            overridden_weight_length_original_CPU_search_resident_fallback_exact=True)
    finally:
        for pf in (source,target,oracle_source,oracle_target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def validate(native,klass):
    records=validate_previous(native,klass)
    for a,b in SIZES:
        for variant in range(8):
            for pattern in range(8):
                records[f'cdf/{a}->{b}/v{variant}/p{pattern}']=check_cdf(native,klass,a,b,variant,pattern)
        records[f'cdf/{a}->{b}/queue']=check_queue(native,klass,a,b)
    for a,b in GATE_SIZES:
        for mode in ('active','decline','missing','mismatch'):
            records[f'cdf/gate/{a}->{b}/{mode}']=check_gate(native,klass,a,b,mode)
    for a,b in ((65536,32768),(32768,65536)):
        for form in ('single','short','long','matrix'):
            records[f'cdf/weight_size/{a}->{b}/{form}']=check_weight_size(native,klass,a,b,form)
    return records
