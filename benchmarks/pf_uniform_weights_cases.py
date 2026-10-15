"""Exact scalar log-weight bits, cache invalidation, and setter fallbacks."""

import math
import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest,_snapshot
from pf_particle_readout_cases import bits_equal
from pf_pool_normalize_cases import validate as validate_previous

COUNTS=(1,31,32,33,255,256,257,65535,65536,65537,100000,1000000)
BITS=(0,0x8000000000000000,1,0x8000000000000001,0x7fefffffffffffff,0xffefffffffffffff,
    0x7ff0000000000000,0xfff0000000000000,0x7ff8000000000041,0xfff8000000000042,
    0x7ff0000000000043,0xfff0000000000044)
MODES=('active','missing','unpaired','instance','class','hook','count_mismatch','uniform_override')


def scalar(count,variant):
    return -math.log(count) if variant==len(BITS) else np.array([BITS[variant]],dtype=np.uint64).view(np.float64)[0].item()


def uniform(pf,value):
    method=getattr(pf,'set_uniform_log_weights',None)
    if method is None:
        pf.set_log_weights(np.full(int(pf.n_particles),value,dtype=np.float64))
    else:
        method(value)


def check_fill(native,klass,count,variant):
    actual,oracle=_new_pf(klass,count),_new_pf(klass,count)
    try:
        prior=np.random.default_rng(count+1960).normal(0.,3.,count)
        for pf in (actual,oracle):
            pf.set_log_weights(prior)
            pf.get_ess()
        value=scalar(count,variant)
        expected=np.full(count,value,dtype=np.float64)
        api=getattr(native,'pf_device_fill_log_weights',None)
        if api is None:
            actual.set_log_weights(expected)
        else:
            api(actual._state,value)
        oracle.set_log_weights(expected)
        a,b=_snapshot(native,actual._state),_snapshot(native,oracle._state)
        bits_equal(a[1],expected)
        for left,right in zip(a,b,strict=True):
            bits_equal(left,right)
        ess_a,ess_b=actual.get_ess(),oracle.get_ess()
        bits_equal(ess_a,ess_b)
        # Read-only queries may quiet temporary NaNs but cannot alter stored bits.
        bits_equal(actual.get_log_weights(),expected)
        return dict(input_sha256=_digest(prior,np.asarray(value)),
            output_sha256=_digest(*a,np.asarray(ess_a)),all_scalar_bits_particle_fields_and_ESS_cache_exact=True)
    finally:
        for pf in (actual,oracle):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_queue(native,klass,count):
    actual,oracle=_new_pf(klass,count),_new_pf(klass,count)
    history=[]
    try:
        for step in range(3):
            value=-math.log(count)+step*.5
            def produce(pf):
                pf.predict([1.,2.,3.],dt=.13,rbpf_velocity_kf=True,velocity_process_noise=.2)
                native.pf_device_resample_megopolis_coalesced(pf._state,4,196+step)
            produce(oracle)
            oracle.set_log_weights(np.full(count,value))
            expected=_snapshot(native,oracle._state)
            # Oracle's host read precedes actual enqueue, so it cannot drain it.
            produce(actual)
            api=getattr(native,'pf_device_fill_log_weights',None)
            if api is None:
                actual.set_log_weights(np.full(count,value))
            else:
                api(actual._state,value)
            current=_snapshot(native,actual._state)
            for a,b in zip(current,expected,strict=True):
                bits_equal(a,b)
            for pf in (actual,oracle):
                pf.predict([3.,2.,1.],dt=.19,rbpf_velocity_kf=True,velocity_process_noise=.13)
                native.pf_device_resample_megopolis_coalesced(pf._state,60,199+step)
            current,expected=_snapshot(native,actual._state),_snapshot(native,oracle._state)
            for a,b in zip(current,expected,strict=True):
                bits_equal(a,b)
            history.extend(current)
        return dict(output_sha256=_digest(*history),queued_predict_gather_fill_next_predict_and_resample_exact=True)
    finally:
        for pf in (actual,oracle):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_public(native,klass,count,mode):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    from gnss_gpu.pf_device_runtime import ParticleFilterDeviceRuntime
    pf=_new_pf(klass,count)
    original_setter=pf.set_log_weights
    original_class=ParticleFilterDeviceRuntime.set_log_weights
    calls=[]
    fills=[]
    try:
        before=_snapshot(native,pf._state)
        api=getattr(native,'pf_device_fill_log_weights',None)
        if api is not None:
            def fill(*args):
                fills.append(1)
                return api(*args)
            pf._pf_device_fill_log_weights=fill
        def spy(*args):
            calls.append(1)
            return original_setter(*args)
        if mode=='missing':
            pf._pf_device_fill_log_weights=None
        elif mode=='unpaired':
            pf._pf_device_fill_log_weights_setter=None
        elif mode=='instance':
            pf.set_log_weights=spy
        elif mode=='class':
            def class_spy(self,*args):
                calls.append(1)
                return original_class(self,*args)
            ParticleFilterDeviceRuntime.set_log_weights=class_spy
        elif mode=='hook':
            hook=pf._pf_device_set_log_weights
            def hook_spy(*args):
                calls.append(1)
                return hook(*args)
            pf._pf_device_set_log_weights=hook_spy
        elif mode=='count_mismatch':
            pf.n_particles=count+1
        elif mode=='uniform_override':
            def forbidden(*args):
                raise AssertionError('Custom uniform method must not bypass original setter')
            pf.set_uniform_log_weights=forbidden
        target_count=int(pf.n_particles)
        try:
            PersistentParticleRuntimePool._set_migration_log_weights(pf,target_count) if hasattr(PersistentParticleRuntimePool,'_set_migration_log_weights') else pf.set_log_weights(np.full(target_count,-math.log(target_count)))
            error=None
        except RuntimeError as exception:
            error=(type(exception).__name__,str(exception))
        if mode=='count_mismatch':
            assert error is not None and 'input size' in error[1]
            after=_snapshot(native,pf._state)
            for a,b in zip(after,before,strict=True):
                bits_equal(a,b)
        else:
            assert error is None
            after=_snapshot(native,pf._state)
            bits_equal(after[0],before[0])
            bits_equal(after[1],np.full(count,-math.log(count)))
        if mode in ('instance','class','hook'):
            assert len(calls)==1 and not fills
        if api is not None:
            assert len(fills)==int(mode=='active' and count>=65536)
        return dict(output_sha256=_digest(*after),error=error,standard_setter_side_effects_and_size_gate_preserved=True)
    finally:
        ParticleFilterDeviceRuntime.set_log_weights=original_class
        pf.n_particles=count
        native.pf_device_destroy(pf._state)
        pf._state=None


def validate(native,klass):
    records=validate_previous(native,klass)
    for count in COUNTS:
        for variant in range(len(BITS)+1):
            records[f'uniform/{count}/v{variant}']=check_fill(native,klass,count,variant)
    for count in (33,65536,100000):
        records[f'uniform/queue/{count}']=check_queue(native,klass,count)
        for mode in MODES:
            records[f'uniform/public/{count}/{mode}']=check_public(native,klass,count,mode)
    return records
