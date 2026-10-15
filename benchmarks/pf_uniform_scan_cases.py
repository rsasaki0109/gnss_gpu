"""Full device-weight finite/equality checks and exact migration fallback."""

from types import SimpleNamespace

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from pf_particle_readout_cases import bits_equal
from pf_pool_cdf_cases import select
from pf_pool_gather_cases import state_values
from pf_uniform_cdf_cases import check_uniform_queue
from pf_uniform_inverse_cases import validate as validate_previous

COUNTS=(65535,65536,65537,100000,1000000)
VARIANTS=tuple(range(18))
MODES=('active','missing','mismatch','decline','python_getter','native_getter','cdf_override',
       'uniform_cdf_override','unpaired_getter','unpaired_state','strict_numpy',
       'source_count_mismatch','target_count_mismatch')


def weight_values(count,variant):
    values=np.zeros(count)
    if variant==1:
        values.fill(-0.)
    elif variant==2:
        values[::2]=-0.
    elif variant in (3,4,5,6,7):
        values.fill({3:7.5,4:-12.5,5:np.nextafter(0.,1.),6:np.finfo(float).max,7:-np.finfo(float).max}[variant])
    elif variant in (8,9,10):
        values[(0,count//2,count-1)[variant-8]]=1.
    elif variant==11:
        values[0]=np.nan
    elif variant==12:
        values[-1]=np.inf
    elif variant in (13,14):
        values.view(np.uint64).fill(0x7ff800000000002a if variant==13 else 0x7ff000000000002a)
    elif variant in (15,16):
        values.fill(np.inf if variant==15 else -np.inf)
    elif variant==17:
        values[::2]=np.nan
    return values


def check_scan(native,klass,count,variant):
    source,target=_new_pf(klass,count),_new_pf(klass,257)
    try:
        states=state_values(source.get_particle_states(),0)
        values=weight_values(count,variant)
        source.set_particle_states(states)
        source.set_log_weights(values)
        finite=bool(np.isfinite(values).all())
        ess=source.get_ess() if finite else None
        target.set_log_weights(np.arange(257,dtype=np.float64))
        target_ess=target.get_ess()
        before=_snapshot(native,target._state)
        expected=count>=65536 and finite and bool(np.all(values==values[0]))
        method=getattr(native,'pf_device_try_gather_uniform_weights',None)
        if method:
            accepted=method(target._state,source._state)
        else:
            accepted=expected and native.pf_device_gather_from_uniform_cdf(target._state,source._state)
        assert accepted==expected
        if accepted:
            cdf=np.cumsum(np.full(count,1./count))
            output=states[select(cdf,257)].copy()
            bits_equal(target.get_particle_states(),output)
            bits_equal(target.get_log_weights(),np.zeros(257))
        else:
            output=before[0]
            for x,y in zip(_snapshot(native,target._state),before,strict=True):
                bits_equal(x,y)
            bits_equal(target.get_ess(),target_ess)
        bits_equal(source.get_particle_states(),states)
        bits_equal(source.get_log_weights(),values)
        if finite:
            bits_equal(source.get_ess(),ess)
        return dict(accepted=bool(accepted),output_sha256=_digest(output,values),error=None,
                    every_weight_checked_all16_fields_and_source_ESS_ownership_preserved=True)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_dispatch(native,klass,count,mode):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    source,target,oracle_source,oracle_target=[_new_pf(klass,n) for n in (count,count//2,count,count//2)]
    calls=[]
    try:
        method=getattr(target,'_pf_device_try_gather_uniform_weights',None)
        if callable(method):
            def wrapped(*args):
                calls.append(1)
                return False if mode=='decline' else method(*args)
            source._pf_device_try_gather_uniform_weights=target._pf_device_try_gather_uniform_weights=wrapped
            if mode=='missing':
                target._pf_device_try_gather_uniform_weights=None
            elif mode=='mismatch':
                source._pf_device_try_gather_uniform_weights=lambda *args:method(*args)
            elif mode=='unpaired_getter':
                source._pf_device_uniform_scan_getter=None
            elif mode=='unpaired_state':
                source._pf_device_uniform_scan_state_type=None
        if mode=='python_getter':
            for pf in (source,oracle_source):
                getter=pf.get_log_weights
                pf.get_log_weights=lambda getter=getter:getter()
        elif mode=='native_getter':
            for pf in (source,oracle_source):
                getter=pf._pf_device_get_log_weights
                pf._pf_device_get_log_weights=lambda *args,getter=getter:getter(*args)
        elif mode in ('cdf_override','uniform_cdf_override'):
            name='_pf_device_gather_from_cdf' if mode=='cdf_override' else '_pf_device_gather_from_uniform_cdf'
            getter=getattr(source,name)
            def wrapper(*args):
                return getter(*args)
            for pf in (source,target,oracle_source,oracle_target):
                setattr(pf,name,wrapper)
        elif mode in ('source_count_mismatch','target_count_mismatch'):
            for pf in ((source,oracle_source) if mode.startswith('source') else (target,oracle_target)):
                pf.n_particles+=1
        oracle_target._pf_device_try_gather_uniform_weights=None
        errors=[]
        with np.errstate(all='raise' if mode=='strict_numpy' else 'ignore'):
            for a,b in ((source,target),(oracle_source,oracle_target)):
                try:
                    PersistentParticleRuntimePool._migrate(a,b)
                    errors.append(None)
                except RuntimeError as error:
                    errors.append((type(error).__name__,str(error)))
        assert errors[0]==errors[1]
        if callable(method):
            assert len(calls)==int(mode in ('active','decline')),(mode,len(calls))
        history=[]
        for a,b in ((source,oracle_source),(target,oracle_target)):
            left,right=_snapshot(native,a._state),_snapshot(native,b._state)
            for x,y in zip(left,right,strict=True):
                bits_equal(x,y)
            history.extend(left)
        return dict(output_sha256=_digest(*history),error=errors[0],original_getter_CDF_policy_and_count_fallback_exact=True)
    finally:
        for pf in (source,target,oracle_source,oracle_target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_queue(native,klass,count):
    proxy=SimpleNamespace(**vars(native))
    proxy.pf_device_gather_from_uniform_cdf=getattr(native,'pf_device_try_gather_uniform_weights',None)
    return check_uniform_queue(proxy,klass,count)


def validate_new(native,klass):
    records={}
    for count in COUNTS:
        for variant in VARIANTS:
            records[f'uniform_scan/{count}/v{variant}']=check_scan(native,klass,count,variant)
    for count in (65536,100000,1000000):
        for mode in MODES:
            records[f'uniform_scan/dispatch/{count}/{mode}']=check_dispatch(native,klass,count,mode)
        records[f'uniform_scan/queue/{count}']=check_queue(native,klass,count)
    return records


def validate(native,klass):
    records=validate_previous(native,klass)
    records.update(validate_new(native,klass))
    return records
