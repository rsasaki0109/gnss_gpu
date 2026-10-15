"""Exact CPU CDF arithmetic, input ownership, and NumPy error policy."""

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np

from bench_pf_covariance_compare import _digest
from pf_particle_readout_cases import bits_equal
from pf_pool_cdf_cases import validate as validate_previous

COUNTS=(1,31,257,65535,65536,65537,100000,1000000)
FORMS=('contiguous','strided','reverse','readonly')
POLICIES=('ignore','warn','raise','call','log','print','warning_error')


def values(count,variant):
    data=np.random.default_rng(count+1950).normal(0.,3.,count)
    if variant==0:
        data[:]=np.resize([-0.,0.],count)
    elif variant==2:
        data=np.random.default_rng(count+1951).uniform(-745.,5.,count)
    elif variant==3:
        data[:]=np.resize([-1e308,1e308],count)
    elif variant==4:
        data[:]=np.resize([5e-324,-5e-324,1e-154,-1e-154],count)
    elif variant==5:
        data[:]=np.resize([np.nan,np.inf,-np.inf,-1e308,0.],count)
    elif variant==6:
        data.fill(-np.inf)
    elif variant==7:
        data.view(np.uint64)[:]=np.random.default_rng(count+1952).integers(0,2**64-1,count,dtype=np.uint64)
    return data


def representation(data,form):
    if form=='strided':
        storage=np.zeros(2*data.size)
        storage[::2]=data
        return storage[::2]
    if form=='reverse':
        return data[::-1]
    if form=='readonly':
        data.flags.writeable=False
    if form=='matrix':
        return data.reshape(256,-1)
    return data


def original_cdf(log_weights):
    log_weights=np.asarray(log_weights,dtype=np.float64)
    finite=np.isfinite(log_weights)
    if not np.any(finite):
        weights=np.full(log_weights.size,1./log_weights.size)
    else:
        floor=float(np.min(log_weights[finite]))-100.
        shifted=np.where(finite,log_weights,floor)
        weights=np.exp(shifted-float(np.max(shifted)))
        weights/=np.sum(weights)
    return np.cumsum(weights)


def evaluate(data,policy,operation):
    calls=[]
    messages=io.StringIO()
    printed=io.StringIO()
    prior=np.geterrcall()
    np.seterrcall(messages if policy=='log' else lambda message,flag:calls.append((message,flag)))
    try:
        # NumPy's print policy writes native fd 2 rather than sys.stderr.
        # Serial tests capture both channels without filling a worker pipe.
        with tempfile.TemporaryFile(dir=Path(__file__).resolve().parents[1]/'dist') as native_output:
            saved_fd=os.dup(2)
            try:
                os.dup2(native_output.fileno(),2)
                with warnings.catch_warnings(record=True) as observed,redirect_stderr(printed):
                    warnings.simplefilter('error' if policy=='warning_error' else 'always',RuntimeWarning)
                    with np.errstate(all='warn' if policy=='warning_error' else policy):
                        try:
                            result=operation(data)
                            error=None
                        except (FloatingPointError,RuntimeWarning) as exception:
                            result=None
                            error=(type(exception).__name__,str(exception))
                    metadata=dict(error=error,callbacks=calls,log=messages.getvalue(),printed=printed.getvalue(),
                        warnings=[(item.category.__name__,str(item.message)) for item in observed])
            finally:
                os.dup2(saved_fd,2)
                os.close(saved_fd)
            native_output.seek(0)
            metadata['printed']+=native_output.read().decode('utf8')
        return result,metadata
    finally:
        np.seterrcall(prior)


def runtime_cdf(data):
    from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
    captured=[]
    count=data.size
    target_count=max(65536,count)
    def cdf_gather(target,source,cdf):
        captured.append(cdf.copy())
        return True
    def ancestors(*args):
        raise AssertionError('CDF capture must be reached')
    def forbidden(*args):
        raise AssertionError('No host full-state migration')
    source=SimpleNamespace(_initialized=True,_state=SimpleNamespace(n_particles=count),
        _pf_device_gather_from=ancestors,_pf_device_gather_from_cdf=cdf_gather,
        get_log_weights=lambda:data,get_particle_states=forbidden)
    target=SimpleNamespace(_initialized=True,_state=SimpleNamespace(n_particles=target_count),
        n_particles=target_count,_pf_device_gather_from=ancestors,_pf_device_gather_from_cdf=cdf_gather,
        set_log_weights=lambda weights:None,set_particle_states=forbidden)
    PersistentParticleRuntimePool._migrate(source,target)
    assert len(captured)==1
    return captured[0]


def check_cdf(count,variant,form,policy='ignore'):
    data=representation(values(count,variant),form)
    before=data.copy()
    expected,metadata=evaluate(data,policy,original_cdf)
    actual,actual_metadata=evaluate(data,policy,runtime_cdf)
    assert metadata==actual_metadata,(count,variant,form,policy,metadata,actual_metadata)
    if policy=='print' and variant in (2,3,5,7):
        assert metadata['printed'], 'Native NumPy warning text must be captured'
    bits_equal(data,before)
    if expected is not None:
        bits_equal(actual,expected)
    return dict(input_sha256=_digest(before),CDF_sha256=None if actual is None else _digest(actual),
        **actual_metadata,caller_weight_bits_preserved=True)


def validate(native,klass):
    records=validate_previous(native,klass)
    for count in COUNTS:
        for variant in range(8):
            for form in FORMS:
                records[f'normalize/{count}/v{variant}/{form}']=check_cdf(count,variant,form)
    for variant in range(8):
        for policy in POLICIES:
            for form in ('contiguous','matrix'):
                records[f'normalize/policy/v{variant}/{policy}/{form}']=check_cdf(65536,variant,form,policy)
    return records
