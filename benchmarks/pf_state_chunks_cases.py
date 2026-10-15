"""Known full-state bits and checkpoint-free PF scratch oracles."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from pf_particle_readout_cases import bits_equal, validate as validate_previous

COUNTS=(1,3,5,7,31,32,33,255,256,257,8193,65535,65536,65537,100000,100001)
METHODS=('coalesced_B4','coalesced_B60','legacy')


def check_state(native,klass,count,variant):
    pf=_new_pf(klass,count)
    try:
        states=pf.get_particle_states()
        values={1:[-0.,0.],2:[5e-324,-5e-324,1e-154],3:[1e308,-1e308,1e154],6:[np.inf,-np.inf]}
        if variant in values:
            states[:]=np.resize(values[variant],states.shape)
        elif variant in (4,5):
            values=[0x7ff8000000000051,0xfff8000000000052] if variant==4 else [0x7ff0000000000053,0xfff0000000000054]
            states.view(np.uint64)[:]=np.resize(np.array(values,dtype=np.uint64),states.shape)
        elif variant==7:
            states.view(np.uint64)[:]=np.random.default_rng(count+1912).integers(0,2**64-1,states.shape,dtype=np.uint64)
        expected=states.copy()
        prior=np.random.default_rng(count+1913).normal(0.,3.,count)
        pf.set_log_weights(prior)
        pf.get_ess()  # Populate a nonuniform cached result before import.
        pf.set_particle_states(states)
        states.fill(-999.)  # Input must no longer be consumed on return.
        bits_equal(pf.get_particle_states(),expected)
        bits_equal(pf.get_log_weights(),np.zeros(count))
        uniform=pf.get_ess()
        pf.set_log_weights(np.zeros(count))
        bits_equal(pf.get_ess(),uniform)
        pf.set_log_weights(prior)
        cached=pf.get_ess()
        first=pf.get_particle_states()
        bits_equal(first,expected)
        first.fill(-999.)
        second=pf.get_particle_states()
        bits_equal(second,expected)
        bits_equal(pf.get_particles(),expected[:,:4])
        bits_equal(pf.get_log_weights(),prior)
        bits_equal(pf.get_ess(),cached)
        return dict(input_sha256=_digest(expected,prior),output_sha256=_digest(second,prior,np.asarray(cached),np.asarray(uniform)),
                    imported_bits_exact=True,reset_weights_and_ess_exact=True,owned_output_and_completed_input=True)
    finally:
        native.pf_device_destroy(pf._state)
        pf._state=None


def check_history(native,klass,count,common,method):
    actual,oracle=_new_pf(klass,count),_new_pf(klass,count)
    history=[]
    try:
        if common:
            for pf in (actual,oracle):
                pf.set_velocity_covariance(np.eye(3)*2.)
        else:
            states=actual.get_particle_states()
            states[:,7:]=(np.eye(3)[None,:,:]*(1.+np.arange(count)[:,None,None]/count)).reshape(count,9)
            for pf in (actual,oracle):
                pf.set_particle_states(states)
        prior=np.random.default_rng(count+1914).normal(0.,2.,count)
        for step in range(1,3):
            for pf in (actual,oracle):
                pf.predict([1.,2.,3.],dt=.17,rbpf_velocity_kf=True,velocity_process_noise=.13)
            # Oracle does not checkpoint/import here. Actual must preserve its
            # completed state and restore weights after the documented reset.
            checkpoint=actual.get_particle_states()
            actual.set_particle_states(checkpoint)
            for pf in (actual,oracle):
                pf.set_log_weights(prior)
                if method=='legacy':
                    native.pf_device_resample_megopolis(pf._state,3,191+step)
                elif method=='systematic':
                    native.pf_device_resample_systematic(pf._state,191+step)
                else:
                    native.pf_device_resample_megopolis_coalesced(pf._state,60 if method.endswith('B60') else 4,191+step)
            actual.get_particle_states()  # After the queued gather pointer swap.
            actual.get_particles()
            native.pf_device_position_mixture_components(actual._state,np.zeros((2,3)),np.zeros(2),6.25)
            for pf in (actual,oracle):
                _kf(native,pf._state,_observations(12))
                pf.predict([1.,2.,3.],dt=.1,rbpf_velocity_kf=True,velocity_process_noise=.1)
            a,b=_snapshot(native,actual._state),_snapshot(native,oracle._state)
            for left,right in zip(a,b,strict=True):
                bits_equal(left,right)
            history.extend(a)
        return dict(output_sha256=_digest(*history),checkpoint_free_oracle_identical=True,
                    queued_prediction_gather_KF_and_geometry_scratch_safe=True)
    finally:
        for pf in (actual,oracle):
            native.pf_device_destroy(pf._state)
            pf._state=None


def check_representation(native,klass,form):
    pf=_new_pf(klass,257)
    try:
        expected=np.arange(257*16,dtype=np.float64).reshape(257,16)
        if form=='F':
            value=np.asfortranarray(expected)
        elif form=='strided':
            storage=np.empty((514,32))
            value=storage[::2,::2]
            value[:]=expected
        else:
            value=expected.copy()
        pf.set_particle_states(value)
        value.fill(-999.)
        actual=pf.get_particle_states()
        bits_equal(actual,expected)
        bits_equal(pf.get_log_weights(),np.zeros(257))
        return dict(output_sha256=_digest(actual),coercion_and_input_completion_exact=True)
    finally:
        native.pf_device_destroy(pf._state)
        pf._state=None


def validate(native,klass):
    records=validate_previous(native,klass)
    for count in COUNTS:
        for variant in range(8):
            records[f'chunks/N{count}/v{variant}']=check_state(native,klass,count,variant)
        for common in (False,True):
            for method in METHODS+(('systematic',) if count<=257 else ()):
                records[f'chunks/N{count}/common{common}/{method}']=check_history(native,klass,count,common,method)
    for form in ('C','F','strided'):
        records[f'chunks/representation/{form}']=check_representation(native,klass,form)
    return records
