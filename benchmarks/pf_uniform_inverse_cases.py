"""Exact direct inverse of sequential FP64 CDF spans, without CDF output."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest, _snapshot
from pf_particle_readout_cases import bits_equal
from pf_pool_cdf_cases import select
from pf_pool_gather_cases import state_values
from pf_uniform_cdf_cases import validate as validate_previous

SOURCE_COUNTS=(65536,65537,100000,131071,500000,999999,1000000,1048577)
TARGET_COUNTS=(129,258,65535,65537,100001)
VARIANTS=(0,2)


def check_inverse(native,klass,count,target_count,variant):
    source,target=_new_pf(klass,count),_new_pf(klass,target_count)
    try:
        states=state_values(source.get_particle_states(),variant)
        weights=np.random.default_rng(count+200).normal(size=count)
        source.set_particle_states(states)
        source.set_log_weights(weights)
        ess=source.get_ess()
        target.set_log_weights(np.arange(target_count,dtype=np.float64))
        target.get_ess()
        before=_snapshot(native,target._state)
        cdf=np.cumsum(np.full(count,1./count))
        expected=states[select(cdf,target_count)].copy()
        accepted=native.pf_device_gather_from_uniform_cdf(target._state,source._state)
        if not accepted:
            for x,y in zip(_snapshot(native,target._state),before,strict=True):
                bits_equal(x,y)
            assert native.pf_device_gather_from_cdf(target._state,source._state,cdf)
        bits_equal(target.get_particle_states(),expected)
        bits_equal(target.get_log_weights(),np.zeros(target_count))
        bits_equal(source.get_particle_states(),states)
        bits_equal(source.get_log_weights(),weights)
        bits_equal(source.get_ess(),ess)
        return dict(output_sha256=_digest(expected,weights,np.asarray(ess)),error=None,
                    accepted=accepted,all16_selected_fields_and_source_ownership_exact_without_CDF_output=True)
    finally:
        for pf in (source,target):
            native.pf_device_destroy(pf._state)
            pf._state=None


def validate(native,klass):
    records=validate_previous(native,klass)
    for count in SOURCE_COUNTS:
        for target in TARGET_COUNTS:
            for variant in VARIANTS:
                records[f'uniform_inverse/{count}/target{target}/v{variant}']=check_inverse(native,klass,count,target,variant)
    return records
