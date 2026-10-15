"""Exact uniform and almost-uniform CDF, policy, and input ownership checks."""

import numpy as np

from bench_pf_covariance_compare import _digest
from pf_particle_readout_cases import bits_equal
from pf_pool_normalize_cases import (
    COUNTS, FORMS, POLICIES, evaluate, original_cdf, representation, runtime_cdf,
)
from pf_uniform_weights_cases import validate as validate_previous

VARIANTS = tuple(range(23))


def values(count, variant):
    scalars = (-0., 0., -1., 1., -np.finfo(float).max, np.finfo(float).max,
               5e-324, -5e-324, np.finfo(float).tiny, -np.finfo(float).tiny,
               np.nextafter(1., 0.), np.nextafter(-1., -np.inf))
    if variant < len(scalars):
        return np.full(count, scalars[variant], dtype=np.float64)
    data = np.zeros(count, dtype=np.float64)
    if variant in (12, 13, 14):
        offset = (0, count//2, count-1)[variant-12]
        data[offset] = np.nextafter(0., 1.)
    elif variant == 15:
        data[count//2] = -5e-324
    elif variant == 16:
        data[-1] = -745.
    elif variant == 17:
        data[0] = -745.
    elif variant == 18:
        data.fill(np.inf)
    elif variant == 19:
        data.view(np.uint64).fill(0x7ff8000000000042)
    elif variant == 20:
        data.fill(-np.inf)
    elif variant == 21:
        data[min(1, count-1)] = -745.
    elif variant == 22:
        data[count//4] = -745.
    return data


def check_cdf(count, variant, form, policy='ignore'):
    data = representation(values(count, variant), form)
    before = data.copy()
    expected, metadata = evaluate(data, policy, original_cdf)
    actual, actual_metadata = evaluate(data, policy, runtime_cdf)
    assert metadata == actual_metadata, (count, variant, form, policy, metadata, actual_metadata)
    bits_equal(data, before)
    if expected is not None:
        bits_equal(actual, expected)
    return dict(input_sha256=_digest(before), CDF_sha256=None if actual is None else _digest(actual),
                **actual_metadata, caller_weight_bits_preserved=True)


def validate(native, klass):
    records = validate_previous(native, klass)
    for count in COUNTS:
        for variant in VARIANTS:
            for form in FORMS:
                records[f'pool_uniform/{count}/v{variant}/{form}'] = check_cdf(count, variant, form)
    for variant in VARIANTS:
        for policy in POLICIES:
            for form in ('contiguous', 'matrix'):
                records[f'pool_uniform/policy/v{variant}/{policy}/{form}'] = check_cdf(65536, variant, form, policy)
    return records
