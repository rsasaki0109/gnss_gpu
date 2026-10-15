"""CDF validator SIMD boundaries, IEEE values, unaligned inputs, and ownership."""

import numpy as np

from bench_pf_covariance_broadcast_compare import _new_pf
from bench_pf_covariance_compare import _digest
from pf_particle_readout_cases import bits_equal
from pf_pool_uniform_cases import validate as validate_previous

COUNTS = (1, 2, 3, 4, 5, 31, 32, 33, 65535, 65536, 65537, 65538, 65539,
          65540, 65541, 65542, 65543, 1000000)
VARIANTS = tuple(range(28))
FORMS = ('contiguous', 'offset', 'byte_unaligned', 'strided', 'readonly')


def values(count, variant):
    if variant == 0:
        return np.cumsum(np.full(count, 1./count))
    if variant in (1, 2):
        return np.resize([-0., 0.] if variant == 2 else [0.], count)
    if variant in (3, 4):
        limit = np.finfo(float).max if variant == 3 else 5e-324
        data = np.full(count, -limit)
        data[count//3:] = -0.
        data[2*count//3:] = limit
        return data
    if variant == 5:
        data = np.random.default_rng(count+198).random(count)
        return np.cumsum(data/np.sum(data))
    if variant in (6, 7):
        thresholds = (np.arange(31, dtype=np.float64)+.5)/31
        data = thresholds[np.linspace(0, 30, count, dtype=np.int64)]
        return data if variant == 6 else np.nextafter(data, -np.inf)
    data = np.arange(count, dtype=np.float64)/max(1, count)
    if 8 <= variant <= 16:
        positions = (0, count-1, count//2, 1, 2, 3, count-3, count-2, count-1)
        patterns = (0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000042,
                    0x7ff0000000000042, 0xfff8000000000042, 0xfff0000000000042,
                    0x7ff0000000000000, 0x7ff8000000000042, 0x7ff0000000000042)
        offset = min(count-1, max(0, positions[variant-8]))
        data.view(np.uint64)[offset] = patterns[variant-8]
    elif 17 <= variant <= 25:
        positions = (1, 2, 3, 4, 5, count//2, count-3, count-2, count-1)
        offset = min(count-1, max(1, positions[variant-17]))
        data[offset] = data[max(0, offset-1)]-1.
    elif variant == 26:
        data.view(np.uint64)[:] = np.random.default_rng(count+199).integers(0, 2**64-1, count, dtype=np.uint64)
    elif variant == 27:
        data = data[::-1].copy()
    return data


def representation(data, form):
    if form == 'offset':
        storage = np.empty(data.size+1)
        storage[1:] = data
        return storage[1:]
    if form == 'byte_unaligned':
        storage = np.empty(data.nbytes+1, dtype=np.uint8)
        result = np.ndarray(data.shape, dtype=np.float64, buffer=storage, offset=1)
        result[:] = data
        assert not result.flags.aligned
        return result
    if form == 'strided':
        storage = np.empty(data.size*2)
        storage[::2] = data
        return storage[::2]
    if form == 'readonly':
        data.flags.writeable = False
    return data


class ValidationSession:
    def __init__(self, native, klass, count):
        self.native, self.count = native, count
        self.source = _new_pf(klass, count)
        self.target = _new_pf(klass, 31)
        self.states = self.source.get_particle_states()
        self.weights = self.source.get_log_weights()
        self.ess = self.source.get_ess()

    def check(self, variant, form):
        data = representation(values(self.count, variant), form)
        before = data.copy()
        with np.errstate(all='ignore'):
            expected = bool(np.all(np.isfinite(data)) and np.all(data[1:] >= data[:-1]))
        prior_states = self.target.get_particle_states()
        prior_weights = self.target.get_log_weights()
        prior_ess = self.target.get_ess()
        accepted = self.native.pf_device_gather_from_cdf(self.target._state, self.source._state, data)
        assert accepted == expected, (self.count, variant, form, accepted, expected)
        states, weights, ess = self.target.get_particle_states(), self.target.get_log_weights(), self.target.get_ess()
        if expected:
            indices = np.minimum(np.searchsorted(data, (np.arange(31)+.5)/31, side='left'), self.count-1)
            bits_equal(states, self.states[indices])
            bits_equal(weights, np.zeros(31))
        else:
            bits_equal(states, prior_states)
            bits_equal(weights, prior_weights)
            bits_equal(ess, prior_ess)
        bits_equal(data, before)
        return dict(input_sha256=_digest(before), output_sha256=_digest(states, weights, np.asarray(ess)),
                    accepted=accepted, error=None, input_and_declined_target_bits_preserved=True,
                    exact_left_search_and_all16_selected_fields=True)

    def close(self):
        try:
            bits_equal(self.source.get_particle_states(), self.states)
            bits_equal(self.source.get_log_weights(), self.weights)
            bits_equal(self.source.get_ess(), self.ess)
        finally:
            for pf in (self.source, self.target):
                self.native.pf_device_destroy(pf._state)
                pf._state = None


def validate(native, klass):
    records = validate_previous(native, klass)
    for count in COUNTS:
        session = ValidationSession(native, klass, count)
        try:
            for variant in VARIANTS:
                for form in FORMS:
                    records[f'cdf_validate/{count}/v{variant}/{form}'] = session.check(variant, form)
        finally:
            session.close()
    return records
