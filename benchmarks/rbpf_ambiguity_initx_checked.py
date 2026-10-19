"""Narrow scalar domain before the frozen bulk-store implementation.

Retain the first version and its timings. Arbitrary-size integers and
custom/wider scalar conversions use the literal original method, including
its original partial state when a conversion raises during diagonal stores.
"""

import numpy as np

from rbpf_ambiguity_initx import (  # noqa: F401
    InitxEngine as LegacyInitxEngine,
    capture_input,
    capture_output,
    exact_output,
    restore,
)

_INTS = (
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_FLOATS = (float, np.float64)


def _store_scalar(value):
    if type(value) in _FLOATS:
        return True
    if type(value) in _INTS:
        return type(value) is not int or value.bit_length() <= 64
    return False


class InitxEngine(LegacyInitxEngine):
    def __call__(self, rtk, x0, v0, i):
        if type(i) in _INTS and _store_scalar(x0) and _store_scalar(v0):
            return super().__call__(rtk, x0, v0, i)
        self.calls += 1
        self.fallback_calls += 1
        return self.original(rtk, x0, v0, i)


def make_type(original):
    engine = InitxEngine(original)

    class CheckedBulkInitx(original):
        def initx(self, x0, v0, i):
            return engine(self, x0, v0, i)

    return CheckedBulkInitx, engine
