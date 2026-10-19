"""Explicit synchronous pinned/pageable experiment;no default installation."""

import ctypes

import numpy as np

from rbpf_native_call_trace import NativeCallScorer
from rbpf_register_lu import RegisterLUScorer


def _check(transport):
    if transport not in ("pageable", "pinned") or type(transport) is not str:
        raise ValueError("Explicit pageable or pinned transport required")


def _select(scorer, transport):
    try:
        choose = scorer.dll.rbpf_host_staging_select
        choose.argtypes, choose.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
        code = choose(scorer.handle, int(transport == "pinned"))
        if code:
            raise RuntimeError("Host staging selection failed:" + str(code))
        info = scorer.dll.rbpf_host_staging_info
        info.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"), ctypes.c_int]
        info.restype = ctypes.c_int
        values = np.empty(5, dtype=np.uint64)
        if info(scorer.handle, values, len(values)) != 5 or values[:3].tolist() != [1, 1, int(transport == "pinned")]:
            raise RuntimeError("Host staging metadata mismatch")
        scorer.transport = transport
        scorer.staging_bytes = dict(input=int(values[3]), output=int(values[4]))
    except BaseException:
        scorer.close()
        raise


class PinnedTransportScorer(RegisterLUScorer):
    def __init__(self, dll, capacity=512, *, transport):
        _check(transport)
        super().__init__(dll, capacity)
        _select(self, transport)


class TracedPinnedTransportScorer(NativeCallScorer):
    def __init__(self, dll, capacity=512, *, transport):
        _check(transport)
        super().__init__(dll, capacity)
        _select(self, transport)
