"""Optional elapsed-cycle diagnostics for the unchanged standalone likelihood math."""

import ctypes

import numpy as np

from rbpf_likelihood_batch import BatchScorer


class LUPhaseScorer(BatchScorer):
    def __init__(self, dll, capacity=512):
        super().__init__(dll, capacity)
        self.dll.rbpf_profile_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_profile_select.restype = ctypes.c_int
        self.dll.rbpf_profile_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_profile_read.restype = ctypes.c_int

    def profiled_score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False, control=None):
        if not self.handle:
            raise RuntimeError("Closed LU phase diagnostic workspace")
        code = self.dll.rbpf_profile_select(self.handle, {None: 2, "copy": 1, "barriers": 3}[control])
        if code:
            raise RuntimeError(f"LU phase diagnostic selection failed:{code}")
        try:
            values, details = BatchScorer.score(self, hypotheses, problem, oracle, mode, diagnostics)
            count = self.dll.rbpf_profile_read(self.handle, None, 0)
            trace = None
            if count:
                if count != details["unique"]:
                    raise RuntimeError("Unexpected LU phase trace count")
                trace = np.empty((count, 48), dtype=np.uint64)
                copied = self.dll.rbpf_profile_read(self.handle, ctypes.c_void_p(trace.ctypes.data), count)
                if copied != count:
                    raise RuntimeError("LU phase trace copy failed")
            return values, details, None if control else trace
        finally:
            self.dll.rbpf_profile_select(self.handle, 0)
