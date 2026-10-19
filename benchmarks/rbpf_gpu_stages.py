"""Optional elapsed-cycle diagnostics for the unchanged standalone likelihood math."""

import ctypes

import numpy as np

from rbpf_likelihood_batch import BatchScorer


class StageScorer(BatchScorer):
    def __init__(self, dll, capacity=512):
        super().__init__(dll, capacity)
        self.dll.rbpf_profile_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_profile_select.restype = ctypes.c_int
        self.dll.rbpf_profile_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_profile_read.restype = ctypes.c_int

    def profiled_score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False, control=False):
        if not self.handle:
            raise RuntimeError("Closed stage diagnostic workspace")
        code = self.dll.rbpf_profile_select(self.handle, 1 if control else 2)
        if code:
            raise RuntimeError(f"Stage diagnostic selection failed:{code}")
        try:
            values, details = BatchScorer.score(self, hypotheses, problem, oracle, mode, diagnostics)
            count = self.dll.rbpf_profile_read(self.handle, None, 0)
            trace = None
            if count:
                if count != details["unique"]:
                    raise RuntimeError("Unexpected stage trace count")
                trace = np.empty((count, 16), dtype=np.uint64)
                copied = self.dll.rbpf_profile_read(self.handle, ctypes.c_void_p(trace.ctypes.data), count)
                if copied != count:
                    raise RuntimeError("Stage trace copy failed")
            return values, details, None if control else trace
        finally:
            self.dll.rbpf_profile_select(self.handle, 0)
