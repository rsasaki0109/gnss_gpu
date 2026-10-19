"""Explicit original/generic/fixed11 standalone likelihood kernel selection."""

import ctypes

from rbpf_likelihood_batch import BatchScorer


class FixedLUScorer(BatchScorer):
    def __init__(self, dll, capacity=512, kernel="fixed"):
        assert kernel in ("original", "generic", "fixed")
        super().__init__(dll, capacity)
        self.kernel = kernel
        self.dll.rbpf_fixed_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_fixed_select.restype = ctypes.c_int
        if self.dll.rbpf_fixed_select(self.handle, ("original", "generic", "fixed").index(kernel)):
            self.close()
            raise RuntimeError("Standalone fixed LU kernel selection failed")
