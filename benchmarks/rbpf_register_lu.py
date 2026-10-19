"""Explicit original/generic/register-row11 standalone likelihood kernel selection."""

import ctypes

from rbpf_likelihood_batch import BatchScorer


class RegisterLUScorer(BatchScorer):
    def __init__(self, dll, capacity=512, kernel="register"):
        assert kernel in ("original", "generic", "register")
        super().__init__(dll, capacity)
        self.kernel = kernel
        self.dll.rbpf_register_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_register_select.restype = ctypes.c_int
        if self.dll.rbpf_register_select(self.handle, ("original", "generic", "register").index(kernel)):
            self.close()
            raise RuntimeError("Standalone register LU kernel selection failed")
