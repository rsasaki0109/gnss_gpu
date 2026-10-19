"""Explicit structural-zero GPU normal products;inherit all frozen score code."""

import ctypes

from rbpf_likelihood_batch import BatchScorer


class SparseNormalScorer(BatchScorer):
    def __init__(self, dll, capacity=512, *, kernel="sparse"):
        if kernel not in ("original", "dense", "sparse"):
            raise ValueError("Select original/dense/sparse GPU kernel")
        super().__init__(dll, capacity)
        try:
            self.dll.rbpf_sparse_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_sparse_select.restype = ctypes.c_int
            self.dll.rbpf_sparse_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_sparse_profile.restype = ctypes.c_int
            self.dll.rbpf_sparse_profile_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
            self.dll.rbpf_sparse_profile_get.restype = ctypes.c_int
            if self.dll.rbpf_sparse_select(self.handle, ("original", "dense", "sparse").index(kernel)):
                raise RuntimeError("Explicit normal kernel selection failed")
        except BaseException:
            self.close()
            raise
        self.kernel = kernel

    def count_score(self, hypotheses, problem, oracle, *, diagnostics=False):
        """Separate attribution call;ordinary score transfers no counters."""
        if not self.handle:
            raise RuntimeError("Closed scorer")
        if self.dll.rbpf_sparse_profile(self.handle, 1):
            raise RuntimeError("Normal counter allocation failed")
        try:
            result = self.score(hypotheses, problem, oracle, "gpu", diagnostics)
            counts = (ctypes.c_ulonglong * 4)()
            code = self.dll.rbpf_sparse_profile_get(self.handle, counts)
            if code not in (0, 1):
                raise RuntimeError("Normal counter query failed")
            data = dict(zip(("dense_stages", "sparse_stages", "skipped_products", "unsafe_stages"), counts, strict=True)) if code == 0 else None
            return *result, data
        finally:
            if self.dll.rbpf_sparse_profile(self.handle, 0):
                raise RuntimeError("Normal counter reset failed")
