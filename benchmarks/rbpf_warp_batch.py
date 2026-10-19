"""Explicit choice of the standalone serial/warp likelihood kernel."""

import ctypes

from rbpf_likelihood_batch import BatchScorer


class WarpBatchScorer(BatchScorer):
    def __init__(self, dll, capacity=512, *, kernel='warp'):
        if kernel not in ('serial', 'warp'):
            raise ValueError('Choose an explicit serial or warp kernel')
        super().__init__(dll, capacity)
        try:
            self.dll.rbpf_set_warp.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_set_warp.restype = ctypes.c_int
            if self.dll.rbpf_set_warp(self.handle, int(kernel == 'warp')):
                raise RuntimeError('Kernel selection failed')
        except BaseException:
            self.close()
            raise
        self.kernel = kernel
