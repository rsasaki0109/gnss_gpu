"""Explicit corrected register scorer with first-system matrix/LU sharing."""

import ctypes

import numpy as np

from rbpf_register_lu import RegisterLUScorer


class InitialShareScorer(RegisterLUScorer):
    """Inherited complete score;usage readback is a separate accuracy query."""

    def __init__(self, dll, capacity=512, *, enabled):
        if type(enabled) is not bool:
            raise TypeError("enabled must be an explicit bool")
        super().__init__(dll, capacity, kernel="register")
        self.enabled = enabled
        integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        self.dll.rbpf_initial_share_enable.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_initial_share_enable.restype = ctypes.c_int
        self.dll.rbpf_initial_share_query.argtypes = [
            ctypes.c_void_p,
            integer,
            integer,
            integer,
        ]
        self.dll.rbpf_initial_share_query.restype = ctypes.c_int
        self.dll.rbpf_initial_share_sizes.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
        ]
        self.dll.rbpf_initial_share_sizes.restype = ctypes.c_int
        if self.dll.rbpf_initial_share_enable(self.handle, int(enabled)):
            self.close()
            raise RuntimeError("Standalone initial-sharing configuration failed")

    def usage(self):
        assert self.handle
        owners, used = (
            np.empty(self.capacity, np.int32),
            np.empty(self.capacity, np.int32),
        )
        summary = np.empty(4, np.int32)
        code = self.dll.rbpf_initial_share_query(self.handle, owners, used, summary)
        if code:
            raise RuntimeError("Initial-sharing accuracy query failed", code)
        count, duplicates, members, launches = map(int, summary)
        assert 0 <= count <= self.capacity
        return dict(
            count=count,
            duplicates=duplicates,
            members=members,
            launches=launches,
            owners=owners[:count].copy(),
            used=used[:count].copy(),
        )

    def memory_layout(self):
        values = np.empty(3, np.uint64)
        assert self.dll.rbpf_initial_share_sizes(values) == 0
        return dict(
            template_bytes=int(values[0]),
            key_bytes=int(values[1]),
            owner_bytes=int(values[2]),
        )
