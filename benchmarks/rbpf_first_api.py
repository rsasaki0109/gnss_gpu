"""Optional first-operation diagnostic;original completed score is inherited."""

import ctypes

import numpy as np

from rbpf_native_call_trace import NativeCallScorer

OPERATIONS = ("none", "query", "sync", "h2d1", "h2d64k", "memset1")
BYTES = (0, 0, 0, 1, 65536, 1)
API_COUNTS = (0, 1, 1, 2, 2, 2)


def parse_probe(raw):
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1 or raw.dtype != np.dtype(np.uint64):
            raise ValueError("Native uint64 vector required")
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) != 20 or any(type(v) is not int or v < 0 for v in raw):
        raise ValueError("20 nonnegative integer probe words required")
    version, op, frequency, byte_count, api_status, completion_status, flags, api_count = raw[:8]
    if version != 1 or not 0 <= op < 6 or frequency <= 0 or api_status or completion_status or flags:
        raise ValueError("Invalid or failed first API observation")
    if byte_count != BYTES[op] or api_count != API_COUNTS[op]:
        raise ValueError("Operation bytes or API count changed")
    rows = [raw[8+i*4:12+i*4] for i in range(3)]
    if [r[0] for r in rows] != [0, 1, 2] or any(b[k] < a[k] for a, b in zip(rows, rows[1:]) for k in (1, 2, 3)):
        raise ValueError("Probe clock/order changed")
    spans = {}
    for phase, a, b in zip(("API", "completion"), rows[:-1], rows[1:], strict=True):
        spans[phase] = dict(wall_ticks=b[1]-a[1], user_100ns=b[2]-a[2], kernel_100ns=b[3]-a[3])
        spans[phase]["thread_100ns"] = spans[phase]["user_100ns"] + spans[phase]["kernel_100ns"]
    total = {key: sum(s[key] for s in spans.values()) for key in ("wall_ticks", "user_100ns", "kernel_100ns", "thread_100ns")}
    assert total["wall_ticks"] == rows[-1][1] - rows[0][1]
    return dict(operation=OPERATIONS[op], frequency=frequency, raw_words=list(raw), bytes=byte_count,
        CUDA_API_count=api_count, spans=spans, total=total,
        scope="Host QPC/thread CPU observation,including runtime/waits/preemption. API plus explicit completion is diagnostic overhead. Not pure device time or speed evidence.")


class FirstApiScorer(NativeCallScorer):
    def __init__(self, dll, capacity=512):
        super().__init__(dll, capacity)
        self.dll.rbpf_first_api_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.dll.rbpf_first_api_prepare.restype = ctypes.c_int
        self.dll.rbpf_first_api_probe.argtypes = [ctypes.c_void_p, ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.dll.rbpf_first_api_probe.restype = ctypes.c_int

    def prepare_probe(self, byte_count=65536):
        assert self.handle
        if type(byte_count) is not int or not 1 <= byte_count <= 65536:
            raise ValueError("Explicit prepare size in1..65536 required")
        if self.dll.rbpf_first_api_prepare(self.handle, byte_count):
            raise ValueError("Prepare exceeds native workspace capacity")

    def probe_raw(self, operation):
        assert self.handle
        if type(operation) is not str or operation not in OPERATIONS:
            raise ValueError("Explicit supported first operation required")
        data = np.empty(20, dtype=np.uint64)
        if self.dll.rbpf_first_api_probe(self.handle, OPERATIONS.index(operation), data, len(data)) != 20:
            raise RuntimeError("First operation probe failed")
        return data
