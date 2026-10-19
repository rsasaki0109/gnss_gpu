"""Optional Windows host API clocks around literal corrected D236 native calls."""

import ctypes

import numpy as np

from rbpf_score_host_phases_v2 import HostPhaseScorer

PHASES = ("entry", "pack", "H2D", "launch", "status", "D2H", "unpack", "exit")


def parse_trace(raw):
    if isinstance(raw, np.ndarray):
        if raw.ndim != 1 or raw.dtype != np.dtype(np.uint64):
            raise ValueError("Native uint64 vector required")
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or any(type(v) is not int or v < 0 for v in raw):
        raise ValueError("Nonnegative integer trace words required")
    if len(raw) < 6:
        raise ValueError("Truncated trace header")
    version, generation, frequency, count, flags, active = raw[:6]
    if version != 1 or generation <= 0 or frequency <= 0 or flags or active:
        raise ValueError("Invalid,active,or unsupported trace")
    if not 2 <= count <= 8 or len(raw) != 6 + 4 * count:
        raise ValueError("Invalid native trace length")
    rows = [raw[6 + i * 4:10 + i * 4] for i in range(count)]
    ids = [r[0] for r in rows]
    if ids[0] != 0 or ids[-1] != 7 or ids != sorted(set(ids)) or any(i > 7 for i in ids):
        raise ValueError("Native phase order changed")
    if any(b[k] < a[k] for a, b in zip(rows, rows[1:]) for k in (1, 2, 3)):
        raise ValueError("A native wall or CPU clock moved backwards")
    segments = []
    for a, b in zip(rows, rows[1:]):
        segments.append(dict(phase=PHASES[a[0]], wall_ticks=b[1]-a[1],
            user_100ns=b[2]-a[2], kernel_100ns=b[3]-a[3],
            thread_100ns=b[2]-a[2]+b[3]-a[3]))
    totals = dict(wall_ticks=rows[-1][1]-rows[0][1], user_100ns=rows[-1][2]-rows[0][2],
        kernel_100ns=rows[-1][3]-rows[0][3])
    totals["thread_100ns"] = totals["user_100ns"] + totals["kernel_100ns"]
    if any(sum(s[key] for s in segments) != value for key, value in totals.items()):
        raise ValueError("Native integer clock conservation failed")
    return dict(version=version, generation=generation, frequency=frequency,
        raw_words=list(raw), full_GPU_path=ids == list(range(8)),
        segments=segments, totals=totals,
        scope="QPC ticks and calling-thread100ns samples;keep zeros and quantization. Transfer/completion phases include runtime waits. Not kernel time,GPU utilization or a speed gate.")


class NativeCallScorer(HostPhaseScorer):
    """Explicit per-instance traced export;returned numeric objects untouched."""

    def __init__(self, dll, capacity=512, *, enabled=True):
        if type(enabled) is not bool:
            raise TypeError("Explicit trace bool required")
        super().__init__(dll, capacity)
        self.native_records = []
        self.last_native_trace = None
        self._native_generation = 0
        if not enabled:
            return
        traced = self.dll.rbpf_score_trace
        traced.argtypes, traced.restype = self.dll.rbpf_score.argtypes, self.dll.rbpf_score.restype
        self.dll.rbpf_score = traced
        reader = self.dll.rbpf_trace_read
        reader.argtypes = [ctypes.c_void_p, np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"), ctypes.c_int]
        reader.restype = ctypes.c_int
        function = self.score

        def recorded(*args, **kwargs):
            self.last_native_trace = None
            try:
                return function(*args, **kwargs)
            finally:
                row = dict(index=len(self.native_records), trace=None, probe_error=None)
                try:
                    if self.handle:
                        data = np.empty(38, dtype=np.uint64)
                        count = reader(self.handle, data, len(data))
                        if count < 0 or count > len(data):
                            raise ValueError("Native trace read failed:" + str(count))
                        if count:
                            parsed = parse_trace(data[:count])
                            if parsed["generation"] != self._native_generation:
                                self._native_generation = parsed["generation"]
                                self.last_native_trace = parsed
                                row["trace"] = parsed
                except Exception as error:
                    row["probe_error"] = type(error).__name__ + ":" + str(error)
                self.native_records.append(row)
        self.score = recorded
