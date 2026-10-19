"""Explicit GPU transport choice;inherit the frozen whole-score math/packing."""

import ctypes
import time

from rbpf_likelihood_batch import BatchScorer


class StreamBatchScorer(BatchScorer):
    def __init__(self, dll, capacity=512, *, stream="nonblocking"):
        if stream not in ("default", "nonblocking"):
            raise ValueError("Select an explicit default/nonblocking stream")
        super().__init__(dll, capacity)
        try:
            self.dll.rbpf_stream_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_stream_select.restype = ctypes.c_int
            self.dll.rbpf_stream_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_stream_profile.restype = ctypes.c_int
            self.dll.rbpf_stream_profile_get.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
            self.dll.rbpf_stream_profile_get.restype = ctypes.c_int
            if self.dll.rbpf_stream_select(self.handle, int(stream == "nonblocking")):
                raise RuntimeError("Explicit stream creation/selection failed")
        except BaseException:
            self.close()
            raise
        self.stream = stream

    def profile_score(self, hypotheses, problem, oracle, *, diagnostics=False):
        """Separate diagnostic only;normal inherited score records no events."""
        if not self.handle:
            raise RuntimeError("Closed scorer")
        if self.dll.rbpf_stream_profile(self.handle, 1):
            raise RuntimeError("Diagnostic event creation failed")
        try:
            start = time.perf_counter_ns()
            values, details = self.score(hypotheses, problem, oracle, "gpu", diagnostics)
            wall_ns = time.perf_counter_ns() - start
            data = (ctypes.c_float * 3)()
            code = self.dll.rbpf_stream_profile_get(self.handle, data)
            if code not in (0, 1):
                raise RuntimeError("Diagnostic event query failed")
            event_ms = dict(zip(("upload", "kernel", "readback"), data, strict=True)) if code == 0 else None
            return values, details, dict(wall_ns=wall_ns, event_ms=event_ms,
                scope="CUDA event intervals include enqueue gaps/contention;whole Python score wall includes preparation/packing/fallback. Diagnostic only.")
        finally:
            if self.dll.rbpf_stream_profile(self.handle, 0):
                raise RuntimeError("Diagnostic event reset failed")
