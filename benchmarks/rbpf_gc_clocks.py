"""Phase-aware GC/cycle diagnosis around unchanged serialized RTK calls."""

from contextlib import contextmanager
import ctypes
import gc
import os
import threading
import time

from rbpf_epoch_clocks import EpochClocks
from rbpf_epoch_lockstep import LockstepClocks


def thread_cycle_clock():
    """Scheduled calling-thread cycles, never converted to seconds."""
    if os.name != "nt":
        raise RuntimeError("The explicit cycle diagnostic requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentThread.restype = ctypes.c_void_p
    kernel.QueryThreadCycleTime.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
    kernel.QueryThreadCycleTime.restype = ctypes.c_int
    handle, value = kernel.GetCurrentThread(), ctypes.c_uint64()
    def read():
        if not kernel.QueryThreadCycleTime(handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value
    return read


class GCPhases:
    """Observe exclusive phases without changing GC enable/threshold/collection.

    Probe errors are retained rather than raised from a GC callback. Events
    that cross phases/threads remain explicit and are not silently attributed.
    """

    def __init__(self, clock=None):
        self.clock = time.perf_counter_ns if clock is None else clock
        self.current, self.epoch = "outside", -1
        self.owner = threading.get_ident()
        self.events, self.errors = [], []
        self._start = self._callback = self._policy = None

    @contextmanager
    def phase(self, name):
        previous = self.current
        self.current = name
        try:
            yield
        finally:
            self.current = previous

    def install(self):
        if self._callback is not None:
            raise RuntimeError("GC observer already installed")
        self._policy = gc.isenabled(), gc.get_threshold()
        self._callback = self.event
        gc.callbacks.append(self._callback)

    def close(self):
        callback = self._callback
        if callback is None:
            raise RuntimeError("GC observer is not installed")
        found = [i for i, value in enumerate(gc.callbacks) if value is callback]
        if len(found) != 1:
            raise RuntimeError("Owned GC callback was changed")
        gc.callbacks.pop(found[0])
        self._callback = None
        if self._start is not None:
            self.errors.append("unfinished GC event")
        if self._policy != (gc.isenabled(), gc.get_threshold()):
            raise RuntimeError("GC policy changed during observation")

    def event(self, phase, info):
        now, owner = self.clock(), threading.get_ident()
        if phase == "start":
            if self._start is not None:
                self.errors.append("overlapping GC start")
            self._start = now, self.current, self.epoch, owner, info.get("generation")
        elif phase == "stop":
            start, self._start = self._start, None
            if start is None:
                self.errors.append("GC stop without start")
                return
            ns = now - start[0]
            if ns < 0 or start[4] != info.get("generation"):
                self.errors.append("invalid GC time or generation")
            self.events.append(dict(epoch=start[2], phase=start[1], end_phase=self.current,
                generation=info.get("generation"), ns=ns,
                collected=info.get("collected", 0), uncollectable=info.get("uncollectable", 0),
                owner_thread=start[3] == self.owner and owner == self.owner,
                crossed_boundary=start[1:4] != (self.current, self.epoch, owner)))
        else:
            self.errors.append("unknown GC callback phase")


class BodyClocks(EpochClocks):
    """Add body cycles and a post-clock hook; original function runs once."""

    def __init__(self, *args, phases=None, cycles=None, after=None, observed=True, name="process", **kwargs):
        super().__init__(*args, **kwargs)
        self.phases = GCPhases() if phases is None else phases
        self.cycles = thread_cycle_clock() if observed and cycles is None else cycles
        self.after, self.observed, self.name = after, observed, name

    def call(self, phase, function, *args, **kwargs):
        count = len(self.records)
        ticks = None
        def measured(*values, **options):
            nonlocal ticks
            if not self.observed:
                return function(*values, **options)
            with self.phases.phase(self.name):
                start = self.cycles()
                try:
                    return function(*values, **options)
                finally:
                    ticks = self.cycles() - start
                    if ticks < 0:
                        raise ValueError("Thread cycle counter moved backwards")
        try:
            result = super().call(phase, measured, *args, **kwargs)
        finally:
            if len(self.records) > count:
                self.records[-1].update(thread_cycles=ticks, GC_observed=self.observed)
        if self.after is not None:
            self.after(self.records[-1])
        return result


class PhaseLockstepClocks(LockstepClocks, BodyClocks):
    """Reuse D271 barriers; post-clock capture finishes before emitting done."""

    def call(self, phase, function, *args, **kwargs):
        self.phases.epoch = len(self.records)
        with self.phases.phase("barrier"):
            return super().call(phase, function, *args, **kwargs)
