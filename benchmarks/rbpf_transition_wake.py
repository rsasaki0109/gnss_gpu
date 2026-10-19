"""Optional one-shot device submission during the original FGO transition.

The owner thread alone may start, score, or close. The worker only sets its
current device and uses D288's private staging probe. Every original score
joins before packing or using that workspace. This is not a keepalive.
"""

import ctypes
from pathlib import Path
import threading
import time
from statistics import median

import numpy as np

from rbpf_first_api import FirstApiScorer, parse_probe
from rbpf_register_lu import RegisterLUScorer


class OneShot:
    """An owned operation with explicit error propagation and a mandatory drain."""

    def __init__(self, prepare, operation):
        self.prepare, self.operation = prepare, operation
        self.owner = threading.get_ident()
        self.thread = None
        self.error = None
        self.state = "idle"
        self.record = {}

    def require_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError("One-shot workspace must be used by its owner")

    def start(self):
        self.require_owner()
        if self.state != "idle":
            return False
        self.record["start_ns"] = time.perf_counter_ns()
        try:
            self.prepare()
        except BaseException as error:
            self.error, self.state = error, "failed"
            raise
        self.state = "running"

        def run():
            self.record["worker_start_ns"] = time.perf_counter_ns()
            self.record["worker_ident"] = threading.get_ident()
            self.record["worker_native_id"] = threading.get_native_id()
            try:
                self.record["result"] = self.operation()
            except BaseException as error:
                self.error = error
                self.record["error"] = type(error).__name__ + ":" + str(error)
            finally:
                self.record["worker_end_ns"] = time.perf_counter_ns()

        try:
            self.thread = threading.Thread(target=run, name="rbpf-transition-once", daemon=False)
            self.thread.start()
        except BaseException:
            self.thread = None
            self.state = "failed"
            raise
        self.record["start_return_ns"] = time.perf_counter_ns()
        return True

    def finish(self):
        self.require_owner()
        if self.thread is not None and self.state == "running":
            self.record["join_start_ns"] = time.perf_counter_ns()
            self.thread.join()
            self.record["join_end_ns"] = time.perf_counter_ns()
            self.state = "failed" if self.error is not None else "done"
        if self.error is not None:
            raise RuntimeError("Transition submission failed") from self.error


class TransitionWakeScorer(RegisterLUScorer):
    """Original score, preceded by a drain of at most one extra 1B operation.

No score/graph/particle/RNG statement is replaced. A shared CUDA runtime is
explicitly supplied; the worker selects the creating thread's device. No
other GPU user or bypass of this wrapper may use the owned workspace.
"""

    prepare_probe = FirstApiScorer.prepare_probe
    probe_raw = FirstApiScorer.probe_raw

    def __init__(self, dll, runtime, *, mode="none", capacity=512):
        if mode not in ("none", "overlap"):
            raise ValueError("Explicit none/overlap mode required")
        self.wake = None
        super().__init__(dll, capacity)
        self.owner = threading.get_ident()
        self.mode, self.transition_seen = mode, False
        self.score_records = []
        try:
            self.runtime = ctypes.CDLL(str(Path(runtime).resolve()))
            self.runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
            self.runtime.cudaGetDevice.restype = ctypes.c_int
            self.runtime.cudaSetDevice.argtypes = [ctypes.c_int]
            self.runtime.cudaSetDevice.restype = ctypes.c_int
            device = ctypes.c_int(-1)
            if self.runtime.cudaGetDevice(ctypes.byref(device)):
                raise RuntimeError("Cannot read workspace device")
            self.device = device.value
            self.dll.rbpf_first_api_prepare.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_first_api_prepare.restype = ctypes.c_int
            self.dll.rbpf_first_api_probe.argtypes = [ctypes.c_void_p, ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS"), ctypes.c_int]
            self.dll.rbpf_first_api_probe.restype = ctypes.c_int
            self.wake = OneShot(lambda: self.prepare_probe(1), self._operation)
        except BaseException:
            super().close()
            raise
        original = self.score

        def completed(*args, **kwargs):
            self.wake.require_owner()
            start = time.perf_counter_ns()
            self.wake.finish()
            drained = time.perf_counter_ns()
            result = original(*args, **kwargs)
            end = time.perf_counter_ns()
            self.score_records.append(dict(index=len(self.score_records), start_ns=start,
                drained_ns=drained, end_ns=end))
            return result

        self.score = completed

    def _operation(self):
        # CDLL releases the GIL. Select the original device on THIS thread.
        begin = time.perf_counter_ns()
        status = self.runtime.cudaSetDevice(self.device)
        selected = time.perf_counter_ns()
        if status:
            raise RuntimeError("Worker device selection failed:" + str(status))
        raw = self.probe_raw("h2d1")
        completed = time.perf_counter_ns()
        return dict(device=self.device, set_device_status=status, begin_ns=begin,
            selected_ns=selected, completed_ns=completed, probe=parse_probe(raw))

    def transition(self):
        self.wake.require_owner()
        if self.transition_seen:
            return False
        self.transition_seen = True
        if self.mode == "overlap":
            return self.wake.start()
        return False

    def close(self):
        if self.wake is not None:
            self.wake.require_owner()
        try:
            if self.wake is not None:
                self.wake.finish()
        finally:
            super().close()


def transition_type(original, begin):
    """Call the original transition once, after the explicit one-shot hook."""

    class Selected(original):
        def _transition_to_phase2(self, *args, **kwargs):
            begin()
            return super()._transition_to_phase2(*args, **kwargs)

    return Selected


ORDERS = (("A", "overlap", "AA"), ("AA", "A", "overlap"),
    ("overlap", "AA", "A"), ("AA", "overlap", "A"), ("A", "AA", "overlap"))


def evaluate(rows):
    """First transition-to-update latency screen; all-400 timing is descriptive.

    Five independent-process triples. Require A/A median absolute <5% and
    at least four absolute rounds <5%. Against BOTH controls require median
    reduction >=5% and at least four positive rounds. No completed old speed
    gate is changed, and no epoch is dropped from saved whole-prefix totals.
    """
    expected = [(r, label) for r, order in enumerate(ORDERS) for label in order]
    if len(rows) != 15 or [(v["trial"]["round"], v["trial"]["variant"]) for v in rows] != expected:
        raise ValueError("All15 new independent-process primary prefixes required")
    rounds = {r: {} for r in range(5)}
    for row in rows:
        if row["trial"]["accuracy"] or not row["verified"] or not row["all_final_and_trajectory_bits_exact"]:
            raise ValueError("Invalid primary prefix")
        records, bounds = row["records"], row["call_bounds_ns"]
        if len(records) != 400 or len(bounds) != 400 or [x["epoch"] for x in records] != list(range(400)):
            raise ValueError("Incomplete or reordered epochs")
        if [x["phase_before"] for x in records] != [1]*200+[2]*200 or not all(x["completed"] for x in records):
            raise ValueError("Incomplete original RTK body")
        if any(type(t) is not int or t < 0 for pair in bounds for t in pair):
            raise ValueError("Invalid absolute clock")
        if any(a > b or b-a < v["wall_ns"] for (a, b), v in zip(bounds, records, strict=True)):
            raise ValueError("RTK clock escaped its complete body")
        if any(a[1] > b[0] for a, b in zip(bounds, bounds[1:])):
            raise ValueError("RTK bodies overlap")
        first, last = row["transition"]["transition_epoch"], row["transition"]["first_score_epoch"]
        if (first, last) != (199, 203):
            raise ValueError("Unexpected transition/score boundary")
        latency = bounds[last][1]-bounds[first][0]
        nested = sum(v["wall_ns"] for v in records[first:last+1])
        if row["transition_through_first_score_wall_ns"] != latency or row["transition_through_first_score_RTK_ns"] != nested:
            raise ValueError("Transition interval does not conserve raw clocks")
        rounds[row["trial"]["round"]][row["trial"]["variant"]] = dict(
            latency=latency, nested=nested, all=sum(v["wall_ns"] for v in records),
            phase1=sum(v["wall_ns"] for v in records[:200]),
            phase2=sum(v["wall_ns"] for v in records[200:]), driver=row["driver_wall_ns"])
    metrics = {}
    for key in ("latency", "nested", "all", "phase1", "phase2", "driver"):
        contrasts = {}
        for label, left, right in (("A", "A", "overlap"), ("AA", "AA", "overlap"), ("control", "A", "AA")):
            values = [100.0*(v[left][key]-v[right][key])/v[left][key] for v in rounds.values()]
            contrasts[label] = dict(paired_percent=values, median_percent=median(values),
                positive_rounds=sum(v > 0 for v in values),
                absolute_median_percent=median(abs(v) for v in values),
                absolute_rounds_under5=sum(abs(v) < 5.0 for v in values))
        control = contrasts["control"]
        stable = control["absolute_median_percent"] < 5.0 and control["absolute_rounds_under5"] >= 4
        gain = all(contrasts[k]["median_percent"] >= 5.0 and contrasts[k]["positive_rounds"] >= 4 for k in ("A", "AA"))
        metrics[key] = dict(contrasts=contrasts, AA_valid=stable,
            pass_gate=stable and gain if key == "latency" else None)
    return dict(metrics=metrics, first_transition_latency_screen=metrics["latency"]["pass_gate"],
        continuous_RTK_throughput_established=False, all400_epochs_retained=True,
        original_failed_gates_unchanged=True)
