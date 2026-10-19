"""Actual unchanged PF states plus sparse GC and method-lifetime diagnostics.

GC settings stay unchanged; recording overhead is uncalibrated. These wall/GC
durations are diagnostic and cannot establish a speedup or a time upper bound.
"""

import gc
import gzip
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace
import weakref

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_register_lu import RegisterLUScorer
from rbpf_view_release import ViewReleaseStep


_CLOCK = time.perf_counter_ns


class Observed:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self._weak_methods = [], []
        self._events = None
        self._gc_start = None
        self._cache_exit_size = None
        compiled = self.compiled
        def watched(view, rtk, ctx):
            self._weak_methods.append(weakref.ref(view._pair_dd))
            try:
                return compiled(view, rtk, ctx)
            finally:
                self._cache_exit_size = len(object.__getattribute__(view, "_bound"))
        self.compiled = watched

    def _gc_event(self, phase, info):
        if phase == "start":
            self._gc_start = _CLOCK()
        elif self._gc_start is not None:
            self._events.append(dict(ns=_CLOCK() - self._gc_start, generation=info["generation"], collected=info["collected"], uncollectable=info["uncollectable"]))
            self._gc_start = None

    def __call__(self, rtk, ctx):
        events = self._events = []
        self._cache_exit_size = None
        before = dict(getattr(self, "release_counts", {}))
        enabled, threshold = gc.isenabled(), gc.get_threshold()
        callback = self._gc_event
        start = _CLOCK()
        gc.callbacks.append(callback)
        try:
            return super().__call__(rtk, ctx)
        finally:
            assert gc.callbacks[-1] is callback
            gc.callbacks.pop()
            elapsed = _CLOCK() - start
            assert gc.isenabled() == enabled and gc.get_threshold() == threshold
            valid = self.last is not None and "batch" in self.last
            if self.last is not None:
                self.last.pop("view_release", None)
            delta = {k: v - before[k] for k, v in getattr(self, "release_counts", {}).items()}
            self.records.append(dict(index=len(self.records), valid=valid, profiled_ns=elapsed,
                GC_events=events, method_cache_at_compiled_exit=self._cache_exit_size,
                watched_method_live_after_step=bool(self._weak_methods and self._weak_methods[-1]() is not None),
                watched_methods_still_live=sum(ref() is not None for ref in self._weak_methods),
                view_release=delta, GC_enabled=enabled, GC_threshold=threshold))
            self._events = None


class BaselineObservedStep(Observed, NativeFeedbackMassStep):
    pass


class CandidateObservedStep(Observed, ViewReleaseStep):
    pass


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    release = os.environ["GNSS_GPU_RBPF_VIEW_RELEASE"] == "1"
    proxy = SimpleNamespace(**vars(field))
    controllers, scorers = [], []
    class SelectedScorer(RegisterLUScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            numeric = self.score
            self.actual_calls = dict(gpu=0, native_cpu=0)
            def counted(hypotheses, problem, oracle, mode="gpu", diagnostics=False):
                self.actual_calls[mode] += 1
                values, detail = numeric(hypotheses, problem, oracle, mode, diagnostics=diagnostics)
                assert detail["scalar_fallbacks"] == 0
                return values, detail
            self.score = counted
            scorers.append(self)
    selected = CandidateObservedStep if release else BaselineObservedStep
    class RecordedStep(selected):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            controllers.append(self)
    def selected_field():
        original = field.main
        return FunctionType(original.__code__, dict(original.__globals__, BatchScorer=SelectedScorer, MoveBatchStep=proxy.MoveBatchStep), original.__name__, original.__defaults__, original.__closure__)()
    proxy.main = selected_field
    original = base.main
    result = FunctionType(original.__code__, dict(original.__globals__, field=proxy, NativeFeedbackMassStep=RecordedStep), original.__name__, original.__defaults__, original.__closure__)()
    assert len(controllers) == len(scorers) == 1
    controller, scorer = controllers[0], scorers[0]
    assert len(controller.records) == 200 and sum(row["valid"] for row in controller.records) == 197
    assert scorer.actual_calls == dict(gpu=197, native_cpu=0)
    if release:
        assert all(row["method_cache_at_compiled_exit"] == 0 and not row["watched_method_live_after_step"] and row["view_release"]["calls"] == 1 and row["view_release"]["declined"] == 0 for row in controller.records)
    with gzip.open(output / "lifecycle.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
        for row in controller.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "transport.json").write_text(json.dumps(dict(actual_score_calls=scorer.actual_calls, native_runtime_diagnostics_unmodified=True, extra_GPU_transfer=False, release_enabled=release, GC_policy_unchanged=True, diagnostic_timing_uncalibrated=True), indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
