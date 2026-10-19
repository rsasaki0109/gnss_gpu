"""Actual PF accuracy/work audit; profiling durations are not speed evidence."""

import gc
import gzip
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace
import weakref

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_direct_view import DirectViewStep
from rbpf_native_canonical import _CanonicalView
from rbpf_register_lu import RegisterLUScorer


class ObservedDirectStep(DirectViewStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self._weak_methods = [], []
        self._lookups, self._cache_size = 0, None
        compiled = self.compiled

        def watched(view, rtk, ctx):
            cache = object.__getattribute__(view, "_bound")
            self._weak_methods.append(weakref.ref(cache["_pair_dd"]))
            try:
                return compiled(view, rtk, ctx)
            finally:
                self._cache_size = len(cache)

        self.compiled = watched

    def _profile(self, frame, event, arg):
        if event == "call" and frame.f_code is _CanonicalView.__getattr__.__code__:
            self._lookups += 1

    def __call__(self, rtk, ctx):
        if sys.getprofile() is not None:
            raise RuntimeError("An existing profiler must not be replaced")
        enabled, threshold = gc.isenabled(), gc.get_threshold()
        before = self.direct_engine.counts.copy()
        self._lookups, self._cache_size = 0, None
        callback = self._profile
        sys.setprofile(callback)
        try:
            return super().__call__(rtk, ctx)
        finally:
            assert sys.getprofile() is callback
            sys.setprofile(None)
            assert gc.isenabled() == enabled and gc.get_threshold() == threshold
            if self.last is not None:
                self.last.pop("direct_view", None)
            self.records.append(dict(index=len(self.records),
                valid=self.last is not None and "batch" in self.last,
                original_getattr_calls=self._lookups, cache_at_compiled_exit=self._cache_size,
                watched_method_live_after_return=bool(self._weak_methods and self._weak_methods[-1]() is not None),
                direct_view={k: v - before[k] for k, v in self.direct_engine.counts.items()},
                GC_enabled=enabled, GC_threshold=threshold, profiler_restored=True))


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
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

    class RecordedStep(ObservedDirectStep):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            controllers.append(self)

    def selected_field():
        original = field.main
        return FunctionType(original.__code__, dict(original.__globals__,
            BatchScorer=SelectedScorer, MoveBatchStep=proxy.MoveBatchStep),
            original.__name__, original.__defaults__, original.__closure__)()

    proxy.main = selected_field
    original = base.main
    result = FunctionType(original.__code__, dict(original.__globals__,
        field=proxy, NativeFeedbackMassStep=RecordedStep), original.__name__,
        original.__defaults__, original.__closure__)()
    assert len(controllers) == len(scorers) == 1
    controller, scorer = controllers[0], scorers[0]
    assert len(controller.records) == 200 and sum(r["valid"] for r in controller.records) == 197
    assert scorer.actual_calls == dict(gpu=197, native_cpu=0)
    assert all(r["original_getattr_calls"] == 3 and r["cache_at_compiled_exit"] == 0
        and not r["watched_method_live_after_return"] and r["direct_view"] ==
        dict(calls=1, promoted=1, declined=0, initial_cached_methods=3) for r in controller.records)
    with gzip.open(output / "lookup.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
        for row in controller.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "transport.json").write_text(json.dumps(dict(actual_score_calls=scorer.actual_calls,
        native_runtime_diagnostics_unmodified=True, extra_GPU_transfer=False,
        GC_policy_unchanged=True, original_numeric_function_objects_unchanged=True,
        profiler_restored=True, diagnostic_timing_uncalibrated=True), indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
