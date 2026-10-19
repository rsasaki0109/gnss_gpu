"""Read-only actual-route copy work audit; no timing or speed acceptance."""

import gc
import gzip
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_copy_work import CopyWork, inventory
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_register_lu import RegisterLUScorer


def _hypotheses(runtime):
    return ([(f"particle:{i}", p.H) for i, p in enumerate(runtime.particles)]
        + [(f"cluster:{i}", v["H"]) for i, v in enumerate(runtime.cluster_states.values())]
        + [(f"shadow:{i}", v["H"]) for i, v in enumerate(runtime.shadow_challengers)])


class ObservedCopyStep(NativeFeedbackMassStep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def __call__(self, rtk, ctx):
        if sys.getprofile() is not None:
            raise RuntimeError("An existing profiler must not be replaced")
        observer = CopyWork(self._source_globals["_copy_H"], {
            self._source_functions["_extend"].__code__: "extend",
            self._source_functions["_shrink_worst"].__code__: "shrink"})
        enabled, threshold = gc.isenabled(), gc.get_threshold()
        before = inventory(_hypotheses(self.runtime))
        completed = False
        try:
            with observer:
                result = super().__call__(rtk, ctx)
                completed = True
                return result
        finally:
            if observer.restored:
                assert gc.isenabled() == enabled and gc.get_threshold() == threshold
                self.records.append(dict(index=len(self.records), completed=completed,
                    valid=self.last is not None and "batch" in self.last,
                    sites=observer.records(), before=before, after=inventory(_hypotheses(self.runtime)),
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

    class RecordedStep(ObservedCopyStep):
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
    with gzip.open(output / "copy_work.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
        for row in controller.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "observation.json").write_text(json.dumps(dict(actual_score_calls=scorer.actual_calls,
        native_runtime_diagnostics_unmodified=True, extra_GPU_transfer=False,
        GC_policy_unchanged=True, numeric_functions_unchanged=True,
        profiler_restored=True, timings_measured=False,
        scope="Read-only copy and dictionary identity counts,not speed,bytes or a hard time bound."),
        indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
