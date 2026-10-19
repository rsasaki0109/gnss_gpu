"""Actual accuracy/work check for the new ordered-H staging plan."""

import gc
import gzip
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_move_plan_checked import MovePlanStep
from rbpf_register_lu import RegisterLUScorer


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

    class RecordedStep(MovePlanStep):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.records = []
            controllers.append(self)

        def __call__(self, rtk, ctx):
            enabled, threshold = gc.isenabled(), gc.get_threshold()
            try:
                return super().__call__(rtk, ctx)
            finally:
                assert gc.isenabled() == enabled and gc.get_threshold() == threshold
                valid = self.last is not None and "batch" in self.last
                self.records.append(dict(index=len(self.records), valid=valid,
                    move_plan=dict(self.last["move_plan"]) if valid else None,
                    GC_enabled=enabled, GC_threshold=threshold))

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
    assert all(r["move_plan"]["calls"] == r["move_plan"]["planned"] == 1
        and r["move_plan"]["fallback"] == 0 for r in controller.records if r["valid"])
    with gzip.open(output / "move_plan.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
        for row in controller.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "transport.json").write_text(json.dumps(dict(actual_score_calls=scorer.actual_calls,
        extra_GPU_transfer=False, numeric_functions_unchanged=True, GC_policy_unchanged=True,
        timing_uncalibrated=True, scope="Accuracy/work only;three actual work diagnostic maps intentionally change."),
        indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
