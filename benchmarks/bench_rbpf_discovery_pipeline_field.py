"""Actual PF-return accuracy check for complete discovery and async scoring."""

import gc
import gzip
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_async_score import AsyncScorer
from rbpf_prior_pipeline import PriorPipelineStep


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    proxy = SimpleNamespace(**vars(field))
    controllers, scorers = [], []

    def selected_scorer(*args, **kwargs):
        scorer = AsyncScorer(*args, **kwargs)
        scorers.append(scorer)
        return scorer

    class RecordedStep(PriorPipelineStep):
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
                    notes={k: dict(self.last[k]) for k in ("move_plan", "discovery_pipeline", "prior_pipeline")} if valid else None,
                    GC_enabled=enabled, GC_threshold=threshold))

    def selected_field():
        original = field.main
        return FunctionType(original.__code__, dict(original.__globals__,
            BatchScorer=selected_scorer, MoveBatchStep=proxy.MoveBatchStep),
            original.__name__, original.__defaults__, original.__closure__)()

    proxy.main = selected_field
    original = base.main
    result = FunctionType(original.__code__, dict(original.__globals__,
        field=proxy, NativeFeedbackMassStep=RecordedStep), original.__name__,
        original.__defaults__, original.__closure__)()
    assert len(controllers) == len(scorers) == 1
    controller, scorer = controllers[0], scorers[0]
    assert len(controller.records) == 200 and sum(r["valid"] for r in controller.records) == 197
    assert scorer.counts == dict(begins=197, completions=197, drains=0, immediate=0, errors=0)
    assert scorer.active is None and scorer.handle is None and scorer.async_handle is None
    for row in controller.records:
        if not row["valid"]:
            continue
        plan, pipeline, prior = [row["notes"][k] for k in ("move_plan", "discovery_pipeline", "prior_pipeline")]
        assert plan["calls"] == plan["planned"] == pipeline["discoveries"] == 1 and plan["fallback"] == 0
        assert pipeline["complete_launches"] == pipeline["finishes"] == 1 and pipeline["drains"] == 0
        assert pipeline["records_after_begin"] == prior["prepared_records"] == plan["logical_registrations"]
        assert prior["prepared_particles"] == plan["particles"] == 64 and prior["literal_weight_calls"] == 0
    with gzip.open(output / "pipeline.jsonl.gz", "wt", encoding="utf8", newline="\n") as stream:
        for row in controller.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (output / "transport.json").write_text(json.dumps(dict(async_counts=scorer.counts,
        contexts=1, complete_batch_per_valid_epoch=1, no_split_or_late_batch=True,
        all_GPU_jobs_completed_and_ownership_closed=True, original_device_math_unchanged=True,
        GC_policy_unchanged=True, timing_uncalibrated=True,
        scope="Accuracy and schedule accounting only. Outstanding job/CPU records do not prove achieved overlap or speed."),
        indent=2) + "\n", encoding="utf8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
