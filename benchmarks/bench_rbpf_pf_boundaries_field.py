"""Actual PF wall boundaries with unchanged D262 scoring and frozen accuracy refs."""

import gzip
import json
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_cluster_parameter_reuse import ClusterBatchScorer
from rbpf_pf_boundaries import BoundaryProfileStep, partition


def main():
    proxy = SimpleNamespace(**vars(field))
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    created = []
    controllers = []

    class SelectedScorer(ClusterBatchScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args, extension=os.environ["GNSS_GPU_RBPF_CLUSTER_BATCH_EXT"], **kwargs
            )
            self.numeric_score = self.score
            self.score = self._counted_score
            self.actual_calls = dict(gpu=0, native_cpu=0)
            created.append(self)

        def _counted_score(
            self, hypotheses, problem, oracle, mode="gpu", diagnostics=False
        ):
            self.actual_calls[mode] += 1
            values, detail = self.numeric_score(
                hypotheses, problem, oracle, mode, diagnostics=diagnostics
            )
            assert detail["scalar_fallbacks"] == 0
            return values, detail

    class RecordedStep(BoundaryProfileStep):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, trace=True, **kwargs)
            self.records = []
            controllers.append(self)

        def __call__(self, rtk, ctx):
            try:
                return super().__call__(rtk, ctx)
            finally:
                assert self.last_trace is not None and self._events is None
                self.records.append(
                    dict(
                        index=len(self.records),
                        trace=self.last_trace,
                        valid=self.last is not None and "batch" in self.last,
                    )
                )

    def selected_field():
        original = field.main
        run = FunctionType(
            original.__code__,
            dict(
                original.__globals__,
                BatchScorer=SelectedScorer,
                MoveBatchStep=proxy.MoveBatchStep,
            ),
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        return run()

    proxy.main = selected_field
    run = FunctionType(
        base.main.__code__,
        dict(base.main.__globals__, field=proxy, NativeFeedbackMassStep=RecordedStep),
        base.main.__name__,
        base.main.__defaults__,
        base.main.__closure__,
    )
    code = run()
    assert len(created) == len(controllers) == 1
    scorer, call = created[0], controllers[0]
    assert (
        scorer.actual_calls == dict(gpu=197, native_cpu=0) and len(call.records) == 200
    )
    assert sum(r["valid"] for r in call.records) == 197
    with gzip.open(
        output / "boundaries.jsonl.gz", "wt", encoding="utf8", newline="\n"
    ) as stream:
        for r in call.records:
            parts = partition(r["trace"])
            assert ("move_scores" in parts) == r["valid"]
            stream.write(json.dumps(r, separators=(",", ":")) + "\n")
    ci = int(os.environ["RBPF_CLUSTER_RELIN"])
    counts = scorer.cluster_engine.counts()
    assert counts["batch_calls"] == 197 * ci and counts["fallback_calls"] == 197 * (
        1 - ci
    )
    (output / "transport.json").write_text(
        json.dumps(
            dict(
                kernel="register",
                actual_score_calls=scorer.actual_calls,
                diagnostic_only=True,
                original_modules_not_mutated=True,
                accuracy_only_full_diagnostics=False,
                native_runtime_diagnostics_unmodified=True,
                cluster_batch=counts,
                old_engine_counters_are_logical_requests=True,
                captured_instance_candidate_score=True,
                boundary_records=200,
                sparse_direct_boundaries=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
