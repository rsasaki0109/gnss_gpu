"""Actual accuracy evidence for per-request guarded staging access reuse."""

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_versioned_staging import VersionedStagingStep
from rbpf_register_lu import RegisterLUScorer


def main():
    proxy = SimpleNamespace(**vars(field))
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    created, controllers = [], []

    class SelectedScorer(RegisterLUScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.numeric_score = self.score
            self.score = self._counted_score
            self.actual_calls = dict(gpu=0, native_cpu=0)
            created.append(self)

        def _counted_score(
            self, hypotheses, problem, oracle, mode="gpu", diagnostics=False
        ):
            self.actual_calls[mode] += 1
            values, details = self.numeric_score(
                hypotheses, problem, oracle, mode, diagnostics=diagnostics
            )
            assert details["scalar_fallbacks"] == 0
            return values, details

    class RecordedStep(VersionedStagingStep):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                staging_extension=os.environ["GNSS_GPU_RBPF_VERSIONED_EXTENSION"],
                **kwargs,
            )
            self.records = []
            controllers.append(self)

        def __call__(self, rtk, ctx):
            try:
                return super().__call__(rtk, ctx)
            finally:
                valid = self.last is not None and "batch" in self.last
                counts = self.last.pop("versioned_staging") if valid else None
                registration = self.last.pop("native_staging") if valid else None
                if valid:
                    assert counts["enabled"] and counts["cached"]
                    assert (
                        counts["attribute_cache_reads"] == self.last["requested_moves"]
                    )
                    assert (
                        counts["fused_hits"] + counts["original_index_calls"]
                        == self.last["requested_moves"]
                    )
                    assert (
                        counts["invalidations"] == counts["cache_deoptimizations"] == 0
                    )
                    assert (
                        registration["native_calls"] == 1
                        and registration["fallback_calls"] == 0
                    )
                    assert registration["native_moves"] == self.last["requested_moves"]
                    assert registration["native_particles"] == self.last["particles"]
                self.records.append(
                    dict(
                        index=len(self.records),
                        valid=valid,
                        counts=counts,
                        registration=registration,
                    )
                )

    def selected_field():
        original = field.main
        return FunctionType(
            original.__code__,
            dict(
                original.__globals__,
                BatchScorer=SelectedScorer,
                MoveBatchStep=proxy.MoveBatchStep,
            ),
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )()

    proxy.main = selected_field
    original = base.main
    code = FunctionType(
        original.__code__,
        dict(original.__globals__, field=proxy, NativeFeedbackMassStep=RecordedStep),
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )()
    assert len(created) == len(controllers) == 1
    scorer, call = created[0], controllers[0]
    backend = "gpu" if scorer.actual_calls["gpu"] else "native_cpu"
    assert scorer.actual_calls == {
        k: 197 if k == backend else 0 for k in scorer.actual_calls
    }
    assert len(call.records) == 200 and sum(r["valid"] for r in call.records) == 197
    with gzip.open(
        output / "versioned_staging.jsonl.gz", "wt", encoding="utf8", newline="\n"
    ) as stream:
        for row in call.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    totals, registration = Counter(), Counter()
    peak = 0
    for row in call.records:
        if row["valid"]:
            totals.update(row["counts"])
            registration.update(row["registration"])
            peak = max(peak, row["counts"]["peak_entries"])
    assert totals["attribute_cache_reads"] > 0 and totals["identity_cache_hits"] > 0
    (output / "transport.json").write_text(
        json.dumps(
            dict(
                kernel="register",
                actual_score_calls=scorer.actual_calls,
                native_runtime_diagnostics_unmodified=True,
                accuracy_only_full_diagnostics=False,
                original_modules_not_mutated=True,
                captured_instance_candidate_score=True,
                versioned_staging=dict(totals),
                native_staging=dict(registration),
                peak_alias_entries=peak,
                extra_GPU_transfer=False,
                performance_gate_passed=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
