"""Actual same-backend accuracy evidence for the isolated extend experiment."""

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_extend_support import NativeExtendSupportStep
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
            values, detail = self.numeric_score(
                hypotheses, problem, oracle, mode, diagnostics=diagnostics
            )
            assert detail["scalar_fallbacks"] == 0
            return values, detail

    class RecordedStep(NativeExtendSupportStep):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                extend_extension=os.environ["GNSS_GPU_RBPF_EXTEND_EXTENSION"],
                **kwargs,
            )
            self.records = []
            controllers.append(self)

        def __call__(self, rtk, ctx):
            try:
                return super().__call__(rtk, ctx)
            finally:
                valid = self.last is not None and "batch" in self.last
                counts = self.last.pop("extend_support") if valid else None
                if valid:
                    assert counts["support_native"] == self.last["extend_calls"]
                    assert counts["support_fallback"] == counts["finish_fallback"] == 0
                self.records.append(
                    dict(index=len(self.records), valid=valid, counts=counts)
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
        output / "extend_support.jsonl.gz", "wt", encoding="utf8", newline="\n"
    ) as stream:
        for row in call.records:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    totals = Counter()
    for row in call.records:
        if row["valid"]:
            totals.update(row["counts"])
    assert totals["support_native"] > 0 and totals["finish_native"] > 0
    (output / "transport.json").write_text(
        json.dumps(
            dict(
                kernel="register",
                actual_score_calls=scorer.actual_calls,
                native_runtime_diagnostics_unmodified=True,
                accuracy_only_full_diagnostics=False,
                original_modules_not_mutated=True,
                captured_instance_candidate_score=True,
                extend_support=dict(totals),
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
