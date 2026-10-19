"""Accuracy-only D236 GPU prefix with pinned CUDA Graph transport."""

import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_gpu_graph import GraphBatchScorer


def main():
    proxy = SimpleNamespace(**vars(field))
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    created = []

    class SelectedScorer(GraphBatchScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.actual_calls = dict(gpu=0, native_cpu=0)
            created.append(self)

        def close(self):
            if self.handle:
                self.final_submissions = self.stats()
            return super().close()

        def score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
            self.actual_calls[mode] += 1
            return super().score(hypotheses, problem, oracle, mode, diagnostics)

    def selected_field():
        original = field.main
        run = FunctionType(original.__code__, dict(original.__globals__, BatchScorer=SelectedScorer,
            MoveBatchStep=proxy.MoveBatchStep), original.__name__, original.__defaults__, original.__closure__)
        return run()

    proxy.main = selected_field
    run = FunctionType(base.main.__code__, dict(base.main.__globals__, field=proxy),
        base.main.__name__, base.main.__defaults__, base.main.__closure__)
    code = run()
    assert len(created) == 1 and created[0].actual_calls == dict(gpu=197, native_cpu=0)
    assert created[0].transport == "graph"
    assert created[0].final_submissions["graph_launches"] == 197
    assert created[0].final_submissions["stream_launches"] == 0
    (output / "transport.json").write_text(json.dumps(dict(transport="graph", submissions=created[0].final_submissions,
        actual_score_calls=created[0].actual_calls, diagnostic_only=True,
        original_modules_not_mutated=True, normal_score_has_no_diagnostic_events=True),
        indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
