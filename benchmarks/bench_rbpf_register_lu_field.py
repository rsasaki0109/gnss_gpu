"""Accuracy-only D236 GPU prefix with register-row11 ordered LU and solves."""

import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import bench_rbpf_feedback_mass_field as base
import bench_rbpf_move_field as field
from rbpf_register_lu import RegisterLUScorer


def main():
    proxy = SimpleNamespace(**vars(field))
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    created = []

    class SelectedScorer(RegisterLUScorer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.actual_calls = dict(gpu=0, native_cpu=0)
            self.dimension_counts = {}
            created.append(self)

        def score(self, hypotheses, problem, oracle, mode="gpu", diagnostics=False):
            self.actual_calls[mode] += 1
            values, detail = super().score(hypotheses, problem, oracle, mode, diagnostics=True)
            assert detail["scalar_fallbacks"] == 0 and "diagnostic" in detail
            for dimension in detail["diagnostic"][:, 0]:
                key = str(int(dimension))
                self.dimension_counts[key] = self.dimension_counts.get(key, 0) + 1
            if not diagnostics:
                detail = {k: v for k, v in detail.items() if k not in ("diagnostic", "unique_hypotheses")}
            return values, detail

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
    assert created[0].kernel == "register" and created[0].dimension_counts["11"] > 0
    (output / "transport.json").write_text(json.dumps(dict(kernel="register",
        actual_score_calls=created[0].actual_calls, diagnostic_only=True,
        original_modules_not_mutated=True, accuracy_only_full_diagnostics=True,
        dimension_counts=created[0].dimension_counts),
        indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
