"""Accuracy-only D246 capture body with the combined update installed."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
from types import FunctionType

import bench_rbpf_ambiguity_initx_field as base
from rbpf_ambiguity_bookkeeping import make_type
from rbpf_ambiguity_initx import capture_input, capture_output
from rbpf_call_profile import ProcessMonitor


class CaptureMonitor(base.CaptureMonitor):
    def instrument_type(self, original):
        measured = ProcessMonitor.instrument_type(self, original)
        optimized, engine = make_type(measured)
        self.engines.append(engine)
        update = optimized.update_ambiguities
        monitor = self

        class Capture(optimized):
            def update_ambiguities(self, obs):
                if getattr(self, "phase", 1) != 2:
                    return update(self, obs)
                with monitor.span("diagnostic_ambiguity_capture"):
                    incoming = capture_input(self, obs)
                    before = engine.counts()
                with monitor.span("rtk.update_ambiguities"):
                    output = update(self, obs)
                with monitor.span("diagnostic_ambiguity_capture"):
                    record = dict(
                        epoch=int(self.epoch2),
                        input=incoming,
                        output=capture_output(self),
                        initx_counts={
                            n: v - before[n] for n, v in engine.counts().items()
                        },
                    )
                    pickle.dump(record, monitor.stream, protocol=5)
                    monitor.captured += 1
                return output

        return Capture


def proof():
    def method(path, classname):
        tree = ast.parse(path.read_bytes())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
        return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "instrument_type")

    original = method(Path(base.__file__), "CaptureMonitor")
    changed = deepcopy(method(Path(__file__), "CaptureMonitor"))
    assert ast.unparse(changed.body[0].value) == "ProcessMonitor.instrument_type(self, original)"
    assert ast.unparse(changed.body[3].value) == "optimized.update_ambiguities"
    changed.body[0].value = deepcopy(original.body[0].value)
    changed.body[3].value = deepcopy(original.body[3].value)
    assert ast.dump(original, include_attributes=False) == ast.dump(changed, include_attributes=False)
    return dict(capture_adapter_undo_equal=True,
        changed="Explicit original ProcessMonitor parent call and select combined update instead of literal update")


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    monitor_proof = proof()
    monitors = []

    class Selected(CaptureMonitor):
        def __init__(self, stream):
            super().__init__(stream)
            monitors.append(self)

    run = FunctionType(base.main.__code__, dict(vars(base), CaptureMonitor=Selected),
        base.main.__name__, base.main.__defaults__, base.main.__closure__)
    code = run()
    (output / "bookkeeping.json").write_text(json.dumps(dict(adapter_proof=monitor_proof,
        engines=[dict(counts=e.scan_counts(), AST_proof=e.proof) for m in monitors for e in m.engines],
        accuracy_only=True), indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
