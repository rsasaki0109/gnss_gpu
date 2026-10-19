"""Accuracy-only actual prefix and phase2 ambiguity-update input capture."""

import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import bench_rbpf_process_profile_field as field
from rbpf_ambiguity_initx import capture_input, capture_output, make_type
from rbpf_call_profile import ProcessMonitor

BASE_SHA256 = "bef18a2bbe1003ca6f3213ae5b022faf01d3c9f5883a5600f53c5953d3023bd0"


def build_main(factory):
    path = Path(field.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen coarse field helper changed")
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    before = deepcopy(node)
    names = next(
        n.value
        for n in node.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "names"
    )
    names.elts.extend(
        ast.Constant(n)
        for n in (
            "benchmarks/rbpf_ambiguity_initx.py",
            "benchmarks/bench_rbpf_ambiguity_initx_field.py",
        )
    )
    restored = deepcopy(node)
    old = next(
        n.value
        for n in restored.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "names"
    )
    del old.elts[-2:]
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        before, include_attributes=False
    )
    scope = dict(field.main.__globals__, ProcessMonitor=factory)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["main"]


class CaptureMonitor(ProcessMonitor):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.engines = []
        self.captured = 0

    def instrument_type(self, original):
        measured = super().instrument_type(original)
        optimized, engine = make_type(measured)
        self.engines.append(engine)
        update = original.update_ambiguities
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


def main():
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    path = output.with_suffix(".ambiguity.pkl.gz")
    assert not path.exists()
    with gzip.open(path, "wb") as stream:
        monitor = CaptureMonitor(stream)
        code = build_main(lambda: monitor)()
    assert monitor.captured == 200 and monitor.engines
    counts = [e.counts() for e in monitor.engines]
    assert sum(c["fast_calls"] for c in counts) > 0
    output.with_suffix(".ambiguity.json").write_text(
        json.dumps(
            dict(
                captured_phase2_updates=monitor.captured,
                initx_counts=counts,
                capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                original_update_source_retained=True,
                diagnostic_only=True,
                no_timing_acceptance=True,
                numeric_DLL_and_PF_unchanged=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
