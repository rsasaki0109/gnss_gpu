"""Actual prefix accuracy and complete QC input/output capture, without timing gates."""

import argparse
import ast
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

import bench_rbpf_process_profile_field as field
from rbpf_call_profile import ProcessMonitor
from rbpf_qcedit_batch import QcEngine, capture_input, capture_output, make_type

BASE_SHA256 = "bef18a2bbe1003ca6f3213ae5b022faf01d3c9f5883a5600f53c5953d3023bd0"


def build_main(factory):
    path = Path(field.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("Frozen field helper changed")
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
    extra = [
        "benchmarks/rbpf_qcedit_batch.py",
        "benchmarks/bench_rbpf_qcedit_batch_field.py",
    ]
    names.elts.extend(ast.Constant(n) for n in extra)
    restored = deepcopy(node)
    old = next(
        n.value
        for n in restored.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "names"
    )
    del old.elts[-len(extra) :]
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
    def __init__(self, stream, variant):
        super().__init__()
        self.stream, self.variant = stream, variant
        self.engines, self.captured = [], 0

    def instrument_type(self, original):
        if self.variant == "batch":
            chosen, engine = make_type(original)
        else:
            chosen, engine = original, QcEngine(original)
        self.engines.append(engine)
        monitor = self

        class Capture(chosen):
            def qcedit(self, obs, rs, dts, svh, rr=None):
                if getattr(self, "phase", 1) != 2:
                    return super().qcedit(obs, rs, dts, svh, rr)
                incoming = capture_input(self, obs, rs, dts, svh, rr)
                result = super().qcedit(obs, rs, dts, svh, rr)
                pickle.dump(
                    dict(
                        epoch=int(self.epoch2),
                        input=incoming,
                        output=capture_output(self, obs, result),
                    ),
                    monitor.stream,
                    protocol=5,
                )
                monitor.captured += 1
                return result

        return Capture


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=("reference", "batch"), required=True)
    own, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    output = Path(rest[rest.index("--output-dir") + 1])
    path = output.with_suffix(".qc.pkl.gz")
    assert not path.exists()
    with gzip.open(path, "wb") as stream:
        monitor = CaptureMonitor(stream, own.variant)
        code = build_main(lambda: monitor)()
    assert monitor.captured == 400 and monitor.engines and code == 0
    output.with_suffix(".qc.json").write_text(
        json.dumps(
            dict(
                variant=own.variant,
                captured_phase2_qc_calls=monitor.captured,
                counts=[e.counts() for e in monitor.engines],
                capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                diagnostic_only=True,
                no_timing_acceptance=True,
                field_main_only_extra_hashes=True,
                numerical_DLL_and_PF_unchanged=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
