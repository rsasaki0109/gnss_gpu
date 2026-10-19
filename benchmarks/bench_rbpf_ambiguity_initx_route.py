"""Controlled complete RTK epochs without diagnostic state serialization."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import bench_rbpf_move_field as field
from bench_pf_private_weights_compare import _pin
from rbpf_ambiguity_initx import make_type
from rbpf_feedback_mass import NativeFeedbackMassStep

ROOT = Path(__file__).resolve().parents[1]
FIELD_SHA256 = "b21c348b72f37c61007a8c0acb7bdcf4114fc2911e9199cbdc279c8336086209"


def adapt(select, install):
    path = Path(field.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FIELD_SHA256
    node = next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    baseline = deepcopy(node)
    at = next(
        i
        for i, n in enumerate(node.body)
        if isinstance(n, ast.Import) and any(a.name == "wp16_run_rbpf" for a in n.names)
    )
    node.body[at + 1 : at + 1] = ast.parse("_d246_install(rbpf_fgo)").body
    recorded = next(
        n
        for n in ast.walk(node)
        if isinstance(n, ast.FunctionDef) and n.name == "recorded"
    )
    start = next(
        i
        for i, n in enumerate(recorded.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "boundary"
    )
    removed = deepcopy(recorded.body[start:-1])
    del recorded.body[start:-1]
    restored = deepcopy(node)
    restored.body.pop(at + 1)
    old = next(
        n
        for n in ast.walk(restored)
        if isinstance(n, ast.FunctionDef) and n.name == "recorded"
    )
    old.body[-1:-1] = removed
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    scope = dict(field.main.__globals__, MoveBatchStep=select, _d246_install=install)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
            dont_inherit=True,
        ),
        scope,
    )
    return scope["main"], dict(
        restored_main_AST_equal=True,
        only_removed="Diagnostic boundary/state/deepcopy/encode/gzip/metadata statements after original selected PF call and PF timing;no estimator computation removed",
        source_sha256=FIELD_SHA256,
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for n in ("extension", "parent-extension", "mass-extension", "output-dir"):
        parser.add_argument("--" + n, type=Path, required=True)
    parser.add_argument("--variant", choices=("reference", "bulk", "AA"), required=True)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    own, remaining = parser.parse_known_args()
    _pin()
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    records, configs, installed, engines = [], [], [], []

    def install(module):
        original = module.RbpfGtsamRtkTc
        installed.append((module, original, module.RbpfFgo.__init__))
        chosen = original
        if own.variant == "bulk":
            chosen, engine = make_type(original)
            engines.append(engine)

        class Timed(chosen):
            def process(self, *args, **kwargs):
                phase = int(self.phase)
                start = time.perf_counter_ns()
                value = super().process(*args, **kwargs)
                duration = time.perf_counter_ns() - start
                records.append(dict(phase_before=phase, ns=duration))
                return value

        module.RbpfGtsamRtkTc = Timed

    def select(runtime, scorer, mode):
        assert mode == "native_cpu" and runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())
        return NativeFeedbackMassStep(
            runtime,
            scorer,
            mode,
            extension=own.extension,
            parent_extension=own.parent_extension,
            mass_extension=own.mass_extension,
        )

    run, proof = adapt(select, install)
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    try:
        code = run()
    finally:
        for module, cls, initialize in reversed(installed):
            module.RbpfGtsamRtkTc, module.RbpfFgo.__init__ = cls, initialize
    assert code == 0 and len(records) == 400
    assert sum(r["phase_before"] == 2 for r in records) == 200
    counts = [e.counts() for e in engines]
    if counts:
        assert counts[0]["fast_calls"] > 0 and counts[0]["fallback_calls"] == 0
    (own.output_dir / "performance.json").write_text(
        json.dumps(
            dict(
                variant=own.variant,
                records=records,
                configs=configs,
                initx_counts=counts,
                diagnostic_state_serialization_in_process_timer=False,
                AST_proof=proof,
                aliases_restored=True,
                scope="Full RTK process including original D236 PF,all400epochs. No diagnostic snapshots/metadata encoding in process timer. Trajectory accuracy checked externally;not per-step saved-state capture.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
