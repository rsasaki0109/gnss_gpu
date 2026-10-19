"""Actual D236 call profile and coarse RTK/GTSAM spans, with exact-state audit."""

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_call_profile import CallProfileStep, ProcessMonitor

ROOT = Path(__file__).resolve().parents[1]
FIELD_SHA256 = "b21c348b72f37c61007a8c0acb7bdcf4114fc2911e9199cbdc279c8336086209"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt_field(monitor, select, install):
    path = ROOT / "benchmarks/bench_rbpf_move_field.py"
    if sha(path) != FIELD_SHA256:
        raise ValueError("Frozen field runner changed")
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
    node.body[at + 1 : at + 1] = ast.parse("_d245_install(rbpf_fgo)").body
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
    assert isinstance(recorded.body[-1], ast.Return)
    span = ast.parse("with _d245_span('diagnostic_snapshot'): pass").body[0]
    span.body = recorded.body[start:-1]
    recorded.body[start:-1] = [span]
    restored = deepcopy(node)
    restored.body.pop(at + 1)
    old_recorded = next(
        n
        for n in ast.walk(restored)
        if isinstance(n, ast.FunctionDef) and n.name == "recorded"
    )
    old_recorded.body[start:-1] = old_recorded.body[start].body
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    scope = dict(field.main.__globals__)
    scope.update(MoveBatchStep=select, _d245_span=monitor.span, _d245_install=install)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope["main"], dict(
        field_source_sha256=FIELD_SHA256,
        restored_main_AST_equal=True,
        changes=[
            "Install diagnostic RTK subclass alias in this process",
            "Wrap original snapshot-only statements in separate span",
        ],
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("extension", "parent-extension", "mass-extension", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    own, remaining = parser.parse_known_args()
    monitor = ProcessMonitor()
    configs, controllers, installed = [], [], []
    names = [
        "benchmarks/rbpf_call_profile.py",
        "benchmarks/bench_rbpf_call_profile_field.py",
        "benchmarks/bench_rbpf_move_field.py",
        "benchmarks/rbpf_feedback_mass.py",
        "benchmarks/rbpf_component_mass.py",
        "benchmarks/rbpf_parent_score.py",
        "benchmarks/rbpf_native_canonical.py",
        "benchmarks/rbpf_move_key_memo.py",
        "benchmarks/rbpf_epoch_memo.py",
        "benchmarks/rbpf_move_reuse.py",
        "benchmarks/rbpf_move_batch.py",
        "experiments/rbpf_fgo/rbpf_fgo.py",
        "experiments/rbpf_fgo/gtsam_rtk_standalone.py",
    ]
    sources = {n: sha(ROOT / n) for n in names}
    binaries = {
        n: sha(getattr(own, n))
        for n in ("extension", "parent_extension", "mass_extension")
    }
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)

    def install(module):
        installed.append((module, module.RbpfGtsamRtkTc, module.RbpfFgo.__init__))
        module.RbpfGtsamRtkTc = monitor.instrument_type(module.RbpfGtsamRtkTc)

    def select(runtime, scorer, mode):
        assert runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())
        controller = CallProfileStep(
            runtime,
            scorer,
            mode,
            monitor=monitor,
            extension=own.extension,
            parent_extension=own.parent_extension,
            mass_extension=own.mass_extension,
        )
        controllers.append(controller)
        return controller

    run, proof = adapt_field(monitor, select, install)
    planpath = own.output_dir.with_suffix(".implementation.json")
    assert not planpath.exists()
    planpath.write_text(
        json.dumps(
            dict(
                source_sha256=sources,
                binary_sha256=binaries,
                explicit_configuration=dict(RBPF_CLUSTER_RELIN=str(own.cluster_relin)),
                field_AST_proof=proof,
                scope="Call-count/wall-profile only around original D236 PF. Coarse wall/process-CPU spans around actual RTK methods and selected GTSAM calls,diagnostic snapshot separate. Probe/scheduler overhead included;not a speed or whole-route gate.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    try:
        code = run()
    finally:
        for module, cls, initialize in reversed(installed):
            module.RbpfGtsamRtkTc, module.RbpfFgo.__init__ = cls, initialize
    assert installed and controllers
    assert all(sha(ROOT / n) == v for n, v in sources.items())
    assert all(sha(getattr(own, n)) == v for n, v in binaries.items())
    assert monitor.current is None
    (own.output_dir / "profiles.json").write_text(
        json.dumps(
            dict(
                records=monitor.records,
                method_AST_proofs=monitor.proofs,
                PF_call_profiles=[c.export() for c in controllers],
                clocks=dict(
                    spans="perf_counter_ns and process_time_ns",
                    call_profile="perf_counter_ns,timeunit1e-9",
                ),
                scope="Deterministic cProfile counts/times include profiling overhead;not statistical sampling or a speedup. Diagnostic snapshot span includes original state/deepcopy/encode/JSON/gzip/metadata.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    for index, controller in enumerate(controllers):
        controller.profile.dump_stats(str(own.output_dir / f"pf_calls_{index}.pstats"))
    (own.output_dir / "implementation_check.json").write_text(
        json.dumps(
            dict(
                configs=configs,
                source_unchanged=True,
                exit_code=code,
                aliases_restored=True,
                implementation_plan_sha256=sha(planpath),
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
