"""Continuous real RTK process screen; snapshots only after the full driver."""

import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace

from bench_pf_private_weights_compare import _pin
from bench_rbpf_ambiguity_initx_route import adapt
from bench_rbpf_move_field import encode
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import state
from rbpf_move_plan_checked import MovePlanStep
from rbpf_move_plan_route import RouteClocks, VARIANTS, retaining_type
from rbpf_register_lu import RegisterLUScorer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("extension", "parent-extension", "mass-extension", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--cluster-relin", type=int, choices=(0,), required=True)
    own, remaining = parser.parse_known_args()
    _pin()
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    monitor = RouteClocks()
    configs, installed, steps = [], [], []
    policy = gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)

    def install(module):
        original = module.RbpfGtsamRtkTc
        installed.append((module, original, module.RbpfFgo.__init__))
        module.RbpfGtsamRtkTc = monitor.instrument_type(original)

    def select(runtime, scorer, mode):
        assert mode == "gpu" and runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())
        planned = own.variant == "candidate"
        cls = MovePlanStep if own.variant in ("off", "candidate") else NativeFeedbackMassStep
        extra = dict(planned=planned) if cls is MovePlanStep else {}
        selected = retaining_type(cls)(runtime, scorer, mode, extension=own.extension,
            parent_extension=own.parent_extension, mass_extension=own.mass_extension, **extra)
        steps.append(selected)
        return selected

    original, proof = adapt(select, install)
    run = FunctionType(original.__code__, dict(original.__globals__, BatchScorer=RegisterLUScorer),
        original.__name__, original.__defaults__, original.__closure__)
    proof.update(unchanged_adapted_code_object=run.__code__ is original.__code__,
        scorer_binding="RegisterLUScorer(register)")
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    start = time.perf_counter_ns()
    try:
        code = run()
    finally:
        elapsed = time.perf_counter_ns() - start
        for module, cls, initialize in reversed(installed):
            module.RbpfGtsamRtkTc, module.RbpfFgo.__init__ = cls, initialize
    # Everything below is outside both process clocks and the driver span.
    assert code == 0 and len(steps) == 1 and not monitor.active
    assert policy == (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks))
    summary = monitor.totals()
    selected = steps[0]
    assert selected.route_calls == 200 and selected.route_last is not None
    rtk, output = selected.route_last
    boundary = SimpleNamespace(_committed=rtk._committed,
        _rbpf_released=getattr(rtk, "_rbpf_released", set()),
        nav=SimpleNamespace(x=rtk.nav.x, P=rtk.nav.P))
    snapshot = encode(state(selected.runtime, boundary, output))
    with gzip.open(own.output_dir / "final_state.json.gz", "wt", encoding="utf8", newline="\n") as stream:
        json.dump(snapshot, stream, separators=(",", ":"))
        stream.write("\n")
    work = selected.staging_engine.counts() if own.variant in ("off", "candidate") else None
    if work is not None:
        assert work["calls"] == 197
        assert work["planned"] == (197 if own.variant == "candidate" else 0)
        assert work["fallback"] == (0 if own.variant == "candidate" else 197)
    (own.output_dir / "performance.json").write_text(json.dumps(dict(variant=own.variant,
        records=monitor.records, summary=summary, driver_wall_ns=elapsed, configs=configs,
        caller_metadata_final=encode(selected.last), work=work, PF_calls=selected.route_calls,
        AST_proof=proof, aliases_restored=True, GC_policy_unchanged=True,
        diagnostic_snapshots_only_after_driver=True, epoch_barriers=False,
        scope="One continuous actual GPU/c0 400-epoch prefix. Same light per-process clocks and latest-return reference retention in all arms. No epoch snapshots/metadata encoding/barriers/GC observer. Driver span includes adapted main setup,input hashing,decoding,graph/PF,trajectory IO and teardown;excludes module imports,final snapshot/performance encoding,and parent audits. Process sums include complete original RTK body,exclude driver decoding/IO. Shared warm Numba cache across serial children,normal GC,nonexclusive host. No fullroute,multiseed,GPU-occupancy or CPU-vs-GPU claim."
    ), indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
