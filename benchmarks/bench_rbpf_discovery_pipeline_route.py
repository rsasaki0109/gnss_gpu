"""First continuous RTK screen of complete-discovery async scheduling."""

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
from rbpf_discovery_pipeline_route import make_scorer, make_step
from rbpf_move_cases import state
from rbpf_move_plan_route import RouteClocks, VARIANTS


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
    configs, installed, steps, scorers = [], [], [], []
    policy = gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks)

    def install(module):
        original = module.RbpfGtsamRtkTc
        installed.append((module, original, module.RbpfFgo.__init__))
        module.RbpfGtsamRtkTc = monitor.instrument_type(original)

    def select(runtime, scorer, mode):
        configs.append(vars(runtime.cfg).copy())
        selected = make_step(runtime, scorer, mode, own.variant, extension=own.extension,
            parent_extension=own.parent_extension, mass_extension=own.mass_extension)
        steps.append(selected)
        return selected

    def scorer_binding(dll):
        scorer = make_scorer(dll, own.variant)
        scorers.append(scorer)
        return scorer

    original, proof = adapt(select, install)
    run = FunctionType(original.__code__, dict(original.__globals__, BatchScorer=scorer_binding),
        original.__name__, original.__defaults__, original.__closure__)
    proof.update(unchanged_adapted_code_object=run.__code__ is original.__code__,
        scorer_binding="AsyncScorer exact type for off/candidate;RegisterLUScorer for reference/AA",
        off_is_matched_blocked_pipeline=True)
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    start = time.perf_counter_ns()
    try:
        code = run()
    finally:
        elapsed = time.perf_counter_ns() - start
        for module, cls, initialize in reversed(installed):
            module.RbpfGtsamRtkTc, module.RbpfFgo.__init__ = cls, initialize
    assert code == 0 and len(steps) == len(scorers) == 1 and not monitor.active
    assert policy == (gc.isenabled(), gc.get_threshold(), tuple(gc.callbacks))
    summary = monitor.totals()
    selected, scorer = steps[0], scorers[0]
    assert selected.route_calls == 200 and selected.route_last is not None
    rtk, output = selected.route_last
    boundary = SimpleNamespace(_committed=rtk._committed,
        _rbpf_released=getattr(rtk, "_rbpf_released", set()),
        nav=SimpleNamespace(x=rtk.nav.x, P=rtk.nav.P))
    snapshot = encode(state(selected.runtime, boundary, output))
    with gzip.open(own.output_dir / "final_state.json.gz", "wt", encoding="utf8", newline="\n") as stream:
        json.dump(snapshot, stream, separators=(",", ":"))
        stream.write("\n")
    work = None
    if own.variant in ("off", "candidate"):
        engine = selected.staging_engine
        work = dict(move_plan=engine.counts(), discovery_pipeline=dict(engine.pipeline), prior_pipeline=dict(engine.prior))
        assert work["move_plan"]["calls"] == work["move_plan"]["planned"] == 197
        assert work["move_plan"]["fallback"] == 0
        assert work["discovery_pipeline"]["complete_launches"] == work["discovery_pipeline"]["finishes"] == 197
        assert work["discovery_pipeline"]["drains"] == work["prior_pipeline"]["literal_weight_calls"] == 0
        assert work["discovery_pipeline"]["hypotheses_submitted"] == 4166
        assert work["discovery_pipeline"]["records_after_begin"] == work["prior_pipeline"]["prepared_records"] == 206522
        assert work["prior_pipeline"]["prepared_particles"] == 12608
        assert scorer.counts == dict(begins=197, completions=197, drains=0, immediate=0, errors=0)
        assert scorer.active is scorer.async_handle is scorer.handle is None
    (own.output_dir / "performance.json").write_text(json.dumps(dict(variant=own.variant,
        records=monitor.records, summary=summary, driver_wall_ns=elapsed, configs=configs,
        caller_metadata_final=encode(selected.last), work=work,
        async_counts=dict(scorer.counts) if work is not None else None, PF_calls=selected.route_calls,
        AST_proof=proof, aliases_restored=True, GC_policy_unchanged=True,
        diagnostic_snapshots_only_after_driver=True, epoch_barriers=False,
        scope="One continuous actual GPU/c0 400-epoch prefix. Same clocks/latest-return reference retention in all arms. No epoch snapshots/metadata encoding/barriers/GC observer. off is same complete-discovery/prior pipeline and async DLL/transport,waits in begin. Original/AA old synced DLL;device kernels identical by static proof. Driver includes adapted setup,input hashing,decoding,graph/PF,trajectory IO/teardown;excludes imports,post-driver snapshot/performance encoding. Process sums include entire original RTK body. Shared warm cache,normal GC,nonexclusive. No fullroute,multiseed,GPU-utilization or CPU-vs-GPU claim."
    ), indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
