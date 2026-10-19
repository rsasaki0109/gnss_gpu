"""GPU A/A clock calibration with post-process PF snapshots and GC phases."""

import argparse
import gc
import gzip
import json
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

from bench_pf_private_weights_compare import _pin
from bench_rbpf_ambiguity_initx_route import adapt
from bench_rbpf_move_field import encode
from rbpf_epoch_clocks import summarize
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_gc_clocks import BodyClocks, GCPhases, PhaseLockstepClocks
from rbpf_move_cases import state
from rbpf_register_lu import RegisterLUScorer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("extension", "parent-extension", "mass-extension", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cluster-relin", type=int, choices=(0,), required=True)
    own, remaining = parser.parse_known_args()
    _pin()
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    role = os.environ["GNSS_GPU_GC_OBSERVED"]
    assert role in ("0", "1")
    observed = role == "1"
    policy = gc.isenabled(), gc.get_threshold()
    phases = GCPhases()
    configs, installed, pending, metadata = [], [], [], []
    snapshots = own.output_dir / "post_states.jsonl.gz"
    stream = None
    captures = 0

    def after_process(sample):
        nonlocal stream, captures
        assert not monitor.active and len(pending) <= 1
        if not pending:
            return
        runtime, rtk, output, last = pending.pop()
        # The parent may postprocess holds after PF return. This is an explicit
        # post-process boundary; runtime/output/RNG are compared to PF-return
        # gold, while the RTK boundary is compared between the two children.
        with phases.phase("snapshot"):
            boundary = SimpleNamespace(
                _committed=rtk._committed,
                _rbpf_released=getattr(rtk, "_rbpf_released", set()),
                nav=SimpleNamespace(x=rtk.nav.x, P=rtk.nav.P),
            )
            snapshot = state(runtime, boundary, output)
        with phases.phase("serialize"):
            if stream is None:
                stream = gzip.open(snapshots, "wt", encoding="utf8", newline="\n")
            stream.write(json.dumps(encode(snapshot), separators=(",", ":")) + "\n")
            stream.flush()
            metadata.append(encode(last))
            captures += 1

    monitor = PhaseLockstepClocks(phases=phases, observed=observed, after=after_process)
    pf = BodyClocks(phases=phases, observed=True, name="pf") if observed else None

    def install(module):
        original = module.RbpfGtsamRtkTc
        installed.append((module, original, module.RbpfFgo.__init__))
        module.RbpfGtsamRtkTc = monitor.instrument_type(original)

    def select(runtime, scorer, mode):
        assert mode == "gpu" and runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())

        class SelectedStep(NativeFeedbackMassStep):
            def __call__(self, rtk, ctx):
                assert not pending
                function = super().__call__
                output = pf.call(2, function, rtk, ctx) if observed else function(rtk, ctx)
                # Retain references only; copying/encoding follows the RTK clock.
                pending.append((runtime, rtk, output, self.last))
                return output

        return SelectedStep(runtime, scorer, mode, extension=own.extension,
            parent_extension=own.parent_extension, mass_extension=own.mass_extension)

    original, proof = adapt(select, install)
    # Freeze the adapted numerical body and bind the already checked scorer.
    scope = dict(original.__globals__, BatchScorer=RegisterLUScorer)
    run = FunctionType(original.__code__, scope, original.__name__,
        original.__defaults__, original.__closure__)
    proof.update(unchanged_adapted_code_object=run.__code__ is original.__code__,
        scorer_binding="RegisterLUScorer(register)")
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    if observed:
        phases.install()
    try:
        code = run()
    finally:
        if observed:
            phases.close()
        if stream is not None:
            stream.close()
        for module, cls, initialize in reversed(installed):
            module.RbpfGtsamRtkTc, module.RbpfFgo.__init__ = cls, initialize
    assert code == 0 and len(monitor.records) == 400 and not monitor.active
    assert sum(v["phase_before"] == 2 for v in monitor.records) == 200
    assert captures == 200 and not pending and not phases.errors
    assert policy == (gc.isenabled(), gc.get_threshold())
    if observed:
        assert len(pf.records) == 200 and not pf.active
    (own.output_dir / "clocks.json").write_text(json.dumps(dict(
        observed=observed, records=monitor.records, summary=summarize(monitor.records),
        PF_records=[] if pf is None else pf.records, configs=configs,
        GC_events=phases.events, GC_errors=phases.errors, GC_policy_unchanged=True,
        post_process_captures=captures, caller_metadata=metadata, AST_proof=proof,
        aliases_restored=True, optimization_candidate=False,
        snapshots_and_protocol_outside_process_clocks=True,
        scope="Unchanged D236 GPU/c0 estimator. Independent epoch barriers,post-process snapshots. GC/cycles/nested PF clocks only when explicitly observed. Primary A/A has no GC observer/cycle/PF probe. CPU cycles are diagnostic counts,not seconds. Forced waits and capture alter GC/cache schedule;not continuous throughput or GPU occupancy."
    ), indent=2) + "\n", encoding="utf8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
