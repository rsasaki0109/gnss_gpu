"""Instrumented A/A complete RTK epochs, with three clocks and no snapshots."""

import argparse
import json
import os
from pathlib import Path
import sys

from bench_pf_private_weights_compare import _pin
from bench_rbpf_ambiguity_initx_route import adapt
from rbpf_epoch_clocks import EpochClocks, summarize
from rbpf_feedback_mass import NativeFeedbackMassStep


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("extension", "parent-extension", "mass-extension", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    own, remaining = parser.parse_known_args()
    _pin()
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    monitor = EpochClocks()
    configs, installed = [], []

    def install(module):
        original = module.RbpfGtsamRtkTc
        installed.append((module, original, module.RbpfFgo.__init__))
        module.RbpfGtsamRtkTc = monitor.instrument_type(original)

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
    assert code == 0 and len(monitor.records) == 400 and not monitor.active
    assert sum(v["phase_before"] == 2 for v in monitor.records) == 200
    (own.output_dir / "clocks.json").write_text(
        json.dumps(
            dict(
                records=monitor.records,
                summary=summarize(monitor.records),
                configs=configs,
                AST_proof=proof,
                aliases_restored=True,
                diagnostic_state_serialization_in_process_timer=False,
                optimization_candidate=False,
                instrumentation_overhead_calibrated=False,
                scope="A/A three-clock diagnosis on frozen D236CPU. All400 epochs including startup. Wall minus CPU is not a scheduler-cause measurement. CPU ticks aggregate before division;no epoch filtering. No uninstrumented share, GPU occupancy or speedup claim.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
