"""Actual D236 phase footprint diagnostic,not a speedup measurement."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_native_canonical import NativeCanonicalStep
from rbpf_component_mass import NativeComponentMassStep
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_phase_profile import MovePhaseProfileStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mass-extension", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--parent-extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    own, remaining = parser.parse_known_args()
    names = [
        "benchmarks/rbpf_move_phase_profile.py",
        "benchmarks/rbpf_phase_profile.py",
        "benchmarks/bench_rbpf_move_phase_profile_field.py",
        "benchmarks/rbpf_parent_score.py",
        "benchmarks/rbpf_component_mass.py",
        "benchmarks/rbpf_feedback_mass.py",
        "benchmarks/rbpf_native_canonical.py",
        "benchmarks/bench_rbpf_move_field.py",
        "benchmarks/rbpf_move_key_memo.py",
        "benchmarks/rbpf_epoch_memo.py",
        "benchmarks/rbpf_move_reuse.py",
        "benchmarks/rbpf_move_batch.py",
    ]
    for folder in ("rbpf_canonical", "rbpf_parent_score", "rbpf_component_mass"):
        names += [
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "benchmarks/probes" / folder).iterdir()
            if p.suffix in {".cpp", ".txt"}
        ]
    sources = {name: sha(ROOT / name) for name in names}
    binaries = dict(
        canonical=sha(own.extension),
        parent=sha(own.parent_extension),
        mass=sha(own.mass_extension),
    )
    planpath = own.output_dir.with_suffix(".implementation.json")
    assert not planpath.exists()
    planpath.write_text(
        json.dumps(
            dict(
                source_sha256=sources,
                binary_sha256=binaries,
                explicit_configuration=dict(RBPF_CLUSTER_RELIN=str(own.cluster_relin)),
                scope="Fine move/scorer literal D236 wall spans for bottleneck diagnosis only; cluster_relin explicitly0or1 as in each frozen reference. Other original model settings unchanged. Not a speed or full-route gate.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    configs, controllers = [], []
    prior = field.MoveBatchStep

    def select(runtime, scorer, mode):
        assert runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())
        call = MovePhaseProfileStep(
            runtime,
            scorer,
            mode,
            extension=own.extension,
            parent_extension=own.parent_extension,
            mass_extension=own.mass_extension,
        )
        original = call.__call__
        profiles = []

        class Recorded:
            @property
            def last(self):
                return call.last

            def __call__(self, rtk, ctx):
                # inherited method guard expects runtime.step to be call
                current = runtime.step
                runtime.step = call
                try:
                    return original(rtk, ctx)
                finally:
                    runtime.step = current
                    profiles.append(call.last_profile)

        controllers.append(profiles)
        return Recorded()

    field.MoveBatchStep = select
    sys.argv = [sys.argv[0], *remaining, "--output-dir", str(own.output_dir)]
    try:
        code = field.main()
    finally:
        field.MoveBatchStep = prior
    assert configs and all(c["cluster_relin"] == own.cluster_relin for c in configs)
    assert all(sha(ROOT / name) == value for name, value in sources.items())
    assert (
        sha(own.extension) == binaries["canonical"]
        and sha(own.parent_extension) == binaries["parent"]
        and sha(own.mass_extension) == binaries["mass"]
    )
    (own.output_dir / "profiles.json").write_text(
        json.dumps(
            dict(
                profiles=controllers,
                scope="Coarse nested wall spans,includes tracing/scheduling overhead,not an optimization or speed gate.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    (own.output_dir / "implementation_check.json").write_text(
        json.dumps(
            dict(
                implementation_plan_sha256=sha(planpath),
                source_unchanged=True,
                exit_code=code,
                configs=configs,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
