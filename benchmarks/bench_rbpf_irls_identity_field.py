"""Accuracy-only actual prefixes for the explicit IRLS identity DLL."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import bench_rbpf_move_field as field
from rbpf_irls_identity import IdentityBatchScorer, IRLSIdentityStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cluster-relin", type=int, choices=(0, 1), required=True)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--reference-dll", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--parent-extension", type=Path, required=True)
    parser.add_argument("--mass-extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    own, remaining = parser.parse_known_args()
    names = [
        "benchmarks/rbpf_irls_identity.py",
        "benchmarks/bench_rbpf_irls_identity_field.py",
        "benchmarks/rbpf_native_canonical.py",
        "benchmarks/rbpf_parent_score.py",
        "benchmarks/rbpf_component_mass.py",
        "benchmarks/rbpf_feedback_mass.py",
        "benchmarks/rbpf_move_key_memo.py",
        "benchmarks/rbpf_epoch_memo.py",
        "benchmarks/rbpf_move_reuse.py",
        "benchmarks/rbpf_move_batch.py",
        "benchmarks/rbpf_likelihood_batch.py",
        "benchmarks/bench_rbpf_move_field.py",
    ]
    for folder in (
        "rbpf_canonical",
        "rbpf_parent_score",
        "rbpf_component_mass",
        "rbpf_irls_identity",
    ):
        names += [
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / "benchmarks/probes" / folder).iterdir()
            if p.suffix in {".cpp", ".cu", ".txt"}
        ]
    sources = {n: sha(ROOT / n) for n in names}
    binaries = dict(
        prior_DLL=sha(own.reference_dll),
        candidate_DLL=sha(own.dll),
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
                scope="Accuracy-only D236 caller with explicit IRLS identity DLL/scorer. Candidate DLL differs;original same-backend outputs/state/cfg/metadata must match. No full-route or speed claim.",
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    os.environ["RBPF_CLUSTER_RELIN"] = str(own.cluster_relin)
    configs = []
    prior_step, prior_scorer = field.MoveBatchStep, field.BatchScorer

    def select(runtime, scorer, mode):
        assert runtime.cfg.cluster_relin == own.cluster_relin
        configs.append(vars(runtime.cfg).copy())
        return IRLSIdentityStep(
            runtime,
            scorer,
            mode,
            extension=own.extension,
            parent_extension=own.parent_extension,
            mass_extension=own.mass_extension,
        )

    # Explicit driver configuration only; no model/runtime method is patched.
    field.MoveBatchStep, field.BatchScorer = select, IdentityBatchScorer
    sys.argv = [
        sys.argv[0],
        *remaining,
        "--dll",
        str(own.dll),
        "--output-dir",
        str(own.output_dir),
    ]
    try:
        code = field.main()
    finally:
        field.MoveBatchStep, field.BatchScorer = prior_step, prior_scorer
    assert configs and all(c["cluster_relin"] == own.cluster_relin for c in configs)
    assert all(sha(ROOT / n) == v for n, v in sources.items())
    assert (
        sha(own.dll) == binaries["candidate_DLL"]
        and sha(own.reference_dll) == binaries["prior_DLL"]
    )
    assert (
        sha(own.extension) == binaries["canonical"]
        and sha(own.parent_extension) == binaries["parent"]
    )
    assert sha(own.mass_extension) == binaries["mass"]
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
