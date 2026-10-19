"""Paired four-epoch PF streams with normal GC and no lifecycle observer."""

import argparse
from copy import deepcopy
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_rbpf_feedback_mass import _bind_fixture_callbacks, _rtk_digest
from bench_rbpf_irls_replay import digest
from bench_rbpf_move_field import encode
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import comparison, state
from rbpf_register_lu import RegisterLUScorer
from rbpf_view_release import ViewReleaseStep

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original_cpu", "candidate_cpu", "original_gpu", "candidate_gpu", "cpu_AA", "gpu_AA")
SEEDS = (20260710, 20260711, 20260712, 20260713)
CONTRASTS = (("cpu", "original_cpu", "candidate_cpu"), ("gpu", "original_gpu", "candidate_gpu"), ("gpu_vs_cpu", "candidate_cpu", "candidate_gpu"), ("cpu_AA", "original_cpu", "cpu_AA"), ("gpu_AA", "original_gpu", "gpu_AA"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(rows):
    times = {}
    for row in rows:
        key = row["round"], row["group"], row["mode"]
        times[key] = times.get(key, 0) + row["ns"]
    contrasts = {}
    for name, baseline, candidate in CONTRASTS:
        rounds = [statistics.median(100 * (1 - times[(r, g, candidate)] / times[(r, g, baseline)]) for g in range(3)) for r in range(5)]
        median, positive = statistics.median(rounds), sum(x > 0 for x in rounds)
        contrasts[name] = dict(reference=baseline, candidate=candidate, round_paired_percent=rounds, median_paired_percent=median, improving_rounds=positive, numeric_screen=median >= 10 and positive >= 4)
    caa = abs(contrasts["cpu_AA"]["median_paired_percent"]) < 5
    gaa = abs(contrasts["gpu_AA"]["median_paired_percent"]) < 5
    return dict(contrasts=contrasts, cpu_AA_valid=caa, gpu_AA_valid=gaa,
                CPU_pass=caa and contrasts["cpu"]["numeric_screen"],
                GPU_same_backend_pass=gaa and contrasts["gpu"]["numeric_screen"],
                GPU_joint_pass=caa and gaa and contrasts["gpu"]["numeric_screen"] and contrasts["gpu_vs_cpu"]["numeric_screen"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "canonical-extension", "parent-extension", "mass-extension", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))
    _pin()
    assert gc.isenabled() and not gc.callbacks
    threshold = gc.get_threshold()
    audit_path = ROOT / "dist/rbpf_view_release_field_audit_20261018.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    assert audit["verified"] and audit["speed_stage_required"]
    parent = json.loads((ROOT / "dist/rbpf_view_release_tests_20261018.plan.json").read_text(encoding="utf8"))
    names = set(parent["source_sha256"]) | {"benchmarks/bench_rbpf_view_release.py", "benchmarks/bench_rbpf_feedback_mass.py", "benchmarks/bench_rbpf_irls_replay.py"}
    names.update(path.relative_to(ROOT).as_posix() for path in (args.dll, args.canonical_extension, args.parent_extension, args.mass_extension))
    frozen = {name: sha(ROOT / name) for name in names}
    conditions = [dict(n=64, u=6, rows=64, groups=4, cluster=0), dict(n=64, u=6, rows=128, groups=8, cluster=1)]
    plan = dict(source_and_binary_sha256=frozen, modes=MODES, seeds=SEEDS, contrasts=CONTRASTS, conditions=conditions,
        rounds=5, groups=3, epochs_per_stream=4, scorer_contexts=1, field_work_audit_sha256=sha(audit_path),
        GC_enabled=True, GC_threshold=threshold, GC_callbacks=False,
        acceptance="Per group sum4seeds x4consecutive fullPF integer-ns calls. Median3groups then median5rounds;>=10percent and>=4positive rounds,matching AAabsmedian<5percent. GPUjoint also candidateGPU versus candidateCPU and bothAA. All conditions/failures retained,no completed timing retry.",
        scope="Two predeclared production-style particle/row/cluster regimes in four-epoch streams. All particles/candidates/FP64/RNG/feedback and complete PF tail,view release guards/counters/cleanup,score transfers/completion are timed. Fresh independent copied epoch problem each step,continuous particles/RNG/RTK holds within stream. Restore/RNG initialization/callback rebind/input epoch advance/deepcopy/hash/validation/serialization outside. Normal GC with no callback/forced collection/threshold/enable changes;one allowed CPU/BLAS1,normalpriority,nonexclusive. Extract/search/time/satellite/graph solve fixture boundaries. No real-RTK speed claim.")
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    timed_count = warm_count = 0
    results = {}
    with RegisterLUScorer(args.dll) as scorer, gzip.open(args.output.with_suffix(".calls.jsonl.gz"), "wt", encoding="utf8", newline="\n") as raw, gzip.open(args.output.with_suffix(".states.jsonl.gz"), "wt", encoding="utf8", newline="\n") as snapshots:
        for ci, condition in enumerate(conditions):
            fixtures = {}
            for mode in MODES:
                runtime, rtk, ctx = make_feedback_case(condition["n"], condition["u"], rows=condition["rows"], groups=condition["groups"], cluster=bool(condition["cluster"]), feedback=True, preload=True)
                initial = deepcopy(vars(runtime))
                backend = "gpu" if mode.endswith("gpu") or mode == "gpu_AA" else "native_cpu"
                cls = ViewReleaseStep if mode.startswith("candidate_") else NativeFeedbackMassStep
                call = cls(runtime, scorer, backend, extension=args.canonical_extension, parent_extension=args.parent_extension, mass_extension=args.mass_extension)
                fixtures[mode] = runtime, rtk, ctx, initial, call

            def stream(mode, seed):
                runtime, template, ctx, initial, call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.rng = np.random.default_rng(seed)
                runtime.step = call
                before = _rtk_digest(template)
                rtk = deepcopy(template)
                _bind_fixture_callbacks(rtk)
                contexts = [deepcopy(ctx) for _ in range(4)]
                records = []
                for epoch, context in enumerate(contexts):
                    rtk.epoch2 = epoch
                    context["obs"].t += epoch
                    start = time.perf_counter_ns()
                    out = call(rtk, context)
                    elapsed = time.perf_counter_ns() - start
                    value, metadata = state(runtime, rtk, out), deepcopy(call.last)
                    release = metadata.pop("view_release", None)
                    if mode.startswith("candidate_"):
                        assert release["calls"] == 1 and release["released_methods"] >= 3 and release["declined"] == 0
                    else:
                        assert release is None
                    records.append((epoch, elapsed, value, metadata, release))
                assert before == _rtk_digest(template)
                assert gc.isenabled() and gc.get_threshold() == threshold and not gc.callbacks
                return records

            references, warm, timed = {}, {}, []
            for mode in MODES:
                for seed in SEEDS:
                    backend = "gpu" if mode.endswith("gpu") or mode == "gpu_AA" else "cpu"
                    for epoch, ns, value, metadata, release in stream(mode, seed):
                        key = backend, seed, epoch
                        if mode.startswith("original_"):
                            references[key] = value, metadata
                        for a, b in zip(references[key], (value, metadata)):
                            check = comparison(a, b)
                            assert check["discrete_equal"] and check["numeric_bits_equal"] and check["max_abs_error"] == 0
                        row = dict(phase="warm", condition=ci, mode=mode, seed=seed, epoch=epoch, ns=ns, state_sha256=digest(value), metadata_sha256=digest(metadata), view_release=release)
                        warm[(mode, seed, epoch)] = row
                        raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                        snapshots.write(json.dumps(dict(condition=ci, mode=mode, seed=seed, epoch=epoch, state=encode(value), metadata=encode(metadata)), separators=(",", ":")) + "\n")
                        warm_count += 1
            for rd in range(5):
                for group in range(3):
                    shift = (ci + rd * 3 + group) % len(MODES)
                    order = MODES[shift:] + MODES[:shift]
                    if (ci + rd + group) % 2:
                        order = order[::-1]
                    shift = (rd + group) % len(SEEDS)
                    seeds = SEEDS[shift:] + SEEDS[:shift]
                    for mode in order:
                        for seed in seeds:
                            for epoch, ns, value, metadata, release in stream(mode, seed):
                                expected = warm[(mode, seed, epoch)]
                                assert digest(value) == expected["state_sha256"] and digest(metadata) == expected["metadata_sha256"] and release == expected["view_release"]
                                row = dict(phase="timed", condition=ci, round=rd, group=group, mode=mode, seed=seed, epoch=epoch, order=order, seed_order=seeds, ns=ns, state_sha256=digest(value), metadata_sha256=digest(metadata), view_release=release)
                                raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                                timed.append(row)
                                timed_count += 1
            results[str(ci)] = dict(condition=condition, summary=summaries(timed))
            raw.flush(); snapshots.flush()
            print(json.dumps(dict(condition_complete=ci, summary=results[str(ci)]["summary"])), flush=True)
    assert warm_count == 192 and timed_count == 2880
    assert all(sha(ROOT / name) == value for name, value in frozen.items())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath), calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")), states_sha256=sha(args.output.with_suffix(".states.jsonl.gz")), warm_calls=warm_count, timed_calls=timed_count, results=results, all_same_backend_state_metadata_RNG_graph_bits_equal=True, source_and_binaries_unchanged=True, no_completed_timing_retry=True, actual_whole_RTK_speed_established=False, default_changed=False), indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
