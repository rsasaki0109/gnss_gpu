"""New class-dispatch candidate, complete PF epochs interleaved before capture."""

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
from rbpf_direct_view import DirectViewStep
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import comparison, state
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original_cpu", "candidate_cpu", "original_gpu", "candidate_gpu", "cpu_AA", "gpu_AA")
SEEDS = (20260710, 20260711, 20260712, 20260713)
CONTRASTS = (("cpu", "original_cpu", "candidate_cpu"), ("gpu", "original_gpu", "candidate_gpu"),
    ("gpu_vs_cpu", "candidate_cpu", "candidate_gpu"), ("cpu_AA", "original_cpu", "cpu_AA"), ("gpu_AA", "original_gpu", "gpu_AA"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(rows):
    times = {}
    for row in rows:
        key = row["round"], row["group"], row["mode"]
        times[key] = times.get(key, 0) + row["ns"]
    contrasts = {}
    for name, original, candidate in CONTRASTS:
        group_values = [[100 * (1 - times[(r, g, candidate)] / times[(r, g, original)]) for g in range(3)] for r in range(5)]
        rounds = [statistics.median(v) for v in group_values]
        absolute = [statistics.median(abs(x) for x in v) for v in group_values]
        median, positive = statistics.median(rounds), sum(x > 0 for x in rounds)
        contrasts[name] = dict(reference=original, candidate=candidate, group_paired_percent=group_values,
            round_paired_percent=rounds, median_paired_percent=median, improving_rounds=positive,
            absolute_round_medians=absolute, absolute_median_percent=statistics.median(absolute),
            absolute_rounds_under5=sum(x < 5 for x in absolute), numeric_screen=median >= 10 and positive >= 4)
    caa, gaa = [contrasts[k]["absolute_median_percent"] < 5 and contrasts[k]["absolute_rounds_under5"] >= 4 for k in ("cpu_AA", "gpu_AA")]
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
    audit_path = ROOT / "dist/rbpf_direct_view_field_audit_20261018.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    assert audit["verified"] and audit["speed_stage_required"]
    prior = json.loads((ROOT / "dist/rbpf_direct_view_observer_tests_20261018.plan.json").read_text(encoding="utf8"))
    names = set(prior["source_sha256"]) | {"benchmarks/bench_rbpf_direct_view.py", "benchmarks/bench_rbpf_feedback_mass.py", "benchmarks/bench_rbpf_irls_replay.py"}
    assert all(sha(ROOT / n) == d for n, d in prior["source_sha256"].items())
    names.update(p.relative_to(ROOT).as_posix() for p in (args.dll, args.canonical_extension, args.parent_extension, args.mass_extension))
    frozen = {n: sha(ROOT / n) for n in names}
    conditions = [dict(n=64, u=6, rows=64, groups=4, cluster=0), dict(n=64, u=6, rows=128, groups=8, cluster=1)]
    plan = dict(source_and_binary_sha256=frozen, modes=MODES, seeds=SEEDS, contrasts=CONTRASTS,
        conditions=conditions, rounds=5, groups=3, epochs_per_stream=4, scorer_contexts=1,
        field_work_audit_sha256=sha(audit_path), GC_enabled=True, GC_threshold=threshold, GC_callbacks=False,
        acceptance="Per group sum4seeds x4consecutive fullPF integer-ns calls. Signed median3groups then median5rounds>=10percent and>=4positive rounds. Matching AA:median absolute group differences per round,median5rounds<5percent AND>=4of5rounds<5percent;no signed cancellation. GPUjoint also candidateGPU versus candidateCPU and bothAA. First sample set only;retain all failures,no retime.",
        scope="New D277 candidate only,no old failed candidate remeasurement. At each epoch prepare ALL6 independent stream inputs before any timer,run6 complete PF calls serially in epoch/seed/round/group rotated/reversed order,then snapshot/hash all6 after all timers. Particles/RNG/holds continuous for4epochs;fresh independent epoch problems. One shared synced scorer/BLAS1/allowedCPU/normalpriority/nonexclusive. No GC callback/forced collection/policy change. Full original PF tail/transfer/complete wait and new type/cache guards/counters included. Restore/prepare/copy/hash/check/serialization outside. Fixture extraction/search/time/satellite/graph boundaries. Call-time sums under a capture schedule,not real RTK/continuous throughput/GPU utilization.")
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    warm_count = timed_count = 0
    results = {}
    with RegisterLUScorer(args.dll) as scorer, gzip.open(args.output.with_suffix(".calls.jsonl.gz"), "wt", encoding="utf8", newline="\n") as raw, gzip.open(args.output.with_suffix(".states.jsonl.gz"), "wt", encoding="utf8", newline="\n") as snapshots:
        for ci, condition in enumerate(conditions):
            fixtures = {}
            for mode in MODES:
                runtime, rtk, ctx = make_feedback_case(condition["n"], condition["u"], rows=condition["rows"], groups=condition["groups"], cluster=bool(condition["cluster"]), feedback=True, preload=True)
                initial = deepcopy(vars(runtime))
                backend = "gpu" if mode.endswith("gpu") or mode == "gpu_AA" else "native_cpu"
                call = (DirectViewStep if mode.startswith("candidate_") else NativeFeedbackMassStep)(runtime, scorer, backend,
                    extension=args.canonical_extension, parent_extension=args.parent_extension, mass_extension=args.mass_extension)
                fixtures[mode] = runtime, rtk, ctx, initial, call
            warm = {}
            timed = []

            def stream(seed, phase, rd, group):
                active = {}
                for mode in MODES:
                    runtime, template, ctx, initial, call = fixtures[mode]
                    runtime.__dict__.clear()
                    runtime.__dict__.update(deepcopy(initial))
                    runtime.rng = np.random.default_rng(seed)
                    runtime.step = call
                    before = _rtk_digest(template)
                    rtk = deepcopy(template)
                    _bind_fixture_callbacks(rtk)
                    contexts = [deepcopy(ctx) for _ in range(4)]
                    active[mode] = runtime, rtk, contexts, call, before
                rows = []
                for epoch in range(4):
                    for runtime, rtk, contexts, call, before in active.values():
                        rtk.epoch2 = epoch
                        contexts[epoch]["obs"].t += epoch
                    shift = (ci + rd * 3 + group + SEEDS.index(seed) + epoch) % len(MODES)
                    order = MODES[shift:] + MODES[:shift]
                    if (ci + rd + group + epoch) % 2:
                        order = order[::-1]
                    outputs = {}
                    for mode in order:
                        runtime, rtk, contexts, call, before = active[mode]
                        start = time.perf_counter_ns()
                        out = call(rtk, contexts[epoch])
                        ns = time.perf_counter_ns() - start
                        outputs[mode] = out, ns
                    captured = {}
                    for mode in MODES:
                        runtime, rtk, contexts, call, before = active[mode]
                        out, ns = outputs[mode]
                        value, metadata = state(runtime, rtk, out), deepcopy(call.last)
                        extra = metadata.pop("direct_view", None)
                        if mode.startswith("candidate_"):
                            assert extra == dict(calls=1, promoted=1, declined=0, initial_cached_methods=3)
                        else:
                            assert extra is None
                        captured[mode] = value, metadata, extra, ns
                    for mode in order:
                        value, metadata, extra, ns = captured[mode]
                        baseline = "original_gpu" if mode.endswith("gpu") or mode == "gpu_AA" else "original_cpu"
                        for a, b in zip(captured[baseline][:2], (value, metadata)):
                            check = comparison(a, b)
                            assert check["discrete_equal"] and check["numeric_bits_equal"] and check["max_abs_error"] == 0
                        row = dict(phase=phase, condition=ci, round=rd, group=group, mode=mode, seed=seed,
                            epoch=epoch, order=order, ns=ns, state_sha256=digest(value), metadata_sha256=digest(metadata), direct_view=extra)
                        key = mode, seed, epoch
                        if phase == "warm":
                            warm[key] = row
                            snapshots.write(json.dumps(dict(condition=ci, mode=mode, seed=seed, epoch=epoch,
                                state=encode(value), metadata=encode(metadata)), separators=(",", ":")) + "\n")
                        else:
                            reference = warm[key]
                            assert all(row[k] == reference[k] for k in ("state_sha256", "metadata_sha256", "direct_view"))
                        raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                        rows.append(row)
                assert all(before == _rtk_digest(fixtures[mode][1]) for mode, (_, _, _, _, before) in active.items())
                assert gc.isenabled() and gc.get_threshold() == threshold and not gc.callbacks
                return rows

            for seed in SEEDS:
                warm_count += len(stream(seed, "warm", 0, 0))
            for rd in range(5):
                for group in range(3):
                    shift = (rd + group) % len(SEEDS)
                    seeds = SEEDS[shift:] + SEEDS[:shift]
                    for seed in seeds:
                        rows = stream(seed, "timed", rd, group)
                        timed.extend(rows)
                        timed_count += len(rows)
            results[str(ci)] = dict(condition=condition, summary=summaries(timed))
            raw.flush()
            snapshots.flush()
            print(json.dumps(dict(condition_complete=ci, summary=results[str(ci)]["summary"])), flush=True)
    assert warm_count == 192 and timed_count == 2880
    assert all(sha(ROOT / n) == d for n, d in frozen.items())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath),
        calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")), states_sha256=sha(args.output.with_suffix(".states.jsonl.gz")),
        warm_calls=warm_count, timed_calls=timed_count, results=results,
        all_same_backend_state_metadata_RNG_feedback_bits_equal=True, source_and_binaries_unchanged=True,
        no_completed_timing_retry=True, actual_whole_RTK_speed_established=False, default_changed=False), indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
