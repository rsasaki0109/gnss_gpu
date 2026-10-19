"""First full PF screen for complete discovery, prior work and async scoring."""

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
from rbpf_async_score import AsyncScorer
from rbpf_discovery_pipeline_validation import check_pipeline_metadata
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import comparison, state
from rbpf_prior_pipeline import PriorPipelineStep
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original_cpu", "original_gpu", "blocked_gpu", "candidate_gpu", "cpu_AA", "gpu_AA")
SEEDS = tuple(range(20260710, 20260726))
CONTRASTS = (("gpu", "original_gpu", "candidate_gpu"), ("overlap", "blocked_gpu", "candidate_gpu"),
    ("gpu_vs_cpu", "original_cpu", "candidate_gpu"), ("cpu_AA", "original_cpu", "cpu_AA"),
    ("gpu_AA", "original_gpu", "gpu_AA"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call_order(ci, rd, group, seed, epoch):
    shift = (ci + rd * 3 + group + SEEDS.index(seed) + epoch) % len(MODES)
    order = MODES[shift:] + MODES[:shift]
    return order[::-1] if (ci + rd + group + epoch) % 2 else order


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
        minimum = 3 if name == "overlap" else 10
        contrasts[name] = dict(reference=original, candidate=candidate, group_paired_percent=group_values,
            round_paired_percent=rounds, median_paired_percent=median, improving_rounds=positive,
            absolute_round_medians=absolute, absolute_median_percent=statistics.median(absolute),
            absolute_rounds_under5=sum(x < 5 for x in absolute), minimum_percent=minimum,
            numeric_screen=median >= minimum and positive >= 4)
    caa, gaa = [contrasts[k]["absolute_median_percent"] < 5 and contrasts[k]["absolute_rounds_under5"] >= 4 for k in ("cpu_AA", "gpu_AA")]
    same_backend = gaa and contrasts["gpu"]["numeric_screen"] and contrasts["overlap"]["numeric_screen"]
    return dict(contrasts=contrasts, cpu_AA_valid=caa, gpu_AA_valid=gaa,
        GPU_same_backend_pass=same_backend, GPU_joint_pass=same_backend and caa and contrasts["gpu_vs_cpu"]["numeric_screen"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "async-dll", "canonical-extension", "parent-extension", "mass-extension", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert all(os.environ.get(k) == "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"))
    _pin()
    assert gc.isenabled() and not gc.callbacks
    threshold = gc.get_threshold()
    audit_path = ROOT / "dist/rbpf_discovery_pipeline_field_audit_20261018.json"
    audit = json.loads(audit_path.read_text(encoding="utf8"))
    assert audit["verified"] and audit["speed_stage_required"]
    prior = json.loads((ROOT / "dist/rbpf_discovery_pipeline_field_20261018/plan.json").read_text(encoding="utf8"))
    names = set(prior["source_sha256"]) | {"benchmarks/bench_rbpf_discovery_pipeline.py", "benchmarks/bench_pf_private_weights_compare.py"}
    assert all(sha(ROOT / n) == d for n, d in prior["source_sha256"].items())
    names.update(p.relative_to(ROOT).as_posix() for p in (args.dll, args.async_dll, args.canonical_extension, args.parent_extension, args.mass_extension))
    frozen = {n: sha(ROOT / n) for n in names}
    conditions = [dict(n=64, u=6, rows=64, groups=4, cluster=0), dict(n=64, u=6, rows=128, groups=8, cluster=1)]
    plan = dict(source_and_binary_sha256=frozen, modes=MODES, seeds=SEEDS, contrasts=CONTRASTS,
        conditions=conditions, rounds=5, groups=3, epochs_per_stream=4, scorer_contexts=3,
        controls="One old synced scorer shared by all original/AA modes;one new async scorer per scheduling arm. Blocked uses exactly the same new DLL/pinned copies/stream/packing/prior/records,waits in begin. Context setup outside;all input/output transfers and waits inside each PF timer.",
        field_accuracy_audit_sha256=sha(audit_path), GC_enabled=True, GC_threshold=threshold, GC_callbacks=False,
        acceptance="Per group sum16seeds x4continuous fullPF integer-ns calls. Signed median3groups then median5rounds:versusD236GPU>=10percent AND versusmatchedblocked>=3percent,each>=4positive rounds. Matching GPUAA:median absolute group differences per round,median5rounds<5percent AND>=4of5rounds<5percent. GPUjoint additionally candidateGPUversusD236CPU>=10percent/>=4positive and CPUAAvalid same absolute rule. No CPUcandidate claim. First measurement set only;retain every failure,no retime. If neither condition passes same-backend,stop performance path;otherwise new actual continuous RTK gate needed before adoption.",
        scope="NEW D282 scheduling implementation,not D279/D280 retiming. Prepare all6 inputs before any timer,serial complete PF calls in rotated/reversed order,snapshot/hash only after all6 timers. Particles/RNG/holds continuous4epochs;independent epoch problems. BLAS1/allowedCPU/normalpriority/nonexclusive;normalGC/noforcedcollection. Timer includes discovery/eligibility/fingerprints/packing/pinnedcopies/queuedGPU/allpriors/materialization/wait/fallback/weights/RNG/commit/feedback. Context setup/inputrestore/hash/serialization outside. Call-time sums under capture schedule,not actualRTKcontinuous throughput or GPU utilization. No speculative or split batches;each original prior operation still executed once;no probability reuse.")
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    warm_count = timed_count = 0
    results = {}
    with RegisterLUScorer(args.dll) as old, AsyncScorer(args.async_dll, blocked=True) as blocked, AsyncScorer(args.async_dll) as asynchronous, gzip.open(args.output.with_suffix(".calls.jsonl.gz"), "wt", encoding="utf8", newline="\n") as raw, gzip.open(args.output.with_suffix(".states.jsonl.gz"), "wt", encoding="utf8", newline="\n") as snapshots:
        for ci, condition in enumerate(conditions):
            fixtures = {}
            for mode in MODES:
                runtime, rtk, ctx = make_feedback_case(condition["n"], condition["u"], rows=condition["rows"], groups=condition["groups"], cluster=bool(condition["cluster"]), feedback=True, preload=True)
                initial = deepcopy(vars(runtime))
                backend = "native_cpu" if mode in ("original_cpu", "cpu_AA") else "gpu"
                cls = PriorPipelineStep if mode in ("blocked_gpu", "candidate_gpu") else NativeFeedbackMassStep
                scorer = {"blocked_gpu": blocked, "candidate_gpu": asynchronous}.get(mode, old)
                call = cls(runtime, scorer, backend, extension=args.canonical_extension,
                    parent_extension=args.parent_extension, mass_extension=args.mass_extension)
                fixtures[mode] = runtime, rtk, ctx, initial, call
            warm, timed = {}, []

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
                    order = call_order(ci, rd, group, seed, epoch)
                    outputs = {}
                    for mode in order:
                        runtime, rtk, contexts, call, before = active[mode]
                        start = time.perf_counter_ns()
                        out = call(rtk, contexts[epoch])
                        ns = time.perf_counter_ns() - start
                        outputs[mode] = out, ns
                    assert asynchronous.active is blocked.active is None
                    captured = {}
                    for mode in MODES:
                        runtime, rtk, contexts, call, before = active[mode]
                        out, ns = outputs[mode]
                        value, metadata = state(runtime, rtk, out), deepcopy(call.last)
                        captured[mode] = value, metadata, ns
                    for mode in order:
                        value, metadata, ns = captured[mode]
                        baseline = "original_cpu" if mode in ("original_cpu", "cpu_AA") else "original_gpu"
                        check = comparison(captured[baseline][0], value)
                        assert check["discrete_equal"] and check["numeric_bits_equal"] and check["max_abs_error"] == 0
                        if mode in ("blocked_gpu", "candidate_gpu"):
                            note = check_pipeline_metadata(captured[baseline][1], metadata)
                        else:
                            assert encode(captured[baseline][1]) == encode(metadata)
                            note = None
                        row = dict(phase=phase, condition=ci, round=rd, group=group, mode=mode, seed=seed,
                            epoch=epoch, order=order, ns=ns, state_sha256=digest(value), metadata_sha256=digest(metadata), notes=note)
                        key = mode, seed, epoch
                        if phase == "warm":
                            warm[key] = row
                            snapshots.write(json.dumps(dict(condition=ci, mode=mode, seed=seed, epoch=epoch,
                                state=encode(value), metadata=encode(metadata)), separators=(",", ":")) + "\n")
                        else:
                            reference = warm[key]
                            assert all(row[k] == reference[k] for k in ("state_sha256", "metadata_sha256", "notes"))
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
                print(json.dumps(dict(condition=ci, round_complete=rd)), flush=True)
            results[str(ci)] = dict(condition=condition, summary=summaries(timed))
            raw.flush()
            snapshots.flush()
            print(json.dumps(dict(condition_complete=ci, summary=results[str(ci)]["summary"])), flush=True)
        counts = dict(blocked=dict(blocked.counts), asynchronous=dict(asynchronous.counts))
    assert warm_count == 768 and timed_count == 11520
    assert all(sha(ROOT / n) == d for n, d in frozen.items())
    assert all(v == dict(begins=2048, completions=2048, drains=0, immediate=0, errors=0) for v in counts.values())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath),
        calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")), states_sha256=sha(args.output.with_suffix(".states.jsonl.gz")),
        warm_calls=warm_count, timed_calls=timed_count, results=results, async_counts=counts,
        all_same_backend_state_metadata_RNG_feedback_bits_equal=True, source_and_binaries_unchanged=True,
        no_completed_timing_retry=True, actual_whole_RTK_speed_established=False, default_changed=False), indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
