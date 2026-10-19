"""Complete-PF paired screen; retain every timing and exact same-backend state."""

import argparse
from copy import deepcopy
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
from rbpf_extend_support import NativeExtendSupportStep
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import comparison, state
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "original_cpu",
    "candidate_cpu",
    "original_gpu",
    "candidate_gpu",
    "cpu_AA",
    "gpu_AA",
)
SEEDS = (20260710, 20260711, 20260712, 20260713)
CONTRASTS = (
    ("cpu", "original_cpu", "candidate_cpu"),
    ("gpu", "original_gpu", "candidate_gpu"),
    ("gpu_vs_cpu", "candidate_cpu", "candidate_gpu"),
    ("cpu_AA", "original_cpu", "cpu_AA"),
    ("gpu_AA", "original_gpu", "gpu_AA"),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(records):
    durations = {}
    for row in records:
        k = row["round"], row["group"], row["mode"]
        durations[k] = durations.get(k, 0) + row["ns"]
    contrasts = {}
    for name, a, b in CONTRASTS:
        rounds = [
            statistics.median(
                100 * (1 - durations[(rd, g, b)] / durations[(rd, g, a)])
                for g in range(3)
            )
            for rd in range(5)
        ]
        median = statistics.median(rounds)
        positive = sum(x > 0 for x in rounds)
        contrasts[name] = dict(
            reference=a,
            candidate=b,
            round_paired_percent=rounds,
            median_paired_percent=median,
            improving_rounds=positive,
            numeric_screen=median >= 10 and positive >= 4,
        )
    caa = abs(contrasts["cpu_AA"]["median_paired_percent"]) < 5
    gaa = abs(contrasts["gpu_AA"]["median_paired_percent"]) < 5
    return dict(
        contrasts=contrasts,
        cpu_AA_valid=caa,
        gpu_AA_valid=gaa,
        CPU_pass=caa and contrasts["cpu"]["numeric_screen"],
        GPU_same_backend_pass=gaa and contrasts["gpu"]["numeric_screen"],
        GPU_joint_pass=caa
        and gaa
        and contrasts["gpu"]["numeric_screen"]
        and contrasts["gpu_vs_cpu"]["numeric_screen"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in (
        "dll",
        "extend-extension",
        "canonical-extension",
        "parent-extension",
        "mass-extension",
        "output",
    ):
        parser.add_argument("--" + n, type=Path, required=True)
    args = parser.parse_args()
    assert all(
        os.environ.get(k) == "1"
        for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    _pin()
    parent = json.loads(
        (ROOT / "dist/rbpf_extend_support_tests_20261017.plan.json").read_text(
            encoding="utf8"
        )
    )
    paths = set(parent["source_sha256"]) | {
        "benchmarks/bench_rbpf_extend_support.py",
        "benchmarks/bench_rbpf_feedback_mass.py",
        "benchmarks/bench_rbpf_irls_replay.py",
    }
    paths.update(
        p.relative_to(ROOT).as_posix()
        for p in (
            args.dll,
            args.extend_extension,
            args.canonical_extension,
            args.parent_extension,
            args.mass_extension,
        )
    )
    frozen = {n: sha(ROOT / n) for n in paths}
    conditions = [
        dict(
            n=n,
            u=u,
            rows=m,
            groups=g,
            cluster=int(m == 128),
            feedback=True,
            preload=True,
        )
        for n, u, m, g in (
            (64, 6, 64, 4),
            (64, 32, 64, 4),
            (64, 64, 64, 4),
            (128, 32, 64, 4),
            (64, 6, 128, 8),
            (128, 128, 128, 8),
        )
    ]
    conditions += [
        dict(n=64, u=6, rows=64, groups=4, cluster=0, feedback=False, preload=False),
        dict(n=64, u=6, rows=64, groups=4, cluster=1, feedback=True, preload=True),
    ]
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    plan = dict(
        source_and_binary_sha256=frozen,
        modes=MODES,
        contrasts=CONTRASTS,
        rounds=5,
        groups=3,
        seeds=SEEDS,
        conditions=conditions,
        scorer_contexts=1,
        acceptance="Sum four complete-PF integer-ns calls per group;paired median3 groups then median5 rounds,>=10percent and>=4positive rounds,matching AAabsmedian<5percent. GPUjoint also candidateGPU versus candidateCPU and bothAA. All failures retained,no completed timing retry.",
        scope="Complete D236 PF versus native extend support/assembly only. All guards,allocation,original solve/matmul/rint,all particles/candidates/priors/FP64/choice/RNG/conditional/FIX/feedback/resampling timed. One shared register-LU scorer for all six modes. Restore/RNG initialization/callback rebind/hash/validation/serialization outside. One allowed CPU/BLAS1/normalpriority/nonexclusive. Extract/search/time/satellite/graph solve are fixture boundaries. No actual whole-route/GTSAM speed claim.",
    )
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    all_results, warm_count, timed_count = {}, 0, 0
    with (
        RegisterLUScorer(args.dll) as scorer,
        gzip.open(
            args.output.with_suffix(".calls.jsonl.gz"),
            "wt",
            encoding="utf8",
            newline="\n",
        ) as raw,
        gzip.open(
            args.output.with_suffix(".states.jsonl.gz"),
            "wt",
            encoding="utf8",
            newline="\n",
        ) as snapshots,
    ):
        for ci, condition in enumerate(conditions):
            fixtures = {}
            for mode in MODES:
                runtime, rtk, ctx = make_feedback_case(
                    condition["n"],
                    condition["u"],
                    rows=condition["rows"],
                    groups=condition["groups"],
                    cluster=bool(condition["cluster"]),
                    feedback=condition["feedback"],
                    preload=condition["preload"],
                )
                initial = deepcopy(vars(runtime))
                backend = (
                    "gpu" if mode.endswith("gpu") or mode == "gpu_AA" else "native_cpu"
                )
                cls = (
                    NativeExtendSupportStep
                    if mode.startswith("candidate_")
                    else NativeFeedbackMassStep
                )
                options = (
                    dict(extend_extension=args.extend_extension)
                    if mode.startswith("candidate_")
                    else {}
                )
                call = cls(
                    runtime,
                    scorer,
                    backend,
                    extension=args.canonical_extension,
                    parent_extension=args.parent_extension,
                    mass_extension=args.mass_extension,
                    **options,
                )
                fixtures[mode] = runtime, rtk, ctx, initial, call

            def run(mode, seed):
                runtime, rtk, ctx, initial, call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.rng = np.random.default_rng(seed)
                runtime.step = call
                template = _rtk_digest(rtk)
                rr, cc = deepcopy(rtk), deepcopy(ctx)
                if condition["feedback"]:
                    _bind_fixture_callbacks(rr)
                start = time.perf_counter_ns()
                output = call(rr, cc)
                elapsed = time.perf_counter_ns() - start
                assert template == _rtk_digest(rtk)
                details = deepcopy(call.last)
                counts = details.pop("extend_support", None)
                if counts is not None:
                    assert counts["support_native"] == details["extend_calls"]
                    assert counts["support_fallback"] == counts["finish_fallback"] == 0
                return elapsed, state(runtime, rr, output), details, counts

            references, meta, warm, records = {}, {}, {}, []
            for mode in MODES:
                for seed in SEEDS:
                    ns, value, details, counts = run(mode, seed)
                    backend = (
                        "gpu"
                        if mode.endswith("gpu") or mode == "gpu_AA"
                        else "native_cpu"
                    )
                    k = backend, seed
                    if mode.startswith("original_"):
                        references[k], meta[k] = value, details
                    for a, b in ((references[k], value), (meta[k], details)):
                        c = comparison(a, b)
                        assert (
                            c["discrete_equal"]
                            and c["numeric_bits_equal"]
                            and c["max_abs_error"] == 0.0
                        )
                    row = dict(
                        phase="warm",
                        condition=ci,
                        mode=mode,
                        seed=seed,
                        ns=ns,
                        state_sha256=digest(value),
                        metadata_sha256=digest(details),
                        extend_support=counts,
                    )
                    warm[(mode, seed)] = row
                    raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                    snapshots.write(
                        json.dumps(
                            dict(
                                condition=ci,
                                mode=mode,
                                seed=seed,
                                state=encode(value),
                                metadata=encode(details),
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    warm_count += 1
            for rd in range(5):
                for g in range(3):
                    shift = (ci + rd * 3 + g) % len(MODES)
                    order = MODES[shift:] + MODES[:shift]
                    if (ci + rd + g) % 2:
                        order = order[::-1]
                    seed_shift = (rd + g) % len(SEEDS)
                    seed_order = SEEDS[seed_shift:] + SEEDS[:seed_shift]
                    for mode in order:
                        for seed in seed_order:
                            ns, value, details, counts = run(mode, seed)
                            expected = warm[(mode, seed)]
                            assert (
                                digest(value) == expected["state_sha256"]
                                and digest(details) == expected["metadata_sha256"]
                            )
                            assert counts == expected["extend_support"]
                            row = dict(
                                phase="timed",
                                condition=ci,
                                round=rd,
                                group=g,
                                mode=mode,
                                seed=seed,
                                order=order,
                                seed_order=seed_order,
                                ns=ns,
                                state_sha256=digest(value),
                                metadata_sha256=digest(details),
                                extend_support=counts,
                            )
                            raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                            records.append(row)
                            timed_count += 1
            all_results[str(ci)] = dict(condition=condition, summary=summaries(records))
            raw.flush()
            snapshots.flush()
            print(
                json.dumps(
                    dict(condition_complete=ci, summary=all_results[str(ci)]["summary"])
                ),
                flush=True,
            )
    assert warm_count == len(conditions) * len(MODES) * len(SEEDS) == 192
    assert timed_count == len(conditions) * len(MODES) * len(SEEDS) * 5 * 3 == 2880
    assert all(sha(ROOT / n) == d for n, d in frozen.items())
    args.output.write_text(
        json.dumps(
            dict(
                plan_sha256=sha(planpath),
                calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
                states_sha256=sha(args.output.with_suffix(".states.jsonl.gz")),
                warm_calls=warm_count,
                timed_calls=timed_count,
                results=all_results,
                all_same_backend_state_metadata_RNG_graph_bits_equal=True,
                source_and_binaries_unchanged=True,
                actual_whole_graph_or_route_speed_established=False,
                default_changed=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
