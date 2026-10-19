"""Calibrate sparse PF boundaries with paired 16-state complete-step cohorts."""

import argparse
from contextlib import ExitStack
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
from rbpf_cluster_parameter_reuse import ClusterBatchScorer
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import comparison, state
from rbpf_pf_boundaries import BoundaryProfileStep, partition

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("original", "clone", "profile", "original_AA", "profile_AA")
MODES = tuple(f"{k}_{b}" for b in ("cpu", "gpu") for k in KINDS)
SAMPLES = 16
CONTRASTS = (
    ("profile_vs_original", "original", "profile"),
    ("profile_vs_clone", "clone", "profile"),
    ("clone_control", "original", "clone"),
    ("original_AA", "original", "original_AA"),
    ("profile_AA", "profile", "profile_AA"),
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(records):
    totals = {
        (rd, g, m): sum(
            r["ns"]
            for r in records
            if (r["round"], r["group"], r["mode"]) == (rd, g, m)
        )
        for rd in range(5)
        for g in range(3)
        for m in MODES
    }
    result = {}
    for b in ("cpu", "gpu"):
        contrasts = {}
        for key, a, c in CONTRASTS:
            rounds = [
                statistics.median(
                    100 * (totals[rd, g, c + "_" + b] / totals[rd, g, a + "_" + b] - 1)
                    for g in range(3)
                )
                for rd in range(5)
            ]
            contrasts[key] = dict(
                reference=a + "_" + b,
                measured=c + "_" + b,
                round_overhead_percent=rounds,
                median_overhead_percent=statistics.median(rounds),
                within_five_percent=abs(statistics.median(rounds)) < 5,
            )
        result[b] = dict(
            contrasts=contrasts,
            calibrated=all(v["within_five_percent"] for v in contrasts.values()),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dll",
        "cluster-extension",
        "canonical-extension",
        "parent-extension",
        "mass-extension",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    assert all(
        os.environ.get(n) == "1"
        for n in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    )
    _pin()
    testplan = json.loads(
        (ROOT / "dist/rbpf_pf_boundaries_tests_20261017.plan.json").read_text(
            encoding="utf8"
        )
    )
    paths = set(testplan["source_sha256"]) | {"benchmarks/bench_rbpf_pf_boundaries.py"}
    paths.update(
        n.relative_to(ROOT).as_posix()
        for n in (
            args.dll,
            args.cluster_extension,
            args.canonical_extension,
            args.parent_extension,
            args.mass_extension,
        )
    )
    frozen = {n: sha(ROOT / n) for n in paths}
    conditions = [
        dict(n=64, u=6, rows=64, groups=4, cluster=0),
        dict(n=64, u=6, rows=64, groups=4, cluster=1),
        dict(n=128, u=32, rows=128, groups=8, cluster=1),
    ]
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    plan = dict(
        source_and_binary_sha256=frozen,
        modes=MODES,
        contrasts=CONTRASTS,
        conditions=conditions,
        samples=SAMPLES,
        rounds=5,
        groups=3,
        expected_timed_calls=7200,
        expected_warm_calls=480,
        RNG_seeds=list(range(20260710, 20260710 + SAMPLES)),
        calibration="Per condition/backend,paired sums of16 completePF step integer-ns in each group;median3groups per round then median5rounds. All5contrasts absmedian<5percent. Retain every negative/noisy round;no completed timing retry. Calibration only,no optimization adoption or D262 speed retest.",
        scope="Full D236 PF controller with identical D262 owning scorer for all modes. Sparse outer/inner/direct move boundaries;clone retains original inner code and wrapper bodies. Includes all guards/clock calls/allocations/trace publication/cleanup and unmodified numerics/transfers/waits/choice. Restore/deepcopy/RTK callback rebinding/hash/validation/serialization outside. One allowed CPU/BLAS1/normal priority/nonexclusive. Extract/search/time/satellite/graph solve are fixture boundaries. Same observations/config/particles per condition;16independent predeclared RNG initial states.",
    )
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    seeds = [
        deepcopy(np.random.default_rng(seed).bit_generator.state)
        for seed in plan["RNG_seeds"]
    ]
    results = {}
    warm_count = timed_count = 0
    with ExitStack() as stack:
        scorer = stack.enter_context(
            ClusterBatchScorer(args.dll, args.cluster_extension)
        )
        raw = stack.enter_context(
            gzip.open(
                args.output.with_suffix(".calls.jsonl.gz"),
                "wt",
                encoding="utf8",
                newline="\n",
            )
        )
        snapshots = stack.enter_context(
            gzip.open(
                args.output.with_suffix(".states.jsonl.gz"),
                "wt",
                encoding="utf8",
                newline="\n",
            )
        )
        for ci, c in enumerate(conditions):
            fixtures = {}
            for m in MODES:
                runtime, rtk, ctx = make_feedback_case(
                    c["n"],
                    c["u"],
                    rows=c["rows"],
                    groups=c["groups"],
                    cluster=bool(c["cluster"]),
                    shadow=True,
                    spawn=True,
                    feedback=True,
                    preload=True,
                    seed=20260710,
                )
                initial = deepcopy(vars(runtime))
                backend = "gpu" if m.endswith("_gpu") else "native_cpu"
                kwargs = dict(
                    extension=args.canonical_extension,
                    parent_extension=args.parent_extension,
                    mass_extension=args.mass_extension,
                )
                if m.startswith("original"):
                    call = NativeFeedbackMassStep(runtime, scorer, backend, **kwargs)
                else:
                    call = BoundaryProfileStep(
                        runtime,
                        scorer,
                        backend,
                        trace=m.startswith("profile"),
                        **kwargs,
                    )
                fixtures[m] = runtime, rtk, ctx, initial, call

            def run(m, s):
                runtime, rtk, ctx, initial, call = fixtures[m]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.rng.bit_generator.state = deepcopy(seeds[s])
                runtime.step = call
                template = _rtk_digest(rtk)
                rr, cc = deepcopy(rtk), deepcopy(ctx)
                _bind_fixture_callbacks(rr)
                before = scorer.cluster_engine.counts()
                start = time.perf_counter_ns()
                output = call(rr, cc)
                elapsed = time.perf_counter_ns() - start
                assert template == _rtk_digest(rtk)
                trace = call.last_trace if m.startswith("profile") else None
                if trace is not None:
                    partition(trace)
                    assert len(trace) == 21 and trace[-1][1] - trace[0][1] <= elapsed
                value, metadata = state(runtime, rr, output), deepcopy(call.last)
                stats = {
                    k: v - before[k] for k, v in scorer.cluster_engine.counts().items()
                }
                return (
                    dict(
                        ns=elapsed,
                        state_sha256=digest(value),
                        metadata_sha256=digest(metadata),
                        cluster_batch=stats,
                        trace=trace,
                    ),
                    value,
                    metadata,
                )

            references = {}
            warm = {}
            records = []
            for s in range(SAMPLES):
                for m in MODES:
                    data, value, metadata = run(m, s)
                    b = "gpu" if m.endswith("_gpu") else "cpu"
                    if m.startswith("original_") and not m.startswith("original_AA"):
                        references[s, b] = value, metadata, data["cluster_batch"]
                    a, meta, stats = references[s, b]
                    for x, y in ((a, value), (meta, metadata)):
                        check = comparison(x, y)
                        assert (
                            check["discrete_equal"]
                            and check["numeric_bits_equal"]
                            and check["max_abs_error"] == 0.0
                        )
                    assert stats == data["cluster_batch"]
                    row = dict(phase="warm", condition=ci, sample=s, mode=m, **data)
                    warm[s, m] = row
                    raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                    snapshots.write(
                        json.dumps(
                            dict(
                                condition=ci,
                                sample=s,
                                mode=m,
                                state=encode(value),
                                metadata=encode(metadata),
                            ),
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    warm_count += 1
            for rd in range(5):
                for g in range(3):
                    for s in range(SAMPLES):
                        shift = (ci + rd * 3 + g + s) % len(MODES)
                        order = MODES[shift:] + MODES[:shift]
                        if (ci + rd + g + s) % 2:
                            order = order[::-1]
                        for m in order:
                            data, _, _ = run(m, s)
                            for k in (
                                "state_sha256",
                                "metadata_sha256",
                                "cluster_batch",
                            ):
                                assert data[k] == warm[s, m][k]
                            row = dict(
                                phase="timed",
                                condition=ci,
                                sample=s,
                                round=rd,
                                group=g,
                                mode=m,
                                order=order,
                                **data,
                            )
                            raw.write(json.dumps(row, separators=(",", ":")) + "\n")
                            records.append(row)
                            timed_count += 1
                    raw.flush()
                print(
                    json.dumps(
                        dict(condition=ci, round_complete=rd, timed_calls=timed_count)
                    ),
                    flush=True,
                )
            results[str(ci)] = dict(condition=c, calibration=summarize(records))
        assert (
            warm_count == plan["expected_warm_calls"]
            and timed_count == plan["expected_timed_calls"]
        )
    assert all(sha(ROOT / n) == d for n, d in frozen.items())
    args.output.write_text(
        json.dumps(
            dict(
                plan_sha256=sha(planpath),
                calls_sha256=sha(args.output.with_suffix(".calls.jsonl.gz")),
                states_sha256=sha(args.output.with_suffix(".states.jsonl.gz")),
                warm_calls=warm_count,
                timed_calls=timed_count,
                conditions=results,
                all_complete_state_metadata_RNG_graph_bits_equal=True,
                sources_and_binaries_unchanged=True,
                numerical_optimization=False,
                default_changed=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
