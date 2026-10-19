"""Paired tracing-overhead calibration, not an optimization speed gate."""

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from bench_rbpf_feedback_mass import _bind_fixture_callbacks, _rtk_digest
from bench_rbpf_move_field import encode
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_batch import BatchScorer
from rbpf_move_cases import comparison, state
from rbpf_phase_profile import PhaseProfileStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dll", "extension", "parent-extension", "mass-extension", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    names = (
        "benchmarks/bench_rbpf_phase_profile.py",
        "benchmarks/rbpf_phase_profile.py",
        "benchmarks/bench_rbpf_feedback_mass.py",
        "benchmarks/rbpf_feedback_cases.py",
        "benchmarks/rbpf_feedback_mass.py",
        "benchmarks/rbpf_component_mass.py",
        "benchmarks/rbpf_parent_score.py",
        "benchmarks/rbpf_native_canonical.py",
        "benchmarks/rbpf_move_key_memo.py",
        "benchmarks/rbpf_move_reuse.py",
        "benchmarks/rbpf_epoch_memo.py",
        "benchmarks/rbpf_move_batch.py",
        "benchmarks/rbpf_likelihood_batch.py",
        "benchmarks/rbpf_likelihood_cases.py",
        "benchmarks/rbpf_likelihood_runtime_adapter.py",
        "benchmarks/rbpf_move_cases.py",
        "benchmarks/bench_rbpf_move_field.py",
        "benchmarks/bench_pf_private_weights_compare.py",
        "experiments/rbpf_fgo/rbpf_fgo.py",
    )
    binaries = {
        n: sha(getattr(args, n))
        for n in ("dll", "extension", "parent_extension", "mass_extension")
    }
    plan = dict(
        source_sha256={n: sha(ROOT / n) for n in names},
        binary_sha256=binaries,
        modes=["reference", "profiled", "AA"],
        rounds=5,
        repeats=3,
        conditions=[
            dict(particles=n, initial_unique=u, rows=r, groups=g)
            for n, u, r, g in ((64, 6, 64, 4), (64, 32, 64, 4), (128, 128, 128, 8))
        ],
        scope="Entire synthetic D236 step versus coarse-profiled D236. Same math/state;calibrate tracer overhead only. Extraction/search inputs are fixtures,so their two extra actual wrappers are not calibrated here. Includes all active method wrappers/spans/metadata work;restore/callback binding/hash/validation outside timer. One allowed CPU/one BLAS thread,normal priority,nonexclusive host/GPU.",
        interpretation="Median paired slowdown within each round,then median5. A/A abs median<5percent and profiled overhead<=5percent are calibration flags,not speedup adoption. Preserve every raw sample and failed control;no retries.",
    )
    output = args.output
    planpath = output.with_suffix(".plan.json")
    snapshots = output.with_suffix(".states.jsonl.gz")
    assert not any(p.exists() for p in (output, planpath, snapshots))
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    _pin()
    results = {}
    with (
        threadpool_limits(limits=1, user_api="blas"),
        BatchScorer(args.dll) as scorer,
        gzip.open(snapshots, "wt", encoding="utf8", newline="\n") as stream,
    ):
        for c in plan["conditions"]:
            key = "/".join(f"{k}{v}" for k, v in c.items())
            fixtures = {}
            for mode in plan["modes"]:
                runtime, rtk, ctx = make_feedback_case(
                    c["particles"],
                    c["initial_unique"],
                    rows=c["rows"],
                    groups=c["groups"],
                    cluster=c["rows"] == 128,
                )
                cls = PhaseProfileStep if mode == "profiled" else NativeFeedbackMassStep
                call = cls(
                    runtime,
                    scorer,
                    "native_cpu",
                    extension=args.extension,
                    parent_extension=args.parent_extension,
                    mass_extension=args.mass_extension,
                )
                fixtures[mode] = (runtime, deepcopy(vars(runtime)), rtk, ctx, call)

            def run(mode):
                runtime, initial, rtk, ctx, call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.step = call
                rr, cc = deepcopy(rtk), deepcopy(ctx)
                _bind_fixture_callbacks(rr)
                before = _rtk_digest(rtk)
                start = time.perf_counter_ns()
                value = call(rr, cc)
                ms = (time.perf_counter_ns() - start) / 1e6
                assert _rtk_digest(rtk) == before
                return (
                    ms,
                    state(runtime, rr, value),
                    call.last,
                    (call.last_profile if mode == "profiled" else None),
                )

            warm = {}
            for mode in plan["modes"]:
                _, value, metadata, spans = run(mode)
                warm[mode] = dict(state=value, metadata=metadata, spans=spans)
                stream.write(
                    json.dumps(
                        dict(condition=key, mode=mode, state=encode(value)),
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            pairs = []
            for ri in range(5):
                for repeat in range(3):
                    offset = (ri * 3 + repeat) % 3
                    order = plan["modes"][offset:] + plan["modes"][:offset]
                    record = dict(round=ri, repeat=repeat, order=order, runs={})
                    for mode in order:
                        ms, value, metadata, spans = run(mode)
                        for a, b in (
                            (warm["reference"]["state"], value),
                            (warm["reference"]["metadata"], metadata),
                        ):
                            check = comparison(a, b)
                            assert (
                                check["discrete_equal"]
                                and check["numeric_bits_equal"]
                                and check["max_abs_error"] == 0.0
                            ), (key, mode, check)
                        digest = hashlib.sha256(
                            json.dumps(encode(value), separators=(",", ":")).encode()
                        ).hexdigest()
                        record["runs"][mode] = dict(
                            ms=ms, state_sha256=digest, spans=spans
                        )
                    pairs.append(record)
            summary = {}
            for mode in ("profiled", "AA"):
                vals = [
                    statistics.median(
                        100 * (p["runs"][mode]["ms"] / p["runs"]["reference"]["ms"] - 1)
                        for p in pairs
                        if p["round"] == ri
                    )
                    for ri in range(5)
                ]
                summary[mode] = dict(
                    p50_ms=statistics.median(p["runs"][mode]["ms"] for p in pairs),
                    round_slowdown_percent=vals,
                    paired_slowdown_percent=statistics.median(vals),
                )
            summary["reference_p50_ms"] = statistics.median(
                p["runs"]["reference"]["ms"] for p in pairs
            )
            results[key] = dict(condition=c, pairs=pairs, summary=summary)
            print(key, json.dumps(summary), flush=True)
    assert all(sha(ROOT / n) == v for n, v in plan["source_sha256"].items())
    assert all(sha(getattr(args, n)) == v for n, v in binaries.items())
    output.write_text(
        json.dumps(
            dict(
                plan=plan,
                plan_sha256=sha(planpath),
                conditions=results,
                snapshots_sha256=sha(snapshots),
                source_and_binaries_unchanged=True,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
