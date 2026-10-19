"""Controlled complete QC calls over every locally captured actual input."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time

from cssrlib.rtk import rtkpos

from bench_pf_private_weights_compare import _pin
from rbpf_qcedit_batch_field_schema import (
    GUARD_PROOF,
    capture_output,
    exact,
    make_type,
    restore,
)

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    h = hashlib.sha256()
    for name, array in value.items():
        h.update(name.encode())
        if isinstance(array, dict):
            h.update(bytes.fromhex(digest(array)))
        else:
            h.update(array.dtype.str.encode())
            h.update(repr(array.shape).encode())
            h.update(array.tobytes())
    return h.hexdigest()


def paired(cases, name):
    rounds = []
    totals = []
    for ri in range(5):
        groups = []
        for gi in range(3):
            sums = {
                m: sum(
                    s["runs"][m]["ns"]
                    for c in cases
                    for s in c["samples"]
                    if s["round"] == ri and s["group"] == gi
                )
                for m in ("reference", "batch", "AA")
            }
            groups.append(100 * (1 - sums[name] / sums["reference"]))
            totals.append(dict(round=ri, group=gi, ns=sums))
        rounds.append(statistics.median(groups))
    return dict(
        round_paired_percent=rounds,
        median_paired_percent=statistics.median(rounds),
        improving_rounds=sum(v > 0 for v in rounds),
        totals=totals,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--capture-sha", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = list(zip(args.capture, args.capture_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p, d in inputs)
    names = [
        "benchmarks/rbpf_qcedit_batch.py",
        "benchmarks/rbpf_qcedit_batch_list.py",
        "benchmarks/rbpf_qcedit_batch_field_schema.py",
        "benchmarks/bench_rbpf_qcedit_batch.py",
    ]
    sources = {n: sha(ROOT / n) for n in names}
    modes = ["reference", "batch", "AA"]
    plan = dict(
        source_sha256=sources,
        inputs=[dict(path=str(p), sha256=d) for p, d in inputs],
        modes=modes,
        rounds=5,
        groups=3,
        cases_per_capture=400,
        guard_AST_proof=GUARD_PROOF,
        gate="Complete QC:median5 of group3 paired reductions>=20percent,atleast4positive rounds,reference A/A absmedian<5percent. All400cases include both receivers and skipped/edited observations. No retries.",
        staged_scope="If both QC conditions pass,verify actual candidate full PF states/trajectory then screen cluster0 full RTK. Phase2 and all400epoch reductions each>=3percent,4of5positive rounds,each AA absmedian<5percent;three complete warm prefixes and5round3mode. Stop if fails;only a pass permits cluster1 screen. No default adoption from component alone.",
        timing="Restore/deepcopy/hash/exact state verification/IO outside. Full QC call with all guards,metadata preparation,allocations,compiled loop,and result included. One allowed CPU,BLAS1,normal priority,nonexclusive. Original FP64 helpers and satellite order;not full RTK speed.",
    )
    planpath = args.output.with_suffix(".plan.json")
    assert not planpath.exists() and not args.output.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    _pin()
    cls, engine = make_type(rtkpos)
    results, timed, warm = {}, 0, 0
    for ci, (path, corpus_sha) in enumerate(inputs):
        cases = []
        with gzip.open(path, "rb") as stream:
            while True:
                try:
                    value = pickle.load(stream)
                except EOFError:
                    break
                index = len(cases)
                expected_sha = digest(value["output"])

                def run(mode):
                    rtk, obs, kwargs = restore(
                        cls if mode == "batch" else rtkpos, value["input"]
                    )
                    before = engine.counts()
                    start = time.perf_counter_ns()
                    returned = rtk.qcedit(obs, **kwargs)
                    ns = time.perf_counter_ns() - start
                    output = capture_output(rtk, obs, returned)
                    exact(output, value["output"])
                    checksum = digest(output)
                    assert checksum == expected_sha
                    counts = {n: v - before[n] for n, v in engine.counts().items()}
                    assert counts == (
                        dict(calls=1, fast_calls=1, fallback_calls=0)
                        if mode == "batch"
                        else dict(calls=0, fast_calls=0, fallback_calls=0)
                    )
                    return dict(ns=ns, output_sha256=checksum, counts=counts)

                warm_results = {m: run(m) for m in modes}
                warm += 3
                samples = []
                for ri in range(5):
                    for gi in range(3):
                        offset = (index + ri + gi) % 3
                        order = modes[offset:] + modes[:offset]
                        samples.append(
                            dict(
                                round=ri,
                                group=gi,
                                order=order,
                                runs={m: run(m) for m in order},
                            )
                        )
                        timed += 3
                cases.append(
                    dict(
                        index=index,
                        epoch=value["epoch"],
                        expected_output_sha256=expected_sha,
                        warm=warm_results,
                        samples=samples,
                    )
                )
        assert len(cases) == 400
        contrast, aa = paired(cases, "batch"), paired(cases, "AA")
        passed = (
            contrast["median_paired_percent"] >= 20
            and contrast["improving_rounds"] >= 4
            and abs(aa["median_paired_percent"]) < 5
        )
        results[str(ci)] = dict(
            capture_sha256=corpus_sha,
            cases=cases,
            contrast=contrast,
            AA=aa,
            component_gate_passed=passed,
        )
        print(
            json.dumps(
                dict(
                    cluster=ci,
                    contrast=contrast["median_paired_percent"],
                    AA=aa["median_paired_percent"],
                    passed=passed,
                )
            ),
            flush=True,
        )
    assert timed == 36000 and warm == 2400
    assert all(sha(ROOT / n) == d for n, d in sources.items())
    assert all(sha(p) == d for p, d in inputs)
    result = dict(
        results=results,
        timed_calls=timed,
        warm_calls=warm,
        all_state_observation_bits_equal=True,
        source_and_inputs_unchanged=True,
        component_passes=sum(v["component_gate_passed"] for v in results.values()),
        plan_sha256=sha(planpath),
        no_completed_timing_retry=True,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
