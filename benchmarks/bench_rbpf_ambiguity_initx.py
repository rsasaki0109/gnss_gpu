"""Controlled whole ambiguity-update replay of every captured phase2 input.

Only load captures produced locally by the field helper and verified by an
explicit SHA-256. Component speed is separate from complete RTK acceptance.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics
import time

from cssrlib.rtk import rtkpos
from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from rbpf_ambiguity_initx import exact_output, make_type, restore

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--capture-sha", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = list(zip(args.capture, args.capture_sha, strict=True))
    assert len(inputs) == 2
    assert all(sha(p) == digest for p, digest in inputs)
    names = [
        "benchmarks/rbpf_ambiguity_initx.py",
        "benchmarks/bench_rbpf_ambiguity_initx.py",
        "benchmarks/bench_pf_private_weights_compare.py",
    ]
    sources = {n: sha(ROOT / n) for n in names}
    external = {
        str(Path(f.__code__.co_filename)): sha(Path(f.__code__.co_filename))
        for f in (
            rtkpos.update_ambiguities,
            rtkpos.initx,
            rtkpos.IB,
            rtkpos._row_has_nonzero,
        )
    }
    plan = dict(
        source_sha256=sources,
        external_source_sha256=external,
        inputs=[dict(path=str(p), sha256=d) for p, d in inputs],
        modes=["reference", "bulk", "AA"],
        rounds=5,
        groups=3,
        cases_per_capture=200,
        all_phase2_inputs_included=True,
        scope="Whole original ambiguity update on every actual captured phase2 input,not initx alone or complete RTK. Include bulk-store guards/counters. Restore/copy/validation/hash/capture IO outside timer. One allowed CPU/BLAS1/normal priority/nonexclusive. No numerical/CUDA changes.",
        gate="Sum each mode duration over all200inputs per round;median5 paired reduction>=20percent,>=4of5positive rounds,AA absmedian<5percent. Component only;whole RTK speed acceptance requires a separate controlled comparison. All failed/noisy conditions retained,no retiming.",
    )
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    _pin()
    optimized, engine = make_type(rtkpos)
    results = {}
    timed, warm = 0, 0
    with threadpool_limits(limits=1, user_api="blas"):
        for index, (path, digest) in enumerate(inputs):
            cases = []
            with gzip.open(path, "rb") as stream:
                while True:
                    try:
                        value = pickle.load(stream)
                    except EOFError:
                        break
                    case_index = len(cases)

                    def run(mode):
                        cls = optimized if mode == "bulk" else rtkpos
                        rtk, obs = restore(cls, value["input"])
                        before = engine.counts()
                        start = time.perf_counter_ns()
                        result = rtk.update_ambiguities(obs)
                        duration = time.perf_counter_ns() - start
                        assert result is None
                        exact_output(rtk, value["output"])
                        for n in ("sat", "L", "P"):
                            assert (
                                getattr(obs, n).tobytes()
                                == value["input"]["obs"][n].tobytes()
                            )
                        counts = {n: v - before[n] for n, v in engine.counts().items()}
                        if mode == "bulk":
                            assert counts == value["initx_counts"], (
                                index,
                                case_index,
                                counts,
                                value["initx_counts"],
                            )
                        else:
                            assert all(v == 0 for v in counts.values())
                        return dict(
                            ns=duration,
                            initx_counts=counts,
                            all_mutated_array_bits_equal=True,
                        )

                    for mode in plan["modes"]:
                        run(mode)
                        warm += 1
                    samples = []
                    for ri in range(5):
                        for group in range(3):
                            offset = (case_index + ri * 3 + group) % 3
                            order = plan["modes"][offset:] + plan["modes"][:offset]
                            samples.append(
                                dict(
                                    round=ri,
                                    group=group,
                                    order=order,
                                    runs={mode: run(mode) for mode in order},
                                )
                            )
                            timed += 3
                    expected_sha = hashlib.sha256(
                        b"".join(
                            value["output"][n].tobytes()
                            for n in sorted(value["output"])
                        )
                    ).hexdigest()
                    cases.append(
                        dict(
                            epoch=value["epoch"],
                            expected_mutated_arrays_sha256=expected_sha,
                            initx_counts=value["initx_counts"],
                            samples=samples,
                        )
                    )
            assert len(cases) == 200
            totals = {
                mode: [
                    sum(
                        s["runs"][mode]["ns"]
                        for c in cases
                        for s in c["samples"]
                        if s["round"] == ri
                    )
                    for ri in range(5)
                ]
                for mode in plan["modes"]
            }
            summary = {}
            for mode in ("bulk", "AA"):
                reductions = [
                    100 * (1 - b / a)
                    for a, b in zip(totals["reference"], totals[mode], strict=True)
                ]
                summary[mode] = dict(
                    round_reduction_percent=reductions,
                    paired_reduction_percent=statistics.median(reductions),
                    improving_rounds=sum(v > 0 for v in reductions),
                )
            passed = (
                summary["bulk"]["paired_reduction_percent"] >= 20
                and summary["bulk"]["improving_rounds"] >= 4
                and abs(summary["AA"]["paired_reduction_percent"]) < 5
            )
            results[str(index)] = dict(
                capture_sha256=digest,
                cases=cases,
                round_total_ns=totals,
                summary=summary,
                component_gate_passed=passed,
                all_state_bits_equal=True,
            )
            print(
                json.dumps(
                    dict(capture=index, summary=summary, component_gate_passed=passed)
                ),
                flush=True,
            )
    assert timed == 18000 and warm == 1200
    assert all(sha(ROOT / n) == digest for n, digest in sources.items())
    assert all(sha(Path(p)) == digest for p, digest in external.items())
    assert all(sha(p) == digest for p, digest in inputs)
    args.output.write_text(
        json.dumps(
            dict(
                plan=plan,
                plan_sha256=sha(planpath),
                captures=results,
                timed_update_calls=timed,
                warm_update_calls=warm,
                all_mutated_array_bits_equal=True,
                sources_and_inputs_unchanged=True,
                component_gate_passes=sum(
                    v["component_gate_passed"] for v in results.values()
                ),
                whole_RTK_speed_gate=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
