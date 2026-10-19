"""Whole-update replay; distinguish bookkeeping gain from the earlier initx.

Load only locally generated captures with an explicit expected SHA-256.
All real phase2 inputs and every unfavorable/control sample are retained.
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
from rbpf_ambiguity_bookkeeping import make_type
from rbpf_ambiguity_initx_checked import make_type as make_initx_type
from rbpf_ambiguity_initx_checked import exact_output, restore

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(a, b):
    values = [100 * (1 - y / x) for x, y in zip(a, b, strict=True)]
    return dict(round_reduction_percent=values, paired_reduction_percent=statistics.median(values),
        improving_rounds=sum(v > 0 for v in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--capture-sha", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = list(zip(args.capture, args.capture_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p, d in inputs)
    names = ["benchmarks/rbpf_ambiguity_bookkeeping.py", "benchmarks/bench_rbpf_ambiguity_bookkeeping.py",
        "benchmarks/rbpf_ambiguity_initx_checked.py", "benchmarks/rbpf_ambiguity_initx.py",
        "benchmarks/rbpf_native_canonical.py", "benchmarks/bench_pf_private_weights_compare.py"]
    sources = {n: sha(ROOT / n) for n in names}
    external = {str(Path(f.__code__.co_filename)): sha(Path(f.__code__.co_filename))
        for f in (rtkpos.update_ambiguities, rtkpos.initx, rtkpos.IB, rtkpos._row_has_nonzero)}
    modes = ["reference", "initx", "combined", "AA", "initx_AA"]
    plan = dict(source_sha256=sources, external_source_sha256=external,
        inputs=[dict(path=str(p), sha256=d) for p, d in inputs], modes=modes, rounds=5, groups=3,
        cases_per_capture=200, all_phase2_inputs_included=True,
        scope="Whole ambiguity update,guards/counters included. Original,earlier checked initx,combined candidate,two A/A controls. Restore/copy/IO/checks outside timer. One allowed CPU/BLAS1/normal priority/nonexclusive. D246 remains unadopted for whole RTK.",
        gate="Per config: sum all200inputs per mode per round,5round medians. Combined reduction vs original>=30percent and vs checked initx>=20percent,each>=4of5positive;both A/A absmedians<5percent. Exact whole state/counters required. Separate whole RTK acceptance needed. No retries.")
    planpath = args.output.with_suffix(".plan.json")
    assert not args.output.exists() and not planpath.exists()
    planpath.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf8")
    _pin()
    combined, engine = make_type(rtkpos)
    initx, old_engine = make_initx_type(rtkpos)
    classes = dict(reference=rtkpos, initx=initx, combined=combined, AA=rtkpos, initx_AA=initx)
    results, timed, warm = {}, 0, 0
    with threadpool_limits(limits=1, user_api="blas"):
        for index, (path, digest) in enumerate(inputs):
            cases = []
            with gzip.open(path, "rb") as stream:
                while True:
                    try:
                        value = pickle.load(stream)
                    except EOFError:
                        break
                    ci = len(cases)

                    def run(mode):
                        rtk, obs = restore(classes[mode], value["input"])
                        selected = engine if mode == "combined" else old_engine
                        before = selected.counts()
                        scans_before = engine.scan_counts()
                        start = time.perf_counter_ns()
                        result = rtk.update_ambiguities(obs)
                        duration = time.perf_counter_ns() - start
                        assert result is None
                        exact_output(rtk, value["output"])
                        for n in ("sat", "L", "P"):
                            assert getattr(obs, n).tobytes() == value["input"]["obs"][n].tobytes()
                        counts = {n: v - before[n] for n, v in selected.counts().items()}
                        if mode in ("initx", "combined", "initx_AA"):
                            assert counts == value["initx_counts"]
                        else:
                            assert not any(counts.values())
                        scans = {n: v - scans_before[n] for n, v in engine.scan_counts().items()}
                        if mode == "combined":
                            nf = value["input"]["nav"]["nf"]
                            assert scans == dict(updates=1, original_updates=0, scans=nf,
                                fast_scans=nf, literal_scans=0)
                        else:
                            assert not any(scans.values())
                        return dict(ns=duration, initx_counts=counts, scan_counts=scans)

                    for mode in modes:
                        run(mode)
                        warm += 1
                    samples = []
                    for ri in range(5):
                        for group in range(3):
                            offset = (ci + ri * 3 + group) % len(modes)
                            order = modes[offset:] + modes[:offset]
                            samples.append(dict(round=ri, group=group, order=order,
                                runs={mode: run(mode) for mode in order}))
                            timed += len(modes)
                    cases.append(dict(epoch=value["epoch"], initx_counts=value["initx_counts"],
                        expected_mutated_arrays_sha256=hashlib.sha256(b"".join(value["output"][n].tobytes()
                            for n in sorted(value["output"]))).hexdigest(), samples=samples))
            assert len(cases) == 200
            totals = {mode: [sum(s["runs"][mode]["ns"] for c in cases for s in c["samples"]
                if s["round"] == ri) for ri in range(5)] for mode in modes}
            summary = dict(combined_vs_original=summarize(totals["reference"], totals["combined"]),
                combined_vs_initx=summarize(totals["initx"], totals["combined"]),
                original_AA=summarize(totals["reference"], totals["AA"]),
                initx_AA=summarize(totals["initx"], totals["initx_AA"]))
            passed = (summary["combined_vs_original"]["paired_reduction_percent"] >= 30
                and summary["combined_vs_initx"]["paired_reduction_percent"] >= 20
                and all(summary[k]["improving_rounds"] >= 4 for k in ("combined_vs_original", "combined_vs_initx"))
                and all(abs(summary[k]["paired_reduction_percent"]) < 5 for k in ("original_AA", "initx_AA")))
            results[str(index)] = dict(capture_sha256=digest, cases=cases, round_total_ns=totals,
                summary=summary, component_gate_passed=passed, all_state_bits_equal=True)
            print(json.dumps(dict(capture=index, summary=summary, component_gate_passed=passed)), flush=True)
    assert timed == 30000 and warm == 2000
    assert all(sha(ROOT / n) == d for n, d in sources.items())
    assert all(sha(Path(p)) == d for p, d in external.items())
    assert all(sha(p) == d for p, d in inputs)
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath), captures=results,
        timed_update_calls=timed, warm_update_calls=warm, all_mutated_array_bits_equal=True,
        sources_and_inputs_unchanged=True, component_gate_passes=sum(v["component_gate_passed"] for v in results.values()),
        whole_RTK_speed_gate=False), indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
