"""Compare DD generation and paired moves on original recorded particle visits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from benchmark_move_pair import pair_records, paired_reference
from benchmark_move_selection import encode, fingerprint


def invoke(evaluator, prob, visits, stage):
    method = evaluator._pair_dd if stage == "dd" else evaluator._move_pair
    return [method(H, prob) for H, _ in visits]


def reduction(baseline, candidate):
    return 100*(1-candidate/baseline) if baseline else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = pair_records(args.manifest)
    if fingerprint(args.baseline)["sha256"] != manifest["runtime"]["sha256"]:
        parser.error("baseline differs from recorded runtime")
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), here/"benchmark_move_pair.py", here/"benchmark_move_selection.py",
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    evaluators = dict(baseline=paired_reference(args.baseline), candidate=paired_reference(args.candidate))
    for prob, visits in records:
        before = [encode(H) for H, _ in visits]
        arrays = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
        base = invoke(evaluators["baseline"], prob, visits, "dd")
        new = invoke(evaluators["candidate"], prob, visits, "dd")
        assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                   for a, b in zip(base, new))
        for evaluator in evaluators.values():
            result = invoke(evaluator, prob, visits, "moves")
            assert [[encode(a), encode(b)] for a, b in result] == [list(e) for _, e in visits]
            assert before == [encode(H) for H, _ in visits]
            assert arrays == {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    stages = {}
    for stage in ["dd", "moves"]:
        trials = []
        for trial in range(-2, args.pairs):
            row = dict(trial=trial, epochOrders=[])
            times = {label: dict(wall=[], cpu=[]) for label in evaluators}
            for epoch, (prob, visits) in enumerate(records):
                order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
                row["epochOrders"].append(order)
                for label in order:
                    start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
                    result = invoke(evaluators[label], prob, visits, stage)
                    elapsed_cpu = time.process_time_ns()-start_cpu
                    elapsed_wall = time.perf_counter_ns()-start_wall
                    times[label]["cpu"].append(elapsed_cpu/1e6)
                    times[label]["wall"].append(elapsed_wall/1e6)
                    del result
            for label in evaluators:
                for metric in ["wall", "cpu"]:
                    row[label+metric.title()+"EpochMs"] = times[label][metric]
                    row[label+metric.title()+"Ms"] = sum(times[label][metric])
            for metric in ["wall", "cpu"]:
                row[metric+"ReductionPercent"] = reduction(row["baseline"+metric.title()+"Ms"],
                                                          row["candidate"+metric.title()+"Ms"])
            trials.append(row)
        stage_result = dict(trials=trials)
        measured = [r for r in trials if r["trial"] >= 0]
        for metric in ["wall", "cpu"]:
            valid = [r[metric+"ReductionPercent"] for r in measured if r[metric+"ReductionPercent"] is not None]
            stage_result[metric] = dict(
                baselineMedianMs=statistics.median(r["baseline"+metric.title()+"Ms"] for r in measured),
                candidateMedianMs=statistics.median(r["candidate"+metric.title()+"Ms"] for r in measured),
                medianReductionPercent=statistics.median(valid) if valid else None,
                validPairs=len(valid), improvedPairs=sum(v > 0 for v in valid))
        stages[stage] = stage_result
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during timing")
    output = dict(schemaVersion=1, sources=sources, originalInputs=manifest["inputs"],
                  captureRuntime=manifest["runtime"], python=platform.python_version(), numpy=np.__version__,
                  epochs=len(records), particleVisits=sum(len(v) for _, v in records),
                  configuredParticles=sorted({r["particles"] for r in manifest["records"]}), stages=stages,
                  scope="Actual recorded structural visits, original order/count within each policy. One DD per visit, or both paired structural outputs. Policies alternate within every epoch window; first policy alternates with trial and epoch. Two warmups per source/stage excluded. No synthetic/expanded workload. Decode, parity and input loading excluded. CPU times may be quantized or zero and remain secondary to wall latency. Not full-filter or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({stage: {metric: value[metric] for metric in ["wall", "cpu"]}
                      for stage, value in stages.items()}))


if __name__ == "__main__":
    main()
