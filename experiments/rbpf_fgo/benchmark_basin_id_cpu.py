"""Record process CPU time alongside wall time for the original ID call stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

from benchmark_basin_id import canonical, json_id, load_calls
from benchmark_move_selection import encode, fingerprint


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
    manifest, records = load_calls(args.manifest)
    if fingerprint(args.baseline) != manifest["runtime"]:
        parser.error("baseline differs from captured runtime")
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), here/"benchmark_basin_id.py", here/"benchmark_move_selection.py",
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    evaluators = dict(baseline=canonical(args.baseline), candidate=canonical(args.candidate))
    for calls in records:
        before = [encode(H) for H, _ in calls]
        for evaluate in evaluators.values():
            assert [json_id(evaluate(H)) for H, _ in calls] == [e for _, e in calls]
            assert before == [encode(H) for H, _ in calls]
    trials = []
    for trial in range(-2, args.pairs):
        order = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
        row = dict(trial=trial, order=order)
        for label in order:
            evaluate = evaluators[label]
            wall, cpu = [], []
            for calls in records:
                start_wall = time.perf_counter_ns()
                start_cpu = time.process_time_ns()
                outputs = [evaluate(H) for H, _ in calls]
                cpu.append((time.process_time_ns()-start_cpu)/1e6)
                wall.append((time.perf_counter_ns()-start_wall)/1e6)
                del outputs
            row[label+"CpuEpochMs"], row[label+"WallEpochMs"] = cpu, wall
            row[label+"CpuMs"], row[label+"WallMs"] = sum(cpu), sum(wall)
        for metric in ["Cpu", "Wall"]:
            row[metric.lower()+"ReductionPercent"] = 100*(1-row["candidate"+metric+"Ms"]/row["baseline"+metric+"Ms"])
        trials.append(row)
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during timing")
    measured = [r for r in trials if r["trial"] >= 0]
    result = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], python=platform.python_version(),
                  epochs=len(records), calls=sum(len(c) for c in records), trials=trials,
                  clocks=dict(cpu=vars(time.get_clock_info("process_time")),
                              wall=vars(time.get_clock_info("perf_counter"))),
                  scope="Original canonical-ID calls in recorded epoch/call order. Includes output construction. Process CPU time excludes time when this process is not scheduled; wall time is retained separately. Two warmups per source excluded. No synthetic/expanded hypotheses, no full-filter/GPU claim.")
    for metric in ["Cpu", "Wall"]:
        result[metric.lower()] = dict(
            baselineMedianMs=statistics.median(r["baseline"+metric+"Ms"] for r in measured),
            candidateMedianMs=statistics.median(r["candidate"+metric+"Ms"] for r in measured),
            medianReductionPercent=statistics.median(r[metric.lower()+"ReductionPercent"] for r in measured),
            improvedPairs=sum(r[metric.lower()+"ReductionPercent"] > 0 for r in measured))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ["epochs", "calls", "cpu", "wall", "clocks"]}))


if __name__ == "__main__":
    main()
