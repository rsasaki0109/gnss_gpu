"""Replay immutable epoch layouts, charging preparation once per epoch window."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np

from benchmark_dd_lookups import invoke, reduction
from benchmark_move_pair import pair_records, paired_reference
from benchmark_move_selection import encode, fingerprint


def epoch_type(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EpochProblem")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["EpochProblem"]


def prepare(prob, source_type):
    source_type.set_pairs(prob, prob.pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["manifest", "baseline", "candidate", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = pair_records(args.manifest)
    if fingerprint(args.baseline)["sha256"] != manifest["runtime"]["sha256"]:
        parser.error("recorded baseline differs")
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), here/"benchmark_dd_lookups.py", here/"benchmark_move_pair.py",
             here/"benchmark_move_selection.py", args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    evaluators = dict(baseline=paired_reference(args.baseline), candidate=paired_reference(args.candidate))
    source_type = epoch_type(args.candidate)
    prepared_records = []
    for prob, visits in records:
        candidate_prob = SimpleNamespace(**vars(prob))
        prepare(candidate_prob, source_type)
        before = [encode(H) for H, _ in visits]
        arrays = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
        old = invoke(evaluators["baseline"], prob, visits, "dd")
        new = invoke(evaluators["candidate"], candidate_prob, visits, "dd")
        assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                   for a, b in zip(old, new))
        for label, problem in [("baseline", prob), ("candidate", candidate_prob)]:
            result = invoke(evaluators[label], problem, visits, "moves")
            assert [[encode(a), encode(b)] for a, b in result] == [list(e) for _, e in visits]
            assert before == [encode(H) for H, _ in visits]
            assert arrays == {k: v.tobytes() for k, v in vars(problem).items() if isinstance(v, np.ndarray)}
        prepared_records.append((prob, candidate_prob, visits))
    stages = {}
    for stage in ["dd", "moves"]:
        trials = []
        for trial in range(-2, args.pairs):
            row = dict(trial=trial, epochOrders=[])
            times = {label: dict(wall=[], cpu=[]) for label in evaluators}
            for epoch, (prob, candidate_prob, visits) in enumerate(prepared_records):
                order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
                row["epochOrders"].append(order)
                for label in order:
                    problem = prob if label == "baseline" else candidate_prob
                    start_wall, start_cpu = time.perf_counter_ns(), time.process_time_ns()
                    if label == "candidate":
                        prepare(problem, source_type)
                    result = invoke(evaluators[label], problem, visits, stage)
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
                row[metric+"ReductionPercent"] = reduction(row["baseline"+metric.title()+"Ms"], row["candidate"+metric.title()+"Ms"])
            trials.append(row)
        measured = [r for r in trials if r["trial"] >= 0]
        stage_result = dict(trials=trials)
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
                  scope="Actual recorded structural visits, original order/count per policy. Candidate layout freezing/preparation charged once per epoch window in both stages. DD-only or paired structural outputs. First policy alternates with trial and epoch. Two warmups per source/stage excluded. No synthetic/expanded hypotheses. Decode/validation excluded. CPU times secondary and may be quantized or zero. Not full-filter/GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({s: {m: v[m] for m in ["wall", "cpu"]} for s, v in stages.items()}))


if __name__ == "__main__":
    main()
