"""Replay actual feedback mass decisions with recorded graph transitions."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform
import statistics
import time
from types import MethodType

import numpy as np

from benchmark_feedback_dd import (evaluator, feedback_fields, invoke, load_records,
                                   outcome, prepare, validate_case)
from benchmark_move_selection import fingerprint


def scalar_visits(evaluation, source, records, fields):
    """Count original scalar finite/int tests outside all timing windows."""
    owner = evaluation[0]
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_feedback")
    loops = [n for n in ast.walk(method) if isinstance(n, ast.For) and n.body
             and isinstance(n.body[0], ast.Assign) and ast.unparse(n.body[0].targets[0]) == "di"]
    assert len(loops) == 1
    loops[0].body[:0] = ast.parse("_count_mass_visit()").body
    count = 0
    def counted():
        nonlocal count
        count += 1
    original = owner._feedback
    namespace = dict(original.__func__.__globals__, _count_mass_visit=counted)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    owner._feedback = MethodType(namespace["_feedback"], owner)
    try:
        for record in records:
            validate_case(evaluation, record, fields)
    finally:
        owner._feedback = original
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["manifest", "baseline", "candidate", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = load_records(args.manifest)
    if fingerprint(args.baseline) != manifest["runtime"]:
        parser.error("baseline differs from observed runtime")
    paths = [Path(__file__), Path(__file__).with_name("benchmark_feedback_dd.py"),
             Path(__file__).with_name("benchmark_move_selection.py"),
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    fields, _ = feedback_fields(args.baseline)
    owners = {label: evaluator(source) for label, source in [("baseline", args.baseline), ("candidate", args.candidate)]}
    counts = {label: scalar_visits(evaluation, source, records, fields)
              for (label, evaluation), source in zip(owners.items(), [args.baseline, args.candidate])}
    trials = []
    for trial in range(-2, args.pairs):
        row, times = dict(trial=trial, epochOrders=[]), {label: [] for label in owners}
        for epoch, record in enumerate(records):
            order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
            row["epochOrders"].append(order)
            for label in order:
                prepared = prepare(owners[label], record)
                begin = time.perf_counter_ns()
                value = invoke(owners[label], prepared)
                times[label].append((time.perf_counter_ns()-begin)/1e6)
                assert outcome(owners[label][0], prepared[0][0], fields, value) == record["expected"]
                assert prepared[0][0].index == len(record["events"])
        for label in owners:
            row[label+"EpochMs"], row[label+"Ms"] = times[label], sum(times[label])
        row["reductionPercent"] = 100*(1-row["candidateMs"]/row["baselineMs"])
        trials.append(row)
    measured = [r for r in trials if r["trial"] >= 0]
    summary = dict(baselineMedianMs=statistics.median(r["baselineMs"] for r in measured),
                   candidateMedianMs=statistics.median(r["candidateMs"] for r in measured),
                   medianReductionPercent=statistics.median(r["reductionPercent"] for r in measured),
                   improvedPairs=sum(r["reductionPercent"] > 0 for r in measured))
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during timing")
    result = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"], originalInputs=manifest["inputs"],
                  epochs=len(records), particleVisits=sum(r["particles"] for r in manifest["records"]),
                  python=platform.python_version(), numpy=np.__version__, scalarDdVisits=counts,
                  graphEvents=sum(len(r["events"]) for r in records), trials=trials, summary=summary,
                  scope="Actual feedback-policy calls in original epoch/particle/pair order/count. Clocks include DD matrix validation/allocation/truncation, matching selection and original ordered scalar weight additions. GTSAM graph operations replaced with recorded transitions of feedback-visible fields; transition playback included. Fresh state restoration, existing component DD preparation, imports/decode/validation excluded from clocks. Two warmups excluded; adjacent epoch windows alternate first policy by epoch/trial. No synthetic/expanded hypotheses or persistent DD cache. Not live feedback/GTSAM, full-filter, full-route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, scalarDdVisits=counts, graphEvents=result["graphEvents"])))


if __name__ == "__main__":
    main()
