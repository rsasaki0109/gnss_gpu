"""Replay recorded canonical-ID calls in their original epoch/call order."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform
import statistics
import time

from benchmark_move_selection import decode, encode, fingerprint


def canonical(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_canon_basin_id")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_canon_basin_id"]


def json_id(value):
    return [[list(g), [list(pair) for pair in pairs]] for g, pairs in value]


def load_calls(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for entry in manifest["records"]:
        path = manifest_path.parent / entry["file"]
        if fingerprint(path) != entry["fingerprint"]:
            raise ValueError("captured calls changed")
        records.append([(decode(c["H"]), c["expected"])
                        for c in json.loads(path.read_text(encoding="utf-8"))])
    if not records or not all(records):
        raise ValueError("nonempty actual call records required")
    return manifest, records


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
        parser.error("baseline must be the captured runtime snapshot")
    paths = [Path(__file__), Path(__file__).with_name("benchmark_move_selection.py"),
             args.baseline, args.candidate, args.manifest]
    frozen = [fingerprint(p) for p in paths]
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
            times = []
            for calls in records:
                begin = time.perf_counter_ns()
                outputs = [evaluate(H) for H, _ in calls]
                times.append((time.perf_counter_ns()-begin)/1e6)
                del outputs
            row[label+"EpochMs"] = times
            row[label+"Ms"] = sum(times)
        row["reductionPercent"] = 100*(1-row["candidateMs"]/row["baselineMs"])
        trials.append(row)
    if frozen != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during timing")
    measured = [r for r in trials if r["trial"] >= 0]
    result = dict(schemaVersion=1, sources=frozen, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], python=platform.python_version(),
                  epochs=len(records), calls=sum(len(c) for c in records), trials=trials,
                  baselineMedianMs=statistics.median(r["baselineMs"] for r in measured),
                  candidateMedianMs=statistics.median(r["candidateMs"] for r in measured),
                  medianReductionPercent=statistics.median(r["reductionPercent"] for r in measured),
                  improvedPairs=sum(r["reductionPercent"] > 0 for r in measured),
                  scope=f"All actual canonical-ID calls inside {len(records)} sampled filter steps, original order/count, including output construction. No synthetic or expanded hypotheses. Two warmups per source excluded. Imports, decoding, parity and hashes outside clocks. Not full-filter or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ["epochs", "calls", "baselineMedianMs",
                     "candidateMedianMs", "medianReductionPercent", "improvedPairs"]}))


if __name__ == "__main__":
    main()
