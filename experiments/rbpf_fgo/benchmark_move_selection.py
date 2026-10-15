"""Replay actual RB-FGO-PF extension/release calls without importing GTSAM."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np


def fingerprint(path):
    path = Path(path)
    return dict(name=path.name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def reference(source):
    """Use the exact method bodies; the two snapshots supply their own logic."""
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    names = {"_pair_dd", "_extend", "_shrink_worst"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in methods} != names:
        raise ValueError("move methods missing")
    copied = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_H")
    reduced = ast.parse("class Reference:\n    pass").body[0]
    reduced.body = methods
    module = ast.fix_missing_locations(ast.Module(body=[copied, reduced], type_ignores=[]))
    namespace = dict(np=np)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Reference"]()


def encode(H):
    if H is None:
        return None
    return [[list(g), [[int(s), int(z)] for s, z in zmap.items()]] for g, zmap in H.items()]


def decode(value):
    return None if value is None else {tuple(g): dict(items) for g, items in value}


def load_records(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["records"]:
        path = manifest_path.parent / record["file"]
        if fingerprint(path) != record["fingerprint"]:
            raise ValueError("recorded input differs")
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        problem = SimpleNamespace(**arrays)
        problem.pairs = [(tuple(g), ref, sat) for g, ref, sat in record["pairs"]]
        calls = [(c["method"], decode(c["H"]), c["expected"]) for c in record["calls"]]
        records.append((problem, calls))
    if not records:
        raise ValueError("recorded calls required")
    return manifest, records


def evaluate(evaluator, problem, calls):
    return [getattr(evaluator, name)(H, problem) for name, H, _ in calls]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=Path(__file__).with_name("rbpf_fgo.py"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = load_records(args.manifest)
    if fingerprint(args.baseline)["sha256"] != manifest["runtime"]["sha256"]:
        parser.error("capture baseline differs")
    paths = [Path(__file__), args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    evaluators = dict(baseline=reference(args.baseline), candidate=reference(args.candidate))
    for problem, calls in records:
        inputs = [encode(H) for _, H, _ in calls]
        arrays = {k: v.tobytes() for k, v in vars(problem).items() if isinstance(v, np.ndarray)}
        for evaluator in evaluators.values():
            results = evaluate(evaluator, problem, calls)
            if [encode(H) for H in results] != [expected for _, _, expected in calls]:
                raise AssertionError("move output differs, including insertion order")
            for result, (_, H, _) in zip(results, calls):
                if result is not None:
                    assert result is not H
                    assert all(result.get(g) is not z for g, z in H.items())
            assert inputs == [encode(H) for _, H, _ in calls]
            assert arrays == {k: v.tobytes() for k, v in vars(problem).items() if isinstance(v, np.ndarray)}
    trials = []
    for trial in range(-2, args.pairs):
        order = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
        row = dict(trial=trial, order=order)
        for label in order:
            timings = []
            for problem, calls in records:
                begin = time.perf_counter_ns()
                outputs = evaluate(evaluators[label], problem, calls)
                timings.append((time.perf_counter_ns() - begin) / 1e6)
                del outputs
            row[label + "Ms"] = sum(timings)
            row[label + "EpochMs"] = timings
        row["reductionPercent"] = 100 * (1-row["candidateMs"]/row["baselineMs"])
        trials.append(row)
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during measurement")
    measured = [row for row in trials if row["trial"] >= 0]
    output = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], epochs=len(records),
                  calls=sum(len(c) for _, c in records), trials=trials,
                  python=platform.python_version(), numpy=np.__version__,
                  baselineMedianMs=statistics.median(r["baselineMs"] for r in measured),
                  candidateMedianMs=statistics.median(r["candidateMs"] for r in measured),
                  medianReductionPercent=statistics.median(r["reductionPercent"] for r in measured),
                  improvedPairs=sum(r["reductionPercent"] > 0 for r in measured),
                  scope="Sum of recorded epochs' actual helper calls, original order/count including output construction. Two warmups per source excluded. No synthesized/expanded hypotheses. Imports, input decoding and parity excluded. Not full-filter or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ["epochs", "calls", "baselineMedianMs", "candidateMedianMs", "medianReductionPercent", "improvedPairs"]}))


if __name__ == "__main__":
    main()
