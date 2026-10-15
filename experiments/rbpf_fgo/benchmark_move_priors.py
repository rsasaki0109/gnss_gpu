"""Replay actual scalar move distributions with epoch-local prior reuse."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np

from benchmark_feedback_dd import pack, unpack
from benchmark_move_selection import fingerprint


def evaluator(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    loop = next(n for n in step.body if isinstance(n, ast.For) and ast.unparse(n.target) == "p"
                and "cfg.track != 1" in ast.unparse(n.iter))
    cached = any(isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "move_logpri" for n in step.body)
    target = "logpri" if cached else "pri"
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == target)
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "pick")
    wrapper = ast.parse("def evaluate(p, moves, H_ext, H_shr, move_logpri):\n    pass").body[0]
    prefix = []
    if cached:
        # Copy only the key assignments and their original branch tests.
        initial = next(n for n in loop.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "prior_key")
        nonempty = next(n for n in loop.body if isinstance(n, ast.If) and ast.unparse(n.test) == "p.H")
        body = [n for n in nonempty.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "prior_key"]
        for node in nonempty.body:
            if isinstance(node, ast.If):
                updates = [n for n in node.body if isinstance(n, ast.AugAssign) and ast.unparse(n.target) == "prior_key"]
                if updates:
                    body.append(ast.If(test=node.test, body=updates, orelse=[]))
        assert len(body) == 3
        prefix = [initial, ast.If(test=nonempty.test, body=body, orelse=[])]
    wrapper.body = prefix+loop.body[start:end]+ast.parse("return p.logw, wsum / tot").body
    namespace = dict(np=np, math=math)
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["evaluate"]


def load_records(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in manifest["records"]:
        file = path.parent/row["file"]
        if fingerprint(file) != row["fingerprint"]:
            raise ValueError("captured move inputs changed")
        calls = unpack(json.loads(file.read_text(encoding="utf-8")))
        assert len(calls) == row["calls"]
        prepared = [(SimpleNamespace(H=c["H"], logw=c["before"]), c) for c in calls]
        records.append(prepared)
    if not records or not all(records):
        raise ValueError("actual move distributions required")
    return manifest, records


def invoke(evaluate, record):
    cache, result = {}, []
    for particle, call in record:
        particle.logw = call["before"]
        result.append(evaluate(particle, call["moves"], call["extended"], call["shrunk"], cache))
    return result


def input_digest(record):
    return hashlib.sha256(json.dumps(pack([c for _, c in record]), separators=(",", ":")).encode()).hexdigest()


def normalizations(evaluate, records):
    count = 0
    class Counter:
        def __getattr__(self, name):
            return getattr(np, name)

        def log(self, value):
            nonlocal count
            count += 1
            return np.log(value)
    evaluate.__globals__["np"] = Counter()
    try:
        for record in records:
            assert pack(invoke(evaluate, record)) == pack([c["expected"] for _, c in record])
    finally:
        evaluate.__globals__["np"] = np
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
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), here/"benchmark_feedback_dd.py", here/"benchmark_move_selection.py",
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    policies = dict(baseline=evaluator(args.baseline), candidate=evaluator(args.candidate))
    for record in records:
        before = input_digest(record)
        for evaluate in policies.values():
            assert pack(invoke(evaluate, record)) == pack([c["expected"] for _, c in record])
            assert before == input_digest(record)
    counts = {label: normalizations(evaluate, records) for label, evaluate in policies.items()}
    trials = []
    for trial in range(-2, args.pairs):
        row, times = dict(trial=trial, epochOrders=[]), {label: [] for label in policies}
        for epoch, record in enumerate(records):
            order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
            row["epochOrders"].append(order)
            for label in order:
                begin = time.perf_counter_ns()
                result = invoke(policies[label], record)
                times[label].append((time.perf_counter_ns()-begin)/1e6)
                del result
        for label in policies:
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
        epochs=len(records), particleVisits=sum(len(r) for r in records), priorNormalizations=counts,
        python=platform.python_version(), numpy=np.__version__, trials=trials, summary=summary,
        scope="Actual scalar move distributions in original epoch/particle/move order/count. Includes new availability-key construction, fresh epoch cache, prior vector/normalization/log, recorded-score combination, max/exp/sum, log-weight addition and probability normalization. Per-call input log-weight reset and helper call included. Move generation, likelihood scoring, RNG sampling, loading/decode/validation excluded. Two warmups excluded; adjacent epoch windows alternate first policy by epoch/trial. No synthetic/expanded inputs or cache across epochs. Not live full-proposal, full-filter, route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, priorNormalizations=counts)))


if __name__ == "__main__":
    main()
