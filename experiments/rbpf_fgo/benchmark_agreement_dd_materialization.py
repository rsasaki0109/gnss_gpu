"""Measure DD materialization separately from agreement's scalar mass sums.

Retain the complete phase study alongside this narrower scope. This controller
selects exact DD assignments and their guards from each production source.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from benchmark_agreement_dd import evaluator, invoke, load_records, phase_bounds, result_digest
from benchmark_move_selection import encode, fingerprint


def materializer(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    start, end = phase_bounds(step)
    keep = {"dd_map", "dd_all", "ne_idx", "ne_mass", "ne_ranked", "dd_ne", "dd_l"}
    def selected(nodes):
        result = []
        for node in nodes:
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in keep:
                result.append(node)
            elif isinstance(node, ast.If):
                body = selected(node.body)
                if body:
                    if node.orelse:
                        raise ValueError("unexpected alternative in DD materialization")
                    result.append(ast.If(test=node.test, body=body, orelse=[]))
        return result
    helper = ast.parse("def agreement(self, prob, w, ranked, rep, map_H):\n    pass").body[0]
    helper.body = selected(step.body[start:end])+ast.parse("return locals()").body
    reference = ast.parse("class Reference:\n    pass").body[0]
    reference.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pair_dd"]+[helper]
    namespace = dict(np=np)
    module = ast.fix_missing_locations(ast.Module(body=[reference], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Reference"]()


def dd_digest(values):
    projection = {k: values.get(k) for k in ["dd_map", "dd_all", "dd_ne", "dd_l"]}
    projection["out"] = {k: None for k in ["nb", "gamma_pair", "nb_at", "ne_mass", "nb_at_ne"]}
    return result_digest(projection)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["manifest", "baseline", "candidate", "full-replay", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = load_records(args.manifest, args.baseline)
    full = json.loads(args.full_replay.read_text(encoding="utf-8"))
    if (fingerprint(args.baseline) != manifest["runtime"]
            or full["sources"][3:5] != [fingerprint(args.baseline), fingerprint(args.candidate)]
            or full["sources"][-1] != fingerprint(args.manifest)):
        parser.error("full-phase study or capture runtime differs")
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), *[here/name for name in ["benchmark_agreement_dd.py",
             "benchmark_dd_layout.py", "benchmark_dd_lookups.py", "benchmark_move_pair.py",
             "benchmark_move_selection.py"]], args.baseline, args.candidate, args.manifest, args.full_replay]
    sources = [fingerprint(p) for p in paths]
    owners = dict(baseline=materializer(args.baseline), candidate=materializer(args.candidate))
    full_owners = dict(baseline=evaluator(args.baseline), candidate=evaluator(args.candidate))
    counts = {label: 0 for label in owners}
    for record in records:
        before = [encode(p.H) for p in record[2]], record[1].tobytes()
        for label, owner in owners.items():
            expected = invoke(full_owners[label], record)
            assert result_digest(expected) == record[-1]
            original = owner._pair_dd
            def counted(H, prob, label=label, original=original):
                counts[label] += 1
                return original(H, prob)
            owner._pair_dd = counted
            observed = invoke(owner, record)
            owner._pair_dd = original
            assert dd_digest(observed) == dd_digest(expected)
            assert before == ([encode(p.H) for p in record[2]], record[1].tobytes())
    assert counts == full["ddCalls"]
    trials = []
    for trial in range(-2, args.pairs):
        row = dict(trial=trial, epochOrders=[])
        times = {label: [] for label in owners}
        for epoch, record in enumerate(records):
            order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
            row["epochOrders"].append(order)
            for label in order:
                begin = time.perf_counter_ns()
                result = invoke(owners[label], record)
                times[label].append((time.perf_counter_ns()-begin)/1e6)
                del result
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
    output = dict(schemaVersion=1, sources=sources, epochs=len(records), particleVisits=full["particleVisits"],
                  originalInputs=manifest["inputs"], captureRuntime=manifest["runtime"],
                  python=platform.python_version(), numpy=np.__version__, ddCalls=counts,
                  trials=trials, summary=summary,
                  scope="DD materialization only: exact source DD assignments, MAP/availability guards, non-empty indices/mass preparation and output arrays. Each projection checked against the observed full phase. Original particle/pair order/count. Scalar component-agreement sums and diagnostic thresholds excluded. Two warmups excluded; adjacent epoch windows alternate first policy by epoch/trial. Import/decode/validation excluded. The separate full-phase study remains required; no full-filter, full-route or GPU speed claim.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, ddCalls=counts)))


if __name__ == "__main__":
    main()
