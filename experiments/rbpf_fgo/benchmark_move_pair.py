"""Compare separate structural moves with the live paired-DD path on real inputs."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from benchmark_move_selection import encode, fingerprint, load_records, reference


def paired_reference(source):
    """Add the exact live pairing method to the original method-body loader."""
    base = reference(source)
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_move_pair")
    derived = ast.parse("class Paired(Base):\n    pass").body[0]
    derived.body = [method]
    module = ast.fix_missing_locations(ast.Module(body=[derived], type_ignores=[]))
    namespace = dict(Base=type(base))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Paired"]()


def separate(evaluator, H, problem):
    return evaluator._extend(H, problem), evaluator._shrink_worst(H, problem)


def pair_records(manifest):
    provenance, records = load_records(manifest)
    paired = []
    for problem, calls in records:
        if len(calls) % 2:
            raise ValueError("unpaired recorded call")
        groups = []
        for start in range(0, len(calls), 2):
            first, second = calls[start:start+2]
            if (first[0], second[0]) != ("_extend", "_shrink_worst") or encode(first[1]) != encode(second[1]):
                raise ValueError("expected adjacent structural calls for the same hypothesis")
            groups.append((first[1], (first[2], second[2])))
        paired.append((problem, groups))
    return provenance, paired


def invoke(evaluator, problem, pairs, paired):
    if paired:
        return [evaluator._move_pair(H, problem) for H, _ in pairs]
    return [(evaluator._extend(H, problem), evaluator._shrink_worst(H, problem))
            for H, _ in pairs]


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
    manifest, records = pair_records(args.manifest)
    paths = [Path(__file__), Path(__file__).with_name("benchmark_move_selection.py"),
             args.baseline, args.candidate, args.manifest]
    frozen = [fingerprint(p) for p in paths]
    evaluators = dict(baseline=reference(args.baseline), candidate=paired_reference(args.candidate))
    dd_counts = {}
    for label, evaluator in evaluators.items():
        original = evaluator._pair_dd
        count = [0]

        def counted(H, problem, original=original, count=count):
            count[0] += 1
            return original(H, problem)

        evaluator._pair_dd = counted
        for problem, pairs in records:
            inputs = [encode(H) for H, _ in pairs]
            arrays = {k: v.tobytes() for k, v in vars(problem).items() if isinstance(v, np.ndarray)}
            results = invoke(evaluator, problem, pairs, label == "candidate")
            observed = [[encode(a), encode(b)] for a, b in results]
            if observed != [list(expected) for _, expected in pairs]:
                raise AssertionError("integer outputs/order differ")
            for (a, b), (H, _) in zip(results, pairs):
                for result in [a, b]:
                    if result is not None:
                        assert result is not H
                        assert all(result.get(g) is not z for g, z in H.items())
                if a is not None and b is not None:
                    assert all(a.get(g) is not z for g, z in b.items())
            assert inputs == [encode(H) for H, _ in pairs]
            assert arrays == {k: v.tobytes() for k, v in vars(problem).items() if isinstance(v, np.ndarray)}
        dd_counts[label] = count[0]
        del evaluator._pair_dd  # remove observation overhead before every clock
    pair_count = sum(len(p) for _, p in records)
    assert dd_counts == dict(baseline=2*pair_count, candidate=pair_count)
    trials = []
    for trial in range(-2, args.pairs):
        order = ["baseline", "candidate"] if trial % 2 == 0 else ["candidate", "baseline"]
        row = dict(trial=trial, order=order)
        for label in order:
            elapsed = []
            for problem, pairs in records:
                start = time.perf_counter_ns()
                outputs = invoke(evaluators[label], problem, pairs, label == "candidate")
                elapsed.append((time.perf_counter_ns()-start)/1e6)
                del outputs
            row[label+"EpochMs"] = elapsed
            row[label+"Ms"] = sum(elapsed)
        row["reductionPercent"] = 100*(1-row["candidateMs"]/row["baselineMs"])
        trials.append(row)
    if frozen != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during measurement")
    measured = [r for r in trials if r["trial"] >= 0]
    output = dict(schemaVersion=1, sources=frozen, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], epochs=len(records), particleVisits=pair_count,
                  configuredParticles=sorted({r["particles"] for r in manifest["records"]}),
                  structuralCalls=2*pair_count, ddCallsPerValidationPass=dd_counts, trials=trials,
                  python=platform.python_version(), numpy=np.__version__,
                  baselineMedianMs=statistics.median(r["baselineMs"] for r in measured),
                  candidateMedianMs=statistics.median(r["candidateMs"] for r in measured),
                  medianReductionPercent=statistics.median(r["reductionPercent"] for r in measured),
                  improvedPairs=sum(r["reductionPercent"] > 0 for r in measured),
                  scope="Actual paired extension/release calls in original epoch/particle order, including DD construction and both outputs. No synthesized/expanded hypotheses. Two warmups per source excluded. Imports, decode, parity and DD counters excluded. Not full-filter/GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({k: output[k] for k in ["epochs", "particleVisits", "structuralCalls", "baselineMedianMs", "candidateMedianMs", "medianReductionPercent", "improvedPairs"]}))


if __name__ == "__main__":
    main()
