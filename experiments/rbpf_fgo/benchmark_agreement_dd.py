"""Replay actual component-agreement phases using their exact source bodies."""

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

from benchmark_dd_layout import epoch_type
from benchmark_move_selection import decode, encode, fingerprint


def phase_bounds(step):
    start = next(i for i, n in enumerate(step.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "dd_map")
    if (isinstance(step.body[start-1], ast.Assign)
            and ast.unparse(step.body[start-1].targets[0]) == "dd_all"):
        start -= 1
    end = next(i for i in range(start, len(step.body)) if isinstance(step.body[i], ast.Assign)
               and ast.unparse(step.body[i].targets[0]) == "xz")
    return start, end


def evaluator(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    start, end = phase_bounds(step)
    helper = ast.parse("def agreement(self, prob, w, ranked, rep, map_H):\n"
                       "    cfg = self.cfg\n    out = {}\n    ddpr_pos = None").body[0]
    helper.body += step.body[start:end]
    helper.body += ast.parse("return locals()").body
    reference = ast.parse("class Reference:\n    pass").body[0]
    reference.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pair_dd"]
    reference.body.append(helper)
    thresholds = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "NB_AT_THRESH")
    namespace = dict(np=np)
    module = ast.fix_missing_locations(ast.Module(body=[thresholds, reference], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Reference"]()


RESULT_KEYS = ["dd_map", "dd_all", "pair_mass", "used_pairs", "gamma_min_used",
               "ne_idx", "ne_mass", "ne_ranked", "dd_ne", "dd_l", "masses_ne"]


def result_digest(values):
    def freeze(value):
        if isinstance(value, np.ndarray):
            return ["array", value.dtype.str, value.shape, value.tobytes().hex()]
        if isinstance(value, (float, np.floating)):
            return ["float", np.float64(value).tobytes().hex()]
        if isinstance(value, dict):
            return ["dict", [[freeze(k), freeze(v)] for k, v in value.items()]]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [freeze(v) for v in value]]
        if isinstance(value, np.integer):
            return int(value)
        return value
    outputs = {k: values.get(k) for k in RESULT_KEYS}
    outputs["out"] = {k: values["out"][k] for k in ["nb", "gamma_pair", "nb_at", "ne_mass", "nb_at_ne"]}
    return hashlib.sha256(json.dumps(freeze(outputs), separators=(",", ":")).encode()).hexdigest()


def load_records(path, source):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, list) else value
    records = []
    for row in manifest["records"]:
        file = path.parent/row["file"]
        if fingerprint(file) != row["fingerprint"]:
            raise ValueError("captured weights changed")
        with np.load(file, allow_pickle=False) as arrays:
            weights = arrays["weights"]
        prob = SimpleNamespace()
        epoch_type(source).set_pairs(prob, [(tuple(g), ref, sat) for g, ref, sat in row["pairs"]])
        particles = [SimpleNamespace(H=decode(H)) for H in row["hypotheses"]]
        ranked = [(tuples(bid), mass) for bid, mass in row["ranked"]]
        rep = {tuples(bid): decode(H) for bid, H in row["representatives"]}
        records.append((prob, weights, particles, SimpleNamespace(**row["config"]),
                        ranked, rep, decode(row["mapH"]), row["resultDigest"]))
    if not records:
        raise ValueError("actual phase records required")
    return manifest, records


def invoke(owner, record):
    prob, weights, owner.particles, owner.cfg, ranked, rep, map_H, _ = record
    return owner.agreement(prob, weights, ranked, rep, map_H)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["manifest", "baseline", "candidate", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = load_records(args.manifest, args.baseline)
    if fingerprint(args.baseline)["sha256"] != manifest["runtime"]["sha256"]:
        parser.error("capture runtime differs")
    here = Path(__file__).resolve().parent
    paths = [Path(__file__), here/"benchmark_dd_layout.py", here/"benchmark_move_selection.py",
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    owners = dict(baseline=evaluator(args.baseline), candidate=evaluator(args.candidate))
    counts = {label: 0 for label in owners}
    for record in records:
        saved = [encode(p.H) for p in record[2]]
        weights = record[1].tobytes()
        for label, owner in owners.items():
            original = owner._pair_dd
            def counted(H, prob, original=original, label=label):
                counts[label] += 1
                return original(H, prob)
            owner._pair_dd = counted
            result = invoke(owner, record)
            owner._pair_dd = original
            assert result_digest(result) == record[-1]
            assert saved == [encode(p.H) for p in record[2]] and weights == record[1].tobytes()
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
    output = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], python=platform.python_version(), numpy=np.__version__,
                  epochs=len(records), particleVisits=sum(len(r[2]) for r in records),
                  configuredParticles=sorted({r["particles"] for r in manifest["records"]}),
                  ddCalls=counts, trials=trials, summary=summary,
                  scope="Actual per-component posterior agreement and non-empty diagnostic phases; original particle, pair and summation order. Includes DD generation and output construction. Adjacent epoch windows alternate first policy with trial and epoch. Two warmups excluded. Loading, decode, validation and basin ranking excluded. No synthetic/expanded hypotheses. No persistent DD cache. Not full-filter, full-route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, ddCalls=counts)))


if __name__ == "__main__":
    main()
