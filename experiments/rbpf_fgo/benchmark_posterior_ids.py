"""Replay actual basin grouping and tracked-output aggregation inputs.

Conditional position solves and cluster publication are excluded explicitly;
their statements remain unchanged in production and run in pipeline controls.
"""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from types import SimpleNamespace

import numpy as np

from benchmark_move_selection import decode, encode, fingerprint


def assigned(node, name):
    return (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name)


def phase_bounds(step):
    start = next(i for i, n in enumerate(step.body) if assigned(n, "masses"))
    top = next(i for i in range(start, len(step.body)) if assigned(step.body[i], "top"))
    after_top = next(i+1 for i in range(top, len(step.body)) if isinstance(step.body[i], ast.Assign)
                     and ast.unparse(step.body[i].targets[0]) == "out['top']")
    end = next(i for i in range(after_top, len(step.body)) if assigned(step.body[i], "dd_all"))
    return start, top, after_top, end


def evaluator(source, stage="aggregation"):
    if stage not in {"basins", "aggregation"}:
        raise ValueError("unknown replay stage")
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    start, top, after_top, end = phase_bounds(step)
    helper = ast.parse("def aggregate(self, w):\n    cfg = self.cfg\n    out = {}").body[0]
    helper.body += step.body[start:top]
    if stage == "aggregation":
        helper.body += step.body[after_top:end]
    helper.body += ast.parse("return locals()").body
    reference = ast.parse("class Reference:\n    pass").body[0]
    reference.body = [helper]
    canon = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_canon_basin_id")
    namespace = dict(np=np, OrderedDict=OrderedDict)
    module = ast.fix_missing_locations(ast.Module(body=[canon, reference], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Reference"](), namespace


def result_digest(values, stage="aggregation"):
    def freeze(value):
        if isinstance(value, np.ndarray):
            return ["array", value.dtype.str, value.shape, value.tobytes().hex()]
        if isinstance(value, (float, np.floating)):
            return ["float", np.float64(value).tobytes().hex()]
        if isinstance(value, dict):
            return ["dict", [[freeze(k), freeze(v)] for k, v in value.items()]]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [freeze(v) for v in value]]
        return int(value) if isinstance(value, np.integer) else value
    outputs = {k: values[k] for k in ["masses", "ranked", "rep"]}
    keys = ["n_basins"]
    if stage == "aggregation":
        outputs.update(gamma=values["gamma"], map_H=values["map_H"])
        keys += ["gamma", "second_gamma", "trk_var", "trk_map", "trk_map_mass", "trk_mix", "trk_mix_mass"]
    outputs["out"] = {k: values["out"][k] for k in keys if k in values["out"]}
    return hashlib.sha256(json.dumps(freeze(outputs), separators=(",", ":")).encode()).hexdigest()


def load_records(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in manifest["records"]:
        file = path.parent/row["file"]
        if fingerprint(file) != row["fingerprint"]:
            raise ValueError("recorded arrays changed")
        with np.load(file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        def state(key):
            return None if key is None else SimpleNamespace(x=arrays[key])
        particles = [SimpleNamespace(H=decode(p["H"]), state=state(p["state"]),
                                    vstates=[state(k) for k in p["variants"]]) for p in row["hypotheses"]]
        cfg = SimpleNamespace(track=row["track"], track_variants=[None]*row["variantCount"])
        records.append((particles, arrays["weights"], cfg, row["resultDigests"]))
    if not records:
        raise ValueError("actual posterior inputs required")
    return manifest, records


def invoke(owner, record):
    owner.particles, weights, owner.cfg, _ = record
    return owner.aggregate(weights)


def input_digest(record):
    particles, weights, _, _ = record
    values = [weights.tobytes()]
    for p in particles:
        values.append(json.dumps(encode(p.H)).encode())
        values += [b"none" if s is None else s.x.tobytes() for s in [p.state, *p.vstates]]
    return hashlib.sha256(b"".join(values)).hexdigest()


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
        parser.error("baseline differs from recorded runtime")
    paths = [Path(__file__), Path(__file__).with_name("benchmark_move_selection.py"),
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    stages = {}
    for stage in ["basins", "aggregation"]:
        owners = {label: evaluator(source, stage) for label, source in
                  [("baseline", args.baseline), ("candidate", args.candidate)]}
        counts = {label: 0 for label in owners}
        for record in records:
            before = input_digest(record)
            for label, (owner, namespace) in owners.items():
                original = namespace["_canon_basin_id"]
                def counted(H, original=original, label=label):
                    counts[label] += 1
                    return original(H)
                namespace["_canon_basin_id"] = counted
                observed = invoke(owner, record)
                namespace["_canon_basin_id"] = original
                assert result_digest(observed, stage) == record[-1][stage]
                assert input_digest(record) == before
        trials = []
        for trial in range(-2, args.pairs):
            row = dict(trial=trial, epochOrders=[])
            times = {label: [] for label in owners}
            for epoch, record in enumerate(records):
                order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
                row["epochOrders"].append(order)
                for label in order:
                    begin = time.perf_counter_ns()
                    result = invoke(owners[label][0], record)
                    times[label].append((time.perf_counter_ns()-begin)/1e6)
                    del result
            for label in owners:
                row[label+"EpochMs"], row[label+"Ms"] = times[label], sum(times[label])
            row["reductionPercent"] = 100*(1-row["candidateMs"]/row["baselineMs"])
            trials.append(row)
        measured = [r for r in trials if r["trial"] >= 0]
        stages[stage] = dict(trials=trials, canonicalCalls=counts,
            baselineMedianMs=statistics.median(r["baselineMs"] for r in measured),
            candidateMedianMs=statistics.median(r["candidateMs"] for r in measured),
            medianReductionPercent=statistics.median(r["reductionPercent"] for r in measured),
            improvedPairs=sum(r["reductionPercent"] > 0 for r in measured))
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during timing")
    output = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], python=platform.python_version(), numpy=np.__version__,
                  epochs=len(records), particleVisits=sum(len(r[0]) for r in records),
                  configuredParticles=sorted({r["particles"] for r in manifest["records"]}), stages=stages,
                  scope="Actual basin grouping and tracked-output aggregation inputs, original particle order/count. Basins includes canonical IDs, ordered mass sums, ranking, first representatives and output construction. Aggregation also includes MAP/non-empty tracked-output selection and weighted means. Conditional top-position solves and cluster-state publication excluded from both scopes; their production statements remain unchanged and are exercised in pipeline controls. Normalized weights are captured inputs. Two warmups excluded. Adjacent epoch windows alternate first policy by epoch/trial. Loading/decode/validation excluded. No synthetic/expanded hypotheses or persistent ID cache. Not full posterior, full-filter, full-route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({stage: {k: row[k] for k in ["canonicalCalls", "baselineMedianMs", "candidateMedianMs",
                      "medianReductionPercent", "improvedPairs"]} for stage, row in stages.items()}))


if __name__ == "__main__":
    main()
