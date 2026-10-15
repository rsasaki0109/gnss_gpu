"""Replay observed likelihood misses and their retained carrier nuisance columns."""

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

from benchmark_move_selection import decode, encode, fingerprint


def evaluator(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    names = {"_loglik", "_pair_dd", "_likelihood_rows", "_nuisance_columns"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if not any(n.name == "_nuisance_columns" for n in methods):
        loglik = next(n for n in methods if n.name == "_loglik")
        start = next(i for i, n in enumerate(loglik.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "grp_ids")
        end = next(i for i, n in enumerate(loglik.body) if isinstance(n, ast.Try))
        block = loglik.body[start:end]
        assert ast.unparse(block[-1].body[-1]) == "A = np.hstack([A, C])"
        # Clone the original block: the full likelihood body remains untouched.
        block = ast.parse(ast.unparse(ast.Module(body=block, type_ignores=[]))).body
        block[-1].body.pop()
        helper = ast.parse("def _nuisance_columns(self, prob, iscp, pi_kept, sig):\n    b = sig").body[0]
        helper.body += block + ast.parse("return (C if G else None), G").body
        methods.append(helper)
    reference = ast.parse("class Reference:\n    pass").body[0]
    reference.body = methods
    canon = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_canon_basin_id")
    constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "_LOG2PI")
    epoch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EpochProblem")
    namespace = dict(np=np, math=math)
    module = ast.fix_missing_locations(ast.Module(body=[constant, canon, epoch, reference], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    owner = namespace["Reference"]()
    owner._cluster_state = lambda H: owner.replay_state
    owner.epoch_type = namespace["EpochProblem"]
    return owner


def exact(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    x, y = np.asarray(a), np.asarray(b)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def load_records(path, source):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    epoch = evaluator(source).epoch_type
    for row in manifest["records"]:
        file = path.parent/row["file"]
        if fingerprint(file) != row["fingerprint"]:
            raise ValueError("observed nuisance inputs changed")
        with np.load(file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        prob = SimpleNamespace(**{k: v for k, v in arrays.items() if not k.startswith("capture_")})
        epoch.set_pairs(prob, [(tuple(g), ref, sat) for g, ref, sat in row["pairs"]])
        visits = []
        for i, call in enumerate(row["calls"]):
            state = (dict(pos=arrays[f"capture_pos_{i}"], P=arrays[f"capture_prior_{i}"])
                     if call["hasState"] else None)
            columns = (arrays[f"capture_C_{i}"] if call["groups"] else None, call["groups"])
            visits.append(dict(H=decode(call["H"]), state=state, columns=columns,
                iscp=arrays[f"capture_iscp_{i}"], pi=arrays[f"capture_pi_{i}"],
                sig=arrays[f"capture_sig_{i}"], ll=arrays[f"capture_ll_{i}"]))
        records.append((prob, SimpleNamespace(**row["config"]), visits))
    if not records or not all(r[2] for r in records):
        raise ValueError("actual likelihood misses required")
    return manifest, records


def invoke(owner, record, stage):
    prob, owner.cfg, visits = record
    if stage == "columns":
        return [owner._nuisance_columns(prob, v["iscp"], v["pi"], v["sig"]) for v in visits]
    result = []
    for visit in visits:
        owner.replay_state = visit["state"]
        result.append(owner._loglik(visit["H"], prob, {}))
    return result


def input_digest(record):
    prob, cfg, visits = record
    digest = hashlib.sha256()
    digest.update(json.dumps(vars(cfg), sort_keys=True).encode())
    digest.update(repr(prob.pairs).encode())
    def array(value):
        digest.update(str(value.dtype).encode()+str(value.shape).encode()+value.tobytes())
    for key, value in vars(prob).items():
        if isinstance(value, np.ndarray):
            digest.update(key.encode())
            array(value)
    for v in visits:
        digest.update(json.dumps(encode(v["H"])).encode())
        for name in ["iscp", "pi", "sig", "ll"]:
            array(v[name])
        if v["state"] is not None:
            array(v["state"]["pos"])
            array(v["state"]["P"])
        if v["columns"][0] is not None:
            array(v["columns"][0])
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["manifest", "baseline", "candidate", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.pairs <= 100:
        parser.error("new output and 1..100 pairs required")
    manifest, records = load_records(args.manifest, args.baseline)
    if fingerprint(args.baseline) != manifest["runtime"]:
        parser.error("baseline differs from observed runtime")
    paths = [Path(__file__), Path(__file__).with_name("benchmark_move_selection.py"),
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    owners = dict(baseline=evaluator(args.baseline), candidate=evaluator(args.candidate))
    for record in records:
        before = input_digest(record)
        for owner in owners.values():
            assert all(exact(x, v["columns"]) for x, v in zip(invoke(owner, record, "columns"), record[2]))
            assert all(exact(np.asarray(x), v["ll"]) for x, v in zip(invoke(owner, record, "likelihood"), record[2]))
            assert before == input_digest(record)
    stages = {}
    for stage in ["columns", "likelihood"]:
        trials = []
        for trial in range(-2, args.pairs):
            row = dict(trial=trial, epochOrders=[])
            times = {label: [] for label in owners}
            for epoch, record in enumerate(records):
                order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
                row["epochOrders"].append(order)
                for label in order:
                    begin = time.perf_counter_ns()
                    result = invoke(owners[label], record, stage)
                    times[label].append((time.perf_counter_ns()-begin)/1e6)
                    del result
            for label in owners:
                row[label+"EpochMs"], row[label+"Ms"] = times[label], sum(times[label])
            row["reductionPercent"] = 100*(1-row["candidateMs"]/row["baselineMs"])
            trials.append(row)
        measured = [r for r in trials if r["trial"] >= 0]
        stages[stage] = dict(trials=trials,
            baselineMedianMs=statistics.median(r["baselineMs"] for r in measured),
            candidateMedianMs=statistics.median(r["candidateMs"] for r in measured),
            medianReductionPercent=statistics.median(r["reductionPercent"] for r in measured),
            improvedPairs=sum(r["reductionPercent"] > 0 for r in measured))
    if sources != [fingerprint(p) for p in paths]:
        raise RuntimeError("source changed during timing")
    output = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"], originalInputs=manifest["inputs"],
        python=platform.python_version(), numpy=np.__version__, epochs=len(records),
        cacheMisses=sum(len(r[2]) for r in records), stages=stages,
        scope="Actual scalar likelihood cache misses in original epoch/call order/count. Columns includes CP indices, group IDs, zero allocation and scaled column writes; retained input selection is excluded. Likelihood also includes canonical ID, DD and row preparation, robust solve and mixture evaluation with fresh caches and recorded resolved cluster states; live cluster lookup is excluded. Two warmups excluded; adjacent epoch windows alternate first policy with epoch/trial. Loading/decode/validation excluded. No synthetic/expanded calls or persistent H cache. Not full-filter, full-route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({s: {k: v[k] for k in ["baselineMedianMs", "candidateMedianMs", "medianReductionPercent", "improvedPairs"]}
                      for s, v in stages.items()}))


if __name__ == "__main__":
    main()
