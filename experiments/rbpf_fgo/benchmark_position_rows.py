"""Replay actual conditional-position LS calls and their row preparation."""

from __future__ import annotations

import argparse
import ast
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
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_ls_offset", "_position_rows"}]
    if not any(n.name == "_position_rows" for n in methods):
        ls = next(n for n in methods if n.name == "_ls_offset")
        start = next(i for i, n in enumerate(ls.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Tuple) and ast.unparse(n.targets[0].elts[0]) == "rows_v")
        end = next(i for i, n in enumerate(ls.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "centre")
        helper = ast.parse("def _position_rows(self, prob, dd, sb):\n    pass").body[0]
        helper.body = ls.body[start:end]+ast.parse(
            "return v, sig, J, np.asarray(grp, dtype=int), len(grp_index)").body
        methods.append(helper)
    reference = ast.parse("class Reference:\n    pass").body[0]
    reference.body = methods
    namespace = dict(np=np, math=math)
    module = ast.fix_missing_locations(ast.Module(body=[reference], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    owner = namespace["Reference"]()
    owner._cluster_state = lambda H: owner.replay_state
    return owner


def exact(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    x, y = np.asarray(a), np.asarray(b)
    return x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def load_records(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in manifest["records"]:
        file = path.parent/row["file"]
        if fingerprint(file) != row["fingerprint"]:
            raise ValueError("captured position inputs changed")
        with np.load(file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        prob = SimpleNamespace(**{k: v for k, v in arrays.items() if not k.startswith("capture_")})
        prob.pairs = [(tuple(g), ref, sat) for g, ref, sat in row["pairs"]]
        visits = []
        for i, call in enumerate(row["calls"]):
            state = (dict(pos=arrays[f"capture_pos_{i}"], P=arrays[f"capture_prior_{i}"])
                     if call["hasState"] else None)
            result = ((arrays[f"capture_sol_{i}"], float(arrays[f"capture_info_{i}"]), arrays[f"capture_cov_{i}"])
                      if call["hasResult"] else None)
            visits.append((decode(call["H"]), arrays[f"capture_dd_{i}"], state, result))
        records.append((prob, SimpleNamespace(**row["config"]), visits))
    if not records:
        raise ValueError("actual LS calls required")
    return manifest, records


def invoke(owner, record, stage):
    prob, owner.cfg, visits = record
    if stage == "rows":
        return [owner._position_rows(prob, dd, math.sqrt(owner.cfg.beta)) for _, dd, _, _ in visits]
    result = []
    for H, dd, state, _ in visits:
        owner.replay_state = state
        result.append(owner._ls_offset(H, prob, dd))
    return result


def input_digest(record):
    import hashlib
    prob, _, visits = record
    digest = hashlib.sha256()
    for name, value in vars(prob).items():
        if isinstance(value, np.ndarray):
            digest.update(name.encode()+value.tobytes())
    for H, dd, state, _ in visits:
        digest.update(json.dumps(encode(H)).encode()+dd.tobytes())
        if state is not None:
            digest.update(state["pos"].tobytes()+state["P"].tobytes())
    return digest.hexdigest()


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
    paths = [Path(__file__), Path(__file__).with_name("benchmark_move_selection.py"),
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    owners = dict(baseline=evaluator(args.baseline), candidate=evaluator(args.candidate))
    for record in records:
        before = input_digest(record)
        a, b = invoke(owners["baseline"], record, "rows"), invoke(owners["candidate"], record, "rows")
        assert all(exact(x, y) for x, y in zip(a, b))
        for owner in owners.values():
            result = invoke(owner, record, "ls")
            assert all(exact(x, v[3]) for x, v in zip(result, record[2]))
            assert input_digest(record) == before
    stages = {}
    for stage in ["rows", "ls"]:
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
                  lsCalls=sum(len(r[2]) for r in records), stages=stages,
                  scope="Actual conditional-position LS calls, original epoch/call order/count. Rows includes finite-CP selection, residual/sigma/J copies, group order and sqrt(beta). LS also includes robust solve, information and covariance with recorded resolved cluster states; live cluster lookup and conditional cache/outer wrapper are excluded. Two warmups excluded. Adjacent epoch windows alternate first policy with epoch/trial. Loading/decode/validation excluded. No synthetic/expanded calls or persistent H cache. Not full-filter, full-route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({s: {k: v[k] for k in ["baselineMedianMs", "candidateMedianMs", "medianReductionPercent", "improvedPairs"]}
                      for s, v in stages.items()}))


if __name__ == "__main__":
    main()
