"""Replay actual scalar likelihood cache misses and their row preparation."""

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

from benchmark_dd_layout import epoch_type
from benchmark_move_selection import decode, encode, fingerprint


def evaluator(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    names = {"_loglik", "_pair_dd", "_likelihood_rows"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if not any(n.name == "_likelihood_rows" for n in methods):
        loglik = next(n for n in methods if n.name == "_loglik")
        start = next(i for i, n in enumerate(loglik.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "m")
        end = next(i for i, n in enumerate(loglik.body) if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "np.any(uniform)")
        helper = ast.parse("def _likelihood_rows(self, prob, dd):\n    cfg = self.cfg").body[0]
        helper.body += loglik.body[start:end]
        helper.body += ast.parse("return v, sig, uniform, const_ll").body
        methods.append(helper)
    reference = ast.parse("class Reference:\n    pass").body[0]
    reference.body = methods
    canon = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_canon_basin_id")
    constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_LOG2PI")
    module = ast.fix_missing_locations(ast.Module(body=[constant, canon, reference], type_ignores=[]))
    namespace = dict(np=np, math=math)
    exec(compile(module, str(source), "exec"), namespace)
    owner = namespace["Reference"]()
    owner._cluster_state = lambda H: owner.replay_state
    return owner


def load_records(path, source):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["records"]:
        file = path.parent/record["file"]
        if fingerprint(file) != record["fingerprint"]:
            raise ValueError("recorded input changed")
        with np.load(file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        prob = SimpleNamespace(**{k: v for k, v in arrays.items()
                                  if not k.startswith("capture_")})
        epoch_type(source).set_pairs(prob, [(tuple(g), ref, sat) for g, ref, sat in record["pairs"]])
        visits = []
        for i, H in enumerate(record["hypotheses"]):
            state = (dict(pos=arrays["capture_pos"][i], P=arrays["capture_P"][i])
                     if arrays["capture_has_state"][i] else None)
            visits.append((decode(H), arrays["capture_dd"][i], state, arrays["capture_ll"][i]))
        records.append((prob, SimpleNamespace(**record["config"]), visits))
    if not records:
        raise ValueError("actual cache misses required")
    return manifest, records


def invoke(owner, prob, cfg, visits, stage):
    owner.cfg = cfg
    if stage == "rows":
        return [owner._likelihood_rows(prob, dd) for _, dd, _, _ in visits]
    result = []
    for H, _, state, _ in visits:
        owner.replay_state = state
        result.append(owner._loglik(H, prob, {}))
    return result


def exact_rows(a, b):
    return all(x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
               for x, y in zip(a[:3], b[:3])) and np.float64(a[3]).tobytes() == np.float64(b[3]).tobytes()


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
    paths = [Path(__file__), here/"benchmark_dd_layout.py", here/"benchmark_dd_lookups.py",
             here/"benchmark_move_pair.py", here/"benchmark_move_selection.py",
             args.baseline, args.candidate, args.manifest]
    sources = [fingerprint(p) for p in paths]
    owners = dict(baseline=evaluator(args.baseline), candidate=evaluator(args.candidate))
    for prob, cfg, visits in records:
        saved = [encode(H) for H, _, _, _ in visits]
        arrays = {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
        old = invoke(owners["baseline"], prob, cfg, visits, "rows")
        new = invoke(owners["candidate"], prob, cfg, visits, "rows")
        assert all(exact_rows(a, b) for a, b in zip(old, new))
        for owner in owners.values():
            values = invoke(owner, prob, cfg, visits, "likelihood")
            assert np.asarray(values).tobytes() == np.asarray([v[3] for v in visits]).tobytes()
            assert saved == [encode(H) for H, _, _, _ in visits]
            assert arrays == {k: v.tobytes() for k, v in vars(prob).items() if isinstance(v, np.ndarray)}
    stages = {}
    for stage in ["rows", "likelihood"]:
        trials = []
        for trial in range(-2, args.pairs):
            row = dict(trial=trial, epochOrders=[])
            times = {label: [] for label in owners}
            for epoch, (prob, cfg, visits) in enumerate(records):
                order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
                row["epochOrders"].append(order)
                for label in order:
                    begin = time.perf_counter_ns()
                    result = invoke(owners[label], prob, cfg, visits, stage)
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
    output = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"],
                  originalInputs=manifest["inputs"], python=platform.python_version(), numpy=np.__version__,
                  epochs=len(records), cacheMisses=sum(len(v) for _, _, v in records), stages=stages,
                  scope="Original scalar cache misses, original epoch/call order per policy. Adjacent epoch windows alternate first policy with trial and epoch. Two warmups per source/stage excluded. Rows include output copies; likelihood includes canonical ID, current DD generation, row preparation and matrix/mixture evaluation with fresh caches and recorded resolved cluster states. Live cluster lookup and input loading/decode/validation excluded. No synthetic or expanded hypotheses. Not live full-filter or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({s: {k: v[k] for k in ["baselineMedianMs", "candidateMedianMs", "medianReductionPercent", "improvedPairs"]}
                      for s, v in stages.items()}))


if __name__ == "__main__":
    main()
