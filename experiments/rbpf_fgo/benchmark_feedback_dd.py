"""Replay actual feedback policy with recorded graph-state transitions.

GTSAM graph operations are replaced by captured transitions of the fields
read by feedback; transition playback is included in timing. This measures
policy replay, not live feedback/graph or full-filter latency.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
from pathlib import Path
import platform
import statistics
import struct
import time
from types import SimpleNamespace

import numpy as np

from benchmark_move_selection import fingerprint


def pack(value):
    if isinstance(value, np.ndarray):
        return ["array", value.dtype.str, list(value.shape), base64.b64encode(value.tobytes()).decode()]
    if isinstance(value, np.generic):
        return ["scalar", value.dtype.str, base64.b64encode(value.tobytes()).decode()]
    if isinstance(value, float):
        return ["float", struct.pack("d", value).hex()]
    if isinstance(value, dict):
        return ["dict", [[pack(k), pack(v)] for k, v in value.items()]]
    if isinstance(value, (tuple, list, set)):
        items = [pack(v) for v in value]
        return [type(value).__name__, sorted(items, key=lambda v: json.dumps(v)) if isinstance(value, set) else items]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported captured type {type(value).__name__}")


def unpack(value):
    if not isinstance(value, list):
        return value
    tag, *parts = value
    if tag == "array":
        dtype, shape, raw = parts
        return np.frombuffer(base64.b64decode(raw), dtype=dtype).copy().reshape(shape)
    if tag == "scalar":
        return np.frombuffer(base64.b64decode(parts[1]), dtype=parts[0])[0]
    if tag == "float":
        return struct.unpack("d", bytes.fromhex(parts[0]))[0]
    if tag == "dict":
        return {unpack(k): unpack(v) for k, v in parts[0]}
    return {"tuple": tuple, "list": list, "set": set}[tag](unpack(v) for v in parts[0])


def feedback_fields(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_feedback")
    def names(base):
        return sorted({n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)
                       and isinstance(n.value, ast.Name) and n.value.id == base})
    fields = [n for n in names("self") if n not in {"cfg", "particles", "_pair_dd"}]
    return fields, names("cfg")


def owner_state(owner, fields):
    return {name: getattr(owner, name) for name in fields}


def graph_state(rtk):
    return dict(committed=dict(rtk._committed), ambKeys=list(rtk.amb_keys),
                x=np.array(rtk.nav.x, copy=True), na=rtk.nav.na)


def evaluator(source, legacy=False):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_feedback")
    accepts = any(a.arg == "dd_all" for a in method.args.args)
    if legacy and accepts:
        assert method.args.args[-1].arg == "dd_all"
        method.args.args.pop()
        method.args.defaults.pop()
        index = next(i for i, n in enumerate(method.body) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "dd_all is None")
        method.body[index:index+1] = method.body[index].body
        accepts = False
    if not legacy:
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step")
        calls = [n for n in ast.walk(step) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "_feedback"]
        assert len(calls) == 1
        supplied = [k for k in calls[0].keywords if k.arg == "dd_all"]
        assert bool(supplied) == accepts
        if supplied:
            assert ast.unparse(supplied[0].value) == "dd_all"
    reduced = ast.parse("class Reference:\n    pass").body[0]
    reduced.body = [method, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_pair_dd")]
    epoch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EpochProblem")
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[epoch, reduced], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["Reference"](), namespace["EpochProblem"], accepts


class GraphReplay:
    def __init__(self, state, events, ib, validate):
        self._committed, self.amb_keys = {}, {}
        self.nav = SimpleNamespace()
        self.apply(state)
        self.events, self.ib, self.validate, self.index = events, ib, validate, 0

    def apply(self, state):
        # Feedback keeps a local alias to _committed, so update it in place.
        self._committed.clear()
        self._committed.update(state["committed"])
        self.amb_keys.clear()
        self.amb_keys.update(dict.fromkeys(state["ambKeys"]))
        self.nav.x, self.nav.na = state["x"].copy(), state["na"]

    def IB(self, sat, frequency, na):
        assert na == self.nav.na
        return self.ib[(int(sat), int(frequency))]

    def transition(self, name, args):
        event = self.events[self.index]
        if self.validate:
            assert event["name"] == name and pack(args) == event["args"]
            assert pack(graph_state(self)) == event["before"]
        self.apply(event["after"])
        self.index += 1
        return event["return"]

    def _release_ambiguity(self, sat, frequency, reason=""):
        return self.transition("release", (sat, frequency, reason))

    def _inject_hold(self, xa, newly):
        return self.transition("inject", (xa, newly))


def load_records(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in manifest["records"]:
        file = path.parent/row["file"]
        if fingerprint(file) != row["fingerprint"]:
            raise ValueError("feedback record changed")
        records.append(json.loads(file.read_text(encoding="utf-8")))
    if not records:
        raise ValueError("actual feedback records required")
    return manifest, records


def prepare(evaluation, record, validate=False):
    owner, epoch, accepts = evaluation
    inputs = unpack(record["inputs"])
    owner.cfg = SimpleNamespace(**inputs["config"])
    owner.particles = [SimpleNamespace(H=H) for H in inputs["hypotheses"]]
    for name, value in inputs["owner"].items():
        setattr(owner, name, value)
    prob = SimpleNamespace(**inputs["problem"])
    epoch.set_pairs(prob, prob.pairs)
    events = []
    for event in record["events"]:
        events.append(dict(event, after=unpack(event["after"]), **{"return": unpack(event["return"])}))
    rtk = GraphReplay(inputs["graph"], events, inputs["ib"], validate)
    kwargs = inputs["kwargs"]
    if accepts:
        kwargs["dd_all"] = inputs["providedDd"]
    return (rtk, prob, inputs["weights"], inputs["ddMap"], inputs["usedPairs"]), kwargs, inputs


def invoke(evaluation, prepared):
    args, kwargs, _ = prepared
    return evaluation[0]._feedback(*args, **kwargs)


def outcome(owner, rtk, fields, value):
    return pack(dict(result=value, owner=owner_state(owner, fields), graph=graph_state(rtk)))


def validate_case(evaluation, record, fields):
    prepared = prepare(evaluation, record, True)
    before = pack([p.H for p in evaluation[0].particles]), pack(prepared[2]["providedDd"]), pack(prepared[0][2])
    value = invoke(evaluation, prepared)
    rtk = prepared[0][0]
    assert outcome(evaluation[0], rtk, fields, value) == record["expected"]
    assert rtk.index == len(rtk.events)
    assert before == (pack([p.H for p in evaluation[0].particles]), pack(prepared[2]["providedDd"]), pack(prepared[0][2]))


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
    fields, _ = feedback_fields(args.baseline)
    owners = {label: evaluator(source) for label, source in [("baseline", args.baseline), ("candidate", args.candidate)]}
    counts = {label: 0 for label in owners}
    for label, evaluation in owners.items():
        original = evaluation[0]._pair_dd
        def counted(H, prob, original=original, label=label):
            counts[label] += 1
            return original(H, prob)
        evaluation[0]._pair_dd = counted
        for record in records:
            validate_case(evaluation, record, fields)
        evaluation[0]._pair_dd = original
    trials = []
    for trial in range(-2, args.pairs):
        row, times = dict(trial=trial, epochOrders=[]), {label: [] for label in owners}
        for epoch, record in enumerate(records):
            order = ["baseline", "candidate"] if (trial+epoch) % 2 == 0 else ["candidate", "baseline"]
            row["epochOrders"].append(order)
            for label in order:
                prepared = prepare(owners[label], record)
                begin = time.perf_counter_ns()
                value = invoke(owners[label], prepared)
                times[label].append((time.perf_counter_ns()-begin)/1e6)
                assert outcome(owners[label][0], prepared[0][0], fields, value) == record["expected"]
                assert prepared[0][0].index == len(record["events"])
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
    result = dict(schemaVersion=1, sources=sources, captureRuntime=manifest["runtime"], originalInputs=manifest["inputs"],
                  epochs=len(records), particleVisits=sum(r["particles"] for r in manifest["records"]),
                  python=platform.python_version(), numpy=np.__version__, ddCalls=counts,
                  graphEvents=sum(len(r["events"]) for r in records), trials=trials, summary=summary,
                  scope="Actual feedback-policy calls in original epoch/particle/pair order/count. GTSAM graph operations replaced with recorded transitions of feedback-visible fields; transition playback included. Fresh state restoration, existing component DD preparation, imports/decode/validation excluded from clocks. Two warmups excluded; adjacent epoch windows alternate first policy by epoch/trial. No synthetic/expanded hypotheses or persistent DD cache. Not live feedback/GTSAM, full-filter, full-route or GPU latency.")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, ddCalls=counts, graphEvents=result["graphEvents"])))


if __name__ == "__main__":
    main()
