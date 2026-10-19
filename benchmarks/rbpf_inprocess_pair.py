"""Single-thread paired RTK call protocol with independent suspended drivers."""

import ast
from copy import deepcopy
import hashlib
from statistics import median
from types import GeneratorType

from rbpf_epoch_clocks import EpochClocks, summarize

LABELS = ("A", "AA")


class PairError(RuntimeError):
    pass


class Owner:
    def __init__(self):
        self.active = False
        self.offers = {}


class Ticket:
    def __init__(self, owner, label, epoch, phase, function, args, kwargs, context):
        self.owner, self.label, self.epoch, self.phase = owner, label, epoch, phase
        self.function, self.args, self.kwargs, self.context = function, args, kwargs, context
        self.state, self.result = "ready", None

    def execute(self, owner, clock):
        if owner is not self.owner or owner.active or self.state != "ready":
            raise PairError("Foreign,recursive,or already executed ticket")
        if len(clock.records) != self.epoch or clock.active:
            raise PairError("Clock and ticket epochs differ")
        self.state, owner.active = "running", True
        try:
            self.result = clock.call(self.phase, self.function, *self.args, **self.kwargs)
        except BaseException:
            self.state = "failed"
            raise
        else:
            self.state = "done"
            return self.result
        finally:
            owner.active = False

    def require_completed(self):
        if self.state != "done":
            raise PairError("Driver advanced before its original RTK call completed")

    def release(self):
        self.require_completed()
        self.function = self.args = self.kwargs = self.context = self.result = None


class Offerer:
    def __init__(self, owner, label):
        if label not in LABELS or label in owner.offers:
            raise PairError("Duplicate or unknown stream label")
        self.owner, self.label, self.last, self.next_epoch = owner, label, None, 0
        owner.offers[label] = self

    def make(self, epoch, phase, function, args=(), kwargs=None, context=None):
        if self.owner.active or (self.last is not None and self.last.state != "done"):
            raise PairError("Input preparation overlapped an unfinished timed call")
        if type(epoch) is not int or epoch != self.next_epoch or type(phase) is not int or phase not in (1, 2):
            raise PairError("Missing,repeated,or invalid epoch/phase")
        self.last = Ticket(self.owner, self.label, epoch, phase, function, args,
            {} if kwargs is None else dict(kwargs), context)
        self.next_epoch += 1
        return self.last


def paired_driver(source, scope, factory, offer, argv, *, expected_sha256, filename):
    """Keep original driver,forward its one RTK call through a yielded ticket.

    The local class factory changes only instance dispatch. Explicit parser
    tokens replace the same sys.argv tokens so two suspended drivers cannot
    change one another's arguments. Removing all three edits restores the
    complete original main AST,including its original call expression.
    """
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("Frozen driver changed")
    tree = ast.parse(source)
    node = deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    original = deepcopy(node)
    if any(isinstance(n, ast.Name) and n.id.startswith("_pair_") for n in ast.walk(node)):
        raise ValueError("Driver uses a reserved protocol name")
    parsers = [n for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ap.parse_args"]
    if len(parsers) != 1 or parsers[0].args or parsers[0].keywords:
        raise ValueError("Unexpected driver argument parser")
    parsers[0].args = [ast.parse("_pair_argv[1:]", mode="eval").body]
    imports = [i for i, n in enumerate(node.body) if isinstance(n, ast.ImportFrom) and n.module == "rbpf_fgo"
        and any(a.name == "RbpfGtsamRtkTc" and a.asname is None for a in n.names)]
    if len(imports) != 1:
        raise ValueError("Unexpected RTK class import")
    index = imports[0] + 1
    node.body[index:index] = ast.parse("RbpfGtsamRtkTc = _pair_factory(RbpfGtsamRtkTc)").body
    sites = [(n.body, i) for n in ast.walk(node) if isinstance(getattr(n, "body", None), list)
        for i, statement in enumerate(n.body) if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func) == "rtk.process"]
    if len(sites) != 1:
        raise ValueError("Expected one direct main RTK process call")
    body, at = sites[0]
    original_call = deepcopy(body[at])
    if ast.unparse(original_call) != "rtk.process(obs, obsb=obsb)":
        raise ValueError("RTK call arguments changed")
    replacement = ast.parse("_pair_ticket = _pair_offer(ne,rtk,obs,obsb)\nyield _pair_ticket\n_pair_ticket.require_completed()").body
    body[at:at + 1] = replacement
    restored = deepcopy(node)
    yielded = [(n.body, i) for n in ast.walk(restored) if isinstance(getattr(n, "body", None), list)
        for i, statement in enumerate(n.body) if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Yield)]
    if len(yielded) != 1:
        raise ValueError("Unexpected pre-existing generator protocol")
    target, pos = yielded[0]
    target[pos - 1:pos + 2] = [original_call]
    restored.body.pop(index)
    parser = next(n for n in ast.walk(restored) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ap.parse_args")
    parser.args = []
    if ast.dump(restored, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError("Original complete driver AST not restored")
    globals_ = dict(scope, _pair_factory=factory, _pair_offer=offer, _pair_argv=tuple(argv))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), filename,
        "exec", dont_inherit=True), globals_)
    return globals_["main"], dict(original_driver_AST_restored_exact=True,
        original_driver_sha256=expected_sha256, explicit_original_parser_tokens=True,
        original_RTK_call_evaluated_once_by_owned_ticket=True,
        class_factory_only_after_original_environment_and_imports=True)


def drive(streams, owner, *, pair_index, epochs, after=None, clocks=None):
    """Prepare both inputs,run both clocks,then advance both original drivers."""
    if (set(streams) != set(LABELS) or set(owner.offers) != set(LABELS)
            or streams["A"] is streams["AA"] or any(type(g) is not GeneratorType for g in streams.values())
            or type(pair_index) is not int or pair_index < 0 or type(epochs) is not int or epochs < 1):
        raise PairError("Two independent generator streams and a valid schedule required")
    clocks = {label: EpochClocks() for label in LABELS} if clocks is None else clocks
    if set(clocks) != set(LABELS) or clocks["A"] is clocks["AA"]:
        raise PairError("Independent clocks required")
    events, completed, tickets = [], {}, {}
    try:
        for epoch in range(epochs):
            order = LABELS[::-1] if (epoch + pair_index) % 2 else LABELS
            prepare = order[::-1]
            for label in prepare:
                try:
                    ticket = next(streams[label])
                except StopIteration as error:
                    raise PairError("Driver ended before all declared epochs") from error
                if (type(ticket) is not Ticket or ticket.owner is not owner or ticket.label != label
                        or ticket.epoch != epoch or ticket.state != "ready"):
                    raise PairError("Foreign or out-of-sequence input ticket")
                tickets[label] = ticket
                events.append(dict(event="ready", epoch=epoch, label=label, phase=ticket.phase))
            if tickets["A"].phase != tickets["AA"].phase:
                raise PairError("Paired RTK phases differ")
            phase = tickets["A"].phase
            for label in order:
                events.append(dict(event="run", epoch=epoch, label=label, phase=phase))
                tickets[label].execute(owner, clocks[label])
                events.append(dict(event="done", epoch=epoch, label=label, phase=phase))
            if after is not None:
                if owner.active or any(c.active for c in clocks.values()):
                    raise PairError("Post-clock capture entered an active estimator")
                after(epoch, tickets, clocks)
            for label in prepare:
                tickets[label].release()
                events.append(dict(event="advance", epoch=epoch, label=label, phase=phase))
        for label in LABELS:
            try:
                next(streams[label])
            except StopIteration as finished:
                if type(finished.value) is not int or finished.value != 0:
                    raise PairError("Original driver did not complete successfully")
                completed[label] = finished.value
            else:
                raise PairError("Driver produced an undeclared extra epoch")
    finally:
        for stream in streams.values():
            stream.close()
    return dict(records={label: list(c.records) for label, c in clocks.items()},
        events=events, completed=completed, pair_index=pair_index,
        prepares_outside_clocks=True, captures_only_after_both_clocks=True)


def evaluate(pairs):
    if len(pairs) != 6 or [p["pair"] for p in pairs] != list(range(6)) or [p["warm"] for p in pairs] != [True] + [False] * 5:
        raise PairError("Exactly one warm pair then five primary pairs required")
    values = []
    for pair in pairs:
        by_label = {}
        for label in LABELS:
            rows = pair["records"][label]
            if len(rows) != 400 or [r["phase_before"] for r in rows] != [1] * 200 + [2] * 200:
                raise PairError("Complete original two-phase RTK prefix required")
            by_label[label] = summarize(rows)
            if any(by_label[label][phase]["wall_ns"] <= 0 for phase in ("all", "1", "2")):
                raise PairError("Positive complete phase wall spans required")
        values.append(by_label)
    metrics = {}
    for phase in ("all", "1", "2"):
        differences = [100.0 * (v["A"][phase]["wall_ns"] - v["AA"][phase]["wall_ns"]) / v["A"][phase]["wall_ns"] for v in values[1:]]
        absolute = list(map(abs, differences))
        metrics[phase] = dict(paired_percent=differences, absolute_median_percent=median(absolute),
            absolute_rounds_under5=sum(v < 5 for v in absolute),
            pass_gate=median(absolute) < 5 and sum(v < 5 for v in absolute) >= 4)
    return dict(metrics=metrics, calibration_pass=all(v["pass_gate"] for v in metrics.values()),
        original_optimization_gates_unchanged=True, optimization_speedup_established=False,
        whole_continuous_RTK_throughput_established=False, first_calibration_set_only=True)
