"""Optional host wall spans around the literal score statements,using live globals."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import time
from types import FunctionType, MethodType

import rbpf_likelihood_batch as original
from rbpf_register_lu import RegisterLUScorer

BATCH_SHA256 = "7f35ae5b466145f07aa28a3de5b18f95121606b3a27a7bc85b2f9810c88bd663"
PHASES = ("entry", "dedup", "support", "fixed_pack", "dd", "parameters", "validate", "native", "finish")
_EVENTS = "__rbpf_host_events_20261017"
_MARK = "__rbpf_host_mark_20261017"
_PUBLISH = "__rbpf_host_publish_20261017"


def _mark(events, name):
    events.append((name, time.perf_counter_ns()))


def _publish(scorer, events):
    # Publication is outside the recorded span but inside the complete score call.
    scorer.last_profile = tuple(events)


def _source():
    source = Path(original.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != BATCH_SHA256:
        raise ValueError("Frozen likelihood caller changed")
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BatchScorer")
    return deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "score"))


def _marker(name):
    return ast.Expr(ast.Call(ast.Name(_MARK, ast.Load()),
        [ast.Name(_EVENTS, ast.Load()), ast.Constant(name)], []))


def _boundary(statement):
    if isinstance(statement, ast.Assign):
        target = ast.unparse(statement.targets[0])
        return {"(keys, unique, scatter)": "dedup", "(m, nb)": "support", "rows": "fixed_pack",
            "dd": "dd", "parameters": "parameters", "config": "validate", "code": "native"}.get(target)
    if isinstance(statement, ast.If) and ast.unparse(statement.test) == "code":
        return "finish"
    return None


def _instrument(method):
    names = {n.id for n in ast.walk(method) if isinstance(n, ast.Name)}
    if names.intersection((_EVENTS, _MARK, _PUBLISH)):
        raise ValueError("Diagnostic names collide with original locals")
    body, found = [], []
    for statement in deepcopy(method.body):
        phase = _boundary(statement)
        if phase is not None:
            body.append(_marker(phase))
            found.append(phase)
        body.append(statement)
    if tuple(found) != PHASES[1:]:
        raise ValueError("Frozen host phase boundaries changed")
    # Keep the docstring out of the try body so restoration preserves its AST.
    doc = body.pop(0)
    wrapped = ast.Try(body=body, handlers=[], orelse=[], finalbody=[_marker("end"),
        ast.Expr(ast.Call(ast.Name(_PUBLISH, ast.Load()),
            [ast.Name("self", ast.Load()), ast.Name(_EVENTS, ast.Load())], []))])
    method.body = [doc, ast.Assign([ast.Name(_EVENTS, ast.Store())], ast.List([], ast.Load())),
        _marker("entry"), wrapped]
    return method


def restored_ast():
    expected = _source()
    method = _instrument(deepcopy(expected))
    wrapped = method.body[-1]
    method.body = [method.body[0]] + [n for n in wrapped.body
        if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name) and n.value.func.id == _MARK)]
    return ast.dump(method, include_attributes=False), ast.dump(expected, include_attributes=False)


def _build(measure):
    method = _source()
    if not measure:
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            str(Path(__file__)), "exec"), scope)
        result = FunctionType(scope["score"].__code__, original.BatchScorer.score.__globals__,
            "score", original.BatchScorer.score.__defaults__)
    else:
        method = _instrument(method)
        factory = ast.FunctionDef(name="_factory",
            args=ast.arguments(posonlyargs=[], args=[ast.arg(_MARK), ast.arg(_PUBLISH)],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
            body=[method, ast.Return(ast.Name("score", ast.Load()))], decorator_list=[])
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
            str(Path(__file__)), "exec"), scope)
        # Helpers are closure cells;numeric names resolve in the original live
        # module dict without inserting or replacing any of its bindings.
        maker = FunctionType(scope["_factory"].__code__, original.BatchScorer.score.__globals__)
        result = maker(_mark, _publish)
    result.__kwdefaults__ = original.BatchScorer.score.__kwdefaults__
    return result


_LITERAL, _PROFILED = _build(False), _build(True)
assert restored_ast()[0] == restored_ast()[1]
assert _LITERAL.__code__.co_code == original.BatchScorer.score.__code__.co_code


def spans(events):
    """Partition elapsed host wall time,including waits and tracing overhead."""
    if not events or events[0][0] != "entry" or events[-1][0] != "end":
        raise ValueError("Incomplete host trace")
    names = [n for n, _ in events[:-1]]
    if names != list(PHASES[:len(names)]):
        raise ValueError("Unexpected host trace topology")
    if any(type(t) is not int for _, t in events) or any(b[1] < a[1] for a, b in zip(events, events[1:])):
        raise ValueError("Invalid host clock sequence")
    parts = {name: 0 for name in PHASES}
    for (name, start), (_, end) in zip(events, events[1:]):
        parts[name] = end - start
    total = events[-1][1] - events[0][1]
    if sum(parts.values()) != total:
        raise ValueError("Host spans do not partition total")
    return dict(total=total, **parts)


class HostPhaseScorer(RegisterLUScorer):
    """Use the frozen corrected register DLL;no new CUDA implementation."""

    def __init__(self, dll, capacity=512, *, measure=True):
        super().__init__(dll, capacity, kernel="register")
        self.last_profile = None
        self.score = MethodType(_PROFILED if measure else _LITERAL, self)
