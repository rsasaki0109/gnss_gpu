"""Sparse direct wall boundaries in the unchanged D236 complete PF caller."""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import time
from types import FunctionType

import rbpf_component_mass as component
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256
from rbpf_move_batch import MoveBatchStep
from rbpf_move_reuse import MoveReuseStep
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_native_canonical import NativeCanonicalStep

_CLOCK = time.perf_counter_ns
_PINS = {
    "rbpf_move_batch.py": "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea",
    "rbpf_move_reuse.py": "9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612",
    "rbpf_move_key_memo.py": "ddcc139b41212bfe813d552f6921b2914b6452571be410e2783a65f0d9cf9632",
    "rbpf_native_canonical.py": "f177e3247617cca70e4aa0c4369f1856dba979703f478d62c1f05cd814c63bd2",
}
for _name, _digest in _PINS.items():
    if (
        hashlib.sha256(Path(__file__).with_name(_name).read_bytes()).hexdigest()
        != _digest
    ):
        raise ValueError("Frozen move helper changed: " + _name)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _method(cls, name):
    tree = ast.parse(Path(__file__).with_name(cls.__module__ + ".py").read_bytes())
    klass = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__
    )
    return deepcopy(
        next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )


def _factory(node, parameter, scope):
    factory = ast.parse(f"def _factory({parameter}): pass").body[0]
    factory.body = [node, ast.Return(ast.Name(node.name, ast.Load()))]
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
            str(Path(__file__)),
            "exec",
        ),
        namespace,
    )
    return FunctionType(namespace["_factory"].__code__, scope)


def _marker(label):
    return ast.parse(f"_mark({label!r})").body[0]


def _strip_marks(nodes):
    return [
        deepcopy(n)
        for n in nodes
        if not (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "_mark"
        )
    ]


def _step_ast():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError("Frozen scalar source changed")
    tree = ast.parse(source)
    klass = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo"
    )
    step = deepcopy(
        next(
            n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "step"
        )
    )
    # Reproduce exactly the two existing D235 substitutions.
    (i,) = [
        i
        for i, n in enumerate(step.body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "new_states"
    ]
    step.body[i : i + 2] = ast.parse(
        "new_states = _rbpf_batch_moves(prob, H_cons, cand_H, spawn_H, cache, ll_burst)"
    ).body
    (block,) = [
        n
        for n in step.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "dd_map is not None"
    ]
    block.body[:3] = ast.parse(
        "pair_mass = _rbpf_component_mass(self, prob, w, dd_map)"
    ).body
    step.name = "_component_mass_step"
    baseline = deepcopy(step)
    # Identify boundaries by literal statements, not line numbers.
    labels = {
        "prob = self._extract(rtk, ctx)": "extract",
        "self._maintenance(rtk, support_sf, ep)": "maintenance_spawn",
        "ddpr_got = self._ddpr_offset(prob)": "ddpr",
        "cache = {}": "burst_shadow",
        "K = prob.cands.shape[1] if prob.cands is not None else 0": "adopt_candidates",
        "new_states = _rbpf_batch_moves(prob, H_cons, cand_H, spawn_H, cache, ll_burst)": "move_adapters",
        "for p, (H, tag) in zip(self.particles, new_states):": "commit",
        "logw = np.array([p.logw for p in self.particles])": "normalize_basins",
        "top = []": "conditional_positions",
        "dd_map = self._pair_dd(map_H, prob) if map_H else None": "fix_witness",
        "if cfg.fb_enable:": "feedback",
        "picks, ess = self._maybe_resample(logw)": "resample",
        "self.last = out": "publish",
    }
    body = []
    used = []
    for node in step.body:
        key = ast.unparse(node).splitlines()[0]
        if key in labels:
            body.append(_marker(labels[key]))
            used.append(labels[key])
        body.append(node)
    assert len(used) == len(labels) == 13 and len(set(used)) == 13
    guarded = ast.parse('try: pass\r\nfinally: _mark("body_exit")').body[0]
    guarded.body = body
    step.body = [_marker("body_entry"), guarded]
    restored = deepcopy(step)
    restored.body = _strip_marks(restored.body[1].body)
    assert _dump(restored) == _dump(baseline)
    return (
        step,
        baseline,
        dict(
            method="D235 complete step", restored_AST_equal=True, sparse_boundaries=15
        ),
    )


def _moves_ast():
    node = _method(MoveBatchStep, "_moves")
    baseline = deepcopy(node)
    labels = {
        "for particle in runtime.particles:": "move_staging",
        "requested = [hypotheses[i] for i in pending]": "move_scores",
        "new_states = []": "move_choice",
    }
    body = []
    used = []
    for n in node.body:
        key = ast.unparse(n).splitlines()[0]
        if key in labels:
            body.append(_marker(labels[key]))
            used.append(labels[key])
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.last":
            body.append(_marker("move_metadata"))
            used.append("move_metadata")
        body.append(n)
    node.body = body
    assert len(used) == 4
    restored = deepcopy(node)
    restored.body = _strip_marks(restored.body)
    assert _dump(restored) == _dump(baseline)
    return node, dict(
        method="MoveBatchStep._moves", restored_AST_equal=True, sparse_boundaries=4
    )


def _adapter(cls, callee):
    node = _method(cls, "_moves")
    baseline = deepcopy(node)
    (call,) = [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == callee
    ]
    call.func = ast.Name("_inner", ast.Load())
    restored = deepcopy(node)
    (call,) = [
        n
        for n in ast.walk(restored)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "_inner"
    ]
    call.func = ast.parse(callee, mode="eval").body
    assert _dump(restored) == _dump(baseline)
    return _factory(node, "_inner", cls._moves.__globals__), dict(
        method=cls.__name__ + "._moves", restored_AST_equal=True
    )


_STEP_AST, _STEP_BASE, _STEP_PROOF = _step_ast()
_MOVES_AST, _MOVES_PROOF = _moves_ast()
_RAW_FACTORY = _factory(_MOVES_AST, "_mark", MoveBatchStep._moves.__globals__)
_ADAPTERS = [
    _adapter(cls, callee)
    for cls, callee in (
        (MoveReuseStep, "MoveBatchStep._moves"),
        (MoveKeyMemoStep, "MoveReuseStep._moves"),
        (NativeCanonicalStep, "MoveKeyMemoStep._moves"),
    )
]
AST_PROOFS = [_STEP_PROOF, _MOVES_PROOF] + [proof for _, proof in _ADAPTERS]
_HELPERS = tuple(
    (cls, cls._moves, cls._moves.__code__)
    for cls in (MoveBatchStep, MoveReuseStep, MoveKeyMemoStep, NativeCanonicalStep)
)


def _cell(value):
    return (lambda: value).__closure__[0]


def _replace_cell(function, name, value):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__, strict=True))
    assert name in cells
    cells[name] = _cell(value)
    clone = FunctionType(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        tuple(cells[n] for n in function.__code__.co_freevars),
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


class BoundaryProfileStep(NativeFeedbackMassStep):
    """Direct, sparse, call-local integer wall markers; math/RNG stay literal.

    The existing two helper-owning wrappers retain their exact code objects.
    Only a new inner step closure and three local adapter callees are supplied.
    Numeric globals remain the same live original/step scope dictionaries.
    trace=False is a same-code inner clone control with no boundary recording.
    """

    def __init__(
        self, runtime, scorer=None, mode="native_cpu", *, trace=True, **extensions
    ):
        if type(trace) is not bool:
            raise TypeError("Explicit bool trace control required")
        super().__init__(runtime, scorer, mode, **extensions)
        self.trace_enabled = trace
        self.last_trace = None
        self._events = None
        self.original_compiled = self.compiled
        base = dict(
            zip(
                self.compiled.__code__.co_freevars,
                self.compiled.__closure__,
                strict=True,
            )
        )["base"].cell_contents
        old_inner = dict(zip(base.__code__.co_freevars, base.__closure__, strict=True))[
            "compiled"
        ].cell_contents
        assert (
            old_inner.__code__ is component._STEP_CODE
            and old_inner.__globals__ is self._scope
        )
        inner = (
            _factory(deepcopy(_STEP_AST), "_mark", self._scope)(self._mark)
            if trace
            else FunctionType(old_inner.__code__, self._scope)
        )
        self.inner_step = inner
        self.compiled = _replace_cell(
            self.compiled, "base", _replace_cell(base, "compiled", inner)
        )
        moves = _RAW_FACTORY(self._mark)
        for factory, _ in _ADAPTERS:
            moves = factory(moves)
        self.boundary_moves = moves

    def _mark(self, name):
        self._events.append((name, _CLOCK()))

    def _moves(self, *args, **kwargs):
        return (
            self.boundary_moves(self, *args, **kwargs)
            if self.trace_enabled
            else super()._moves(*args, **kwargs)
        )

    def __call__(self, rtk, ctx):
        if self._events is not None:
            raise RuntimeError("One filter cannot run concurrent boundary traces")
        self.last_trace = None
        for cls, fn, code in _HELPERS:
            if cls._moves is not fn or fn.__code__ is not code:
                raise ValueError("Frozen move helper changed after binding")
        if not self.trace_enabled:
            return super().__call__(rtk, ctx)
        self._events = [("call_entry", _CLOCK())]
        try:
            return super().__call__(rtk, ctx)
        finally:
            self._events.append(("call_return", _CLOCK()))
            self.last_trace = tuple(self._events)
            self._events = None


def partition(trace):
    """Adjacent wall spans partition the recorded call; skipped phases absent."""
    assert (
        type(trace) is tuple
        and trace[0][0] == "call_entry"
        and trace[-1][0] == "call_return"
    )
    assert all(type(n) is str and type(t) is int for n, t in trace)
    assert len({n for n, _ in trace}) == len(trace)
    spans = {a[0]: b[1] - a[1] for a, b in zip(trace, trace[1:])}
    assert all(v >= 0 for v in spans.values())
    assert sum(spans.values()) == trace[-1][1] - trace[0][1]
    return spans
