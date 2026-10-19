"""Call-count diagnosis inside D236 plus coarse actual RTK/GTSAM spans.

The profiler changes no equations or dispatch defaults. Timings include probe
overhead and scheduling; they locate work, not accepted speedups or ceilings.
"""

import ast
from contextlib import contextmanager
from copy import deepcopy
import cProfile
from pathlib import Path
import sys
import time
from types import FunctionType

from rbpf_feedback_mass import NativeFeedbackMassStep

ROOT = Path(__file__).resolve().parents[1]
_ADAPTER = "_d245_call"
_C_CALLS = {
    "gtsam.Marginals": "gtsam.Marginals",
    "mg.jointMarginalCovariance": "gtsam.jointMarginalCovariance",
    "mg.marginalCovariance": "gtsam.marginalCovariance",
    "self.smoother.update": "gtsam.smoother.update",
    "self.isam.update": "gtsam.isam.update",
    "self.smoother.calculateEstimate": "gtsam.smoother.calculateEstimate",
    "self.isam.calculateEstimate": "gtsam.isam.calculateEstimate",
}
_METHODS = (
    "process",
    "_process_phase2",
    "_prepare_common",
    "_build_dd_factors",
    "_build_dd_factors_arm",
    "_write_back",
    "_write_back_tc",
    "_apply_fde",
    "_tc_literal_postfit_quality",
    "_tc_literal_postprocess",
    "_do_ar",
    "_seed_phase2_graph",
    "_inject_hold",
    "_release_ambiguity",
)


class _Spans:
    def __init__(self):
        self.stack, self.values = [], {}

    def start(self, name):
        frame = [name, time.perf_counter_ns(), time.process_time_ns(), 0, 0]
        self.stack.append(frame)
        return frame

    def stop(self, frame):
        wall, cpu = time.perf_counter_ns(), time.process_time_ns()
        assert self.stack.pop() is frame
        wall, cpu = wall - frame[1], cpu - frame[2]
        if self.stack:
            self.stack[-1][3] += wall
            self.stack[-1][4] += cpu
        v = self.values.setdefault(
            frame[0],
            dict(calls=0, wall_ns=0, cpu_ns=0, exclusive_wall_ns=0, exclusive_cpu_ns=0),
        )
        v["calls"] += 1
        v["wall_ns"] += wall
        v["cpu_ns"] += cpu
        v["exclusive_wall_ns"] += wall - frame[3]
        v["exclusive_cpu_ns"] += cpu - frame[4]


class ProcessMonitor:
    """External integer spans; no profiler/controller stored on RTK/RBPF state."""

    def __init__(self):
        self.current = None
        self.records, self.proofs = [], []

    @contextmanager
    def span(self, name, **context):
        root = self.current is None
        if root:
            self.current = _Spans()
        spans = self.current
        frame = spans.start(name)
        try:
            yield
        finally:
            spans.stop(frame)
            if root:
                self.current = None
                assert not spans.stack
                for prefix, key in (
                    ("exclusive_wall_ns", "wall_ns"),
                    ("exclusive_cpu_ns", "cpu_ns"),
                ):
                    assert (
                        sum(v[prefix] for v in spans.values.values())
                        == spans.values[name][key]
                    )
                self.records.append(
                    dict(root=name, context=context, spans=spans.values)
                )

    def call(self, name, function, *args, **kwargs):
        with self.span(name):
            return function(*args, **kwargs)

    def instrument_type(self, original):
        """A diagnostic subclass; original class/functions/globals stay intact.

        Standalone source compilation retains the original __class__ closure
        for zero-argument super(). Only selected C-call expressions redirect
        through the monitor; undoing them recovers each literal method AST.
        """
        methods = {}
        for name in _METHODS:
            fn = getattr(original, name, None)
            if not isinstance(fn, FunctionType):
                continue
            called, proof, scope = _instrument_function(fn, self.call)
            self.proofs.append(proof)

            def wrap(
                self_rtk,
                *args,
                _fn=fn,
                _called=called,
                _scope=scope,
                _name=name,
                **kwargs,
            ):
                if _scope is not None:
                    _scope.clear()
                    _scope.update(_fn.__globals__)
                    _scope[_ADAPTER] = self.call
                context = (
                    dict(phase_before=int(getattr(self_rtk, "phase", 1)))
                    if _name == "process"
                    else {}
                )
                with self.span("rtk." + _name, **context):
                    return _called(self_rtk, *args, **kwargs)

            methods[name] = wrap
        return type("CallProfile" + original.__name__, (original,), methods)


def _instrument_function(fn, adapter):
    path = Path(fn.__code__.co_filename)
    if ".claude" in [p.casefold() for p in path.parts]:
        raise ValueError("Excluded source path")
    tree = ast.parse(path.read_bytes())
    choices = [
        (c, n)
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef)
        and n.name == fn.__name__
        and min([n.lineno, *[x.lineno for x in n.decorator_list]])
        == fn.__code__.co_firstlineno
    ]
    if len(choices) != 1:
        raise ValueError("Cannot identify literal method source")
    klass, node = choices[0]
    baseline = deepcopy(node)
    replacements = []

    class Forward(ast.NodeTransformer):
        def visit_Call(self, n):
            n = self.generic_visit(n)
            target = ast.unparse(n.func)
            if target not in _C_CALLS:
                return n
            replacements.append(target)
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=_ADAPTER, ctx=ast.Load()),
                    args=[ast.Constant(_C_CALLS[target]), n.func, *n.args],
                    keywords=n.keywords,
                ),
                n,
            )

    class Undo(ast.NodeTransformer):
        def visit_Call(self, n):
            n = self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == _ADAPTER:
                return ast.copy_location(
                    ast.Call(func=n.args[1], args=n.args[2:], keywords=n.keywords), n
                )
            return n

    # Methods assigning module globals or closing over unrelated locals use
    # only the coarse forwarding wrapper, retaining their original function.
    coarse_only = any(
        isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node)
    ) or any(v != "__class__" for v in fn.__code__.co_freevars)
    changed = deepcopy(node) if coarse_only else Forward().visit(deepcopy(node))
    restored = Undo().visit(deepcopy(changed))
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    proof = dict(
        method=fn.__qualname__,
        source=str(path),
        restored_AST_equal=True,
        replaced_C_calls=replacements,
        coarse_only=coarse_only,
    )
    if coarse_only or not replacements:
        return fn, proof, None
    cls = deepcopy(klass)
    cls.body = [changed]
    futures = [
        deepcopy(n)
        for n in tree.body
        if isinstance(n, ast.ImportFrom) and n.module == "__future__"
    ]
    module = compile(
        ast.fix_missing_locations(ast.Module(body=[*futures, cls], type_ignores=[])),
        str(path),
        "exec",
        dont_inherit=True,
    )
    class_code = next(
        c for c in module.co_consts if getattr(c, "co_name", None) == klass.name
    )
    code = next(
        c for c in class_code.co_consts if getattr(c, "co_name", None) == fn.__name__
    )
    assert code.co_freevars == fn.__code__.co_freevars
    scope = dict(fn.__globals__)
    assert _ADAPTER not in scope
    scope[_ADAPTER] = adapter
    clone = FunctionType(code, scope, fn.__name__, fn.__defaults__, fn.__closure__)
    clone.__kwdefaults__ = fn.__kwdefaults__
    return clone, proof, scope


def _code_label(code):
    if isinstance(code, str):
        return dict(file="~", line=0, function=code)
    path = Path(code.co_filename)
    try:
        name = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        name = str(path).replace("\\", "/")
    return dict(file=name, line=code.co_firstlineno, function=code.co_name)


class CallProfileStep(NativeFeedbackMassStep):
    """Original D236 step with a deterministic wall-clock call profiler.

    Counts/times are cumulative across calls. Export only outside all step
    timers. Profiling overhead changes time attribution, not model outputs.
    """

    def __init__(self, *args, monitor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor if monitor is not None else ProcessMonitor()
        self.profile = cProfile.Profile(timer=time.perf_counter_ns, timeunit=1e-9)
        self._profiling = False

    def __call__(self, rtk, ctx):
        if self._profiling or sys.getprofile() is not None:
            raise RuntimeError("Call profiling requires an idle profiling hook")
        with self.monitor.span("pf"):
            self._profiling = True
            self.profile.enable()
            try:
                return super().__call__(rtk, ctx)
            finally:
                self.profile.disable()
                self._profiling = False

    def export(self):
        if self._profiling:
            raise RuntimeError("Cannot export an active profiler")
        rows = []
        for value in self.profile.getstats():
            rows.append(
                dict(
                    code=_code_label(value.code),
                    calls=value.callcount,
                    recursive_calls=value.reccallcount,
                    self_seconds=value.inlinetime,
                    cumulative_seconds=value.totaltime,
                    children=[
                        dict(
                            code=_code_label(c.code),
                            calls=c.callcount,
                            recursive_calls=c.reccallcount,
                            self_seconds=c.inlinetime,
                            cumulative_seconds=c.totaltime,
                        )
                        for c in value.calls or ()
                    ],
                )
            )
        return sorted(
            rows,
            key=lambda x: (
                -x["self_seconds"],
                x["code"]["file"],
                x["code"]["line"],
                x["code"]["function"],
            ),
        )
