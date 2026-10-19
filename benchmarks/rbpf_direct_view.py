"""Use unchanged PF functions through ordinary class method descriptors."""

import ast
import hashlib
from types import FunctionType, MethodType

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256
from rbpf_native_canonical import _CanonicalView


def _fields():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError("Frozen RBPF source changed")
    klass = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    fields = set()
    for node in ast.walk(klass):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            fields.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr", "setattr", "delattr", "vars"}:
            if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                raise ValueError("Dynamic self access requires the original view")
    return frozenset(fields)


_FIELDS = _fields()


class _ActualField:
    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name

    def __get__(self, view, owner=None):
        if view is None:
            return self
        return getattr(object.__getattribute__(view, "_actual"), self.name)


class DirectViewEngine:
    """Promote the same private view, retaining all numerical function objects.

    Ordinary class lookup allows CPython's method-call path to avoid the
    original __getattr__/cached MethodType route. The three oracle bindings
    already constructed by D233 remain callable. No per-epoch view is reused.
    Only the original exact view, function map and ordinary cache are eligible.
    This is a fixed private-view API, not arbitrary namespace mutation support.
    """

    def __init__(self, runtime, functions):
        if type(functions) is not dict or any(type(k) is not str or type(v) is not FunctionType for k, v in functions.items()):
            raise ValueError("Ordinary private function table required")
        self.runtime, self.functions = runtime, functions
        self.frozen_functions = functions.copy()
        reserved = set(dir(_CanonicalView))
        methods = {k: v for k, v in functions.items() if k not in reserved}
        names = (_FIELDS | set(vars(runtime))) - set(methods) - reserved
        self.fields = {k: _ActualField(k) for k in names}
        namespace = dict(__slots__=_CanonicalView.__slots__,
            __init__=_CanonicalView.__init__, __setattr__=_CanonicalView.__setattr__,
            **self.fields, **methods)
        self.view_type = type("_DirectCanonicalView", (), namespace)
        self.class_bindings = {k: v for k, v in namespace.items() if k != "__slots__"}
        self.counts = dict(calls=0, promoted=0, declined=0, initial_cached_methods=0)

    def promote(self, view):
        self.counts["calls"] += 1
        table = self.functions
        kind = self.view_type
        eligible = (type(view) is _CanonicalView and type(table) is dict
            and len(table) == len(self.frozen_functions)
            and all(table.get(k) is v for k, v in self.frozen_functions.items())
            and kind.__getattribute__ is object.__getattribute__
            and not hasattr(kind, "__getattr__")
            and all(kind.__dict__.get(k) is v for k, v in self.class_bindings.items())
            and all(v.name == k for k, v in self.fields.items()))
        if eligible:
            eligible = (object.__getattribute__(view, "_actual") is self.runtime
                and object.__getattribute__(view, "_functions") is table)
        cache = object.__getattribute__(view, "_bound") if eligible else None
        if eligible:
            eligible = (type(cache) is dict and all(type(k) is str and type(v) is MethodType
                and v.__self__ is view and v.__func__ is table.get(k) for k, v in cache.items()))
        if eligible:
            try:
                object.__setattr__(view, "__class__", kind)
            except TypeError:
                eligible = False
        if not eligible:
            self.counts["declined"] += 1
            return False
        self.counts["initial_cached_methods"] += len(cache)
        cache.clear()
        self.counts["promoted"] += 1
        return True


class DirectViewStep(NativeFeedbackMassStep):
    """D236 with one private view promotion before its unchanged PF body."""

    def __init__(self, runtime, scorer=None, mode="scalar", **extensions):
        super().__init__(runtime, scorer, mode, **extensions)
        self.direct_engine = DirectViewEngine(runtime, self._functions)
        self.original_compiled = base = self.compiled

        def call(view, rtk, ctx):
            self.direct_engine.promote(view)
            return base(view, rtk, ctx)

        self.compiled = call

    def __call__(self, rtk, ctx):
        before, previous = self.direct_engine.counts.copy(), self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if self.last is not previous and self.last is not None and "batch" in self.last:
                self.last["direct_view"] = {k: v - before[k] for k, v in self.direct_engine.counts.items()}
