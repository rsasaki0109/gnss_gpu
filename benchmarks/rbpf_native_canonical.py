"""Explicit native integer basin keys with the frozen D229 numerical caller."""

import builtins
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256, make_oracle
from rbpf_move_batch import _function
from rbpf_move_key_memo import MoveKeyMemoStep


def _code_key(code):
    return (code.co_code, code.co_consts, code.co_names, code.co_varnames,
        code.co_argcount, code.co_kwonlyargcount, code.co_freevars, code.co_cellvars)


def _literal_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError('Frozen RBPF source changed')
    module = compile(source, str(SCALAR_SOURCE), 'exec', dont_inherit=True)
    imported = next(c for c in module.co_consts if getattr(c, 'co_name', None) == '_canon_basin_id')
    return (_code_key(make_oracle().basin_id.__code__), _code_key(imported))


_CODES = _literal_codes()
_SORTED, _MIN, _TUPLE = builtins.sorted, builtins.min, builtins.tuple


class CanonicalEngine:
    """No cache: one fresh exact key, or the original custom Python result.

    Only the literal frozen function with ordinary builtin bindings is native.
    Recheck bindings per call: a custom sorted/min/tuple still runs normally.
    Integer types/mappings outside the exact native domain bypass in C++ before
    arithmetic, preserving Python callbacks, exceptions and integer semantics.
    """

    def __init__(self, extension, original):
        if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
            raise RuntimeError('The explicit key probe requires CPython 3.12')
        path = Path(extension).resolve()
        spec = importlib.util.spec_from_file_location('_rbpf_canonical_probe', path)
        if spec is None or spec.loader is None:
            raise ValueError('An explicit canonical extension is required')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original = original
        self.code = getattr(original, '__code__', None)
        self.literal = (type(original) is FunctionType
            and Path(original.__code__.co_filename).resolve() == SCALAR_SOURCE.resolve()
            and _code_key(original.__code__) in _CODES)
        self.native_calls = self.fallback_calls = 0

    def canonical(self, hypothesis):
        function = self.original
        if self.literal and function.__code__ is self.code:
            scope, base = function.__globals__, function.__builtins__
            if (scope.get('sorted', base.get('sorted')) is _SORTED
                    and scope.get('min', base.get('min')) is _MIN
                    and scope.get('tuple', base.get('tuple')) is _TUPLE):
                key = self.module.canonical(hypothesis)
                if key is not None:
                    self.native_calls += 1
                    return key
        self.fallback_calls += 1
        return function(hypothesis)

    def counts(self):
        return dict(native_calls=self.native_calls, fallback_calls=self.fallback_calls)


class _CanonicalView:
    __slots__ = ('_actual', '_functions', '_bound')

    def __init__(self, actual, functions):
        object.__setattr__(self, '_actual', actual)
        object.__setattr__(self, '_functions', functions)
        object.__setattr__(self, '_bound', {})

    def __getattr__(self, name):
        if name in self._functions:
            if name not in self._bound:
                self._bound[name] = MethodType(self._functions[name], self)
            return self._bound[name]
        return getattr(self._actual, name)

    def __setattr__(self, name, value):
        setattr(self._actual, name, value)


class NativeCanonicalStep(MoveKeyMemoStep):
    """Opt-in C++ key construction; no DD, scoring or model changes.

    Original method code and refreshed globals execute through a per-call view.
    Only the canonical function is redirected; every integer difference remains
    arbitrary precision. D229 move reuse/key memo, RNG order, all candidates,
    math and strict-policy original-step fallback remain. Nothing patches the
    actual runtime methods/module, and no additional H/key cache is retained.
    """

    def __init__(self, runtime, scorer=None, mode='scalar', *, extension):
        super().__init__(runtime, scorer, mode)
        self.engine = CanonicalEngine(extension, self.canonical)
        if not self.engine.literal:
            raise ValueError('Native keys require the frozen literal canonical function')
        self._source_globals = self.methods['step'].__globals__
        self._scope = dict(self._source_globals)
        self._source_functions, self._functions = {}, {}
        for name in dir(type(runtime)):
            method = getattr(runtime, name)
            function = getattr(method, '__func__', None)
            if (getattr(method, '__self__', None) is runtime and function is not None
                    and function.__globals__ is self._source_globals):
                self._source_functions[name] = function
                clone = FunctionType(function.__code__, self._scope, function.__name__,
                    function.__defaults__, function.__closure__)
                clone.__kwdefaults__ = function.__kwdefaults__
                self._functions[name] = clone
        original = self.compiled
        self.compiled = FunctionType(original.__code__, self._scope, original.__name__,
            original.__defaults__, original.__closure__)
        self._view = self._oracle = None

    def __call__(self, rtk, ctx):
        if self._view is not None:
            raise RuntimeError('One filter cannot run concurrent native-key epochs')
        for name, function in {**self._source_functions, **self.methods}.items():
            current = getattr(self.runtime, name)
            if name == 'step' and current is self:
                continue
            if _function(current) is not function:
                raise ValueError(f'RBPF method changed after binding: {name}')
        if (self._source_globals['_canon_basin_id'] is not self.canonical
                or self.canonical.__code__ is not self.engine.code):
            raise ValueError('RBPF canonical function changed after binding')
        self.last = None
        policy = np.geterr()
        if any(v not in ('ignore', 'warn') for v in policy.values()) or policy['under'] != 'ignore':
            self.last = dict(original_step_fallback='numpy_error_policy')
            return self.original(rtk, ctx)
        self._scope.clear()
        self._scope.update(self._source_globals)
        self._scope.update(_canon_basin_id=self.engine.canonical, _rbpf_batch_moves=self._moves)
        view = self._view = _CanonicalView(self.runtime, self._functions)
        self._oracle = SimpleNamespace(cfg=self.runtime.cfg, basin_id=self.engine.canonical,
            _pair_dd=view._pair_dd, _cluster_state=view._cluster_state, _loglik=view._loglik)
        before = self.engine.counts()
        try:
            return self.compiled(view, rtk, ctx)
        finally:
            if self.last is not None and 'batch' in self.last:
                self.last['native_canonical'] = {key:value-before[key] for key,value in self.engine.counts().items()}
            self._scope['_canon_basin_id'] = self.canonical
            self._view = self._oracle = None

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        context = SimpleNamespace(runtime=self._view, oracle=self._oracle, canonical=self.engine.canonical,
            scorer=self.scorer, mode=self.mode, last=None)
        result = MoveKeyMemoStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        return result
