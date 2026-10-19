"""Explicit, source-pinned native DD input expansion; unchanged score math."""

import ast
from copy import deepcopy
import ctypes
import hashlib
from pathlib import Path
import sys
from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

import rbpf_likelihood_batch as original_batch
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256, make_oracle
from rbpf_move_batch import _function
from rbpf_move_key_memo import MoveKeyMemoStep

BATCH_SHA256 = '7f35ae5b466145f07aa28a3de5b18f95121606b3a27a7bc85b2f9810c88bd663'


def _code_key(code):
    return (code.co_code, code.co_consts, code.co_names, code.co_varnames,
        code.co_argcount, code.co_kwonlyargcount, code.co_freevars, code.co_cellvars)


_DD_CODE = _code_key(make_oracle()._pair_dd.__func__.__code__)


def _literal_dd(method):
    function = getattr(method, '__func__', None)
    return (type(method) is MethodType and function is not None
        and _code_key(function.__code__) == _DD_CODE
        and function.__globals__.get('np') is np)


class DdPacker:
    """Owning PyDLL reference; one fresh FP64 result for every request.

    Exact builtin containers/ints use the C API. Other mappings, numeric
    subclasses and custom DD hooks keep the original Python implementation.
    There is no DD/key cache and no integer narrowing or input mutation.
    """

    def __init__(self, dll):
        if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
            raise RuntimeError('The explicit DD probe requires CPython 3.12')
        self.dll = ctypes.PyDLL(str(Path(dll).resolve()))
        self.fill = self.dll.rbpf_pack_dd
        self.fill.argtypes = [ctypes.py_object, ctypes.py_object, ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_double]
        self.fill.restype = ctypes.c_int
        self.source_dd = None
        self.native_calls = self.native_hypotheses = self.fallback_calls = 0

    def batch(self, hypotheses, problem, source_dd):
        output = np.empty((len(hypotheses), len(problem.pairs)))
        if self.fill(hypotheses, problem.pairs, output.ctypes.data, output.size, np.nan) == 1:
            self.native_calls += 1
            self.native_hypotheses += len(hypotheses)
            return output
        self.fallback_calls += 1
        return np.ascontiguousarray([source_dd(h, problem) for h in hypotheses])

    def one(self, hypothesis, problem):
        output = np.empty(len(problem.pairs))
        if self.fill([hypothesis], problem.pairs, output.ctypes.data, output.size, np.nan) == 1:
            self.native_calls += 1
            self.native_hypotheses += 1
            return output
        self.fallback_calls += 1
        return self.source_dd(hypothesis, problem)

    def counts(self):
        return dict(native_calls=self.native_calls, native_hypotheses=self.native_hypotheses,
            fallback_calls=self.fallback_calls)


def _pack_dd(scorer, unique, problem, oracle):
    method = oracle._pair_dd
    owner = getattr(method, '__self__', None)
    own = (type(owner) is DdPacker and getattr(method, '__func__', None) is DdPacker.one
        and _literal_dd(owner.source_dd))
    if own or _literal_dd(method):
        return scorer.packer.batch(unique, problem, method)
    return np.ascontiguousarray([method(h, problem) for h in unique])


def _score_function():
    path = Path(original_batch.__file__)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != BATCH_SHA256:
        raise ValueError('Frozen batch scorer changed')
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BatchScorer')
    method = deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'score'))
    targets = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'dd']
    if len(targets) != 1 or ast.unparse(targets[0].value) != 'np.ascontiguousarray([oracle._pair_dd(h, problem) for h in unique])':
        raise ValueError('Frozen DD packing boundary changed')
    targets[0].value = ast.parse('_pack_dd(self, unique, problem, oracle)', mode='eval').body
    scope = dict(vars(original_batch), _pack_dd=_pack_dd)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(Path(__file__)), 'exec'), scope)
    return scope['score']


class PackedDdScorer(original_batch.BatchScorer):
    """Replace just one DD packing expression in the unchanged batch caller."""

    score = _score_function()

    def __init__(self, dll, capacity=512, *, pack_dll):
        self.packer = DdPacker(pack_dll)
        super().__init__(dll, capacity)


class _DdView:
    __slots__ = ('_actual', '_packer', '_functions', '_bound')

    def __init__(self, actual, packer, functions):
        object.__setattr__(self, '_actual', actual)
        object.__setattr__(self, '_packer', packer)
        object.__setattr__(self, '_functions', functions)
        object.__setattr__(self, '_bound', {})

    def __getattr__(self, name):
        if name == '_pair_dd':
            return self._packer.one
        if name in self._functions:
            if name not in self._bound:
                self._bound[name] = MethodType(self._functions[name], self)
            return self._bound[name]
        return getattr(self._actual, name)

    def __setattr__(self, name, value):
        setattr(self._actual, name, value)


class PackedDdStep(MoveKeyMemoStep):
    """Opt-in DD expansion for literal full-step numeric consumers as well.

    Original method code and module globals run through a per-call state
    view. Only _pair_dd uses native input expansion; no canonical/DD caching
    is added. All model equations, conditionals, moves and RNG stay literal.
    Runtime methods are never temporarily overwritten. Strict error policies
    execute the original entire step, preserving its partial error state.
    """

    def __init__(self, runtime, scorer=None, mode='scalar', *, pack_dll=None):
        super().__init__(runtime, scorer, mode)
        if hashlib.sha256(SCALAR_SOURCE.read_bytes()).hexdigest() != SCALAR_SHA256 or not _literal_dd(runtime._pair_dd):
            raise ValueError('Native DD supports only the frozen literal DD method')
        self.packer = DdPacker(pack_dll if pack_dll is not None else scorer.packer.dll._name)
        self.packer.source_dd = runtime._pair_dd
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
            raise RuntimeError('One filter cannot run concurrent packed-DD epochs')
        for name, function in {**self._source_functions, **self.methods}.items():
            current = getattr(self.runtime, name)
            if name == 'step' and current is self:
                continue
            if _function(current) is not function:
                raise ValueError(f'RBPF method changed after binding: {name}')
        self.last = None
        policy = np.geterr()
        if any(v not in ('ignore', 'warn') for v in policy.values()) or policy['under'] != 'ignore':
            self.last = dict(original_step_fallback='numpy_error_policy')
            return self.original(rtk, ctx)
        self._scope.clear()
        self._scope.update(self._source_globals)
        self._scope['_rbpf_batch_moves'] = self._moves
        view = self._view = _DdView(self.runtime, self.packer, self._functions)
        self._oracle = SimpleNamespace(cfg=self.runtime.cfg, basin_id=self.canonical,
            _pair_dd=view._pair_dd, _cluster_state=view._cluster_state, _loglik=view._loglik)
        before = self.packer.counts()
        batch_before = self.scorer.packer.counts() if self.scorer is not None else dict(before)
        try:
            return self.compiled(view, rtk, ctx)
        finally:
            if self.last is not None and 'batch' in self.last:
                self.last['DD_pack'] = {key:value-before[key] for key,value in self.packer.counts().items()}
                if self.scorer is not None:
                    for key, value in self.scorer.packer.counts().items():
                        self.last['DD_pack'][key] += value-batch_before[key]
            self._view = self._oracle = None

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        context = SimpleNamespace(runtime=self._view, oracle=self._oracle, canonical=self.canonical,
            scorer=self.scorer, mode=self.mode, last=None)
        result = MoveKeyMemoStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        return result
