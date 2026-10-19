"""Explicit epoch-local canonical/DD reuse for the frozen RBPF model.

Literal method code objects execute through a per-call state view. Globals
are refreshed from the original module each call; only canonical lookup is
redirected to the owned memo. State writes still target the actual filter.
The previous move/weight/choice implementation and native DLL are unchanged.
"""

from types import FunctionType, MethodType, SimpleNamespace

import numpy as np

from rbpf_move_batch import _function
from rbpf_move_reuse import MoveReuseStep, _ordered_integer_key


class _EpochMemo:
    def __init__(self, canonical, pair_dd, *, cache_dd=True):
        self.source_canonical, self.source_dd = canonical, pair_dd
        self.cache_dd = cache_dd
        self.identities, self.problems, self.dd = {}, {}, {}
        self.counts = dict(canonical_requests=0, canonical_source_calls=0,
            canonical_identity_hits=0, canonical_empty_hits=0, canonical_bypass=0,
            DD_requests=0, DD_source_calls=0, DD_hits=0, DD_bypass=0)

    def canonical(self, hypothesis):
        counts = self.counts
        counts['canonical_requests'] += 1
        if type(hypothesis) is dict and not hypothesis:
            counts['canonical_empty_hits'] += 1
            return ()
        identity = id(hypothesis)
        if identity in self.identities:
            counts['canonical_identity_hits'] += 1
            return self.identities[identity][1]
        counts['canonical_source_calls'] += 1
        key = self.source_canonical(hypothesis)
        if _ordered_integer_key(hypothesis) is None:
            counts['canonical_bypass'] += 1
        else:
            # Retain the object: its id cannot be recycled during the epoch.
            self.identities[identity] = (hypothesis, key)
        return key

    def pair_dd(self, hypothesis, problem):
        counts = self.counts
        counts['DD_requests'] += 1
        if not self.cache_dd:
            counts['DD_source_calls'] += 1
            return self.source_dd(hypothesis, problem)
        canonical = self.canonical(hypothesis)
        supported = ((type(hypothesis) is dict and not hypothesis)
                     or id(hypothesis) in self.identities)
        if not supported:
            counts['DD_source_calls'] += 1
            counts['DD_bypass'] += 1
            return self.source_dd(hypothesis, problem)
        # Different problem identities never share a DD vector. The frozen
        # model does not mutate pairs within an epoch. Keep each problem alive.
        identity = id(problem)
        self.problems[identity] = problem
        key = (identity, canonical)
        if key not in self.dd:
            counts['DD_source_calls'] += 1
            self.dd[key] = self.source_dd(hypothesis, problem)
        else:
            counts['DD_hits'] += 1
        return self.dd[key]

    def summary(self):
        return dict(self.counts, identity_entries=len(self.identities),
            problem_entries=len(self.problems), DD_entries=len(self.dd),
            DD_payload_bytes=sum(array.nbytes for array in self.dd.values()),
            DD_reuse_enabled=self.cache_dd)


class _EpochView:
    __slots__ = ('_actual', '_memo', '_functions', '_bound')

    def __init__(self, actual, memo, functions):
        object.__setattr__(self, '_actual', actual)
        object.__setattr__(self, '_memo', memo)
        object.__setattr__(self, '_functions', functions)
        object.__setattr__(self, '_bound', {})

    def __getattr__(self, name):
        if name == '_pair_dd':
            return self._memo.pair_dd
        if name in self._functions:
            if name not in self._bound:
                self._bound[name] = MethodType(self._functions[name], self)
            return self._bound[name]
        return getattr(self._actual, name)

    def __setattr__(self, name, value):
        setattr(self._actual, name, value)


class EpochMemoStep(MoveReuseStep):
    """Opt-in full-epoch memo, not a default model/runtime replacement.

    Canonical results are immutable tuples. DD sharing is allowed only for
    plain integer dictionaries and the same problem identity; gauge-equivalent
    integer dictionaries imply the same original FP64 DD vector. All frozen
    consumers read DD arrays without modifying them. The original commit copies
    particle dictionaries before any later particle mutation.
    """

    def __init__(self, runtime, scorer=None, mode='scalar', *, cache_dd=True):
        super().__init__(runtime, scorer, mode)
        self.cache_dd = cache_dd
        self._source_globals = self.methods['step'].__globals__
        self._scope = dict(self._source_globals)
        self._source_functions, self._functions = {}, {}
        for name in dir(type(runtime)):
            method = getattr(runtime, name)
            function = getattr(method, '__func__', None)
            if (getattr(method, '__self__', None) is runtime
                    and function is not None and function.__globals__ is self._source_globals):
                self._source_functions[name] = function
                clone = FunctionType(function.__code__, self._scope, function.__name__,
                    function.__defaults__, function.__closure__)
                clone.__kwdefaults__ = function.__kwdefaults__
                self._functions[name] = clone
        original = self.compiled
        self.compiled = FunctionType(original.__code__, self._scope, original.__name__,
            original.__defaults__, original.__closure__)
        self._epoch = self._view = self._oracle = None

    def __call__(self, rtk, ctx):
        if self._epoch is not None:
            raise RuntimeError('One filter cannot run concurrent memoized epochs')
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
        memo = _EpochMemo(self.canonical, self.methods['_pair_dd'].__get__(self.runtime), cache_dd=self.cache_dd)
        self._epoch = memo
        self._scope.clear()
        self._scope.update(self._source_globals)
        self._scope.update(_canon_basin_id=memo.canonical, _rbpf_batch_moves=self._moves)
        view = self._view = _EpochView(self.runtime, memo, self._functions)
        self._oracle = SimpleNamespace(cfg=self.runtime.cfg, basin_id=memo.canonical,
            _pair_dd=view._pair_dd, _cluster_state=view._cluster_state, _loglik=view._loglik)
        try:
            return self.compiled(view, rtk, ctx)
        finally:
            if self.last is not None and 'batch' in self.last:
                self.last['epoch_memo'] = memo.summary()
            # The persistent function globals must not retain the epoch memo.
            self._scope['_canon_basin_id'] = self.canonical
            self._epoch = self._view = self._oracle = None

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        context = SimpleNamespace(runtime=self._view, oracle=self._oracle, canonical=self._epoch.canonical,
            scorer=self.scorer, mode=self.mode, last=None)
        result = MoveReuseStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        return result
