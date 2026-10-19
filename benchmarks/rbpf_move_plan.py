"""Call-local ordered-H staging plans with the literal D236 numeric tail.

Reuse registration results only in a guarded, callback-free staging domain.
All particle moves, priors, original weight arithmetic and RNG calls remain.
This is a new explicit experiment, not a product/default replacement.
"""

import ast
import builtins
import hashlib
import math
import sys
from types import FunctionType

import numpy as np

from rbpf_epoch_memo import _EpochMemo
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256
from rbpf_move_reuse import _MoveView, _ordered_integer_key
from rbpf_native_canonical import CanonicalEngine, _CanonicalView, _code_key
from rbpf_native_staging import NativeStagingStep, _CODES, _ORIGINALS, _RAW_STAGE, _PRIORS

_BUILTINS = {n: getattr(builtins, n) for n in
    ("len", "range", "enumerate", "dict", "int", "float", "abs", "max", "type", "all", "any", "tuple", "id")}
_NUMPY = {n: getattr(np, n) for n in ("full", "where", "isfinite", "ix_", "rint")}
_SOLVE, _LINALG_ERROR = np.linalg.solve, np.linalg.LinAlgError
_CANONICAL = CanonicalEngine.canonical
_CANONICAL_CODE = _CANONICAL.__code__
_VIEW_METHODS = {n: getattr(_MoveView, n) for n in ("__getattr__", "_key", "_evaluate", "_extend", "_shrink_worst")}
_VIEW_CODES = {n: f.__code__ for n, f in _VIEW_METHODS.items()}
_ORDERED_CODE, _MEMO = _ordered_integer_key.__code__, _EpochMemo.canonical
_MEMO_CODE = _MEMO.__code__


def _literal_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError("Frozen RBPF source changed")
    tree = ast.parse(source)
    module = compile(source, str(SCALAR_SOURCE), "exec", dont_inherit=True)
    cls = next(c for c in module.co_consts if getattr(c, "co_name", None) == "RbpfFgo")
    names = ("_extend", "_shrink_worst", "_pair_dd")
    codes = {n: _code_key(next(c for c in cls.co_consts if getattr(c, "co_name", None) == n)) for n in names}
    codes["_copy_H"] = _code_key(next(c for c in module.co_consts if getattr(c, "co_name", None) == "_copy_H"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    assert all(any(isinstance(n, ast.FunctionDef) and n.name == name for n in klass.body) for name in names)
    return codes


_LITERAL = _literal_codes()


class MovePlanEngine:
    """Plan only immutable builtin inputs; unsupported domains run raw once.

    Pure source canonical calls on redundant registrations are intentionally
    skipped. Existing work counters report actual executed requests/calls;
    they must not be described as unchanged. No numeric result is cached across
    scoring, choice, commit, graph callbacks, or epochs. Concurrent foreign
    mutation/audit hooks/OOM are outside this fixed private-probe domain.
    """

    def __init__(self, owner, scope, enabled):
        self.owner, self.enabled = owner, enabled
        self.original = FunctionType(_RAW_STAGE, scope)
        self.native_canonical = owner.engine.module.canonical
        self.cfg_type = owner._scope["RbpfConfig"]
        self.problem_type = owner.mass_engine.problem_type
        self.particle_type = owner.mass_engine.particle_type
        self.stats = dict(calls=0, planned=0, fallback=0, particles=0,
            logical_registrations=0, executed_registrations=0,
            avoided_registrations=0, prefix_plans=0, prefix_replays=0,
            common_records=0, common_replayed_records=0)

    def counts(self):
        return dict(self.stats)

    def _eligible(self, context, runtime, cfg, prob, H_cons, cand_H, spawn_H, cache):
        owner = self.owner
        scope = owner._source_globals
        base = owner._source_functions["_extend"].__builtins__
        engine = owner.engine
        function = engine.original
        memo = getattr(context.canonical, "__self__", None)
        if (not self.enabled or sys.getprofile() is not None or sys.gettrace() is not None
                or type(runtime) is not _MoveView or type(runtime.runtime) is not _CanonicalView
                or runtime.runtime._actual is not owner.runtime
                or type(owner.runtime).__getattribute__ is not object.__getattribute__
                or type(cfg) is not self.cfg_type or self.cfg_type.__getattribute__ is not object.__getattribute__
                or type(prob) is not self.problem_type or self.problem_type.__getattribute__ is not object.__getattribute__
                or type(cand_H) is not list or type(spawn_H) is not list or type(cache) is not dict
                or type(engine) is not CanonicalEngine or CanonicalEngine.canonical is not _CANONICAL
                or _CANONICAL.__code__ is not _CANONICAL_CODE or engine.literal is not True
                or "canonical" in vars(engine) or engine.module.canonical is not self.native_canonical
                or type(engine.native_calls) is not int or type(engine.fallback_calls) is not int
                or function is not owner.canonical or function.__code__ is not engine.code
                or getattr(context.canonical, "__func__", None) is not _MEMO
                or _EpochMemo.canonical is not _MEMO or type(memo) is not _EpochMemo
                or _MEMO.__code__ is not _MEMO_CODE
                or _ordered_integer_key.__code__ is not _ORDERED_CODE):
            return None
        if (getattr(memo.source_canonical, "__self__", None) is not engine
                or getattr(memo.source_canonical, "__func__", None) is not _CANONICAL
                or type(memo.identities) is not dict or memo.identities
                or type(memo.counts) is not dict or any(type(v) is not int for v in memo.counts.values())):
            return None
        if (any(getattr(_MoveView, n) is not f or f.__code__ is not _VIEW_CODES[n] for n, f in _VIEW_METHODS.items())
                or any(scope.get(n, base.get(n)) is not v for n, v in _BUILTINS.items())
                or scope.get("np") is not np or any(getattr(np, n) is not f for n, f in _NUMPY.items())
                or np.linalg.solve is not _SOLVE or np.linalg.LinAlgError is not _LINALG_ERROR):
            return None
        for f in (_MEMO, _ordered_integer_key, *self._view_functions()):
            if any(f.__globals__.get(n, f.__builtins__.get(n)) is not _BUILTINS[n]
                    for n in ("type", "dict", "id", "tuple", "int", "all", "any")):
                return None
        if (_MEMO.__globals__.get("_ordered_integer_key") is not _ordered_integer_key
                or _MoveView._key.__globals__.get("_ordered_integer_key") is not _ordered_integer_key
                or self.original.__globals__.get("len", self.original.__builtins__.get("len")) is not builtins.len):
            return None
        for n in ("sorted", "min", "tuple"):
            if function.__globals__.get(n, function.__builtins__.get(n)) is not getattr(builtins, n):
                return None
        for n, code in _LITERAL.items():
            f = scope.get(n) if n == "_copy_H" else owner._source_functions[n]
            if type(f) is not FunctionType or _code_key(f.__code__) != code:
                return None
        if any(self.cfg_type.__dict__.get(n) is not None
                or type(vars(cfg).get(n)) is not float or not math.isfinite(vars(cfg)[n])
                or vars(cfg)[n] <= 0.0 for n in _PRIORS):
            return None
        fields = ("pairs", "Qb", "y", "rows_v0", "rows_iscp", "rows_pair", "rows_bmeas", "rows_lam", "rows_sig")
        if any(n not in vars(prob) or self.problem_type.__dict__.get(n) is not None for n in fields):
            return None
        if type(prob.pairs) is not list:
            return None
        for pair in prob.pairs:
            if (type(pair) is not tuple or len(pair) != 3 or type(pair[0]) is not tuple
                    or len(pair[0]) != 2 or any(type(x) is not int for x in (*pair[0], pair[1], pair[2]))):
                return None
        for n in fields[1:]:
            a = vars(prob)[n]
            if type(a) is not np.ndarray or a.dtype.kind not in "fib" or a.dtype.hasobject:
                return None
        if ("particles" not in vars(owner.runtime) or type(vars(owner.runtime)["particles"]) is not list
                or self.particle_type.__getattribute__ is not object.__getattribute__):
            return None
        particles = vars(owner.runtime)["particles"]
        planned = []
        for p in particles:
            if type(p) is not self.particle_type:
                return None
            h = p.H
            key = _ordered_integer_key(h)
            if key is None:
                return None
            planned.append((h, key))
        for h in ([] if H_cons is None else [H_cons]) + cand_H + spawn_H:
            if _ordered_integer_key(h) is None:
                return None
        return planned

    @staticmethod
    def _view_functions():
        return tuple(_VIEW_METHODS.values())

    def stage(self, context, runtime, cfg, prob, H_cons, cand_H, spawn_H,
              cache, keys, hypotheses, lookup, pending, staged):
        self.stats["calls"] += 1
        plan = self._eligible(context, runtime, cfg, prob, H_cons, cand_H, spawn_H, cache)
        if plan is None:
            self.stats["fallback"] += 1
            return self.original(context, runtime, cfg, prob, H_cons, cand_H, spawn_H,
                cache, keys, hypotheses, lookup, pending, staged)
        self.stats["planned"] += 1
        held = 0
        prefixes, common = {}, None
        executed = prefix_replays = common_replayed = 0

        def index(hypothesis):
            nonlocal executed
            executed += 1
            key = context.canonical(hypothesis)
            if key not in lookup:
                lookup[key] = len(keys)
                keys.append(key)
                hypotheses.append(hypothesis)
                if cache.get(key) is None:
                    pending.append(len(keys) - 1)
            return lookup[key]

        for h, ordered in plan:
            moves = []

            def add(prior, hypothesis, tag):
                moves.append((prior, index(hypothesis), hypothesis, tag))

            held += int(bool(h))
            if ordered not in prefixes:
                if h:
                    add(cfg.p_survive, h, "survive")
                    extended = runtime._extend(h, prob)
                    if extended is not None:
                        add(cfg.p_extend, extended, "extend")
                    shrunk = runtime._shrink_worst(h, prob)
                    if shrunk is not None:
                        add(cfg.p_shrink, shrunk, "shrink")
                    add(cfg.p_release, {}, "release")
                else:
                    add(cfg.p_survive + cfg.p_extend + cfg.p_shrink + cfg.p_release, h, "stay")
                prefixes[ordered] = tuple(moves)
            else:
                prefix_replays += 1
                for prior, i, original_h, tag in prefixes[ordered]:
                    # Survive/stay use THIS particle's own H. Release remains
                    # a fresh dictionary for every particle, as in the source.
                    selected = h if tag in ("survive", "stay") else ({} if tag == "release" else original_h)
                    moves.append((prior, i, selected, tag))
            if common is None:
                start = len(moves)
                if H_cons is not None:
                    add(cfg.p_adopt_cons, H_cons, "adopt_cons")
                for candidate in cand_H:
                    add(cfg.p_adopt / len(cand_H), candidate, "adopt")
                for spawned in spawn_H:
                    add(cfg.p_spawn / len(spawn_H), spawned, "spawn")
                common = tuple(moves[start:])
            else:
                common_replayed += len(common)
                for prior, i, hypothesis, tag in common:
                    moves.append((prior, i, hypothesis, tag))
            staged.append(moves)
        logical = sum(map(len, staged))
        self.stats["particles"] += len(plan)
        self.stats["logical_registrations"] += logical
        self.stats["executed_registrations"] += executed
        self.stats["avoided_registrations"] += logical - executed
        self.stats["prefix_plans"] += len(prefixes)
        self.stats["prefix_replays"] += prefix_replays
        self.stats["common_records"] += len(common) if common else 0
        self.stats["common_replayed_records"] += common_replayed
        return held


class MovePlanStep(NativeStagingStep):
    """New Python staging plan; original numerical functions and tail execute.

    Reuse only the D241 reversible adapter redirects, not its rejected C++ loop.
    Off executes the literal Python staging block via those same adapters.
    Methods/globals refresh/guards are inherited without weakening them.
    """

    def __init__(self, runtime, scorer=None, mode="scalar", *, planned=True, **extensions):
        if type(planned) is not bool:
            raise TypeError("planned must be explicit bool")
        NativeFeedbackMassStep.__init__(self, runtime, scorer, mode, **extensions)
        self._staging_clones = []
        for code, original in zip(_CODES, _ORIGINALS, strict=True):
            clone = FunctionType(code, dict(original.__globals__), original.__name__,
                original.__defaults__, original.__closure__)
            clone.__kwdefaults__ = original.__kwdefaults__
            self._staging_clones.append(clone)
        self.staging_engine = MovePlanEngine(self, self._staging_clones[0].__globals__, planned)

    def __call__(self, rtk, ctx):
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if self.last is not previous and self.last is not None and "native_staging" in self.last:
                self.last["move_plan"] = self.last.pop("native_staging")
