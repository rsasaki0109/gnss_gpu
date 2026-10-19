"""Literal D236 moves with bounded phase spans; no model transformation."""

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType

from rbpf_likelihood_batch import BatchScorer
from rbpf_move_batch import MoveBatchStep
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_move_reuse import MoveReuseStep
from rbpf_native_canonical import NativeCanonicalStep, _code_key
from rbpf_phase_profile import PhaseProfileStep

_SOURCES = {
    "rbpf_move_batch.py": "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea",
    "rbpf_move_reuse.py": "9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612",
    "rbpf_move_key_memo.py": "ddcc139b41212bfe813d552f6921b2914b6452571be410e2783a65f0d9cf9632",
    "rbpf_native_canonical.py": "f177e3247617cca70e4aa0c4369f1856dba979703f478d62c1f05cd814c63bd2",
    "rbpf_likelihood_batch.py": "7f35ae5b466145f07aa28a3de5b18f95121606b3a27a7bc85b2f9810c88bd663",
}
_METHODS = (
    (MoveBatchStep, "_moves"),
    (MoveReuseStep, "_moves"),
    (MoveKeyMemoStep, "_moves"),
    (NativeCanonicalStep, "_moves"),
    (BatchScorer, "score"),
)


def _dump(node):
    return ast.dump(node, include_attributes=False)


def _wrap(name, statements):
    node = ast.parse(f"with _profile_span('{name}'): pass").body[0]
    node.body = statements
    return node


def _compile():
    codes, originals, proofs = [], [], []
    for (cls, name), (filename, digest) in zip(_METHODS, _SOURCES.items(), strict=True):
        path = Path(__file__).with_name(filename)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("Frozen helper changed: " + filename)
        tree = ast.parse(source)
        klass = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == cls.__name__
        )
        fn = deepcopy(
            next(
                n
                for n in klass.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
        )
        baseline = deepcopy(fn)
        literal = compile(source, str(path), "exec", dont_inherit=True)
        klasscode = next(
            c for c in literal.co_consts if getattr(c, "co_name", None) == cls.__name__
        )
        literalcode = next(
            c for c in klasscode.co_consts if getattr(c, "co_name", None) == name
        )
        original = getattr(cls, name)
        if _code_key(original.__code__) != _code_key(literalcode):
            raise ValueError("Helper code no longer literal: " + filename)
        if cls is MoveBatchStep:
            (stage,) = [
                i
                for i, n in enumerate(fn.body)
                if isinstance(n, ast.For) and ast.unparse(n.iter) == "runtime.particles"
            ]
            (choice,) = [
                i
                for i, n in enumerate(fn.body)
                if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "zip(runtime.particles, staged)"
            ]
            score = stage + 1
            assert ast.unparse(fn.body[score]).startswith("requested = ")
            assert ast.unparse(fn.body[choice - 1]) == "new_states = []"
            assert ast.unparse(fn.body[choice].body[-2]).startswith(
                "pick = int(runtime.rng.choice("
            )
            fn.body = (
                fn.body[:stage]
                + [
                    _wrap("_move_staging", [fn.body[stage]]),
                    _wrap("_move_scores", fn.body[score : choice - 1]),
                    _wrap("_move_choice", fn.body[choice - 1 : choice + 1]),
                ]
                + fn.body[choice + 1 :]
            )
        elif cls is BatchScorer:
            (native,) = [
                i
                for i, n in enumerate(fn.body)
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and ast.unparse(n.targets[0]) == "code"
            ]
            assert ast.unparse(fn.body[native].value.func) == "self.dll.rbpf_score"
            fn.body = fn.body[:1] + [
                _wrap("_score_pack", fn.body[1:native]),
                _wrap("_score_native", [fn.body[native]]),
                _wrap("_score_finish", fn.body[native + 1 :]),
            ]
        else:
            previous = {
                MoveReuseStep: "MoveBatchStep._moves",
                MoveKeyMemoStep: "MoveReuseStep._moves",
                NativeCanonicalStep: "MoveKeyMemoStep._moves",
            }[cls]
            target = {
                MoveReuseStep: "_profile_raw_moves",
                MoveKeyMemoStep: "_profile_reuse_moves",
                NativeCanonicalStep: "_profile_key_moves",
            }[cls]
            calls = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == previous
            ]
            assert len(calls) == 1
            calls[0].func = ast.Name(id=target, ctx=ast.Load())
        restored = deepcopy(fn)
        if cls in (MoveBatchStep, BatchScorer):
            restored.body = [
                item
                for node in restored.body
                for item in (node.body if isinstance(node, ast.With) else [node])
            ]
        else:
            (call,) = [
                n
                for n in ast.walk(restored)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == target
            ]
            call.func = ast.parse(previous, mode="eval").body
        assert _dump(restored) == _dump(baseline), (filename, name)
        module = compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(Path(__file__)),
            "exec",
            dont_inherit=True,
        )
        codes.append(
            next(c for c in module.co_consts if getattr(c, "co_name", None) == name)
        )
        originals.append(original)
        proofs.append(
            dict(
                source=filename,
                method=name,
                source_sha256=digest,
                restored_AST_equal=True,
            )
        )
    return codes, originals, proofs


_CODES, _ORIGINALS, AST_PROOFS = _compile()
_ORIGINAL_CODES = tuple(f.__code__ for f in _ORIGINALS)


@contextmanager
def _span(spans, name):
    frame = spans.start(name)
    try:
        yield
    finally:
        spans.stop(frame)


class _ScorerView:
    def __init__(self, owner, function):
        self.owner, self.function = owner, function

    def __getattr__(self, name):
        return getattr(self.owner, name)

    def score(self, *args, **kwargs):
        return self.function(self, *args, **kwargs)


class MovePhaseProfileStep(PhaseProfileStep):
    """D238 plus three move and three scorer spans, all original expressions.

    Helper scopes refresh before each step. Every helper's literal code and
    call target is pinned, and undoing only phase wrappers/adapter target
    redirects recovers the exact original AST. No per-particle tracing, RNG
    proxy, changed candidate order, floating-point rewrite or new cache.
    """

    EXTRA_PHASES = (
        "_move_staging",
        "_move_scores",
        "_move_choice",
        "_score_pack",
        "_score_native",
        "_score_finish",
    )

    def __init__(self, runtime, scorer=None, mode="native_cpu", **extensions):
        if mode != "scalar" and type(scorer) is not BatchScorer:
            raise ValueError("Fine profiler requires the exact owning BatchScorer")
        self._clones = []
        for code, original in zip(_CODES, _ORIGINALS, strict=True):
            clone = FunctionType(
                code,
                dict(original.__globals__),
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            clone.__kwdefaults__ = original.__kwdefaults__
            self._clones.append(clone)
        wrapped = None if scorer is None else _ScorerView(scorer, self._clones[4])
        super().__init__(runtime, wrapped, mode, **extensions)

    def __call__(self, rtk, ctx):
        for (cls, name), original, code in zip(
            _METHODS, _ORIGINALS, _ORIGINAL_CODES, strict=True
        ):
            if getattr(cls, name) is not original or original.__code__ is not code:
                raise ValueError(
                    "Profiled helper changed after binding: "
                    + cls.__name__
                    + "."
                    + name
                )
        additions = {
            "_profile_span": lambda name: _span(self._spans, name),
            "_profile_raw_moves": self._clones[0],
            "_profile_reuse_moves": self._clones[1],
            "_profile_key_moves": self._clones[2],
        }
        for clone, original in zip(self._clones, _ORIGINALS, strict=True):
            clone.__globals__.clear()
            clone.__globals__.update(original.__globals__)
            clone.__globals__.update(additions)
        return super().__call__(rtk, ctx)

    def _moves(self, *args, **kwargs):
        frame = self._spans.start("_moves")
        try:
            return self._clones[3](self, *args, **kwargs)
        finally:
            self._spans.stop(frame)
