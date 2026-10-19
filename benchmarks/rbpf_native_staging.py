"""Register all original particle moves through a standalone native loop.

Only staging moves to C++; canonical calls, value hashing, probabilities,
RNG choice and all existing memo accounting remain. D236 is the baseline.
"""

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

from rbpf_epoch_memo import _EpochMemo
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_batch import MoveBatchStep
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_move_reuse import MoveReuseStep, _MoveView
from rbpf_native_canonical import NativeCanonicalStep, _CanonicalView, _code_key

_LEN = builtins.len
_SPECS = (
    (
        MoveBatchStep,
        "rbpf_move_batch.py",
        "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea",
        None,
        None,
    ),
    (
        MoveReuseStep,
        "rbpf_move_reuse.py",
        "9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612",
        "MoveBatchStep._moves",
        "_staging_raw_moves",
    ),
    (
        MoveKeyMemoStep,
        "rbpf_move_key_memo.py",
        "ddcc139b41212bfe813d552f6921b2914b6452571be410e2783a65f0d9cf9632",
        "MoveReuseStep._moves",
        "_staging_reuse_moves",
    ),
    (
        NativeCanonicalStep,
        "rbpf_native_canonical.py",
        "f177e3247617cca70e4aa0c4369f1856dba979703f478d62c1f05cd814c63bd2",
        "MoveKeyMemoStep._moves",
        "_staging_key_moves",
    ),
)


def _compile():
    codes, originals, proofs = [], [], []
    for cls, filename, digest, previous, target in _SPECS:
        path = Path(__file__).with_name(filename)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("Frozen native-staging helper changed: " + filename)
        klass = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == cls.__name__
        )
        fn = deepcopy(
            next(
                n
                for n in klass.body
                if isinstance(n, ast.FunctionDef) and n.name == "_moves"
            )
        )
        baseline = deepcopy(fn)
        module = compile(source, str(path), "exec", dont_inherit=True)
        kc = next(
            c for c in module.co_consts if getattr(c, "co_name", None) == cls.__name__
        )
        literal = next(
            c for c in kc.co_consts if getattr(c, "co_name", None) == "_moves"
        )
        original = cls._moves
        if _code_key(original.__code__) != _code_key(literal):
            raise ValueError("Native-staging helper is no longer literal: " + filename)
        if previous:
            (call,) = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == previous
            ]
            call.func = ast.Name(id=target, ctx=ast.Load())
        else:
            (begin,) = [
                i
                for i, n in enumerate(fn.body)
                if isinstance(n, ast.FunctionDef) and n.name == "index"
            ]
            oldbody = deepcopy(fn.body[begin : begin + 3])
            assert len(oldbody) == 3 and ast.unparse(oldbody[1]) == "held = 0"
            assert ast.unparse(oldbody[2].iter) == "runtime.particles"
            raw = ast.parse(
                "def _original_stage(self,runtime,cfg,prob,H_cons,cand_H,spawn_H,cache,keys,hypotheses,lookup,pending,staged): pass"
            ).body[0]
            raw.body = deepcopy(oldbody) + ast.parse("return held").body
            fn.body[begin : begin + 3] = ast.parse(
                "held = _rbpf_stage(self,runtime,cfg,prob,H_cons,cand_H,spawn_H,cache,keys,hypotheses,lookup,pending,staged)"
            ).body
        restored = deepcopy(fn)
        if previous:
            (call,) = [
                n
                for n in ast.walk(restored)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == target
            ]
            call.func = ast.parse(previous, mode="eval").body
        else:
            restored.body[begin : begin + 1] = deepcopy(oldbody)
        assert ast.dump(restored, include_attributes=False) == ast.dump(
            baseline, include_attributes=False
        )
        module = compile(
            ast.fix_missing_locations(
                ast.Module(body=([raw, fn] if not previous else [fn]), type_ignores=[])
            ),
            str(Path(__file__)),
            "exec",
            dont_inherit=True,
        )
        codes.append(
            next(c for c in module.co_consts if getattr(c, "co_name", None) == "_moves")
        )
        if not previous:
            rawcode = next(
                c
                for c in module.co_consts
                if getattr(c, "co_name", None) == "_original_stage"
            )
        originals.append(original)
        proofs.append(
            dict(
                source=filename,
                method=cls.__name__ + "._moves",
                source_sha256=digest,
                restored_AST_equal=True,
            )
        )
    return codes, originals, rawcode, proofs


_CODES, _ORIGINALS, _RAW_STAGE, AST_PROOFS = _compile()
_ORIGINAL_CODES = tuple(f.__code__ for f in _ORIGINALS)
_PRIORS = (
    "p_survive",
    "p_extend",
    "p_shrink",
    "p_release",
    "p_adopt_cons",
    "p_adopt",
    "p_spawn",
)


class StagingEngine:
    """No new cache: every canonical request and value lookup still executes."""

    def __init__(self, extension, owner, scope):
        spec = importlib.util.spec_from_file_location(
            "_rbpf_native_staging_probe", Path(extension).resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError("An explicit native-staging extension is required")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.owner = owner
        self.original = FunctionType(_RAW_STAGE, scope)
        self.native_calls = self.fallback_calls = self.native_particles = (
            self.native_moves
        ) = 0

    def stage(
        self,
        context,
        runtime,
        cfg,
        prob,
        H_cons,
        cand_H,
        spawn_H,
        cache,
        keys,
        hypotheses,
        lookup,
        pending,
        staged,
    ):
        owner = self.owner
        scope, base = self.original.__globals__, self.original.__builtins__
        ordinary = (
            scope.get("len", base.get("len")) is _LEN
            and type(runtime) is _MoveView
            and type(runtime.runtime) is _CanonicalView
            and type(cfg) is owner._scope["RbpfConfig"]
            and type(cfg).__getattribute__ is object.__getattribute__
            and all(
                type(cfg).__dict__.get(n) is None and type(vars(cfg).get(n)) is float
                for n in _PRIORS
            )
            and type(cand_H) is list
            and type(spawn_H) is list
            and type(cache) is dict
            and type(getattr(context.canonical, "__self__", None)) is _EpochMemo
            and getattr(context.canonical, "__func__", None) is _EpochMemo.canonical
        )
        if ordinary:
            actual = runtime.runtime._actual
            ordinary = (
                type(actual) is type(owner.runtime)
                and type(actual).__getattribute__ is object.__getattribute__
                and "particles" in vars(actual)
                and type(vars(actual)["particles"]) is list
            )
            particles = vars(actual)["particles"] if ordinary else ()
            ordinary = (
                ordinary
                and owner.mass_engine.particle_type.__getattribute__
                is object.__getattribute__
                and all(type(p) is owner.mass_engine.particle_type for p in particles)
            )
        args = (
            context,
            runtime,
            cfg,
            prob,
            H_cons,
            cand_H,
            spawn_H,
            cache,
            keys,
            hypotheses,
            lookup,
            pending,
            staged,
        )
        if ordinary:
            held = self.module.stage(*args)
            self.native_calls += 1
            self.native_particles += len(staged)
            self.native_moves += sum(map(len, staged))
            return held
        self.fallback_calls += 1
        return self.original(*args)

    def counts(self):
        return dict(
            native_calls=self.native_calls,
            fallback_calls=self.fallback_calls,
            native_particles=self.native_particles,
            native_moves=self.native_moves,
        )


class NativeStagingStep(NativeFeedbackMassStep):
    """D236 plus full candidate registration; original likelihood/choice tail.

    No D239 index aliases, D240 key rewrite, common-record/probability cache or
    profiler. All adapter/helper scope work and native allocations are timed.
    """

    def __init__(
        self,
        runtime,
        scorer=None,
        mode="native_cpu",
        *,
        staging_extension,
        **extensions,
    ):
        super().__init__(runtime, scorer, mode, **extensions)
        self._staging_clones = []
        for code, original in zip(_CODES, _ORIGINALS, strict=True):
            clone = FunctionType(
                code,
                dict(original.__globals__),
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            clone.__kwdefaults__ = original.__kwdefaults__
            self._staging_clones.append(clone)
        self.staging_engine = StagingEngine(
            staging_extension, self, self._staging_clones[0].__globals__
        )

    def __call__(self, rtk, ctx):
        for (cls, *_), original, code in zip(
            _SPECS, _ORIGINALS, _ORIGINAL_CODES, strict=True
        ):
            if cls._moves is not original or original.__code__ is not code:
                raise ValueError(
                    "Native-staging helper changed after binding: " + cls.__name__
                )
        for original, name, target in (
            (_ORIGINALS[1], "MoveBatchStep", MoveBatchStep),
            (_ORIGINALS[1], "_MoveView", _MoveView),
            (_ORIGINALS[2], "MoveReuseStep", MoveReuseStep),
            (_ORIGINALS[2], "_EpochMemo", _EpochMemo),
            (_ORIGINALS[3], "MoveKeyMemoStep", MoveKeyMemoStep),
        ):
            if original.__globals__.get(name) is not target:
                raise ValueError("Native-staging helper call target changed: " + name)
        additions = dict(
            _rbpf_stage=self.staging_engine.stage,
            _staging_raw_moves=self._staging_clones[0],
            _staging_reuse_moves=self._staging_clones[1],
            _staging_key_moves=self._staging_clones[2],
        )
        for clone, original in zip(self._staging_clones, _ORIGINALS, strict=True):
            clone.__globals__.clear()
            clone.__globals__.update(original.__globals__)
            clone.__globals__.update(additions)
        before, previous = self.staging_engine.counts(), self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if (
                self.last is not previous
                and self.last is not None
                and "batch" in self.last
            ):
                self.last["native_staging"] = {
                    k: v - before[k] for k, v in self.staging_engine.counts().items()
                }

    def _moves(self, *args, **kwargs):
        return self._staging_clones[3](self, *args, **kwargs)
