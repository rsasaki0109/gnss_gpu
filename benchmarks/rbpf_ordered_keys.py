"""D236 with exact ordered keys and allocation-free cache eligibility checks.

The original canonical calls, memo lifetime and all move math still execute.
No score-index aliases, profiler, integer normalization or numerical changes.
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
from rbpf_move_reuse import MoveReuseStep, _MoveView, _ordered_integer_key
from rbpf_native_canonical import NativeCanonicalStep, _code_key

_BINDINGS = tuple(
    getattr(builtins, n) for n in ("type", "dict", "tuple", "int", "any", "len")
)
_KEY_CODE = _ordered_integer_key.__code__
_SPECS = (
    (
        MoveReuseStep,
        "_moves",
        "rbpf_move_reuse.py",
        "9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612",
        "MoveBatchStep._moves",
        "_ordered_raw_moves",
    ),
    (
        MoveKeyMemoStep,
        "_moves",
        "rbpf_move_key_memo.py",
        "ddcc139b41212bfe813d552f6921b2914b6452571be410e2783a65f0d9cf9632",
        "MoveReuseStep._moves",
        "_ordered_reuse_moves",
    ),
    (
        NativeCanonicalStep,
        "_moves",
        "rbpf_native_canonical.py",
        "f177e3247617cca70e4aa0c4369f1856dba979703f478d62c1f05cd814c63bd2",
        "MoveKeyMemoStep._moves",
        "_ordered_key_moves",
    ),
    (
        _MoveView,
        "_key",
        "rbpf_move_reuse.py",
        "9d1c35239afcbac4c4ed17494c5072cb2fe793a24e8dd3548d5c85b2933b4612",
        None,
        None,
    ),
    (
        _EpochMemo,
        "canonical",
        "rbpf_epoch_memo.py",
        "178c4302ee1e6c6dcaea05be4e91161e82ce407168666cd864a1449f81bdedeb",
        None,
        None,
    ),
)


def _compile():
    codes, originals, proofs = [], [], []
    for cls, name, filename, digest, previous, target in _SPECS:
        path = Path(__file__).with_name(filename)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("Frozen ordered-key helper changed: " + filename)
        klass = next(
            n
            for n in ast.parse(source).body
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
        if filename == "rbpf_move_reuse.py":
            keycode = next(
                c
                for c in literal.co_consts
                if getattr(c, "co_name", None) == "_ordered_integer_key"
            )
            if _code_key(_KEY_CODE) != _code_key(keycode):
                raise ValueError("Ordered-key source helper is no longer literal")
        kc = next(
            c for c in literal.co_consts if getattr(c, "co_name", None) == cls.__name__
        )
        lc = next(c for c in kc.co_consts if getattr(c, "co_name", None) == name)
        original = getattr(cls, name)
        if _code_key(original.__code__) != _code_key(lc):
            raise ValueError("Ordered-key helper is no longer literal: " + filename)
        if previous:
            (call,) = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == previous
            ]
            call.func = ast.Name(id=target, ctx=ast.Load())
        elif cls is _EpochMemo:
            (test,) = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "_ordered_integer_key(hypothesis) is None"
            ]
            oldtest = deepcopy(test.test)
            test.test = ast.parse(
                "not _ordered_supported(hypothesis)", mode="eval"
            ).body
        restored = deepcopy(fn)
        if previous:
            (call,) = [
                n
                for n in ast.walk(restored)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == target
            ]
            call.func = ast.parse(previous, mode="eval").body
        elif cls is _EpochMemo:
            (test,) = [
                n
                for n in ast.walk(restored)
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "not _ordered_supported(hypothesis)"
            ]
            test.test = oldtest
        assert ast.dump(restored, include_attributes=False) == ast.dump(
            baseline, include_attributes=False
        )
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
                method=cls.__name__ + "." + name,
                source_sha256=digest,
                restored_AST_equal=True,
            )
        )
    # The raw staging/weights/probabilities/choice body is never transformed.
    raw = Path(__file__).with_name("rbpf_move_batch.py").read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != "69faf808e0a128bd11bc9d53d92ead87b47fda0a81a0c48d4299ab0510b7e7ea"
    ):
        raise ValueError("Frozen raw move helper changed")
    return codes, originals, proofs


_CODES, _ORIGINALS, AST_PROOFS = _compile()
_ORIGINAL_CODES = tuple(f.__code__ for f in _ORIGINALS)
_RAW = MoveBatchStep._moves
_RAW_CODE = _RAW.__code__


class OrderedKeyStep(NativeFeedbackMassStep):
    """Local method scopes; original raw moves and exact memo bookkeeping.

    Native calls check original helper code and builtin/global bindings on
    EVERY request, including after source-canonical callbacks. Custom bindings
    execute the original helper with its original callback/exception order.
    """

    def __init__(
        self,
        runtime,
        scorer=None,
        mode="native_cpu",
        *,
        ordered_extension,
        **extensions,
    ):
        super().__init__(runtime, scorer, mode, **extensions)
        spec = importlib.util.spec_from_file_location(
            "_rbpf_ordered_keys_probe", Path(ordered_extension).resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError("An explicit ordered-key extension is required")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.ordered_engine = module.OrderedKeys(
            _ordered_integer_key, _KEY_CODE, _BINDINGS
        )
        self._ordered_clones = []
        for code, original in zip(_CODES, _ORIGINALS, strict=True):
            clone = FunctionType(
                code,
                dict(original.__globals__),
                original.__name__,
                original.__defaults__,
                original.__closure__,
            )
            clone.__kwdefaults__ = original.__kwdefaults__
            self._ordered_clones.append(clone)
        self._ordered_view = type(
            "_OrderedMoveView", (_MoveView,), {"_key": self._ordered_clones[3]}
        )
        self._ordered_memo = type(
            "_OrderedEpochMemo", (_EpochMemo,), {"canonical": self._ordered_clones[4]}
        )

    def __call__(self, rtk, ctx):
        for (cls, name, *_), original, code in zip(
            _SPECS, _ORIGINALS, _ORIGINAL_CODES, strict=True
        ):
            if getattr(cls, name) is not original or original.__code__ is not code:
                raise ValueError(
                    "Ordered-key helper changed after binding: "
                    + cls.__name__
                    + "."
                    + name
                )
        if MoveBatchStep._moves is not _RAW or _RAW.__code__ is not _RAW_CODE:
            raise ValueError("Ordered-key raw helper changed after binding")
        for original, name, target in (
            (_ORIGINALS[0], "MoveBatchStep", MoveBatchStep),
            (_ORIGINALS[0], "_MoveView", _MoveView),
            (_ORIGINALS[1], "MoveReuseStep", MoveReuseStep),
            (_ORIGINALS[1], "_EpochMemo", _EpochMemo),
            (_ORIGINALS[2], "MoveKeyMemoStep", MoveKeyMemoStep),
        ):
            if original.__globals__.get(name) is not target:
                raise ValueError("Ordered-key helper call target changed: " + name)
        additions = dict(
            _ordered_raw_moves=_RAW,
            _ordered_reuse_moves=self._ordered_clones[0],
            _ordered_key_moves=self._ordered_clones[1],
            _MoveView=self._ordered_view,
            _EpochMemo=self._ordered_memo,
        )
        for clone, original in zip(self._ordered_clones, _ORIGINALS, strict=True):
            scope = clone.__globals__
            scope.clear()
            scope.update(original.__globals__)
            scope.update(additions)
        helper = _ORIGINALS[3].__globals__["_ordered_integer_key"]
        self._ordered_clones[3].__globals__["_ordered_integer_key"] = (
            self.ordered_engine.key if helper is _ordered_integer_key else helper
        )
        helper = _ORIGINALS[4].__globals__["_ordered_integer_key"]
        self._ordered_clones[4].__globals__["_ordered_supported"] = (
            self.ordered_engine.supported
            if helper is _ordered_integer_key
            else lambda h: helper(h) is not None
        )
        before, previous = self.ordered_engine.summary(), self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if (
                self.last is not previous
                and self.last is not None
                and "batch" in self.last
            ):
                self.last["ordered_keys"] = {
                    k: v - before[k] for k, v in self.ordered_engine.summary().items()
                }

    def _moves(self, *args, **kwargs):
        return self._ordered_clones[2](self, *args, **kwargs)
