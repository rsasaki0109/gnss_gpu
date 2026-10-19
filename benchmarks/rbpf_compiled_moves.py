"""Compile the complete literal move controller, preserving numerical calls."""

import builtins
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rbpf_epoch_memo import _EpochMemo
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_batch import MoveBatchStep
from rbpf_move_key_memo import MoveKeyMemoStep
from rbpf_move_reuse import MoveReuseStep, _MoveView
from rbpf_native_canonical import NativeCanonicalStep

SPECS = (MoveBatchStep, MoveReuseStep, MoveKeyMemoStep, NativeCanonicalStep)
_BUILTINS = {
    n: getattr(builtins, n)
    for n in ("dict", "float", "int", "len", "map", "max", "sum", "zip")
}
_NUMPY = {n: getattr(np, n) for n in ("array", "log", "exp", "max", "sum", "logaddexp")}
_LOG = math.log
_ABSENT = object()


def load_extension(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_rbpf_compiled_moves", path)
    if spec is None or spec.loader is None:
        raise ValueError("Explicit compiled-moves extension required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompiledMovesStep(NativeFeedbackMassStep):
    """D236 with a compiled literal controller, or the original off control.

    No input conversion, row batch, changed reduction, new RNG algorithm, or
    native numerical implementation. NumPy/scorer calls remain Python calls.
    Entry binding guards and counters are inside complete-step measurements.
    """

    def __init__(self, *args, compiled_extension, enabled=True, **kwargs):
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        super().__init__(*args, **kwargs)
        self.enabled = enabled
        self.module = load_extension(compiled_extension)
        self.compiled_move_calls = self.literal_move_calls = 0
        self._originals = tuple(cls._moves for cls in SPECS)
        self._codes = tuple(fn.__code__ for fn in self._originals)
        self._compiled_classes = tuple(
            getattr(self.module, cls.__name__) for cls in SPECS
        )
        self._compiled_functions = tuple(cls._moves for cls in self._compiled_classes)
        self._entry = self._compiled_functions[-1]
        aliases = dict(
            math=math,
            np=np,
            SimpleNamespace=SimpleNamespace,
            _EpochMemo=_EpochMemo,
            _MoveView=_MoveView,
        )
        self._module_bindings = dict(
            aliases,
            **{
                cls.__name__: new
                for cls, new in zip(SPECS, self._compiled_classes, strict=True)
            },
        )
        names = set(aliases) | {cls.__name__ for cls in SPECS} | set(_BUILTINS)
        self._global_bindings = [
            (fn.__globals__, {n: fn.__globals__.get(n, _ABSENT) for n in names})
            for fn in self._originals
        ]
        expected = {}
        import hashlib

        for cls, fn in zip(SPECS, self._originals, strict=True):
            path = Path(fn.__code__.co_filename)
            expected[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.module.SOURCE_METHOD_SHA256 != expected:
            raise ValueError(
                "Compiled method sources do not match current frozen helpers"
            )

    def _eligible(self):
        return (
            all(
                cls._moves is fn and fn.__code__ is code
                for cls, fn, code in zip(
                    SPECS, self._originals, self._codes, strict=True
                )
            )
            and all(
                cls._moves is fn
                for cls, fn in zip(
                    self._compiled_classes, self._compiled_functions, strict=True
                )
            )
            and all(
                vars(self.module).get(n) is value
                for n, value in self._module_bindings.items()
            )
            and all(
                scope.get(n, _ABSENT) is value
                for scope, expected in self._global_bindings
                for n, value in expected.items()
            )
            and all(getattr(builtins, n) is value for n, value in _BUILTINS.items())
            and all(getattr(np, n) is value for n, value in _NUMPY.items())
            and math.log is _LOG
        )

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        if self.enabled and self._eligible():
            self.compiled_move_calls += 1
            return self._entry(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.literal_move_calls += 1
        return NativeCanonicalStep._moves(
            self, prob, H_cons, cand_H, spawn_H, cache, ll_burst
        )

    def move_counts(self):
        return dict(compiled=self.compiled_move_calls, literal=self.literal_move_calls)
