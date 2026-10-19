"""Exact native parent overlap on top of the frozen D233 complete PF caller."""

import builtins
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import FunctionType

from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256, make_oracle
from rbpf_native_canonical import NativeCanonicalStep, _code_key


def _literal_codes():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError('Frozen RBPF source changed')
    module = compile(source, str(SCALAR_SOURCE), 'exec', dont_inherit=True)
    klass = next(c for c in module.co_consts if getattr(c, 'co_name', None) == 'RbpfFgo')
    imported = next(c for c in klass.co_consts if getattr(c, 'co_name', None) == '_basin_parent_score')
    return (_code_key(make_oracle()._basin_parent_score.__code__), _code_key(imported))


_CODES = _literal_codes()
_SET, _SORTED, _LEN = builtins.set, builtins.sorted, builtins.len


class ParentScoreEngine:
    """Native exact builtin integers, otherwise one unchanged Python call.

    No parent/child cache, gauge conversion, fixed-width integers or FP64
    conversion. The reference is the minimum COMMON satellite. Both maps are
    preflighted before arithmetic so custom types retain callbacks/errors.
    Counting integer equalities is independent of group/satellite scan order;
    Python still executes the original cluster thresholds, ranking and ties.
    """

    def __init__(self, extension, original):
        if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
            raise RuntimeError('The explicit parent-score probe requires CPython 3.12')
        spec = importlib.util.spec_from_file_location('_rbpf_parent_score_probe', Path(extension).resolve())
        if spec is None or spec.loader is None:
            raise ValueError('An explicit parent-score extension is required')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original = original
        self.code = getattr(original, '__code__', None)
        self.literal = (type(original) is FunctionType
            and Path(original.__code__.co_filename).resolve() == SCALAR_SOURCE.resolve()
            and _code_key(original.__code__) in _CODES)
        self.native_calls = self.fallback_calls = 0

    def score(self, child, parent):
        function = self.original
        if self.literal and function.__code__ is self.code:
            scope, base = function.__globals__, function.__builtins__
            if (scope.get('set', base.get('set')) is _SET
                    and scope.get('sorted', base.get('sorted')) is _SORTED
                    and scope.get('len', base.get('len')) is _LEN):
                result = self.module.score(child, parent)
                if result is not None:
                    self.native_calls += 1
                    return result
        self.fallback_calls += 1
        return function(child, parent)

    def counts(self):
        return dict(native_calls=self.native_calls, fallback_calls=self.fallback_calls)


class NativeParentScoreStep(NativeCanonicalStep):
    """Redirect only the static integer-overlap method in the D233 local view.

    No actual runtime/module patch. Every existing bound method keeps the same
    code, including the entire cluster selection loop and all floating math.
    The numerical scorer, candidates, ordering, thresholds and RNG stay intact.
    """

    def __init__(self, runtime, scorer=None, mode='scalar', *, extension, parent_extension):
        super().__init__(runtime, scorer, mode, extension=extension)
        self.parent_engine = ParentScoreEngine(parent_extension, runtime._basin_parent_score)
        if not self.parent_engine.literal:
            raise ValueError('Native overlap requires the frozen literal parent score')
        engine = self.parent_engine

        def parent_score(_view, child, parent):
            return engine.score(child, parent)

        self._functions['_basin_parent_score'] = parent_score

    def __call__(self, rtk, ctx):
        before = self.parent_engine.counts()
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if self.last is not previous and self.last is not None and 'batch' in self.last:
                self.last['parent_score'] = {
                    key:value-before[key] for key,value in self.parent_engine.counts().items()}
