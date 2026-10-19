"""Explicit row-batched proposal arithmetic; every particle still draws once.

No proposal cache, shared probability template, changed candidates or CUDA.
Rows of equal length use contiguous NumPy fast-axis reductions. Exact bits
must be validated on the recorded NumPy build before any speed acceptance.
"""

import ast
import builtins
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from types import FunctionType, MemberDescriptorType

import numpy as np

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_native_staging import _SPECS

_NP = {
    n: getattr(np, n)
    for n in (
        "array",
        "log",
        "exp",
        "max",
        "sum",
        "isfinite",
        "all",
        "errstate",
        "geterr",
    )
}
_BUILTINS = {n: getattr(builtins, n) for n in ("float", "int", "len", "zip")}
_LOG = math.log
_GENERATORS = (
    np.random.PCG64,
    np.random.PCG64DXSM,
    np.random.Philox,
    np.random.SFC64,
    np.random.MT19937,
)
MAX_CELLS = 131072


def prepare_rows(staged, likelihood):
    """Compute all rows once; return None before draws for unsafe numbers.

    Suppressed probe arithmetic cannot hide original warnings/errors: an
    invalid row returns to the literal selection loop for the entire call.
    Only exact built-in floats enter this probe, so conversion runs no hooks.
    """
    if (
        type(staged) is not list
        or type(likelihood) is not list
        or any(type(v) is not float for v in likelihood)
    ):
        return None
    groups, cells = {}, 0
    for i, records in enumerate(staged):
        if type(records) is not list or not records:
            return None
        for record in records:
            if (
                type(record) is not tuple
                or len(record) != 4
                or type(record[0]) is not float
                or type(record[1]) is not int
                or not 0 <= record[1] < len(likelihood)
            ):
                return None
        cells += len(records)
        if cells > MAX_CELLS:
            return None
        groups.setdefault(len(records), []).append(i)
    prepared = [None] * len(staged)
    with np.errstate(all="ignore"):
        for width, indices in groups.items():
            records = [staged[i] for i in indices]
            pri = np.array([[r[0] for r in row] for row in records], dtype=float)
            values = np.array([[likelihood[r[1]] for r in row] for row in records])
            if not (
                np.all(np.isfinite(pri))
                and np.all(pri > 0.0)
                and np.all(np.isfinite(values))
            ):
                return None
            # Same row order/width and contiguous reduction as the 1-D source.
            pri /= pri.sum(axis=1)[:, None]
            if not (np.all(pri > 0.0) and np.all(np.isfinite(pri))):
                return None
            lam = np.log(pri) + values
            if not np.all(np.isfinite(lam)):
                return None
            mx = np.max(lam, axis=1)
            wsum = np.exp(lam - mx[:, None])
            tot = np.sum(wsum, axis=1)
            probabilities = wsum / tot[:, None]
            if not (
                np.all(np.isfinite(tot))
                and np.all(tot > 0.0)
                and np.all(np.isfinite(probabilities))
            ):
                return None
            assert probabilities.shape == (len(indices), width)
            for j, i in enumerate(indices):
                prepared[i] = (float(mx[j]), float(tot[j]), probabilities[j])
    return prepared, dict(
        rows=len(staged),
        cells=cells,
        row_batches=len(groups),
        probability_payload_bytes=cells * 8,
    )


def _compile():
    codes, originals, proofs = [], [], []
    for cls, filename, digest, previous, target in _SPECS:
        path = Path(__file__).with_name(filename)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError("Frozen weight-row caller changed: " + filename)
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
        original = cls._moves
        if previous:
            (call,) = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func) == previous
            ]
            call.func = ast.Name(id=target, ctx=ast.Load())
        else:
            (index,) = [
                i
                for i, n in enumerate(fn.body)
                if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "zip(runtime.particles, staged)"
            ]
            old = deepcopy(fn.body[index - 1 : index + 1])
            assert ast.unparse(old[0]) == "new_states = []"
            literal = ast.parse(
                "def _literal_choices(runtime, staged, likelihood): pass"
            ).body[0]
            literal.body = deepcopy(old) + ast.parse("return new_states").body
            fn.body[index - 1 : index + 1] = ast.parse(
                "new_states = _weight_choices(runtime, staged, likelihood, _literal_choices)"
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
            restored.body[index - 1 : index] = old
        assert ast.dump(restored, include_attributes=False) == ast.dump(
            baseline, include_attributes=False
        )
        module = compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=([literal, fn] if not previous else [fn]), type_ignores=[]
                )
            ),
            str(Path(__file__)),
            "exec",
            dont_inherit=True,
        )
        codes.append(
            next(c for c in module.co_consts if getattr(c, "co_name", None) == "_moves")
        )
        if not previous:
            literal_code = next(
                c
                for c in module.co_consts
                if getattr(c, "co_name", None) == "_literal_choices"
            )
        originals.append(original)
        proofs.append(
            dict(source=filename, source_sha256=digest, restored_AST_equal=True)
        )
    return codes, originals, literal_code, proofs


_CODES, _ORIGINALS, _LITERAL, AST_PROOFS = _compile()
_ORIGINAL_CODES = tuple(f.__code__ for f in _ORIGINALS)


class WeightRowsStep(NativeFeedbackMassStep):
    """D236 with row-batched preparation, or a literal-tail clone control."""

    def __init__(self, *args, enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        if type(enabled) is not bool:
            raise TypeError("enabled must be an explicit bool")
        self.enabled, self.last_weight_rows = enabled, None
        self._row_clones = []
        for code, original in zip(_CODES, _ORIGINALS, strict=True):
            self._row_clones.append(
                FunctionType(
                    code,
                    dict(original.__globals__),
                    original.__name__,
                    original.__defaults__,
                    original.__closure__,
                )
            )
        self._literal = FunctionType(_LITERAL, self._row_clones[0].__globals__)
        self._particle = self._source_globals["Particle"]
        self._slots = {
            n: self._particle.__dict__[n] for n in ("H", "logw", "last_move")
        }

    def _choices(self, runtime, staged, likelihood, literal):
        self.last_weight_rows = dict(
            enabled=self.enabled,
            used=False,
            rows=len(staged),
            cells=sum(map(len, staged)),
            row_batches=0,
            probability_payload_bytes=0,
        )
        scope, base = literal.__globals__, literal.__builtins__
        ordinary = (
            self.enabled
            and scope.get("np") is np
            and scope.get("math") is math
            and math.log is _LOG
            and all(getattr(np, n) is v for n, v in _NP.items())
            and all(scope.get(n, base.get(n)) is v for n, v in _BUILTINS.items())
            and type(runtime.rng) is np.random.Generator
            and type(runtime.rng.bit_generator) in _GENERATORS
            and self._particle.__getattribute__ is object.__getattribute__
            and self._particle.__setattr__ is object.__setattr__
            and all(
                self._particle.__dict__.get(n) is v and type(v) is MemberDescriptorType
                for n, v in self._slots.items()
            )
            and all(
                type(p) is self._particle and type(p.logw) is float
                for p in runtime.particles
            )
        )
        prepared = prepare_rows(staged, likelihood) if ordinary else None
        if prepared is None:
            return literal(runtime, staged, likelihood)
        values, stats = prepared
        self.last_weight_rows.update(stats, used=True)
        new_states = []
        for particle, records, (mx, tot, probabilities) in zip(
            runtime.particles, staged, values
        ):
            # Original scalar math and sequential RNG/particle mutation order.
            particle.logw += mx + math.log(tot)
            pick = int(runtime.rng.choice(len(records), p=probabilities))
            new_states.append((records[pick][2], records[pick][3]))
        return new_states

    def __call__(self, rtk, ctx):
        for (cls, *_), original, code in zip(
            _SPECS, _ORIGINALS, _ORIGINAL_CODES, strict=True
        ):
            if cls._moves is not original or original.__code__ is not code:
                raise ValueError(
                    "Weight-row caller changed after binding: " + cls.__name__
                )
        for original, name, target in (
            (_ORIGINALS[1], "MoveBatchStep", _SPECS[0][0]),
            (_ORIGINALS[2], "MoveReuseStep", _SPECS[1][0]),
            (_ORIGINALS[3], "MoveKeyMemoStep", _SPECS[2][0]),
        ):
            if original.__globals__.get(name) is not target:
                raise ValueError("Weight-row call target changed: " + name)
        additions = dict(
            _weight_choices=self._choices,
            _literal_choices=self._literal,
            _staging_raw_moves=self._row_clones[0],
            _staging_reuse_moves=self._row_clones[1],
            _staging_key_moves=self._row_clones[2],
        )
        for clone, original in zip(self._row_clones, _ORIGINALS, strict=True):
            clone.__globals__.clear()
            clone.__globals__.update(original.__globals__)
            clone.__globals__.update(additions)
        self.last_weight_rows = None
        previous = self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if (
                self.last is not previous
                and self.last is not None
                and "batch" in self.last
            ):
                self.last["weight_rows"] = self.last_weight_rows

    def _moves(self, *args, **kwargs):
        return self._row_clones[3](self, *args, **kwargs)
