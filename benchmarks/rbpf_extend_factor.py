"""Call-local exact-matrix LU reuse for the frozen conditional extend solve.

Explicit Windows/NumPy-build experiment. All gathers, right-hand sides,
matrix products, rounding, candidates and original PF decisions remain.
"""

import ast
from copy import deepcopy
import ctypes
import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256

_SOLVE = np.linalg.solve
_FINITE, _ABS, _MAX, _ARRAY = np.isfinite, np.abs, np.max, np.array
_BLAS_SHA = "860d95b1c38e637ce4509f5fa24fbf2a98ba8696f9f3d28bf184bee74ad9a325"
_BLAS_NAME = "libscipy_openblas64_-860d95b1c38e637ce4509f5fa24fbf2a.dll"
_F64 = np.dtype("float64")


def _compile():
    source = SCALAR_SOURCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == SCALAR_SHA256
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo")
    method = deepcopy(next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_extend"))
    original = deepcopy(method)
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call) and ast.unparse(n.func) == "np.linalg.solve"]
    assert len(calls) == 1
    assert ast.unparse(calls[0]) == "np.linalg.solve(QHH, prob.y[idxH] - dd[idxH])"
    calls[0].func = ast.Name(id="_rbpf_extend_solve", ctx=ast.Load())
    restored = deepcopy(method)
    call = next(n for n in ast.walk(restored) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_rbpf_extend_solve")
    call.func = ast.parse("np.linalg.solve", mode="eval").body
    assert ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False)
    module = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(Path(__file__)), "exec", dont_inherit=True)
    return next(c for c in module.co_consts if getattr(c, "co_name", None) == "_extend")


_CODE = _compile()
AST_PROOF = dict(restored_AST_equal=True, source_sha256=SCALAR_SHA256, replacement="one np.linalg.solve call in _extend only")


class ExtendFactorCache:
    """Cache only exact matrix bytes; factorization never depends on the RHS.

    Uses the same pinned ILP64 LAPACK library as NumPy. No inverse, batched
    RHS, mixed precision, changed pivoting, approximate key or retained epoch.
    Unusual inputs/functions, singular factors and nonfinite results execute
    the original solve. A finite bounded input guard limits this prototype.
    """

    def __init__(self):
        if np.__version__ != "2.3.4":
            raise RuntimeError("This explicit experiment requires NumPy 2.3.4")
        path = Path(np.__file__).resolve().parent.parent / "numpy.libs" / _BLAS_NAME
        if hashlib.sha256(path.read_bytes()).hexdigest() != _BLAS_SHA:
            raise RuntimeError("The pinned NumPy LAPACK library changed")
        self.library = ctypes.CDLL(str(path))
        self.factor = self.library.scipy_dgetrf_64_
        self.factor.argtypes = [ctypes.c_void_p] * 6
        self.factor.restype = None
        self.substitute = self.library.scipy_dgetrs_64_
        self.substitute.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_size_t]
        self.substitute.restype = None
        self.cache = None
        self.counts = dict(requests=0, factorizations=0, factor_hits=0, substitutions=0, fallbacks=0)
        self.peak_payload_bytes = 0

    def begin(self):
        if self.cache is not None:
            raise RuntimeError("Concurrent factor-cache epochs are unsupported")
        self.cache = {}

    def end(self):
        self.cache = None

    def solve(self, matrix, rhs):
        self.counts["requests"] += 1
        ordinary = (
            self.cache is not None
            and np.linalg.solve is _SOLVE
            and np.isfinite is _FINITE and np.abs is _ABS
            and np.max is _MAX and np.array is _ARRAY
            and type(matrix) is np.ndarray and matrix.dtype == _F64
            and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
            and 0 < matrix.shape[0] <= 128 and matrix.flags.aligned
            and type(rhs) is np.ndarray and rhs.dtype == _F64
            and rhs.shape == (matrix.shape[0],) and rhs.flags.aligned
        )
        if ordinary:
            ordinary = bool(_FINITE(matrix).all() and _FINITE(rhs).all()
                            and _MAX(_ABS(matrix)) <= 2.0**500
                            and _MAX(_ABS(rhs)) <= 2.0**500)
        if not ordinary:
            return self._fallback(matrix, rhs)
        size = matrix.shape[0]
        # Full bytes preserve changed matrices, strides, signed zero and 1 ULP.
        # Gathers remain original, so no immutable-problem assumption is needed.
        key = size, matrix.tobytes(order="C")
        pair = self.cache.get(key)
        n, one, info = ctypes.c_int64(size), ctypes.c_int64(1), ctypes.c_int64(0)
        if pair is None:
            if len(self.cache) >= 128:
                return self._fallback(matrix, rhs)
            lu = _ARRAY(matrix, dtype=_F64, order="F", copy=True)
            piv = np.empty(size, dtype=np.int64)
            self.factor(ctypes.byref(n), ctypes.byref(n), lu.ctypes.data,
                        ctypes.byref(n), piv.ctypes.data, ctypes.byref(info))
            self.counts["factorizations"] += 1
            if info.value != 0 or not _FINITE(lu).all():
                return self._fallback(matrix, rhs)
            pair = lu, piv
            self.cache[key] = pair
            self.peak_payload_bytes = max(self.peak_payload_bytes, sum(len(k[1]) + v[0].nbytes + v[1].nbytes for k, v in self.cache.items()))
        else:
            self.counts["factor_hits"] += 1
        lu, piv = pair
        out = _ARRAY(rhs, dtype=_F64, order="C", copy=True)
        trans = ctypes.c_char(b"N")
        self.substitute(ctypes.byref(trans), ctypes.byref(n), ctypes.byref(one),
                        lu.ctypes.data, ctypes.byref(n), piv.ctypes.data,
                        out.ctypes.data, ctypes.byref(n), ctypes.byref(info), 1)
        self.counts["substitutions"] += 1
        if info.value != 0 or not _FINITE(out).all():
            return self._fallback(matrix, rhs)
        return out

    def _fallback(self, matrix, rhs):
        self.counts["fallbacks"] += 1
        return np.linalg.solve(matrix, rhs)


class ExtendFactorStep(NativeFeedbackMassStep):
    """D236 with one exact-matrix solve substitution and call-local cleanup."""

    def __init__(self, runtime, scorer=None, mode="scalar", **extensions):
        super().__init__(runtime, scorer, mode, **extensions)
        self.extend_factor = ExtendFactorCache()
        original = self._source_functions["_extend"]
        replacement = FunctionType(_CODE, self._scope, original.__name__, original.__defaults__, original.__closure__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        self._functions["_extend"] = replacement
        base = self.compiled

        def call(view, rtk, ctx):
            self._scope["_rbpf_extend_solve"] = self.extend_factor.solve
            try:
                return base(view, rtk, ctx)
            finally:
                del self._scope["_rbpf_extend_solve"]

        self.compiled = call

    def __call__(self, rtk, ctx):
        engine = self.extend_factor
        before, previous = engine.counts.copy(), self.last
        engine.begin()
        try:
            return super().__call__(rtk, ctx)
        finally:
            engine.end()
            if self.last is not previous and self.last is not None and "batch" in self.last:
                self.last["extend_factor"] = {k: v - before[k] for k, v in engine.counts.items()}
