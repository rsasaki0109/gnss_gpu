"""Exact support extraction and integer assembly for the D236 extend move."""

import ast
import builtins
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType

import numpy as np

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_likelihood_cases import SCALAR_SOURCE, SCALAR_SHA256
from rbpf_native_canonical import _code_key


def _compile():
    source = SCALAR_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SCALAR_SHA256:
        raise ValueError("Frozen RBPF source changed")
    tree = ast.parse(source)
    klass = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RbpfFgo"
    )
    method = deepcopy(
        next(
            n
            for n in klass.body
            if isinstance(n, ast.FunctionDef) and n.name == "_extend"
        )
    )
    baseline = deepcopy(method)
    support = deepcopy(method.body[1:4])
    assert [ast.unparse(n.targets[0]) for n in support] == ["dd", "idxH", "idxC"]
    assert ast.unparse(support[0].value) == "self._pair_dd(H, prob)"
    tail = deepcopy(method.body[-3:])
    assert ast.unparse(tail[0]) == "H2 = _copy_H(H)"
    assert ast.unparse(tail[1].iter) == "enumerate(idxC)"
    assert ast.unparse(tail[-1]) == "return H2"
    raw_support = ast.parse("def _original_support(self, H, prob): pass").body[0]
    raw_support.body = support + ast.parse("return dd, idxH, idxC").body
    raw_finish = ast.parse("def _original_finish(H, prob, idxC, b_cond): pass").body[0]
    raw_finish.body = tail
    method.body[-3:] = ast.parse(
        "return _rbpf_extend_finish(H, prob, idxC, b_cond)"
    ).body
    method.body[1:4] = ast.parse(
        "dd, idxH, idxC = _rbpf_extend_support(self, H, prob)"
    ).body
    restored = deepcopy(method)
    restored.body[1:2] = deepcopy(support)
    restored.body[-1:] = deepcopy(tail)
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        baseline, include_attributes=False
    )
    module = compile(
        ast.fix_missing_locations(
            ast.Module(body=[raw_support, raw_finish, method], type_ignores=[])
        ),
        str(Path(__file__)),
        "exec",
        dont_inherit=True,
    )
    codes = {c.co_name: c for c in module.co_consts if hasattr(c, "co_name")}
    copy_node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_H"
    )
    compiled_copy = compile(
        ast.fix_missing_locations(ast.Module(body=[copy_node], type_ignores=[])),
        str(SCALAR_SOURCE),
        "exec",
        dont_inherit=True,
    )
    codes["copy_literal"] = next(
        c for c in compiled_copy.co_consts if getattr(c, "co_name", None) == "_copy_H"
    )
    return codes


_CODES = _compile()
AST_PROOF = dict(
    source_sha256=SCALAR_SHA256,
    method="RbpfFgo._extend",
    restored_AST_equal=True,
    original_solve_matmul_rint_integer_order=True,
)
_BINDINGS = {n: getattr(builtins, n) for n in ("len", "enumerate", "int", "dict")}
_ARRAY, _F64, _INTP = np.ndarray, np.dtype("float64"), np.dtype("intp")
_FULL, _ARRAY_FN, _FINITE, _WHERE, _RINT, _NAN = (
    np.full,
    np.array,
    np.isfinite,
    np.where,
    np.rint,
    np.nan,
)


class ExtendSupportEngine:
    """No numerical solve rewrite, approximate integer, or retained input cache."""

    def __init__(self, extension, component):
        self.component, self.scope = component, component.scope
        spec = importlib.util.spec_from_file_location(
            "_rbpf_extend_support_probe", Path(extension).resolve()
        )
        if spec is None or spec.loader is None:
            raise ValueError("An explicit extend-support extension is required")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.original_support = FunctionType(_CODES["_original_support"], self.scope)
        self.original_finish = FunctionType(_CODES["_original_finish"], self.scope)
        self.copy = self.scope["_copy_H"]
        self.copy_code = self.copy.__code__
        self.copy_literal = _code_key(self.copy_code) == _code_key(
            _CODES["copy_literal"]
        )
        self.support_native = self.support_fallback = self.finish_native = (
            self.finish_fallback
        ) = 0
        self.native_pairs = self.assembled_integers = 0

    def _ordinary(self, problem):
        component = self.component
        scope, base = self.scope, self.original_support.__builtins__
        policy = np.geterr()
        return (
            component._ordinary()
            and scope.get("np") is np
            and all(scope.get(k, base.get(k)) is v for k, v in _BINDINGS.items())
            and np.full is _FULL
            and np.array is _ARRAY_FN
            and np.isfinite is _FINITE
            and np.where is _WHERE
            and np.nan is _NAN
            and type(problem) is component.problem_type
            and component.problem_type.__getattribute__ is object.__getattribute__
            and component.problem_type.__dict__.get("pairs") is None
            and "pairs" in vars(problem)
            and type(vars(problem)["pairs"]) is list
            and policy["under"] == "ignore"
            and all(v in ("ignore", "warn") for v in policy.values())
        )

    def support(self, runtime, hypothesis, problem):
        if self._ordinary(problem):
            pairs = problem.pairs
            dd = np.full(len(pairs), np.nan)
            result = self.module.support(hypothesis, pairs, dd)
            if result is not None:
                held, missing = result
                self.support_native += 1
                self.native_pairs += len(pairs)
                return dd, np.array(held, dtype=_INTP), missing
        self.support_fallback += 1
        return self.original_support(runtime, hypothesis, problem)

    def finish(self, hypothesis, problem, indices, conditional):
        scope, base = self.scope, self.original_finish.__builtins__
        ordinary = (
            self._ordinary(problem)
            and self.copy_literal
            and scope.get("_copy_H") is self.copy
            and self.copy.__code__ is self.copy_code
            and self.copy.__globals__.get("dict", self.copy.__builtins__.get("dict"))
            is _BINDINGS["dict"]
            and scope.get("int", base.get("int")) is _BINDINGS["int"]
            and np.rint is _RINT
            and type(conditional) is _ARRAY
            and conditional.dtype == _F64
            and conditional.ndim == 1
            and conditional.flags.c_contiguous
            and conditional.flags.aligned
        )
        if ordinary:
            result = self.module.finish(
                hypothesis, problem.pairs, indices, conditional, _RINT
            )
            if result is not None:
                self.finish_native += 1
                self.assembled_integers += len(indices)
                return result
        self.finish_fallback += 1
        return self.original_finish(hypothesis, problem, indices, conditional)

    def counts(self):
        return dict(
            support_native=self.support_native,
            support_fallback=self.support_fallback,
            finish_native=self.finish_native,
            finish_fallback=self.finish_fallback,
            native_pairs=self.native_pairs,
            assembled_integers=self.assembled_integers,
        )


class NativeExtendSupportStep(NativeFeedbackMassStep):
    """Explicit D236 caller; only pure extend support and assembly are grouped."""

    def __init__(
        self, runtime, scorer=None, mode="scalar", *, extend_extension, **extensions
    ):
        super().__init__(runtime, scorer, mode, **extensions)
        self.extend_engine = ExtendSupportEngine(extend_extension, self.mass_engine)
        original = self._source_functions["_extend"]
        replacement = FunctionType(
            _CODES["_extend"],
            self._scope,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        replacement.__kwdefaults__ = original.__kwdefaults__
        self._functions["_extend"] = replacement
        base = self.compiled

        def call(view, rtk, ctx):
            self._scope["_rbpf_extend_support"] = self.extend_engine.support
            self._scope["_rbpf_extend_finish"] = self.extend_engine.finish
            try:
                return base(view, rtk, ctx)
            finally:
                del self._scope["_rbpf_extend_support"]
                del self._scope["_rbpf_extend_finish"]

        self.compiled = call

    def __call__(self, rtk, ctx):
        before, previous = self.extend_engine.counts(), self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if (
                self.last is not previous
                and self.last is not None
                and "batch" in self.last
            ):
                self.last["extend_support"] = {
                    k: v - before[k] for k, v in self.extend_engine.counts().items()
                }
