"""Copy ambiguity cross-covariances from one existing GTSAM joint matrix.

Only the nested scalar block-copy loop is replaced. Position transforms,
diagonal/cross blocks, graph solves and held-state publication stay literal.
"""

import ast
import builtins
from copy import deepcopy
import hashlib
import inspect
from pathlib import Path
from types import FunctionType

import gtsam
import numpy as np
from cssrlib.gnss import Nav

from rbpf_native_canonical import _code_key

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/rbpf_fgo/gtsam_rtk_standalone.py"
SOURCE_SHA256 = "1cdf4a7931bce541a3ad0c41a7d7e7e5aa404ca11e47bacdbbd832b17e5f8f38"
GTSAM_PE_SHA256 = "76d018af6634358e5245c25e6676864abefa56044e3b5862b3361b78128f18a3"
_IB_SOURCE_SHA256 = "701779669f357b6401a751fa8f1c126fa84d604ccca311d456e3a53b0079bcbc"
_JOINT, _AT, _FULL = (
    gtsam.JointMarginal,
    gtsam.JointMarginal.at,
    gtsam.JointMarginal.fullMatrix,
)
_NUMPY = {
    n: getattr(np, n)
    for n in (
        "array",
        "asarray",
        "triu_indices",
        "shares_memory",
        "ndarray",
        "float64",
        "intp",
    )
}
_BUILTINS = {n: getattr(builtins, n) for n in ("enumerate", "int", "range")}


def scatter_pairs(target, dense, indices, offsets):
    """Independent stores only; use original active-order upper entries.

    Do not symmetrize or average the source matrix: the original loop reads
    at(k1,k2) for i<j, then mirrors exactly that value to both destinations.
    """
    i, j = np.triu_indices(len(indices), 1)
    values = dense[offsets[i], offsets[j]]
    target[indices[i], indices[j]] = values
    target[indices[j], indices[i]] = values


class JointCovarianceEngine:
    def __init__(self, original):
        source = SOURCE.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Frozen covariance writer changed")
        pe = Path(gtsam.gtsam.__file__)
        if hashlib.sha256(pe.read_bytes()).hexdigest() != GTSAM_PE_SHA256:
            raise ValueError("Uncertified GTSAM matrix layout")
        self.ib = original.IB
        ib_source = Path(self.ib.__code__.co_filename)
        if ".claude" in [p.casefold() for p in ib_source.parts]:
            raise ValueError("Excluded source path")
        # The literal index method is arithmetic on ordinary integer inputs.
        if hashlib.sha256(ib_source.read_bytes()).hexdigest() != _IB_SOURCE_SHA256:
            raise ValueError("Frozen ambiguity indexing source changed")
        literal_module = compile(
            ib_source.read_bytes(), str(ib_source), "exec", dont_inherit=True
        )
        pppos = next(
            c
            for c in literal_module.co_consts
            if getattr(c, "co_name", None) == "pppos"
        )
        ib_code = next(
            c for c in pppos.co_consts if getattr(c, "co_name", None) == "IB"
        )
        if _code_key(self.ib.__code__) != _code_key(ib_code):
            raise ValueError("Nonliteral ambiguity index method")
        self.ib_code = self.ib.__code__
        self.gnss = self.ib.__globals__["uGNSS"]
        self.maxsat = int(self.gnss.MAXSAT)
        self.calls = self.fast_calls = self.fallback_calls = self.full_matrix_calls = (
            self.copied_pairs
        ) = 0
        self.methods, self.proofs = {}, []
        tree = ast.parse(source)
        compiled = compile(source, str(SOURCE), "exec", dont_inherit=True)
        for owner, name, dim in (
            ("GtsamRtk", "_write_back", 3),
            ("GtsamRtkTc", "_write_back_tc", 6),
        ):
            fn = getattr(original, name)
            cls = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
            )
            node = deepcopy(
                next(
                    n
                    for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == name
                )
            )
            cls_code = next(
                c for c in compiled.co_consts if getattr(c, "co_name", None) == owner
            )
            literal = next(
                c for c in cls_code.co_consts if getattr(c, "co_name", None) == name
            )
            if fn.__code__.co_freevars or _code_key(fn.__code__) != _code_key(literal):
                raise ValueError("Expected literal covariance method: " + name)
            before = deepcopy(node)
            choices = [
                (body, i, n)
                for parent in ast.walk(node)
                for field, body in ast.iter_fields(parent)
                if field in ("body", "orelse", "finalbody") and isinstance(body, list)
                for i, n in enumerate(body)
                if isinstance(n, ast.For)
                and ast.unparse(n.target) == "(i, (s1, f1, k1))"
            ]
            if len(choices) != 1:
                raise ValueError("Cannot identify the covariance pair loop")
            body, index, loop = choices[0]
            assert ast.unparse(loop.iter) == "enumerate(active)"
            scope = dict(fn.__globals__)
            raw = ast.parse("def literal_pairs(self, active, jm): pass").body[0]
            raw.body = [deepcopy(loop)]
            raw_scope = {}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[raw], type_ignores=[])),
                    str(SOURCE),
                    "exec",
                    dont_inherit=True,
                ),
                raw_scope,
            )
            raw_fn = FunctionType(raw_scope["literal_pairs"].__code__, fn.__globals__)
            body[index] = ast.parse(
                f"_d272_copy_pairs(self, active, jm, key_x, {dim})"
            ).body[0]
            restored = deepcopy(node)

            class Undo(ast.NodeTransformer):
                def visit_Expr(self, expr):
                    if (
                        isinstance(expr.value, ast.Call)
                        and isinstance(expr.value.func, ast.Name)
                        and expr.value.func.id == "_d272_copy_pairs"
                    ):
                        return deepcopy(loop)
                    return self.generic_visit(expr)

            restored = Undo().visit(restored)
            assert ast.dump(restored, include_attributes=False) == ast.dump(
                before, include_attributes=False
            )
            code = compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(SOURCE),
                "exec",
                dont_inherit=True,
            )
            transformed = FunctionType(
                next(c for c in code.co_consts if getattr(c, "co_name", None) == name),
                scope,
                fn.__name__,
                fn.__defaults__,
            )
            transformed.__kwdefaults__ = fn.__kwdefaults__
            self.methods[name] = (
                fn,
                fn.__code__,
                fn.__defaults__,
                transformed,
                scope,
                raw_fn,
            )
            self.proofs.append(
                dict(
                    method=name,
                    restored_AST_equal=True,
                    position_dim=dim,
                    only_replaced="Nested ambiguity cross-covariance copy loop",
                )
            )

    def ordinary(self, rtk, active, jm, key_x, dim, scope):
        if not (
            type(rtk).__getattribute__ is object.__getattribute__
            and type(rtk.nav) is Nav
            and type(rtk.nav.na) is int
            and "IB" not in vars(rtk)
            and inspect.getattr_static(type(rtk), "IB") is self.ib
            and self.ib.__code__ is self.ib_code
            and self.ib.__globals__.get("uGNSS") is self.gnss
            and int(self.gnss.MAXSAT) == self.maxsat
            and type(jm) is _JOINT
            and gtsam.JointMarginal is _JOINT
            and _JOINT.at is _AT
            and _JOINT.fullMatrix is _FULL
            and scope.get("np") is np
            and scope.get("gtsam") is gtsam
            and all(getattr(np, n) is v for n, v in _NUMPY.items())
            and all(
                scope.get(n, builtins.__dict__[n]) is v for n, v in _BUILTINS.items()
            )
            and type(active) is list
            and len(active) <= 512
            and type(key_x) is int
            and 0 <= key_x < 2**64
            and dim in (3, 6)
        ):
            return False
        p = rtk.nav.P
        if not (
            type(p) is np.ndarray
            and p.dtype == np.float64
            and p.ndim == 2
            and p.flags.writeable
            and p.flags.aligned
            and (p.flags.c_contiguous or p.flags.f_contiguous)
        ):
            return False
        return all(
            type(row) is tuple
            and len(row) == 3
            and all(type(v) is int for v in row)
            and 1 <= row[0] <= self.maxsat
            and 0 <= row[1] < 3
            and 0 <= row[2] < 2**64
            for row in active
        )

    def copy_pairs(self, rtk, active, jm, key_x, dim, scope, literal):
        self.calls += 1
        if not self.ordinary(rtk, active, jm, key_x, dim, scope):
            self.fallback_calls += 1
            return literal(rtk, active, jm)
        keys = [row[2] for row in active]
        indices = [self.ib(rtk, s, f, rtk.nav.na) for s, f, _ in active]
        p = rtk.nav.P
        if (
            len(set([key_x, *keys])) != len(keys) + 1
            or len(set(indices)) != len(indices)
            or any(not 0 <= i < min(p.shape) for i in indices)
        ):
            self.fallback_calls += 1
            return literal(rtk, active, jm)
        if len(keys) < 2:
            self.fast_calls += 1
            return None
        dense = jm.fullMatrix()
        if not (
            type(dense) is np.ndarray
            and dense.dtype == np.float64
            and dense.shape == (len(keys) + dim, len(keys) + dim)
            and not np.shares_memory(dense, p)
        ):
            self.fallback_calls += 1
            return literal(rtk, active, jm)
        starts, offset = {}, 0
        for key in sorted([key_x, *keys]):
            starts[key] = offset
            offset += dim if key == key_x else 1
        scatter_pairs(
            p,
            dense,
            np.asarray(indices, dtype=np.intp),
            np.asarray([starts[k] for k in keys], dtype=np.intp),
        )
        self.fast_calls += 1
        self.full_matrix_calls += 1
        self.copied_pairs += len(keys) * (len(keys) - 1) // 2
        return None

    def counts(self):
        return dict(
            calls=self.calls,
            fast_calls=self.fast_calls,
            fallback_calls=self.fallback_calls,
            full_matrix_calls=self.full_matrix_calls,
            copied_pairs=self.copied_pairs,
        )


def make_type(original):
    engine = JointCovarianceEngine(original)
    methods = {}
    for name, (
        fn,
        code,
        defaults,
        transformed,
        scope,
        literal,
    ) in engine.methods.items():

        def wrap(
            self_rtk,
            *args,
            _fn=fn,
            _code=code,
            _defaults=defaults,
            _transformed=transformed,
            _scope=scope,
            _literal=literal,
            **kwargs,
        ):
            if _fn.__code__ is not _code or _fn.__defaults__ is not _defaults:
                return _fn(self_rtk, *args, **kwargs)
            _scope.clear()
            _scope.update(_fn.__globals__)
            _scope["_d272_copy_pairs"] = lambda rtk, active, jm, key_x, dim: (
                engine.copy_pairs(rtk, active, jm, key_x, dim, _scope, _literal)
            )
            return _transformed(self_rtk, *args, **kwargs)

        methods[name] = wrap
    return type("JointCovariance" + original.__name__, (original,), methods), engine
