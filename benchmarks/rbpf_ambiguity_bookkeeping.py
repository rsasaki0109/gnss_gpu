"""Batch independent integer ambiguity bookkeeping, retain reset/call order.

Experimental frozen-cssrlib adapter. No observation, estimator equation,
floating-point expression, prior, threshold, particle or RNG changes.
"""

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np
from cssrlib.gnss import Nav

from rbpf_ambiguity_initx_checked import make_type as make_initx_type
from rbpf_native_canonical import _code_key

SOURCE_SHA256 = "24c941e8f05205df7825961ac7c2791c46994f5057cc44a7df57dab405cf3b70"


def _method(value):
    return getattr(value, "__func__", value)


class BookkeepingEngine:
    def __init__(self, original, initx_type, initx_engine):
        self.original = original.update_ambiguities
        self.initx_engine = initx_engine
        path = Path(self.original.__code__.co_filename)
        if ".claude" in [p.casefold() for p in path.parts]:
            raise ValueError("Excluded source path")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Frozen ambiguity bookkeeping source changed")
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "rtkpos")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "update_ambiguities")
        compiled = compile(source, str(path), "exec", dont_inherit=True)
        cls_code = next(c for c in compiled.co_consts if getattr(c, "co_name", None) == "rtkpos")
        self.codes = {}
        for name in ("update_ambiguities", "_row_has_nonzero"):
            fn = getattr(original, name)
            literal = next(c for c in cls_code.co_consts if getattr(c, "co_name", None) == name)
            if _code_key(fn.__code__) != _code_key(literal):
                raise ValueError("Expected literal ambiguity method: " + name)
            self.codes[fn] = fn.__code__
        self.initx = initx_type.initx
        self.ib, self.predicate = original.IB, original._row_has_nonzero
        ib_path = Path(self.ib.__code__.co_filename)
        if ".claude" in [p.casefold() for p in ib_path.parts]:
            raise ValueError("Excluded source path")
        ib_source = ib_path.read_bytes()
        if hashlib.sha256(ib_source).hexdigest() != "701779669f357b6401a751fa8f1c126fa84d604ccca311d456e3a53b0079bcbc":
            raise ValueError("Frozen ambiguity index source changed")
        ib_module = compile(ib_source, str(ib_path), "exec", dont_inherit=True)
        ib_class = next(c for c in ib_module.co_consts if getattr(c, "co_name", None) == "pppos")
        ib_code = next(c for c in ib_class.co_consts if getattr(c, "co_name", None) == "IB")
        if _code_key(self.ib.__code__) != _code_key(ib_code):
            raise ValueError("Expected literal ambiguity index method")
        self.codes[self.ib] = self.ib.__code__
        self.codes[self.initx_engine.original] = self.initx_engine.original.__code__
        self.globals = self.original.__globals__
        self.bindings = {n: self.globals[n] for n in self.original.__code__.co_names if n in self.globals}
        self.range = self.globals.get("range", self.original.__builtins__["range"])
        self.np_any, self.np = np.any, np
        self.maxsat = int(self.globals["uGNSS"].MAXSAT)
        sat2prn = self.globals["sat2prn"]
        self.codes[sat2prn] = sat2prn.__code__
        groups = {}
        for i in range(self.maxsat):
            system, _ = sat2prn(i + 1)
            groups.setdefault(system, []).append(i)
        self.system_groups = tuple((s, np.array(v, dtype=np.intp)) for s, v in groups.items())
        self.key_types = (int, self.globals["uGNSS"])
        self.updates = self.original_updates = self.scans = self.fast_scans = self.literal_scans = 0

        before = deepcopy(node)
        loop = next(n for n in node.body if isinstance(n, ast.For))
        scan = loop.body[0]
        assert isinstance(scan, ast.For) and ast.unparse(scan.iter) == "range(uGNSS.MAXSAT)"
        literal_node = ast.FunctionDef(
            name="literal_scan",
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in ("self", "obs", "f")],
                kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[deepcopy(scan)], decorator_list=[],
        )
        scope = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[literal_node], type_ignores=[])),
            str(path), "exec", dont_inherit=True), scope)
        self.literal = FunctionType(scope["literal_scan"].__code__, self.globals)
        loop.body[0] = ast.parse("_d247_engine.scan(self, obs, f)").body[0]
        restored = deepcopy(node)
        next(n for n in restored.body if isinstance(n, ast.For)).body[0] = scan
        assert ast.dump(restored, include_attributes=False) == ast.dump(before, include_attributes=False)
        self.proof = dict(restored_update_AST_equal=True, source_sha256=SOURCE_SHA256,
            changed="Only the first integer bookkeeping loop in each frequency;original observation/floating-point loop retained verbatim")
        scope = dict(self.globals, _d247_engine=self)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path), "exec", dont_inherit=True), scope)
        self.transformed = scope["update_ambiguities"]

    def counts(self):
        return self.initx_engine.counts()

    def scan_counts(self):
        return dict(updates=self.updates, original_updates=self.original_updates,
            scans=self.scans, fast_scans=self.fast_scans, literal_scans=self.literal_scans)

    def bindings_valid(self):
        return (
            all(fn.__code__ is code for fn, code in self.codes.items())
            and all(self.globals.get(n) is v for n, v in self.bindings.items())
            and self.globals.get("range", self.original.__builtins__["range"]) is self.range
            and np.any is self.np_any
            and self.predicate.__globals__["np"] is self.np
            and self.ib.__globals__["uGNSS"] is self.bindings["uGNSS"]
            and int(self.bindings["uGNSS"].MAXSAT) == self.maxsat
            and self.initx_engine.original.__globals__.get("range",
                self.initx_engine.original.__builtins__["range"]) is range
        )

    def update(self, rtk, obs):
        self.updates += 1
        if not self.bindings_valid():
            self.original_updates += 1
            return self.original(rtk, obs)
        return self.transformed(rtk, obs)

    def eligible(self, rtk, obs, f):
        nav = rtk.nav
        if not (
            type(rtk).__getattribute__ is object.__getattribute__
            and type(nav) is Nav and type(nav.nf) is int and type(nav.na) is int
            and type(nav.nx) is int and type(nav.maxout) is int
            and -(2**31) <= nav.maxout < 2**31 and 0 <= f < nav.nf
            and type(obs.sig) is dict and all(type(k) in self.key_types for k in obs.sig)
            and _method(rtk.initx) is self.initx
            and _method(rtk.IB) is self.ib
            and _method(rtk._row_has_nonzero) is self.predicate
        ):
            return False
        flags = (nav.outc, nav.edt, nav.slip)
        if not all(type(a) is np.ndarray and a.dtype == np.dtype("int64")
            and a.shape == (self.maxsat, nav.nf)
            and (a.flags.c_contiguous or a.flags.f_contiguous) for a in flags):
            return False
        if not (nav.outc.flags.writeable and nav.slip.flags.writeable):
            return False
        x, p = nav.x, nav.P
        if not (type(x) is type(p) is np.ndarray and x.dtype == p.dtype == np.dtype("float64")
            and x.ndim == 1 and p.ndim == 2 and x.flags.c_contiguous
            and (p.flags.c_contiguous or p.flags.f_contiguous) and x.flags.writeable and p.flags.writeable
            and 0 <= nav.na and nav.na + nav.nf * self.maxsat <= nav.nx <= min(x.size, *p.shape)):
            return False
        arrays = (x, p, *flags)
        if any(np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1:]):
            return False
        # Scalar signed-overflow warnings/strict-policy partial state stay
        # with the original loop. No integer narrowing or counter clipping.
        if np.any(nav.outc[:, f] == np.iinfo(np.int64).max):
            return False
        start = nav.na + f * self.maxsat
        return bool(np.all(np.isfinite(x[start:start + self.maxsat])))

    def scan(self, rtk, obs, f):
        self.scans += 1
        try:
            eligible = self.eligible(rtk, obs, f)
        except (AttributeError, TypeError, ValueError):
            eligible = False
        if not eligible:
            self.literal_scans += 1
            return self.literal(rtk, obs, f)
        self.fast_scans += 1
        nav = rtk.nav
        start = nav.na + f * self.maxsat
        # These arrays cannot alias. Literal initx only touches x/P; flag
        # stores are independent across satellites. Keep each reset call and
        # its flag stores in the original ascending satellite order.
        nav.outc[:, f] += 1
        reset = ((nav.outc[:, f] > nav.maxout)
            | np.any(nav.edt != 0, axis=1) | np.any(nav.slip != 0, axis=1))
        allowed = np.empty(self.maxsat, dtype=bool)
        for system, indices in self.system_groups:
            allowed[indices] = system in obs.sig
        selected = np.flatnonzero(reset & allowed & (nav.x[start:start + self.maxsat] != 0.0))
        for index in selected:
            i = int(index)
            rtk.initx(0.0, 0.0, start + i)
            nav.outc[i, f] = 0
            nav.slip[i, f] = 0


def make_type(original):
    initx_type, initx_engine = make_initx_type(original)
    engine = BookkeepingEngine(original, initx_type, initx_engine)

    class BatchedBookkeeping(initx_type):
        def update_ambiguities(self, obs):
            return engine.update(self, obs)

    return BatchedBookkeeping, engine
