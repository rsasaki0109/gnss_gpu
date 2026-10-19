"""Explicit covariance row/column initialization with unchanged scalar values.

Only ordinary contiguous float64 arrays use bulk stores. The original
method handles unusual buffers, arguments, bindings and strict error policy.
No estimator equation, observation, particle, threshold or RNG changes.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cssrlib.gnss import Nav

from rbpf_native_canonical import _code_key

SOURCE_SHA256 = "701779669f357b6401a751fa8f1c126fa84d604ccca311d456e3a53b0079bcbc"
_RANGE = range
_NUMERIC = (int, float, np.integer, np.floating)
NAV_FIELDS = (
    "nf",
    "na",
    "nx",
    "maxout",
    "sig_n0",
    "glo_ch",
    "x",
    "P",
    "outc",
    "edt",
    "slip",
)
MUTATED = ("x", "P", "outc", "edt", "slip")


def _method(value):
    return getattr(value, "__func__", value)


class InitxEngine:
    def __init__(self, original):
        self.original = original.initx
        path = Path(self.original.__code__.co_filename)
        if ".claude" in [p.casefold() for p in path.parts]:
            raise ValueError("Excluded source path")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError("Frozen covariance initializer source changed")
        module = compile(source, str(path), "exec", dont_inherit=True)
        cls = next(
            c for c in module.co_consts if getattr(c, "co_name", None) == "pppos"
        )
        code = next(c for c in cls.co_consts if getattr(c, "co_name", None) == "initx")
        if _code_key(code) != _code_key(self.original.__code__):
            raise ValueError("Covariance initializer is not the literal source method")
        self.code = self.original.__code__
        self.calls = self.fast_calls = self.fallback_calls = 0

    def counts(self):
        return dict(
            calls=self.calls,
            fast_calls=self.fast_calls,
            fallback_calls=self.fallback_calls,
        )

    def __call__(self, rtk, x0, v0, i):
        self.calls += 1
        fn = self.original
        nav = rtk.nav
        valid = (
            fn.__code__ is self.code
            and fn.__globals__.get("range", fn.__builtins__.get("range")) is _RANGE
            and type(nav) is Nav
            and type(nav.nx) is int
            and isinstance(i, (int, np.integer))
            and not isinstance(i, (bool, np.bool_))
            and isinstance(x0, _NUMERIC)
            and isinstance(v0, _NUMERIC)
            and type(nav.x) is np.ndarray
            and type(nav.P) is np.ndarray
            and nav.x.dtype == nav.P.dtype == np.dtype("float64")
            and nav.x.ndim == 1
            and nav.P.ndim == 2
            and nav.x.flags.writeable
            and nav.P.flags.writeable
            and (nav.P.flags.c_contiguous or nav.P.flags.f_contiguous)
            and 0 <= i < nav.nx <= min(nav.x.size, *nav.P.shape)
            and all(v in ("ignore", "warn") for v in np.geterr().values())
            and np.geterr()["under"] == "ignore"
        )
        if not valid:
            self.fallback_calls += 1
            return fn(rtk, x0, v0, i)
        # No float arithmetic: the original stores +0 off-diagonal and v0
        # twice at the diagonal. Ordinary nonoverlapping array elements allow
        # the zero stores to be grouped, with the same final scalar bits.
        nav.x[i] = x0
        nav.P[: nav.nx, i] = 0
        nav.P[i, : nav.nx] = 0
        nav.P[i, i] = v0
        self.fast_calls += 1


def make_type(original):
    engine = InitxEngine(original)

    class BulkInitx(original):
        def initx(self, x0, v0, i):
            return engine(self, x0, v0, i)

    return BulkInitx, engine


def capture_input(rtk, obs):
    return dict(
        nav={n: deepcopy(getattr(rtk.nav, n)) for n in NAV_FIELDS},
        obs={n: deepcopy(getattr(obs, n)) for n in ("sat", "L", "P", "sig")},
    )


def capture_output(rtk):
    return {n: getattr(rtk.nav, n).copy() for n in MUTATED}


def restore(cls, value):
    rtk = object.__new__(cls)
    rtk.nav = object.__new__(Nav)
    vars(rtk.nav).update(deepcopy(value["nav"]))
    return rtk, SimpleNamespace(**deepcopy(value["obs"]))


def exact_output(rtk, expected):
    for name, value in expected.items():
        actual = getattr(rtk.nav, name)
        assert actual.dtype == value.dtype and actual.shape == value.shape
        assert actual.tobytes() == value.tobytes(), name
