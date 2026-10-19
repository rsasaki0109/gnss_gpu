"""Bitwise original stores, malformed-buffer fallbacks and whole update state."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from cssrlib.gnss import Nav, rSigRnx, uGNSS, uTYP
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_ambiguity_initx import (  # noqa: E402
    InitxEngine,
    capture_input,
    capture_output,
    exact_output,
    make_type,
    restore,
)


def instance(cls, nx=17, order="C"):
    rtk = object.__new__(cls)
    rtk.nav = object.__new__(Nav)
    rtk.nav.nx = nx
    rtk.nav.x = np.arange(nx + 2, dtype=np.float64)
    rtk.nav.P = np.array(
        np.arange((nx + 2) ** 2).reshape(nx + 2, nx + 2), dtype=np.float64, order=order
    )
    return rtk


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("i", [0, 8, 16])
@pytest.mark.parametrize(
    "values", [(3.25, 100.0), (-0.0, -0.0), (np.float64(np.nan), np.float64(np.nan))]
)
def test_all_array_bits_and_untouched_border(order, i, values):
    cls, engine = make_type(rtkpos)
    a, b = instance(rtkpos, order=order), instance(cls, order=order)
    a.initx(*values, i)
    b.initx(*values, i)
    for name in ("x", "P"):
        assert getattr(a.nav, name).tobytes() == getattr(b.nav, name).tobytes()
    assert engine.counts() == dict(calls=1, fast_calls=1, fallback_calls=0)


@pytest.mark.parametrize(
    "kind", ["strided", "float32", "readonly", "bounds", "strict", "zero", "subclass"]
)
def test_fallback_matches_errors_and_partial_state(kind):
    cls, engine = make_type(rtkpos)
    a, b = instance(rtkpos), instance(cls)
    index = 8
    for rtk in (a, b):
        if kind == "strided":
            rtk.nav.P = rtk.nav.P[::2, ::2]
            rtk.nav.nx = 8
        elif kind == "float32":
            rtk.nav.P = rtk.nav.P.astype(np.float32)
        elif kind == "readonly":
            rtk.nav.P.flags.writeable = False
        elif kind == "bounds":
            index = 100
        elif kind == "zero":
            rtk.nav.nx = 0
        elif kind == "subclass":

            class Array(np.ndarray):
                pass

            rtk.nav.P = rtk.nav.P.view(Array)
    with np.errstate(under="raise" if kind == "strict" else "ignore"):
        errors = []
        for rtk in (a, b):
            try:
                rtk.initx(3.25, 100.0, index)
            except (IndexError, ValueError) as exc:
                errors.append(type(exc))
            else:
                errors.append(None)
    assert errors[0] is errors[1]
    for name in ("x", "P"):
        assert getattr(a.nav, name).tobytes() == getattr(b.nav, name).tobytes()
    assert engine.fallback_calls == 1 and engine.fast_calls == 0


@pytest.mark.parametrize("nf", [1, 2])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
def test_whole_update_four_epochs_state_and_observations(nf, seed):
    from types import SimpleNamespace

    rng = np.random.default_rng(seed)
    cls, engine = make_type(rtkpos)
    maxsat = int(uGNSS.MAXSAT)
    a = instance(rtkpos, nx=3 + nf * maxsat)
    nav = a.nav
    nav.nf, nav.na, nav.maxout, nav.sig_n0, nav.glo_ch = nf, 3, 3, 10.0, {}
    nav.outc = rng.integers(0, 5, (maxsat, nf), dtype=np.int64)
    nav.edt = rng.integers(0, 2, (maxsat, nf), dtype=np.int64)
    nav.slip = rng.integers(0, 2, (maxsat, nf), dtype=np.int64)
    nav.x[:] = 0.0
    nav.x[::3] = 100.0
    obs = SimpleNamespace(
        sat=np.arange(1, 13, dtype=np.int64),
        L=rng.normal(1e8, 1e3, (12, nf)),
        P=rng.normal(2e7, 1e3, (12, nf)),
        sig={uGNSS.GPS: {uTYP.L: [rSigRnx("GL1C"), rSigRnx("GL2W")][:nf]}},
    )
    for epoch in range(4):
        incoming = capture_input(a, obs)
        b, copied = restore(cls, incoming)
        a.update_ambiguities(obs)
        b.update_ambiguities(copied)
        exact_output(b, capture_output(a))
        assert (
            obs.L.tobytes() == copied.L.tobytes()
            and obs.P.tobytes() == copied.P.tobytes()
        )
    assert engine.fast_calls > 0 and engine.fallback_calls == 0
