"""Exact full updates, ordered reset arguments and partial-state fallbacks."""

from copy import deepcopy
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from cssrlib.gnss import Nav, rSigRnx, uGNSS, uTYP
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_ambiguity_bookkeeping import make_type  # noqa: E402
from rbpf_ambiguity_initx_checked import capture_input, capture_output, exact_output, restore  # noqa: E402


def example(nf=3, seed=20260710):
    rng = np.random.default_rng(seed)
    rtk = object.__new__(rtkpos)
    nav = rtk.nav = object.__new__(Nav)
    ns = int(uGNSS.MAXSAT)
    nav.nf, nav.na, nav.nx = nf, 3, 3 + ns * nf
    nav.maxout, nav.sig_n0, nav.glo_ch = 3, 10.0, {}
    nav.x = rng.normal(100, 20, nav.nx + 2)
    nav.x[::3] = 0.0
    nav.P = rng.normal(size=(nav.nx + 2, nav.nx + 2))
    nav.outc = rng.integers(0, 5, (ns, nf), dtype=np.int64)
    nav.edt = rng.integers(0, 2, (ns, nf), dtype=np.int64)
    nav.slip = rng.integers(0, 2, (ns, nf), dtype=np.int64)
    obs = SimpleNamespace(sat=np.arange(1, 13, dtype=np.int64),
        L=rng.normal(1e8, 1e3, (12, nf)), P=rng.normal(2e7, 1e3, (12, nf)),
        sig={uGNSS.GPS: {uTYP.L: [rSigRnx("GL1C") for _ in range(nf)]}})
    return rtk, obs


def traced_candidate(cls, engine, value):
    rtk, obs = restore(cls, value)
    trace = []
    base = type(engine.initx_engine)

    class Trace(base):
        def __call__(self, target, x0, v0, i):
            trace.append((x0, v0, i))
            return super().__call__(target, x0, v0, i)

    engine.initx_engine.__class__ = Trace
    return rtk, obs, trace


@pytest.mark.parametrize("nf", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
def test_full_update_four_epochs_and_all_initx_arguments(nf, seed):
    cls, engine = make_type(rtkpos)
    a, obs = example(nf, seed)
    reference_trace = []

    class Reference(rtkpos):
        def initx(self, x0, v0, i):
            reference_trace.append((x0, v0, i))
            return super().initx(x0, v0, i)

    a.__class__ = Reference
    for _ in range(4):
        incoming = capture_input(a, obs)
        reference_trace.clear()
        b, copied, trace = traced_candidate(cls, engine, incoming)
        a.update_ambiguities(obs)
        b.update_ambiguities(copied)
        exact_output(b, capture_output(a))
        assert pickle.dumps(trace, protocol=5) == pickle.dumps(reference_trace, protocol=5)
        for n in ("sat", "L", "P"):
            assert getattr(copied, n).tobytes() == incoming["obs"][n].tobytes()
    assert engine.scan_counts() == dict(updates=4, original_updates=0, scans=4 * nf,
        fast_scans=4 * nf, literal_scans=0)
    assert engine.proof["restored_update_AST_equal"]


@pytest.mark.parametrize("kind", ["overflow", "strict_overflow", "readonly_outc", "readonly_slip",
    "readonly_x", "readonly_P", "strided", "overlapping_elements", "alias", "float_flags",
    "nan_x", "infinity_x", "signedzero", "F_order", "empty_sig", "bad_index", "short_x"])
def test_fallback_errors_warnings_and_all_partial_state(kind):
    cls, engine = make_type(rtkpos)
    initial, obs = example()
    a, x = restore(rtkpos, capture_input(initial, obs))
    b, y = restore(cls, capture_input(initial, obs))
    for target in (a, b):
        nav = target.nav
        if kind in ("overflow", "strict_overflow"):
            nav.outc[4, 0] = np.iinfo(np.int64).max
        elif kind.startswith("readonly_"):
            getattr(nav, kind.removeprefix("readonly_")).flags.writeable = False
        elif kind == "strided":
            nav.outc = np.repeat(nav.outc, 2, axis=0)[::2]
        elif kind == "overlapping_elements":
            nav.outc = np.lib.stride_tricks.as_strided(nav.outc[:1], shape=nav.outc.shape,
                strides=(0, nav.outc.strides[1]), writeable=True)
        elif kind == "alias":
            nav.edt = nav.outc
        elif kind == "float_flags":
            nav.edt = nav.edt.astype(np.float64)
        elif kind == "nan_x":
            nav.x[3] = np.nan
        elif kind == "infinity_x":
            nav.x[3] = np.inf
        elif kind == "signedzero":
            nav.x[3::2] = -0.0
        elif kind == "F_order":
            nav.P = np.asfortranarray(nav.P)
            for n in ("outc", "edt", "slip"):
                setattr(nav, n, np.asfortranarray(getattr(nav, n)))
        elif kind == "bad_index":
            nav.na = -1
        elif kind == "short_x":
            nav.x = nav.x[:12].copy()
    if kind == "empty_sig":
        x.sig = y.sig = {}
    outcomes = []
    for target, incoming in ((a, x), (b, y)):
        with np.errstate(over="raise" if kind == "strict_overflow" else "warn"):
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                try:
                    target.update_ambiguities(incoming)
                except (IndexError, ValueError, FloatingPointError) as exc:
                    error = type(exc)
                else:
                    error = None
                outcomes.append((error, [(v.category, str(v.message)) for v in seen]))
    assert outcomes[0] == outcomes[1]
    exact_output(b, capture_output(a))
    if kind not in ("signedzero", "F_order", "empty_sig"):
        assert engine.scan_counts()["literal_scans"] > 0


def test_custom_callback_observes_original_counter_order():
    cls, engine = make_type(rtkpos)
    initial, obs = example()
    value = capture_input(initial, obs)
    traces = []
    for base in (rtkpos, cls):
        trace = []

        class Custom(base):
            def initx(self, x0, v0, i):
                trace.append(self.nav.outc.tobytes())
                return super().initx(x0, v0, i)

        target, copied = restore(Custom, value)
        target.update_ambiguities(copied)
        traces.append((trace, capture_output(target)))
    assert traces[0][0] == traces[1][0] and len(traces[0][0]) > 0
    for n in traces[0][1]:
        assert traces[0][1][n].tobytes() == traces[1][1][n].tobytes()
    assert engine.scan_counts()["fast_scans"] == 0


def test_changed_global_binding_uses_original_whole_update(monkeypatch):
    cls, engine = make_type(rtkpos)
    a, obs = example()
    b, copied = restore(cls, capture_input(a, obs))
    scope = rtkpos.update_ambiguities.__globals__
    original = scope["sat2prn"]
    monkeypatch.setitem(scope, "sat2prn", lambda s: original(s))
    a.update_ambiguities(obs)
    b.update_ambiguities(copied)
    exact_output(b, capture_output(a))
    assert engine.scan_counts()["original_updates"] == 1
