"""QC control flow, exact geometry/GF bits and original-method fallbacks."""

from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pytest
from cssrlib.gnss import Nav, Obs, gtime_t, prn2sat, rSigRnx, uGNSS, uTYP
from cssrlib.rtk import rtkpos
from cssrlib import pppssr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_qcedit_batch import (  # noqa: E402
    QcEngine,
    capture_input,
    capture_output,
    exact,
    make_type,
    restore,
)


def fixture(nf=2, seed=20260710, dtype="int32"):
    rng = np.random.default_rng(seed)
    rtk, nav, obs = object.__new__(rtkpos), object.__new__(Nav), object.__new__(Obs)
    rtk.nav = nav
    ns = int(uGNSS.MAXSAT)
    nav.t, obs.t = gtime_t(100, 0.25), gtime_t(101, 0.75)
    nav.nf, nav.pmode, nav.monlevel = nf, 1, 0
    nav.x = np.array([-3954500.0, 3354900.0, 3700000.0, 0.5, -0.2, 0.1])
    nav.el = rng.normal(0, 0.4, ns)
    nav.gf = rng.normal(0, 0.2, ns)
    nav.slip = rng.integers(0, 2, (ns, nf), dtype=np.int64)
    nav.edt = rng.integers(0, 2, (ns, nf), dtype=np.int64)
    nav.cnr_min, nav.cnr_min_gpy = 30.0, 35.0
    nav.elmin, nav.thresslip = -0.5, 0.05
    nav.excl_sat = [2]
    nav.glo_ch = {prn2sat(uGNSS.GLO, i): i - 4 for i in range(1, 5)}
    ids = [(uGNSS.GPS, i) for i in range(1, 10)] + [(uGNSS.GLO, i) for i in range(1, 5)]
    ids += [(uGNSS.GAL, i) for i in range(1, 5)]
    obs.sat = np.array([prn2sat(s, i) for s, i in ids], dtype=dtype)
    obs.sig = {}
    for system, letter, bands in (
        (uGNSS.GPS, "G", ("1C", "2W")),
        (uGNSS.GLO, "R", ("1C", "2C")),
        (uGNSS.GAL, "E", ("1C", "5Q")),
    ):
        obs.sig[system] = {
            typ: [rSigRnx(letter + prefix + band) for band in bands[:nf]]
            for typ, prefix in ((uTYP.C, "C"), (uTYP.L, "L"), (uTYP.S, "S"))
        }
    n = obs.sat.size
    obs.P = rng.normal(2e7, 1e4, (n, nf))
    obs.L = rng.normal(1e8, 1e4, (n, nf))
    obs.S = rng.uniform(20, 50, (n, nf))
    obs.lli = rng.integers(0, 3, (n, nf), dtype=np.int32)
    obs.P[5, 0], obs.L[6, 0] = 0.0, 0.0
    rs = rng.normal(0, 1, (n, 3))
    rs *= 2.6e7 / np.linalg.norm(rs, axis=1)[:, None]
    dts = rng.normal(0, 1e-5, n)
    svh = np.zeros(n, dtype=np.int64)
    svh[3] = 1
    rs[4, 1] = np.nan
    dts[7] = np.nan
    return rtk, obs, dict(rs=rs, dts=dts, svh=svh, rr=None)


def pair(rtk, obs, args, require_fast=True):
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(rtk, obs, **args))
    ra, rb = rtk.qcedit(obs, **args), b.qcedit(ob, **ab)
    exact(capture_output(rtk, obs, ra), capture_output(b, ob, rb))
    assert engine.fast_calls == int(require_fast)
    assert engine.fallback_calls == int(not require_fast)
    return b, ob, ab


@pytest.mark.parametrize("nf", [1, 2])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_eight_epochs_all_qc_state_and_observation_bits(nf, seed, dtype):
    a, oa, aa = fixture(nf, seed, dtype)
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, oa, **aa))
    for epoch in range(8):
        for o in (oa, ob):
            o.L[:, 0] += float(epoch) * 0.03125
        ra, rb = a.qcedit(oa, **aa), b.qcedit(ob, **ab)
        exact(capture_output(a, oa, ra), capture_output(b, ob, rb))
    assert engine.fast_calls == 8 and engine.fallback_calls == 0


@pytest.mark.parametrize(
    "kind",
    [
        "duplicates",
        "empty",
        "base",
        "signed_nan",
        "all_excluded",
        "gf_missing",
        "boundaries",
    ],
)
def test_order_corner_cases(kind):
    a, o, args = fixture()
    if kind == "duplicates":
        o.sat[8] = o.sat[0]
    elif kind == "empty":
        o.sat = o.sat[:0]
        for n in ("P", "L", "S", "lli"):
            setattr(o, n, getattr(o, n)[:0])
        args = {k: v[:0] if isinstance(v, np.ndarray) else v for k, v in args.items()}
    elif kind == "base":
        args["rr"] = a.nav.x[:3].copy()
    elif kind == "signed_nan":
        a.nav.gf.view("uint64")[0:4] = [
            0,
            0x8000000000000000,
            0x7FF8000000000042,
            0xFFF8000000000063,
        ]
        o.L.view("uint64")[8, 0] = 0x7FF8000000000021
    elif kind == "all_excluded":
        a.nav.excl_sat = [int(s) for s in o.sat]
    elif kind == "gf_missing":
        for sigs in o.sig.values():
            sigs[uTYP.L] = sigs[uTYP.L][:1]
    elif kind == "boundaries":
        o.S[:, :] = a.nav.cnr_min
        o.S[0, 0] = np.nextafter(a.nav.cnr_min, -np.inf)
        o.P[1, 0] = -0.0
    pair(a, o, args)


@pytest.mark.parametrize(
    "kind",
    [
        "strided",
        "float32",
        "readonly",
        "logging",
        "base_list",
        "strict",
        "bad_sat",
        "custom_array",
        "custom_signal",
        "aliased",
    ],
)
def test_original_fallback_outputs_errors_and_partial_state(kind):
    a, o, args = fixture()
    if kind == "strided":
        o.P = np.asfortranarray(o.P)
    elif kind == "float32":
        o.P = o.P.astype(np.float32)
    elif kind == "readonly":
        a.nav.el.flags.writeable = False
    elif kind == "logging":
        a.nav.monlevel, a.nav.fout = 1, StringIO()
    elif kind == "base_list":
        args["rr"] = a.nav.x[:3].tolist()
    elif kind == "bad_sat":
        o.sat[0] = int(uGNSS.MAXSAT) + 10
    elif kind == "custom_array":

        class Array(np.ndarray):
            pass

        o.P = o.P.view(Array)
    elif kind == "custom_signal":

        class Signal(rSigRnx):
            pass

        o.sig[uGNSS.GPS][uTYP.S][0] = Signal("GS1C")
    elif kind == "aliased":
        a.nav.x = a.nav.el[:6]
        a.nav.x[:] = [-3954500.0, 3354900.0, 3700000.0, 0.5, -0.2, 0.1]
    cls, engine = make_type(rtkpos)
    b, ob, ab = restore(cls, capture_input(a, o, **args))
    # deepcopy intentionally repairs readonly/alias layouts; reproduce them.
    if kind == "readonly":
        b.nav.el.flags.writeable = False
    if kind == "logging":
        b.nav.fout = StringIO()
    if kind == "aliased":
        b.nav.x = b.nav.el[:6]
    errors, output = [], []
    with np.errstate(under="raise" if kind == "strict" else "ignore"):
        for rtk, obs, ar in ((a, o, args), (b, ob, ab)):
            try:
                result = rtk.qcedit(obs, **ar)
            except (ValueError, IndexError, TypeError, ZeroDivisionError) as exc:
                errors.append((type(exc), str(exc)))
                result = np.empty(0, dtype=int)
            else:
                errors.append(None)
            output.append(capture_output(rtk, obs, result))
    assert errors[0] == errors[1]
    exact(*output)
    if kind == "logging":
        assert a.nav.fout.getvalue() == b.nav.fout.getvalue()
    assert engine.fast_calls == 0 and engine.fallback_calls == 1


def test_mutated_dependency_uses_original(monkeypatch):
    a, o, args = fixture()
    engine = QcEngine(rtkpos)
    original, calls = pppssr._qc_signal_checks, []

    def custom(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(pppssr, "_qc_signal_checks", custom)
    engine(a, o, **args)
    assert calls and engine.fast_calls == 0 and engine.fallback_calls == 1


def test_saved_inputs_are_owned():
    a, o, args = fixture()
    data = capture_input(a, o, **args)
    before = deepcopy(data)
    cls, _ = make_type(rtkpos)
    b, ob, ab = restore(cls, data)
    b.qcedit(ob, **ab)
    for n in ("x", "el", "gf", "slip", "edt"):
        assert data["nav"][n].tobytes() == before["nav"][n].tobytes()
