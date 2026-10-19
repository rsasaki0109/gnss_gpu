"""QC-to-ambiguity interaction, original callback order and complete arrays."""

from copy import deepcopy
from io import StringIO
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
from cssrlib.gnss import uGNSS
from cssrlib.rtk import rtkpos

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from rbpf_preprocess_bundle import make_type  # noqa: E402
from rbpf_qcedit_batch import capture_output, exact  # noqa: E402
from rbpf_qcedit_capture_check import exact_tree  # noqa: E402
from test_rbpf_qcedit_batch import fixture  # noqa: E402


def example(nf, seed, base_list):
    rtk, obs, args = fixture(nf, seed)
    nav = rtk.nav
    ns = int(uGNSS.MAXSAT)
    nav.na, nav.nx, nav.maxout, nav.sig_n0 = 6, 6 + ns * nf, 3, 10.0
    rng = np.random.default_rng(seed + 1)
    x = rng.normal(100, 20, nav.nx)
    x[::3] = 0.0
    x[:6] = nav.x
    nav.x = x
    nav.P = rng.normal(size=(nav.nx, nav.nx))
    nav.outc = rng.integers(0, 5, (ns, nf), dtype=np.int64)
    nav.cnr_min, nav.cnr_min_gpy = 30, 35
    for signals in obs.sig.values():
        for values in signals.values():
            while len(values) < nf:
                values.append(deepcopy(values[-1]))
            for signal in values:
                signal.str()
    if base_list:
        args["rr"] = nav.x[:3].tolist()
    return rtk, obs, args


def output(rtk, obs, result):
    values = capture_output(rtk, obs, result)
    values.update(x=rtk.nav.x.copy(), P=rtk.nav.P.copy(), outc=rtk.nav.outc.copy())
    return values


@pytest.mark.parametrize("nf", [1, 2, 3])
@pytest.mark.parametrize("seed", [20260710, 20260711, 20260712])
@pytest.mark.parametrize("base_list", [False, True])
def test_qc_then_ambiguity_eight_epochs_all_bits(nf, seed, base_list):
    a, oa, aa = example(nf, seed, base_list)
    cls, engine = make_type(rtkpos)
    b, ob, ab = deepcopy((a, oa, aa))
    b.__class__ = cls
    for epoch in range(8):
        for obs in (oa, ob):
            obs.L[:, 0] += float(epoch) * 0.03125
        ra, rb = a.qcedit(oa, **aa), b.qcedit(ob, **ab)
        exact(output(a, oa, ra), output(b, ob, rb))
        a.update_ambiguities(oa)
        b.update_ambiguities(ob)
        exact(output(a, oa, ra), output(b, ob, rb))
        exact_tree(vars(oa), vars(ob))
    assert engine.qc.counts() == dict(calls=8, fast_calls=8, fallback_calls=0)
    assert engine.bookkeeping.scan_counts() == dict(
        updates=8, original_updates=0, scans=8 * nf, fast_scans=8 * nf, literal_scans=0
    )
    assert engine.bookkeeping.initx_engine.fast_calls > 0
    assert engine.bookkeeping.initx_engine.fallback_calls == 0


@pytest.mark.parametrize(
    "kind",
    ["logging", "custom_initx", "overflow", "strided_flags", "strict", "changed_qc"],
)
def test_combined_fallbacks_errors_partial_state_and_order(kind, monkeypatch):
    from cssrlib import pppssr

    a, oa, aa = example(2, 20260710, True)
    cls, engine = make_type(rtkpos)
    b, ob, ab = deepcopy((a, oa, aa))
    b.__class__ = cls
    traces = [[], []]
    if kind == "changed_qc":
        old = pppssr._qc_signal_checks
        monkeypatch.setattr(pppssr, "_qc_signal_checks", lambda *args: old(*args))
    for index, rtk in enumerate((a, b)):
        if kind == "logging":
            rtk.nav.monlevel, rtk.nav.fout = 1, StringIO()
        elif kind == "custom_initx":
            parent, trace = type(rtk), traces[index]

            class Trace(parent):
                def initx(self, x0, v0, i, _trace=trace):
                    _trace.append((x0, v0, i, self.nav.outc.tobytes()))
                    return super().initx(x0, v0, i)

            rtk.__class__ = Trace
        elif kind == "overflow":
            rtk.nav.outc[0, 0] = np.iinfo(np.int64).max
        elif kind == "strided_flags":
            rtk.nav.outc = np.repeat(rtk.nav.outc, 2, axis=0)[::2]
    outputs, errors = [], []
    with np.errstate(under="raise" if kind == "strict" else "ignore"):
        for rtk, obs, args in ((a, oa, aa), (b, ob, ab)):
            try:
                result = rtk.qcedit(obs, **args)
                rtk.update_ambiguities(obs)
            except (FloatingPointError, ValueError, IndexError) as exc:
                errors.append(type(exc))
                result = np.empty(0, dtype=int)
            else:
                errors.append(None)
            outputs.append(output(rtk, obs, result))
    assert errors[0] == errors[1]
    exact(*outputs)
    if kind == "custom_initx":
        assert traces[0] and pickle.dumps(traces[0], protocol=5) == pickle.dumps(
            traces[1], protocol=5
        )
        assert engine.bookkeeping.fast_scans == 0
    if kind == "logging":
        assert a.nav.fout.getvalue() == b.nav.fout.getvalue()
    if kind in ("logging", "strict", "changed_qc"):
        assert engine.qc.fallback_calls == 1
