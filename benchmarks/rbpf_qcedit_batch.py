"""Explicit, frozen-fork experiment: one compiled observation QC loop.

Retain satellite order and the original compiled geometry/signal/GF helpers.
Logging, custom objects, unusual buffers or bindings use the original method.
No observation selection, threshold, precision or estimator changes.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np
from numba import njit
from cssrlib.gnss import Nav, Obs, SAT_SYS_ARR, rSigRnx, uGNSS, uTYP, uSIG
from cssrlib import geometry, pppssr

SOURCE_SHA256 = "701779669f357b6401a751fa8f1c126fa84d604ccca311d456e3a53b0079bcbc"
GEOMETRY_SHA256 = "542f0f1ca4e5e9f0d64ffc181d4691f4f61c8a43b3fe02b1376ae9dbb8cf2b7d"
_geodist = geometry.geodist
_satazel = geometry.satazel
_qc = pppssr._qc_signal_checks
_gf = pppssr._gf_slip_check
_CACHE = pppssr._qcedit_system_cache
_NP = {
    n: getattr(np, n)
    for n in ("ones", "asarray", "atleast_2d", "isnan", "any", "geterr")
}
_SIG = {n: getattr(rSigRnx, n) for n in ("wavelength", "frequency", "isGPS_PY")}
NAV_FIELDS = (
    "t",
    "x",
    "pmode",
    "nf",
    "excl_sat",
    "monlevel",
    "cnr_min_gpy",
    "cnr_min",
    "elmin",
    "thresslip",
    "glo_ch",
    "edt",
    "el",
    "slip",
    "gf",
)
OBS_FIELDS = ("t", "sat", "sig", "P", "L", "S", "lli")


@njit(cache=True, fastmath=False)
def _loop(
    sats,
    rs,
    dts,
    svh,
    rr,
    pos,
    P,
    L,
    S,
    lli,
    thresholds,
    excluded,
    enabled,
    wavelengths,
    elmin,
    thresslip,
    edt,
    el,
    slip,
    gf,
):
    accepted = np.empty(sats.size, dtype=np.int64)
    count = 0
    nf = edt.shape[1]
    for j in range(sats.size):
        sat = int(sats[j])
        i = sat - 1
        sys = SAT_SYS_ARR[sat]
        edt[i, :] = 0
        if excluded[sat]:
            edt[i, :] = 1
            continue
        if np.isnan(rs[j, :]).any() or np.isnan(dts[j]):
            edt[i, :] = 1
            continue
        if svh[j] > 0:
            edt[i, :] = 1
            continue
        _, e = _geodist(rs[j, :], rr)
        _, elevation = _satazel(pos, e)
        el[i] = elevation
        if elevation < elmin:
            edt[i, :] = 1
            continue
        codes = _qc(P[j, :nf], L[j, :nf], S[j, :nf], lli[j, :nf], thresholds[sys])
        for f in range(nf):
            code = int(codes[f])
            if code == 0:
                continue
            if code == 1:
                slip[i, f] = 1
            else:
                edt[i, f] = 1
        if enabled[sys]:
            lam1, lam2 = wavelengths[sat, 0], wavelengths[sat, 1]
            gf1, slipped = _gf(
                float(L[j, 0]),
                float(L[j, 1]),
                float(lam1),
                float(lam2),
                float(gf[sat]),
                float(thresslip),
            )
            if gf1 != 0.0:
                gf[sat] = gf1
            if slipped:
                slip[i, :2] = 1
        edited = False
        for f in range(nf):
            if edt[i, f] > 0:
                edited = True
                break
        if not edited:
            accepted[count] = sat
            count += 1
    return accepted[:count].copy()


def _array(a, dtype, shape):
    return (
        type(a) is np.ndarray
        and a.dtype == np.dtype(dtype)
        and a.shape == shape
        and a.flags.c_contiguous
        and a.flags.aligned
    )


class QcEngine:
    def __init__(self, original):
        self.original = original.qcedit
        for fn, expected in (
            (self.original, SOURCE_SHA256),
            (geometry.geodist.py_func, GEOMETRY_SHA256),
        ):
            path = Path(fn.__code__.co_filename)
            if ".claude" in [p.casefold() for p in path.parts]:
                raise ValueError("Excluded source path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen QC/geometry source changed")
        if (
            type(self.original) is not FunctionType
            or self.original is not pppssr.pppos.qcedit
        ):
            raise ValueError("The original frozen QC method is required")
        self.code = self.original.__code__
        self.bindings = {
            n: self.original.__globals__[n]
            for n in self.code.co_names
            if n in self.original.__globals__
        }
        self.helper_codes = [
            (f, f.py_func.__code__) for f in (_geodist, _satazel, _qc, _gf)
        ]
        self.calls = self.fast_calls = self.fallback_calls = 0

    def counts(self):
        return dict(
            calls=self.calls,
            fast_calls=self.fast_calls,
            fallback_calls=self.fallback_calls,
        )

    def eligible(self, rtk, obs, rs, dts, svh, rr):
        fn = self.original
        if (
            fn.__code__ is not self.code
            or any(fn.__globals__.get(k) is not v for k, v in self.bindings.items())
            or any(getattr(np, k) is not v for k, v in _NP.items())
            or any(getattr(rSigRnx, k) is not v for k, v in _SIG.items())
            or any(f.py_func.__code__ is not c for f, c in self.helper_codes)
            or _CACHE is not pppssr._qcedit_system_cache
            or type(rtk.nav) is not Nav
            or type(obs) is not Obs
        ):
            return False
        nav = rtk.nav
        if not all(n in vars(nav) for n in NAV_FIELDS) or not all(
            n in vars(obs) for n in OBS_FIELDS
        ):
            return False
        ns = int(uGNSS.MAXSAT)
        if (
            type(nav.nf) is not int
            or not 1 <= nav.nf <= 8
            or type(nav.monlevel) is not int
            or nav.monlevel != 0
            or type(nav.pmode) is not int
            or nav.pmode not in (0, 1)
            or type(nav.excl_sat) not in (list, tuple, set)
            or not all(type(s) is int and 1 <= s <= ns for s in nav.excl_sat)
            or type(nav.glo_ch) is not dict
            or type(obs.sig) is not dict
            or not all(
                type(v) in (float, np.float64)
                for v in (nav.elmin, nav.thresslip, nav.cnr_min, nav.cnr_min_gpy)
            )
            or type(obs.sat) is not np.ndarray
            or obs.sat.ndim != 1
        ):
            return False
        n, nf = obs.sat.size, nav.nf
        if not (
            (_array(obs.sat, "int64", (n,)) or _array(obs.sat, "int32", (n,)))
            and np.all((obs.sat >= 1) & (obs.sat < ns))
            and _array(rs, "float64", (n, 3))
            and _array(dts, "float64", (n,))
            and _array(svh, "int64", (n,))
            and type(nav.x) is np.ndarray
            and _array(nav.x, "float64", nav.x.shape)
            and nav.x.ndim == 1
            and nav.x.size >= 6
            and _array(nav.el, "float64", (ns,))
            and _array(nav.gf, "float64", (ns,))
            and _array(nav.slip, "int64", (ns, nf))
        ):
            return False
        arrays = [obs.P, obs.L, obs.S, obs.lli]
        if not all(
            type(a) is np.ndarray
            and a.ndim == 2
            and a.shape[0] == n
            and a.shape[1] >= nf
            and a.flags.c_contiguous
            and a.flags.aligned
            for a in arrays
        ):
            return False
        if not all(a.dtype == np.dtype("float64") for a in arrays[:3]):
            return False
        if obs.lli.dtype.kind not in "iuf" or obs.lli.dtype.itemsize > 8:
            return False
        if rr is not None and not _array(rr, "float64", (3,)):
            return False
        if not all(a.flags.writeable for a in (nav.el, nav.gf, nav.slip)):
            return False
        if any(
            np.may_share_memory(a, b)
            for a in (nav.el, nav.gf, nav.slip)
            for b in (rs, dts, svh, nav.x, *arrays)
        ):
            return False
        for sys, sigs in obs.sig.items():
            if (
                type(sys) is not uGNSS
                or not 0 <= int(sys) < int(uGNSS.GNSSMAX)
                or type(sigs) is not dict
            ):
                return False
            if not all(
                k in sigs
                and type(sigs[k]) is list
                and all(
                    type(s) is rSigRnx
                    and set(vars(s)) == {"sys", "typ", "sig"}
                    and s.sys == sys
                    and s.typ == k
                    and s.sig != uSIG.NONE
                    for s in sigs[k]
                )
                for k in (uTYP.C, uTYP.L, uTYP.S)
            ):
                return False
        if not all(SAT_SYS_ARR[int(s)] in obs.sig for s in obs.sat):
            return False
        if obs.L.shape[1] > 1:
            for sat in obs.sat:
                sys = SAT_SYS_ARR[int(sat)]
                if sys == uGNSS.GLO and len(obs.sig[sys][uTYP.L]) >= 2:
                    if (
                        int(sat) not in nav.glo_ch
                        or type(nav.glo_ch[int(sat)]) is not int
                    ):
                        return False
        return (
            all(v in ("ignore", "warn") for v in np.geterr().values())
            and np.geterr()["under"] == "ignore"
        )

    def __call__(self, rtk, obs, rs, dts, svh, rr=None):
        self.calls += 1
        if not self.eligible(rtk, obs, rs, dts, svh, rr):
            self.fallback_calls += 1
            return self.original(rtk, obs, rs, dts, svh, rr)
        nav = rtk.nav
        # Same prefix expressions and original metadata-cache builder.
        tt = pppssr.timediff(obs.t, nav.t)
        if rr is None:
            rr_ = nav.x[0:3].copy()
            if nav.pmode > 0:
                rr_ += nav.x[3:6] * tt
        else:
            rr_ = np.asarray(rr, dtype=np.float64)
        pos = pppssr.ecef2pos(rr_)
        ns = int(uGNSS.MAXSAT)
        nav.edt = np.ones((ns, nav.nf), dtype=int)
        system_cache = _CACHE(obs, nav)
        thresholds = np.zeros((int(uGNSS.GNSSMAX), nav.nf), dtype=np.float64)
        enabled = np.zeros(int(uGNSS.GNSSMAX), dtype=np.bool_)
        wavelengths = np.zeros((ns + 1, 2), dtype=np.float64)
        excluded = np.zeros(ns + 1, dtype=np.bool_)
        for sat in nav.excl_sat:
            excluded[sat] = True
        for sys, (_, _, _, cnr, pair) in system_cache.items():
            thresholds[int(sys), :] = cnr
            if obs.L.shape[1] > 1 and len(obs.sig[sys][uTYP.L]) >= 2:
                enabled[int(sys)] = True
                for sat in obs.sat:
                    if SAT_SYS_ARR[int(sat)] != sys:
                        continue
                    if sys == uGNSS.GLO:
                        a, b = obs.sig[sys][uTYP.L][:2]
                        ch = nav.glo_ch[int(sat)]
                        wavelengths[int(sat)] = a.wavelength(ch), b.wavelength(ch)
                    else:
                        wavelengths[int(sat)] = pair
        result = _loop(
            obs.sat,
            rs,
            dts,
            svh,
            rr_,
            pos,
            obs.P,
            obs.L,
            obs.S,
            np.asarray(obs.lli, dtype=np.float64),
            thresholds,
            excluded,
            enabled,
            wavelengths,
            nav.elmin,
            nav.thresslip,
            nav.edt,
            nav.el,
            nav.slip,
            nav.gf,
        )
        self.fast_calls += 1
        return result


def make_type(original):
    engine = QcEngine(original)

    class BatchedQc(original):
        def qcedit(self, obs, rs, dts, svh, rr=None):
            return engine(self, obs, rs, dts, svh, rr)

    return BatchedQc, engine


def capture_input(rtk, obs, rs, dts, svh, rr=None):
    return deepcopy(
        dict(
            nav={n: getattr(rtk.nav, n) for n in NAV_FIELDS},
            obs={n: getattr(obs, n) for n in OBS_FIELDS},
            rs=rs,
            dts=dts,
            svh=svh,
            rr=rr,
        )
    )


def restore(cls, data):
    data = deepcopy(data)
    rtk, nav, obs = object.__new__(cls), object.__new__(Nav), object.__new__(Obs)
    vars(nav).update(data.pop("nav"))
    vars(obs).update(data.pop("obs"))
    rtk.nav = nav
    return rtk, obs, data


def capture_output(rtk, obs, result):
    return dict(
        result=result.copy(),
        edt=rtk.nav.edt.copy(),
        el=rtk.nav.el.copy(),
        gf=rtk.nav.gf.copy(),
        slip=rtk.nav.slip.copy(),
        obs={n: getattr(obs, n).copy() for n in ("sat", "P", "L", "S", "lli")},
    )


def exact(a, b):
    assert a.keys() == b.keys()
    for name, x in a.items():
        y = b[name]
        if isinstance(x, dict):
            exact(x, y)
        else:
            assert (
                x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            ), name
