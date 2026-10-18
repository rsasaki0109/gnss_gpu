"""WP61: can PLATEAU-predicted LOS / NLOS states and reflection delays locate
the antenna better than the real-time output? (negative; see D-060)

Subcommands:
  sd       cache 1 Hz single-difference observations (rover - base, GLONASS
           excluded, first band) of one run
  truth    per-satellite rows at the ground truth: map LOS, shortest
           single-bounce reflection (plateau_reflect.py), SD code residual,
           C/N0
  summary  LOS / NLOS / reflection statistics of truth rows (DD to the
           highest LOS satellite of each system)
  calib    -log densities from the truth rows of other runs: residual by
           class, residual minus predicted reflection, C/N0 by class and
           elevation
  grid     score a horizontal grid of candidates around the real-time output
           (or the truth) and report the horizontal error of the best one
  carrier  continuity of LOS DD carrier pairs at the truth
  selloss  split the ensemble selector's loss by the distance between its
           output and the nearest candidate within 0.5 m

Example (Nagoya 2, its 375 s canyon 555996-556371 s):
  python wp61_3dma.py sd --run nagoya 2
  python wp61_3dma.py truth --run nagoya 2 --tri tri_nagoya2.npz
  python wp61_3dma.py calib --rows tokyo1 tokyo2 tokyo3 --name tokyo
  python wp61_3dma.py grid --run nagoya 2 --tri tri_nagoya2.npz \\
      --window 555996 556371 --calib tokyo --ens-out ens_out --cn0 --refl
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE / "tc" / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

OMGE_C = 7.2921151467e-5 / 299792458.0


def _sagnac_range(rs, x):
    """Geometric range with the Sagnac term; rs (..., 3), x broadcastable."""
    return (np.linalg.norm(rs - x, axis=-1)
            + OMGE_C * (rs[..., 0] * x[..., 1] - rs[..., 1] * x[..., 0]))


def _truth(data_root, city, run):
    a = np.genfromtxt(Path(data_root) / city / f"run{run}" / "reference.csv",
                      delimiter=",", skip_header=1)
    return {round(float(t), 1): x for t, x in zip(a[:, 0], a[:, 5:8])}


def cmd_sd(a):
    import cssrlib.gnss as gn
    import cssrlib.rinex as rn
    from cssrlib.ephemeris import satposs
    from cssrlib.gnss import sat2prn, time2gpst, uGNSS, uTYP
    from gnss_fgo.utils.sig_autodetect import auto_detect_signals
    city, run = a.run[0], int(a.run[1])
    D = Path(a.data_root) / city / f"run{run}"
    dec, decb = rn.rnxdec(), rn.rnxdec()
    dec.decode_obsh(str(D / "rover.obs")); decb.decode_obsh(str(D / "base.obs"))
    sigs, sigsb = auto_detect_signals(dec.sig_map, decb.sig_map, max_freq=1,
                                      required=(uTYP.C, uTYP.L, uTYP.S))
    dec.setSignals(sigs); decb.setSignals(sigsb)
    nav = gn.Nav(nf=1); dec.decode_nav(str(D / "base.nav"), nav)
    rb = np.array(list(decb.pos))
    rec = []
    for obs, obsb, dt in rn.sync_obs_hold(dec, decb, maxage=0.5):
        if obsb is None or abs(dt) > 1e-3:        # base epochs only (1 Hz)
            continue
        tow = round(time2gpst(obs.t)[1], 1)
        rs, _, _, svh, _ = satposs(obs, nav)
        bidx = {s: i for i, s in enumerate(obsb.sat)}
        for i, s in enumerate(obs.sat):
            sys_, _ = sat2prn(s)
            if (sys_ == uGNSS.GLO or s not in bidx or svh[i] != 0
                    or np.linalg.norm(rs[i]) < 1e6):
                continue
            j = bidx[s]
            pr, pb = obs.P[i, 0], obsb.P[j, 0]
            if not (pr > 1e6 and pb > 1e6):
                continue
            gb = _sagnac_range(rs[i], rb)
            lam = obs.sig[sys_][uTYP.L][0].wavelength()
            lr, lb = obs.L[i, 0], obsb.L[j, 0]
            cp = (lr - lb) * lam + gb if (lr != 0 and lb != 0) else np.nan
            # SD code / carrier plus the base range: residual(x) = value - range(x)
            rec.append((tow, s, int(sys_), *rs[i], (pr - pb) + gb, cp, lam, obs.S[i, 0]))
    out = Path(a.work) / f"sd_{city}{run}.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(rec, dtype=float))
    print(out, "rows", len(rec))


def _tracers(tri):
    from plateau_los import GridTracer
    from plateau_reflect import ReflectionModel
    tr = GridTracer(np.load(tri)["triangles"])
    return tr, ReflectionModel(tr)


def cmd_truth(a):
    from plateau_los import _enu_rot
    city, run = a.run[0], int(a.run[1])
    sd = np.load(Path(a.work) / f"sd_{city}{run}.npy")
    truth = _truth(a.data_root, city, run)
    tr, rm = _tracers(a.tri)
    R = None
    rows = []
    for t in np.unique(sd[:, 0]):
        if t not in truth:
            continue
        m = sd[:, 0] == t
        x = truth[t]; rs = sd[m, 3:6]
        if R is None:
            R = _enu_rot(x)
        u = rs - x; u /= np.linalg.norm(u, axis=1, keepdims=True)
        los = tr.check_los(x, x + 2e7 * u)
        ext, _, _ = rm.paths(x[None], u[None], radius=60.0)
        enu = u @ R.T
        el = np.degrees(np.arcsin(enu[:, 2]))
        az = np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360
        res = sd[m, 6] - _sagnac_range(rs, x)
        for j in range(m.sum()):
            rows.append((t, sd[m][j, 2], sd[m][j, 1], el[j], az[j], sd[m][j, 9], los[j],
                         ext[0, j], res[j]))
    out = Path(a.work) / f"truth_{city}{run}.npy"
    np.save(out, np.array(rows, dtype=float))
    print(out, "rows", len(rows))


def _truth_errors(rows):
    """Residual minus per-system bias minus a per-epoch clock, at truth."""
    tow, sy, _, el, _, _, los, _, sd = rows.T
    los = los > 0.5
    clk = {}
    for t in np.unique(tow):
        m = (tow == t) & los & (sy == 0) & (el > 30)
        if m.sum() >= 2:
            clk[t] = np.median(sd[m])
    c = np.array([clk.get(t, np.nan) for t in tow])
    isb = {s: np.nanmedian((sd - c)[(sy == s) & los & (el > 30)]) for s in np.unique(sy)}
    r = sd - np.array([isb[s] for s in sy])
    e = np.full(len(rows), np.nan)
    for t in np.unique(tow):
        m = tow == t
        L = m & los & (el > 15)
        if L.sum() >= 3:
            e[m] = r[m] - np.median(r[L])
    return e, isb


def cmd_summary(a):
    rows = np.load(Path(a.work) / f"truth_{a.name}.npy")
    tow, sy, _, el, _, _, los, ext, sd = rows.T
    los = los > 0.5
    dd = np.full(len(rows), np.nan); pred = np.full(len(rows), np.nan)
    isref = np.zeros(len(rows), bool)
    for t in np.unique(tow):
        for s in np.unique(sy[tow == t]):
            m = np.nonzero((tow == t) & (sy == s))[0]
            L = m[los[m] & (el[m] > 20)]
            if len(L) == 0 or len(m) < 2:
                continue
            r = L[np.argmax(el[L])]
            isref[r] = True
            dd[m] = sd[m] - sd[r]
            pred[m] = np.where(los[m], 0.0, ext[m])
    ok = np.isfinite(dd) & ~isref
    sel = (tow >= a.window[0]) & (tow <= a.window[1]) if a.window else np.ones(len(rows), bool)
    s = ok & sel
    print(f"{a.name}: {int(s.sum())} DD rows over {len(np.unique(tow[sel]))} epochs, "
          f"NLOS share {np.mean(~los[sel]):.0%}")
    for cat, m in (("LOS", s & los), ("NLOS, reflection found", s & ~los & np.isfinite(ext)),
                   ("NLOS, no reflection", s & ~los & ~np.isfinite(ext))):
        if m.sum() == 0:
            continue
        e = dd[m]
        line = (f"  {cat:24s} n={int(m.sum()):6d}  median res {np.median(e):6.2f} m  "
                f"median |res| {np.median(np.abs(e)):5.2f}")
        if "found" in cat:
            line += (f" | median pred {np.median(pred[m]):5.2f}  |res-pred| "
                     f"{np.median(np.abs(e - pred[m])):5.2f}  corr {np.corrcoef(e, pred[m])[0, 1]:.2f}")
        print(line)


def cmd_calib(a):
    ed = np.arange(-20.0, 80.5, 0.5); er = np.arange(-40.0, 60.5, 0.5)
    ec = np.arange(10.0, 61.0, 1.0); eb = np.arange(0, 91, 10)
    h = {"los": np.zeros(len(ed) - 1), "nlos": np.zeros(len(ed) - 1)}
    hr = np.zeros(len(er) - 1); hc = np.zeros((2, len(eb) - 1, len(ec) - 1))
    for name in a.rows:
        rows = np.load(Path(a.work) / f"truth_{name}.npy")
        e, isb = _truth_errors(rows)
        _, _, _, el, _, cn0, los, ext, _ = rows.T
        los = los > 0.5; ok = np.isfinite(e)
        print(name, "system biases", {int(k): round(float(v), 2) for k, v in isb.items()})
        h["los"] += np.histogram(e[ok & los], ed)[0]
        h["nlos"] += np.histogram(e[ok & ~los], ed)[0]
        k = ok & ~los & np.isfinite(ext)
        hr += np.histogram(e[k] - ext[k], er)[0]
        for i, cls in enumerate((los, ~los)):
            hc[i] += np.histogram2d(el[cls], cn0[cls], [eb, ec])[0]

    def nll(x, w):
        return -np.log((x + 0.5) / (x.sum() + 0.5 * len(x)) / w)
    pc = (hc + 0.5) / (hc.sum(axis=2, keepdims=True) + 0.5 * hc.shape[2])
    out = Path(a.work) / f"calib_{a.name}.npz"
    np.savez(out, ed=ed, los=nll(h["los"], 0.5), nlos=nll(h["nlos"], 0.5), er=er,
             refl=nll(hr, 0.5), ec=ec, eb=eb, cn0_los=-np.log(pc[0]), cn0_nlos=-np.log(pc[1]))
    for k, x, edges in (("LOS", h["los"], ed), ("NLOS", h["nlos"], ed), ("NLOS minus reflection", hr, er)):
        q = np.cumsum(x) / x.sum()
        print(f"  {k:22s} rows {int(x.sum()):7d}  10/50/90%:",
              [round(float(edges[np.searchsorted(q, v)]), 1) for v in (0.1, 0.5, 0.9)])
    for b in range(len(eb) - 1):
        med = [ec[np.searchsorted(np.cumsum(hc[i, b]) / hc[i, b].sum(), 0.5)] for i in (0, 1)]
        print(f"  C/N0 median, elevation {eb[b]:2d}-{eb[b + 1]:2d}: LOS {med[0]:.0f}, NLOS {med[1]:.0f} dB-Hz")
    print(out)


class _Score:
    def __init__(self, path):
        self.c = np.load(path)

    def _lut(self, edges, table, x):
        i = np.clip(np.floor((np.nan_to_num(x) - edges[0]) / (edges[1] - edges[0])).astype(int),
                    0, len(edges) - 2)
        return table[i]

    def res(self, los, e):
        return np.where(los, self._lut(self.c["ed"], self.c["los"], e),
                        self._lut(self.c["ed"], self.c["nlos"], e))

    def refl(self, x):
        return self._lut(self.c["er"], self.c["refl"], x)

    def cn0(self, los, el, cn):
        b = np.clip((el // 10).astype(int), 0, 8)
        j = np.clip((cn - 10).astype(int), 0, len(self.c["ec"]) - 2)
        return np.where(los, self.c["cn0_los"][b, j], self.c["cn0_nlos"][b, j])


def cmd_grid(a):
    from plateau_los import _enu_rot
    city, run = a.run[0], int(a.run[1])
    sd = np.load(Path(a.work) / f"sd_{city}{run}.npy")
    truth = _truth(a.data_root, city, run)
    o = np.load(Path(a.ens_out) / f"{'TK' if city == 'tokyo' else 'NG'}run{run}.npz")
    out = {round(float(t), 1): x for t, x in zip(o["tow"], o["pos"])}
    sc = _Score(Path(a.work) / f"calib_{a.calib}.npz")
    tr, rm = _tracers(a.tri)
    c0, c1 = a.window
    tow_all = sd[:, 0]

    def residuals(X, rs, v):
        return v[None] - _sagnac_range(rs[None], X[:, None])

    # receiver inter-system biases from epochs before the window, at the output (causal)
    acc = {}
    for t in np.unique(tow_all[tow_all < c0 - 5]):
        if t not in out:
            continue
        m = tow_all == t; rs = sd[m, 3:6]; sy = sd[m, 2]; x = out[t]
        r = residuals(x[None], rs, sd[m, 6])[0]
        u = rs - x; los = tr.check_los(x, x + 2e7 * u / np.linalg.norm(u, axis=1, keepdims=True))
        g = los & (sy == 0)
        if g.sum() < 3:
            continue
        for s in np.unique(sy):
            acc.setdefault(s, []).extend((r[los & (sy == s)] - np.median(r[g])).tolist())
    isb = {s: float(np.median(v)) for s, v in acc.items() if len(v) > 20}
    R = _enu_rot(next(iter(truth.values())))
    g1 = np.arange(-a.half, a.half + 1e-9, a.step)
    OFF = np.array([[e, n, 0.0] for n in g1 for e in g1]) @ R
    cgrid = np.linspace(-3, 3, 31)
    hist, res = [], []
    for t in np.unique(tow_all[(tow_all >= c0) & (tow_all <= c1)]):
        if t not in truth or t not in out:
            continue
        m = tow_all == t
        sy = sd[m, 2]; keep = np.array([s in isb for s in sy])
        if keep.sum() < 4:
            continue
        rs = sd[m, 3:6][keep]; cn = sd[m, 9][keep]
        v = sd[m, 6][keep] - np.array([isb[s] for s in sy[keep]])
        X = (truth[t] if a.center == "truth" else out[t])[None] + OFF
        U = rs[None] - X[:, None]; U /= np.linalg.norm(U, axis=2, keepdims=True)
        los = tr.check_los_batch(X, X[:, None] + 2e7 * U)
        r = residuals(X, rs, v)
        el = np.degrees(np.arcsin(np.clip(U @ R[2], -1, 1)))
        extra = rm.paths(X, U, radius=50.0)[0] if a.refl else None
        base = sc.cn0(los, el, cn[None]).sum(axis=1) if a.cn0 else 0.0
        nl = los.sum(axis=1)
        with warnings.catch_warnings():       # candidates with no LOS satellite
            warnings.simplefilter("ignore", RuntimeWarning)
            cc = np.nanmedian(np.where(los, r, np.nan), axis=1)
        cost = np.full(len(X), np.inf)
        for dc in cgrid:                      # receiver clock: best of a +-3 m search
            e = r - (cc + dc)[:, None]
            c = sc.res(los, e)
            if a.refl:
                c = np.where(~los & np.isfinite(extra), sc.refl(e - extra), c)
            cost = np.minimum(cost, c.sum(axis=1) + base)
        cost[nl < 2] = np.inf
        hist.append(cost); hist = hist[-a.acc:]
        cost = np.sum(hist, axis=0)
        k = int(np.argmin(cost))
        res.append((t, *((out[t] - truth[t]) @ R.T), *((X[k] - truth[t]) @ R.T)))
    z = np.array(res)
    tag = f"center={a.center} step={a.step} cn0={int(a.cn0)} refl={int(a.refl)} acc={a.acc}"
    print(f"{city}{run} [{c0:.0f}, {c1:.0f}], {len(z)} epochs, {tag}")
    for name, j in (("real-time output", 1), ("best candidate", 4)):
        hh = np.hypot(z[:, j], z[:, j + 1])
        print(f"  {name:17s} horizontal error median {np.median(hh):5.2f} m, <1 m {np.mean(hh < 1):5.1%}, "
              f"<2 m {np.mean(hh < 2):5.1%};  median E / N {np.median(z[:, j]):5.2f} / {np.median(z[:, j + 1]):5.2f} m")


def cmd_carrier(a):
    from plateau_los import GridTracer, _enu_rot
    city, run = a.run[0], int(a.run[1])
    sd = np.load(Path(a.work) / f"sd_{city}{run}.npy")
    truth = _truth(a.data_root, city, run)
    tr = GridTracer(np.load(a.tri)["triangles"])
    R = None; prev = {}; runlen = {}; stat = []
    for t in np.unique(sd[:, 0]):
        if t < a.window[0] or t > a.window[1] or t not in truth:
            continue
        m = sd[:, 0] == t; x = truth[t]
        sat, sy, rs, cp = sd[m, 1], sd[m, 2], sd[m, 3:6], sd[m, 7]
        u = rs - x; u /= np.linalg.norm(u, axis=1, keepdims=True)
        if R is None:
            R = _enu_rot(x)
        el = np.degrees(np.arcsin(u @ R[2]))
        los = tr.check_los(x, x + 2e7 * u)
        res = cp - _sagnac_range(rs, x)
        grp = np.where(sy == 3, 3, 0) if a.cross_system else sy   # GPS/GAL/QZS share L1/E1
        cur = {}; npair = nlong = 0
        for s in np.unique(grp):
            k = np.nonzero((grp == s) & los & np.isfinite(res) & (el > 15))[0]
            if len(k) < 2:
                continue
            rf = k[np.argmax(el[k])]
            for j in k:
                if j == rf:
                    continue
                key = (int(sat[rf]), int(sat[j])); v = res[j] - res[rf]
                cur[key] = v; npair += 1
                if key in prev and abs(v - prev[key]) < 0.03:
                    runlen[key] = runlen.get(key, 0) + 1
                    nlong += runlen[key] >= 10
                else:
                    runlen[key] = 0
        prev = cur
        stat.append((npair, nlong, int(los.sum()), len(los)))
    z = np.array(stat)
    print(f"{city}{run} [{a.window[0]:.0f}, {a.window[1]:.0f}] {len(z)} epochs, "
          f"{'cross-system' if a.cross_system else 'per-system'} reference")
    print(f"  map-LOS satellites (median) {np.median(z[:, 2]):.0f} of {np.median(z[:, 3]):.0f}; "
          f"LOS DD carrier pairs {np.median(z[:, 0]):.0f}, continuous for 10 s {np.median(z[:, 1]):.0f}")
    print("  epochs with >= 3 / 4 / 5 / 6 pairs continuous for 10 s: "
          + " / ".join(f"{np.mean(z[:, 1] >= n):.0%}" for n in (3, 4, 5, 6)))


def cmd_selloss(a):
    from rt_ensemble import CANDS, RUNS, fmt, reference
    bins = [0.5, 1.0, 1.5, 2.0, 3.0, 1e9]
    tot = np.zeros(len(bins))
    for ct, r in RUNS:
        rt, x, w = reference(a.data_root, ct, r)
        rk = {round(float(t), 1): i for i, t in enumerate(rt)}
        o = np.load(Path(a.ens_out) / f"{ct.upper()}run{r}.npz")
        ok_ = {round(float(t), 1): i for i, t in enumerate(o["tow"])}
        d = [np.load(Path(a.results) / fmt(m, ct, r) / f"run{r}.npz") for m in a.members]
        C = np.stack([np.asarray(z[c], float) for z in d for c in CANDS], axis=1)
        loss = np.zeros(len(bins))
        for j, t in enumerate(d[0]["tow"]):
            k = rk.get(round(float(t), 1)); i = ok_.get(round(float(t), 1))
            if k is None or i is None or np.linalg.norm(o["pos"][i] - x[k]) <= 0.5:
                continue
            e = np.linalg.norm(C[j] - x[k], axis=1)
            if not np.any(e <= 0.5):
                continue
            loss[np.searchsorted(bins, np.linalg.norm(o["pos"][i] - C[j][np.nanargmin(e)]))] += w[k]
        loss = 100 * loss / w.sum(); tot += loss
        print(f"{ct}{r}: selector loss {loss.sum():4.1f} pp; " + " ".join(f"{v:4.2f}" for v in loss))
    print("mean: " + " ".join(f"{v:4.2f}" for v in tot / len(RUNS))
          + "  (output to good candidate: <0.5, 0.5-1, 1-1.5, 1.5-2, 2-3, >3 m)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", default="datasets/PPC-Dataset-data")
    ap.add_argument("--work", default=str(HERE / "results" / "wp61"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sd"); p.add_argument("--run", nargs=2, required=True)
    p = sub.add_parser("truth"); p.add_argument("--run", nargs=2, required=True)
    p.add_argument("--tri", required=True)
    p = sub.add_parser("summary"); p.add_argument("--name", required=True)
    p.add_argument("--window", nargs=2, type=float)
    p = sub.add_parser("calib"); p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--name", required=True)
    p = sub.add_parser("grid"); p.add_argument("--run", nargs=2, required=True)
    p.add_argument("--tri", required=True); p.add_argument("--window", nargs=2, type=float, required=True)
    p.add_argument("--calib", required=True); p.add_argument("--ens-out", required=True)
    p.add_argument("--center", choices=("out", "truth"), default="out")
    p.add_argument("--half", type=float, default=12.0); p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--cn0", action="store_true"); p.add_argument("--refl", action="store_true")
    p.add_argument("--acc", type=int, default=1, help="sum the last N epochs' score maps")
    p = sub.add_parser("carrier"); p.add_argument("--run", nargs=2, required=True)
    p.add_argument("--tri", required=True); p.add_argument("--window", nargs=2, type=float, required=True)
    p.add_argument("--cross-system", action="store_true")
    p = sub.add_parser("selloss"); p.add_argument("--ens-out", required=True)
    p.add_argument("--results", default=str(HERE / "results"))
    p.add_argument("--members", nargs="+", default=[
        "af56{CT}_s10_r{r}", "af56{CT}_s11_r{r}", "af56{CT}_s12_r{r}", "w58b_{ct}{r}",
        "on3_{ct}{r}", "on3s11_{ct}{r}", "on3s12_{ct}{r}"])
    a = ap.parse_args()
    {"sd": cmd_sd, "truth": cmd_truth, "summary": cmd_summary, "calib": cmd_calib, "grid": cmd_grid,
     "carrier": cmd_carrier, "selloss": cmd_selloss}[a.cmd](a)


if __name__ == "__main__":
    main()
