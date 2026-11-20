"""WP62 diagnostic: would per-basin factor graphs separate contested basins?

Plan 5b (internal_docs/plan.md) proposes giving each top basin its own FGO
with its own holds and history, branched from the hold-free witness graph
(`RBPF_WITNESS=1`). Before building that into the filter, this probe asks
the deciding question offline: when the PF's two heaviest basins compete,
does a graph that carries one basin's integers for a few seconds fit the
following observations better for the correct basin than for the wrong one?

Per contested epoch (second basin mass >= RBPF_BRANCH_MIN_MASS2, the two
basins disagree on at least one shared DD pair, no branch active):

1. copy the witness smoother's ISAM2 twice (`gtsam.ISAM2(other)`; the
   fixed-lag smoother itself cannot be copied from Python);
2. add each basin's DD integers as tight between-factors on the witness
   ambiguity keys (N_ref - N_j = z, cycles);
3. feed both copies every factor the witness adds for the next
   RBPF_BRANCH_EPOCHS epochs (no marginalization; a factor removal or a
   graph re-seed in the witness ends the event as `killed`);
4. score each branch by the error of those post-branch factors at its own
   estimate, and by its final antenna error against the reference.

Truth labels use the reference trajectory, so this is a diagnostic only.
The probe reads the PF and copies the witness graph; it changes neither,
and with RBPF_BRANCH_PROBE unset nothing here runs.
"""

from __future__ import annotations

import csv
import os

import gtsam
import numpy as np

from rbpf_fgo import RbpfGtsamRtkTc, _canon_basin_id

FIELDS = [
    "ep", "tow", "mass1", "mass2", "n_pairs", "n_diff", "n_hold1", "n_hold2",
    "unit_res1", "unit_res2", "y_off_med", "y_off_int", "y_off_n", "corr1", "corr2", "cm_err1", "cm_err2",
    "status", "age", "fit1", "fit2", "end_err1", "end_err2",
    "pf_end_mass1", "pf_end_mass2", "pf_end_map",
]


class _SmootherTee:
    """Forwards to the witness smoother and mirrors each update into the
    probe's live branches."""

    def __init__(self, inner, probe):
        self._inner = inner
        self._probe = probe

    def update(self, graph=None, values=None, ts=None, *rest, **kw):
        if graph is None:
            out = self._inner.update()
        else:
            out = self._inner.update(graph, values, ts, *rest, **kw)
        removal = rest[0] if rest else kw.get("factorsToRemove")
        self._probe._mirror(graph, values, removal)
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Branch:
    def __init__(self, isam):
        self.isam = isam
        self.graphs = []


class BasinBranchProbe:
    def __init__(self, witness, out_csv):
        self.w = witness
        self.epochs = int(os.environ.get("RBPF_BRANCH_EPOCHS", "15"))
        self.min_mass2 = float(os.environ.get("RBPF_BRANCH_MIN_MASS2", "0.05"))
        self.cooldown = int(os.environ.get("RBPF_BRANCH_COOLDOWN", "5"))
        self.hold_sigma = float(os.environ.get("RBPF_BRANCH_HOLD_SIGMA", "1e-3"))
        self.event = None
        self.since = 10**9
        self.n_events = 0
        self._fh = open(out_csv, "w", newline="")
        self._csv = csv.DictWriter(self._fh, fieldnames=FIELDS)
        self._csv.writeheader()

    # ---- witness hook -------------------------------------------------
    def wrap(self):
        sm = self.w.smoother
        if sm is not None and not isinstance(sm, _SmootherTee):
            if self.event is not None:  # graph re-seeded under the branches
                self._finish("killed_reseed", None)
            self.w.smoother = _SmootherTee(sm, self)

    def _mirror(self, graph, values, removal):
        ev = self.event
        if ev is None or graph is None:
            return
        if removal is not None and len(removal) > 0:
            ev["killed"] = "killed_removal"
            return
        if ev.get("killed"):
            return
        try:
            for br in ev["branches"]:
                # The branch never marginalizes, so a key the witness
                # re-creates after its lag window dropped it is already
                # here: keep the branch value, add only the factors.
                vals = gtsam.Values(values) if values is not None else gtsam.Values()
                for k in list(vals.keys()):
                    if br.isam.valueExists(k):
                        vals.erase(k)
                br.isam.update(graph, vals)
                br.graphs.append(graph)
        except RuntimeError as exc:
            if os.environ.get("RBPF_BRANCH_DEBUG"):
                import traceback
                traceback.print_exc()
            msg = str(exc).splitlines()[0][:60] if str(exc) else ""
            ev["killed"] = ("killed " + type(exc).__name__ + " " + msg).replace(",", ";")

    # ---- per epoch ----------------------------------------------------
    def after_epoch(self, rtk, ep, tow):
        self.wrap()
        self.since += 1
        if self.event is not None:
            self.event["age"] += 1
            if self.event.get("killed"):
                self._finish(self.event["killed"], rtk)
            elif self.event["age"] >= self.epochs:
                self._finish("done", rtk)
            return
        if self.since < self.cooldown:
            return
        self._maybe_start(rtk, ep, tow)

    def _basin_masses(self, pf):
        lw = np.array([p.logw for p in pf.particles], dtype=float)
        if not len(lw) or not np.all(np.isfinite(lw)):
            return {}
        w = np.exp(lw - lw.max())
        w /= w.sum()
        mass, rep = {}, {}
        for p, wi in zip(pf.particles, w):
            if not p.H:
                continue
            bid = _canon_basin_id(p.H)
            mass[bid] = mass.get(bid, 0.0) + float(wi)
            rep.setdefault(bid, p.H)
        return {b: (m, rep[b]) for b, m in mass.items()}

    def _maybe_start(self, rtk, ep, tow):
        pf = rtk.rbpf
        prob = getattr(pf, "_last_prob", None)
        sm = self.w.smoother
        if prob is None or not getattr(prob, "ok", False) or sm is None:
            return
        masses = self._basin_masses(pf)
        if len(masses) < 2:
            return
        top = sorted(masses.items(), key=lambda kv: -kv[1][0])[:2]
        (b1, (m1, H1)), (b2, (m2, H2)) = top
        if m2 < self.min_mass2:
            return
        d1, d2 = pf._pair_dd(H1, prob), pf._pair_dd(H2, prob)
        both = np.isfinite(d1) & np.isfinite(d2)
        if not np.any(both & (d1 != d2)):
            return
        truth = getattr(rtk, "_rbpf_truth_ecef", None)
        base = sm.getISAM2()
        est0 = base.calculateEstimate()
        branches, nhold, unit = [], [], []
        for dd in (d1, d2):
            isam = gtsam.ISAM2(base)
            g = gtsam.NonlinearFactorGraph()
            noise = gtsam.noiseModel.Isotropic.Sigma(1, self.hold_sigma)
            res = []
            for pi, (grp, ref, j) in enumerate(prob.pairs):
                if not np.isfinite(dd[pi]):
                    continue
                kr = self.w.amb_keys.get((int(ref), grp[1]))
                kj = self.w.amb_keys.get((int(j), grp[1]))
                if kr is None or kj is None or not (est0.exists(kr) and est0.exists(kj)):
                    continue
                g.add(gtsam.BetweenFactorDouble(kj, kr, float(dd[pi]), noise))
                res.append(est0.atDouble(kr) - est0.atDouble(kj) - dd[pi])
            if g.size() == 0:
                return
            try:
                isam.update(g, gtsam.Values())
            except RuntimeError:
                return
            branches.append(_Branch(isam))
            nhold.append(int(g.size()))
            unit.append(float(np.median(np.abs(res))))
        # PF float DD (main graph) vs witness float DD on the same pairs:
        # an integer offset would mean the two graphs define the SD
        # ambiguities differently and the holds are mis-mapped.
        offs = []
        for pi, (grp, ref, j) in enumerate(prob.pairs):
            kr = self.w.amb_keys.get((int(ref), grp[1]))
            kj = self.w.amb_keys.get((int(j), grp[1]))
            if kr is None or kj is None or not (est0.exists(kr) and est0.exists(kj)):
                continue
            offs.append(est0.atDouble(kr) - est0.atDouble(kj) - float(prob.y[pi]))
        offs = np.asarray(offs)
        y_off_med = float(np.median(np.abs(offs))) if len(offs) else float("nan")
        y_off_int = (int(np.sum((np.abs(np.rint(offs)) >= 1)
                                & (np.abs(offs - np.rint(offs)) < 0.2)))
                     if len(offs) else 0)
        corr, cm_err = [], []
        for H in (H1, H2):
            xz = pf._conditional_mean(H, prob)[0]
            e = (float(np.linalg.norm(xz - truth))
                 if xz is not None and truth is not None else float("nan"))
            cm_err.append(e)
            corr.append(int(e <= 0.5) if np.isfinite(e) else -1)
        self.event = dict(
            branches=branches, bids=(b1, b2), age=0,
            row=dict(ep=ep, tow=round(float(tow), 2), mass1=round(m1, 4),
                     mass2=round(m2, 4), n_pairs=len(prob.pairs),
                     n_diff=int(np.sum(both & (d1 != d2))),
                     n_hold1=nhold[0], n_hold2=nhold[1],
                     unit_res1=round(unit[0], 3), unit_res2=round(unit[1], 3),
                     y_off_med=round(y_off_med, 3), y_off_int=y_off_int,
                     y_off_n=len(offs),
                     corr1=corr[0], corr2=corr[1],
                     cm_err1=round(cm_err[0], 3), cm_err2=round(cm_err[1], 3)))
        self.n_events += 1

    def _finish(self, status, rtk):
        ev, self.event = self.event, None
        self.since = 0
        row = dict(ev["row"], status=status, age=ev["age"])
        if status == "done" and rtk is not None:
            truth = getattr(rtk, "_rbpf_truth_ecef", None)
            for i, br in enumerate(ev["branches"], start=1):
                try:
                    est = br.isam.calculateEstimate()
                    row[f"fit{i}"] = round(sum(g.error(est) for g in br.graphs), 3)
                    pose = est.atPose3(self.w.PX(self.w.last_valid_epoch))
                    ant = RbpfGtsamRtkTc._antenna_ecef(self.w, pose)
                    row[f"end_err{i}"] = (round(float(np.linalg.norm(ant - truth)), 3)
                                          if truth is not None else "")
                except RuntimeError:
                    row[f"fit{i}"] = ""
            masses = self._basin_masses(rtk.rbpf)
            row["pf_end_mass1"] = round(masses.get(ev["bids"][0], (0.0,))[0], 4)
            row["pf_end_mass2"] = round(masses.get(ev["bids"][1], (0.0,))[0], 4)
            top = max(masses.items(), key=lambda kv: kv[1][0])[0] if masses else None
            row["pf_end_map"] = (1 if top == ev["bids"][0] else
                                 2 if top == ev["bids"][1] else 0)
        self._csv.writerow(row)
        self._fh.flush()

    def close(self):
        if self.event is not None:
            self._finish("unfinished", None)
        self._fh.close()
