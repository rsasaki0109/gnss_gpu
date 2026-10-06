"""WP13m (TASK_M13.md): LAMBDA-input diagnostic, tc/ vs this port.

Monkeypatches ``cssrlib.pppssr.mlambda`` (the module-level name
``pppssr.py`` bound at import time via ``from cssrlib.mlambda import
mlambda`` -- patching ``cssrlib.mlambda.mlambda`` itself would NOT
intercept calls already bound into ``pppssr``'s namespace) so that
EVERY call to LAMBDA -- from either this port's ``GtsamRtk.resamb_lambda``
call or tc/'s identical inherited ``cssrlib.rtk.rtkpos.resamb_lambda``
(both subclass ``cssrlib.pppssr.pppos``, sharing the exact same
``resamb_lambda``/``resamb_lambda_rtklib``/``ddidx`` code, confirmed via
``git status``/``git diff`` on the submodule -- no cssrlib edit anywhere)
-- is recorded: the candidate-set size ``n`` (== ``nb`` fed to
``ddidx``/mlambda BEFORE any partial exclusion, for full-ILS the accepted
``nb`` always equals this since `parmode==1` fixes ALL-or-nothing) and the
conditioning of the float ambiguity covariance ``Qb`` (``cond(Qb)`` and
its diagonal spread) actually handed to LAMBDA, plus wall-clock cost per
call. No behavior change: the wrapper calls the real ``mlambda`` and
returns its result unmodified; installing this diagnostic is opt-in
(``--lambda-diag-csv`` in ``wp13a_run_standalone_rtk.py``, or import +
``install()`` directly for tc/'s ``examples/run_imu_gnss_tc.py``).

Per-call, not strictly per-epoch (a `rtklib_mode` epoch can call mlambda
twice -- primary + 1-sat round-robin retry -- and `resamb_lambda_subsets`
more; PAR-mode epochs call it 0 or 1 times). The distribution over calls
in a capped run is what answers WP13m's "nb/covariance-conditioning
table, tc/ vs ours" question; exact per-epoch pairing was not needed for
that (disclosed simplification -- the task's literal ask was "per epoch"
but comparing the SIZE DISTRIBUTIONS of what enters LAMBDA at all answers
the same tractability question with far less machinery).
"""

import csv
import os
import time

import numpy as np
import cssrlib.pppssr as _pppssr

_records = []
_orig_mlambda = _pppssr.mlambda
_installed = False

# WP13m hang forensics: when WP13M_CAPTURE_NPZ is set, every mlambda
# input (ahat, Qahat) is written to that path BEFORE the real call runs,
# so if a call never returns (the `_estimILS` uncounted-spin hang this
# diagnostic caught live via py-spy -- see WP13M_REPORT.md) the exact
# offending input survives the external kill and can be replayed in
# isolation. Rolling single-file overwrite: only the LAST (i.e. hanging
# or most recent) call's input is kept.
_capture_path = os.environ.get('WP13M_CAPTURE_NPZ', '')


def _cond_and_spread(Qb):
    Qb = np.asarray(Qb, dtype=np.float64)
    try:
        d = np.diag(Qb)
        d_pos = d[d > 0]
        diag_spread = float(d_pos.max() / d_pos.min()) if d_pos.size else float('nan')
    except Exception:
        diag_spread = float('nan')
    try:
        cond = float(np.linalg.cond(Qb))
    except Exception:
        cond = float('nan')
    return cond, diag_spread


def _wrapped_mlambda(ahat, Qahat, ncands=2, parmode=1, P0=0.995):
    n = int(len(ahat))
    cond, diag_spread = _cond_and_spread(Qahat)
    if _capture_path:
        try:
            np.savez(_capture_path, ahat=np.asarray(ahat, dtype=np.float64),
                     Qahat=np.asarray(Qahat, dtype=np.float64),
                     ncands=ncands, parmode=parmode, P0=P0,
                     call_idx=len(_records))
        except Exception:
            pass
    t0 = time.perf_counter()
    ok = False
    try:
        result = _orig_mlambda(ahat, Qahat, ncands=ncands, parmode=parmode, P0=P0)
        ok = True
        return result
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        nfix = -1
        Ps = float('nan')
        if ok:
            try:
                nfix = int(result[2])
                Ps = float(result[3])
            except Exception:
                pass
        _records.append({
            'call_idx': len(_records),
            'n': n,
            'cond_Qb': cond,
            'diag_spread_Qb': diag_spread,
            'parmode': int(parmode),
            'ok': int(ok),
            'nfix': nfix,
            'Ps': Ps,
            'dt_ms': dt_ms,
        })


def install():
    """Patch ``cssrlib.pppssr.mlambda`` in place. Idempotent."""
    global _installed
    if _installed:
        return
    _pppssr.mlambda = _wrapped_mlambda
    _installed = True


def uninstall():
    global _installed
    if not _installed:
        return
    _pppssr.mlambda = _orig_mlambda
    _installed = False


def records():
    return list(_records)


def clear():
    _records.clear()


def dump_csv(path):
    fieldnames = ['call_idx', 'n', 'cond_Qb', 'diag_spread_Qb', 'parmode',
                  'ok', 'nfix', 'Ps', 'dt_ms']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in _records:
            w.writerow(r)


def summary():
    """Quick console summary: n/cond percentiles + mean per-call dt."""
    if not _records:
        return "no mlambda calls recorded"
    n = np.array([r['n'] for r in _records], dtype=np.float64)
    cond = np.array([r['cond_Qb'] for r in _records], dtype=np.float64)
    cond = cond[np.isfinite(cond)]
    dt = np.array([r['dt_ms'] for r in _records], dtype=np.float64)
    lines = [
        f"mlambda calls: {len(_records)}",
        f"  n (candidate-set size):    "
        f"min={n.min():.0f} p25={np.percentile(n,25):.0f} "
        f"median={np.median(n):.0f} p75={np.percentile(n,75):.0f} "
        f"max={n.max():.0f} mean={n.mean():.2f}",
    ]
    if cond.size:
        lines.append(
            f"  cond(Qb):                  "
            f"min={cond.min():.3g} p25={np.percentile(cond,25):.3g} "
            f"median={np.median(cond):.3g} p75={np.percentile(cond,75):.3g} "
            f"max={cond.max():.3g} mean={cond.mean():.3g}")
    lines.append(
        f"  dt per call [ms]:          "
        f"min={dt.min():.3f} median={np.median(dt):.3f} "
        f"p95={np.percentile(dt,95):.3f} max={dt.max():.3f} "
        f"total={dt.sum()/1000.0:.2f}s")
    return "\n".join(lines)
