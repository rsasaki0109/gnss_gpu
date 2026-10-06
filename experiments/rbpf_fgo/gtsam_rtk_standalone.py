"""WP13a: standalone RTK-on-FGO (GtsamRtk), self-contained port.

Ported from inuex35/gnss-gtsam-rtk `src/gnss_fgo/rtk.py` (~300 lines,
IMU-free), adapted to the cssrlib fork installed in this venv
(`.venv/src/cssrlib`, a "minimal core" fork that renamed/removed a few
`rtkpos`/`pppos` methods -- see the deltas noted inline below) and to the
gtsam build installed in this venv, which already exposes the DD factors
as C++ `NoiseModelFactor`s under the names
`DoubleDifferenceCarrierPhaseFactor` / `DoubleDifferencePseudorangeFactor`
(NOT `DDCarrierPhaseFactor` -- that name doesn't exist, which is what an
earlier `hasattr(gtsam, 'DDCarrierPhaseFactor')` check found). Using the
already-built, already-integration-tested C++ factors is more faithful to
the reference than re-deriving the DD Jacobian as a Python CustomFactor,
and does not require rebuilding gtsam.

No EKF: position comes from GTSAM (ISAM2 or IncrementalFixedLagSmoother),
covariance for LAMBDA AR comes from `gtsam.Marginals.jointMarginalCovariance`
over [x, active ambiguity keys] -- THE crux of WP13a. `nav.x`/`nav.P` are
overwritten from the GTSAM estimate + joint marginal every epoch and then
handed to cssrlib's `resamb_lambda` (LAMBDA AR), `valpos` (chi-square
residual gate) and `holdamb_flags` (fix-and-hold bookkeeping).

Deltas from the upstream/TC-repro reference (both consequences of the
installed cssrlib fork being a rewritten "minimal core", not upstream
cssrlib):
  * `rtkpos.base_process(...)` returns `(None, None, iu, obs_sd)` (the
    fork removed the undifferenced zdres path); we use
    `single_differences` directly, matching `prepare_double_difference_
    measurements`'s own internals.
  * `rtkpos.holdamb(xa)` (the reference's Kalman-update hold) does not
    exist in this fork -- only `holdamb_flags()`, documented as the
    intended replacement "in pipelines that overwrite nav.x/nav.P from
    another source (e.g. GTSAM marginals) every epoch". The TC repro's
    own copy of this module (`tc/src/gnss_fgo/utils/rtk.py`) still calls
    the non-existent `self.holdamb(xa)`, silently swallowed by
    `_do_ar`'s bare `except Exception: return` -- i.e. fix-and-hold
    re-injection was a no-op there. Fixed here to call `holdamb_flags()`.

WP13c (TASK_M3.md) adds an AR acceptance gate + hold-release on top of
this: WP13a found that `rtkpos.valpos` is a no-op in this fork (always
returns True), so LAMBDA's own ratio test (`s[1]/s[0] >= nav.thresar`,
default 2.0-3.0) was the ONLY thing standing between a wrong integer and
a *permanent* GTSAM prior via fix-and-hold -- and it isn't enough: 100%
"fixed" epochs with 20-192m FixRMS (`results/wp13a/WP13A_REPORT.md`
section 4). `_do_ar`/`_inject_hold`/`_check_release` below replace that
single gate with three independently-tunable ones (extra ratio bar,
N-consecutive-epoch agreement, absolute post-fit residual cap) that
must all pass before a hold is ever committed, plus a release mechanism
for a hold that turns out wrong after all -- see `_do_ar`'s own
docstring for the full design and `results/wp13c/WP13C_REPORT.md` for
the sweep evidence and final numbers. WP13c's gate only *relabels*
which epochs report Q=4 -- it never touches `nav.x` itself, so it could
not fix WP13a/c's real problem: the underlying FLOAT trajectory
(`nav.x`) is poor (AllRMS 57-192m).

WP13d (TASK_M4.md) targets that float trajectory directly, by
transcribing inuex35's GICI-style Fault Detection & Exclusion
(`apply_fde` + helpers, from `tightly-coupled-gnss-imu-fgo`
`src/gnss_fgo/validation/postfit.py`) onto this GNSS-only standalone:
`_apply_fde`/`_fde_collect_residuals`/`_fde_pick_rejects_single_pass`/
`_fde_pick_rejects_iterative`/`_fde_reset_rejected_amb` below evaluate
each DD-PR/DD-CP factor THIS EPOCH added to the smoother, in meters,
and REMOVE (`smoother.update(..., removeFactorIndices=...)`) the ones
whose residual is a statistical outlier relative to the epoch's own
median -- this changes `nav.x` itself (the estimate written back after
FDE runs), unlike WP13c's reporting-only gate. A rejected DD-CP factor
is treated as a cycle slip: `_fde_reset_rejected_amb` abandons that
(sat,freq)'s ambiguity via the same `_release_ambiguity` mechanism
WP13c's hold-release uses (bump generation, drop from `amb_keys`,
release any WP13c hold) -- one shared mechanism, two triggers.

Disclosed adaptation vs. the literal reference: the reference locates
"this epoch's newly-added factor indices" via `fi_start = nf_total -
g3.size()` (assumes the smoother's factor vector grows by *exactly*
`g3.size()` per update). Empirically verified NOT reliable for our
`IncrementalFixedLagSmoother` at `--lag 1.0`: once the lag window is
full, marginalization appends its own extra `LinearContainerFactor`(s)
*after* our own new factors on the same `update()` call, so `nf_total -
graph.size()` under-shoots (the marginal factor(s) get incorrectly
counted as part of "this epoch's factors", and worse, would shift
`fi_start` too far forward if more than one is added). Fixed here by
instead recording `nf_before = factors.size()` immediately BEFORE the
epoch's own `update()` call -- verified empirically that our own
`graph`'s factors always land FIRST, at `[nf_before, nf_before +
graph.size())`, with any marginalization byproduct appended strictly
after -- so this range is exact, not an approximation, for our smoother
(see WP13D_REPORT.md for the empirical check that established this).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import gtsam

from cssrlib.rtk import rtkpos
from cssrlib.gnss import (
    uGNSS, uTYP, sat2prn, geodist, timediff, time2gpst, ecef2pos, gpst2time,
    id2sat)
# Private helper (no cssrlib edit -- just calling an existing function):
# `sdres` returns a single mixed PR+CP residual vector `v`, and a naive
# "any |v[i]| exceeds thres" gate (WP13a's own disclosed-negative
# VALPOS_THRES attempt, and this WP13c task's first attempt at the same
# idea) is dominated by ordinary meter-scale pseudorange multipath noise
# (urban Tokyo), collapsing the pipeline (rejects ~every epoch) long
# before it ever says anything about ambiguity CORRECTNESS. Re-deriving
# `_sdres_build_plan`'s own is_phase mask (same call `sdres` makes
# internally, same row order) lets HOLD_VALPOS_THRES look at only the
# carrier-phase rows, whose residual scale (mm-cm for a correct integer,
# ~1 wavelength or more for a wrong one) is actually diagnostic.
from cssrlib.pppssr import _sdres_build_plan

# WP13s (TASK_M19.md, べた移植): import tc/'s OWN modules directly where
# interfaces allow (precedent: wp13a_run_standalone_rtk.py already
# sys.path's tc/src and imports gnss_fgo.utils.*). tc/ stays READ-ONLY --
# these are plain imports. Guarded so importing this module without tc/
# on disk still works (TC_LITERAL mode is then unavailable).
_TC_SRC_DIR = Path(__file__).resolve().parent / "tc" / "src"
if _TC_SRC_DIR.is_dir() and str(_TC_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_TC_SRC_DIR))
try:
    from gnss_fgo.preprocess.sat_quality import (  # noqa: E402
        SatQualityState as _TcSatQualityState)
    from gnss_fgo.buildfactor.factors_support import (  # noqa: E402
        compute_cp_build_policy as _tc_compute_cp_build_policy)
    from gnss_fgo.buildfactor.factors import (  # noqa: E402
        _make_ddcp_factor_with_held_n as _tc_make_ddcp_factor_with_held_n)
    from gnss_fgo.utils.geometry import (  # noqa: E402
        compute_gdop as _tc_compute_gdop)
except ImportError:  # pragma: no cover - tc/ not present
    _TcSatQualityState = None
    _tc_compute_cp_build_policy = None
    _tc_make_ddcp_factor_with_held_n = None
    _tc_compute_gdop = None

# WP13m (TASK_M13.md): guard cssrlib's module-level `mlambda` binding
# against NON-FINITE input -- the root cause of WP13k's ">40x slower"
# full-ILS measurement, which this WP re-diagnosed with py-spy on a live
# hung run + a captured-input replay (results/wp13m_smoke/
# last_mlambda_input.npz): the VERY FIRST `resamb_lambda` call of a
# Phase-1 cold start can carry a `Qb` containing NaN (GTSAM
# `jointMarginalCovariance` on a still-under-determined cold-start graph
# -- finite `ahat`, NaN covariance). `mlambda.parsearch` (parmode=2/PAR,
# this port's previous only path) happens to DEGRADE GRACEFULLY on NaN:
# `while Ps < P0` is entered with Ps=NaN, every NaN comparison is False,
# so it falls straight through to `nfix=0` (reject, report float) in
# microseconds. `_estimILS` (parmode=1/full ILS, tc/'s literal path) does
# NOT: its LOOPMAX abort counter is only incremented INSIDE its two inner
# `while newdist < Chi2` / `while newdist >= Chi2` loops, and a NaN
# `newdist` makes BOTH loop conditions False, so the OUTER
# `while not endSearch` loop spins forever with `loop_count` never
# advancing -- an infinite, LOOPMAX-bypassing hang inside njit code
# (confirmed: 3x py-spy samples all inside `estimILS`, 14+ CPU-minutes on
# a single call; replaying the captured input reproduces the hang in
# isolation, and the SAME input returns nfix=0 in <1ms under parmode=2).
# tc/ never trips this because its EKF covariance is finite by
# construction. The guard raises `LambdaError` -- the SAME exception
# cssrlib's own `ldldecom` already raises for a non-PD `Qb`, caught by
# the SAME `except` in `_run_lambda_attempts` (-> nb=0, report float) --
# i.e. full-ILS now degrades on NaN exactly the way PAR always did here,
# no other behavior change, no cssrlib edit (monkeypatch of the
# `cssrlib.pppssr.mlambda` module-level name from OUR module, same
# technique as the opt-in `wp13m_lambda_diag`; order-independent with it).
import cssrlib.pppssr as _cssrlib_pppssr  # noqa: E402
from cssrlib.mlambda import LambdaError as _LambdaError  # noqa: E402

_wp13m_unguarded_mlambda = _cssrlib_pppssr.mlambda


def _wp13m_finite_guarded_mlambda(ahat, Qahat, ncands=2, parmode=1, P0=0.995):
    if not (np.all(np.isfinite(ahat)) and np.all(np.isfinite(Qahat))):
        raise _LambdaError(
            "non-finite LAMBDA input (WP13m guard: cold-start GTSAM "
            "marginal produced NaN covariance; parmode=1 _estimILS would "
            "hang past LOOPMAX on this input -- reject like PAR does)")
    return _wp13m_unguarded_mlambda(ahat, Qahat, ncands=ncands,
                                    parmode=parmode, P0=P0)


_cssrlib_pppssr.mlambda = _wp13m_finite_guarded_mlambda

# WP13e (TASK_M5.md): reuse (not reimplement) inuex35's own IMU
# preintegration + geometry utility functions -- `tc/src/gnss_fgo/utils/
# imu.py` (make_imu_params/build_pim/estimate_stationary_bias) and
# `.../utils/geometry.py` (euler_to_R_body2enu/heading_from_vel), pure
# functions with no dependency on their ImuGnssTc FSM object. Idempotent
# path insert -- `wp13a_run_standalone_rtk.py` already does this before
# importing this module, but keep this module independently importable.
_TC_SRC = Path(__file__).resolve().parent / "tc" / "src"
if str(_TC_SRC) not in sys.path:
    sys.path.insert(0, str(_TC_SRC))
from gnss_fgo.utils.imu import (  # noqa: E402
    make_imu_params, build_pim, estimate_stationary_bias, compute_zupt_stats)
from gnss_fgo.utils.geometry import (  # noqa: E402
    euler_to_R_body2enu, heading_from_vel, load_imu_csv)
# WP13h (TASK_M8.md): NHC/ZUPT/Doppler motion-constraint backbone. The
# LS math (`doppler_velocity_ls`) is a pure function, reused directly
# (not reimplemented) exactly like `build_pim`/`estimate_stationary_bias`
# above. The NHC/ZUPT *factor construction* (`buildfactor/{nhc,zupt}.py`)
# is transcribed as GtsamRtkTc's own methods below instead of imported
# verbatim, to match this file's established flat `self.*` config
# convention (`self.nhc_enable`, not `tc.cfg.nhc_enable`) -- same
# transcription style WP13g used for `postfit.py`'s sanity family.
from gnss_fgo.utils.ls_solvers import doppler_velocity_ls as _tc_doppler_velocity_ls  # noqa: E402


# WP13j (TASK_M10.md): inject the PLATEAU ray-traced NLOS prior --
# per-epoch, per-satellite is_los masks generated by
# `gnss_gpu/experiments/build_per_epoch_nlos_csv.py` at
# `gnss_gpu/experiments/results/plateau_nlos_phase33/tokyo_run{N}_
# per_epoch_nlos.csv` (columns tow,epoch_idx,prn,is_los,system,svid,
# elevation_deg,receiver_source,receiver_time_delta_s -- `prn` is an
# RTKLIB-style id string, e.g. "G05"/"C11"/"E24", directly consumable by
# cssrlib's own `id2sat`). Confirmed (wp13j_nlos_align_check.py) TOW-
# aligned to this port's own synced rover epochs at ~100% (run1 11924/
# 11928, run2/3 exact), covering 83-86% of observed sat-epochs (the mask
# ray-traces down to its own elevation floor, not every satellite this
# port tracks) with a 30-36% NLOS rate among covered sat-epochs (~39-44%
# of approximate DD pairs touch at least one NLOS-flagged sat) -- see
# WP13J_REPORT.md for the full alignment table.
#
# Injected at three independently-ablatable layers (this task's own
# experiment, WP7/WP8 already established layer (b) negative on the
# older EKF -- kept here only for completeness):
#   (a) AR candidate exclusion (NLOS_AR_EXCLUDE): a flagged satellite's
#       `nav.vsat` row is temporarily zeroed before `resamb_lambda` (same
#       mechanism `resamb_lambda_rtklib`'s own round-robin exclusion
#       uses -- `_ddidx_core` skips any (sat,freq) with `vsat==0`), then
#       restored -- so LAMBDA never considers that satellite as a DD
#       reference or candidate this epoch, using the mask as the
#       INDEPENDENT signal WP13i's post-fit DD-PR cross-check could not
#       be (multipath corrupts PR+CP together; ray-tracing does not).
#   (c) Wrong-fix rejection (NLOS_COMMIT_REJECT): a LAMBDA candidate
#       whose satellite is flagged NLOS this epoch is never promoted
#       into `self._committed` (the permanent hold set) even if it
#       otherwise clears WP13c's ratio/confirm-count gate; and if a
#       MAJORITY of this epoch's active candidates are NLOS-flagged, the
#       whole epoch's Q=4 report is refused (falls back to float) --
#       the independent gate the inert DD-PR cross-check (WP13i Part 3)
#       could not provide.
#   (b) DD measurement weighting (NLOS_MEAS_MODE): inflate (mode=2,
#       `NLOS_MEAS_SIGMA_MULT`) or hard-exclude (mode=1) a DD-PR/DD-CP
#       factor when either its ref or target satellite is NLOS-flagged
#       this epoch.
def _load_nlos_mask(run):
    """Load `tokyo_run{run}_per_epoch_nlos.csv` into
    ``{tow_rounded_0.1s: set(sat_int for NLOS (is_los==0) rows)}``.
    Returns (mask_dict, n_rows, n_nlos_rows, n_bad_prn) for the
    alignment-stats report. Pure stdlib `csv` (no pandas dependency)."""
    import csv as _csv
    # NLOS_MASK_DIR (WP39): the phase33 default was ray-traced at the
    # REFERENCE (ground-truth) position; point this at a mask built from an
    # estimated trajectory for a non-oracle run.
    mask_dir = os.environ.get(
        'NLOS_MASK_DIR',
        r"C:\Users\rsasa\Workspace\old\gnss_gpu\experiments\results"
        r"\plateau_nlos_phase33")
    path = Path(mask_dir) / f"tokyo_run{run}_per_epoch_nlos.csv"
    mask = {}
    n_rows = 0
    n_nlos = 0
    n_bad_prn = 0
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            n_rows += 1
            if int(row["is_los"]) != 0:
                continue
            n_nlos += 1
            try:
                sat = id2sat(row["prn"])
            except Exception:
                sat = -1
            if sat <= 0:
                n_bad_prn += 1
                continue
            tow_r = round(float(row["tow"]), 1)
            mask.setdefault(tow_r, set()).add(int(sat))
    return mask, n_rows, n_nlos, n_bad_prn


def _sorted_sys_ids(sig_map):
    """Deterministic system-id iteration order (GPS=1, GLO=2, ...)."""
    return sorted(sig_map.keys(), key=int)


def _sorted_amb_items(amb_dict):
    """Deterministic ``(sat, freq) -> key`` iteration, sorted by sat then freq."""
    return sorted(amb_dict.items(), key=lambda item: (int(item[0][0]), int(item[0][1])))


def _get_wavelengths(nav, obs_sd, sat):
    """Carrier-phase wavelengths [m] for one satellite, indexed by frequency."""
    sys_i, _ = sat2prn(sat)
    if sys_i not in obs_sd.sig:
        return []
    if uTYP.L not in obs_sd.sig[sys_i]:
        return []
    sigs = obs_sd.sig[sys_i][uTYP.L]
    if sys_i == uGNSS.GLO:
        ch_map = getattr(nav, 'glo_ch', {}) or {}
        return [s.wavelength(ch_map.get(sat, 0)) for s in sigs]
    return [s.wavelength() for s in sigs]


class GtsamRtk(rtkpos):
    """Standalone GNSS-only RTK on a GTSAM factor graph (WP13a)."""

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, pos0, logfile)
        self.epoch = 0
        self.epoch_time = 0.0

        # WP13m (TASK_M13.md): this port never overrode cssrlib's own
        # Nav-class defaults for `elmin`/`cnr_min` (15deg/25dBHz, set in
        # cssrlib/gnss.py's Nav.__init__ and never touched by anything in
        # this file before now) -- unlike tc/'s `runner.py` (`self.nav.
        # elmin = deg2rad(cfg.elmin_deg); self.nav.cnr_min = cfg.
        # cnr_min_dbhz`), which under the `tokyo_mode2_satbad_cponly`
        # preset actually used for `results/tc_run*.npz` sets elmin_deg=25
        # (vs cssrlib's/this port's 15) and cnr_min_dbhz=30 (vs 25).
        # `qcedit` (cssrlib/pppssr.py, shared unmodified code -- both tc/
        # and this port subclass `pppos`) is what actually enforces these
        # per obs epoch, upstream of `prepare_double_difference_
        # measurements`/`ddidx`: a satellite below `nav.elmin` or `nav.
        # cnr_min` never enters `sat`/`el` (and therefore never enters a
        # DD-CP/DD-PR factor or the LAMBDA candidate set) at all, for
        # EITHER port -- the only difference is what these two thresholds
        # are set TO. This port's own `ddidx`-internal AR mask
        # (`nav.elmaskar`, cssrlib default 20deg, also never overridden by
        # either port) sits BETWEEN the two: with elmin=15, satellites in
        # [15,20)deg reach the DD-CP/DD-PR factor build but still get
        # dropped from the LAMBDA candidate set by elmaskar, so this
        # port's DE-FACTO AR floor was already ~20deg, not 15 -- but 30
        # dBHz CNR and 25deg elmin (tc/'s tighter pre-DD-factor filter)
        # still admit meaningfully fewer, cleaner satellites into the
        # DD-CP/DD-PR factors themselves (and hence the float covariance
        # LAMBDA reads), not just the AR candidate set. Opt-in via CLI
        # (`--elmin-deg`/`--cnr-min-dbhz`, defaults 15/25 = cssrlib's own
        # unchanged defaults, i.e. a no-op unless explicitly overridden --
        # WP13l's shipped config is bit-identical without these flags).
        self.nav.elmin = np.deg2rad(float(os.environ.get('ELMIN_DEG', '15.0')))
        self.nav.cnr_min = float(os.environ.get('CNR_MIN_DBHZ', '25.0'))

        lag = float(os.environ.get('LAG', '0'))
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        if lag > 0:
            self.smoother = gtsam.IncrementalFixedLagSmoother(lag, params)
            self.isam = None
        else:
            self.isam = gtsam.ISAM2(params)
            self.smoother = None

        self.amb_keys = {}
        self.current_estimate = None
        # WP13c (TASK_M3.md): AR acceptance gate + hold-release. All state
        # below is OUR OWN bookkeeping, never read by cssrlib and never
        # conflated with `nav.fix` -- ddidx() rewrites `nav.fix` from
        # scratch on EVERY resamb_lambda call (see `_do_ar`'s docstring),
        # so any persistent "is this ambiguity trustworthy" concept must
        # live somewhere ddidx never touches. That's what these are.
        self.hold_ratio = float(os.environ.get('HOLD_RATIO', '0'))
        self.hold_confirm_n = int(os.environ.get('HOLD_CONFIRM_N', '1'))
        self.hold_valpos_thres = float(os.environ.get('HOLD_VALPOS_THRES', '0'))
        self.streak_tol = float(os.environ.get('HOLD_STREAK_TOL', '0.05'))
        self.min_committed_for_fix = int(os.environ.get('MIN_COMMITTED_FOR_FIX', '0'))
        self.release_k = int(os.environ.get('RELEASE_K', '0'))
        self.release_thres = float(os.environ.get('RELEASE_THRES', '1.0'))
        # WP13i (TASK_M9.md): post-fit DD-PR cross-check in the AR accept
        # gate (tc/'s `optimize/stage.py` `ar_ddpr_xvalidate_thresh`/
        # `_delta_thresh`, config.py defaults 10.0/0.0; our old local_fgo.py
        # had a `ddpr_reject_threshold` doing the same job). DD-PR residuals
        # depend ONLY on position, never on ambiguity values -- an
        # AMBIGUITY-INDEPENDENT cross-check the ratio test itself cannot
        # provide. Targets WP13h's disclosed false-fix increase (NHC/ZUPT
        # tightening the joint-marginal covariance the ratio test
        # normalizes by, without the underlying integer resolution actually
        # becoming more often correct): reject (this epoch only, report
        # float, `self._committed`/`self._streak` untouched -- see
        # `_ddpr_ar_xvalidate`'s docstring) a candidate LAMBDA fix whose
        # own DD-PR residual (evaluated at the fix's OWN, LAMBDA-refined
        # position `xa[0:3]`) is worse than `ar_ddpr_xvalidate_thresh`
        # (absolute, meters) or has worsened by more than
        # `ar_ddpr_xvalidate_delta` versus this epoch's pre-AR float DD-PR
        # residual (`self.last_main_ddpr_rms`). Only meaningful in Phase 2
        # (Pose3+lever) -- gated internally, a no-op for the GNSS-only
        # Point3 float this class also serves (Phase 1 already performs at
        # ~0.03m per WP13F_REPORT.md). `_delta` defaults off (0.0, matches
        # tc/'s own config.py default) since the absolute threshold alone
        # is tc/'s literal default behavior.
        self.ar_ddpr_xvalidate_thresh = float(
            os.environ.get('AR_DDPR_XVALIDATE_THRESH', '10.0'))
        self.ar_ddpr_xvalidate_delta = float(
            os.environ.get('AR_DDPR_XVALIDATE_DELTA_THRESH', '0.0'))
        self._ar_ddpr_xvalidate_reject_count = 0
        # nav.parmode already defaults to 2 (PAR, success-rate bootstrap)
        # via pppos.__init__ -- exposed here only so a controlled A/B
        # (parmode=1, full ILS) can be swept without a cssrlib edit.
        self.nav.parmode = int(os.environ.get('PARMODE', str(self.nav.parmode)))
        self.nav.par_P0 = float(os.environ.get('PAR_P0', str(self.nav.par_P0)))

        # WP13k (TASK_M11.md): tc/'s actual `_run_lambda_attempts` chain
        # (`optimize/ar.py`), transcribed WITHOUT editing cssrlib. Root
        # cause found (not just assumed) for why this port's fix rate
        # stayed ~5-9% while parmode=2 was already cssrlib's own default:
        # cssrlib's `mlambda.parsearch` (the actual PAR/parmode=2
        # implementation) hardcodes `exclmax=1` in its OWN signature
        # (`parsearch(..., exclmax=1)`, `mlambda()` never passes a
        # different value) -- i.e. cssrlib's own "partial AR" can only
        # ever exclude AT MOST ONE ambiguity from the fixed set before
        # giving up entirely (nfix=0) if reaching the P0=0.995 bootstrapped
        # success rate needs excluding 2+. In Tokyo urban-canyon geometry
        # that is the common case, so parmode=2 was structurally almost as
        # strict as full ILS here despite being "on". tc/'s OWN default AR
        # path does NOT rely on this at all: `rtklib_mode=1` (config.py's
        # own default) routes every call through `resamb_lambda_rtklib`
        # (full ILS, parmode=1 forced internally, + RTKLIB-style ONE-
        # satellite round-robin exclusion with `arfilter`) and, only when
        # THAT still fails outright, `optimize/ar.py`'s own `_try_subset_ar`
        # (drop up to `subset_ar_max_drop` "dirty" sats, ranked by DD-PR
        # residual/elevation, combinatorial retry) -- this is what tc/
        # itself calls "subset/partial AR" and is the actual 69%-fix-rate
        # mechanism, not cssrlib's own exclmax-capped PAR. Both cssrlib
        # methods (`resamb_lambda_rtklib`/`resamb_lambda`) are already
        # inherited for free via `GtsamRtk(rtkpos)` -> `rtkpos(pppos)`
        # (cssrlib/rtk.py, cssrlib/pppssr.py) -- no cssrlib edit, just
        # calling what's already there instead of bypassing it.
        #
        # WP13k measured `resamb_lambda_rtklib` as ">40x slower / did not
        # reach epoch 50 in 90s" and defaulted it OFF. WP13m (TASK_M13.md)
        # RE-DIAGNOSED that measurement with py-spy on a live hung run and
        # a captured-input replay: the slowness was NOT full-ILS search
        # cost at all -- it was ONE cold-start epoch whose GTSAM joint
        # marginal contained NaN, sending `mlambda._estimILS` into an
        # infinite LOOPMAX-bypassing spin (see the module-level
        # `_wp13m_finite_guarded_mlambda` docstring above for the full
        # mechanism + evidence chain). With that single input guarded
        # (reject-as-float, exactly what PAR already did on the same
        # input), full-ILS runs at ~24-30 ep/s here (vs ~33 ep/s PAR;
        # per-mlambda-call median 0.55ms, 334 calls / 0.27s total on a
        # 300ep capped smoke) -- fully tractable, and it is tc/'s literal
        # `rtklib_mode=1` path. The flag default stays 0 only so WP13l's
        # shipped config reproduces bit-identically without flags; WP13m's
        # own measured configs pass `--ar-rtklib-mode 1` explicitly.
        self.ar_rtklib_mode = bool(int(os.environ.get('AR_RTKLIB_MODE', '0')))
        self.nav.arfilter = bool(int(os.environ.get('AR_ARFILTER', '1')))
        self.nav.minfixsats = int(os.environ.get('AR_MINFIXSATS', str(self.nav.minfixsats)))
        self.subset_ar_enable = bool(int(os.environ.get('SUBSET_AR_ENABLE', '1')))
        self.subset_ar_max_drop = int(os.environ.get('SUBSET_AR_MAX_DROP', '2'))
        self.subset_ar_min_nb = int(os.environ.get('SUBSET_AR_MIN_NB', '4'))
        self.subset_ar_max_candidates = int(os.environ.get('SUBSET_AR_MAX_CANDIDATES', '5'))
        # WP13k: separate, lower ratio-report bar for subset-AR-sourced
        # epochs (see `_do_ar`'s docstring note at `ratio_ok` for why the
        # literal `hold_ratio` -- calibrated for full-ILS ratios -- is
        # structurally unreachable by PAR-mode's own `s1/s0`). Default 1.0
        # (nfix>0 alone is trusted, matching PAR's own acceptance
        # semantics: `par_P0`=0.995 bootstrapped success rate already
        # gated it inside `parsearch`) -- swept alongside fix%/false-fix%.
        self.subset_ar_hold_ratio = float(os.environ.get('SUBSET_AR_HOLD_RATIO', '1.0'))
        # WP13o: subset-cascade attempt routing -- see `_try_subset_ar`.
        self.subset_ar_use_par = int(os.environ.get('SUBSET_AR_USE_PAR', '0'))
        # WP15 (TASK_M21.md): batched-CUDA LAMBDA for the subset-AR
        # cascade (opt-in, `CUDA_LAMBDA=1`). The WP15 profile (cProfile +
        # py-spy on run2-3000ep, Q4+COND_HOLD) measured `_try_subset_ar`
        # at 16-19% of wall time: 2,039 of 3,000 epochs fire the cascade,
        # ~29.5k of the 34.6k total mlambda calls come from it, and each
        # CPU combo evaluation also pays cssrlib's `restamb` Python loop
        # (~1.8 ms) on every internally-accepted candidate whose result
        # is then discarded. `_try_subset_ar_cuda` evaluates ALL combos'
        # (y, Qb) in ONE gnss_gpu.lambda_batch kernel launch (a faithful
        # mlambda port, verified bit-identical on all 34,569 finite
        # captured calls -- see gnss_gpu/results/wp15/WP15_REPORT.md) and
        # re-fires only the winning subset through the normal CPU
        # `resamb_lambda`, so every nav.* side effect of the ACCEPTED
        # result is produced by the same code as before. Automatic CPU
        # fallback when the extension/GPU is unavailable and (for exact
        # state emulation) on the rare cascades where a combo was
        # internally accepted but rejected by `min_nb` (its transient
        # nav.xa/Pa/restamb side effects cannot be skipped safely).
        # Default 0 = bit-identical CPU path.
        self.cuda_lambda = int(os.environ.get('CUDA_LAMBDA', '0'))
        self._cuda_mlambda_batch = None
        self.cuda_lambda_batch_count = 0
        self.cuda_lambda_combo_count = 0
        self.cuda_lambda_fallback_count = 0
        # WP15_TIME=1: opt-in cascade wall-time accumulator (see the
        # `_try_subset_ar` dispatcher); print-only, default off.
        self._wp15_time = int(os.environ.get('WP15_TIME', '0'))
        self.wp15_cascade_seconds = 0.0
        if self.cuda_lambda:
            try:
                _gp = os.environ.get(
                    'GNSS_GPU_PY',
                    str(Path(__file__).resolve().parent.parent
                        / 'gnss_gpu' / 'python'))
                if _gp not in sys.path:
                    sys.path.insert(0, _gp)
                from gnss_gpu.lambda_batch import (
                    mlambda_batch as _wp15_mlb,
                    HAS_LAMBDA_BATCH as _wp15_has)
                if not _wp15_has:
                    raise ImportError(
                        '_gnss_gpu_lambda_batch native module missing')
                # One-shot probe: raises if no CUDA device / driver.
                _wp15_mlb([np.array([0.1, 0.2])], [np.eye(2) * 0.1],
                          ncands=2, parmode=1)
                self._cuda_mlambda_batch = _wp15_mlb
                print('WP15: CUDA_LAMBDA active (gnss_gpu.lambda_batch)')
            except Exception as _e_wp15:
                print(f'WP15: CUDA_LAMBDA=1 but GPU path unavailable '
                      f'({_e_wp15}); using CPU fallback')
        self._last_resamb_raw_nb = -1
        self._last_subset_ar_used = False
        self._last_subset_ar_drop = []
        self.subset_ar_used_count = 0
        self.subset_ar_attempt_count = 0
        # This-epoch's own pre-AR per-satellite DD-PR/DD-CP residual [m]
        # (tc/'s `_last_main_ddpr_per_sat` / `_cached_ddpr_res_pre`
        # analogue) -- populated by `_compute_main_dd_res` (Phase 2 only;
        # stays `{}` in Phase 1's brief GNSS-only bootstrap, degrading
        # `_rank_subset_drop_sats` to an elevation-only ranking there,
        # which is fine -- Phase 1 is a ~200-epoch warm-up, not the
        # operative long-run path per WP13i's own disclosed scope note).
        self._main_ddpr_per_sat = {}

        # WP13k priority 2: sat_badness EWMA (tc/'s `preprocess/
        # sat_quality.py` `SatQualityState.sat_badness`/
        # `update_observation_quality`), reduced to its dominant term for
        # this port (the DD-PR-residual EWMA feeding `sigma_scale_cp`,
        # `alpha_ddpr`/`sigma_scale_cp` are the two non-zero knobs in tc/'s
        # own `tokyo_mode2_satbad_cponly` preset with the largest
        # contribution) -- disclosed simplification: tc/'s fuller score
        # also folds in CP-PR-reject counts, per-sat/per-ref/per-pair
        # short-memory badness and el/SNR penalties, which need
        # infrastructure (`rejc_cp_pr`, ref-sat bookkeeping) this port
        # does not track; those terms are left at their tc/-preset-implied
        # zero contribution here (`alpha_cppr` etc. effectively 0) rather
        # than fabricated.
        self.sat_badness_enable = bool(int(os.environ.get('SAT_BADNESS_ENABLE', '0')))
        self.sat_badness_ddpr_thresh = float(os.environ.get('SAT_BADNESS_DDPR_THRESH', '1.0'))
        self.sat_badness_sigma_scale_cp = float(os.environ.get('SAT_BADNESS_SIGMA_SCALE_CP', '1.5'))
        self.sat_badness_ewma_alpha = float(os.environ.get('SAT_BADNESS_EWMA_ALPHA', '0.3'))
        self.sat_badness_decay = float(os.environ.get('SAT_BADNESS_DECAY', '0.9'))
        self._sat_badness_ewma = {}

        # WP13k priority 3: cp_hold FSM, reduced scope -- tc/'s `state.py`
        # `trigger_cp_hold`/`effective_cp_hold_epochs` (global DDCP-disable
        # for N epochs after a trigger) + `runtime_state.py`'s
        # `RecoveryState.recov_cp_hold` countdown. This port's DD-PR-
        # always change (WP13i) already gives every epoch an absolute,
        # ambiguity-independent position anchor even with DD-CP fully
        # suppressed -- exactly the bounded fallback tc/'s cp_hold relies
        # on -- so "hold DDCP" here means "skip adding DD-CP factors
        # entirely for this pair this epoch" (PR-only), not a Kalman-style
        # freeze. Trigger: this port's own already-tracked `_main_ddpr_res`
        # (main-graph DD residual RMS, 1-epoch-lag same as sat_badness)
        # exceeding `cp_hold_trigger_thresh` -- a reduced-scope stand-in
        # for tc/'s richer trigger set (slip-burst count, pose-innovation,
        # FDE-safeguard) this port doesn't track in comparable form.
        # Disclosed simplification, not fabricated: cp_hold_dirty_reset /
        # cooldown / probation / per-sat suspect-streak machinery (tc/'s
        # `SatQualityState` hold_quarantine/release_probation/
        # dirty_cooldown) is NOT transcribed here -- only the core
        # trigger-and-hold-N-epochs primitive.
        self.cp_hold_enable = bool(int(os.environ.get('CP_HOLD_ENABLE', '0')))
        self.cp_hold_trigger_thresh = float(os.environ.get('CP_HOLD_TRIGGER_THRESH', '3.0'))
        self.cp_hold_epochs = int(os.environ.get('CP_HOLD_EPOCHS', '5'))
        self._cp_hold_remaining = 0
        self.cp_hold_trigger_count = 0
        self.cp_hold_active_epochs = 0

        # WP13l (TASK_M12.md): tc/'s ACTUAL downstream purity stack that
        # replaces the ratio test for PAR-sourced fixes, read directly out
        # of `tc/src/gnss_fgo/optimize/ar.py` (`_run_lambda_attempts`,
        # `_validate_fix`/`_ar_context_reject`) and
        # `validation/postprocess.py` (`_decide_fix_or_flt`) -- NOT
        # assumed. Root cause, confirmed by reading the source: tc/ NEVER
        # ratio-gates `resamb_lambda`'s own `nfix>0` result --
        # `_run_lambda_attempts` only checks `nb<=0` and (for
        # non-`rtklib_mode` paths ONLY -- tc/'s own default is
        # `rtklib_mode=1`, so this never even applies to tc/'s real run)
        # `nb < ar_min_nb`. cssrlib's OWN `mlambda.parsearch` (parmode=2/
        # PAR) already enforces `Ps>=par_P0` INTERNALLY before it ever
        # returns `nfix>0` (`.venv/src/cssrlib/src/cssrlib/mlambda.py`:
        # `while Ps<P0: k+=1; Ps=sr_boost(d[k:])`; `if k<=exclmax and
        # Ps>P0: ...nfix=n-k else: nfix=0`) -- so "gate on par_P0" and
        # "trust `nb>0` from a parmode=2 call" are the SAME thing. WP13k's
        # `ratio=1.0` sweep (WP13K_REPORT.md Part 2) already measured
        # exactly that in isolation (fix%=90.8, false-fix=79.6%, median
        # err 307cm) -- catastrophic, because trusting `nb>0` ALONE is not
        # what makes tc/ pure; tc/'s real purity comes from FOUR
        # downstream checks this port never transcribed before this task:
        # `_ar_context_reject` (burst-context reject, ar.py) and THREE
        # checks in `_decide_fix_or_flt` gating the FIX/FLT *label* itself,
        # independent of any ratio: `lambda_corr_hard_max` (reject if the
        # LAMBDA-fixed antenna position `xa[0:3]` jumps more than 1.0m from
        # THIS epoch's own float antenna position `nav.x[0:3]` -- an
        # ambiguity-blind sanity check catching exactly the "wrong integer,
        # plausible nb" case a ratio test never could) and two "cold-start"
        # gates (`low_nb_fix_reject_nb_max`=6, `weak_fix_nb_max`=2) that
        # refuse a LOW-nb fix as the FIRST fix right after a float epoch
        # (tc/'s own `prev_smode==5` check). Preset
        # `tokyo_mode2_satbad_cponly` sets the two `*_max_prev_fix_streak`
        # override knobs to 0, which (per postprocess.py's own
        # `if int(...)>0:` guard) disables their "OR extend past the first
        # post-float epoch" leniency clause entirely -- i.e. tc/'s own
        # tuned preset uses the STRICT prev_smode==5-only form transcribed
        # here. All thresholds below are tc/'s own `config.py` DEFAULTS
        # (this preset does not override any of them). `pair_bad_max`
        # (tc/'s per-pair EWMA, `preprocess/sat_quality.py`) is NOT
        # transcribed here (WP13k already disclosed `alpha_recent_pair=0`
        # in this port's reduced sat_badness scope) so its own
        # context-reject term is fixed at 0.0 (never fires) below --
        # disclosed, not fabricated.
        self.lambda_corr_max = float(os.environ.get('LAMBDA_CORR_MAX', '0.0'))
        self.lambda_corr_hard_max = float(os.environ.get('LAMBDA_CORR_HARD_MAX', '1.0'))
        self.low_nb_fix_reject_nb_max = int(os.environ.get('LOW_NB_FIX_REJECT_NB_MAX', '6'))
        self.weak_fix_nb_max = int(os.environ.get('WEAK_FIX_NB_MAX', '2'))
        self.weak_fix_lambda_corr_max = float(os.environ.get('WEAK_FIX_LAMBDA_CORR_MAX', '0.08'))
        self.weak_fix_main_ddpr_res_max = float(os.environ.get('WEAK_FIX_MAIN_DDPR_RES_MAX', '0.8'))
        self.ar_context_main_ddpr_max = float(os.environ.get('AR_CONTEXT_MAIN_DDPR_MAX', '1.2'))
        self.ar_context_worst_sat_max = float(os.environ.get('AR_CONTEXT_WORST_SAT_MAX', '4.0'))
        self.ar_context_nb_max = int(os.environ.get('AR_CONTEXT_NB_MAX', '6'))
        self.ar_context_reject_during_cp_hold = bool(int(
            os.environ.get('AR_CONTEXT_REJECT_DURING_CP_HOLD', '1')))
        self.ar_context_reject_during_ddpr_bad = bool(int(
            os.environ.get('AR_CONTEXT_REJECT_DURING_DDPR_BAD', '1')))
        # tc/'s Layer-3 `prev_smode = tc.nav.smode` is captured BEFORE
        # Layer-5 resets `nav.smode=5` for the new epoch (`optimize/
        # stage.py` `run()` calls `_build_factor_block` before
        # `_run_lambda_ar`). This port's `process()` already resets
        # `self.nav.smode = 5` (via `_write_back`) BEFORE `_do_ar` runs
        # each epoch (see `process()`), so by the time `_do_ar` starts,
        # last epoch's report state is already gone from `nav.smode` --
        # tracked separately here instead.
        self._last_report_was_fix = False
        self.par_gate_reject_count = 0
        self.par_gate_reject_by_reason = {}
        # Disclosed, non-tc/ addition -- see the `ratio_ok` computation in
        # `_do_ar` for why. Default 0 = pure tc/-literal trust-PAR-fully
        # behavior.
        self.par_ratio_min = float(os.environ.get('PAR_RATIO_MIN', '0'))

        # WP13n (TASK_M14.md) piece 1: cycle-slip / outage-driven GTSAM
        # ambiguity resets -- tc/'s `preprocess/slip_detect.py`
        # `detect_slips_and_manage_amb`, the piece of tc/'s ambiguity
        # LIFECYCLE this port never had. The factor-by-factor diff
        # (WP13N_REPORT.md Part 1) found that with the WP13l shipped
        # config (FDE off, RELEASE_K=0) this port's GTSAM N keys are
        # NEVER reset for the whole run: cssrlib's own `update_
        # ambiguities` (our `_manage_ambiguities`) consumes `nav.slip`
        # (LLI+GF flags qcedit set, rover+base merged) but only zeroes
        # `nav.x` -- which `_write_back_tc` overwrites from GTSAM every
        # epoch anyway -- so a slipped carrier keeps feeding DD-CP
        # factors into the SAME N variable with a jumped integer, and
        # the smoother splits the difference: float means diffuse, s1/s0
        # ~1 (exactly WP13m's measured signature). tc/ instead bumps
        # `amb_gen` (-> brand-new N variable) on: LLI, GF jump
        # (thres_slip 0.15m), CMC jump (cmc_thresh 3.0m), MW jump
        # (mw_thresh 1.0 cyc, time-averaged mean, preset value), and
        # outage > maxout (5) epochs. Transcribed here onto the existing
        # `_release_ambiguity` mechanism (same key-abandon semantics).
        # LLI+GF come for free from `nav.slip` (read BEFORE
        # `_manage_ambiguities` clears it); MW and CMC are transcribed
        # from tc/'s `_detslp_mw`/CMC-jump blocks (rover-only MW, exact
        # formula). All opt-in (0 = WP13l shipped config bit-identical).
        self.slip_reset_enable = int(os.environ.get('SLIP_RESET_ENABLE', '0'))
        self.slip_reset_maxout = int(os.environ.get('SLIP_RESET_MAXOUT', '5'))
        self.slip_reset_mw_thresh = float(os.environ.get('SLIP_RESET_MW_THRESH', '0'))
        self.slip_reset_cmc_thresh = float(os.environ.get('SLIP_RESET_CMC_THRESH', '0'))
        self._amb_last_seen = {}   # (sat,f) -> last epoch idx the sat was in `sat`
        self._mw_state = {}        # (sat,f) -> [sum, n, mean]  (tc/'s _mw_state)
        self._cmc_state = {}       # (sat,f) -> last CMC value [m]
        self.slip_reset_count = 0    # LLI/GF (nav.slip) triggered releases
        self.mw_reset_count = 0
        self.cmc_reset_count = 0
        self.outage_reset_count = 0

        # WP13n piece 2: tc/'s `write_marginals` AR candidate hygiene --
        # (a) `ar_wait_new` (tc/ config default 3, used by the preset):
        # a NEWLY-seeded ambiguity is excluded from the LAMBDA candidate
        # set (`vsat=0`) until it has aged `ar_wait_new` epochs in the
        # graph ("Exclude new ambiguities from AR until converged",
        # optimize/ar.py line ~249). This port put every CP-visible
        # ambiguity into LAMBDA the epoch it was born, with its sigma=30
        # prior barely constrained -- flattening s1/s0 for the whole
        # set. Committed (held) keys are exempt, mirroring tc/'s
        # held-state vsat=1 path. (b) `per_sat_gate` (preset
        # per_sat_gate_enable=1, per_sat_res_thresh=3.0m,
        # preprocess/prefit.py `apply_per_sat_residual_gate`): zero vsat
        # for sats whose previous-epoch main-graph DD-PR residual
        # exceeded the threshold, keeping multipath-biased sats out of
        # LAMBDA. Both opt-in (0/0 = shipped config bit-identical).
        self.ar_wait_new = int(os.environ.get('AR_WAIT_NEW', '0'))
        self.per_sat_gate_enable = int(os.environ.get('PER_SAT_GATE_ENABLE', '0'))
        self.per_sat_res_thresh = float(os.environ.get('PER_SAT_RES_THRESH', '3.0'))
        self._amb_init_epoch = {}  # (sat,f) -> epoch idx the key was seeded
        self.per_sat_gate_drop_count = 0

        # WP13o (TASK_M15.md): accept-stack recalibration to tc/'s
        # operating point + PR-only context signals. All opt-in (unset =
        # WP13n B config bit-identical).
        # (a) THRESAR: cssrlib's `nav.thresar` (the ratio bar INSIDE
        #     `resamb_lambda`/`_rtklib`, i.e. the gate `raw_nb>0` means)
        #     has been the cssrlib default 2.0 for the whole campaign;
        #     tc/ sets 3.0 (`config.py ar_thresar=3.0` via
        #     `_apply_nav_config`). Collapsing to tc/'s SINGLE ratio bar
        #     = THRESAR=3.0 + --hold-ratio 0 (tc/ has no second bar on
        #     top of its internal one).
        # (b) AR_CONTEXT_PR_ONLY / PER_SAT_GATE_PR_ONLY: tc/'s
        #     `_ar_context_reject` and per-sat residual gate consume
        #     PR-ONLY residuals (`validation/postfit.py
        #     main_ddpr_residuals` evaluates only 'Pseudorange'-named
        #     factors). This port's `_compute_main_dd_res` folds DD-CP
        #     residuals in (WP13g adaptation, which predates WP13i's
        #     DD-PR-always change) -- so a wrong/slipped float N inflates
        #     the very context signal that decides whether its own
        #     CORRECTING fix is accepted. WP13o mining
        #     (`wp13o_mine_rejections.py` on the WP13n B full runs)
        #     measured `ar_context_reject` as THE dominant
        #     pass-but-no-commit rejector (1525/662/2479 rejected epochs
        #     on runs 1/2/3, vs 62/126/175 accepted fixes). With
        #     DD-PR-always the tc/-literal PR-only signal now exists on
        #     this port too; these flags re-point the two gates at it
        #     (computed in the same `_compute_main_dd_res` pass).
        _thresar_env = os.environ.get('THRESAR', '')
        if _thresar_env:
            self.nav.thresar = float(_thresar_env)
        self.ar_context_pr_only = int(os.environ.get('AR_CONTEXT_PR_ONLY', '0'))
        self.per_sat_gate_pr_only = int(os.environ.get('PER_SAT_GATE_PR_ONLY', '0'))
        self._main_ddpr_res_pr = 0.0
        self._main_ddpr_per_sat_pr = {}

        # WP13p (TASK_M16.md): churn reconciliation WITHOUT the O13 mask
        # purity break. All opt-in (0 = WP13o O11 config bit-identical).
        # (a) Two-tier AR membership: O13 measured that tc/'s preset obs
        #     masks (elmin 25deg / cnr 30dBHz vs this port's 15/25) cut
        #     the ambiguity-reset churn into tc/'s range and halve the
        #     float, but killing the marginal sats EVERYWHERE (tc/'s
        #     qcedit-level mask) also removes their geometry/PR
        #     cross-check and breaks purity (24.4% false, the WP13m mask
        #     trap). Two-tier keeps marginal (low-el / low-CNR) sats in
        #     the FLOAT graph but out of the AR machinery:
        #       AR_TIER_MODE=1: zero `nav.vsat` for marginal (sat,f)
        #         right before the LAMBDA attempt chain (same mechanism
        #         as the WP13n per-sat residual gate) -- excluded from
        #         the LAMBDA candidate pool AND (since ddidx's fix==2
        #         candidates require vsat==1) from N-hold eligibility;
        #         their DD-PR *and* DD-CP factors still stabilize the
        #         float.
        #       AR_TIER_MODE=2: additionally never build a DD-CP factor
        #         for a pair touching a marginal sat (PR-only, exactly
        #         the GLO_CP_EXCLUDE mechanism) -- the marginal sat then
        #         never owns an N variable at all, so it cannot churn
        #         the reset machinery either (slip/MW/outage resets fire
        #         per tracked N key).
        #     AR_TIER_HELD_EXEMPT=1 (default) keeps an already-committed
        #     hold eligible even if its sat dips marginal later --
        #     mirrors tc/'s held-state vsat=1 path and ar_wait_new's own
        #     committed-key exemption. Phase-2 scoped (the O3 probe
        #     measured accept-stack changes damaging ONLY the phase-1
        #     stationary bootstrap).
        self.ar_tier_mode = int(os.environ.get('AR_TIER_MODE', '0'))
        self.ar_tier_elmin_deg = float(os.environ.get('AR_TIER_ELMIN_DEG', '25.0'))
        self.ar_tier_cnr_min = float(os.environ.get('AR_TIER_CNR_MIN', '30.0'))
        self.ar_tier_held_exempt = int(os.environ.get('AR_TIER_HELD_EXEMPT', '1'))
        self.ar_tier_drop_count = 0       # mode-1 vsat zeroings (sat,f)-epochs
        self.ar_tier_cp_excl_count = 0    # mode-2 DD-CP factor suppressions
        # (b) Small-nb purity backstop: WP13o's honest purity cost
        #     (false-fix 10-25%, med fix err 12-26cm) lives in the
        #     small-nb partial fixes (the nb>=9 rich-search subset is
        #     0.9%/5.0cm on run2); the O8 anatomy measured PAR-cascade
        #     fixes at nb 5-8 & ratio>=10 -> 5.7% false vs the single
        #     ratio>=3 bar's mixed population. SMALL_NB_MAX_P2>0 demands
        #     a HIGHER ratio (SMALL_NB_RATIO_P2) whenever this epoch's
        #     nb <= SMALL_NB_MAX_P2, leaving rich searches on the normal
        #     HOLD_RATIO_P2 bar. Phase-2 scoped, applied inside _do_ar.
        self.small_nb_max_p2 = int(os.environ.get('SMALL_NB_MAX_P2', '0'))
        self.small_nb_ratio_p2 = float(os.environ.get('SMALL_NB_RATIO_P2', '10.0'))
        self.small_nb_bar_reject_count = 0
        # (c) Churn-source triage instrumentation (WP13P_DBG_CHURN=1,
        #     print-only): every `_release_ambiguity` call logs its
        #     trigger + the sat's this-epoch elevation/CNR (from the
        #     context map refreshed each phase-2 epoch) + whether the
        #     released key was a committed hold.
        self._churn_dbg_ctx = None
        # (d) Held-pool per-epoch diagnostics for the npz (report-only):
        #     this epoch's LAMBDA candidate count and how many of them
        #     were already-committed holds (tc/ runs at ~90% held).
        self._last_ar_ncand = 0
        self._last_ar_ncommit = 0
        # (e) Slip-flag hysteresis for marginal sats (churn triage
        #     follow-up, opt-in): SLIP_MARGINAL_CONFIRM_N>0 requires a
        #     marginal (below-tier) sat's LLI/GF slip flag to persist N
        #     consecutive flagged epochs before its N key is released --
        #     a flapping slip bit on a low-el/low-CNR sat otherwise
        #     releases (and re-seeds at sigma-30) the same key every few
        #     epochs. Committed holds and above-tier sats keep the
        #     immediate (tc/-literal) release.
        self.slip_marginal_confirm_n = int(os.environ.get('SLIP_MARGINAL_CONFIRM_N', '0'))
        self._slip_flag_streak = {}
        # (f) FDE_RELEASE_REF=1: tc/-literal FDE-CP ambiguity release
        #     (BOTH satellites of a rejected DD-CP factor, via the
        #     reference's generic fac.keys() loop) instead of WP13d's
        #     j-sat-only adaptation. The WP13p churn triage measured the
        #     FDE-CP path as THE dominant churn source (23k releases /
        #     3000ep vs 748 slip/MW/outage on the O11 run2 probe, median
        #     key age 0, 84% from ABOVE-tier sats): a poisoned shared
        #     REFERENCE N is never released under the adaptation, so
        #     every pair against it keeps failing the 0.5m CP gate and
        #     releases its (good) j-sat forever -- the loop that starves
        #     the LAMBDA pool below ar_wait_new age. WP13d measured the
        #     literal both-sat release wedging the smoother, but that
        #     predates WP13n piece 7 (solve-exception warm reset), which
        #     is exactly the recovery tc/ itself relies on. Default 0 =
        #     WP13o O11 bit-identical.
        self.fde_release_ref = int(os.environ.get('FDE_RELEASE_REF', '0'))

        # WP13q (TASK_M17.md): tc/-literal HELD-release re-seed pin.
        # The WP13q between-burst triage (wp13q_burst_diag.py on the
        # WP13p T3 full runs) measured the held pool decaying through
        # fix gaps mostly via fde_cp/fde_cp_ref releases of HEALTHY,
        # mature keys (run3: 500 of 1,115 held releases, med key age
        # 125-145 ep) -- each one, under this port's `_release_
        # ambiguity`, is FULL amnesia: fresh sigma-30 seed + ar_wait_new
        # exclusion + reconvergence + HOLD_CONFIRM_N before the key can
        # hold again. tc/ treats exactly this path differently
        # (`validation/postfit.py _fde_reset_rejected_amb`: a rejected
        # DD-CP factor whose (sat,f) is HELD calls `release_hold(seed=
        # True)`, and `buildfactor/factors.py _seed_one_amb_prior` then
        # seeds the NEXT generation of that N at `last_held_value` with
        # a fixed sigma-0.1-cycle prior -- their literal `_noise1(0.1)`;
        # `validation/postprocess.py _release_suspicious_held_on_flt`
        # (their release_k analog) uses the same path). Non-held FDE
        # releases in tc/ get amnesia like ours (`amb_key=None` drops
        # the key from `_carry_prev_amb`'s continuity dict), and true
        # slips/outages call `clear_hold()` (pin forgotten) -- both
        # verified in the tc/ source, transcribed exactly:
        #   RELEASE_SEED_HELD=1: on a soft-release (`fde_cp`/
        #   `fde_cp_ref`/`release_k`) of a COMMITTED key, remember its
        #   committed value; the next `_build_dd_factors_arm` fresh
        #   seed of that (sat,f) starts at that value with sigma =
        #   RELEASE_SEED_SIGMA (cycles, x lambda -- tc/'s N lives in
        #   cycles, this port's in meters) instead of sigma_amb0=30.
        #   One-shot (popped on use); forgotten on any hard release
        #   (slip/mw/cmc/outage/sanity_wipe), on a Phase-2 re-seed, and
        #   when unconsumed for > slip_reset_maxout epochs (tc/'s
        #   SatState pending is likewise cleared by its outage
        #   `clear_hold` after maxout unseen epochs).
        # Default 0 = WP13p T3 bit-identical.
        self.release_seed_held = int(os.environ.get('RELEASE_SEED_HELD', '0'))
        self.release_seed_sigma = float(os.environ.get('RELEASE_SEED_SIGMA', '0.1'))
        # RELEASE_SEED_REF_ONLY=1 narrows the pin to the COLLATERAL
        # release of the shared reference (`fde_cp_ref` + `release_k`),
        # leaving the directly-suspected j-sat (`fde_cp`) on the amnesia
        # path. Disclosed deviation from tc/'s both-sat pin: on this
        # port the j-sat of a rejected DD-CP factor is the FDE's actual
        # suspect (its value just failed the 0.5 m gate), while the
        # reference is released only because it shares the factor -- the
        # capped run3 probe measured the both-sat pin dragging burst
        # ratios (s1/s0 med 3.37->2.93, fix% 35->30) by re-imposing
        # suspect j-sat values into the LAMBDA pool.
        self.release_seed_ref_only = int(os.environ.get('RELEASE_SEED_REF_ONLY', '0'))
        # RELEASE_SEED_MAX_POOL>0 gates the pin on the committed-pool
        # size AT RELEASE TIME (the released key still counts): pin
        # only when the pool is already small (the decayed/re-entry
        # regime the mechanism was built for), keep T3's amnesia when a
        # strong hold equilibrium is running. Disclosed, evidence-driven
        # condition (not tc/-literal): the full-scale A/B measured the
        # unconditional pin regressing run3's strong bursts (fix 86->72
        # / 12->2 / 61->47 per-1000ep blocks; <50cm_full 61.8->60.7)
        # while producing its gains in the churny low-pool regimes of
        # run1/run2. 0 = unconditional (tc/-literal).
        self.release_seed_max_pool = int(os.environ.get('RELEASE_SEED_MAX_POOL', '0'))
        self._release_seed_pending = {}  # (sat,f) -> (value, ep2_stored)
        self.release_seed_stored = 0
        self.release_seed_used = 0
        self.release_seed_expired = 0

        # WP13r (TASK_M18.md) lever 1: conditioned-out holds -- tc/'s
        # `write_marginals` held-state branch (optimize/ar.py lines
        # ~276-285), the structural piece WP13q's Part-1 diagnosis
        # identified behind run3's raw_nb==0 mega-gaps. In tc/, a HELD
        # ambiguity has left the graph entirely (`activate_hold` sets
        # `amb_key=None`; its DD-CP factors carry the integer as a
        # CONSTANT) and its LAMBDA mirror is written as
        #   nav.x[IB] = held_value, nav.P[IB,IB] = varholdamb (0.001),
        #   all cross-covariances 0, vsat = 1 iff CP-visible this epoch
        # -- i.e. the held integer stays IN ddidx's candidate set but
        # enters mlambda with near-zero variance: the ILS over held
        # components is trivial (any candidate flipping a held integer
        # costs ~1/varholdamb), so s1/s0 is decided by the young/unheld
        # REMAINDER alone -- increment-only AR. When the whole pool is
        # held (tc/'s steady state, 42-54 keys through our dead run3
        # gaps), s0 itself is tiny -> ratio clears -> the fix keeps
        # flowing epoch after epoch, which is exactly what this port's
        # whole-pool re-search (committed keys re-entering LAMBDA with
        # their LIVE ISAM2 joint marginals, re-searched jointly with
        # every young key) never achieves in the long gaps (raw_nb==0 on
        # 95% of gap epochs at healthy nsat, WP13Q_REPORT Part 1).
        #   COND_HOLD=1: `_write_back_tc` writes every committed
        #   (sat,f) exactly like tc/'s held branch (value = the
        #   committed integer `self._committed[key]`, the same quantity
        #   tc/ pins -- both are the LAMBDA xa value at hold time; P
        #   diag = VAR_HOLDAMB; vsat per CP-visibility) and EXCLUDES it
        #   from the ISAM2 joint-marginal query (`_ar_eligible` False --
        #   its graph variable keeps its WP13c one-time tight prior, the
        #   graph-side analogue of tc/'s conditioned-out constant).
        #   Phase-2 scoped. Default 0 = Q4/T3 bit-identical.
        self.cond_hold = int(os.environ.get('COND_HOLD', '0'))
        self.cond_hold_epochs = 0       # epochs with >=1 conditioned key
        self.cond_hold_key_epochs = 0   # sum of conditioned keys over epochs
        # COND_FIX_RES_MAX (m, 0=off): disclosed, non-tc/ report gate
        # needed BECAUSE of conditioning -- with the held pool at
        # varholdamb the joint s1/s0 no longer reflects float quality
        # (a wrong/bad-epoch candidate flipping a held integer costs
        # ~1/varholdamb, so the ratio stays huge on epochs the
        # whole-pool re-search used to veto), and xa[0:3] collapses to
        # the FLOAT position (held keys carry no pose cross-covariance,
        # same as tc/), so a transient 0.5-1 m float excursion would be
        # REPORTED as a Q=4 fix at the float's error. Measured on the
        # run1 hard window (r1_cond_hard_r1): 60 false fixes, median
        # err 0.54 m, ratio med 38 -- borderline float epochs, not
        # wrong integers. This gate vetoes the Q=4 report (same
        # par_gate mechanism as tc/'s own accept stack; commits are
        # skipped too, matching every other gate) when THIS epoch's
        # CP-folded main DD residual exceeds the cap.
        self.cond_fix_res_max = float(os.environ.get('COND_FIX_RES_MAX', '0'))
        self.cond_fix_res_reject_count = 0
        self._main_ddcp_res_med = -1.0   # CP-only median, this epoch
        self._main_ddcp_res_n = 0
        # COND_MIN_NB (0=off): report-only nb floor under COND_HOLD --
        # the WP13p small-nb lesson re-tooled for the conditioned
        # world: nb<=8 fixes measured 54.9% false / med 55 cm on the
        # full COND run3 (253 of 3,139; they drive FixRMS 0.899) and
        # 19.3% false on run2, while nb>=9 is 1.1% false / 3.2 cm --
        # and the ratio bar that used to screen small-nb searches is
        # structurally neutralized by conditioning (flipping a held
        # integer costs ~1/varholdamb, so tiny held-dominated pools
        # clear ANY ratio). Reporting those epochs float (sol == xa =~
        # float either way; only the label + the prev-smode gate
        # feedback change) is the honest call, and the same intent as
        # tc/'s own low_nb_fix_reject family.
        self.cond_min_nb = int(os.environ.get('COND_MIN_NB', '0'))
        self.cond_min_nb_reject_count = 0

        # WP13r lever 2: post-outage re-entry stack -- the tc/ recovery
        # pieces (validation/recovery.py + postfit.py + state.py) this
        # port carried only partially (WP13g transcribed the sanity
        # trigger/persist/pose-replace core; measured net-negative THEN
        # -- pre-DD-PR-always, pre-holds -- and shipped disabled via
        # --sanity-enable 0). Re-transcribed additions, all opt-in:
        #   RECOV_CP_HOLD_EPOCHS>0: tc/'s `recov_cp_hold` (config
        #   default 5; state.py `trigger_cp_hold` / recovery.py
        #   `reset_ambiguities_with_cp_hold` / `warm_reset_phase2`) --
        #   after a sanity reset or a solve-exception warm reset, and on
        #   every sanity-triggered (residual-bad) epoch (tc/'s
        #   `_ddpr_sanity_persist` fires `trigger_cp_hold` on EACH bad
        #   epoch), suppress ALL DD-CP factor construction for N epochs
        #   (PR-only float; the WP13i DD-PR-always anchor carries the
        #   pose) so freshly re-seeded ambiguities are born at a
        #   PR-pulled pose instead of the dirty CP basin.
        #   SANITY_MAX_MEDIAN_RATIO>0: tc/'s `_ddpr_multipath_dominated`
        #   (postfit.py; preset 5.0, min sats 6): skip the whole sanity
        #   ladder when max/median of the per-sat DD-PR residuals says
        #   ONE satellite dominates (canyon multipath, not a wrong
        #   pose) -- protects fix streaks from spurious resets.
        # Disclosed omissions (scope, mirrors WP13g): tc/'s DDPR-only-LS
        # anchor stages are diagnostics/label-only in its sanity path
        # (`_ddpr_sanity_apply_reset` ignores the anchor value; both
        # anchor-fail paths run the SAME `_apply_sanity_reset`), so no
        # `_ddpr_only_position` solver is ported; `sanity_break_pim` has
        # no analogue in this port's single-chain PIM.
        self.recov_cp_hold_epochs = int(os.environ.get('RECOV_CP_HOLD_EPOCHS', '0'))
        # tc/'s residual-conditioned release (preprocess/gate.py lines
        # ~113-128; preset recov_cp_release_thresh=2.0m, count=5): the
        # window only truly expires once the PR-only main DD residual
        # has been clean (<= thresh) for `count` consecutive held
        # epochs -- otherwise it re-arms itself 1 epoch at a time.
        self.recov_cp_release_thresh = float(os.environ.get('RECOV_CP_RELEASE_THRESH', '0'))
        self.recov_cp_release_count = int(os.environ.get('RECOV_CP_RELEASE_COUNT', '5'))
        self.sanity_max_median_ratio = float(os.environ.get('SANITY_MAX_MEDIAN_RATIO', '0'))
        self.sanity_max_median_min_sats = int(os.environ.get('SANITY_MAX_MEDIAN_MIN_SATS', '6'))
        # SANITY_PR_ONLY=1: tc/-literal PR-only sanity signals (see
        # `_ddpr_sanity_check`) -- the WP13g CP-folded adaptation's
        # justification predates WP13i's DD-PR-always. Default 0 =
        # WP13g behavior.
        self.sanity_pr_only = int(os.environ.get('SANITY_PR_ONLY', '0'))
        self._recov_cp_hold_remaining = 0
        self._recov_cp_release_streak = 0
        self._recov_cp_skip_now = False  # this epoch's pre-build snapshot (tc/'s ed.skip_cp_now)
        self.recov_cp_hold_trigger_count = 0
        self.recov_cp_hold_cp_suppressed_epochs = 0
        self.sanity_skip_multipath_count = 0

        # WP13r lever 1 companion: tc/'s `ar_persist_bad` per-sat
        # quarantine (optimize/stage.py lines ~228-254 + sat_quality.py
        # `tick`/`persist_bad_hold`, config DEFAULTS ON in tc/: enable=1,
        # res_thresh=2.0 m, streak=4, hold=10) -- the purity control the
        # conditioned-hold world needs: a held integer's LAMBDA ratio is
        # inflated by its own conditioning (any competitor flipping it
        # costs ~1/varholdamb), so ratio-based gates can no longer
        # reject a wrong hold; tc/ instead watches the PR-only per-sat
        # residual and, after `streak` consecutive epochs above
        # `res_thresh`, quarantines the SATELLITE for `hold` epochs:
        #   * excluded from the LAMBDA candidate set (write_marginals
        #     `held_bad` -> vsat=0 on BOTH the float and held branches),
        #   * its DD-CP factors are not built at all while quarantined
        #     (forced_hold -> compute_cp_build_policy with tc/'s
        #     cp_hold_sigma_penalty default 0.0 = CP disabled),
        #   * its FLOAT N is regenerated (amb_gen bump; tc/ does NOT
        #     clear_hold -- a held integer survives quarantine and
        #     returns to the pool when the hold expires).
        # First measured need: the r1_cond_hard_r1 probe (COND_HOLD
        # alone) broke the run1 hard-window purity gate (p2 false 0.2%
        # -> 8.6%, canyon 3 false) exactly via persistent wrong holds.
        self.ar_persist_bad_enable = int(os.environ.get('AR_PERSIST_BAD_ENABLE', '0'))
        self.ar_persist_bad_res_thresh = float(os.environ.get('AR_PERSIST_BAD_RES_THRESH', '2.0'))
        self.ar_persist_bad_streak = int(os.environ.get('AR_PERSIST_BAD_STREAK', '4'))
        self.ar_persist_bad_hold = int(os.environ.get('AR_PERSIST_BAD_HOLD', '10'))
        self._persist_bad_streak = {}   # sat -> consecutive bad epochs
        self._persist_bad_hold = {}     # sat -> remaining quarantine epochs
        self.persist_bad_hold_count = 0    # quarantine activations
        self.persist_bad_release_count = 0  # float-N regenerations

        # WP13n piece 3: tc/'s elevation-dependent measurement variance
        # (`preprocess/prefit.py` `varerr_dd_sigma`, RTKLIB-demo5
        # rtkpos.c:402 port; enabled by tc/'s config default
        # varerr_enable=1 with err_a=err_b=0.001, err_eratio_pr=100,
        # err_sclkstab=5e-12, el floor el_mask_deg=10):
        #   var_SD = 2*(a^2 + b^2/sin^2 el) + (c*sclkstab*dt)^2,
        #   sigma_DD = sqrt(2*var_SD); a=b=fact*err, fact=eratio (PR) / 1 (CP)
        # replacing this port's flat sigma_pr*sqrt(2) / sigma_cp*sqrt(2).
        # Relative PR-vs-CP weighting vs elevation shapes how much the
        # float N absorbs low-elevation code multipath (TASK_M14 target
        # (c)). Opt-in (0 = shipped config bit-identical).
        self.varerr_enable = int(os.environ.get('VARERR_ENABLE', '0'))
        self.varerr_err_a = float(os.environ.get('VARERR_ERR_A', '0.001'))
        self.varerr_err_b = float(os.environ.get('VARERR_ERR_B', '0.001'))
        self.varerr_eratio_pr = float(os.environ.get('VARERR_ERATIO_PR', '100.0'))
        self.varerr_sclkstab = float(os.environ.get('VARERR_SCLKSTAB', '5e-12'))
        self.varerr_el_mask_deg = float(os.environ.get('VARERR_EL_MASK_DEG', '10.0'))
        self._epoch_dt = 0.2  # tc/'s runner default; refreshed per epoch

        # WP13n piece 5: GLONASS DD-CP exclusion -- tc/'s
        # `_build_cp_for_pair` (buildfactor/factors.py line ~453) starts
        # with `if sys_id == uGNSS.GLO: return 0`: GLO gets DD-PR
        # factors but NEVER a DD-CP factor, so a GLO (sat,f) never
        # enters `amb_dict` -> never gets vsat=1 in write_marginals ->
        # never reaches LAMBDA. Reason: GLONASS FDMA inter-frequency
        # (channel-dependent) hardware biases make the DD ambiguity
        # NON-INTEGER across heterogeneous rover/base receivers, so
        # feeding GLO N to an integer search poisons the candidate set.
        # This port (transcribed from tc/'s OLD GNSS-only utils/rtk.py,
        # which predates the exclusion) has been building GLO DD-CP and
        # handing GLO ambiguities to LAMBDA on EVERY epoch since WP13a
        # -- a per-epoch, structural s1/s0 flattener. `ddidx` (shared
        # cssrlib) has no GLO filter of its own: what keeps GLO out of
        # tc/'s LAMBDA is exactly this build-time exclusion. Opt-in
        # (0 = shipped config bit-identical).
        self.glo_cp_exclude = int(os.environ.get('GLO_CP_EXCLUDE', '0'))

        # WP13n piece 6: BDS GEO exclusion -- tc/ preset
        # `tokyo_mode2_satbad_cponly` sets `exclude_bds_geo=1`
        # (DdFactorBuilder.run(): GEO j-sats skipped for BOTH PR and CP;
        # `pick_ref_sat_idx` refuses GEO refs). BeiDou GEO slots (PRN
        # <=5 / 59-63) sit at fixed azimuth with chronic urban
        # multipath. Opt-in (0 = shipped config bit-identical).
        self.exclude_bds_geo = int(os.environ.get('EXCLUDE_BDS_GEO', '0'))

        self._streak = {}      # (sat,freq) -> (last_resolved_value_m, consecutive_count)
        self._committed = {}   # (sat,freq) -> value_m at commit time (a VALIDATED hold)
        self._amb_gen = {}     # (sat,freq) -> generation counter, bumped on release so
                                #   self.N() mints a fresh, never-reused gtsam key instead
                                #   of re-touching a poisoned one still live in the graph
                                #   (a released hold's OLD prior can't be un-injected --
                                #   see _check_release / WP13C_REPORT.md)
        self._bad_streak_by_key = {}  # (sat,freq) -> consecutive bad-residual count (for release)

        # WP13d (TASK_M4.md): GICI-style FDE, transcribed from inuex35's
        # `validation/postfit.py` (`apply_fde` + helpers). Config mirrors
        # their TcConfig defaults exactly (fde_pr=4.0, fde_cp=0.5,
        # fde_max_frac=0.5, fde_max_iter=1 i.e. single-pass).
        self.fde_enable = int(os.environ.get('FDE_ENABLE', '1'))
        self.fde_pr = float(os.environ.get('FDE_PR', '4.0'))
        self.fde_cp = float(os.environ.get('FDE_CP', '0.5'))
        self.fde_max_frac = float(os.environ.get('FDE_MAX_FRAC', '0.5'))
        self.fde_max_iter = int(os.environ.get('FDE_MAX_ITER', '1'))
        self.fde_median_sub = int(os.environ.get('FDE_MEDIAN_SUB', '0'))
        # Disclosed experimental knob (WP13D_REPORT.md): whether a rejected
        # DD-CP factor is ALSO structurally removed via removeFactorIndices
        # (1, matches the literal reference) or left in the graph and
        # handled purely via the ambiguity cycle-slip reset (0, mirrors
        # WP13c's own release mechanism, which never physically removes a
        # factor either -- it just lets an abandoned key's timestamp stop
        # being refreshed so it ages out of the lag window passively).
        self.fde_remove_cp = int(os.environ.get('FDE_REMOVE_CP', '1'))
        # Per-epoch DD factor tag tracking (mirrors their `_last_ddpr_sat_tags`
        # / `_last_custom_ddcp_global`): populated by `_build_dd_factors`
        # (LOCAL index within this epoch's `graph`) then globalized by
        # `_apply_fde` (see its docstring for why via `nf_before`, not the
        # reference's own `nf_total - g3.size()` offset trick).
        self._epoch_ddpr_tags_local = []
        self._epoch_ddcp_tags_local = []
        self._last_ddpr_sat_tags = []   # [(global_fi, ref_sat, j_sat, freq), ...]
        self._last_ddcp_meta = {}       # global_fi -> (ref_sat, j_sat, freq)
        # FDE diagnostics (for the on/off ablation report).
        self.fde_total_rejected = 0
        self.fde_epochs_rejected = 0
        self.fde_epochs_safeguard_skipped = 0
        self.last_fde_reject = 0
        self.last_main_ddpr_rms = 0.0
        self.last_main_ddpr_worst = None  # (sat, res_m) or None
        self.main_ddpr_rms_sum = 0.0
        self.main_ddpr_rms_n = 0

        # WP13j (TASK_M10.md): PLATEAU ray-traced NLOS prior. See the
        # module-level comment above `_load_nlos_mask` for the 3-layer
        # design. NLOS_ENABLE gates loading the mask at all (0 cost when
        # off); NLOS_RUN selects which run's mask file (falls back to
        # parsing it from the process's own argv `run` positional if
        # unset, since the driver always sets NLOS_RUN explicitly).
        self.nlos_enable = int(os.environ.get('NLOS_ENABLE', '0'))
        self.nlos_ar_exclude = int(os.environ.get('NLOS_AR_EXCLUDE', '0'))
        self.nlos_commit_reject = int(os.environ.get('NLOS_COMMIT_REJECT', '0'))
        # 0=off, 1=hard-exclude the DD-PR/DD-CP factor, 2=inflate its sigma
        # by NLOS_MEAS_SIGMA_MULT.
        self.nlos_meas_mode = int(os.environ.get('NLOS_MEAS_MODE', '0'))
        self.nlos_meas_sigma_mult = float(os.environ.get('NLOS_MEAS_SIGMA_MULT', '5.0'))
        # Fraction of this epoch's ACTIVE candidates that must be
        # NLOS-flagged before the whole-epoch Q=4 report is refused (layer
        # (c), second half). Default 0.5 = strict majority; 0.0 = refuse
        # on ANY NLOS-flagged active candidate (much more aggressive).
        self.nlos_report_frac = float(os.environ.get('NLOS_REPORT_FRAC', '0.5'))
        self._nlos_by_tow = {}
        self._nlos_mask_n_rows = 0
        self._nlos_mask_n_nlos = 0
        if self.nlos_enable:
            run_n = int(os.environ.get('NLOS_RUN', '0'))
            if run_n in (1, 2, 3):
                (self._nlos_by_tow, self._nlos_mask_n_rows,
                 self._nlos_mask_n_nlos, _bad) = _load_nlos_mask(run_n)
            else:
                print(f"WARNING: NLOS_ENABLE=1 but NLOS_RUN={run_n} invalid; "
                      f"mask NOT loaded (all layers become no-ops)")
        self._nlos_cache_key = None
        self._nlos_cache_val = frozenset()
        # Alignment/ablation diagnostics.
        self.nlos_lookup_epochs = 0
        self.nlos_lookup_hit = 0
        self.nlos_lookup_miss = 0
        self.nlos_ar_excluded_epochs = 0
        self.nlos_ar_excluded_sat_epochs = 0
        self.nlos_commit_rejected = 0
        self.nlos_report_rejected = 0
        self.nlos_meas_pr_affected = 0
        self.nlos_meas_cp_affected = 0

        self.sigma_pr = float(os.environ.get('SIG_PR', '0.3'))
        self.sigma_cp = float(os.environ.get('SIG_CP', '0.003'))
        self.sigma_dyn = float(os.environ.get('SIG_DYN', '0.1'))
        self.sigma_amb0 = float(os.environ.get('SIG_AMB', '30.0'))
        self.huber_pr = float(os.environ.get('HUBER_PR', '0'))
        self.nav.armode = int(os.environ.get('AR_MODE', '3'))
        # `rtkpos.valpos` in this cssrlib fork is a documented no-op (always
        # returns True -- see module docstring / WP13A_REPORT.md section 2.3).
        # VALPOS_THRES>0 replaces it, IN THIS FILE ONLY (no cssrlib edit),
        # with an actual RTKLIB-style post-fit chi-square gate: reject (and
        # do not hold) a LAMBDA fix whose worst per-measurement residual
        # exceeds VALPOS_THRES sigma. 0 = literal no-op (old behavior).
        self.valpos_thres = float(os.environ.get('VALPOS_THRES', '0'))

        # Diagnostics: last epoch's AR covariance source, for the
        # jointMarginalCovariance-vs-temporal-proxy sanity dump.
        self.last_joint_P = None
        self.last_diag_P = None

        # Deliberate deviation from the literal reference `process()`:
        # track the last epoch symbol that actually made it into the
        # ISAM2/IFLS Bayes tree separately from `self.epoch` (a simple
        # monotonic symbol counter). The reference always wires
        # `BetweenFactorPoint3(X(epoch-1), X(epoch), ...)` and increments
        # `self.epoch` even on epochs where no factor was ever committed
        # (nv<4, or the ISAM2 update itself throws) -- the *next* real
        # update then references a phantom `X(epoch-1)` key that was
        # never inserted, which throws again, permanently wedging the
        # chain after the first such gap. Tokyo run1-3 have exactly this
        # kind of gap (e.g. the tow 188990-189070 canyon), so this isn't
        # a hypothetical: wiring the between-factor to `last_valid_epoch`
        # instead of `epoch-1` keeps the graph well-formed across gaps.
        self.last_valid_epoch = None
        self.last_valid_time = None

        # WP13g (TASK_M7.md): raw LAMBDA-resolved-ambiguity count from the
        # most recent `_do_ar` call (0 if AR wasn't reached/failed this
        # epoch), independent of the downstream accept-gate's fix/float
        # decision -- mirrors the reference's `ed.nb` (set by
        # `_tc_ar.run_ar`, consumed by `_ddpr_sanity_fast_path`'s `nb<=0`
        # check). Declared on the parent since `_do_ar` (which sets it) is
        # inherited unchanged by `GtsamRtkTc`; only Phase-2's sanity check
        # reads it.
        self._last_nb = 0

        # WP13s (TASK_M19.md, べた移植): TC_LITERAL=1 forces tc/'s OWN
        # decision path + its `tokyo_mode2_satbad_cponly` preset knobs,
        # replacing THIS PORT'S ADAPTATIONS (hold_ratio/confirm-N/
        # min-committed accept extras, hard quarantine CP-drop, ref-only
        # release pin, per-run knob sets) with tc/'s literal ones. All
        # literal behavior is gated on `self.tc_literal` -- default 0 =
        # every prior config bit-identical. NOTE: TC_LITERAL deliberately
        # OVERRIDES conflicting CLI/env knobs (the whole point is ONE
        # tc/-preset-shaped config, like tc/ itself).
        self.tc_literal = int(os.environ.get('TC_LITERAL', '0'))
        # WP13s l5 A/B: tc/'s residual scale for the ladder signals
        # (sqrt(2*fac.error)*sigma_pr*sqrt(2), i.e. whitened by the
        # actual varerr sigma then re-scaled by the nominal one).
        # JUSTIFIED DEVIATION (default 0 = raw evaluateError meters),
        # measured TWICE: tc/'s scale runs HOTTER than raw at high
        # elevation (sigma_varerr < sigma_nominal there) in THIS port's
        # residual regime -- l5 vs l4 (per-epoch stack): fix 349 -> 122;
        # l8 vs l7 (persistent stack): decision agreement 60.8% -> 45.9%,
        # fix 34.9% -> 21.9%. TC_LIT_WRES=1 restores the literal scale.
        self.tc_lit_wres = int(os.environ.get('TC_LIT_WRES', '0'))
        # tc/-literal per-(sat,freq) quality counters (buildfactor/
        # factors.py `rejc_cp_pr` / optimize/stage.py `rejc_post_ddpr`)
        # + per-sat fix streak (postprocess.py) + tc/'s own
        # SatQualityState instance (imported, not reimplemented).
        self.rejc_cp_pr = {}
        self._rejc_post_ddpr = {}
        self._fix_streak = {}
        self.ref_sats = {}
        self._sat_el_deg = {}
        self._sat_snr_dbhz = {}
        self._last_pair_rows = []
        self._last_pair_bad_max = 0.0
        self._last_main_ddpr_per_sat = {}
        self._last_slip_keys = set()
        self.dirty_reset_count = 0
        self.rejc_wipe_count = 0
        self.flt_release_count = 0
        self.cp_pr_reject_count = 0
        self.lit_ar_skip_hold_count = 0
        self.lit_commit_count = 0
        self._last_candidates = []
        self._sq = _TcSatQualityState() if _TcSatQualityState else None
        self.cfg = None
        if self.tc_literal:
            if self._sq is None:
                raise RuntimeError(
                    "TC_LITERAL=1 requires tc/src on disk (gnss_fgo import)")
            self._apply_tc_literal_preset()

    def _apply_tc_literal_preset(self):
        """WP13s: force tc/'s `tokyo_mode2_satbad_cponly` preset values
        onto this port's knobs (config.py TC_PRESETS + TcConfig defaults
        + runner._apply_nav_config, transcribed value-for-value). Called
        at the end of __init__ so it overrides every env/CLI-derived
        attribute -- ONE config, like tc/'s own preset."""
        import types
        # runner._apply_nav_config with the tokyo preset
        self.nav.elmin = np.deg2rad(25.0)      # elmin_deg
        self.nav.cnr_min = 30.0                # cnr_min_dbhz
        self.nav.thresar = 3.0                 # ar_thresar
        self.nav.arfilter = True               # ar_arfilter
        self.nav.minfixsats = 4                # ar_minfixsats
        self.nav.armode = 3                    # ar_mode (fix-and-hold)
        self.ar_rtklib_mode = True             # rtklib_mode=1
        self.subset_ar_enable = True
        self.subset_ar_max_drop = 2
        self.subset_ar_min_nb = 4
        self.subset_ar_max_candidates = 5
        self.subset_ar_use_par = 0
        self.ar_ddpr_xvalidate_thresh = 10.0
        self.ar_ddpr_xvalidate_delta = 0.0
        self.valpos_thres = 0.0                # cssrlib valpos untouched
        # This port's OWN accept-stack additions -- neutralized (the
        # literal _do_ar branch bypasses them; zeroed for coherence).
        self.hold_ratio = 0.0
        self.hold_confirm_n = 1
        self.hold_valpos_thres = 0.0
        self.min_committed_for_fix = 0
        self.release_k = 0
        # AR context / decide gates (TcConfig defaults + preset)
        self.ar_context_main_ddpr_max = 1.2
        self.ar_context_worst_sat_max = 4.0
        self.ar_context_nb_max = 6
        self.ar_context_reject_during_cp_hold = True
        self.ar_context_reject_during_ddpr_bad = True
        self.ar_context_pr_only = 1
        self.per_sat_gate_enable = 1
        self.per_sat_gate_pr_only = 1
        self.per_sat_res_thresh = 3.0
        self.ar_wait_new = 3
        self.lambda_corr_max = 0.0
        self.lambda_corr_hard_max = 1.0
        self.low_nb_fix_reject_nb_max = 6
        self.weak_fix_nb_max = 2
        self.weak_fix_lambda_corr_max = 0.08
        self.weak_fix_main_ddpr_res_max = 0.8
        # Measurement model
        self.varerr_enable = 1
        self.varerr_err_a = 0.001
        self.varerr_err_b = 0.001
        self.varerr_eratio_pr = 100.0
        self.varerr_sclkstab = 5e-12
        self.varerr_el_mask_deg = 10.0
        self.sigma_pr = 0.3
        self.sigma_cp = 0.003
        self.sigma_amb0 = 30.0
        self.huber_pr = 0.0
        self.exclude_bds_geo = 1
        self.glo_cp_exclude = 1
        self.sat_badness_enable = True         # scored via tc/'s own sq
        self.sat_badness_sigma_scale_cp = 1.5
        # Slip / outage (preprocess/slip_detect with preset mw_thresh)
        self.slip_reset_enable = 1
        self.slip_reset_maxout = 5
        self.slip_reset_mw_thresh = 1.0        # mw_thresh preset
        self.slip_reset_cmc_thresh = 3.0       # cmc_thresh default
        self.slip_marginal_confirm_n = 0
        self.ar_tier_mode = 0
        self.small_nb_max_p2 = 0
        self.par_ratio_min = 0.0
        # FDE (validation/postfit)
        self.fde_enable = 1
        self.fde_pr = 4.0
        self.fde_cp = 0.5
        self.fde_max_frac = 0.5
        self.fde_max_iter = 1
        self.fde_median_sub = 0
        self.fde_remove_cp = 1
        self.fde_release_ref = 1               # tc/ releases BOTH sats
        # Held-release re-seed (factors.py _seed_one_amb_prior: sigma
        # 0.1 CYCLES -- literal now that literal N is in cycles too)
        self.release_seed_held = 1
        self.release_seed_ref_only = 0         # tc/ pins both sats
        self.release_seed_sigma = 0.1
        self.release_seed_max_pool = 0
        # Conditioned-out holds (WP13r lever 1 = tc/'s write_marginals
        # held branch -- the graph side keeps the WP13c one-time tight
        # prior as the conditioned-constant analogue, disclosed).
        self.cond_hold = 1
        self.cond_fix_res_max = 0.0
        self.cond_min_nb = 0
        # Per-sat quarantine (optimize/stage.py ar_persist_bad, preset on)
        self.ar_persist_bad_enable = 1
        self.ar_persist_bad_res_thresh = 2.0
        self.ar_persist_bad_streak = 4
        self.ar_persist_bad_hold = 10
        # This port's own non-tc/ layers: OFF
        self.nlos_enable = 0
        self.nlos_ar_exclude = 0
        self.nlos_commit_reject = 0
        self.nlos_meas_mode = 0
        self.cp_hold_enable = False            # our reduced FSM, not tc/'s
        # alias: tc/'s per-sat quarantine state lives in the imported
        # SatQualityState so sq.tick/compute_cp_build_policy see it.
        self._persist_bad_hold = self._sq.persist_bad_hold
        self._persist_bad_streak = self._sq.persist_bad_streak
        # tc/-cfg namespace for the IMPORTED tc/ functions
        # (sq.sat_badness / compute_cp_build_policy read tc.cfg.*).
        # Values = TcConfig defaults overlaid with the tokyo preset.
        self.cfg = types.SimpleNamespace(
            # sat_badness (preset)
            sat_badness_enable=1,
            sat_badness_ddpr_thresh=1.0,
            sat_badness_cppr_thresh=1,
            sat_badness_alpha_ddpr=1.0,
            sat_badness_alpha_cppr=1.0,
            sat_badness_alpha_recent_cppr=0.25,
            sat_badness_alpha_recent_worst=0.75,
            sat_badness_alpha_recent_ref=0.5,
            sat_badness_alpha_recent_pair=0.0,
            sat_badness_alpha_obsq_ewma=1.0,
            sat_badness_alpha_obsq_streak=0.5,
            sat_badness_alpha_el=0.05,
            sat_badness_alpha_snr=0.05,
            sat_badness_sigma_scale_pr=0.0,
            sat_badness_sigma_scale_cp=1.5,
            sat_badness_el_ref_deg=30.0,
            sat_badness_snr_ref_dbhz=40.0,
            sat_badness_snr_span_db=10.0,
            obsq_res_thresh=2.0,
            obsq_bad_streak_cap=8,
            # CP build policy tiers (factors_support)
            cp_hold_sigma_penalty=0.0,
            cp_hold_dirty_reset_penalty=15.0,
            cp_release_probation_penalty=10.0,
            pair_bad_cp_hold_thresh=0.0,
            pair_bad_cp_hold_penalty=0.0,
            # dirty-sat reset ladder (preset)
            cp_hold_dirty_reset_enable=1,
            cp_hold_dirty_reset_hold=5,
            cp_hold_dirty_reset_suspect_count=2,
            cp_hold_dirty_reset_cooldown=3,
            recov_cp_release_thresh=2.0,
            recov_cp_release_count=5,
            # CP-vs-PR innovation gate (preset)
            cp_pr_innov_thresh=10.0,
            cp_pr_rejc_max=2,
            # post-fit DDPR per-sat counters (preset)
            post_ddpr_reset_thresh=1.5,
            post_ddpr_reset_count=3,
            ar_context_pair_bad_max=4.0,
        )

    def X(self, ep):
        return gtsam.symbol('x', ep)

    def N(self, sat, f):
        # WP13c: fold in a per-(sat,freq) generation counter (bumped by
        # _check_release) so a released ambiguity gets a brand-new gtsam
        # key -- the old key's tight hold prior can never be un-injected,
        # so "release" means abandoning it, not mutating it. Generation
        # stays small (a handful of releases per run at most), so this
        # keeps sat/freq/generation collision-free within a 64-bit symbol.
        gen = self._amb_gen.get((sat, f), 0)
        return gtsam.symbol('n', (sat * 10 + f) * 1000 + gen)

    # An ambiguity-pruning scheme was attempted here (bound self.amb_keys
    # to a rolling window instead of "every satellite ever tracked this
    # run", replacing this deterministic `self.N` lookup with a fresh
    # gtsam key on reacquisition after a long gap) to fix the superlinear
    # per-epoch cost growth over a long run. It roughly halved wall-clock
    # time on a 6000-epoch run2 window, but a 6000-epoch correctness check
    # caught a severe divergence (AllRMS ~7000 m from ep~3600 on) not
    # present in this un-pruned code or in shorter (<=2000 ep) windows --
    # almost certainly because forcing a *fresh* ambiguity re-
    # initialization (from the current, possibly-imprecise position
    # estimate) every time a satellite reappears after a gap is materially
    # worse than this design's "initialize once, keep forever" ambiguity,
    # which reuses whatever value that satellite had already converged to.
    # Reverted -- see WP13A_REPORT.md section 3c. `_build_dd_factors`
    # below calls `self.N` directly, as originally designed.

    def _manage_ambiguities(self, obs):
        """Cycle slip detection + new ambiguity initialization (cssrlib fork helper)."""
        self.update_ambiguities(obs)

    def _get_nlos_sats(self, obs):
        """WP13j: this epoch's NLOS-flagged satellite set (int sat ids),
        keyed by rover TOW rounded to the mask's own 0.2s tick. Cached by
        TOW so repeated calls within the same epoch (AR-exclude in
        `_do_ar`, measurement weighting in `_build_dd_factors*`) are O(1)
        after the first. Returns an empty frozenset when NLOS_ENABLE=0 or
        this TOW has no mask entry at all (tracked in
        `nlos_lookup_hit`/`_miss` for the alignment report)."""
        if not self.nlos_enable:
            return frozenset()
        _week, tow = time2gpst(obs.t)
        key = round(tow, 1)
        if key == self._nlos_cache_key:
            return self._nlos_cache_val
        self.nlos_lookup_epochs += 1
        sats = self._nlos_by_tow.get(key)
        if sats is None:
            self.nlos_lookup_miss += 1
            sats = frozenset()
        else:
            self.nlos_lookup_hit += 1
            sats = frozenset(sats)
        self._nlos_cache_key = key
        self._nlos_cache_val = sats
        return sats

    def _build_dd_factors(self, graph, new_values, obs, obsb, obs_sd,
                           rs, rsb, sat, el, iu, ir, pos_pred, ep):
        """Build DD-PR and DD-CP factors, initialize new ambiguities.

        Reference sat per system = max elevation. Rover-side geometry uses
        rover-time satellite positions `rs`; base-side geometry uses
        base-time satellite positions `rsb` -- kept separate (NOT folded
        into a scalar base range like `local_fgo`), matching the
        reference `rtk.py`'s `_build_dd_factors`.
        """
        nv = 0
        new_amb = {}
        # (sat, f) ambiguities that appear in a DD-CP factor THIS epoch. Only
        # these feed the current epoch's LAMBDA (ddidx selects sats present in
        # the current `sat` set), so restricting the joint-marginal covariance
        # to them is a pure speed-up with identical AR input -- see
        # _write_back / process() perf note.
        cp_amb = set()
        # WP13d: reset this epoch's DD factor tag lists (LOCAL index within
        # `graph`, i.e. `graph.size()` at the moment each factor is added --
        # `_apply_fde` globalizes these once it knows this epoch's
        # `nf_before` offset into the smoother's own factor vector).
        self._epoch_ddpr_tags_local = []
        self._epoch_ddcp_tags_local = []
        base_pt = gtsam.Point3(*self.nav.rb)
        rb = np.array(self.nav.rb)
        key_x = self.X(ep)

        for sys_id in _sorted_sys_ids(obs_sd.sig):
            idx_sys = self.sysidx(sat, sys_id)
            if len(idx_sys) < 2:
                continue
            ref_idx = idx_sys[np.argmax(el[idx_sys])]
            ref_sat = sat[ref_idx]
            lams = _get_wavelengths(self.nav, obs_sd, ref_sat)

            for j_idx in idx_sys:
                if j_idx == ref_idx:
                    continue
                sat_j = sat[j_idx]
                ri, ji = ref_idx, j_idx

                for f in range(self.nav.nf):
                    if f >= len(lams):
                        continue
                    lam_f = lams[f]
                    if obs_sd.P[ri, f] == 0 or obs_sd.P[ji, f] == 0:
                        continue

                    rs_ref = gtsam.Point3(*rs[iu[ri], :3])
                    rs_j = gtsam.Point3(*rs[iu[ji], :3])
                    rsb_ref = gtsam.Point3(*rsb[ir[ri], :3])
                    rsb_j = gtsam.Point3(*rsb[ir[ji], :3])

                    has_cp = obs_sd.L[ri, f] != 0 and obs_sd.L[ji, f] != 0
                    if has_cp:
                        kr = self.N(ref_sat, f)
                        kj = self.N(sat_j, f)
                        for sn, kn, sd_cp, rs_s in [
                                (ref_sat, kr, obs_sd.L[ri, f] * lam_f, rs[iu[ri], :3]),
                                (sat_j, kj, obs_sd.L[ji, f] * lam_f, rs[iu[ji], :3])]:
                            if (sn, f) not in self.amb_keys and (sn, f) not in new_amb:
                                r_r, _ = geodist(rs_s, pos_pred)
                                r_b, _ = geodist(rs_s, rb)
                                n0 = sd_cp - (r_r - r_b)
                                new_values.insert(kn, n0)
                                graph.addPriorDouble(
                                    kn, n0,
                                    gtsam.noiseModel.Isotropic.Sigma(1, self.sigma_amb0))
                                new_amb[(sn, f)] = kn
                                self.nav.x[self.IB(sn, f, self.nav.na)] = n0

                        ddcp_local_idx = graph.size()
                        graph.add(gtsam.DoubleDifferenceCarrierPhaseFactor(
                            key_x, kr, kj,
                            float(obs.L[iu[ri], f]) * lam_f,
                            float(obsb.L[ir[ri], f]) * lam_f,
                            float(obs.L[iu[ji], f]) * lam_f,
                            float(obsb.L[ir[ji], f]) * lam_f,
                            rs_ref, rs_j, rsb_ref, rsb_j, base_pt, lam_f,
                            gtsam.noiseModel.Isotropic.Sigma(1, self.sigma_cp * np.sqrt(2))))
                        self._epoch_ddcp_tags_local.append(
                            (ddcp_local_idx, int(ref_sat), int(sat_j), f))
                        cp_amb.add((int(ref_sat), f))
                        cp_amb.add((int(sat_j), f))
                        nv += 1
                    else:
                        noise_pr = gtsam.noiseModel.Isotropic.Sigma(
                            1, self.sigma_pr * np.sqrt(2))
                        if self.huber_pr > 0:
                            noise_pr = gtsam.noiseModel.Robust.Create(
                                gtsam.noiseModel.mEstimator.Huber.Create(self.huber_pr),
                                noise_pr)
                        ddpr_local_idx = graph.size()
                        graph.add(gtsam.DoubleDifferencePseudorangeFactor(
                            key_x,
                            float(obs.P[iu[ri], f]), float(obsb.P[ir[ri], f]),
                            float(obs.P[iu[ji], f]), float(obsb.P[ir[ji], f]),
                            rs_ref, rs_j, rsb_ref, rsb_j, base_pt, noise_pr))
                        self._epoch_ddpr_tags_local.append(
                            (ddpr_local_idx, int(ref_sat), int(sat_j), f))
                        nv += 1

        # NOTE: do NOT merge new_amb into self.amb_keys here. This graph
        # (and the ambiguity priors/values just built for new_amb) may still
        # be abandoned by the caller this epoch (nv<4, or the ISAM2/IFLS
        # update itself throws) -- committing new_amb into self.amb_keys
        # unconditionally would then leave "ghost" keys that self.amb_keys
        # believes exist but were never actually inserted into the
        # smoother/ISAM2, which later crashes `smoother.update()` with
        # `IndexError: invalid map<K, T> key` once a per-epoch timestamp
        # refresh (`_sorted_amb_items(self.amb_keys)` in `process()`)
        # references one. `process()` merges new_amb into self.amb_keys
        # itself, only after a successful commit.
        return nv, new_amb, cp_amb

    def _write_back(self, estimate, key_x, cp_amb=None):
        """Write GTSAM estimate + joint-marginal covariance into nav.x/nav.P.

        THE crux of WP13a: `nav.P`'s ambiguity block (and its cross-
        covariance with position) comes from `gtsam.Marginals.
        jointMarginalCovariance([x] + active N keys)` -- the FGO's own
        joint posterior -- not a temporal-variance proxy. `resamb_lambda`
        (called right after this by `_do_ar`) reads exactly this `nav.P`.

        Perf/correctness note (disclosed deviation from the literal
        reference): the joint marginal + `nav.vsat` are restricted to the
        ambiguities carrying a DD-CP factor THIS epoch (`cp_amb`), instead
        of every ambiguity ever seen. LAMBDA's `ddidx` only reads `nav.P`
        rows/cols for satellites present in the current epoch's `sat` set,
        so the two are identical as LAMBDA input; but the reference's
        "all ever-seen" set grows over a long run (with IFLS refreshing
        every ambiguity's timestamp they never age out), making the
        per-epoch `jointMarginalCovariance` cost grow ~linearly and the
        whole run ~quadratic (measured: run1 decayed 4.6->2.1 ep/s by
        ep2000, extrapolating to ~5h). Restricting to `cp_amb` (~current
        sat count, ~20-40 keys) bounds it to constant per-epoch cost with
        no change to the AR decision. `cp_amb=None` falls back to the
        literal all-ever-seen behavior.
        """
        self.nav.P[:, :] = 0
        self.nav.vsat[:, :] = 0
        self.nav.x[0:3] = np.array(estimate.atPoint3(key_x))

        for (s, f), k in _sorted_amb_items(self.amb_keys):
            if estimate.exists(k):
                self.nav.x[self.IB(s, f, self.nav.na)] = estimate.atDouble(k)
                if cp_amb is None or (int(s), int(f)) in cp_amb:
                    self.nav.vsat[s - 1, f] = 1

        try:
            mg = (gtsam.Marginals(self.smoother.getFactors(), estimate)
                  if self.smoother else
                  gtsam.Marginals(self.isam.getFactorsUnsafe(), estimate))
            active = [(s, f, k) for (s, f), k in _sorted_amb_items(self.amb_keys)
                      if estimate.exists(k)
                      and (cp_amb is None or (int(s), int(f)) in cp_amb)]
            if active:
                keys = gtsam.KeyVector()
                keys.append(key_x)
                for s, f, k in active:
                    keys.append(k)
                jm = mg.jointMarginalCovariance(keys)
                self.nav.P[0:3, 0:3] = jm.at(key_x, key_x)
                for s, f, k in active:
                    idx = self.IB(s, f, self.nav.na)
                    Pxn = jm.at(key_x, k)
                    self.nav.P[0:3, idx] = Pxn[:, 0]
                    self.nav.P[idx, 0:3] = Pxn[:, 0]
                    self.nav.P[idx, idx] = jm.at(k, k)[0, 0]
                for i, (s1, f1, k1) in enumerate(active):
                    i1 = self.IB(s1, f1, self.nav.na)
                    for j, (s2, f2, k2) in enumerate(active):
                        if i >= j:
                            continue
                        i2 = self.IB(s2, f2, self.nav.na)
                        c = jm.at(k1, k2)[0, 0]
                        self.nav.P[i1, i2] = c
                        self.nav.P[i2, i1] = c
                self.last_joint_P = self.nav.P.copy()
            else:
                self.nav.P[0:3, 0:3] = mg.marginalCovariance(key_x)
                self.last_joint_P = self.nav.P.copy()
        except RuntimeError:
            self.last_joint_P = None

    def _valpos_real(self, v, R, thres):
        """RTKLIB-style post-fit residual chi-square gate (our own code, not
        a cssrlib edit -- see __init__ / VALPOS_THRES). Rejects the fix if
        any single-difference residual exceeds `thres` sigma."""
        var = np.diag(R)
        ok = var > 0
        if not np.any(ok):
            return True
        return bool(np.all(v[ok] ** 2 <= thres ** 2 * var[ok]))

    # ---- WP13k (TASK_M11.md): partial/subset AR, transcribed from tc/'s
    # `optimize/ar.py` `_run_lambda_attempts`/`_try_subset_ar`/
    # `_rank_subset_drop_sats` -- see __init__'s docstring block above for
    # why this (not cssrlib's own parmode=2 PAR, which is exclmax=1-capped)
    # is the real "partial AR" lever. No cssrlib edit: `resamb_lambda_
    # rtklib`/`resamb_lambda` are cssrlib methods already inherited via
    # `GtsamRtk(rtkpos)` -> `rtkpos(pppos)`; this only calls them in a
    # different order/combination than the port's own `_do_ar` used to.

    def _rank_subset_drop_sats(self, sat, el):
        """Rank this epoch's satellites worst-first (DD-PR residual, then
        lowest elevation) as subset-AR drop candidates -- tc/'s
        `_rank_subset_drop_sats`, minus the `rejc_cp_pr`/cppr term this
        port doesn't track (see __init__'s sat_badness docstring for the
        same disclosed gap)."""
        per_sat = self._main_ddpr_per_sat or {}
        sat_el = {}
        for i, s in enumerate(sat):
            si = int(s)
            sat_el[si] = max(sat_el.get(si, 0.0), float(el[i]))
        seen = set()
        rows = []
        for s in sat:
            si = int(s)
            if si in seen:
                continue
            seen.add(si)
            rows.append((float(per_sat.get(si, 0.0)), -float(sat_el.get(si, 0.0)), si))
        rows.sort(reverse=True)
        max_candidates = max(0, self.subset_ar_max_candidates)
        return [s for *_rest, s in rows[:max_candidates]]

    def _try_subset_ar(self, sat, el):
        """Retry AR with up to `subset_ar_max_drop` candidate bad
        satellites excluded (combinatorial, smallest-drop-count and
        highest-ratio preferred on ties) -- tc/'s `_try_subset_ar`,
        reduced to its core mres/dirty-sat gates omitted (this port
        doesn't track `rejc_cp_pr` for the dirty-sat-count gate; the
        candidate ranking itself already demotes DD-PR-dirty sats first,
        which is the dominant effect tc/'s own gates protect against).

        WP15: with `CUDA_LAMBDA=1` (and the USE_PAR routing this
        campaign ships), the combo evaluations are batched into one
        gnss_gpu.lambda_batch kernel launch; the winning subset is then
        re-fired through the normal CPU `resamb_lambda` so all nav.*
        side effects come from the unchanged code path. See the
        `__init__` WP15 block for the exact-emulation contract and the
        automatic CPU fallbacks."""
        self.subset_ar_attempt_count += 1
        if not self._wp15_time:
            if (self._cuda_mlambda_batch is not None
                    and self.subset_ar_use_par):
                handled, nb, xa = self._try_subset_ar_cuda(sat, el)
                if handled:
                    return nb, xa
                self.cuda_lambda_fallback_count += 1
            return self._try_subset_ar_cpu(sat, el)
        # WP15_TIME=1: same code, wrapped with a perf accumulator so the
        # cascade cost can be compared across runs using the run's OWN
        # non-cascade time as the machine-state control (the benchmark
        # host's wall-clock drifts >2x between identical runs).
        import time as _t
        _t0 = _t.perf_counter()
        try:
            if (self._cuda_mlambda_batch is not None
                    and self.subset_ar_use_par):
                handled, nb, xa = self._try_subset_ar_cuda(sat, el)
                if handled:
                    return nb, xa
                self.cuda_lambda_fallback_count += 1
            return self._try_subset_ar_cpu(sat, el)
        finally:
            self.wp15_cascade_seconds += _t.perf_counter() - _t0

    def _try_subset_ar_cpu(self, sat, el):
        """The original (WP13k/WP13o) sequential cascade -- unchanged
        except that the attempt counter moved to the `_try_subset_ar`
        dispatcher above."""
        from itertools import combinations
        min_nb = max(1, self.subset_ar_min_nb)
        max_drop = max(1, self.subset_ar_max_drop)
        candidates = self._rank_subset_drop_sats(sat, el)
        # Performance (disclosed adaptation, not in tc/'s ar.py, which
        # does not need it running under real hardware): stop as soon as a
        # combo clears a STRONG ratio bar (thresar+0.5, same margin tc/'s
        # OWN `resamb_lambda_subsets` uses to skip searching further
        # subsets) instead of exhaustively scoring every C(candidates,k)
        # combination -- WP13k found the naive exhaustive search made a
        # capped 400-epoch smoke test take >150 CPU-s without reaching
        # epoch 100 (most epochs fail the primary AR call in Tokyo
        # geometry, so nearly every epoch was paying the full combinatorial
        # cost). A strong-enough combo is accepted immediately; only when
        # NONE clears the strong bar does this fall back to the best
        # (highest-ratio) SUBAR bar combo found, same tie-break as before.
        strong_ratio = float(self.nav.thresar) + 0.5
        best = None  # (score, excl_set)
        vsat_snap = self.nav.vsat.copy()
        found_strong = False
        for k in range(1, max_drop + 1):
            if found_strong:
                break
            if k > len(candidates):
                break
            for combo in combinations(candidates, k):
                excl = set(int(x) for x in combo)
                sub_sat = [s for s in sat if int(s) not in excl]
                if len(sub_sat) < self.nav.minfixsats:
                    continue
                try:
                    for s in excl:
                        if 0 < s <= self.nav.vsat.shape[0]:
                            self.nav.vsat[s - 1, :] = 0
                    # WP13o: tc/'s `_run_single_ar_attempt` routes subset
                    # retries through `resamb_lambda_rtklib` (full ILS,
                    # ratio>=thresar gated INTERNALLY) when rtklib_mode --
                    # the WP13k-era `resamb_lambda(parmode=2)` call here
                    # let subset-sourced fixes bypass the ratio bar
                    # entirely (PAR's armode==2 short-circuits the ratio
                    # test): measured 44% phase-2 false-fix on
                    # results/wp13o/o8_subset_r2 before this fix.
                    # SUBSET_AR_USE_PAR=1 (disclosed non-literal option):
                    # keep cssrlib's PAR partial search in the cascade --
                    # full ILS is all-or-nothing and returns nb=0 whenever
                    # ANY remaining candidate is bad, PAR fixes the
                    # resolvable subset -- and let `hold_ratio` (set to
                    # tc/'s single 3.0 bar) gate its result OUTSIDE, which
                    # the o8 breakdown measured as a clean separator
                    # (ratio>=3: 0.0-5.7% false; ratio<2: 72-96% false).
                    if self.ar_rtklib_mode and not self.subset_ar_use_par:
                        nb, xa = self.resamb_lambda_rtklib(sub_sat)
                    else:
                        nb, xa = self.resamb_lambda(sub_sat, self.nav.parmode, self.nav.par_P0)
                except (SystemExit, Exception):
                    nb, xa = 0, None
                finally:
                    self.nav.vsat[:, :] = vsat_snap
                if nb < min_nb or xa is None:
                    continue
                ratio = (self._last_s1 / self._last_s0) if self._last_s0 > 0 else 0.0
                score = (float(ratio), int(nb), -k)
                if best is None or score > best[0]:
                    best = (score, excl)
                if ratio >= strong_ratio:
                    found_strong = True
                    break
        if best is None:
            self._last_subset_ar_used = False
            self._last_subset_ar_drop = []
            return 0, None
        _, excl = best
        sub_sat = [s for s in sat if int(s) not in excl]
        for s in excl:
            if 0 < s <= self.nav.vsat.shape[0]:
                self.nav.vsat[s - 1, :] = 0
        try:
            # WP13o: same rtklib_mode/USE_PAR routing as the search loop.
            if self.ar_rtklib_mode and not self.subset_ar_use_par:
                nb, xa = self.resamb_lambda_rtklib(sub_sat)
            else:
                nb, xa = self.resamb_lambda(sub_sat, self.nav.parmode, self.nav.par_P0)
        except (SystemExit, Exception):
            nb, xa = 0, None
        if nb < min_nb or xa is None:
            self._last_subset_ar_used = False
            self._last_subset_ar_drop = []
            return 0, None
        self._last_subset_ar_used = True
        self._last_subset_ar_drop = sorted(excl)
        self.subset_ar_used_count += 1
        if os.environ.get('WP13K_DBG_SUBSET'):
            ratio = (self._last_s1 / self._last_s0) if self._last_s0 > 0 else 0.0
            print(f"DBG_SUBSET drop={sorted(excl)} nb={nb} ratio={ratio:.3f} "
                  f"thresar={self.nav.thresar} hold_ratio={self.hold_ratio}")
        return nb, xa

    def _try_subset_ar_cuda(self, sat, el):
        """WP15: batched-CUDA evaluation of the subset-AR cascade.

        Exact-emulation contract (verified against the CPU loop line by
        line; the kernel itself is bit-identical to cssrlib `mlambda` on
        all 34,569 finite captured pipeline calls):

        - Combos are enumerated in the CPU loop's exact order (k
          ascending, `combinations` order) with the same
          `len(sub_sat) < minfixsats` skip.
        - Per combo, the SAME `ddidx` call runs (so `nav.fix` sees the
          same rewrite sequence) and (y, Qb) are built with the same
          expressions `resamb_lambda` uses; vsat is snapshot-restored
          exactly like the CPU loop's `finally`.
        - `self._last_s0/_last_s1` are updated per evaluated combo in
          order from the kernel's bit-identical s values (matching
          `resamb_lambda`'s own stash); combos whose input trips the
          WP13m finite guard / non-PD LambdaError update nothing,
          matching the CPU exception path.
        - Acceptance per combo replicates `resamb_lambda`:
          `nfix>0 and (armode==2 or s0<=0 or s1/s0>=thresar)`; nb is
          `len(ix)` on accept else 0. Scoring/strong-bar/break logic is
          the dispatcher-visible part of the CPU loop, verbatim.
        - The WINNING subset is re-fired through the normal CPU
          `resamb_lambda` (same vsat zeroing, no restore), so nav.xa/
          nav.Pa/nav.fix/restamb and the final `_last_s0/_last_s1` come
          from the unchanged code path -- the CPU cascade ends with the
          identical re-fire, so the final nav.* state matches by
          construction.
        - Returns (handled=False, ...) to request the exact CPU rerun
          when a combo was internally ACCEPTED (nfix>0 -> transient
          nav.xa/Pa/restamb side effects on the CPU) but rejected by
          `min_nb` with no winner found, or when the kernel reports an
          unsupported problem -- the only cases whose CPU side effects
          this emulation does not reproduce.
        """
        from itertools import combinations
        min_nb = max(1, self.subset_ar_min_nb)
        max_drop = max(1, self.subset_ar_max_drop)
        candidates = self._rank_subset_drop_sats(sat, el)
        strong_ratio = float(self.nav.thresar) + 0.5
        vsat_snap = self.nav.vsat.copy()
        armode = int(self.nav.parmode)
        na, nx = self.nav.na, self.nav.nx

        # Phase 1: enumerate + gather every combo's LAMBDA input.
        combos = []  # dicts: k, excl, nb_ix, y, Qb, state
        for k in range(1, max_drop + 1):
            if k > len(candidates):
                break
            for combo in combinations(candidates, k):
                excl = set(int(x) for x in combo)
                sub_sat = [s for s in sat if int(s) not in excl]
                if len(sub_sat) < self.nav.minfixsats:
                    continue
                c = {'k': k, 'excl': excl, 'nb_ix': 0,
                     'y': None, 'Qb': None, 'state': 'ok'}
                try:
                    for s in excl:
                        if 0 < s <= self.nav.vsat.shape[0]:
                            self.nav.vsat[s - 1, :] = 0
                    ix = self.ddidx(self.nav, sub_sat)
                    if len(ix) <= 0:
                        # resamb_lambda's own no-DD path (returns -1).
                        print("no valid DD")
                        c['state'] = 'no_dd'
                    else:
                        y = self.nav.x[ix[:, 0]] - self.nav.x[ix[:, 1]]
                        DP = (self.nav.P[ix[:, 0], na:nx]
                              - self.nav.P[ix[:, 1], na:nx])
                        Qb = DP[:, ix[:, 0] - na] - DP[:, ix[:, 1] - na]
                        if not (np.all(np.isfinite(y))
                                and np.all(np.isfinite(Qb))):
                            # WP13m finite guard -> LambdaError on CPU.
                            c['state'] = 'raised'
                        else:
                            c['nb_ix'] = len(ix)
                            c['y'] = y
                            c['Qb'] = Qb
                except (SystemExit, Exception):
                    c['state'] = 'raised'
                finally:
                    self.nav.vsat[:, :] = vsat_snap
                combos.append(c)

        # Phase 2: one kernel launch for every evaluable combo.
        evals = [c for c in combos if c['state'] == 'ok']
        if evals:
            try:
                res = self._cuda_mlambda_batch(
                    [c['y'] for c in evals], [c['Qb'] for c in evals],
                    ncands=2, parmode=armode, P0=self.nav.par_P0)
            except Exception:
                return False, 0, None  # GPU hiccup -> exact CPU rerun
            self.cuda_lambda_batch_count += 1
            self.cuda_lambda_combo_count += len(evals)
            for c, r in zip(evals, res):
                c['res'] = r
                if r.status == 2:
                    return False, 0, None  # unsupported dims -> CPU

        # Phase 3: the CPU loop's selection logic, verbatim.
        best = None  # (score, excl)
        found_strong = False
        accepted_but_small_nb = False
        for c in combos:
            if found_strong:
                break
            if c['state'] == 'no_dd':
                continue  # nb=-1 < min_nb on the CPU path
            if c['state'] == 'raised':
                continue  # nb, xa = 0, None on the CPU path
            r = c['res']
            if r.status == 1:
                # cssrlib LambdaError inside mlambda: the CPU except
                # path leaves _last_s0/_last_s1 untouched.
                continue
            s0 = float(r.s[0]) if len(r.s) > 0 else 0.0
            s1 = float(r.s[1]) if len(r.s) > 1 else 0.0
            # resamb_lambda's own stash, in evaluation order.
            self._last_s0 = s0
            self._last_s1 = s1
            accepted = (r.nfix > 0
                        and (armode == 2 or s0 <= 0.0
                             or s1 / s0 >= self.nav.thresar))
            nb = c['nb_ix'] if accepted else 0
            if nb < min_nb:
                if nb > 0:
                    # CPU would have run restamb/nav.xa updates here.
                    accepted_but_small_nb = True
                continue
            ratio = (self._last_s1 / self._last_s0
                     if self._last_s0 > 0 else 0.0)
            score = (float(ratio), int(nb), -c['k'])
            if best is None or score > best[0]:
                best = (score, c['excl'])
            if ratio >= strong_ratio:
                found_strong = True
                break

        if best is None:
            if accepted_but_small_nb:
                # Transient nav.xa/Pa/restamb effects not reproduced:
                # hand the whole cascade back to the exact CPU loop.
                return False, 0, None
            self._last_subset_ar_used = False
            self._last_subset_ar_drop = []
            return True, 0, None

        # Phase 4: winner re-fire through the unchanged CPU path (same
        # trailing statements as the CPU cascade).
        _, excl = best
        sub_sat = [s for s in sat if int(s) not in excl]
        for s in excl:
            if 0 < s <= self.nav.vsat.shape[0]:
                self.nav.vsat[s - 1, :] = 0
        try:
            nb, xa = self.resamb_lambda(sub_sat, self.nav.parmode,
                                        self.nav.par_P0)
        except (SystemExit, Exception):
            nb, xa = 0, None
        if nb < min_nb or xa is None:
            self._last_subset_ar_used = False
            self._last_subset_ar_drop = []
            return True, 0, None
        self._last_subset_ar_used = True
        self._last_subset_ar_drop = sorted(excl)
        self.subset_ar_used_count += 1
        if os.environ.get('WP13K_DBG_SUBSET'):
            ratio = (self._last_s1 / self._last_s0
                     if self._last_s0 > 0 else 0.0)
            print(f"DBG_SUBSET drop={sorted(excl)} nb={nb} "
                  f"ratio={ratio:.3f} thresar={self.nav.thresar} "
                  f"hold_ratio={self.hold_ratio}")
        return True, nb, xa

    def _run_lambda_attempts(self, sat, el):
        """tc/'s `_run_lambda_attempts`: primary AR call (RTKLIB-faithful
        full-ILS + 1-sat round-robin exclusion via `resamb_lambda_rtklib`
        when `AR_RTKLIB_MODE` -- tc/'s own `rtklib_mode=1` default -- else
        the plain `resamb_lambda(sat, nav.parmode, nav.par_P0)` this port
        used before WP13k), THEN -- only if that fails outright (nb<=0) --
        the subset-AR fallback above."""
        self._last_resamb_raw_nb = -1
        # Reset every call -- `_try_subset_ar` only sets this when it is
        # actually invoked; without a reset here it would carry a stale
        # True/False from whichever PAST epoch last ran subset AR into an
        # epoch where the PRIMARY call succeeded directly (bug found
        # during WP13k's own capped verification).
        self._last_subset_ar_used = False
        try:
            if self.ar_rtklib_mode:
                nb, xa = self.resamb_lambda_rtklib(sat)
            else:
                nb, xa = self.resamb_lambda(sat, self.nav.parmode, self.nav.par_P0)
        except (SystemExit, Exception):
            return 0, None
        self._last_resamb_raw_nb = int(nb)
        # Subset-AR fallback gated to Phase 2 only (GtsamRtkTc; GtsamRtk's
        # GNSS-only Point3 float has no `.phase` attr and is treated as
        # Phase 2 for this gate) -- Phase 1 is a brief (~200-epoch)
        # stationary-assumption GNSS-only bootstrap, not the operative
        # long-run path (see WP13i's own disclosed scope note), and its
        # geometry/DD-PR history is not yet meaningful for dirty-sat
        # ranking. Skipping it there also removes ~200 epochs' worth of
        # wasted combinatorial search from every capped/full run.
        if (nb <= 0 and self.subset_ar_enable
                and len(sat) >= self.subset_ar_min_nb + 1
                and getattr(self, 'phase', 2) == 2):
            try:
                nb, xa = self._try_subset_ar(sat, el)
            except (SystemExit, Exception):
                nb, xa = 0, None
        if nb <= 0:
            return 0, None
        return nb, xa

    # ---- WP13k priority 2: sat_badness EWMA (tc/'s `preprocess/
    # sat_quality.py`), see __init__'s docstring for the reduced scope.

    def _update_sat_badness(self):
        if not self.sat_badness_enable:
            return
        per_sat = self._main_ddpr_per_sat or {}
        alpha = min(max(self.sat_badness_ewma_alpha, 0.0), 1.0)
        decay = min(max(self.sat_badness_decay, 0.0), 1.0)
        seen = set()
        for s, r in per_sat.items():
            si = int(s)
            seen.add(si)
            prev = float(self._sat_badness_ewma.get(si, 0.0))
            cur = float(r)
            self._sat_badness_ewma[si] = cur if prev <= 0 else (
                (1.0 - alpha) * prev + alpha * cur)
        for si in list(self._sat_badness_ewma.keys()):
            if si not in seen:
                self._sat_badness_ewma[si] *= decay

    def _sat_badness_score(self, sat_id):
        if not self.sat_badness_enable:
            return 0.0
        thr = max(1e-6, self.sat_badness_ddpr_thresh)
        r = float(self._sat_badness_ewma.get(int(sat_id), 0.0))
        return max(0.0, r / thr)

    # ---- WP13k priority 3: cp_hold FSM (reduced scope), see __init__'s
    # docstring block.

    def _update_cp_hold(self):
        """Tick the global cp-hold countdown, then check THIS epoch's own
        `_main_ddpr_res` (already computed by `_compute_main_dd_res`) as a
        trigger for a NEW hold window -- tc/'s `trigger_cp_hold`, reduced
        to a single trigger source (main-DDPR-residual spike) instead of
        tc/'s richer set (slip-burst/innovation/FDE-safeguard)."""
        if not self.cp_hold_enable:
            return
        if self._cp_hold_remaining > 0:
            self._cp_hold_remaining -= 1
            self.cp_hold_active_epochs += 1
        if self._main_ddpr_res > self.cp_hold_trigger_thresh:
            if self._cp_hold_remaining <= 0:
                self.cp_hold_trigger_count += 1
            self._cp_hold_remaining = max(self._cp_hold_remaining, self.cp_hold_epochs)

    # ---- WP13n (TASK_M14.md) piece 1: slip/outage-driven GTSAM ambiguity
    # reset -- tc/'s `preprocess/slip_detect.py` transcription. See
    # `__init__`'s WP13n block for the full derivation.

    def _mw_n_wl(self, obs, iu_idx, s, f, lams):
        """Melbourne-Wuebbena N_WL [cyc] for sat s, (L1, Lf) pair --
        tc/'s `_compute_mw_n_wl`, verbatim math (rover-only)."""
        if (iu_idx >= obs.L.shape[0] or obs.L.shape[1] <= f
                or obs.P.shape[1] <= f or f < 1):
            return None
        L1 = obs.L[iu_idx, 0]
        Lf = obs.L[iu_idx, f]
        P1 = obs.P[iu_idx, 0]
        Pf = obs.P[iu_idx, f]
        if L1 == 0.0 or Lf == 0.0 or P1 == 0.0 or Pf == 0.0:
            return None
        if len(lams) <= f or lams[0] <= 0 or lams[f] <= 0 or lams[0] == lams[f]:
            return None
        f1 = 1.0 / lams[0]
        f2 = 1.0 / lams[f]
        lam_wl = 1.0 / (f1 - f2)
        if lam_wl <= 0:
            return None
        phi_wl = L1 - Lf
        p_nl = (f1 * P1 + f2 * Pf) / (f1 + f2)
        return phi_wl - p_nl / lam_wl

    def _reset_slipped_ambiguities(self, obs, obsb, obs_sd, sat, iu, ir, ep,
                                   el=None):
        """tc/'s `detect_slips_and_manage_amb` analogue: release (bump
        generation -> fresh GTSAM key next build) every tracked (sat,f)
        ambiguity hit this epoch by (a) an LLI/GF slip flag (`nav.slip`,
        set by qcedit on BOTH receivers and merged by
        `single_differences` -- read here BEFORE `_manage_ambiguities`
        consumes+clears it), (b) an MW jump (rover-only, time-averaged
        mean, tc/'s `_check_slip_mw_avg`), (c) a CMC jump (rover-base,
        tc/'s inline CMC block), or (d) an outage longer than
        `slip_reset_maxout` epochs. Uses the existing
        `_release_ambiguity` (same mechanism as WP13c release / WP13d
        FDE slip-reset -- two triggers before, three now)."""
        if not self.slip_reset_enable:
            return 0
        nf = self.nav.nf
        ns = len(sat)
        sat_set = {int(s) for s in sat}
        ir_map = {int(sat[i]): ir[i] for i in range(ns)} if ir is not None else {}

        reset_keys = set()

        # (a) LLI + GF from nav.slip (cssrlib qcedit, rover|base merged).
        # WP13p (e): SLIP_MARGINAL_CONFIRM_N>0 debounces the LLI/GF
        # trigger for MARGINAL (below-AR-tier el/CNR) non-held sats: the
        # flag must persist N consecutive flagged epochs before the key
        # is released. Above-tier sats and committed holds keep the
        # immediate (tc/-literal) release. Default 0 = off.
        _hyst_n = self.slip_marginal_confirm_n
        _el_map, _cnr_map = {}, {}
        if _hyst_n > 0 and el is not None:
            for _i in range(ns):
                _s = int(sat[_i])
                _el_map[_s] = float(np.rad2deg(el[_i]))
                for _f in range(nf):
                    try:
                        _cnr_map[(_s, _f)] = float(obs.S[iu[_i], _f])
                    except (IndexError, TypeError, ValueError):
                        _cnr_map[(_s, _f)] = 0.0

        def _is_marginal(s_, f_):
            _e = _el_map.get(s_)
            if (_e is not None and self.ar_tier_elmin_deg > 0
                    and _e < self.ar_tier_elmin_deg):
                return True
            _c = _cnr_map.get((s_, f_), 0.0)
            return bool(self.ar_tier_cnr_min > 0
                        and 0.0 < _c < self.ar_tier_cnr_min)

        _flagged_now = set()
        for (s, f) in list(self.amb_keys):
            if 1 <= s <= self.nav.slip.shape[0] and f < self.nav.slip.shape[1]:
                if self.nav.slip[s - 1, f]:
                    if (_hyst_n > 0 and (s, f) not in self._committed
                            and _is_marginal(int(s), int(f))):
                        _flagged_now.add((s, f))
                        n_flag = self._slip_flag_streak.get((s, f), 0) + 1
                        self._slip_flag_streak[(s, f)] = n_flag
                        if n_flag >= _hyst_n:
                            reset_keys.add((s, f))
                            self._slip_flag_streak.pop((s, f), None)
                    else:
                        reset_keys.add((s, f))
        if _hyst_n > 0:
            for _key in list(self._slip_flag_streak):
                if _key not in _flagged_now:
                    del self._slip_flag_streak[_key]

        # Per-sat wavelengths cached once per sat this epoch.
        lam_cache = {}

        def _lams(s_):
            if s_ not in lam_cache:
                try:
                    lam_cache[s_] = _get_wavelengths(self.nav, obs_sd, s_)
                except Exception:
                    lam_cache[s_] = []
            return lam_cache[s_]

        # (b) MW jump, tc/'s time-averaged variant (mw_avg_enable=1).
        mw_hits = set()
        if self.slip_reset_mw_thresh > 0 and nf >= 2:
            seen_mw = set()
            for i in range(ns):
                s = int(sat[i])
                for f in range(1, nf):
                    n_wl = self._mw_n_wl(obs, iu[i], s, f, _lams(s))
                    if n_wl is None:
                        continue
                    seen_mw.add((s, f))
                    state = self._mw_state.get((s, f))
                    if state is None:
                        self._mw_state[(s, f)] = [float(n_wl), 1, float(n_wl)]
                        continue
                    if abs(n_wl - state[2]) > self.slip_reset_mw_thresh:
                        # MW(L1, Lf) jump can be L1 or Lf slip -- reset both.
                        mw_hits.add((s, 0))
                        mw_hits.add((s, f))
                        self._mw_state[(s, f)] = [float(n_wl), 1, float(n_wl)]
                    else:
                        state[0] += float(n_wl)
                        state[1] += 1
                        state[2] = state[0] / state[1]
            for key in list(self._mw_state.keys()):
                if key not in seen_mw:
                    del self._mw_state[key]

        # (c) CMC jump (rover-base code-minus-carrier, per (s,f)).
        cmc_hits = set()
        if self.slip_reset_cmc_thresh > 0 and obsb is not None and ir_map:
            seen_cmc = set()
            for i in range(ns):
                s = int(sat[i])
                if s not in ir_map:
                    continue
                lams = _lams(s)
                for f in range(nf):
                    if f >= len(lams) or lams[f] <= 0:
                        continue
                    if (iu[i] >= obs.P.shape[0] or f >= obs.P.shape[1]
                            or ir_map[s] >= obsb.P.shape[0]
                            or f >= obsb.P.shape[1]):
                        continue
                    pr_rov = obs.P[iu[i], f]
                    cp_rov = obs.L[iu[i], f]
                    pr_bas = obsb.P[ir_map[s], f]
                    cp_bas = obsb.L[ir_map[s], f]
                    if pr_rov == 0 or cp_rov == 0 or pr_bas == 0 or cp_bas == 0:
                        continue
                    cmc = (pr_rov - pr_bas) - (cp_rov - cp_bas) * lams[f]
                    seen_cmc.add((s, f))
                    prev = self._cmc_state.get((s, f))
                    if prev is not None and abs(cmc - prev) > self.slip_reset_cmc_thresh:
                        cmc_hits.add((s, f))
                    self._cmc_state[(s, f)] = cmc
            for key in list(self._cmc_state.keys()):
                if key not in seen_cmc:
                    del self._cmc_state[key]

        # (d) outage: tracked amb unseen for > maxout epochs.
        outage_hits = set()
        for (s, f) in list(self.amb_keys):
            if s in sat_set:
                self._amb_last_seen[(s, f)] = ep
            else:
                last = self._amb_last_seen.get((s, f))
                if last is not None and (ep - last) > self.slip_reset_maxout:
                    outage_hits.add((s, f))

        n_released = 0
        # WP13s: stash this epoch's slip-flagged keys for tc/'s
        # sq.update_cp_lock (gate.py passes ed.slip_keys) -- read-only.
        self._last_slip_keys = (set(reset_keys) | set(mw_hits)
                                | set(cmc_hits) | set(outage_hits))
        for bucket, counter_name, _trig in (
                (reset_keys, 'slip_reset_count', 'slip'),
                (mw_hits, 'mw_reset_count', 'mw'),
                (cmc_hits, 'cmc_reset_count', 'cmc'),
                (outage_hits, 'outage_reset_count', 'outage')):
            for key in bucket:
                if key in self.amb_keys:
                    self._release_ambiguity(*key, reason=_trig)
                    self._amb_last_seen.pop(key, None)
                    self._amb_init_epoch.pop(key, None)
                    setattr(self, counter_name, getattr(self, counter_name) + 1)
                    n_released += 1
        return n_released

    # ---- WP13n piece 3: elevation-dependent DD sigma (tc/'s
    # `varerr_dd_sigma`, RTKLIB-demo5 varerr port).

    def _varerr_dd_sigma(self, code, el_ref_rad, el_j_rad):
        """sigma_DD [m] for one (code?, ref-el, j-el) pair -- tc/'s
        `prefit.varerr_dd_sigma` with tc/'s own pair-elevation choice
        (`max(min(el_ref, el_j), el_min)`, factors.py `pair_sigma`)."""
        el_min = np.radians(max(1.0, self.varerr_el_mask_deg))
        el = max(min(float(el_ref_rad), float(el_j_rad)), el_min)
        fact = self.varerr_eratio_pr if code else 1.0
        a = fact * self.varerr_err_a
        b = fact * self.varerr_err_b
        d = 299792458.0 * self.varerr_sclkstab * float(self._epoch_dt)
        sinel = max(np.sin(el), 0.05)
        var_sd = 2.0 * (a * a + b * b / (sinel * sinel)) + d * d
        return float(np.sqrt(2.0 * var_sd))

    def _do_ar(self, obs, rs, vs, dts, sat, el, iu):
        """LAMBDA AR + WP13c-gated fix-and-hold (TASK_M3.md).

        WP13a's original version treated LAMBDA's own ratio-test pass
        (`nb>0`, i.e. ratio>=nav.thresar) as sufficient to BOTH report
        this epoch Q=4 "fix" AND permanently inject a tight hold prior
        for every current candidate, via cssrlib's `nav.fix==2->3`
        bookkeeping. WP13A_REPORT.md section 3b/4 found this locks in
        wrong integers forever (FixRMS 20-192m at 100% "fix").

        This version separates "LAMBDA passed its own ratio test this
        instant" from "trustworthy enough to commit a permanent hold"
        from "trustworthy enough to report Q=4", using ONLY our own
        state (`self._streak`/`self._committed`, never `nav.fix` --
        `ddidx()` rewrites `nav.fix` from scratch on every
        `resamb_lambda` call, so anything living there cannot survive
        past the current call, let alone across epochs; committing a
        hold or rejecting a candidate must never be implemented as "read
        nav.fix again later", exactly the pitfall this task warns about):

          1. `hold_ratio` (env HOLD_RATIO, 0=off): an EXTRA ratio bar on
             top of nav.thresar that must clear before a candidate is
             even eligible to accrue streak credit toward a hold.
          2. `hold_confirm_n` (env HOLD_CONFIRM_N, default 1): consecutive
             qualifying epochs an (sat,freq)'s resolved value must stay
             unchanged (within `streak_tol`) before it is newly committed
             -- promoted into `self._committed` and given its one (and
             only one, ever -- see `_inject_hold`) tight GTSAM prior.
          3. `hold_valpos_thres` (env HOLD_VALPOS_THRES, 0=off): a real,
             absolute post-fit residual cap in METERS (not the tiny
             nominal CP sigma WP13A_REPORT.md 3b item 3's disclosed-
             negative `VALPOS_THRES` attempt used, which rejected nearly
             every epoch and collapsed the pipeline by disabling holding
             altogether). Gates ONLY whether a candidate is promoted this
             epoch -- never whether an already-committed hold keeps
             being used, and never the whole epoch's fix/float report by
             itself.

        Q=4 is reported only when this epoch's active candidate set
        includes at least `min_committed_for_fix` already-committed
        (validated) holds -- the concrete, in-our-own-code replacement
        for the no-op `rtkpos.valpos`.

        A rejected/not-yet-confirmed candidate NEVER mutates
        `self._committed` (the held set) -- only `_check_release`
        (explicit, evidence-based) removes an entry from it. Verified:
        seeing this docstring's guarantee hold, `self._committed`'s size
        is monotonically non-decreasing except at an explicit release
        (see WP13C_REPORT.md's unit-check).
        """
        self._last_nb = 0
        # WP13p: held-pool diagnostics default to "no candidates" -- only
        # overwritten below once ddidx has produced this epoch's set.
        self._last_ar_ncand = 0
        self._last_ar_ncommit = 0
        # WP13l: capture tc/'s `prev_smode==5` equivalent BEFORE any early
        # return below, and default this epoch's own report state to
        # "float" (only flipped True at the bottom if report_ok) -- see
        # `__init__`'s WP13l block docstring for why this can't just read
        # `self.nav.smode` the way tc/'s Layer-3 `_build_factor_block`
        # does (this port's `process()` has already reset it to 5 by the
        # time `_do_ar` runs).
        prev_was_flt = not self._last_report_was_fix
        self._last_report_was_fix = False
        # WP13s literal: tc/'s optimize/stage.py `_ar_eligibility` --
        # NO AR while the recovery CP-hold window is active, nor within
        # the first post-transition epochs (kk < n_collect + 3 = 8).
        # (ar_max_frac = 1.0 in the preset -> that term never gates.)
        _lit = bool(self.tc_literal) and getattr(self, 'phase', 1) == 2
        if _lit and (self._recov_cp_skip_now or self.epoch2 < 8):
            self._last_candidates = []
            return
        # WP13j (TASK_M10.md) layer (a): this epoch's NLOS set (computed
        # once regardless of which layer needs it -- (a)'s AR-exclude and
        # (c)'s commit-reject both read it below).
        nlos_sats = self._get_nlos_sats(obs) if self.nlos_enable else frozenset()
        # Temporarily zero `nav.vsat` for NLOS-flagged satellites BEFORE
        # calling resamb_lambda, exactly like `resamb_lambda_rtklib`'s own
        # round-robin exclusion mechanism -- `_ddidx_core` (cssrlib
        # pppssr.py) skips any (sat,freq) with `vsat==0` as both a
        # candidate AND a reference, so this drops NLOS sats from the
        # LAMBDA candidate set entirely for this epoch only (restored
        # right after, whether or not resamb_lambda throws).
        # WP13n piece 2(b): tc/'s per-sat residual gate (preprocess/
        # prefit.py `apply_per_sat_residual_gate`, preset
        # per_sat_gate_enable=1 / per_sat_res_thresh=3.0): zero vsat for
        # sats whose THIS-epoch main-graph DD-PR residual exceeded the
        # threshold, keeping multipath-biased sats out of the LAMBDA
        # candidate set. nav.vsat is rewritten from scratch by
        # `_write_back*` every epoch, so no restore needed (tc/ does not
        # restore either). Phase-2 only (the residuals exist there).
        if self.per_sat_gate_enable and getattr(self, 'phase', 1) == 2:
            # WP13o: tc/'s gate reads PR-only residuals (see __init__'s
            # WP13o block) -- opt-in signal re-point, default unchanged.
            _per_sat_cur = ((self._main_ddpr_per_sat_pr
                             if self.per_sat_gate_pr_only
                             else self._main_ddpr_per_sat) or {})
            for _s, _rmax in _per_sat_cur.items():
                _s = int(_s)
                if _rmax > self.per_sat_res_thresh and 1 <= _s <= self.nav.vsat.shape[0]:
                    for _f in range(self.nav.nf):
                        if self.nav.vsat[_s - 1, _f] == 1:
                            self.nav.vsat[_s - 1, _f] = 0
                            self.per_sat_gate_drop_count += 1

        # WP13p (a) AR_TIER_MODE=1: two-tier AR membership -- zero vsat
        # for marginal (below-tier el/CNR) satellites so they never enter
        # the LAMBDA candidate pool or the hold-eligible (ddidx fix==2)
        # set, while their DD-PR/DD-CP factors keep stabilizing the float
        # (mode 2 additionally suppresses their DD-CP at build time --
        # see `_build_dd_factors_arm`). nav.vsat is rewritten from
        # scratch by `_write_back*` every epoch, same as the per-sat gate
        # above. Committed holds exempt by default (AR_TIER_HELD_EXEMPT).
        if (self.ar_tier_mode >= 1 and getattr(self, 'phase', 1) == 2
                and (self.ar_tier_elmin_deg > 0 or self.ar_tier_cnr_min > 0)):
            _el_floor = np.deg2rad(self.ar_tier_elmin_deg)
            for _i in range(len(sat)):
                _s = int(sat[_i])
                if not (1 <= _s <= self.nav.vsat.shape[0]):
                    continue
                _low_el = (self.ar_tier_elmin_deg > 0
                           and float(el[_i]) < _el_floor)
                for _f in range(self.nav.nf):
                    if self.nav.vsat[_s - 1, _f] != 1:
                        continue
                    if (self.ar_tier_held_exempt
                            and (_s, _f) in self._committed):
                        continue
                    _low_cnr = False
                    if self.ar_tier_cnr_min > 0:
                        try:
                            _cnr_v = float(obs.S[iu[_i], _f])
                        except (IndexError, TypeError, ValueError):
                            _cnr_v = 0.0
                        # S==0 means "no reading on this band" (qcedit
                        # already floor-checked the recorded ones) --
                        # only a POSITIVE reading below the tier drops.
                        _low_cnr = 0.0 < _cnr_v < self.ar_tier_cnr_min
                    if _low_el or _low_cnr:
                        self.nav.vsat[_s - 1, _f] = 0
                        self.ar_tier_drop_count += 1

        saved_vsat = {}
        if self.nlos_ar_exclude and nlos_sats:
            for s in nlos_sats:
                if 0 < s <= self.nav.vsat.shape[0]:
                    saved_vsat[s] = self.nav.vsat[s - 1, :].copy()
                    self.nav.vsat[s - 1, :] = 0
            if saved_vsat:
                self.nlos_ar_excluded_epochs += 1
                self.nlos_ar_excluded_sat_epochs += len(saved_vsat)
        try:
            # WP13k (TASK_M11.md): tc/'s actual `_run_lambda_attempts`
            # chain (RTKLIB-faithful full-ILS + 1-sat round-robin +
            # subset-AR fallback) in place of the bare `resamb_lambda`
            # call this port used through WP13j -- see `_run_lambda_
            # attempts`'s docstring / __init__'s WP13k block for why.
            nb, xa = self._run_lambda_attempts(sat, el)
        except (SystemExit, Exception):
            return
        finally:
            for s, row in saved_vsat.items():
                self.nav.vsat[s - 1, :] = row
        # WP13g: record raw LAMBDA nb regardless of what happens below --
        # matches the reference's `ed.nb` semantics (AR's own resolved-
        # ambiguity count, not gated by valpos/ratio/hold-confirm).
        self._last_nb = nb
        if nb <= 0:
            return
        _xval_reject = self._ddpr_ar_xvalidate(xa)
        if _xval_reject and not _lit:
            # WP13i: mirrors tc/'s own `ed.nb = 0; ed.xa = None` on reject
            # (`optimize/stage.py` ~line 420) -- treat this exactly like a
            # LAMBDA non-fix (`nb<=0`) for every downstream consumer of
            # `self._last_nb` (e.g. WP13g sanity's fast-path `nb<=0`
            # check), report float (nav.smode already 5, untouched), and
            # do NOT touch self._committed/self._streak (WP13c's ddidx-
            # strips-holds pitfall).
            # WP13s literal mode: tc/ runs this xvalidate AFTER run_ar
            # has ALREADY fix-and-held (stage.py `_run_ar_with_marginals`
            # -- reject downgrades the REPORT, the holds stay), so the
            # literal path defers the reject to the report decision
            # below instead of returning here.
            self._last_nb = 0
            return
        yu, eu, _ = self.zdres(obs, None, None, rs, vs, dts, xa[0:3])
        v_fix, _, R_fix = self.sdres(obs, xa, yu[iu], eu[iu], sat, el)
        if not self.valpos(v_fix, R_fix):
            return
        if self.valpos_thres > 0 and not self._valpos_real(v_fix, R_fix, self.valpos_thres):
            return

        # WP13l: `lc` = tc/'s `lambda_correction` -- distance between the
        # LAMBDA-fixed antenna position (`xa[0:3]`) and THIS epoch's own
        # float antenna position. `self.nav.x[0:3]` is still the float
        # value here: `process()`'s `_write_back` sets it before `_do_ar`
        # runs, and nothing between then and here overwrites it (`zdres`/
        # `sdres` above take `xa[0:3]` as a parameter, they don't mutate
        # `nav.x`) -- exactly tc/'s own `pose_tc_antenna` vs `ed.xa[0:3]`
        # comparison in `validation/postprocess.py._decide_fix_or_flt`.
        lc = float(np.linalg.norm(np.asarray(xa[0:3]) - np.asarray(self.nav.x[0:3])))

        ratio = (self._last_s1 / self._last_s0) if self._last_s0 > 0 else 0.0

        # Carrier-phase residual, attributed PER (sat,freq) -- see the
        # _sdres_build_plan import note. A first attempt at this gate used
        # a single WHOLE-EPOCH max (either over all of v_fix, or over its
        # phase-only rows): both collapsed the pipeline exactly like
        # WP13a's original VALPOS_THRES, and a diagnostic run explained
        # why -- one already-wrong ambiguity's residual sits at a constant
        # ~50m EVERY epoch (a smoking gun for the wrong-fix-lock-in root
        # cause itself), so gating on the worst residual across ALL
        # candidates holds every OTHER, perfectly fine satellite's
        # commit hostage to that one satellite forever. Attributing each
        # phase row to its own (sat,freq) and gating per-candidate avoids
        # this: one bad satellite blocks only itself.
        resid_by_satfreq = {}
        try:
            plan = _sdres_build_plan(obs, sat, el, yu[iu], self.nav)
            sat_idx_arr, freq_idx_arr, is_phase_arr = plan[1], plan[2], plan[8]
            if len(is_phase_arr) == len(v_fix):
                for row in range(len(v_fix)):
                    if not is_phase_arr[row]:
                        continue
                    sp = int(sat[int(sat_idx_arr[row])])
                    fr = int(freq_idx_arr[row])
                    val = abs(float(v_fix[row]))
                    key = (sp, fr)
                    if val > resid_by_satfreq.get(key, 0.0):
                        resid_by_satfreq[key] = val
        except Exception:
            resid_by_satfreq = {}

        # Candidates ddidx flagged THIS epoch (fix==2, freshly computed by
        # the resamb_lambda call above) -- read immediately, before
        # holdamb_flags or anything else touches nav.fix again.
        fix_idx = np.argwhere(self.nav.fix == 2)
        candidates = [(int(s) + 1, int(f)) for s, f in fix_idx]
        cur_set = set(candidates)
        for key in list(self._streak):
            if key not in cur_set:
                del self._streak[key]  # evidence chain broke (sat dropped out); not a reject

        # WP13k found (per-epoch diff vs tc/'s own `results/tc_run2.npz`):
        # of 1998 phase-2 epochs where tc/ fixes and we don't, 1381 (69%)
        # are epochs where our own primary AR call ALSO returned nb>0,
        # rejected only by `hold_ratio`'s literal ratio bar -- a metric
        # mismatch, since `nav.parmode==2` (PAR)'s `resamb_lambda` never
        # gates on `s1/s0` at all (`armode==2` short-circuits the ratio
        # check in cssrlib's own `resamb_lambda`; PAR's real gate is
        # `Ps>=par_P0` INSIDE `mlambda.parsearch`, already enforced before
        # `nfix>0` is ever returned). WP13l (TASK_M12.md) replaces the
        # ratio bar for PAR-sourced fixes with exactly what tc/ does
        # instead (read from `tc/src/gnss_fgo/optimize/ar.py` +
        # `validation/postprocess.py`, not assumed -- see `__init__`'s
        # WP13l docstring for the full citation): trust `nb>0` outright
        # (already `par_P0`-gated internally), then apply tc/'s OWN
        # downstream purity stack (`_ar_context_reject` + `lambda_corr_
        # hard_max` position-jump sanity + `low_nb_fix_reject`/
        # `weak_fix_reject` cold-start gates) below, in place of a ratio
        # PAR was never designed to produce. The remaining 477/1998 (24%)
        # tc-fixes-we-miss are genuine AR failures (`raw_nb<=0`) -- a
        # candidate-set/covariance-quality gap sat_badness/cp_hold target,
        # not a gate-threshold issue; unaffected by this change.
        _par_sourced = ((self.nav.parmode == 2) and not self.ar_rtklib_mode)
        if _lit:
            # WP13s literal: tc/'s ONLY ratio gate is the thresar test
            # INSIDE resamb_lambda_rtklib (nb>0 here means it passed) --
            # no hold_ratio / PAR bar / small-nb bar extras.
            ratio_ok = True
        elif _par_sourced:
            # `par_ratio_min` (env PAR_RATIO_MIN, default 0=off) is NOT
            # part of tc/'s own accept path -- disclosed, empirically-
            # motivated addition (see WP13L_REPORT.md): a per-epoch diff
            # against `results/tc_run2.npz` showed tc/'s own downstream
            # stack above barely fires on ITS regime (full-ILS, nb mostly
            # >>6) because tc/'s ratio-free trust in `nb>0` is safe THERE
            # only because full-ILS's OWN ratio test already screened it
            # before `_ar_context_reject`/`_decide_fix_or_flt` ever run.
            # This port's LAMBDA path is PAR (WP13k Part 1: literal
            # `rtklib_mode=1` full ILS measured >40x too slow to run
            # here), and PAR's `s1/s0` -- while not the metric its OWN
            # `Ps>=par_P0` bootstrap gates acceptance on -- still
            # correlates with fix correctness in THIS port's covariance
            # (measured on this exact run: false-fix epochs' ratio median
            # 1.05 vs true-fix 8.83; a low floor here is a screen tc/
            # gets for free from full ILS and this port does not).
            ratio_ok = (self.par_ratio_min <= 0) or (ratio >= self.par_ratio_min)
        else:
            ratio_ok = (self.hold_ratio <= 0) or (ratio >= self.hold_ratio)

        # WP13p (b): small-nb purity backstop -- a LOW-nb (partial)
        # search must clear a HIGHER ratio bar than a rich one. WP13o's
        # O8 anatomy: at nb 5-8 only ratio>=10 separates true from false
        # (5.7% false there vs 72-96% below 2), while nb>=9 fixes are
        # already near-pure at the ordinary ratio>=3 bar -- one bar for
        # both populations was the honest-purity leak. Phase-2 scoped;
        # 0 = off = WP13o O11 behavior.
        if (ratio_ok and self.small_nb_max_p2 > 0
                and getattr(self, 'phase', 1) == 2
                and nb <= self.small_nb_max_p2
                and ratio < self.small_nb_ratio_p2):
            ratio_ok = False
            self.small_nb_bar_reject_count += 1
            self.par_gate_reject_by_reason['small_nb_bar'] = (
                self.par_gate_reject_by_reason.get('small_nb_bar', 0) + 1)

        # WP13l: tc/'s real downstream purity stack (see __init__ docstring
        # for the full citation/derivation) -- applied whenever the AR
        # result is otherwise accepted (`ratio_ok`), regardless of which
        # path produced it, exactly like tc/'s own `_ar_context_reject`/
        # `_decide_fix_or_flt` (both run unconditionally on any nb>0
        # `run_ar` success, not gated on parmode).
        par_gate_reason = None
        if ratio_ok:
            # WP13o: tc/'s `_ar_context_reject` reads the PR-only main
            # residual (`_cached_ddpr_res_pre` <- `main_ddpr_residuals`,
            # PR-only) -- opt-in re-point, default = WP13l/n behavior
            # (CP-folded signal). Also feeds `weak_fix_reject`'s
            # main_res term below, matching tc/'s single signal source.
            if self.ar_context_pr_only:
                main_res = float(self._main_ddpr_res_pr or 0.0)
                per_sat = self._main_ddpr_per_sat_pr or {}
            else:
                main_res = float(self._main_ddpr_res or 0.0)
                per_sat = self._main_ddpr_per_sat or {}
            worst_res = float(max(per_sat.values())) if per_sat else 0.0
            cp_hold_active = self._cp_hold_remaining > 0
            ddpr_bad_active = int(self._ddpr_bad_count or 0) > 0
            # pair_bad_max: tc/'s per-pair EWMA (preprocess/sat_quality.py)
            # is not tracked in this port's reduced sat_badness scope
            # (WP13k disclosed alpha_recent_pair=0) -- fixed at 0.0, this
            # term never fires, matching that same disclosed gap.
            # WP13s literal: tracked for real via the IMPORTED
            # SatQualityState (update_pair_quality on this epoch's
            # PR pair rows), gated at the preset's 4.0.
            if _lit:
                pair_bad_max = float(self._last_pair_bad_max or 0.0)
                _pair_bad_trip = pair_bad_max > float(
                    self.cfg.ar_context_pair_bad_max)
                # tc/'s cp_hold term reads _recov_cp_hold -- already
                # enforced by the eligibility skip above; our reduced
                # cp_hold FSM is off in literal mode.
                cp_hold_active = self._recov_cp_skip_now
            else:
                pair_bad_max = 0.0
                _pair_bad_trip = False
            burst_like = (
                (self.ar_context_reject_during_cp_hold and cp_hold_active)
                or (self.ar_context_reject_during_ddpr_bad and ddpr_bad_active)
                or (main_res > self.ar_context_main_ddpr_max)
                or (worst_res > self.ar_context_worst_sat_max)
                or _pair_bad_trip
            )
            if burst_like and nb <= self.ar_context_nb_max:
                par_gate_reason = 'ar_context_reject'
            elif self.lambda_corr_max > 0 and lc > self.lambda_corr_max:
                par_gate_reason = 'lambda_corr'
            elif self.lambda_corr_hard_max > 0 and lc > self.lambda_corr_hard_max:
                par_gate_reason = 'lambda_corr_hard'
            elif (self.low_nb_fix_reject_nb_max > 0
                    and nb <= self.low_nb_fix_reject_nb_max and prev_was_flt):
                par_gate_reason = 'low_nb_fix_reject'
            elif (self.weak_fix_nb_max > 0 and nb <= self.weak_fix_nb_max
                    and prev_was_flt
                    and (lc > self.weak_fix_lambda_corr_max
                         or main_res > self.weak_fix_main_ddpr_res_max)):
                par_gate_reason = 'weak_fix_reject'
            if par_gate_reason is not None:
                ratio_ok = False
                self.par_gate_reject_count += 1
                self.par_gate_reject_by_reason[par_gate_reason] = (
                    self.par_gate_reject_by_reason.get(par_gate_reason, 0) + 1)

        # WP13s literal commit/report split: in tc/, fix-and-hold runs
        # INSIDE run_ar (optimize/ar.py `_apply_fix_and_hold`), i.e.
        # BEFORE postprocess's `_decide_fix_or_flt` -- an epoch
        # downgraded to FLT by lambda_corr_hard / low_nb / weak_fix
        # STILL keeps its holds. Only valpos (already returned above)
        # and ar_context_reject (run_ar's own `_validate_fix`) block the
        # hold itself.
        _lit_commit_ok = _lit and (par_gate_reason is None
                                   or par_gate_reason not in
                                   ('ar_context_reject',))

        if os.environ.get('WP13O_DBG_GATE') and getattr(self, 'phase', 1) == 2:
            # WP13o: one line per phase-2 nb>0 epoch -- everything the
            # accept ladder read, plus the verdict (diagnostic only).
            print(f"WP13O_GATE ep={self.epoch2} nb={nb} ratio={ratio:.2f} "
                  f"lc={lc:.3f} prev_flt={prev_was_flt} "
                  f"reason={par_gate_reason} ratio_ok={ratio_ok} "
                  f"mres_pr={self._main_ddpr_res_pr:.2f} "
                  f"mres={float(self._main_ddpr_res or 0.0):.2f} "
                  f"ncand={len(candidates)} ncommit={len(self._committed)}")

        if os.environ.get('WP13I_DIAG_XVAL') and getattr(self, 'phase', 1) == 2:
            # WP13i diagnostic (non-destructive): compute res_at_xa for
            # EVERY phase-2 nb>0 epoch and log it beside ratio_ok, so we
            # can correlate (offline, via the npz err3d) whether would-be
            # fixes (ratio_ok=True) actually carry a large DD-PR residual
            # at xa. Only runs when WP13I_DIAG_XVAL is set; the real gate
            # (AR_DDPR_XVALIDATE_THRESH) is OFF in this diagnostic mode.
            try:
                key_x = self.PX(self.epoch2)
                cur_pose = self.current_estimate.atPose3(key_x)
                Rb = self.ecef_T_nav.compose(cur_pose).rotation().matrix()
                body_ecef_xa = np.asarray(xa[0:3], dtype=float) - Rb @ self.lever_arm
                body_nav_xa = self.ecef_T_nav.transformTo(gtsam.Point3(*body_ecef_xa))
                xa_pose = gtsam.Pose3(cur_pose.rotation(), body_nav_xa)
                res_xa = self._ddpr_res_at_pose(xa_pose)
            except (RuntimeError, AttributeError):
                res_xa = None
            print(f"DIAG_XVAL ep2={self.epoch2} ratio={ratio:.3f} "
                  f"ratio_ok={ratio_ok} res_xa={res_xa} "
                  f"res_pre={self.last_main_ddpr_rms:.2f} ncand={len(candidates)}")

        if os.environ.get('WP13F_DBG_GATE'):
            ep_dbg = getattr(self, 'epoch2', None) if getattr(self, 'phase', 1) == 2 else self.epoch
            print(f"DBG_GATE ep={ep_dbg} phase={getattr(self, 'phase', 1)} "
                  f"hold_ratio={self.hold_ratio} fde_enable={self.fde_enable} "
                  f"armode={self.nav.armode} ratio={ratio:.3f} ratio_ok={ratio_ok} "
                  f"n_cand={len(candidates)} n_committed={len(self._committed)}")

        newly_committed = []
        if _lit:
            # WP13s literal fix-and-hold: EVERY ddidx fix==2 candidate
            # is held IMMEDIATELY at its LAMBDA value (tc/'s
            # holdamb_flags -> _activate_phase2_hold_states: no
            # confirm-N, no extra ratio bar, no per-candidate residual
            # cap, no min-committed floor). Re-fix re-pins the stored
            # value at the new xa, like tc/'s activate_hold every fix
            # epoch.
            if _lit_commit_ok:
                for (s, f) in candidates:
                    val = float(xa[self.IB(s, f, self.nav.na)])
                    if not np.isfinite(val):
                        continue
                    if (s, f) not in self._committed:
                        newly_committed.append((s, f))
                        self.lit_commit_count += 1
                    self._committed[(s, f)] = val
            else:
                self.lit_ar_skip_hold_count += 1
        elif ratio_ok:
            for (s, f) in candidates:
                val = float(xa[self.IB(s, f, self.nav.na)])
                prev = self._streak.get((s, f))
                count = (prev[1] + 1 if prev is not None
                         and abs(prev[0] - val) <= self.streak_tol else 1)
                self._streak[(s, f)] = (val, count)
                if (s, f) not in self._committed and count >= self.hold_confirm_n:
                    res = resid_by_satfreq.get((s, f), 0.0)
                    if self.hold_valpos_thres > 0 and res > self.hold_valpos_thres:
                        continue  # this candidate's own residual too large; streak
                                  # is kept (not reset), just not committed yet
                    # WP13j (TASK_M10.md) layer (c): refuse to promote this
                    # candidate into a permanent hold if ITS OWN satellite
                    # is NLOS-flagged THIS epoch -- the independent,
                    # ambiguity-blind signal WP13i's post-fit DD-PR cross-
                    # check could not be (multipath corrupts PR+CP
                    # together; ray-tracing does not). Streak is kept (not
                    # reset), same as the hold_valpos_thres reject above --
                    # a later epoch where this sat is no longer NLOS-
                    # flagged can still commit it.
                    if self.nlos_commit_reject and s in nlos_sats:
                        self.nlos_commit_rejected += 1
                        continue
                    self._committed[(s, f)] = val
                    newly_committed.append((s, f))
        # else: leave self._streak untouched -- a single ratio-bar-failing
        # epoch doesn't erase an otherwise-converging streak, it just
        # isn't extended this epoch.

        if self.nav.armode == 3:
            # cssrlib fork delta: `holdamb(xa)` (Kalman re-update) does not
            # exist here; `holdamb_flags()` is the fork's documented
            # replacement for pipelines that own nav.x/P externally (us).
            # Kept for its diagnostic n_held count; downstream WP13c logic
            # never reads nav.fix==3 (see docstring above).
            self.holdamb_flags()
            self._inject_hold(xa, newly_committed)

        # Reporting gate: Q=4 requires THIS epoch's own joint ratio to
        # clear the (possibly raised) bar -- ratio is inherently a joint
        # quantity (LAMBDA's ratio test is never per-satellite), so this
        # is the right level to gate the label at -- AND, optionally
        # (min_committed_for_fix>0), a minimum number of this epoch's
        # active candidates must already be validated holds. Default 0
        # makes this purely ratio_ok, i.e. the direct in-our-own-code
        # replacement for the no-op rtkpos.valpos: reject (report float)
        # this SPECIFIC epoch without touching self._committed /
        # self._streak either way.
        n_active_committed = sum(1 for k in candidates if k in self._committed)
        # WP13p (d): held-pool diagnostics (report-only; tc/'s LAMBDA
        # pool is ~90% held ambiguities -- this measures ours per epoch).
        self._last_ar_ncand = len(candidates)
        self._last_ar_ncommit = n_active_committed
        report_ok = ratio_ok and n_active_committed >= self.min_committed_for_fix
        # WP13s literal: tc/'s ar_ddpr_xvalidate runs AFTER run_ar's
        # holds -- a reject downgrades the REPORT only (ed.nb=0,
        # smode=5; stage.py ~420), holds stay.
        if _lit and _xval_reject:
            report_ok = False
            self._last_nb = 0
        # WP13r: conditioned-world float-quality REPORT veto (see
        # __init__'s COND_FIX_RES_MAX comment) -- report-only: streaks/
        # commits/holds proceed unchanged (a first 0.5 m-bar variant
        # that also blocked commits collapsed run1 hard-window fix% 9.2
        # -> 2.6 by never letting the pool form). CP-folded signal on
        # purpose: the committed pool's CP residuals are exactly what a
        # float excursion inflates.
        # Signal = MEDIAN of THIS epoch's DD-CP residuals (CP-only, from
        # `_compute_main_dd_res`'s same pass), not the CP+PR RMS/per-sat
        # map: a float excursion lifts every CP row together (common
        # mode -> median rises), canyon multipath spikes a FEW rows
        # (median stays low) -- the same max-vs-median discriminator
        # tc/'s `_ddpr_multipath_dominated` uses, applied in reverse.
        # (An RMS-1.0m variant vetoed every canyon fix -- r4 probe; a
        # CP+PR per-sat-median 0.4m variant vetoed nearly ALL fixes
        # because the folded map is dominated by metre-scale PR noise
        # -- r5 probe.)
        if (report_ok and self.cond_fix_res_max > 0 and self.cond_hold
                and getattr(self, 'phase', 1) == 2
                and getattr(self, '_main_ddcp_res_n', 0) >= 4
                and self._main_ddcp_res_med > self.cond_fix_res_max):
            report_ok = False
            self.cond_fix_res_reject_count += 1
        # WP13r: report-only nb floor under conditioning (see __init__'s
        # COND_MIN_NB comment) -- streaks/commits/holds proceed.
        if (report_ok and self.cond_min_nb > 0 and self.cond_hold
                and getattr(self, 'phase', 1) == 2
                and nb < self.cond_min_nb):
            report_ok = False
            self.cond_min_nb_reject_count += 1
        # WP13j (TASK_M10.md) layer (c), second half: refuse the WHOLE
        # epoch's Q=4 report (not just future commits) when a MAJORITY of
        # this epoch's active candidates are NLOS-flagged -- "a candidate
        # fix that depends on NLOS satellites" per the task spec, applied
        # at the reporting level so an already-wrong-but-committed hold
        # from before NLOS coverage existed can't keep reporting Q=4
        # epoch after epoch either.
        if report_ok and self.nlos_commit_reject and candidates:
            n_nlos_active = sum(1 for (s, _f) in candidates if s in nlos_sats)
            frac = n_nlos_active / len(candidates)
            if frac > self.nlos_report_frac or (self.nlos_report_frac <= 0.0 and n_nlos_active > 0):
                report_ok = False
                self.nlos_report_rejected += 1
        if report_ok:
            self.nav.smode = 4
        # else: leave nav.smode as process() set it before calling
        # _do_ar (5 = float) -- an honest float report, no cssrlib state
        # mutated to get there.
        # WP13l: feed NEXT epoch's `prev_was_flt` (see top of this
        # function) -- tc/'s own `prev_smode` is exactly this epoch's
        # final reported smode, captured before next epoch resets it.
        self._last_report_was_fix = bool(report_ok)
        # WP13s literal: candidates consumed by `_tc_literal_postprocess`
        # (fix-streak update / suspicious-held release on FLT).
        self._last_candidates = list(candidates) if _lit else []

        self._check_release(candidates, resid_by_satfreq)

    def _inject_hold(self, xa, newly_committed):
        """Inject a ONE-TIME tight prior for each newly-committed
        (sat,freq) (WP13c: `self._committed`, never cssrlib's
        `nav.fix==3` -- see `_do_ar`'s docstring for why). Unlike
        WP13a's original `_inject_hold` (which re-injected a fresh prior
        for every CURRENT `nav.fix==3` ambiguity on EVERY epoch it
        passed), each (sat,freq) gets exactly one prior, ever -- avoiding
        compounding re-affirmation of a value that later turns out
        wrong (a mechanism the WP13a port had: a stable-but-wrong,
        e.g. multipath-consistent, integer that keeps re-passing the
        ratio test would previously get ANOTHER tight prior stacked on
        top of itself every single epoch, making it progressively
        harder, not easier, to ever recover from)."""
        if not newly_committed:
            return
        if self.tc_literal and self.per_epoch_n and self.cond_hold:
            # WP13s N4: tc/'s phase-2 fix-and-hold adds NO hold-prior
            # factors at all (`_apply_fix_and_hold`: hg stays empty when
            # fix_pose_anchor_sigma=0) -- the hold IS the next epoch's
            # folded-constant DDCP factors + the conditioned mirror.
            return
        graph = gtsam.NonlinearFactorGraph()
        for (s, f) in newly_committed:
            k = self.amb_keys.get((s, f))
            if k is None:
                continue
            graph.addPriorDouble(
                k, xa[self.IB(s, f, self.nav.na)],
                gtsam.noiseModel.Isotropic.Sigma(1, np.sqrt(self.VAR_HOLDAMB)))
        if graph.size() > 0:
            try:
                if self.smoother:
                    self.smoother.update(graph, gtsam.Values(),
                                          gtsam.FixedLagSmootherKeyTimestampMap())
                elif self.isam:
                    self.isam.update(graph, gtsam.Values())
            except RuntimeError:
                pass

    def _check_release(self, candidates, resid_by_satfreq):
        """Hold-release / dirty-reset (WP13c step 4; mirrors the intent
        of inuex35's TC-FGO `cp_hold` dirty-reset, our own code, no
        cssrlib edit): if a SPECIFIC committed (sat,freq)'s own post-fit
        CP residual stays large for `release_k` consecutive epochs where
        it's an active candidate, release just THAT ambiguity back to
        float -- bump its generation counter (so `self.N` mints a fresh
        gtsam key -- the old key's tight prior can't be un-injected, see
        `N`'s docstring), drop it from `self.amb_keys` (so
        `_build_dd_factors` re-initializes it from scratch next time it's
        seen) and reset its `nav.x` slot to 0. Per-(sat,freq) scoped (not
        a global reset) since a diagnostic run showed a single
        persistently-wrong ambiguity does not imply the rest of the held
        set is also wrong. Disabled when `release_k<=0` (default)."""
        if self.release_k <= 0:
            return
        cur = set(candidates)
        for key in list(self._bad_streak_by_key):
            if key not in cur:
                del self._bad_streak_by_key[key]  # not a candidate this epoch; don't accrue
        for key in list(self._committed):
            if key not in cur:
                continue
            res = resid_by_satfreq.get(key)
            bad = res is not None and res > self.release_thres
            n = (self._bad_streak_by_key.get(key, 0) + 1) if bad else 0
            if n == 0:
                self._bad_streak_by_key.pop(key, None)
            else:
                self._bad_streak_by_key[key] = n
            if n >= self.release_k:
                s, f = key
                self._release_ambiguity(s, f, reason='release_k')

    def _release_ambiguity(self, sat, freq, reason=''):
        """Abandon (sat,freq)'s current gtsam key (shared by WP13c's
        residual-triggered `_check_release` dirty-reset and WP13d's
        FDE-triggered cycle-slip reset, `_fde_reset_rejected_amb` -- same
        mechanism, two different triggers): bump the generation counter
        (so `self.N` mints a fresh, never-reused gtsam key -- an
        already-injected prior can't be un-injected, see `N`'s
        docstring), drop it from `self.amb_keys` (so `_build_dd_factors`
        re-initializes it from scratch next time that satellite is
        seen), reset its `nav.x` slot, and release any WP13c hold /
        streak progress on it. Does NOT touch any OTHER (sat,freq)'s
        held state -- the task's guardrail ("held set must survive
        [FDE] unless the CP factor is explicitly rejected-as-slip").

        WP13p: `reason` is print-only churn-triage metadata (one line
        per release when WP13P_DBG_CHURN is set, using the per-epoch
        el/CNR context map `self._churn_dbg_ctx`) -- zero behavior."""
        key = (sat, freq)
        if os.environ.get('WP13P_DBG_CHURN'):
            ctx = self._churn_dbg_ctx or {}
            _elm = ctx.get('el', {})
            _cnrm = ctx.get('cnr', {})
            _el_s = _elm.get(int(sat))
            _cnr_s = _cnrm.get((int(sat), int(freq)))
            try:
                _sys_i, _prn = sat2prn(int(sat))
                _sys_i = int(_sys_i)
            except Exception:
                _sys_i, _prn = -1, int(sat)
            _init = self._amb_init_epoch.get(key)
            _ep = ctx.get('ep', -1)
            print(f"WP13P_CHURN ep={_ep} trig={reason or 'other'} "
                  f"sat={int(sat)} sys={_sys_i} prn={_prn} f={int(freq)} "
                  f"el={_el_s if _el_s is None else f'{_el_s:.1f}'} "
                  f"cnr={_cnr_s if _cnr_s is None else f'{_cnr_s:.1f}'} "
                  f"held={1 if key in self._committed else 0} "
                  f"age={-1 if _init is None else _ep - _init}")
        # WP13q: tc/-literal held-release re-seed pin (see __init__) --
        # a SOFT release (tc/'s `release_hold(seed=True)` trigger set:
        # FDE-CP reject / release_k) of a COMMITTED key remembers its
        # held value for the next fresh seed; every HARD trigger
        # (slip/mw/cmc/outage/sanity -- tc/'s `clear_hold()` set)
        # forgets any pending pin for the key instead.
        if self.release_seed_held:
            # WP13s: 'flt_release' = tc/'s `_release_suspicious_held_on_
            # flt` (release_hold(seed=True) -- a SOFT, seeded release).
            _soft = (('fde_cp_ref', 'release_k', 'flt_release')
                     if self.release_seed_ref_only
                     else ('fde_cp', 'fde_cp_ref', 'release_k',
                           'flt_release'))
            _pool_ok = (self.release_seed_max_pool <= 0
                        or len(self._committed) <= self.release_seed_max_pool)
            if reason in _soft and key in self._committed and _pool_ok:
                _val = float(self._committed[key])
                if np.isfinite(_val):
                    self._release_seed_pending[key] = (
                        _val, int(getattr(self, 'epoch2', 0)))
                    self.release_seed_stored += 1
            else:
                self._release_seed_pending.pop(key, None)
        self._amb_gen[key] = self._amb_gen.get(key, 0) + 1
        self.amb_keys.pop(key, None)
        self.initx(0.0, 0.0, self.IB(sat, freq, self.nav.na))
        self._committed.pop(key, None)
        self._streak.pop(key, None)
        self._bad_streak_by_key.pop(key, None)
        # WP13n piece 4: also break the per-epoch continuity chain so the
        # next build re-seeds fresh (tc/'s slip_detect pops prev_amb via
        # amb_key=None + amb_gen bump).
        prev_vals = getattr(self, '_prev_amb_values', None)
        if prev_vals is not None:
            prev_vals.pop(key, None)

    # ---- WP13i (TASK_M9.md): post-fit DD-PR cross-check in the AR accept
    # gate, transcribed from tc/'s `optimize/stage.py` `_run_ar_with_
    # marginals` ~line 391-424 (`ar_ddpr_xvalidate_thresh`/
    # `_delta_thresh`). Only meaningful for `GtsamRtkTc` (Phase 2,
    # Pose3+lever) -- gated internally on `self.phase == 2`, a no-op for
    # the base `GtsamRtk` (Point3) float this method is nonetheless
    # defined on (mirrors `_do_ar`'s own shared-base-class placement).

    def _ddpr_ar_xvalidate(self, xa):
        """Ambiguity-INDEPENDENT cross-check: DD-PR residuals depend only
        on position, never on ambiguity values, so they can catch a
        LAMBDA fix whose own (LAMBDA-refined) implied position `xa[0:3]`
        disagrees with the raw pseudoranges -- independent of whatever
        the ratio test says. Targets WP13h's disclosed false-fix increase
        (NHC/ZUPT tightening the joint-marginal covariance the ratio
        test's denominator uses, without the underlying integer
        resolution actually becoming more often correct): reject (return
        True) when this epoch's DD-PR residual AT `xa[0:3]` exceeds
        `ar_ddpr_xvalidate_thresh` (absolute, meters) or has worsened by
        more than `ar_ddpr_xvalidate_delta` versus `self.
        last_main_ddpr_rms` (the pre-AR float DD-PR residual, tc/'s own
        `_cached_ddpr_res_pre`/`main_res_pre_fde`). The caller (`_do_ar`)
        treats a True return exactly like `nb<=0` -- an early return that
        never touches `self._committed`/`self._streak` (WP13c's ddidx-
        strips-holds pitfall: only `_check_release` may remove a hold)."""
        if getattr(self, 'phase', 1) != 2:
            return False
        if self.ar_ddpr_xvalidate_thresh <= 0 and self.ar_ddpr_xvalidate_delta <= 0:
            return False
        if not self._last_ddpr_sat_tags or self.current_estimate is None:
            return False
        try:
            key_x = self.PX(self.epoch2)
            cur_pose = self.current_estimate.atPose3(key_x)
            R_body_to_ecef = self.ecef_T_nav.compose(cur_pose).rotation().matrix()
            antenna_ecef_xa = np.asarray(xa[0:3], dtype=float)
            body_ecef_xa = antenna_ecef_xa - R_body_to_ecef @ self.lever_arm
            body_nav_xa = self.ecef_T_nav.transformTo(gtsam.Point3(*body_ecef_xa))
            xa_pose = gtsam.Pose3(cur_pose.rotation(), body_nav_xa)
        except RuntimeError:
            return False
        res_xa = self._ddpr_res_at_pose(xa_pose)
        if res_xa is None:
            return False
        res_pre = self.last_main_ddpr_rms
        reject = False
        if self.ar_ddpr_xvalidate_thresh > 0 and res_xa > self.ar_ddpr_xvalidate_thresh:
            reject = True
        if (self.ar_ddpr_xvalidate_delta > 0 and res_pre > 0
                and (res_xa - res_pre) > self.ar_ddpr_xvalidate_delta):
            reject = True
        if reject:
            self._ar_ddpr_xvalidate_reject_count += 1
            if os.environ.get('WP13I_DBG_XVALIDATE'):
                print(f"DBG_XVALIDATE ep={self.epoch2} res_xa={res_xa:.2f} "
                      f"res_pre={res_pre:.2f} thresh={self.ar_ddpr_xvalidate_thresh} "
                      f"delta_thresh={self.ar_ddpr_xvalidate_delta}")
        return reject

    def _ddpr_res_at_pose(self, pose):
        """Literal PR-only DD residual RMS at an arbitrary Pose3 (not
        necessarily the current graph estimate) -- feeds `_ddpr_ar_
        xvalidate`. Uses raw `evaluateError()`, not `.error()` (robust-
        safe -- see `_compute_main_dd_res`'s adaptation-2 docstring for
        why `.error()` on a Huber-`Robust`-wrapped factor under-reports
        exactly the large residuals this cross-check needs to see;
        `HUBER_PR` defaults off so this rarely matters in practice, but
        costs nothing to get right)."""
        factors_all = (self.smoother.getFactors() if self.smoother
                       else self.isam.getFactorsUnsafe())
        res_sq = []
        for fi, _ref, _j, _f in self._last_ddpr_sat_tags:
            fac = factors_all.at(fi)
            if fac is None:
                continue
            try:
                r = float(fac.evaluateError(pose)[0])
            except RuntimeError:
                continue
            res_sq.append(r * r)
        return float(np.sqrt(np.mean(res_sq))) if res_sq else None

    # ---- WP13d (TASK_M4.md): GICI-style FDE, transcribed from inuex35's
    # `validation/postfit.py` -- `apply_fde` + `_fde_collect_residuals` +
    # `_fde_pick_rejects_single_pass`/`_fde_pick_rejects_iterative` +
    # `_fde_reset_rejected_amb`. See the module docstring for the
    # disclosed `nf_before`-vs-`nf_total - g3.size()` adaptation.

    def _globalize_epoch_ddpr_tags(self, fi_start, graph_size):
        """Convert this epoch's LOCAL DD-PR/DD-CP factor-index tags
        (`_epoch_ddpr_tags_local`/`_epoch_ddcp_tags_local`, populated by
        `_build_dd_factors`/`_build_dd_factors_arm`) into GLOBAL smoother
        factor indices, `[fi_start, fi_start+graph_size)`. Shared by
        `_apply_fde` (WP13d) and WP13g's own `main_ddpr_res`/
        `_compute_res_at_pred` (both need `self._last_ddpr_sat_tags`
        globalized the same way, so this is factored out once instead of
        duplicated)."""
        del graph_size  # kept in the signature for readability; local
                        # tags already carry their own within-epoch index
        self._last_ddpr_sat_tags = [
            (fi_start + li, r, j, f) for li, r, j, f in self._epoch_ddpr_tags_local]
        self._last_ddcp_meta = {
            fi_start + li: (r, j, f) for li, r, j, f in self._epoch_ddcp_tags_local}

    def _fde_collect_residuals(self, factors_all, fi_start, fi_end, estimate):
        """Collect (fi, residual_in_meters) for this epoch's DD-PR/DD-CP
        factors (transcribed from `_fde_collect_residuals`). Our DD
        factors are the real gtsam C++ classes (not a CustomFactor
        wrapper), so a plain type-name check is sufficient to classify
        each -- unlike the reference, which additionally special-cases
        a `custom_cp_global` id-set because some of ITS DD-CP factors
        are CustomFactor-wrapped and don't carry 'CarrierPhase' in their
        type name."""
        pr_entries = []
        cp_entries = []
        for fi in range(fi_start, fi_end):
            fac = factors_all.at(fi)
            if fac is None:
                continue
            fname = type(fac).__name__
            try:
                err = fac.error(estimate)
            except RuntimeError:
                continue
            if 'Pseudorange' in fname:
                res_m = float(np.sqrt(2.0 * err)) * self.sigma_pr * np.sqrt(2)
                pr_entries.append((fi, res_m))
            elif 'CarrierPhase' in fname or fi in self._last_ddcp_meta:
                # WP13s N4: held-folded DDCP factors are CustomFactors
                # ('CustomFactor' type name) -- classified via the
                # per-epoch DDCP index tags, exactly tc/'s
                # `custom_cp_global` special case in its own
                # `_fde_collect_residuals`.
                res_m = float(np.sqrt(2.0 * err)) * self.sigma_cp * np.sqrt(2)
                cp_entries.append((fi, res_m))
        return pr_entries, cp_entries

    def _fde_pick_rejects_single_pass(self, pr_entries, cp_entries, pr_median, cp_median):
        """Single-pass FDE (default, `fde_max_iter=1`): collect every PR
        + CP entry whose |residual - median| exceeds its threshold, in
        one batch (transcribed from `_fde_pick_rejects_single_pass`)."""
        reject_fi = []
        for fi, res in pr_entries:
            if abs(res - pr_median) > self.fde_pr:
                reject_fi.append(fi)
        for fi, res in cp_entries:
            if abs(res - cp_median) > self.fde_cp:
                reject_fi.append(fi)
        return reject_fi

    def _fde_pick_rejects_iterative(self, pr_entries, cp_entries, pr_median, cp_median):
        """Iterative FDE (`fde_max_iter>1`): pick only the SINGLE largest
        outlier across PR and CP this pass (transcribed from
        `_fde_pick_rejects_iterative`); `_apply_fde` re-evaluates
        residuals from scratch and calls this again next pass, up to
        `fde_max_iter` times."""
        best_d = 0.0
        best_fi = None
        for fi, res in pr_entries:
            d = abs(res - pr_median)
            if d > self.fde_pr and d > best_d:
                best_d, best_fi = d, fi
        for fi, res in cp_entries:
            d = abs(res - cp_median)
            if d > self.fde_cp and d > best_d:
                best_d, best_fi = d, fi
        return [best_fi] if best_fi is not None else []

    def _fde_reset_rejected_amb(self, reject_fi):
        """Treat every rejected DD-CP factor as a cycle slip (transcribed
        from `_fde_reset_rejected_amb`, which releases BOTH satellites
        tied to the factor via a generic `fac.keys()` loop).

        Disclosed adaptation (WP13D_REPORT.md): only release the
        NON-reference satellite (`j_sat`), not the shared reference
        satellite (`ref_sat`). Our per-system reference satellite (max
        elevation) backs EVERY DD pair of that constellation this epoch
        (empirically 4-14 pairs/reference on this dataset) -- resetting
        it because ONE paired non-reference satellite looked like an
        outlier conflates a single-satellite fault with a whole-system
        one, and empirically cascades: every other still-good pair
        sharing that reference is forced to re-anchor against a freshly
        re-initialized (unconverged) reference ambiguity next epoch,
        which then itself tends to look like an outlier, repeating
        forever (`_committed` collapses to 0 and never recovers, then
        the smoother eventually hits `IndeterminantLinearSystemException`
        and permanently wedges). Releasing only `j_sat` still delivers
        the task's core intent (a specific flagged satellite's bad
        ambiguity gets a fresh start next epoch) without destabilizing
        every other satellite that happens to share the same anchor.
        Rejected DD-PR factors (`fi` not in `_last_ddcp_meta`) carry no
        ambiguity -- nothing to reset."""
        released_refs = set()
        for fi in reject_fi:
            meta = self._last_ddcp_meta.get(fi)
            if meta is None:
                continue
            _ref_sat, j_sat, freq = meta
            self._release_ambiguity(j_sat, freq, reason='fde_cp')
            # WP13p (f): tc/-literal both-sat release (see __init__) --
            # the reference's fac.keys() loop releases the shared ref N
            # too, giving a poisoned reference ambiguity a fresh start
            # instead of letting it fail every pair against it forever.
            if (self.fde_release_ref
                    and (_ref_sat, freq) not in released_refs
                    and (_ref_sat, freq) in self.amb_keys):
                self._release_ambiguity(_ref_sat, freq, reason='fde_cp_ref')
                released_refs.add((_ref_sat, freq))

    def _apply_fde(self, graph, key_x, nv, nf_before, estimate):
        """GICI-style Fault Detection and Exclusion (transcribed from
        `apply_fde`). Evaluates THIS epoch's just-committed DD-PR/DD-CP
        factors' residuals in meters and removes statistical outliers
        from the smoother via `update(..., removeFactorIndices=...)`,
        resetting (as a cycle slip) any ambiguity tied to a rejected
        DD-CP factor. Returns the (possibly updated) estimate -- unlike
        WP13c's AR gate, this changes what gets written back to
        `nav.x`/`nav.P`, which is the whole point (WP13c only relabels
        Q, never touches the float trajectory itself).

        Single-pass (`fde_max_iter=1`, the default and this task's
        starting point): collect every over-threshold entry in one
        batch. Safeguard (`fde_max_frac`, default 0.5): if a single-pass
        batch would reject more than that fraction of this epoch's `nv`
        DD factors, skip FDE entirely this epoch (mirrors the
        reference's `fde_safeguard` -- avoids nuking a whole epoch's
        worth of factors on a geometry-wide bad moment, e.g. a canyon
        bounce, where "everything looks like an outlier" is itself the
        signal that the reference frame, not individual satellites, is
        the problem).

        Iterative mode (`fde_max_iter>1`): remove only the single
        largest outlier per pass, re-evaluating residuals from scratch
        each time, up to `fde_max_iter` passes.
        """
        factors_all = (self.smoother.getFactors() if self.smoother
                       else self.isam.getFactorsUnsafe())
        fi_start = nf_before
        fi_end = fi_start + graph.size()
        # Globalize this epoch's LOCAL DD factor tags (see module
        # docstring for why `nf_before`, not the reference's own
        # `nf_total - g3.size()`, is the robust offset here).
        self._globalize_epoch_ddpr_tags(fi_start, graph.size())

        max_iter = max(1, self.fde_max_iter)
        iterative = max_iter > 1
        total_rejected = 0
        for _it in range(max_iter):
            pr_entries, cp_entries = self._fde_collect_residuals(
                factors_all, fi_start, fi_end, estimate)
            pr_median = (float(np.median([r for _, r in pr_entries]))
                         if self.fde_median_sub and pr_entries else 0.0)
            cp_median = (float(np.median([r for _, r in cp_entries]))
                         if self.fde_median_sub and cp_entries else 0.0)
            if iterative:
                reject_fi = self._fde_pick_rejects_iterative(
                    pr_entries, cp_entries, pr_median, cp_median)
                if not reject_fi:
                    break
            else:
                reject_fi = self._fde_pick_rejects_single_pass(
                    pr_entries, cp_entries, pr_median, cp_median)
                if not reject_fi:
                    break
                if len(reject_fi) > self.fde_max_frac * max(1, nv):
                    self.fde_epochs_safeguard_skipped += 1
                    # WP13s literal: tc/'s fde_safeguard fires
                    # trigger_cp_hold(skip_if_active=True) -- arm the
                    # recovery CP-hold window (+ sat-quality clear)
                    # only when no hold is already active.
                    if (self.tc_literal and self.recov_cp_hold_epochs > 0
                            and self._recov_cp_hold_remaining <= 0):
                        self.recov_cp_hold_trigger_count += 1
                        self._recov_cp_hold_remaining = \
                            self.recov_cp_hold_epochs
                        self._recov_cp_release_streak = 0
                        if self._sq is not None:
                            self._sq.clear()
                    self._log_main_ddpr(factors_all, fi_start, fi_end, estimate)
                    return estimate
                # Disclosed adaptation (WP13D_REPORT.md): this is a
                # GNSS-only graph (no IMU factor providing redundant
                # connectivity across epochs, unlike the reference's
                # tightly-coupled pipeline) -- removing DD factors down
                # to fewer than the SAME `nv>=4` floor `process()` already
                # uses to decide an epoch is well-posed empirically
                # produces an underconstrained X(ep) a few epochs later
                # (`gtsam.IndeterminantLinearSystemException`) that
                # permanently wedges the IncrementalFixedLagSmoother (every
                # subsequent update() throws "invalid map<K,T> key" -- no
                # recovery). Extending the existing fraction-based
                # safeguard with this absolute floor is a minimal,
                # motivated adaptation, not a threshold hack.
                if nv - len(reject_fi) < 4:
                    self.fde_epochs_safeguard_skipped += 1
                    self._log_main_ddpr(factors_all, fi_start, fi_end, estimate)
                    return estimate

            self._fde_reset_rejected_amb(reject_fi)
            total_rejected += len(reject_fi)
            removal_fi = (reject_fi if self.fde_remove_cp else
                          [fi for fi in reject_fi if fi not in self._last_ddcp_meta])
            if removal_fi:
                try:
                    if self.smoother:
                        self.smoother.update(gtsam.NonlinearFactorGraph(), gtsam.Values(),
                                              gtsam.FixedLagSmootherKeyTimestampMap(), removal_fi)
                        est_fde = self.smoother.calculateEstimate()
                    else:
                        self.isam.update(gtsam.NonlinearFactorGraph(), gtsam.Values(), removal_fi)
                        self.isam.update()
                        est_fde = self.isam.calculateEstimate()
                    if est_fde.exists(key_x):
                        estimate = est_fde
                except (RuntimeError, IndexError):
                    break
            if not iterative:
                break
            factors_all = (self.smoother.getFactors() if self.smoother
                           else self.isam.getFactorsUnsafe())

        if total_rejected:
            self.fde_total_rejected += total_rejected
            self.fde_epochs_rejected += 1
            self.last_fde_reject = total_rejected
        else:
            self.last_fde_reject = 0
        self._log_main_ddpr(factors_all, fi_start, fi_end, estimate)
        return estimate

    def _log_main_ddpr(self, factors_all, fi_start, fi_end, estimate):
        """Per-epoch DDPR residual RMS + worst satellite -- diagnostic
        monitor only (transcribed from `main_ddpr_residuals`; TASK_M4.md
        item 2, "the monitor feeding FDE and diagnostics"). For this
        port it feeds diagnostics only, evaluated AFTER FDE has already
        run (so it reports the CLEANED epoch's residual quality) -- the
        IMU-predicted-pose sanity/anchor-fallback machinery that
        consumes this upstream in the reference is out of scope here
        (module docstring)."""
        res_sq = []
        per_sat = {}
        pair_rows = []
        for fi, ref, j, _f in self._last_ddpr_sat_tags:
            if fi < fi_start or fi >= fi_end:
                continue
            fac = factors_all.at(fi)
            if fac is None:
                continue
            try:
                err = fac.error(estimate)
            except RuntimeError:
                continue
            res_m = float(np.sqrt(2.0 * max(err, 0.0)) * self.sigma_pr * np.sqrt(2))
            res_sq.append(res_m * res_m)
            if res_m > per_sat.get(ref, 0.0):
                per_sat[ref] = res_m
            if res_m > per_sat.get(j, 0.0):
                per_sat[j] = res_m
            # WP13s: tc/'s main_ddpr_residuals(with_pairs=True) rows --
            # feed sq.update_pair_quality / pair_bad_max in literal mode.
            pair_rows.append({'ref': int(ref), 'sat': int(j),
                              'freq': int(_f), 'res': res_m})
        self._last_pair_rows = pair_rows
        self.last_main_ddpr_rms = float(np.sqrt(np.mean(res_sq))) if res_sq else 0.0
        self.last_main_ddpr_worst = (
            max(per_sat.items(), key=lambda kv: kv[1]) if per_sat else None)
        if res_sq:
            self.main_ddpr_rms_sum += self.last_main_ddpr_rms
            self.main_ddpr_rms_n += 1

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        if len(obs.sat) == 0:
            return

        # Position from GTSAM's own previous estimate (not an EKF predict).
        # See __init__ docstring note: reference against last_valid_epoch,
        # not epoch-1, so a skipped epoch can't desync this from the
        # actual ISAM2 Bayes tree contents.
        if self.current_estimate is not None and self.last_valid_epoch is not None:
            prev = self.X(self.last_valid_epoch)
            pos_pred = (np.array(self.current_estimate.atPoint3(prev))
                        if self.current_estimate.exists(prev)
                        else self.nav.x[0:3].copy())
        else:
            pos_pred = self.nav.x[0:3].copy()

        prep = self.prepare_double_difference_measurements(
            obs, obsb, pos_pred=pos_pred, cs=cs, orb=orb, bsx=bsx,
            compute_zdres=False)
        if prep is None:
            return

        rs = prep['rs']
        vs = prep['vs']
        dts = prep['dts']
        rsb = prep['rsb']
        iu = prep['iu']
        ir = prep['ir']
        sat = prep['sat']
        el = prep['el']
        obs_ = prep['obs_sd']

        # Ambiguity management (no EKF)
        self._manage_ambiguities(obs_)
        ns = len(iu)
        if ns < 6:
            return

        # Factor graph
        graph = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()
        ep = self.epoch
        key_x = self.X(ep)

        if self.last_valid_epoch is None:
            graph.addPriorPoint3(key_x, gtsam.Point3(*pos_pred),
                                  gtsam.noiseModel.Isotropic.Sigma(3, self.nav.sig_p0))
        else:
            dt = (timediff(obs.t, self.last_valid_time)
                  if self.last_valid_time is not None and self.last_valid_time.time > 0
                  else 1.0)
            graph.add(gtsam.BetweenFactorPoint3(
                self.X(self.last_valid_epoch), key_x, gtsam.Point3(0, 0, 0),
                gtsam.noiseModel.Isotropic.Sigma(
                    3, self.sigma_dyn * np.sqrt(max(abs(dt), 0.1)))))
        new_values.insert(key_x, gtsam.Point3(*pos_pred))

        nv, new_amb, cp_amb = self._build_dd_factors(
            graph, new_values, obs, obsb, obs_, rs, rsb, sat, el, iu, ir, pos_pred, ep)
        if nv < 4:
            if os.environ.get('DEBUG_EXC'):
                print("DEBUG nv<4", ep, nv, "amb_keys=", len(self.amb_keys),
                      "new_amb=", len(new_amb), "cp_amb=", len(cp_amb))
            self.epoch += 1
            self.nav.t = obs.t
            return

        # ISAM2 / IFLS update
        # WP13d: capture the factor count BEFORE this epoch's own update --
        # `_apply_fde` needs this to locate exactly which factor indices are
        # THIS epoch's DD-PR/DD-CP factors (see module docstring: verified
        # empirically that our own new factors always land first, at
        # [nf_before, nf_before+graph.size()), with any lag-window
        # marginalization byproduct appended strictly after).
        nf_before = (self.smoother.getFactors().size() if self.smoother
                     else self.isam.getFactorsUnsafe().size())
        try:
            if self.smoother:
                ts = gtsam.FixedLagSmootherKeyTimestampMap()
                ts[key_x] = self.epoch_time
                for (s, f), k in _sorted_amb_items(new_amb):
                    ts[k] = self.epoch_time
                for (s, f), k in _sorted_amb_items(self.amb_keys):
                    if (s, f) not in new_amb:
                        # Refresh EVERY key still in self.amb_keys (matches
                        # the reference: ambiguities never age out of the
                        # LAG window on their own). See WP13A_REPORT.md
                        # section 3c for a reverted attempt to narrow this
                        # to "recently seen" for performance -- it caused a
                        # severe divergence at scale and was backed out.
                        ts[k] = self.epoch_time
                self.smoother.update(graph, new_values, ts)
                estimate = self.smoother.calculateEstimate()
            else:
                self.isam.update(graph, new_values)
                self.isam.update()
                estimate = self.isam.calculateEstimate()
            self.current_estimate = estimate
        except (RuntimeError, IndexError) as _DEBUG_EXC:
            if os.environ.get('DEBUG_EXC'):
                import traceback as _tb
                print("DEBUG_EXC", type(_DEBUG_EXC).__name__, _DEBUG_EXC)
                _tb.print_exc()
            # IndexError ("invalid map<K, T> key") observed from
            # IncrementalFixedLagSmoother.update() at short LAG values: an
            # ambiguity's per-epoch timestamp refresh (the loop just above)
            # is skipped on any epoch this same try block is never reached
            # (nv<4 abandon, or a prior throw here) -- with a short enough
            # LAG, even one such skipped refresh can push that ambiguity's
            # last-known timestamp outside the smoother's own lag window
            # before the next successful update, and the C++ smoother then
            # rejects its own now-stale key. Treated exactly like the
            # existing RuntimeError case: skip the epoch, do not corrupt
            # self.amb_keys/nav state, and keep last_valid_epoch anchored
            # at the last epoch that actually committed.
            self.epoch += 1
            self.epoch_time += 1.0
            self.nav.t = obs.t
            return

        # Committed: this epoch's node now exists in the ISAM2/IFLS Bayes
        # tree, so it's safe as the next epoch's between-factor anchor. Only
        # now is it safe to tell self.amb_keys that new_amb's keys actually
        # exist in the smoother -- see _build_dd_factors's note.
        self.last_valid_epoch = ep
        self.last_valid_time = obs.t
        self.amb_keys.update(new_amb)
        # NOTE: an ambiguity-pruning scheme (bound self.amb_keys to a
        # rolling window instead of "every satellite ever tracked this
        # run", to fix the superlinear per-epoch cost growth over a long
        # run) was tried and reverted here -- see WP13A_REPORT.md section
        # 3c and `_amb_key_for_UNUSED`'s docstring above: it roughly halved
        # wall-clock time but caused a severe divergence (AllRMS ~7000 m)
        # starting a few thousand epochs in, not caught by short-window
        # testing. `self.amb_keys` therefore still grows for the life of
        # the run, same as the original design -- this is a real
        # throughput cost on long runs (disclosed, not silently reverted).

        # WP13d: FDE runs AFTER the merge above (so a same-epoch freshly-
        # initialized ambiguity that FDE immediately rejects is correctly
        # dropped, not silently re-added by a merge that ran after it) and
        # BEFORE _write_back (so a cleaned/replaced `estimate` -- not the
        # original -- is what gets written into nav.x/nav.P, unlike WP13c's
        # AR gate which only ever relabels Q after nav.x is already fixed).
        if self.fde_enable:
            estimate = self._apply_fde(graph, key_x, nv, nf_before, estimate)
            self.current_estimate = estimate

        self._write_back(estimate, key_x, cp_amb)
        self.nav.smode = 5

        if self.nav.armode > 0:
            self._do_ar(obs, rs, vs, dts, sat, el, iu)

        self.nav.t = obs.t
        self.epoch += 1
        self.epoch_time += 1.0


class GtsamRtkTc(GtsamRtk):
    """WP13e (TASK_M5.md): transcribe inuex35's IMU tight coupling onto the
    standalone, to supply the stable pose their FDE (WP13d) presupposes.

    WP13a-d proved the GNSS-only float is fundamentally under-constrained
    (AllRMS 57-192m): a `BetweenFactorPoint3` process model with an
    isotropic `sigma_dyn` "prior" is not a real dynamics constraint, so
    the position node has essentially no cross-epoch backbone other than
    that epoch's own (frequently poor-geometry) DD-PR/DD-CP factors.
    inuex35's own `postfit.run_ddpr_sanity` explicitly uses the
    IMU-predicted `pred.pose()` as their sanity/recovery reference --
    i.e. the IMU chain IS the stable pose in their design, not an
    optional refinement. This class transcribes the MINIMAL slice that
    supplies it, per TASK_M5.md's explicit exclusion list (no NHC/ZUPT/
    Doppler/sat_badness/recovery FSMs -- those are later refinements):

      * State: `Pose3` (attitude+position) + `Velocity3` +
        `imuBias.ConstantBias`, keyed per epoch (mirrors tc/'s own
        `Xpose`/`Vel`/`Bias` scheme, `runner.py` lines 37-41) -- own
        namespace (`PX`/`PV`/`PB`, symbol chars 'P'/'V'/'B') in a FRESH
        ISAM2/IFLS instance, entirely separate from Phase 1's Point3
        smoother (see `_transition_to_phase2`).
      * IMU chain: `CombinedImuFactor` + `BetweenFactorConstantBias` +
        an absolute bias prior anchored to the Phase-1->2 transition's
        bias estimate every epoch (a simplified, always-on variant of
        tc/'s `bias_prior_anchor` mode 1 -- "keep it simple" per the
        task), preintegrated from `imu.csv` via tc/'s own `build_pim`
        (`utils/imu.py`, reused directly, not reimplemented).
      * DD factors: the **Arm** variants (`DoubleDifferencePseudorange
        FactorArm` / `DoubleDifferenceCarrierPhaseFactorArm`, from tc/'s
        `buildfactor/factors.py` lines ~78/251) -- take the vehicle
        `Pose3` (in a LOCAL ENU frame anchored at the base station,
        `ecef_T_nav`, matching tc/'s own `runner._init_base_frame`) +
        a body-FLU lever arm instead of a bare ECEF `Point3` -- see
        `_build_dd_factors_arm`.
      * Phase-1 init: a BRIEF (`PHASE1_EPOCHS`, default 40) GNSS-only
        bootstrap that is simply the UNCHANGED parent `GtsamRtk` (WP13a-
        d's own Point3 float/AR/FDE pipeline, verbatim, via `super().
        process()`) run for its own sake -- not to replicate tc/'s own
        multi-fix-collection Phase 1 (`initialization.py`, which
        requires actual validated LAMBDA "FIX" epochs before collecting,
        a strong assumption our much weaker standalone float cannot
        reliably satisfy early in a run). Attitude (pitch/roll from
        stationary accel tilt, heading from bootstrap displacement) and
        bias (stationary IMU average) are computed exactly as tc/'s own
        `tightly_coupled.transition_to_tc`, just from this simpler
        bootstrap's position history instead of their `collected_fixes`
        list -- see `_transition_to_phase2`.

    WP13c's accept-gate (`_do_ar`/`_inject_hold`/`_check_release`) and
    WP13d's FDE (`_apply_fde` + helpers) are inherited COMPLETELY
    UNCHANGED: both only ever read/write `nav.x`/`nav.P`/`self.amb_keys`
    and evaluate DD factor residuals via `fac.error(estimate)` + a
    `type(fac).__name__` substring check -- none of that cares whether
    the position node backing `nav.x` is a bare `Point3` or a `Pose3`'s
    translation (+lever arm), and the Arm factor classes still carry
    'Pseudorange'/'CarrierPhase' in their type names. The ONLY new glue
    is `_write_back_tc` (this class), which is the Pose3-aware analogue
    of the parent's `_write_back` -- everything downstream of nav.x/P is
    untouched inherited code.
    """

    def __init__(self, nav, imu_data, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, pos0, logfile)
        self.imu_data = imu_data
        self.imu_idx = 0

        self.phase = 1
        self.phase1_epochs = int(os.environ.get('PHASE1_EPOCHS', '40'))
        self._phase1_hist = []  # [(obs.t, ecef), ...] committed Phase-1 epochs

        lever_str = os.environ.get('LEVER_ARM', '0.31,0,0.55')
        self.lever_arm = np.array([float(x) for x in lever_str.split(',')])
        self.lever_pt = gtsam.Point3(*self.lever_arm)

        self.imu_params = make_imu_params(
            accel_noise=float(os.environ.get('IMU_ACCEL_NOISE', '2.84e-4')),
            gyro_noise=float(os.environ.get('IMU_GYRO_NOISE', '4.01e-5')),
            accel_bias_sigma=float(os.environ.get('IMU_ACCEL_BIAS_SIGMA', '3.14e-4')),
            gyro_bias_sigma=float(os.environ.get('IMU_GYRO_BIAS_SIGMA', '9.70e-6')),
            scale=float(os.environ.get('IMU_SCALE', '1.0')),
            integ_cov=float(os.environ.get('IMU_INTEG_COV', '1e-3')))
        # tc/'s own TcConfig defaults (config.py lines 250-253) -- not
        # swept, task says "keep it simple".
        self.bias_between_acc_sigma = float(os.environ.get('BIAS_BETWEEN_ACC_SIGMA', '3e-4'))
        self.bias_between_gyro_sigma = float(os.environ.get('BIAS_BETWEEN_GYRO_SIGMA', '3e-5'))
        self.bias_prior_acc_sigma = float(os.environ.get('BIAS_PRIOR_ACC_SIGMA', '3e-3'))
        self.bias_prior_gyro_sigma = float(os.environ.get('BIAS_PRIOR_GYRO_SIGMA', '3e-4'))
        # 0=off, 1=anchor to the Phase-1->2 transition's initial bias
        # estimate EVERY epoch (tc/'s own `bias_prior_mode=1`), 2=anchor
        # to the PREVIOUS epoch's own committed bias estimate every
        # epoch (tc/'s own DEFAULT, `bias_prior_mode=2` -- a soft
        # regularizer that discourages a big epoch-to-epoch bias jump,
        # but does NOT forever pin the bias close to a possibly-wrong
        # initial stationary-window estimate the way mode 1 does).
        self.bias_prior_mode = int(os.environ.get('BIAS_PRIOR_MODE', '2'))

        # Bug fix (WP13f/TASK_M6.md item 3, "float quality"): a per-epoch
        # dump localized a recurring few-meter STEP jump in the Phase-2
        # float trajectory to the exact epoch a brand-new (sat,freq)
        # ambiguity is first seeded (nv jumping e.g. 14->16->18 within 1-2
        # epochs, `bias_acc` simultaneously jumping ~50% and the ECEF pose
        # oscillating by several meters epoch-to-epoch) -- root cause: a
        # fresh ambiguity's initial value is seeded from THIS epoch's own
        # (possibly several-meter-off, since Phase 2 has no NHC/ZUPT
        # backbone between GNSS updates) `pos_pred`, then immediately used
        # in a `DoubleDifferenceCarrierPhaseFactorArm` with the SAME tiny
        # sigma_cp (~4mm) a converged, correct ambiguity needs for cm-level
        # accuracy -- so a wrong-by-meters seed is trusted just as tightly
        # as a right one, and the optimizer drags the POSE (not just the
        # ambiguity) to satisfy it. `HUBER_CP` (0=off, matches the existing
        # `HUBER_PR`/`self.huber_pr` pattern already used for DD-PR) caps a
        # single bad/fresh CP factor's influence to a linear, not
        # quadratic, penalty beyond its threshold -- applied ONLY in this
        # Phase-2 Arm DD-CP path (`_build_dd_factors_arm`), not the
        # Phase-1/parent Point3 path, which already performs at tc/'s own
        # reference quality (~0.03m) and is left untouched.
        self.huber_cp = float(os.environ.get('HUBER_CP', '0'))
        # WP17 (TASK_M23 item 2, B1 root-fix): SEED_N_CYCLES=1 seeds a
        # FRESH phase-2 ambiguity in CYCLES, N0 = (cp - pr)/lam, in the
        # NON-literal Arm builder, matching the Arm factor's own model
        # (error = ddModel + lam*(Nref-Nj) - dd_obs,
        # CarrierPhaseFactor.cpp:417 -- N is in cycles). The non-literal
        # port seeded METERS since WP13e (WP13s audit B1 / WP13f
        # "fresh-seed step jump"), and BOTH the meters seed and the
        # naive cycles seed (cp - geometric range)/lam carry the epoch's
        # SD receiver-clock term, so seeds minted at different epochs
        # disagree by the clock RAMP -- under HUBER_CP>0 the robustified
        # CP factor cannot snap the thousands-of-cycles-off-lattice seed
        # (it crawls ~16 cyc/epoch), which is why WP16 had to run the
        # shared filter with HUBER_CP=0. Anchoring on the SAME epoch's
        # pseudorange cancels the clock per-seed. The release-seed pin
        # sigma drops its x lam_f meter conversion accordingly (the
        # pinned value itself is the committed graph estimate, which the
        # Arm factor already keeps in cycles). Phase-1 (Point3 builder)
        # untouched. Default 0 = every prior config bit-identical.
        self.seed_n_cycles = int(os.environ.get('SEED_N_CYCLES', '0'))

        self.tc_bias_init = None   # set at _transition_to_phase2
        self.ecef_T_nav = None     # fixed Pose3, base-anchored local-ENU -> ECEF
        self.R_enu2ecef = None
        self.epoch2 = 0            # Phase-2's own epoch counter/key namespace

        # WP13g (TASK_M7.md): DDPR-sanity / pose-innovation check,
        # transcribed from inuex35's `validation/postfit.py`
        # `run_ddpr_sanity` family (`_ddpr_sanity_trigger` +
        # `_ddpr_sanity_persist` + `_ddpr_sanity_fast_path` +
        # `_sanity_report_translation` + `_compute_res_at_pred`; the
        # `_ddpr_sanity_fetch_anchor`/`_ddpr_sanity_anchor_vs_imu` DDPR-
        # only-LS-anchor refinement stages are OUT OF SCOPE here -- this
        # standalone has no `_ddpr_only_position` solver, and TASK_M7.md's
        # own numbered item list doesn't ask for them -- so a persisted
        # trigger goes straight from `_ddpr_sanity_persist` to the same
        # reset+report action the reference's own anchor-FALLBACK path
        # uses when its anchor is unavailable/untrusted, i.e.
        # `_ddpr_sanity_anchor_fallback` -> `_apply_sanity_reset`).
        # Defaults are the reference's own (config.py / the
        # `tokyo_mode2_satbad_cponly` preset, which is itself tuned for
        # THIS dataset): sanity_pose_replace_thresh/main_ddpr_res_
        # catastrophic/ddpr_fast_worst_sat_min come from that Tokyo
        # preset (5.0 / 15.0 / 10.0); ddpr_sanity_persist/main_ddpr_res_
        # thresh are unmodified by the preset (3 / 3.0).
        self.sanity_enable = int(os.environ.get('SANITY_ENABLE', '1'))
        self.sanity_pose_replace_thresh = float(
            os.environ.get('SANITY_POSE_REPLACE_THRESH', '5.0'))
        self.main_ddpr_res_catastrophic = float(
            os.environ.get('MAIN_DDPR_RES_CATASTROPHIC', '15.0'))
        self.ddpr_sanity_persist = int(os.environ.get('DDPR_SANITY_PERSIST', '3'))
        self.main_ddpr_res_thresh = float(
            os.environ.get('MAIN_DDPR_RES_THRESH', '3.0'))
        self.ddpr_fast_worst_sat_min = float(
            os.environ.get('DDPR_FAST_WORST_SAT_MIN', '10.0'))
        # Disclosed adaptation (not in postfit.py): minimum consecutive
        # bad epochs before the CATASTROPHIC FAST PATH may act (the
        # reference's own fast path needs only 1 -- see
        # `_ddpr_sanity_check`'s docstring for the false-positive this
        # fixes, found on run2's clean milestone window).
        self.ddpr_fast_min_persist = int(os.environ.get('DDPR_FAST_MIN_PERSIST', '2'))
        # Disclosed adaptation (not in postfit.py): whether a sanity fire
        # ALSO does the full-ambiguity-wipe graph surgery
        # (`reset_ambiguities_with_cp_hold`, 1=literal reference) or only
        # the report-layer action (report FLT, possibly pose-replace, 0
        # -- see `_ddpr_sanity_check`'s docstring for why this knob
        # exists: the full wipe was found to regress the ALREADY-SOLVED
        # tow 188990-189070 canyon, since this port has no NHC/ZUPT
        # backbone to bound the post-reset IMU-only drift the reference
        # relies on).
        self.sanity_do_reset = int(os.environ.get('SANITY_DO_RESET', '1'))
        self._ddpr_bad_count = 0      # consecutive-bad-epoch counter (tc._ddpr_bad_count)
        self._main_ddpr_res = 0.0     # this epoch's own pre-FDE main DDPR RMS [m]
        self._main_ddpr_worst_res = 0.0  # worst single-satellite residual this epoch [m]
        self.sanity_fire_count = 0    # diagnostic: total epochs sanity fired on
        self.sanity_fast_count = 0    # diagnostic: of those, how many via the fast path
        self.sanity_replace_count = 0  # diagnostic: of those, how many replaced the pose

        # WP13h (TASK_M8.md): NHC + ZUPT/ZARU + Doppler motion-constraint
        # backbone -- established across WP13a/c/d/f/g that adding
        # outlier/reset machinery (RELEASE_K, WP13g sanity) is net-
        # NEGATIVE on this port because it lacks the per-epoch velocity
        # constraints inuex35's own pipeline always has. Defaults mirror
        # tc/'s own TcConfig defaults (config.py lines ~312-327 / the
        # `tokyo_mode2_satbad_cponly` preset, which has nhc_enable=1).
        self.nhc_enable = int(os.environ.get('NHC_ENABLE', '1'))
        self.nhc_min_speed = float(os.environ.get('NHC_MIN_SPEED', '0.0'))
        self.nhc_sigma_lat = float(os.environ.get('NHC_SIGMA_LAT', '0.3'))
        self.nhc_sigma_vert = float(os.environ.get('NHC_SIGMA_VERT', '0.2'))
        nhc_lever_str = os.environ.get('NHC_LEVER', '0,0,0')
        self.nhc_lever = np.array([float(x) for x in nhc_lever_str.split(',')])

        self.zupt_enable = int(os.environ.get('ZUPT_ENABLE', '1'))
        self.zupt_max_acc_std = float(os.environ.get('ZUPT_MAX_ACC_STD', '0.55'))
        self.zupt_max_gyro_std = float(os.environ.get('ZUPT_MAX_GYRO_STD', '0.030'))
        self.zupt_max_gyro_median = float(os.environ.get('ZUPT_MAX_GYRO_MEDIAN', '0.020'))
        self.zupt_min_samples = int(os.environ.get('ZUPT_MIN_SAMPLES', '5'))
        self.zupt_sigma_zero_velocity = float(os.environ.get('ZUPT_SIGMA_ZERO_VELOCITY', '0.5'))
        self.zupt_sigma_zero_rotation = float(os.environ.get('ZUPT_SIGMA_ZERO_ROTATION', '0.010'))

        self.doppler_vel_sigma = float(os.environ.get('DOPPLER_VEL_SIGMA', '0.5'))
        self.doppler_max_res = float(os.environ.get('DOPPLER_MAX_RES', '2.0'))
        self._nhc_fire_count = 0
        self._zupt_fire_count = 0
        self._doppler_fire_count = 0

        # WP13n (TASK_M14.md) piece 4: tc/'s per-epoch ambiguity state
        # model (TASK_M14 target (b)) -- the largest structural
        # difference between the two float backbones. tc/ mints a NEW N
        # variable EVERY epoch (`buildfactor/factors.py` line ~476:
        # `tc.N(sat, f, dd_epoch*100 + amb_gen)` with `dd_epoch=ed.kk`)
        # and couples it to last epoch's N with (i) a continuity prior
        # at last epoch's ESTIMATED value, sigma=sigma_cont=1.0 cyc
        # (`_seed_one_amb_prior`), and (ii) a
        # `BetweenFactorDouble(k_old, k_new, 0.0)` with
        # sigma=sigma_n_between=0.01 cyc when the previous epoch
        # reported FIX / 0.1 cyc when FLT (`optimize/stage.py`
        # `_build_factor_block`, `betweenn_enable=1`,
        # `sigma_n_between_warmup=0` in tc/'s defaults+preset). Old N
        # states age out of the fls_lag=1.0 window (~5 epochs at 5Hz)
        # and are marginalized CLEANLY each epoch -- unlike this port's
        # persistent-N scheme, whose single N key is re-stamped forever
        # and accretes stale-linearization LinearContainerFactor
        # information for the whole run. `_prev_amb_values` mirrors
        # tc/'s `ed.prev_amb_tc` (preprocess/gate.py `_carry_prev_amb`):
        # rebuilt from the CURRENT estimate after every successful
        # update; an (s,f) absent one epoch drops out and re-seeds
        # fresh (sigma_amb0), exactly like tc/. Opt-in (0 = shipped
        # config bit-identical; persistent-N path untouched).
        self.per_epoch_n = int(os.environ.get('PER_EPOCH_N', '0'))
        self.sigma_cont = float(os.environ.get('SIG_CONT', '1.0'))
        self.sigma_n_between = float(os.environ.get('SIG_N_BETWEEN', '0.01'))
        self.sigma_n_between_flt = float(os.environ.get('SIG_N_BETWEEN_FLT', '0.1'))
        self._prev_amb_values = {}   # (s,f) -> (key, value) at last committed epoch
        self.n_between_count = 0
        self.n_cont_seed_count = 0
        self.n_fresh_seed_count = 0

        # WP13n piece 7: solve-exception warm reset -- tc/'s
        # `validation/recovery.py` `handle_solve_exception` /
        # `warm_reset_phase2`, reduced scope (disclosed: tc/ first tries
        # a DDPR-only-LS re-anchor, which this port has no solver for;
        # here the reset re-seeds from the IMU-predicted NavState, tc/'s
        # own fallback when the DDPR anchor is unavailable). Motivation
        # measured this WP: the piece-1 slip resets stop re-stamping
        # released persistent-N keys; the delayed batch marginalization
        # can hit `IndeterminantLinearSystemException`, and
        # IncrementalFixedLagSmoother.update is NOT transactional -- one
        # thrown update leaves ISAM2 permanently broken, after which
        # EVERY subsequent epoch throws (measured: run wedged from
        # ep~1300/ep~550, all later epochs abandoned). tc/ survives the
        # same class of failure precisely because a solve exception
        # triggers a warm reset instead of an infinite abandon loop.
        # After `solve_reset_after` CONSECUTIVE solve failures the
        # Phase-2 graph is re-seeded in place via `_seed_phase2_graph`
        # (the same machinery `--resume` already uses) + a cp_hold
        # window (tc/ arms `recov_cp_hold` on reset too). 0 = off =
        # shipped config bit-identical.
        self.solve_reset_after = int(os.environ.get('SOLVE_RESET_AFTER', '0'))
        self._solve_fail_streak = 0
        self.solve_reset_count = 0

        # WP13s: tc/'s PIM-discontinuity flag (one-shot; consumed by the
        # next epoch's IMU-chain build). Literal-mode only.
        self._pim_discontinuity = False
        self.pim_break_count = 0
        self.sanity_break_pim = 0
        self.pim_break_trans_sigma = 100.0
        # WP13s: tc/'s GDOP gate + sanity GDOP guard (literal-mode only)
        self._last_gdop = 0.0
        self.gdop_skip_count = 0
        self.sanity_max_gdop = 0.0
        self.sanity_skip_gdop_count = 0

        # WP13s (TASK_M19.md): phase-2 half of the TC_LITERAL preset --
        # runs at the very end of __init__ so it overrides every env/
        # CLI-derived attribute set above (see GtsamRtk's
        # `_apply_tc_literal_preset` docstring).
        if self.tc_literal:
            self._apply_tc_literal_preset_p2()

    def _apply_tc_literal_preset_p2(self):
        """WP13s: tc/'s tokyo preset, phase-2-scoped knobs (sanity /
        recovery / IMU-side defaults from TcConfig + the preset)."""
        # GDOP guards (preset sanity_max_gdop=5.0; gdop_max=10.0 wired
        # at the gate site with tc/'s IMPORTED compute_gdop).
        self.sanity_max_gdop = 5.0
        # IMU-chain break after sanity reset (preset sanity_break_pim=1,
        # pim_break_trans_sigma=100.0): the epoch after a reset gets NO
        # CombinedImuFactor -- pose prior at the IMU prediction with a
        # near-free translation sigma, so the pose snaps to that epoch's
        # own DD-PR solution (tc/'s re-anchor mechanism,
        # imu_preintegration.add_imu_chain).
        self.sanity_break_pim = 1
        self.pim_break_trans_sigma = 100.0
        # DDPR sanity ladder (validation/postfit.py + preset)
        self.sanity_enable = 1
        self.sanity_pr_only = 1                # tc/-literal PR-only signal
        self.sanity_max_median_ratio = 5.0     # _ddpr_multipath_dominated
        self.sanity_max_median_min_sats = 6
        self.sanity_pose_replace_thresh = 5.0
        self.main_ddpr_res_catastrophic = 15.0
        self.ddpr_sanity_persist = 3
        self.main_ddpr_res_thresh = 3.0
        self.ddpr_fast_worst_sat_min = 10.0
        # tc/'s fast path fires on the FIRST catastrophic epoch (the
        # WP13g min-persist=2 adaptation was justified against the
        # CP-folded signal, which literal mode does not use).
        self.ddpr_fast_min_persist = 1
        self.sanity_do_reset = 1
        # Recovery CP-hold (state.py trigger_cp_hold + gate.py release
        # condition, preset recov_cp_hold=5 / 2.0 m / 5 clean epochs)
        self.recov_cp_hold_epochs = 5
        self.recov_cp_release_thresh = 2.0
        self.recov_cp_release_count = 5
        # tc/ warm-resets on the FIRST solve exception
        # (validation/recovery.py handle_solve_exception).
        self.solve_reset_after = 1
        # Motion constraints: preset nhc on, TcConfig defaults
        self.nhc_enable = 1
        self.nhc_min_speed = 0.0
        self.nhc_sigma_lat = 0.3
        self.nhc_sigma_vert = 0.2
        self.zupt_enable = 1
        self.doppler_vel_sigma = 0.5
        self.doppler_max_res = 2.0
        self.huber_cp = 0.0                    # tc/ has no CP robustifier

    # ---- Phase-2 key scheme (mirrors tc/'s runner.py Xpose/Vel/Bias,
    # symbol chars 'P'/'V'/'B' -- distinct from the parent's Phase-1 'x'
    # Point3 key and the shared 'n' ambiguity key, which Phase 2 keeps
    # using via the inherited `N()`).
    def PX(self, ep):
        return gtsam.symbol('P', ep)

    def N_ep(self, sat, f, ep):
        """WP13n piece 4: per-epoch ambiguity key -- tc/'s
        `N(sat, f, dd_epoch*100 + amb_gen)` scheme (runner.py:
        `symbol('n', gen*1000000 + s*10 + f)`), with the epoch
        multiplier widened from 100 to 1000 (disclosed deviation:
        this port's slip-reset generation counter can exceed 100 over
        a full run, which under tc/'s literal *100 packing would
        collide with a neighboring epoch's key space; 15301 epochs *
        1000 * 1e6 + sat*10+f still fits a 56-bit gtsam symbol index
        with 3 orders of margin)."""
        sat = int(sat)  # np.int32 sat would OverflowError in the mixed
        f = int(f)      # numpy/python-int arithmetic below (tc/ casts too)
        gen = int(self._amb_gen.get((sat, f), 0)) % 1000
        return gtsam.symbol('n', (int(ep) * 1000 + gen) * 1000000 + sat * 10 + f)

    def PV(self, ep):
        return gtsam.symbol('V', ep)

    def PB(self, ep):
        return gtsam.symbol('B', ep)

    def _bias_between_noise(self):
        ag = self.bias_between_acc_sigma
        gy = self.bias_between_gyro_sigma
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([ag, ag, ag, gy, gy, gy]))

    def _bias_prior_noise(self):
        ag = self.bias_prior_acc_sigma
        gy = self.bias_prior_gyro_sigma
        return gtsam.noiseModel.Diagonal.Sigmas(np.array([ag, ag, ag, gy, gy, gy]))

    # ---- WP13h (TASK_M8.md): NHC + ZUPT/ZARU + Doppler backbone ----

    def _add_nhc_factor(self, graph, ep, speed, gyro_mean_rh, bias_gyro_current):
        """Non-Holonomic Constraint at the rear-axle center in the FLU
        body frame -- transcribed from tc/'s `buildfactor/nhc.py`
        `add_nhc_factor` (own method, not a direct import, to match this
        file's flat `self.*` config convention -- see the import-block
        comment above). Body lateral+vertical velocity are constrained
        to ~0 every Phase-2 epoch, evaluated at `nhc_lever`'s offset from
        the IMU origin (a per-epoch `omega x lever` correction) when a
        non-zero lever is configured; with the default `nhc_lever=
        '0,0,0'` (task's own default) this offset is always exactly zero
        and `gyro_mean_rh`/`bias_gyro_current` are unused, matching the
        reference's own dead-branch behavior at a zero lever."""
        if not self.nhc_enable or speed < self.nhc_min_speed:
            return False
        lever = self.nhc_lever
        if gyro_mean_rh is not None and np.linalg.norm(lever) > 0:
            bias_gyro = (np.asarray(bias_gyro_current)
                         if bias_gyro_current is not None else np.zeros(3))
            omega = np.asarray(gyro_mean_rh) - bias_gyro
            offset = np.cross(omega, lever)
        else:
            offset = np.zeros(3)
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([float(self.nhc_sigma_lat), float(self.nhc_sigma_vert)]))
        key_x = self.PX(ep)
        key_v = self.PV(ep)

        def error_fn(this, values, jacobians):
            pose = values.atPose3(this.keys()[0])
            v = np.array(values.atVector(this.keys()[1]))
            R = np.array(pose.rotation().matrix())
            v_body = R.T @ v + offset
            err = np.array([v_body[1], v_body[2]])
            if jacobians is not None:
                skew_v = np.array([[0, -v[2], v[1]],
                                    [v[2], 0, -v[0]],
                                    [-v[1], v[0], 0]])
                dR = (R.T @ skew_v)[1:3, :]
                jacobians[0] = np.hstack([dR, np.zeros((2, 3))])
                jacobians[1] = R.T[1:3, :]
            return err

        graph.add(gtsam.CustomFactor(noise, [key_x, key_v], error_fn))
        self._nhc_fire_count += 1
        return True

    def _zupt_check_and_add(self, graph, ep, prev_ep, imu_window):
        """ZUPT (zero-velocity prior) + ZARU (zero-rotation between
        factor) on GICI-style stationary detection over THIS epoch's own
        IMU integration window -- transcribed from tc/'s `buildfactor/
        zupt.py` `add_zupt_factors`. The stats themselves (`acc_std`/
        `gyro_std`/`gyro_median`) are computed by tc/'s own
        `compute_zupt_stats` (`utils/imu.py`), reused directly -- pure
        function, no FSM deps, exactly like `build_pim` above.

        Disclosed simplification vs. the literal reference: the
        streak-anchor pseudo-measurement (`zupt_anchor_*`) is NOT
        transcribed -- it only ever fires when `gnss_available` is False
        (their GNSS-outage recovery path via `validation/recovery.py`),
        and every Phase-2 epoch reaching this call already has a
        committed DD solve (`nv>=4`, no outage-recovery FSM in this
        port), so that branch would never be reachable here anyway."""
        if not self.zupt_enable:
            return False
        n_imu = len(imu_window)
        if n_imu < self.zupt_min_samples:
            return False
        if self.tc_bias_init is not None:
            ref_acc = np.asarray(self.tc_bias_init.accelerometer())
            ref_gyro = np.asarray(self.tc_bias_init.gyroscope())
        else:
            ref_acc = ref_gyro = None
        stats = compute_zupt_stats(imu_window, ref_acc, ref_gyro)
        if stats is None:
            return False
        if stats['acc_std'] > self.zupt_max_acc_std:
            return False
        if stats['gyro_std'] > self.zupt_max_gyro_std:
            return False
        if stats['gyro_median'] > self.zupt_max_gyro_median:
            return False
        any_added = False
        if self.zupt_sigma_zero_velocity > 0:
            graph.add(gtsam.PriorFactorVector(
                self.PV(ep), np.zeros(3, dtype=np.float64),
                gtsam.noiseModel.Isotropic.Sigma(3, self.zupt_sigma_zero_velocity)))
            any_added = True
        if self.zupt_sigma_zero_rotation > 0 and ep > 0:
            sr = self.zupt_sigma_zero_rotation
            sigmas_pose = np.array([sr, sr, sr, 1e3, 1e3, 1e3])
            graph.add(gtsam.BetweenFactorPose3(
                self.PX(prev_ep), self.PX(ep), gtsam.Pose3(),
                gtsam.noiseModel.Diagonal.Sigmas(sigmas_pose)))
            any_added = True
        if any_added:
            self._zupt_fire_count += 1
        return any_added

    def _add_doppler_vel_prior(self, graph, ep, obs, obs_sd, rs, vs, iu, sat,
                                 pos_pred_ecef, vel_pred_enu):
        """Per-epoch `PriorFactorVector(Vel(ep), v_doppler_enu, sigma)`
        from rover Doppler (`obs.D`) -- transcribed from tc/'s
        `buildfactor/doppler.py` `add_doppler_vel_prior`. The LS solve
        itself (`doppler_velocity_ls`, GICI-style per-satellite LOS
        Doppler + iterative-Huber outlier rejection) is tc/'s own pure
        `utils/ls_solvers.py` function, reused directly, not
        reimplemented -- same pattern as `build_pim`."""
        sigma = self.doppler_vel_sigma
        if sigma <= 0:
            return False
        vel_pred_ecef = self.R_enu2ecef @ np.asarray(vel_pred_enu)
        v_dop_ecef, _clkdr, dop_ok, dop_res, _dop_n = _tc_doppler_velocity_ls(
            obs, obs_sd, rs, vs, iu, sat, pos_pred_ecef,
            nav_nf=self.nav.nf,
            get_wavelengths=lambda o, s: _get_wavelengths(self.nav, o, s),
            vel_pred_ecef=vel_pred_ecef,
            outlier_thresh_m_s=float(os.environ.get('DOP_OUTLIER_M_S', '3.0')))
        if not dop_ok or dop_res >= self.doppler_max_res:
            return False
        v_dop_enu = self.R_enu2ecef.T @ v_dop_ecef
        graph.add(gtsam.PriorFactorVector(
            self.PV(ep), v_dop_enu,
            gtsam.noiseModel.Isotropic.Sigma(3, sigma)))
        self._doppler_fire_count += 1
        return True

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        if len(obs.sat) == 0:
            return
        if self.phase == 1:
            prev_committed = self.last_valid_epoch
            super().process(obs, cs=cs, orb=orb, bsx=bsx, obsb=obsb)
            if self.last_valid_epoch != prev_committed:
                self._phase1_hist.append((obs.t, self.nav.x[0:3].copy()))
                if len(self._phase1_hist) >= self.phase1_epochs:
                    self._transition_to_phase2(obs)
            return
        self._process_phase2(obs, obsb)

    def _transition_to_phase2(self, obs):
        """Phase-1 -> Phase-2 handoff (TASK_M5.md item 4): brief GNSS-only
        bootstrap -> IMU-tight Pose3/Vel/Bias state. Attitude (pitch/roll
        from stationary accel tilt, heading from bootstrap displacement)
        and bias (stationary IMU average) computed exactly as tc/'s own
        `tightly_coupled.transition_to_tc`, from this simpler bootstrap's
        `_phase1_hist` instead of their multi-fix `collected_fixes`
        (see class docstring for why that simplification is warranted
        here). Starts a FRESH ISAM2/IFLS (Phase 1's Point3 smoother is
        abandoned, matching tc/'s own fresh `tc.isam2` at transition) and
        resets ALL WP13c/d ambiguity-state dicts (`amb_keys`/`_streak`/
        `_committed`/`_amb_gen`/`_bad_streak_by_key`) since they keyed
        into that now-abandoned smoother's Bayes tree -- carrying them
        forward would make `_build_dd_factors_arm` skip re-initializing
        an ambiguity value/prior that was never actually inserted into
        the NEW graph, crashing the first Phase-2 update.
        """
        base_ecef = np.array(self.nav.rb, dtype=float)
        lat, lon, _ = ecef2pos(base_ecef)
        sl, cl = np.sin(lat), np.cos(lat)
        sn, cn = np.sin(lon), np.cos(lon)
        self.R_enu2ecef = np.array([
            [-sn, -sl * cn, cl * cn],
            [cn, -sl * sn, cl * sn],
            [0, cl, sl]])
        self.ecef_T_nav = gtsam.Pose3(
            gtsam.Rot3(self.R_enu2ecef), gtsam.Point3(*base_ecef))

        n_static = min(500, len(self.imu_data))
        bias_acc, bias_gyro = estimate_stationary_bias(self.imu_data, n_static)
        bias0 = gtsam.imuBias.ConstantBias(bias_acc, bias_gyro)
        self.tc_bias_init = bias0

        if n_static > 0:
            acc_avg = np.mean([im['acc'] for im in self.imu_data[:n_static]], axis=0)
        else:
            acc_avg = np.array([0.0, 0.0, 9.81])
        acc_tilt = acc_avg - bias_acc
        pitch_rad = float(np.arctan2(acc_tilt[0], np.sqrt(acc_tilt[1] ** 2 + acc_tilt[2] ** 2)))
        roll_rad = float(np.arctan2(acc_tilt[1], acc_tilt[2]))

        # Bug fix (WP13f/TASK_M6.md item 3): `_phase1_hist[0]` is the FIRST
        # committed Phase-1 (GNSS-only Point3) epoch -- diagnosed via a
        # per-epoch dump: this epoch's own position is a transient,
        # barely-converged single-epoch solution (observed ~17m off on
        # tokyo run2, vs <0.04m from _phase1_hist[2] onward), because the
        # very first GTSAM node only has a wide position PRIOR (`nav.
        # sig_p0`) behind it, not yet the DD-factor-converged estimate the
        # rest of the bootstrap enjoys. The reference (`tightly_coupled.
        # transition_to_tc`/`_init_tc_heading_series`) does not have this
        # problem because its `collected_fixes` only start accumulating
        # once cssrlib has reported an actual validated LAMBDA FIX, i.e.
        # already past this transient. Using the raw first point as the
        # t0/ecef0 endpoint for the displacement/heading/velocity seed
        # corrupts it badly when the REAL displacement over the bootstrap
        # window is small (a few meters horizontally): a 17m transient
        # error at one endpoint, against a ~4m true displacement, swings
        # the estimated heading by ~28 degrees (empirically: -67.2 deg vs
        # tc/'s own -39.3 deg reference on this exact run2 window) -- and
        # that heading error is exactly what turns into the
        # steadily-growing (not bounded) Phase-2 float drift this task
        # set out to fix (a wrong forward-direction estimate misdirects
        # every IMU-predicted displacement between GNSS updates). Fix:
        # skip a short burn-in prefix (matches empirically where our own
        # per-epoch error settles to its steady <0.05m floor) before
        # picking the t0 endpoint.
        burn_in = min(10, len(self._phase1_hist) - 2)
        t0, ecef0 = self._phase1_hist[burn_in]
        t1, ecef1 = self._phase1_hist[-1]
        dt_span = max(float(timediff(t1, t0)), 1.0)
        enu0 = self.R_enu2ecef.T @ (ecef0 - base_ecef)
        enu1 = self.R_enu2ecef.T @ (ecef1 - base_ecef)
        disp_enu = enu1 - enu0
        vel_enu = disp_enu / dt_span
        heading_rad = heading_from_vel(vel_enu, fallback=0.0, disp_enu=disp_enu)

        R_b2e = euler_to_R_body2enu(roll_rad, pitch_rad, heading_rad)
        body_enu = enu1 - R_b2e @ self.lever_arm
        pose0 = gtsam.Pose3(gtsam.Rot3(R_b2e), gtsam.Point3(*body_enu))

        if os.environ.get('WP13F_DBG_TC'):
            print(f"DBG_TRANSITION pitch={np.degrees(pitch_rad):.1f} "
                  f"roll={np.degrees(roll_rad):.1f} heading={np.degrees(heading_rad):.1f} deg "
                  f"bias_acc0={bias_acc} bias_gyro0_deg_s={np.degrees(bias_gyro)} "
                  f"vel_enu={vel_enu} disp_enu={disp_enu} dt_span={dt_span} "
                  f"n_phase1_hist={len(self._phase1_hist)}")

        _, tow0 = time2gpst(obs.t)
        _, _, _, imu_idx0 = build_pim(
            self.imu_params, self.tc_bias_init, self.imu_data, 0, target_tow=tow0)
        self._seed_phase2_graph(pose0, vel_enu, bias0, obs.t, imu_idx0)

    def _seed_phase2_graph(self, pose0, vel_enu, bias0, obs_time, imu_idx0):
        """Seed a FRESH Phase-2 ISAM2/IFLS at ep=0 with the given
        pose/vel/bias (WP13g/TASK_M7.md "also fix": factored out of
        `_transition_to_phase2` so `--resume` can call this directly
        with a checkpointed NavState via
        `_restore_phase2_from_checkpoint`, bypassing the stationary-
        bootstrap pitch/roll/heading estimation entirely). Resets ALL
        WP13c/d ambiguity-state dicts (`amb_keys`/`_streak`/`_committed`/
        `_amb_gen`/`_bad_streak_by_key`) -- a fresh smoother has no
        ambiguity keys yet either way, bootstrap or resume."""
        lag = float(os.environ.get('LAG', '0'))
        params = gtsam.ISAM2Params()
        if self.tc_literal and int(os.environ.get('TC_LIT_ISAM2', '1')):
            # tc/'s own phase-2 smoother parameters (solver.make_isam2 /
            # TcConfig isam2_relinearize_skip=10 / _threshold=0.05).
            # TC_LIT_ISAM2=0 keeps this port's 0.01/skip-1 instead --
            # A/B knob: tc/'s lazy relinearization is tuned for its
            # per-epoch-N model; with THIS port's persistent-N keys it
            # is a stale-linearization hazard (WP13n).
            params.setRelinearizeThreshold(0.05)
            params.relinearizeSkip = 10
        else:
            params.setRelinearizeThreshold(0.01)
            params.relinearizeSkip = 1
        if lag > 0:
            self.smoother = gtsam.IncrementalFixedLagSmoother(lag, params)
            self.isam = None
        else:
            self.isam = gtsam.ISAM2(params)
            self.smoother = None

        self.amb_keys = {}
        self._streak = {}
        self._committed = {}
        self._amb_gen = {}
        self._bad_streak_by_key = {}
        self._epoch_ddpr_tags_local = []
        self._epoch_ddcp_tags_local = []
        self._last_ddpr_sat_tags = []
        self._last_ddcp_meta = {}
        self._ddpr_bad_count = 0
        # WP13n: piece-1/2/4 state must not survive a re-seed either
        # (stale prev-epoch keys/stamps reference the DISCARDED smoother).
        self._amb_init_epoch = {}
        self._amb_last_seen = {}
        self._mw_state = {}
        self._cmc_state = {}
        self._prev_amb_values = {}
        # WP13o: context-signal maps reference the discarded smoother's
        # factor evaluations -- reset alongside (refreshed next commit).
        self._main_ddpr_per_sat = {}
        self._main_ddpr_res_pr = 0.0
        self._main_ddpr_per_sat_pr = {}
        # WP13p: per-epoch state must not survive a re-seed either.
        self._slip_flag_streak = {}
        self._churn_dbg_ctx = None
        # WP13q: pending held-release pins reference the discarded
        # smoother's committed values -- a fresh graph starts unpinned.
        self._release_seed_pending = {}
        # WP13r: recovery cp-hold state restarts with the fresh graph
        # (the warm-reset caller re-arms it AFTER this returns, exactly
        # like tc/'s warm_reset_phase2 sets _recov_cp_hold post-init).
        self._recov_cp_hold_remaining = 0
        self._recov_cp_release_streak = 0
        self._recov_cp_skip_now = False
        # WP13r: per-sat quarantine state restarts too (tc/'s
        # warm_reset_phase2 calls sat_quality.clear()).
        self._persist_bad_streak = {}
        self._persist_bad_hold = {}
        # WP13s literal: tc/'s warm_reset_phase2 clears the WHOLE
        # sat-quality state (sat_quality.clear()) -- and the persist-bad
        # dicts must stay ALIASED to the imported SatQualityState so
        # sq.tick / compute_cp_build_policy keep seeing them.
        if self._sq is not None:
            self._sq.clear()
        if self.tc_literal:
            self._persist_bad_hold = self._sq.persist_bad_hold
            self._persist_bad_streak = self._sq.persist_bad_streak
        self.rejc_cp_pr = {}
        self._rejc_post_ddpr = {}
        self._fix_streak = {}
        self.ref_sats = {}
        self._sat_el_deg = {}
        self._sat_snr_dbhz = {}
        self._last_pair_rows = []
        self._last_pair_bad_max = 0.0
        self._last_main_ddpr_per_sat = {}
        self._pim_discontinuity = False
        # WP13o: PHASE-2-SCOPED accept-stack overrides. The O3 probe
        # (results/wp13o/o3_tcstack_r2) measured that applying the
        # tc/-point collapse (THRESAR=3.0 / hold_ratio 0 / lambda_corr
        # 1.0) GLOBALLY damages ONLY the phase-1 stationary bootstrap
        # (run2-3000ep: ep0-200 RMS 1.75->4.05, phase-1 fixes 197->159;
        # phase-2 AllRMS bit-flat at 1.496) -- phase 1 keeps the B
        # accept stack, and these overrides engage exactly when the
        # phase-2 graph is seeded (transition, --resume restore and the
        # piece-7 warm reset all funnel through this method; idempotent).
        for _env, _attr, _cast in (
                ('THRESAR_P2', None, float),                     # nav.thresar
                ('HOLD_RATIO_P2', 'hold_ratio', float),
                ('LAMBDA_CORR_HARD_MAX_P2', 'lambda_corr_hard_max', float),
                ('LOW_NB_FIX_REJECT_NB_MAX_P2', 'low_nb_fix_reject_nb_max', int),
                ('AR_CONTEXT_NB_MAX_P2', 'ar_context_nb_max', int),
                ('WEAK_FIX_NB_MAX_P2', 'weak_fix_nb_max', int),
                ('HOLD_CONFIRM_N_P2', 'hold_confirm_n', int),
                ('MIN_COMMITTED_FOR_FIX_P2', 'min_committed_for_fix', int),
                ('RELEASE_K_P2', 'release_k', int),
                ('RELEASE_THRES_P2', 'release_thres', float)):
            _v = os.environ.get(_env, '')
            if _v:
                if _attr is None:
                    self.nav.thresar = _cast(_v)
                else:
                    setattr(self, _attr, _cast(_v))

        seed_ep = 0
        key_x, key_v, key_b = self.PX(seed_ep), self.PV(seed_ep), self.PB(seed_ep)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        values.insert(key_x, pose0)
        values.insert(key_v, vel_enu)
        values.insert(key_b, bias0)
        # Roll/pitch tight (from accel tilt or the checkpointed estimate),
        # yaw looser, position loose (a LOCAL seed, not a validated LAMBDA
        # fix) -- same relative-tightness pattern as tc/'s own
        # `add_tc_seed_epoch` i==0 case, simplified to one seed epoch.
        graph.addPriorPose3(key_x, pose0, gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.3, 3.0, 3.0, 3.0])))
        graph.addPriorVector(key_v, vel_enu, gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
        graph.addPriorConstantBias(key_b, bias0, gtsam.noiseModel.Isotropic.Sigma(6, 0.02))

        ts = gtsam.FixedLagSmootherKeyTimestampMap()
        ts[key_x] = self.epoch_time
        ts[key_v] = self.epoch_time
        ts[key_b] = self.epoch_time
        try:
            if self.smoother:
                self.smoother.update(graph, values, ts)
                estimate = self.smoother.calculateEstimate()
            else:
                self.isam.update(graph, values)
                estimate = self.isam.calculateEstimate()
            self.current_estimate = estimate
        except RuntimeError:
            self.current_estimate = values

        self.last_valid_epoch = seed_ep
        self.last_valid_time = obs_time
        self.epoch2 = seed_ep + 1
        self.imu_idx = imu_idx0

        pose0_ecef = self.ecef_T_nav.compose(pose0)
        self.nav.x[0:3] = (np.array(pose0_ecef.translation())
                           + pose0_ecef.rotation().matrix() @ self.lever_arm)
        self.nav.smode = 5
        self.phase = 2

    def _restore_phase2_from_checkpoint(self, pose0, vel_enu, bias0,
                                          week, tow, imu_idx0):
        """WP13g Part 2 (TASK_M7.md "also fix"): bypass the stationary-
        assumption Phase-1 bootstrap entirely on `--resume` by seeding
        Phase 2 directly from a checkpointed NavState (position/
        velocity/attitude via `pose0`/`vel_enu`, accel/gyro bias via
        `bias0`, IMU sample cursor via `imu_idx0`), instead of
        constructing a brand-new `GtsamRtkTc` that starts at `phase=1`
        and re-runs `_transition_to_phase2`'s accel-tilt pitch/roll +
        bootstrap-displacement heading estimation on data that is
        actually moving (WP13F_REPORT.md's disclosed bug -- ~9500m-scale
        artifacts at the resume boundary). `pose0`/`vel_enu` are in the
        LOCAL ENU frame anchored at `nav.rb` (recomputed here exactly as
        `_transition_to_phase2` does -- the base station position is
        fixed per run, so `R_enu2ecef`/`ecef_T_nav` are bit-for-bit
        reproducible across a process restart). See
        `wp13a_run_standalone_rtk.py`'s `--resume` handling for what gets
        persisted into the checkpoint `.npz` and read back here."""
        base_ecef = np.array(self.nav.rb, dtype=float)
        lat, lon, _ = ecef2pos(base_ecef)
        sl, cl = np.sin(lat), np.cos(lat)
        sn, cn = np.sin(lon), np.cos(lon)
        self.R_enu2ecef = np.array([
            [-sn, -sl * cn, cl * cn],
            [cn, -sl * sn, cl * sn],
            [0, cl, sl]])
        self.ecef_T_nav = gtsam.Pose3(
            gtsam.Rot3(self.R_enu2ecef), gtsam.Point3(*base_ecef))
        self.tc_bias_init = bias0

        obs_t = gpst2time(week, tow)
        self._seed_phase2_graph(pose0, vel_enu, bias0, obs_t, imu_idx0)

    def _tc_literal_pick_ref(self, sys_id, idx_sys, sat, el):
        """tc/'s preprocess/prefit.py `pick_ref_sat_idx`, literal:
        previously-locked reference kept for continuity; rejected when
        it left view, is a BDS GEO, or its PREV-epoch post-fit DDPR
        residual crossed `per_sat_res_thresh`; fresh pick = highest
        elevation among non-GEO, non-multipath sats (ref_bad tier
        disabled by the tokyo preset's ref_bad_reject_thresh=0.0).
        NOTE tc/ does NOT reset system ambiguities on a phase-2 ref
        switch (its reset branch requires slip_keys None, never true in
        phase 2)."""
        def _is_geo(s_):
            _sys, _prn = sat2prn(int(s_))
            return _sys == uGNSS.BDS and (_prn <= 5 or 59 <= _prn <= 63)

        prev_ref = self.ref_sats.get(sys_id)
        sats_in_sys = [sat[i] for i in idx_sys]
        last_res = self._main_ddpr_per_sat_pr or {}
        thresh = float(self.per_sat_res_thresh)
        prev_is_geo = (sys_id == uGNSS.BDS and prev_ref is not None
                       and _is_geo(prev_ref))
        prev_res = float(last_res.get(int(prev_ref), 0.0)) \
            if prev_ref is not None else 0.0
        if (prev_ref is not None and prev_ref in sats_in_sys
                and not prev_is_geo and prev_res <= thresh):
            ref_idx = idx_sys[sats_in_sys.index(prev_ref)]
            self.ref_sats[sys_id] = prev_ref
            return ref_idx, prev_ref

        def ok(i):
            s_ = sat[i]
            if sys_id == uGNSS.BDS and _is_geo(s_):
                return False
            if float(last_res.get(int(s_), 0.0)) > thresh:
                return False
            return True

        pool = [i for i in idx_sys if ok(i)]
        if not pool:
            pool = [i for i in idx_sys
                    if not (sys_id == uGNSS.BDS and _is_geo(sat[i]))]
        if not pool:
            pool = idx_sys
        pool_arr = np.array(pool)
        ref_idx = int(pool_arr[np.argmax(el[pool_arr])])
        self.ref_sats[sys_id] = sat[ref_idx]
        return ref_idx, sat[ref_idx]

    def _tc_literal_cp_pr_reject(self, obs, obsb, iu, ir, ri, ji,
                                  ref_sat, j_sat, f, lam_f, new_amb):
        """tc/'s buildfactor/factors.py `_cp_pr_innovation_rejects`,
        literal: |DD_PR - (DD_CP - lam*(N_ref - N_j))| > 10.0 m (preset
        cp_pr_innov_thresh) on a NON-fresh pair rejects the DDCP row
        this epoch and increments the j-sat's rejc_cp_pr counter (wiped
        keys handled by `_tc_literal_reset_failing_n` at 2). N values in
        CYCLES: the committed integer when held, else the current graph
        estimate. Returns True on reject."""
        cfg = self.cfg
        if not (cfg.cp_pr_innov_thresh > 0 and int(cfg.cp_pr_rejc_max) > 0):
            return False
        # fresh pair: either side seeded THIS epoch (no prev key/hold)
        if (ref_sat, f) in new_amb or (j_sat, f) in new_amb:
            return False

        def _n_val(s_):
            key = (s_, f)
            if key in self._committed:
                v = float(self._committed[key])
                return v if np.isfinite(v) else None
            k = self.amb_keys.get(key)
            if (k is not None and self.current_estimate is not None
                    and self.current_estimate.exists(k)):
                return float(self.current_estimate.atDouble(k))
            return None

        n_ref = _n_val(ref_sat)
        n_j = _n_val(j_sat)
        if n_ref is None or n_j is None:
            return False
        try:
            pr_ref_r = float(obs.P[iu[ri], f])
            pr_ref_b = float(obsb.P[ir[ri], f])
            pr_j_r = float(obs.P[iu[ji], f])
            pr_j_b = float(obsb.P[ir[ji], f])
            cp_ref_r = float(obs.L[iu[ri], f]) * lam_f
            cp_ref_b = float(obsb.L[ir[ri], f]) * lam_f
            cp_j_r = float(obs.L[iu[ji], f]) * lam_f
            cp_j_b = float(obsb.L[ir[ji], f]) * lam_f
        except (IndexError, TypeError, ValueError):
            return False
        if 0.0 in (pr_ref_r, pr_ref_b, pr_j_r, pr_j_b,
                   cp_ref_r, cp_ref_b, cp_j_r, cp_j_b):
            return False
        dd_pr = (pr_ref_r - pr_j_r) - (pr_ref_b - pr_j_b)
        dd_cp = (cp_ref_r - cp_j_r) - (cp_ref_b - cp_j_b)
        innov = abs(dd_pr - (dd_cp - lam_f * (n_ref - n_j)))
        if innov > float(cfg.cp_pr_innov_thresh):
            key_j = (j_sat, f)
            self.rejc_cp_pr[key_j] = self.rejc_cp_pr.get(key_j, 0) + 1
            self.cp_pr_reject_count += 1
            return True
        return False

    def _build_dd_factors_arm(self, graph, new_values, obs, obsb, obs_sd,
                                rs, rsb, sat, el, iu, ir, pos_pred_ecef, ep):
        """Arm-factor analogue of the parent's `_build_dd_factors`
        (TASK_M5.md item 3): same reference-satellite selection / new-
        ambiguity seeding logic, `DoubleDifferencePseudorangeFactorArm`/
        `DoubleDifferenceCarrierPhaseFactorArm` (tc/'s `buildfactor/
        factors.py` lines ~78/251) in place of the bare-Point3 factors,
        keyed on this epoch's `Pose3` (`PX(ep)`) + this rover's lever arm
        + the fixed `ecef_T_nav` local-ENU->ECEF transform."""
        nv = 0
        new_amb = {}
        cp_amb = set()
        self._epoch_ddpr_tags_local = []
        self._epoch_ddcp_tags_local = []
        base_pt = gtsam.Point3(*self.nav.rb)
        rb = np.array(self.nav.rb)
        key_x = self.PX(ep)
        lever = self.lever_pt

        # WP13j layer (b): this epoch's NLOS set, only computed when the
        # measurement-weighting layer is actually enabled (0 cost off).
        nlos_sats = self._get_nlos_sats(obs) if self.nlos_meas_mode else frozenset()

        # WP13s literal: tc/'s `_reset_persistently_failing_n` runs at
        # the START of every DD build (buildfactor/factors.py builder
        # __init__) -- wipe keys whose cp_pr / post_ddpr counters
        # crossed the preset bars BEFORE this epoch re-seeds them.
        if self.tc_literal:
            self._tc_literal_reset_failing_n()

        for sys_id in _sorted_sys_ids(obs_sd.sig):
            idx_sys = self.sysidx(sat, sys_id)
            if len(idx_sys) < 2:
                continue
            if self.tc_literal:
                # tc/'s prefit.pick_ref_sat_idx, literal: previously-
                # locked reference kept for continuity; fresh pick =
                # highest elevation excluding BDS GEO and sats whose
                # prev-epoch post-fit DDPR residual crossed
                # per_sat_res_thresh (ref_bad_reject_thresh = 0.0 in
                # the tokyo preset -> that tier disabled).
                ref_idx, ref_sat = self._tc_literal_pick_ref(
                    sys_id, idx_sys, sat, el)
            else:
                ref_idx = idx_sys[np.argmax(el[idx_sys])]
                ref_sat = sat[ref_idx]
            lams = _get_wavelengths(self.nav, obs_sd, ref_sat)

            for j_idx in idx_sys:
                if j_idx == ref_idx:
                    continue
                sat_j = sat[j_idx]
                # WP13n piece 6: tc/ preset `exclude_bds_geo=1` -- skip
                # BeiDou GEO j-sats (PRN <=5 / 59-63) for BOTH PR and CP
                # (tc/'s DdFactorBuilder.run() line ~591).
                if (self.exclude_bds_geo and sys_id == uGNSS.BDS):
                    _prn_j = sat2prn(int(sat_j))[1]
                    if _prn_j <= 5 or 59 <= _prn_j <= 63:
                        continue
                ri, ji = ref_idx, j_idx

                # WP13j layer (b): this DD pair touches an NLOS-flagged sat
                # if EITHER its ref or target satellite is flagged this
                # epoch (a wrong ref choice is a known caveat here -- ref
                # selection stays max-elevation, unchanged; see the
                # module-level WP13j comment).
                pair_nlos = int(ref_sat) in nlos_sats or int(sat_j) in nlos_sats

                for f in range(self.nav.nf):
                    if f >= len(lams):
                        continue
                    lam_f = lams[f]
                    if obs_sd.P[ri, f] == 0 or obs_sd.P[ji, f] == 0:
                        continue

                    if self.nlos_meas_mode == 1 and pair_nlos:
                        # hard-exclude: skip this pair/freq entirely (no PR,
                        # no CP, no ambiguity seed) this epoch.
                        continue

                    rs_ref = gtsam.Point3(*rs[iu[ri], :3])
                    rs_j = gtsam.Point3(*rs[iu[ji], :3])
                    rsb_ref = gtsam.Point3(*rsb[ir[ri], :3])
                    rsb_j = gtsam.Point3(*rsb[ir[ji], :3])

                    pr_sigma_mult = (self.nlos_meas_sigma_mult
                                      if (self.nlos_meas_mode == 2 and pair_nlos) else 1.0)
                    if pr_sigma_mult != 1.0:
                        self.nlos_meas_pr_affected += 1

                    # WP13i (TASK_M9.md): DD-PR built EVERY epoch ALONGSIDE
                    # DD-CP (tc/'s buildfactor/factors.py ~line 609, "PR
                    # factor: added even without CP" -- DdFactorBuilder.run()
                    # always calls _build_pr_for_pair, THEN separately
                    # _build_cp_for_pair, for the SAME pair). WP13a-h built
                    # these MUTUALLY EXCLUSIVELY (CP whenever carrier lock
                    # held, PR only as the no-lock fallback), starving the
                    # float of any absolute, ambiguity-independent anchor
                    # whenever CP was available -- diagnosed (WP13d/g/h) as
                    # the root cause of the ~7-10m broadly-distributed float
                    # floor. Un-conditional now, matching tc/ exactly.
                    # WP13n piece 3: tc/'s elevation-dependent varerr DD
                    # sigma in place of the flat sigma_pr*sqrt(2) when
                    # VARERR_ENABLE=1 (tc/'s own config default; see
                    # `_varerr_dd_sigma`). The nlos/badness multipliers
                    # stack on top exactly as before, mirroring tc/'s
                    # own `pair_sigma` -> bad-scale ordering.
                    _sig_pr_base = (self._varerr_dd_sigma(1, el[ri], el[ji])
                                    if self.varerr_enable
                                    else self.sigma_pr * np.sqrt(2))
                    noise_pr = gtsam.noiseModel.Isotropic.Sigma(
                        1, _sig_pr_base * pr_sigma_mult)
                    if self.huber_pr > 0:
                        noise_pr = gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Huber.Create(self.huber_pr),
                            noise_pr)
                    ddpr_local_idx = graph.size()
                    graph.add(gtsam.DoubleDifferencePseudorangeFactorArm(
                        key_x,
                        float(obs.P[iu[ri], f]), float(obsb.P[ir[ri], f]),
                        float(obs.P[iu[ji], f]), float(obsb.P[ir[ji], f]),
                        rs_ref, rs_j, rsb_ref, rsb_j, base_pt,
                        lever, self.ecef_T_nav, noise_pr))
                    self._epoch_ddpr_tags_local.append(
                        (ddpr_local_idx, int(ref_sat), int(sat_j), f))
                    nv += 1

                    has_cp = obs_sd.L[ri, f] != 0 and obs_sd.L[ji, f] != 0
                    # WP13n piece 5: GLONASS never gets a DD-CP factor
                    # (tc/'s `_build_cp_for_pair` GLO early-return; FDMA
                    # DD ambiguities are non-integer) -- PR-only, which
                    # the DD-PR factor above already provided.
                    if self.glo_cp_exclude and sys_id == uGNSS.GLO:
                        has_cp = False
                    # WP13p (a) AR_TIER_MODE=2: never build a DD-CP
                    # factor for a pair touching a MARGINAL (below-tier
                    # el/CNR) satellite -- PR-only membership, exactly
                    # the GLO_CP_EXCLUDE mechanism above: the marginal
                    # sat then owns no N variable at all, so it can
                    # neither enter LAMBDA nor churn the slip/MW/outage
                    # reset machinery, while its DD-PR factor (already
                    # added above) keeps anchoring the float. Committed
                    # keys exempt by default (AR_TIER_HELD_EXEMPT).
                    if has_cp and self.ar_tier_mode == 2 and (
                            self.ar_tier_elmin_deg > 0
                            or self.ar_tier_cnr_min > 0):
                        _tier_el_floor = np.deg2rad(self.ar_tier_elmin_deg)
                        for _si, _row, _el_i in ((int(ref_sat), ri, el[ri]),
                                                 (int(sat_j), ji, el[ji])):
                            if (self.ar_tier_held_exempt
                                    and (_si, f) in self._committed):
                                continue
                            _low_el = (self.ar_tier_elmin_deg > 0
                                       and float(_el_i) < _tier_el_floor)
                            _low_cnr = False
                            if self.ar_tier_cnr_min > 0:
                                try:
                                    _c = float(obs.S[iu[_row], f])
                                except (IndexError, TypeError, ValueError):
                                    _c = 0.0
                                _low_cnr = 0.0 < _c < self.ar_tier_cnr_min
                            if _low_el or _low_cnr:
                                has_cp = False
                                self.ar_tier_cp_excl_count += 1
                                break
                    # WP13k priority 3: global cp_hold -- skip DD-CP
                    # entirely (PR-only, relying on WP13i's DD-PR-always
                    # anchor) while a hold window is active. Ambiguity
                    # tracking itself is untouched (no seed/reset here) --
                    # a held-open ambiguity simply doesn't get a fresh CP
                    # observation this epoch, same as any other no-lock
                    # epoch this port already handles.
                    if self.cp_hold_enable and self._cp_hold_remaining > 0:
                        has_cp = False
                    # WP13r lever 2: recovery CP-hold window (tc/'s
                    # `_recov_cp_hold`, armed by sanity resets / solve-
                    # exception warm resets / sanity-triggered bad
                    # epochs) -- PR-only float while it runs, exactly
                    # the mechanism above but on the recovery counter.
                    # `_recov_cp_skip_now` is this epoch's pre-tick
                    # snapshot (tc/'s gate.py `ed.skip_cp_now`,
                    # computed BEFORE the decrement). WP13s literal
                    # mode routes this through tc/'s OWN
                    # compute_cp_build_policy below instead (its
                    # `skip_cp` argument), AFTER the N seed -- tc/
                    # seeds N even on policy-dropped pairs.
                    if self._recov_cp_skip_now and not self.tc_literal:
                        has_cp = False
                    # WP13r: quarantined sat gets NO DD-CP factor while
                    # its hold runs -- tc/'s forced_hold pairs through
                    # `compute_cp_build_policy` with the default
                    # cp_hold_sigma_penalty=0.0 (penalty<=0 -> CP
                    # disabled for the pair). DD-PR (above) still
                    # anchors the float, same as tc/. (Literal mode:
                    # via the imported compute_cp_build_policy below.)
                    if (not self.tc_literal
                            and self.ar_persist_bad_enable and has_cp
                            and (int(ref_sat) in self._persist_bad_hold
                                 or int(sat_j) in self._persist_bad_hold)):
                        has_cp = False
                    if has_cp:
                        if self.per_epoch_n:
                            # WP13n piece 4: fresh N variable per epoch,
                            # coupled to last epoch's via continuity
                            # prior (sigma_cont at last ESTIMATE) +
                            # BetweenFactorDouble(0, sigma 0.01 FIX /
                            # 0.1 FLT) -- tc/'s `_seed_one_amb_prior` +
                            # stage.py BetweenN chain, merged at seed
                            # site (equivalent factor set per epoch).
                            kr = self.N_ep(ref_sat, f, ep)
                            kj = self.N_ep(sat_j, f, ep)
                        else:
                            kr = self.N(ref_sat, f)
                            kj = self.N(sat_j, f)
                        for sn, kn, sd_cp, sd_pr, rs_s in [
                                (ref_sat, kr, obs_sd.L[ri, f] * lam_f,
                                 float(obs_sd.P[ri, f]), rs[iu[ri], :3]),
                                (sat_j, kj, obs_sd.L[ji, f] * lam_f,
                                 float(obs_sd.P[ji, f]), rs[iu[ji], :3])]:
                            if self.per_epoch_n:
                                # WP13s N4: a HELD (committed) ambiguity
                                # owns NO graph variable at all (tc/'s
                                # activate_hold: amb_key=None; the DDCP
                                # factor below folds the integer into
                                # its constant).
                                if (self.tc_literal and self.cond_hold
                                        and (int(sn), f) in self._committed):
                                    continue
                                if (sn, f) in new_amb:
                                    continue
                                prev = self._prev_amb_values.get((sn, f))
                                if (prev is not None
                                        and self.current_estimate is not None
                                        and self.current_estimate.exists(prev[0])):
                                    k_old, n_prev = prev
                                    new_values.insert(kn, n_prev)
                                    graph.addPriorDouble(
                                        kn, n_prev,
                                        gtsam.noiseModel.Isotropic.Sigma(
                                            1, self.sigma_cont))
                                    sig_btw = (self.sigma_n_between
                                               if self._last_report_was_fix
                                               else self.sigma_n_between_flt)
                                    graph.add(gtsam.BetweenFactorDouble(
                                        k_old, kn, 0.0,
                                        gtsam.noiseModel.Isotropic.Sigma(
                                            1, sig_btw)))
                                    self.n_between_count += 1
                                    self.n_cont_seed_count += 1
                                    self.nav.x[self.IB(sn, f, self.nav.na)] = n_prev
                                else:
                                    r_r, _ = geodist(rs_s, pos_pred_ecef)
                                    r_b, _ = geodist(rs_s, rb)
                                    n0 = sd_cp - (r_r - r_b)
                                    sig_n0 = self.sigma_amb0
                                    if self.seed_n_cycles and not self.tc_literal:
                                        # WP17 B1 root-fix: cycles seed
                                        # ANCHORED ON THE PSEUDORANGE,
                                        # N0 = (cp - pr) / lam. The
                                        # geometric-range anchor keeps
                                        # the epoch's SD receiver-clock
                                        # term in the seed, so seeds
                                        # minted at different epochs are
                                        # mutually off by the clock RAMP
                                        # (measured: DD float 36k cycles
                                        # off-lattice under HUBER_CP=1);
                                        # cp - pr cancels the clock
                                        # per-seed (residual = code
                                        # noise/multipath, 5-25 cyc,
                                        # healed in ~1-2 solves inside
                                        # the ar_wait_new window).
                                        if sd_pr != 0.0:
                                            n0 = (sd_cp - sd_pr) / lam_f
                                        else:
                                            n0 = n0 / lam_f
                                    elif self.tc_literal:
                                        # cycles (see the persistent-N
                                        # branch's B1 comment) + tc/'s
                                        # release-seed pin
                                        # (_seed_one_amb_prior second
                                        # branch: last_held_value at
                                        # sigma 0.1 cyc).
                                        n0 = n0 / lam_f
                                        if self.release_seed_held:
                                            _pend = (self.
                                                     _release_seed_pending
                                                     .pop((sn, f), None))
                                            if _pend is not None:
                                                _pv, _pep = _pend
                                                if ((ep - _pep)
                                                        > self.slip_reset_maxout):
                                                    self.release_seed_expired += 1
                                                else:
                                                    n0 = _pv
                                                    sig_n0 = self.release_seed_sigma
                                                    self.release_seed_used += 1
                                    new_values.insert(kn, n0)
                                    graph.addPriorDouble(
                                        kn, n0,
                                        gtsam.noiseModel.Isotropic.Sigma(
                                            1, sig_n0))
                                    self.n_fresh_seed_count += 1
                                    # tc/ stamps amb_init_epoch only on a
                                    # FRESH seed -- continuing keys keep
                                    # their original age for ar_wait_new.
                                    self._amb_init_epoch[(sn, f)] = ep
                                    self.nav.x[self.IB(sn, f, self.nav.na)] = n0
                                new_amb[(sn, f)] = kn
                            elif (sn, f) not in self.amb_keys and (sn, f) not in new_amb:
                                r_r, _ = geodist(rs_s, pos_pred_ecef)
                                r_b, _ = geodist(rs_s, rb)
                                n0 = sd_cp - (r_r - r_b)
                                # WP13s literal: the Arm CP factor's
                                # error is ddModel + lam*(Nref-Nj) -
                                # dd_obs, i.e. N is in CYCLES
                                # (CarrierPhaseFactor.cpp:417; tc/'s
                                # _init_dd_ambiguity_priors divides by
                                # lam). This port seeded METERS since
                                # WP13e -- a 1/lam-scaled-wrong initial
                                # value/prior on every fresh seed (the
                                # WP13f "fresh-seed step jump").
                                # WP17 (B1 root-fix): SEED_N_CYCLES=1
                                # seeds the non-literal builder in
                                # CYCLES anchored on the PSEUDORANGE,
                                # N0 = (cp - pr)/lam -- see the
                                # per-epoch branch comment (the
                                # geometric-range anchor keeps the
                                # per-epoch SD receiver clock in the
                                # seed; the clock RAMP across seed
                                # epochs is what left the DD float
                                # thousands of cycles off-lattice
                                # under HUBER_CP>0). Default 0 =
                                # prior configs bit-identical.
                                if self.seed_n_cycles and not self.tc_literal:
                                    if sd_pr != 0.0:
                                        n0 = (sd_cp - sd_pr) / lam_f
                                    else:
                                        n0 = n0 / lam_f
                                elif self.tc_literal:
                                    n0 = n0 / lam_f
                                sig_n0 = self.sigma_amb0
                                # WP13q: consume a pending held-release
                                # pin (tc/'s `_seed_one_amb_prior`
                                # release_seed_pending branch: seed at
                                # last_held_value, sigma 0.1 CYCLES --
                                # x lam_f here, this port's N is in
                                # meters). One-shot; stale pins (key
                                # unconsumed > maxout epochs, i.e. the
                                # sat left view) expire to the normal
                                # fresh seed, mirroring tc/'s outage
                                # clear_hold.
                                if self.release_seed_held:
                                    _pend = self._release_seed_pending.pop(
                                        (sn, f), None)
                                    if _pend is not None:
                                        _pv, _pep = _pend
                                        if (ep - _pep) > self.slip_reset_maxout:
                                            self.release_seed_expired += 1
                                        else:
                                            n0 = _pv
                                            # WP13s literal: sigma 0.1
                                            # CYCLES (tc/'s _noise1(0.1))
                                            # -- N is in cycles now, so
                                            # the x lam_f meter-scale
                                            # conversion drops out.
                                            # (WP17: same under
                                            # SEED_N_CYCLES.)
                                            sig_n0 = (
                                                self.release_seed_sigma
                                                if (self.tc_literal
                                                    or self.seed_n_cycles)
                                                else
                                                self.release_seed_sigma
                                                * lam_f)
                                            self.release_seed_used += 1
                                new_values.insert(kn, n0)
                                graph.addPriorDouble(
                                    kn, n0,
                                    gtsam.noiseModel.Isotropic.Sigma(1, sig_n0))
                                new_amb[(sn, f)] = kn
                                self.nav.x[self.IB(sn, f, self.nav.na)] = n0

                        # WP13s literal: tc/'s DDCP build policy tiers
                        # (IMPORTED compute_cp_build_policy: forced-hold
                        # -> CP dropped, dirty pre-reset -> sigma x15,
                        # probation -> x10, then skip_cp) + the CP-vs-PR
                        # innovation gate. CP-visibility (cp_amb) is
                        # recorded BEFORE the policy (tc/ marks
                        # _ar_cp_visible_sf before its policy call), so
                        # a policy-dropped pair still counts CP-visible
                        # for the held-mirror vsat.
                        _lit_mult = 1.0
                        if self.tc_literal:
                            cp_amb.add((int(ref_sat), f))
                            cp_amb.add((int(sat_j), f))
                            _lit_ok, _lit_mult = _tc_compute_cp_build_policy(
                                self, self._sq, int(ref_sat), int(sat_j), f,
                                self._recov_cp_skip_now)
                            if not _lit_ok:
                                continue
                            if self._tc_literal_cp_pr_reject(
                                    obs, obsb, iu, ir, ri, ji,
                                    int(ref_sat), int(sat_j), f, lam_f,
                                    new_amb):
                                continue
                        cp_sigma_mult = (self.nlos_meas_sigma_mult
                                          if (self.nlos_meas_mode == 2 and pair_nlos) else 1.0)
                        if cp_sigma_mult != 1.0:
                            self.nlos_meas_cp_affected += 1
                        cp_sigma_mult *= _lit_mult
                        # WP13k priority 2: sat_badness CP-sigma scaling
                        # (tc/'s `sat_badness_sigma_scale_cp`) -- worse of
                        # this pair's two satellites' badness scores widens
                        # DD-CP sigma, softening (not excluding) a
                        # persistently-dirty satellite's pull on the float
                        # without the AR-candidate-exclusion mechanism
                        # WP13j found net-negative for the same intent.
                        if self.sat_badness_enable:
                            if self.tc_literal:
                                # tc/'s OWN multi-term score (imported
                                # SatQualityState.sat_badness with the
                                # tokyo preset alphas), not this port's
                                # reduced DD-PR EWMA.
                                badness = max(
                                    self._sq.sat_badness(
                                        self, int(ref_sat), f),
                                    self._sq.sat_badness(
                                        self, int(sat_j), f,
                                        ref_sat=int(ref_sat)))
                            else:
                                badness = max(self._sat_badness_score(ref_sat),
                                              self._sat_badness_score(sat_j))
                            cp_sigma_mult *= (1.0 + self.sat_badness_sigma_scale_cp * badness)
                        # WP13n piece 3: varerr CP sigma (code=0) when
                        # enabled -- see the PR-side comment above.
                        _sig_cp_base = (self._varerr_dd_sigma(0, el[ri], el[ji])
                                        if self.varerr_enable
                                        else self.sigma_cp * np.sqrt(2))
                        noise_cp = gtsam.noiseModel.Isotropic.Sigma(
                            1, _sig_cp_base * cp_sigma_mult)
                        if self.huber_cp > 0:
                            noise_cp = gtsam.noiseModel.Robust.Create(
                                gtsam.noiseModel.mEstimator.Huber.Create(self.huber_cp),
                                noise_cp)
                        ddcp_local_idx = graph.size()
                        # WP13s N4 (literal + per-epoch N): a held
                        # ambiguity's integer is folded into the DDCP
                        # factor CONSTANT (tc/'s `_add_ddcp_factor` /
                        # `_make_ddcp_factor_with_held_n`, IMPORTED) --
                        # the held (sat,f) has no graph variable.
                        _ref_held = _j_held = None
                        if (self.tc_literal and self.per_epoch_n
                                and self.cond_hold):
                            _ref_held = self._committed.get((int(ref_sat), f))
                            _j_held = self._committed.get((int(sat_j), f))
                        if _ref_held is not None or _j_held is not None:
                            _dd_obs_cp = (
                                (float(obs.L[iu[ri], f])
                                 - float(obs.L[iu[ji], f])) * lam_f
                                - (float(obsb.L[ir[ri], f])
                                   - float(obsb.L[ir[ji], f])) * lam_f)
                            if _ref_held is not None and _j_held is not None:
                                _kf, _off, _coef = (
                                    None,
                                    lam_f * (_ref_held - _j_held), 0.0)
                            elif _ref_held is not None:
                                _kf, _off, _coef = (
                                    kj, lam_f * _ref_held, -lam_f)
                            else:
                                _kf, _off, _coef = (
                                    kr, -lam_f * _j_held, lam_f)
                            graph.add(_tc_make_ddcp_factor_with_held_n(
                                key_x, _kf, noise_cp,
                                np.asarray(rs[iu[ri], :3], dtype=float),
                                np.asarray(rs[iu[ji], :3], dtype=float),
                                np.asarray(rsb[ir[ri], :3], dtype=float),
                                np.asarray(rsb[ir[ji], :3], dtype=float),
                                rb, lam_f, _dd_obs_cp,
                                self.lever_arm, self.ecef_T_nav,
                                offset_m=_off, coeff_m=_coef))
                        else:
                            graph.add(gtsam.DoubleDifferenceCarrierPhaseFactorArm(
                                key_x, kr, kj,
                                float(obs.L[iu[ri], f]) * lam_f,
                                float(obsb.L[ir[ri], f]) * lam_f,
                                float(obs.L[iu[ji], f]) * lam_f,
                                float(obsb.L[ir[ji], f]) * lam_f,
                                rs_ref, rs_j, rsb_ref, rsb_j, base_pt, lam_f,
                                lever, self.ecef_T_nav, noise_cp))
                        self._epoch_ddcp_tags_local.append(
                            (ddcp_local_idx, int(ref_sat), int(sat_j), f))
                        cp_amb.add((int(ref_sat), f))
                        cp_amb.add((int(sat_j), f))
                        nv += 1

        return nv, new_amb, cp_amb

    def _write_back_tc(self, estimate, key_x, cp_amb=None):
        """Pose3-aware analogue of the parent's `_write_back` (TASK_M5.md
        item 3): `nav.x[0:3]` = antenna ECEF position (`ecef_T_nav`-
        composed pose translation + rotated lever arm); `nav.P`'s
        position block comes from the POSE's OWN [3:6,3:6] joint-marginal
        sub-block (the `rho`/translation part of GTSAM's 6-D Pose3
        tangent -- verified empirically that gtsam's default (non-EXPMAP)
        Pose3 retraction keeps `rho` as a plain world/nav-frame
        translation delta, independent of the rotation part, so this
        sub-block IS the position covariance in the local ENU frame),
        rotated by the FIXED `R_enu2ecef` to match nav.P's ECEF-frame
        convention (same convention the parent's Point3 `_write_back`
        already uses). Disclosed approximation: the lever-arm's own
        ROTATION-uncertainty contribution to antenna position covariance
        is neglected (small lever, ~0.3-0.5m) -- everything downstream
        (WP13c's `_do_ar`/LAMBDA, WP13d's `_apply_fde`) is otherwise
        IDENTICAL inherited code, unaware this came from a Pose3."""
        self.nav.P[:, :] = 0
        self.nav.vsat[:, :] = 0
        pose = estimate.atPose3(key_x)
        pose_ecef = self.ecef_T_nav.compose(pose)
        R = pose_ecef.rotation().matrix()
        self.nav.x[0:3] = np.array(pose_ecef.translation()) + R @ self.lever_arm

        # WP13n piece 2: tc/'s `ar_wait_new` age gate (write_marginals,
        # optimize/ar.py ~line 249: "Exclude new ambiguities from AR
        # until converged") -- a newly-seeded N stays OUT of the LAMBDA
        # candidate set (vsat=0, not in the joint marginal) until it has
        # aged `ar_wait_new` epochs. Committed holds are exempt (tc/'s
        # held-state path likewise reports vsat=1). Default 0 = off =
        # WP13l shipped behavior, bit-identical.
        # WP13r lever 1 (COND_HOLD): committed keys leave the live-
        # marginal path entirely -- they are mirrored from the committed
        # integer with P = VAR_HOLDAMB in the held loop below (tc/'s
        # write_marginals held branch), never from the ISAM2 joint
        # marginal. Phase-2 scoped (this method only runs there).
        _cond_hold = bool(self.cond_hold) and getattr(self, 'phase', 1) == 2

        def _ar_eligible(s, f):
            if _cond_hold and (int(s), int(f)) in self._committed:
                return False  # conditioned-out: held loop below owns it
            # WP13r: quarantined sat (tc/'s `held_bad` in
            # write_marginals' float branch: `age >= ar_wait_new and
            # not held_bad`) never enters the candidate set.
            if self.ar_persist_bad_enable and int(s) in self._persist_bad_hold:
                return False
            if cp_amb is not None and (int(s), int(f)) not in cp_amb:
                return False
            if self.ar_wait_new > 0 and (s, f) not in self._committed:
                init_ep = self._amb_init_epoch.get((s, f))
                if init_ep is not None and (self.epoch2 - init_ep) < self.ar_wait_new:
                    return False
            return True

        for (s, f), k in _sorted_amb_items(self.amb_keys):
            if estimate.exists(k):
                self.nav.x[self.IB(s, f, self.nav.na)] = estimate.atDouble(k)
                if _ar_eligible(s, f):
                    self.nav.vsat[s - 1, f] = 1

        try:
            mg = (gtsam.Marginals(self.smoother.getFactors(), estimate)
                  if self.smoother else
                  gtsam.Marginals(self.isam.getFactorsUnsafe(), estimate))
            active = [(s, f, k) for (s, f), k in _sorted_amb_items(self.amb_keys)
                      if estimate.exists(k) and _ar_eligible(s, f)]
            if active:
                keys = gtsam.KeyVector()
                keys.append(key_x)
                for s, f, k in active:
                    keys.append(k)
                jm = mg.jointMarginalCovariance(keys)
                pose_cov6 = np.array(jm.at(key_x, key_x))
            else:
                jm = None
                pose_cov6 = np.array(mg.marginalCovariance(key_x))

            pos_cov_enu = pose_cov6[3:6, 3:6]
            pos_cov_ecef = self.R_enu2ecef @ pos_cov_enu @ self.R_enu2ecef.T
            self.nav.P[0:3, 0:3] = pos_cov_ecef

            if active:
                for s, f, k in active:
                    idx = self.IB(s, f, self.nav.na)
                    Pxn = np.array(jm.at(key_x, k))[3:6, :]
                    Pxn_ecef = self.R_enu2ecef @ Pxn
                    self.nav.P[0:3, idx] = Pxn_ecef[:, 0]
                    self.nav.P[idx, 0:3] = Pxn_ecef[:, 0]
                    self.nav.P[idx, idx] = jm.at(k, k)[0, 0]
                for i, (s1, f1, k1) in enumerate(active):
                    i1 = self.IB(s1, f1, self.nav.na)
                    for j, (s2, f2, k2) in enumerate(active):
                        if i >= j:
                            continue
                        i2 = self.IB(s2, f2, self.nav.na)
                        c = jm.at(k1, k2)[0, 0]
                        self.nav.P[i1, i2] = c
                        self.nav.P[i2, i1] = c
            self.last_joint_P = self.nav.P.copy()
        except RuntimeError:
            self.last_joint_P = None

        # WP13r lever 1 (COND_HOLD): tc/'s write_marginals held-state
        # branch (optimize/ar.py ~276-285), transcribed onto this
        # port's committed set: x = the committed integer (the value
        # tc/ pins as `held_value` -- both are LAMBDA's xa at hold
        # time), P diag = varholdamb (cssrlib VAR_HOLDAMB, 0.001),
        # cross-covariances stay 0 (nav.P was zeroed above; tc/ writes
        # only the diagonal too), vsat = 1 iff the (sat,f) carries a
        # DD-CP factor THIS epoch (tc/'s `is_visible`/cp_visible_sf
        # gate with its HELD_VSAT_HOLD_EPOCHS extension at default 0).
        # Runs AFTER the float/marginal writes so a committed key's
        # live graph estimate never reaches LAMBDA.
        if _cond_hold and self._committed:
            _n_cond = 0
            for (s, f) in sorted(self._committed):
                _val = float(self._committed[(s, f)])
                if not np.isfinite(_val):
                    continue
                _idx = self.IB(s, f, self.nav.na)
                self.nav.x[_idx] = _val
                self.nav.P[_idx, _idx] = self.VAR_HOLDAMB
                _vis = (cp_amb is None) or ((int(s), int(f)) in cp_amb)
                # WP13r: quarantined sat sits out (tc/'s held branch:
                # `1 if is_visible and not held_bad else 0`); the
                # committed integer itself survives for after expiry.
                if (self.ar_persist_bad_enable
                        and int(s) in self._persist_bad_hold):
                    _vis = False
                self.nav.vsat[s - 1, f] = 1 if _vis else 0
                if _vis:
                    _n_cond += 1
            if _n_cond:
                self.cond_hold_epochs += 1
                self.cond_hold_key_epochs += _n_cond

    # ---- WP13g (TASK_M7.md): DDPR-sanity / pose-innovation check,
    # transcribed from inuex35's `validation/postfit.py` `run_ddpr_sanity`
    # family. See `__init__`'s docstring comment for the scope this port
    # covers vs. what's left out (the DDPR-only-LS anchor stages).

    def _compute_main_dd_res(self, fi_start, graph_size, estimate, key_x, new_amb=None):
        """Main-graph DD residual RMS + worst-satellite residual, THIS
        epoch's own newly-added factors only -- the signal source for
        `_ddpr_sanity_trigger`/`_ddpr_sanity_fast_path` (postfit.py's
        `main_ddpr_residuals`, consumed via `info['main_ddpr_res']`/
        `info['main_ddpr_sat_worst']`).

        TWO disclosed adaptations from the literal reference, both
        empirically necessary (verified on the task's own tow
        188300-188620 debug window -- a literal PR-only, `.error()`-
        based signal never fired once, `sanity_fire_count=0` end to
        end):

        1. postfit.py's `main_ddpr_residuals` evaluates ONLY
           'Pseudorange'-named factors. That's a fine signal in tc/'s
           OWN pipeline because its `DdFactorBuilder.run()`
           (buildfactor/factors.py ~line 609, "PR factor: added even
           without CP") always adds a DD-PR factor REDUNDANTLY alongside
           a DD-CP factor for the same pair whenever both obs types
           exist. This port's `_build_dd_factors_arm` (established since
           WP13a, out of this task's scope to redesign) instead builds
           DD-PR and DD-CP MUTUALLY EXCLUSIVELY per pair (CP whenever
           carrier lock is available, PR only as the cycle-slip/no-lock
           fallback) -- so a literal PR-only signal is ~0 almost always
           on this port, INCLUDING throughout a wrong-but-LOCKED-
           ambiguity excursion (carrier lock holds -> no PR factor for
           that pair). Fold DD-CP's own residual in too.
        2. `fac.error(estimate)` on a Huber-`Robust`-wrapped factor (see
           `HUBER_CP`, WP13f) returns the ROBUSTIFIED (capped, linear-
           not-quadratic-beyond-threshold) cost, not the raw chi-square
           -- which is BY DESIGN desensitized to exactly the large
           residuals this signal needs to see (verified: with
           `.error()`, this signal stayed <1.7m even deep inside the
           100-300m excursion). Use `fac.evaluateError(...)` instead --
           GTSAM's raw, noise-model-INDEPENDENT residual vector (same
           value whether or not the factor is Robust-wrapped) -- mirrors
           the INTENT of postfit.py's own `_ddpr_factor_error`
           `rebuild_for_robust` branch (which reconstructs a literal
           chi-square from `evaluateError` specifically to bypass this
           same robust-desensitization for DD-PR; simplifies here to
           `res_m = |evaluateError()[0]|` directly, since the DD noise
           models here are 1-D isotropic and the chi-square round-trip
           postfit.py does is self-cancelling for a 1-D residual).

        THIRD disclosed adaptation (`new_amb`, optional): exclude any DD-
        CP factor touching a (sat,freq) FRESHLY seeded THIS epoch. A
        brand-new ambiguity's value is seeded from `pos_pred` (WP13f Bug
        2c) and can legitimately carry a multi-metre residual for a
        single epoch as a pure seeding artifact, not evidence the POSE is
        wrong -- empirically this was the exact cause of a false-positive
        catastrophic-fast-path fire on run2's otherwise-clean milestone
        window (`main_res=1169` from one freshly-seeded satellite,
        WP13G_REPORT.md) that made an already-recovering trajectory worse
        by resetting it. Excluding fresh ambiguities removes that false
        trigger without touching the real (sustained, multi-epoch,
        already-converged-ambiguity) excursion signal this mechanism
        exists to catch."""
        factors_all = (self.smoother.getFactors() if self.smoother
                       else self.isam.getFactorsUnsafe())
        fi_end = fi_start + graph_size
        try:
            pose = estimate.atPose3(key_x)
        except RuntimeError:
            return 0.0, 0.0
        res_sq = []
        per_sat = {}
        # WP13o: PR-only aggregates of the SAME evaluation pass -- the
        # tc/-literal signal (`postfit.main_ddpr_residuals` is PR-only)
        # consumed by `_do_ar` when AR_CONTEXT_PR_ONLY /
        # PER_SAT_GATE_PR_ONLY are set. Zero extra factor evaluations.
        res_sq_pr = []
        per_sat_pr = {}
        # WP13r: residuals of DD-CP rows whose BOTH sats are committed
        # holds -- the conditioned-world common-mode float-quality
        # signal consumed by COND_FIX_RES_MAX. Only both-committed rows
        # carry it: a row touching a YOUNG key has a free N that
        # absorbs the offset (residual ~0 regardless of pose quality),
        # which diluted an all-CP-rows median below usefulness (r6
        # probe: veto barely fired at 0.2 m). A float excursion lifts
        # every both-committed row together; canyon multipath spikes a
        # few -- median separates.
        res_cp = []
        fresh = new_amb or {}

        def _accum(fi, ref, j, is_cp, freq):
            if is_cp and ((int(ref), int(freq)) in fresh
                          or (int(j), int(freq)) in fresh):
                return
            fac = factors_all.at(fi)
            if fac is None:
                return
            try:
                if is_cp:
                    fkeys = fac.keys()
                    if len(fkeys) < 3:
                        # WP13s N4: held-folded DDCP (tc/'s 0/1-ary
                        # custom factor) -- no direct evaluateError
                        # binding; skip in this CP-folded diagnostic
                        # (the literal sanity signal is PR-only).
                        return
                    nr = estimate.atDouble(fkeys[1])
                    nj = estimate.atDouble(fkeys[2])
                    r = float(fac.evaluateError(pose, nr, nj)[0])
                else:
                    r = float(fac.evaluateError(pose)[0])
            except RuntimeError:
                return
            res_m = abs(r)
            res_sq.append(res_m * res_m)
            if res_m > per_sat.get(ref, 0.0):
                per_sat[ref] = res_m
            if res_m > per_sat.get(j, 0.0):
                per_sat[j] = res_m
            if not is_cp:
                res_m_pr = res_m
                if self.tc_literal and self.tc_lit_wres:
                    # WP13s literal: tc/'s main_ddpr_residuals "meters"
                    # are sqrt(2*fac.error)*sigma_pr*sqrt(2) -- i.e.
                    # WHITENED by the factor's ACTUAL (varerr,
                    # elevation-dependent) sigma and re-scaled by the
                    # NOMINAL one. Raw evaluateError meters overstate
                    # low-elevation rows by sigma_actual/sigma_nominal
                    # (up to ~7x), which made every residual-fed ladder
                    # (sanity/persist-bad/post-DDPR/dirty/ar_context)
                    # fire an order of magnitude more than tc/'s.
                    try:
                        err_w = fac.error(estimate)
                        res_m_pr = float(
                            np.sqrt(2.0 * max(err_w, 0.0))
                            * self.sigma_pr * np.sqrt(2))
                    except RuntimeError:
                        pass
                res_sq_pr.append(res_m_pr * res_m_pr)
                if res_m_pr > per_sat_pr.get(ref, 0.0):
                    per_sat_pr[ref] = res_m_pr
                if res_m_pr > per_sat_pr.get(j, 0.0):
                    per_sat_pr[j] = res_m_pr
            else:
                if ((int(ref), int(freq)) in self._committed
                        and (int(j), int(freq)) in self._committed):
                    res_cp.append(res_m)

        for fi, ref, j, f in self._last_ddpr_sat_tags:
            if fi_start <= fi < fi_end:
                _accum(fi, ref, j, False, f)
        for fi, (ref, j, f) in self._last_ddcp_meta.items():
            if fi_start <= fi < fi_end:
                _accum(fi, ref, j, True, f)

        rms = float(np.sqrt(np.mean(res_sq))) if res_sq else 0.0
        worst = max(per_sat.values()) if per_sat else 0.0
        # WP13k: stash the per-sat map itself (tc/'s `_last_main_ddpr_
        # per_sat`/`_cached_ddpr_res_pre` analogue) -- consumed THIS same
        # epoch by `_do_ar`'s subset-AR dirty-sat ranking, and next epoch
        # by `_build_dd_factors_arm`'s sat_badness CP-sigma scaling (a
        # deliberate 1-epoch lag there, matching tc/'s own architecture).
        self._main_ddpr_per_sat = dict(per_sat)
        # WP13o: PR-only counterparts (tc/-literal signal source).
        self._main_ddpr_res_pr = (
            float(np.sqrt(np.mean(res_sq_pr))) if res_sq_pr else 0.0)
        self._main_ddpr_per_sat_pr = dict(per_sat_pr)
        # WP13r: CP-only median (see res_cp above); -1 = no CP rows.
        self._main_ddcp_res_med = (
            float(np.median(res_cp)) if res_cp else -1.0)
        self._main_ddcp_res_n = len(res_cp)
        self._update_sat_badness()
        return rms, worst

    def _compute_res_at_pred(self, nf_before, graph_size, pred, key_x):
        """DD-PR residual evaluated at the IMU-predicted pose (transcribed
        from `_compute_res_at_pred`, LITERAL PR-only, matching the
        reference exactly here -- not extended with DD-CP the way
        `_compute_main_dd_res` is). Returns `None` when this epoch added
        no DD-PR factor at all (common on this port -- see
        `_compute_main_dd_res`'s docstring), which `_sanity_report_
        translation` already treats as "can't verify, don't replace" --
        the correct behavior, not the `0.0` ("perfectly clean") a naive
        empty-list default would give.

        Why NOT fold in DD-CP here (unlike `_compute_main_dd_res`,
        deliberately): this residual's ONLY job is
        `_sanity_report_translation`'s "is the IMU-predicted pose
        trustworthy enough to report" gate. Empirically (this task's own
        debug window, WP13G_REPORT.md), folding DD-CP in here made this
        gate ALWAYS fail during the excursion, defeating the whole
        pose-replace mechanism -- because a WRONG ambiguity disagrees
        with ANY plausible pose's CP prediction, including the IMU's, so
        a CP-inclusive residual can never look "clean" precisely in the
        failure mode this mechanism exists to catch. A literal PR-only
        residual doesn't have that problem (PR factors carry no
        ambiguity), matching the reference's own intent even though PR
        factors are sparse here."""
        try:
            factors_all = (self.smoother.getFactors() if self.smoother
                           else self.isam.getFactorsUnsafe())
            pose_pred = pred.pose()
            fi_end = nf_before + graph_size
            res_sq = []
            v_pred = None
            if self.tc_literal and self.tc_lit_wres:
                v_pred = gtsam.Values()
                v_pred.insert(key_x, pose_pred)
            for fi, _ref, _j, _f in self._last_ddpr_sat_tags:
                if fi < nf_before or fi >= fi_end:
                    continue
                fac = factors_all.at(fi)
                if fac is None:
                    continue
                try:
                    if self.tc_literal and self.tc_lit_wres:
                        # tc/'s scale (see _compute_main_dd_res): the
                        # whitened-then-renormalized pseudo-meters.
                        err_w = fac.error(v_pred)
                        r = float(np.sqrt(2.0 * max(err_w, 0.0))
                                  * self.sigma_pr * np.sqrt(2))
                    else:
                        r = float(fac.evaluateError(pose_pred)[0])
                except RuntimeError:
                    continue
                res_sq.append(r * r)
            return float(np.sqrt(np.mean(res_sq))) if res_sq else None
        except RuntimeError:
            return float('inf')

    def _reset_all_ambiguities_with_cp_hold(self):
        """Warm reset: abandon EVERY currently-tracked ambiguity
        (transcribed from postfit.py's `reset_ambiguities_with_cp_hold`,
        REUSING this file's own `_release_ambiguity` -- the same per-
        (sat,freq) abandon mechanism WP13c's hold-release and WP13d's FDE
        cycle-slip reset already use -- rather than duplicating a
        parallel graph-surgery path). Disclosed adaptation: unlike the
        reference (which ALSO structurally removes each abandoned
        ambiguity's own factors from the ISAM2 Bayes tree via
        `removeFactorIndices`), this relies on the SAME passive-expiry
        mechanism `_release_ambiguity` already uses everywhere else in
        this file -- an abandoned key's timestamp stops being refreshed,
        so it ages out of the `--lag` window on its own (already an
        accepted equivalence here, see WP13d's `FDE_REMOVE_CP=0` variant,
        module docstring)."""
        keys = list(self.amb_keys.keys())
        for (s, f) in keys:
            self._release_ambiguity(s, f, reason='sanity_wipe')
        # WP13r lever 2: the reference's namesake cp-hold arming
        # (recovery.py `reset_ambiguities_with_cp_hold` line ~116:
        # `tc._recov_cp_hold = effective_cp_hold_epochs(tc)`) -- the
        # piece WP13g's transcription left out. Opt-in.
        if self.recov_cp_hold_epochs > 0:
            if self._recov_cp_hold_remaining < self.recov_cp_hold_epochs:
                self.recov_cp_hold_trigger_count += 1
            self._recov_cp_hold_remaining = max(
                self._recov_cp_hold_remaining, self.recov_cp_hold_epochs)
            self._recov_cp_release_streak = 0
            # WP13s literal: tc/'s reset_ambiguities_with_cp_hold also
            # clears the whole sat-quality state.
            if self.tc_literal and self._sq is not None:
                self._sq.clear()
        return len(keys)

    def _sanity_report_translation(self, pose_tc, pred, pred_res):
        """Pose translation+rotation to report when sanity recovery
        fires (transcribed from `_sanity_report_translation`): if the
        IMU-predicted pose's OWN DD residual (`pred_res`) is clean
        (<= `sanity_pose_replace_thresh`) AND its translation disagrees
        with the just-solved `pose_tc` by MORE than that same threshold,
        REPLACE the reported translation (and rotation) with the IMU
        prediction's -- otherwise keep `pose_tc` as-is. Returns
        `(translation_enu, rotation, replaced)`."""
        tc_t = np.array(pose_tc.translation())
        thr = self.sanity_pose_replace_thresh
        if thr <= 0 or pred is None:
            return tc_t, pose_tc.rotation(), False
        if pred_res is None or pred_res > thr:
            return tc_t, pose_tc.rotation(), False
        pred_t = np.array(pred.pose().translation())
        gap = float(np.linalg.norm(tc_t - pred_t))
        if gap > thr:
            return pred_t, pred.pose().rotation(), True
        return tc_t, pose_tc.rotation(), False

    def _apply_sanity_report(self, pose_tc, pred, pred_res, key_x):
        """Shared tail of `_apply_sanity_reset`/`_ddpr_sanity_fast_path`:
        decide the reported translation via `_sanity_report_translation`,
        convert it to an antenna ECEF position (same lever-arm convention
        `_write_back_tc` uses) and write it + force `nav.smode=5` (FLT --
        sanity firing NEVER reports a fix, matching the reference's
        `finalize_epoch(..., 'FLT', 0, ...)`)."""
        t, rot, replaced = self._sanity_report_translation(pose_tc, pred, pred_res)
        pose_report = gtsam.Pose3(rot, gtsam.Point3(*t))
        pose_ecef = self.ecef_T_nav.compose(pose_report)
        R = pose_ecef.rotation().matrix()
        self.nav.x[0:3] = np.array(pose_ecef.translation()) + R @ self.lever_arm
        self.nav.smode = 5
        if replaced:
            self.sanity_replace_count += 1
        return replaced

    def _ddpr_sanity_check(self, nf_before, graph_size, pose_tc, pred, key_x, nb):
        """Orchestrates the transcribed subset of `run_ddpr_sanity`:
        Stage 1 (`_ddpr_sanity_trigger`) -> Stage 2 fast path
        (`_ddpr_sanity_fast_path`, catastrophic + no-fix-this-epoch +
        one dominant bad satellite -> immediate warm reset, no persist
        wait) -> Stage 3 persist (`_ddpr_sanity_persist`, N consecutive
        trigger-passing epochs -> warm reset). A persisted-bad reset goes
        straight to the same reset+report action the reference's own
        anchor-FALLBACK path uses when its DDPR-only-LS anchor is
        unavailable/untrusted (`_ddpr_sanity_anchor_fallback` ->
        `_apply_sanity_reset`) -- this port has no `_ddpr_only_position`
        solver to attempt the anchor refinement itself (out of scope,
        see class docstring). Returns True iff sanity fired this epoch
        (caller should treat the epoch as reported, forced FLT)."""
        # WP13r: SANITY_PR_ONLY=1 re-points every sanity signal at the
        # PR-only aggregates -- the tc/-LITERAL quantity
        # (postfit.main_ddpr_residuals evaluates only Pseudorange
        # factors). The WP13g CP-folded adaptation was justified when
        # DD-PR and DD-CP were built mutually exclusively (PR-only
        # signal ~0 always); WP13i's DD-PR-always removed that reason,
        # and the CP-folded signal measured PATHOLOGICAL under the
        # lever-2 stack on full run1 (sanity fired 399x -- every dirty
        # CP stretch, not pose breaks -- re-wiping the pool into a
        # wrong basin: post-tunnel re-fix 24.0% but 56% false).
        if self.sanity_pr_only:
            main_res = float(self._main_ddpr_res_pr or 0.0)
            _ps_pr = self._main_ddpr_per_sat_pr or {}
            worst_res_sanity = float(max(_ps_pr.values())) if _ps_pr else 0.0
        else:
            main_res = self._main_ddpr_res
            worst_res_sanity = self._main_ddpr_worst_res
        # Stage 1: trigger. A clean residual resets the persist counter
        # and does nothing else this epoch.
        if main_res <= self.main_ddpr_res_thresh:
            self._ddpr_bad_count = 0
            return False

        # WP13r lever 2: tc/'s `_ddpr_multipath_dominated` skip
        # (postfit.py, run BETWEEN trigger and everything else): when
        # ONE satellite dominates the per-sat DD-PR residual
        # distribution (max/median > SANITY_MAX_MEDIAN_RATIO with at
        # least SANITY_MAX_MEDIAN_MIN_SATS sats), the spike is canyon
        # multipath, not a wrong pose -- skip the whole ladder this
        # epoch WITHOUT touching the persist counter (tc/ returns None
        # after `_ddpr_sanity_trigger` already passed; the counter
        # increments only in its later persist stage). PR-only per-sat
        # map (tc/'s own signal), CP-folded fallback when absent.
        if self.sanity_max_median_ratio > 0:
            _per_sat_m = (self._main_ddpr_per_sat_pr
                          or self._main_ddpr_per_sat or {})
            if len(_per_sat_m) >= self.sanity_max_median_min_sats:
                _vals = sorted(float(v) for v in _per_sat_m.values())
                _med = _vals[len(_vals) // 2]
                if _med > 1e-3 and _vals[-1] / _med > self.sanity_max_median_ratio:
                    self.sanity_skip_multipath_count += 1
                    return False

        # Count THIS epoch as bad before the fast-path check (disclosed
        # adaptation, `ddpr_fast_min_persist`, default 2 -- see below).
        self._ddpr_bad_count += 1

        # WP13r lever 2: tc/'s `_ddpr_sanity_persist` fires
        # `trigger_cp_hold('ddpr_main_res')` on EVERY sanity-triggered
        # bad epoch (state.py line ~52: max-arm + streak reset), i.e.
        # DD-CP construction stays suppressed for the whole bad stretch
        # -- the float is PR-pulled while the residual says the CP
        # basin is dirty. Opt-in via RECOV_CP_HOLD_EPOCHS.
        if self.recov_cp_hold_epochs > 0:
            if self._recov_cp_hold_remaining < self.recov_cp_hold_epochs:
                self.recov_cp_hold_trigger_count += 1
            self._recov_cp_hold_remaining = max(
                self._recov_cp_hold_remaining, self.recov_cp_hold_epochs)
            self._recov_cp_release_streak = 0
            # WP13s literal: tc/'s trigger_cp_hold ('ddpr_main_res',
            # fired on EVERY sanity-bad epoch) clears sat-quality state.
            if self.tc_literal and self._sq is not None:
                self._sq.clear()

        pred_res = self._compute_res_at_pred(nf_before, graph_size, pred, key_x)

        # Stage 2: catastrophic fast path -- fires without waiting for
        # the FULL persist count, when this epoch's own residual is a
        # clear spike, LAMBDA did NOT resolve a fix this epoch (nb<=0 --
        # a genuine fix would be strong contrary evidence against "the
        # pose is wrong"), and one satellite dominates the spike (not a
        # geometry-wide effect).
        #
        # Disclosed adaptation (`ddpr_fast_min_persist`, default 2, NOT
        # the reference's literal 1-epoch-is-enough fast path):
        # empirically, this port's CP-inclusive signal (see
        # `_compute_main_dd_res`'s docstring) produces occasional single-
        # epoch spikes into the hundreds-of-metres range that are NOT a
        # sustained wrong-fix -- verified concretely on run2's otherwise-
        # clean 3000-epoch milestone window: one epoch spiked to
        # `main_res=1169` (excluding freshly-seeded ambiguities did NOT
        # remove it -- some other single-epoch transient), the reference-
        # literal 1-epoch fast path reset on it, and that reset made an
        # ALREADY-recovering trajectory worse (7.8m and falling ->
        # resets, then plateaus at 7.5-7.8m -- see WP13G_REPORT.md).
        # Requiring 2 consecutive bad epochs before the fast path (not
        # the full `ddpr_sanity_persist=3`) keeps it reacting fast to a
        # REAL sustained spike (which stays bad for many epochs, per the
        # run1 excursion) while filtering exactly this one-epoch-only
        # false-alarm class.
        if (main_res > self.main_ddpr_res_catastrophic
                and int(nb) <= 0
                and worst_res_sanity >= self.ddpr_fast_worst_sat_min
                and self._ddpr_bad_count >= self.ddpr_fast_min_persist):
            self._ddpr_bad_count = 0
            self.sanity_fire_count += 1
            self.sanity_fast_count += 1
            if self.sanity_do_reset:
                self._reset_all_ambiguities_with_cp_hold()
            if self.tc_literal and self.sanity_break_pim:
                self._pim_discontinuity = True
            self._apply_sanity_report(pose_tc, pred, pred_res, key_x)
            return True

        # Stage 3: persist -- require N consecutive trigger-passing
        # epochs before acting (avoid over-triggering on one noisy
        # epoch; RELEASE_K's per-(sat,freq) residual signal has already
        # failed 4x on this exact failure mode -- see WP13F_REPORT.md --
        # this is a DIFFERENT, main-graph-residual-vs-IMU-prediction
        # signal).
        if self._ddpr_bad_count < self.ddpr_sanity_persist:
            return False
        # WP13s literal: tc/'s `_ddpr_sanity_gdop_ok` (preset
        # sanity_max_gdop=5.0) -- abort the persisted reset when the
        # geometry is too weak to trust the anchor (counter/cp-hold
        # arming above already happened, exactly like tc/).
        if (self.tc_literal and self.sanity_max_gdop > 0
                and getattr(self, '_last_gdop', 0.0) > self.sanity_max_gdop):
            self.sanity_skip_gdop_count += 1
            return False
        self._ddpr_bad_count = 0
        self.sanity_fire_count += 1
        if self.sanity_do_reset:
            self._reset_all_ambiguities_with_cp_hold()
        if self.tc_literal and self.sanity_break_pim:
            self._pim_discontinuity = True
        self._apply_sanity_report(pose_tc, pred, pred_res, key_x)
        return True

    # ---- WP13s (TASK_M19.md, べた移植): tc/-literal pipeline stages,
    # driving tc/'s OWN imported SatQualityState / compute_cp_build_policy
    # where interfaces allow, verbatim transcriptions otherwise. All
    # gated on self.tc_literal (default 0 = bit-identical).

    def _tc_literal_gate_stage(self, obs, sat, el, iu):
        """tc/'s preprocess/gate.py phases 2-5, literal: per-sat el/SNR
        telemetry, sq.tick (quarantine/cooldown/suspect decay + forced
        holds incl. persist-bad expansion), the dirty-sat reset ladder
        (`_apply_dirty_sat_reset`, preset 5/2/15.0/3), and
        update_cp_lock. Runs at the gate position (pre-build), reading
        the PREVIOUS epoch's post-fit per-sat DDPR residuals exactly
        like tc/'s `tc._mres_signals.per_sat`."""
        sq = self._sq
        cfg = self.cfg
        ns = len(sat)
        # per-sat telemetry (gate.py _collect_sat_telemetry_and_holds)
        self._sat_el_deg = {
            int(sat[i]): float(np.degrees(el[i])) for i in range(ns)}
        sat_snr = {}
        for i in range(ns):
            try:
                vals = np.asarray(obs.S[iu[i]], float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if vals.size:
                    sat_snr[int(sat[i])] = float(np.max(vals))
            except (ValueError, TypeError, IndexError):
                pass
        self._sat_snr_dbhz = sat_snr

        amb_keys_now = dict(self.amb_keys)
        forced_hold = sq.tick(amb_keys_now, {})

        # dirty-sat reset (gate.py _apply_dirty_sat_reset, literal)
        penalized_hold = set()
        q_map = sq.hold_quarantine
        cooldown_map = sq.dirty_cooldown
        suspect_map = sq.dirty_suspect
        if (bool(cfg.cp_hold_dirty_reset_enable)
                and cfg.recov_cp_release_thresh > 0):
            thr = float(cfg.recov_cp_release_thresh)
            hold_n = int(cfg.cp_hold_dirty_reset_hold)
            if hold_n <= 0:
                hold_n = int(self.recov_cp_hold_epochs)
            suspect_need = max(1, int(cfg.cp_hold_dirty_reset_suspect_count))
            cooldown_n = max(0, int(cfg.cp_hold_dirty_reset_cooldown))
            per_sat = self._main_ddpr_per_sat_pr or {}
            for (s, f) in list(amb_keys_now.keys()):
                key = (s, f)
                if key in forced_hold:
                    continue
                res_s = float(per_sat.get(s, 0.0))
                if res_s <= thr:
                    suspect_map.pop(key, None)
                    continue
                cppr_count = int(self.rejc_cp_pr.get(key, 0))
                if cppr_count <= 0:
                    suspect_map.pop(key, None)
                    continue
                streak = suspect_map.get(key, 0) + 1
                suspect_map[key] = streak
                if cooldown_map.get(key, 0) > 0:
                    penalized_hold.add(key)
                    continue
                if streak < suspect_need:
                    penalized_hold.add(key)
                    continue
                # reset: amb_gen bump + clear_hold + counter wipes
                self._release_ambiguity(s, f, reason='dirty_reset')
                self.rejc_cp_pr.pop(key, None)
                self._rejc_post_ddpr.pop(key, None)
                self._fix_streak.pop(key, None)
                q_map[key] = hold_n
                if cooldown_n > 0:
                    cooldown_map[key] = cooldown_n
                suspect_map.pop(key, None)
                forced_hold.add(key)
                self.dirty_reset_count += 1
            if self.dirty_reset_count:
                sq.reset_cp_lock(forced_hold)
        sq.forced_hold_per_sat = forced_hold
        sq.penalized_per_sat = penalized_hold

        visible_keys = {(int(s), f) for s in sat for f in range(self.nav.nf)}
        sq.update_cp_lock(visible_keys,
                          slip_keys=getattr(self, '_last_slip_keys', None),
                          forced_hold=forced_hold)

    def _tc_literal_reset_failing_n(self):
        """tc/'s buildfactor/factors.py `_reset_persistently_failing_n`,
        literal: wipe (hard release) any (sat,freq) whose CP-vs-PR
        innovation reject count or post-fit DDPR reject count crossed
        the preset bars (cp_pr_rejc_max=2 / post_ddpr_reset_count=3)."""
        cfg = self.cfg
        cp_pr_active = (cfg.cp_pr_innov_thresh > 0
                        and int(cfg.cp_pr_rejc_max) > 0)
        post_active = (cfg.post_ddpr_reset_thresh > 0
                       and int(cfg.post_ddpr_reset_count) > 0)
        if not (cp_pr_active or post_active):
            return
        for key in set(list(self.rejc_cp_pr.keys())
                       + list(self._rejc_post_ddpr.keys())):
            wipe = ((cp_pr_active
                     and self.rejc_cp_pr.get(key, 0) >= int(cfg.cp_pr_rejc_max))
                    or (post_active
                        and self._rejc_post_ddpr.get(key, 0)
                        >= int(cfg.post_ddpr_reset_count)))
            if not wipe:
                continue
            s, f = key
            if key in self.amb_keys or key in self._committed:
                self._release_ambiguity(s, f, reason='rejc_wipe')
                self.rejc_wipe_count += 1
            self.rejc_cp_pr.pop(key, None)
            self._rejc_post_ddpr.pop(key, None)
            self._fix_streak.pop(key, None)

    def _tc_literal_postfit_quality(self):
        """tc/'s optimize/stage.py `_compute_postfit_diagnostics` quality
        bookkeeping, literal, driven by THIS epoch's PR-only per-sat
        residuals (already computed by `_log_main_ddpr`, tc/'s
        `main_ddpr_residuals` analogue): persist-bad streak/quarantine,
        reference / observation / pair quality EWMAs (imported
        SatQualityState methods), post-fit DDPR per-sat counters, and
        the `pair_bad_max` context signal."""
        sq = self._sq
        cfg = self.cfg
        per_sat = {int(k): float(v)
                   for k, v in (self._main_ddpr_per_sat_pr or {}).items()}
        # alias consumed by sq.sat_badness (tc._last_main_ddpr_per_sat)
        self._last_main_ddpr_per_sat = per_sat
        worst_sat = (max(per_sat, key=per_sat.get) if per_sat else None)

        # persist-bad (stage.py, preset on: 2.0 m / 4 / 10)
        if self.ar_persist_bad_enable:
            thr = float(self.ar_persist_bad_res_thresh)
            streak_need = max(1, int(self.ar_persist_bad_streak))
            hold_len = max(1, int(self.ar_persist_bad_hold))
            seen = set()
            for s, rmax in per_sat.items():
                seen.add(s)
                if rmax > thr:
                    st = sq.persist_bad_streak.get(s, 0) + 1
                    sq.persist_bad_streak[s] = st
                    if st >= streak_need:
                        if sq.persist_bad_hold.get(s, 0) <= 0:
                            self.persist_bad_hold_count += 1
                        sq.persist_bad_hold[s] = max(
                            int(sq.persist_bad_hold.get(s, 0)), hold_len)
                        for f in range(self.nav.nf):
                            key = (s, f)
                            # tc/: amb_gen bump regenerates the FLOAT N;
                            # a held integer is NOT cleared here.
                            if key in self.amb_keys and key not in self._committed:
                                self._release_ambiguity(s, f, reason='persist_bad')
                                self.persist_bad_release_count += 1
                            self.rejc_cp_pr.pop(key, None)
                            self._rejc_post_ddpr.pop(key, None)
                            self._fix_streak.pop(key, None)
                else:
                    sq.persist_bad_streak[s] = 0
            for s in list(sq.persist_bad_streak.keys()):
                if s not in seen:
                    sq.persist_bad_streak[s] = 0

        # reference / observation / pair quality EWMAs (imported)
        cppr_sat = {}
        for (s, f), c in self.rejc_cp_pr.items():
            cppr_sat[int(s)] = max(cppr_sat.get(int(s), 0), int(c))
        sq.update_reference_quality(cfg, dict(self.ref_sats), per_sat)
        sq.update_observation_quality(
            cfg, per_sat, worst_sat=worst_sat, cppr_sat=cppr_sat,
            sat_el_deg=self._sat_el_deg, sat_snr_dbhz=self._sat_snr_dbhz)
        pair_rows = self._last_pair_rows or []
        sq.update_pair_quality(cfg, pair_rows)
        self._last_pair_bad_max = max(
            (float(sq.recent_pair_bad.get(
                (int(r['ref']), int(r['sat']), int(r['freq'])), 0.0) or 0.0)
             for r in pair_rows), default=0.0)

        # post-fit DDPR per-(sat,f) counters (stage.py, preset 1.5 m / 3)
        if cfg.post_ddpr_reset_thresh > 0 and int(cfg.post_ddpr_reset_count) > 0:
            thr_post = float(cfg.post_ddpr_reset_thresh)
            sats_seen = set(per_sat.keys())
            for s, rmax in per_sat.items():
                for f in range(self.nav.nf):
                    key = (s, f)
                    if rmax > thr_post:
                        self._rejc_post_ddpr[key] = (
                            self._rejc_post_ddpr.get(key, 0) + 1)
                    else:
                        self._rejc_post_ddpr.pop(key, None)
            for (s, f) in list(self._rejc_post_ddpr.keys()):
                if s not in sats_seen:
                    self._rejc_post_ddpr.pop((s, f), None)

    def _tc_literal_release_suspicious_held(self):
        """tc/'s validation/postprocess.py
        `_release_suspicious_held_on_flt`, literal: on a FLT epoch,
        soft-release (seeded, sigma-0.1-cycle re-seed pin) the single
        most suspicious committed ambiguity, scored by post-fit per-sat
        DDPR residual + worst-sat bonus + CP-vs-PR reject count."""
        per_sat = {int(k): float(v)
                   for k, v in (self._main_ddpr_per_sat_pr or {}).items()}
        worst = (max(per_sat.items(), key=lambda kv: kv[1])
                 if per_sat else None)
        res_thr = max(2.0, 0.5 * float(self.ar_context_worst_sat_max))
        candidates = []
        for (s, f) in self._committed:
            s_i, f_i = int(s), int(f)
            sat_res = per_sat.get(s_i, 0.0)
            cppr = int(self.rejc_cp_pr.get((s_i, f_i), 0))
            score = 0.0
            if sat_res >= res_thr:
                score += sat_res
            if worst is not None and s_i == int(worst[0]) and worst[1] >= res_thr:
                score += max(1.0, 0.25 * float(worst[1]))
            if cppr > 0:
                score += 10.0 + float(cppr)
            if score > 0.0:
                candidates.append((score, sat_res, cppr, s_i, f_i))
        if not candidates:
            return 0
        candidates.sort(reverse=True)
        _, _, _, s_i, f_i = candidates[0]
        self._release_ambiguity(s_i, f_i, reason='flt_release')
        self._fix_streak.pop((s_i, f_i), None)
        self.flt_release_count += 1
        return 1

    def _tc_literal_postprocess(self, candidates_fixed):
        """tc/'s postprocess.py `_update_streaks_and_post_hooks`,
        literal: per-(sat,freq) fix-streak update; on FLT, zero all
        streaks and release the single most suspicious held."""
        if self.nav.smode == 4:
            cand = set(candidates_fixed or ())
            for key in list(self._fix_streak.keys()):
                if key not in cand:
                    self._fix_streak[key] = 0
            for key in cand:
                self._fix_streak[key] = self._fix_streak.get(key, 0) + 1
        else:
            for key in list(self._fix_streak.keys()):
                self._fix_streak[key] = 0
            self._tc_literal_release_suspicious_held()

    def _process_phase2(self, obs, obsb):
        """Phase-2 per-epoch update (TASK_M5.md items 2-3): CombinedImuFactor
        chain (preintegrated via tc/'s `build_pim`, reused directly) +
        Arm DD factors, into the SAME ISAM2/IFLS instance `_transition_to_
        phase2` created. Abandon-and-roll-back on any failure (nv<4, too
        few sats, or the smoother update itself throwing) rolls `self.
        imu_idx` back to its pre-attempt snapshot so the NEXT successful
        epoch's PIM re-integrates the FULL span since the last COMMITTED
        epoch (not just since the abandoned attempt) -- this is what
        keeps the CombinedImuFactor's own time span consistent with the
        Pose3 key pair it actually connects, across an arbitrary run of
        skipped epochs (mirrors the parent's `last_valid_epoch`-not-
        `epoch-1` between-factor anchoring for the same reason)."""
        prev_ep = self.last_valid_epoch
        prev_pose = self.current_estimate.atPose3(self.PX(prev_ep))
        prev_vel = np.array(self.current_estimate.atVector(self.PV(prev_ep)))
        prev_bias = self.current_estimate.atConstantBias(self.PB(prev_ep))

        _, tow = time2gpst(obs.t)
        idx_snapshot = self.imu_idx
        pim, n_int, gyro_mean, new_idx = build_pim(
            self.imu_params, prev_bias, self.imu_data, self.imu_idx, target_tow=tow)
        if n_int < 1:
            self.imu_idx = idx_snapshot
            self.epoch2 += 1
            self.nav.t = obs.t
            return
        self.imu_idx = new_idx

        nav_pred = pim.predict(gtsam.NavState(prev_pose, prev_vel), prev_bias)
        pose_pred = nav_pred.pose()
        vel_pred = np.array(nav_pred.velocity())
        pos_pred_ecef = np.array(self.ecef_T_nav.compose(pose_pred).translation())

        prep = self.prepare_double_difference_measurements(
            obs, obsb, pos_pred=pos_pred_ecef, cs=None, orb=None, bsx=None,
            compute_zdres=False)
        if prep is None:
            self.imu_idx = idx_snapshot
            self.epoch2 += 1
            self.nav.t = obs.t
            return

        rs = prep['rs']
        vs = prep['vs']
        dts = prep['dts']
        rsb = prep['rsb']
        iu = prep['iu']
        ir = prep['ir']
        sat = prep['sat']
        el = prep['el']
        obs_ = prep['obs_sd']

        # WP13n: refresh the real epoch dt (tc/'s `_update_epoch_dt`) --
        # consumed by the varerr sigma's clock-stability term.
        if self.nav.t.time > 0:
            _dt_ep = timediff(obs.t, self.nav.t)
            if _dt_ep > 0:
                self._epoch_dt = float(_dt_ep)

        # WP13p: churn-triage context map (print-only, built only when
        # the debug env is set) -- consumed by `_release_ambiguity`'s
        # WP13P_DBG_CHURN line for EVERY release this epoch (slip/MW/
        # outage here, FDE-CP later in this same epoch).
        if os.environ.get('WP13P_DBG_CHURN'):
            _elm_dbg = {int(sat[_i]): float(np.rad2deg(el[_i]))
                        for _i in range(len(sat))}
            _cnrm_dbg = {}
            for _i in range(len(sat)):
                for _f in range(self.nav.nf):
                    try:
                        _cnrm_dbg[(int(sat[_i]), _f)] = float(obs.S[iu[_i], _f])
                    except (IndexError, TypeError, ValueError):
                        pass
            self._churn_dbg_ctx = {'ep': self.epoch2, 'el': _elm_dbg,
                                   'cnr': _cnrm_dbg}

        # WP13n piece 1: slip/outage GTSAM ambiguity resets -- MUST run
        # BEFORE `_manage_ambiguities` (cssrlib's `update_ambiguities`
        # consumes and clears `nav.slip`; tc/'s own slip stage likewise
        # runs upstream of factor construction).
        self._reset_slipped_ambiguities(
            obs, obsb, obs_, prep['sat'], prep['iu'], prep['ir'],
            self.epoch2, el=el)

        self._manage_ambiguities(obs_)

        # WP13r lever 2: recovery CP-hold tick -- tc/'s gate stage
        # (preprocess/gate.py ~81/113-128): snapshot the active flag for
        # this epoch's factor build BEFORE decrementing (tc/'s
        # `ed.skip_cp_now`), then decrement; with
        # RECOV_CP_RELEASE_THRESH>0 the window re-arms 1 epoch at a
        # time until the PR-only main residual (last committed epoch's,
        # tc/'s `_last_main_ddpr_res`) has been clean for
        # RECOV_CP_RELEASE_COUNT consecutive held epochs.
        self._recov_cp_skip_now = self._recov_cp_hold_remaining > 0
        if self._recov_cp_skip_now:
            self.recov_cp_hold_cp_suppressed_epochs += 1
            self._recov_cp_hold_remaining -= 1
            if self.recov_cp_release_thresh > 0:
                _last_res_pr = float(self._main_ddpr_res_pr or 0.0)
                if 0.0 < _last_res_pr <= self.recov_cp_release_thresh:
                    self._recov_cp_release_streak += 1
                else:
                    self._recov_cp_release_streak = 0
                if (self._recov_cp_hold_remaining <= 0
                        and self._recov_cp_release_streak
                        < self.recov_cp_release_count):
                    self._recov_cp_hold_remaining = 1

        # WP13r: per-sat quarantine tick -- tc/'s sat_quality.tick
        # (gate stage, pre-build): decrement each active hold; expired
        # sats rejoin the AR candidate set / CP build.
        if (not self.tc_literal) and self.ar_persist_bad_enable and self._persist_bad_hold:
            for _s_pb in list(self._persist_bad_hold.keys()):
                _rem_pb = int(self._persist_bad_hold[_s_pb]) - 1
                if _rem_pb > 0:
                    self._persist_bad_hold[_s_pb] = _rem_pb
                else:
                    self._persist_bad_hold.pop(_s_pb, None)

        ns = len(iu)
        # WP13s literal: tc/'s GDOP gate (gate.py `_check_gdop_nsat_gate`,
        # preset gdop_max=10): a bad-geometry epoch never reaches the DD
        # build. tc/ advances the graph IMU-only (`process_gdop_skip`);
        # this port's abandon-epoch path carries the pose and lets the
        # next committed epoch's PIM re-integrate the full gap --
        # disclosed approximation (audit P7).
        _gdop_bad = False
        if self.tc_literal and _tc_compute_gdop is not None:
            _gdop_val = float(_tc_compute_gdop(
                np.array(pose_pred.translation()), ns, rs, iu,
                self.R_enu2ecef, np.array(self.nav.rb, dtype=float)))
            self._last_gdop = _gdop_val
            _gdop_bad = not (_gdop_val < 10.0)
            if _gdop_bad:
                self.gdop_skip_count += 1
        if ns < 6 or _gdop_bad:
            self.imu_idx = idx_snapshot
            self.epoch2 += 1
            self.nav.t = obs.t
            return

        # WP13s literal gate stage: tc/'s preprocess/gate.py phases 2-5
        # via the IMPORTED SatQualityState (tick + update_cp_lock +
        # dirty-sat reset ladder). Runs AFTER the nsat gate, exactly
        # where tc/'s gate.run places it (its GDOP/nsat early-return
        # comes first), replacing the reduced WP13r tick above.
        if self.tc_literal:
            self._tc_literal_gate_stage(obs, sat, el, iu)

        ep = self.epoch2
        key_x, key_v, key_b = self.PX(ep), self.PV(ep), self.PB(ep)
        graph = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()
        new_values.insert(key_x, pose_pred)
        new_values.insert(key_v, vel_pred)
        new_values.insert(key_b, prev_bias)

        # WP13s literal: tc/'s PIM-discontinuity break
        # (imu_preintegration.add_imu_chain): the epoch after a sanity
        # reset is UNTIED from the previous pose -- priors at the IMU
        # prediction (trans sigma 100 m = near-free) instead of the
        # CombinedImuFactor chain, letting this epoch's DD-PR pull the
        # pose back into the right basin.
        _pim_broke = False
        if self.tc_literal and getattr(self, '_pim_discontinuity', False):
            _pim_broke = True
            _tsig = float(self.pim_break_trans_sigma)
            graph.addPriorPose3(key_x, pose_pred,
                gtsam.noiseModel.Diagonal.Sigmas(np.array(
                    [0.1, 0.1, 0.3, _tsig, _tsig, _tsig])))
            graph.addPriorVector(key_v, vel_pred,
                gtsam.noiseModel.Isotropic.Sigma(3, 2.0))
            graph.addPriorConstantBias(key_b, prev_bias,
                gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
            self._pim_discontinuity = False
            self.pim_break_count += 1
        else:
            graph.add(gtsam.CombinedImuFactor(
                self.PX(prev_ep), self.PV(prev_ep), key_x, key_v,
                self.PB(prev_ep), key_b, pim))
            graph.add(gtsam.BetweenFactorConstantBias(
                self.PB(prev_ep), key_b,
                gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3)),
                self._bias_between_noise()))
        # Bias anchor (TASK_M5.md "keep it simple" -- one absolute prior
        # mode, not tc/'s full `bias_prior_anchor` 3-way dispatch): mode 1
        # pins the bias close to the Phase-1->2 transition's initial
        # stationary-window estimate FOREVER, mode 2 (tc/'s own default)
        # re-anchors to the PREVIOUS epoch's own committed estimate each
        # time (a soft "don't jump" regularizer, not a hard pin to a
        # possibly-wrong initial guess), mode 0 omits it (rely on the
        # BetweenFactorConstantBias random walk alone).
        if _pim_broke:
            pass  # tc/'s break path adds only its own sigma-0.01 anchor
        elif self.bias_prior_mode == 1:
            graph.addPriorConstantBias(key_b, self.tc_bias_init, self._bias_prior_noise())
        elif self.bias_prior_mode == 2:
            graph.addPriorConstantBias(key_b, prev_bias, self._bias_prior_noise())

        nv, new_amb, cp_amb = self._build_dd_factors_arm(
            graph, new_values, obs, obsb, obs_, rs, rsb, sat, el, iu, ir,
            pos_pred_ecef, ep)
        if nv < 4:
            if os.environ.get('DEBUG_EXC'):
                print("DEBUG(TC) nv<4", ep, nv, "amb_keys=", len(self.amb_keys))
            self.imu_idx = idx_snapshot
            self.epoch2 += 1
            self.nav.t = obs.t
            return

        # WP13h (TASK_M8.md): NHC + ZUPT/ZARU + Doppler motion-constraint
        # backbone, added to the SAME per-epoch graph as the DD factors
        # above, mirroring tc/'s own `optimize/stage.py` order (Doppler ->
        # NHC -> ZUPT, all after DD-factor construction, before the
        # smoother update). `imu_window` is exactly the IMU samples this
        # epoch's own `build_pim` call just integrated (`idx_snapshot` ->
        # `new_idx`), matching tc/'s `imu_idx_prev`/`n_imu` ZUPT window.
        self._add_doppler_vel_prior(
            graph, ep, obs, obs_, rs, vs, iu, sat, pos_pred_ecef, vel_pred)
        speed_for_nhc = float(np.linalg.norm(prev_vel[:2]))
        self._add_nhc_factor(
            graph, ep, speed_for_nhc, gyro_mean, prev_bias.gyroscope())
        imu_window = self.imu_data[idx_snapshot:new_idx]
        self._zupt_check_and_add(graph, ep, prev_ep, imu_window)

        nf_before = (self.smoother.getFactors().size() if self.smoother
                     else self.isam.getFactorsUnsafe().size())
        try:
            if self.smoother:
                ts = gtsam.FixedLagSmootherKeyTimestampMap()
                ts[key_x] = self.epoch_time
                ts[key_v] = self.epoch_time
                ts[key_b] = self.epoch_time
                for (s, f), k in _sorted_amb_items(new_amb):
                    ts[k] = self.epoch_time
                if self.per_epoch_n:
                    # WP13n piece 4: stamp last epoch's N keys one more
                    # epoch (tc/'s `_solve_isam2` keep_keys: `extra.
                    # append(k_old)` for sats with a current key) so the
                    # BetweenN chain's k_old survives this update; keys
                    # older than that are NOT re-stamped and age out of
                    # the lag window -- the clean per-epoch
                    # marginalization tc/'s N model relies on.
                    for (s, f) in sorted(new_amb):
                        prev = self._prev_amb_values.get((s, f))
                        if prev is not None:
                            ts[prev[0]] = self.epoch_time
                else:
                    for (s, f), k in _sorted_amb_items(self.amb_keys):
                        if (s, f) not in new_amb:
                            ts[k] = self.epoch_time
                self.smoother.update(graph, new_values, ts)
                estimate = self.smoother.calculateEstimate()
            else:
                self.isam.update(graph, new_values)
                self.isam.update()
                estimate = self.isam.calculateEstimate()
            self.current_estimate = estimate
        except (RuntimeError, IndexError) as _DEBUG_EXC:
            if os.environ.get('DEBUG_EXC'):
                import traceback as _tb
                print("DEBUG_EXC(TC)", type(_DEBUG_EXC).__name__, _DEBUG_EXC)
                _tb.print_exc()
            # WP13n piece 7: consecutive-failure warm reset (tc/'s
            # handle_solve_exception, reduced -- see __init__ docstring).
            self._solve_fail_streak += 1
            if (self.solve_reset_after > 0
                    and self._solve_fail_streak >= self.solve_reset_after):
                try:
                    self._seed_phase2_graph(
                        pose_pred, vel_pred, prev_bias, obs.t, new_idx)
                    # arm a cp_hold window like tc/'s warm_reset_phase2
                    if self.cp_hold_enable:
                        self._cp_hold_remaining = max(
                            self._cp_hold_remaining, self.cp_hold_epochs)
                    # WP13r lever 2: tc/'s warm_reset_phase2 arms the
                    # RECOVERY cp-hold too (recovery.py line ~79).
                    if self.recov_cp_hold_epochs > 0:
                        self._recov_cp_hold_remaining = max(
                            self._recov_cp_hold_remaining,
                            self.recov_cp_hold_epochs)
                        self._recov_cp_release_streak = 0
                        self.recov_cp_hold_trigger_count += 1
                    self.solve_reset_count += 1
                    self._solve_fail_streak = 0
                    self.nav.t = obs.t
                    return
                except (RuntimeError, IndexError):
                    pass
            self.imu_idx = idx_snapshot
            self.epoch2 += 1
            self.nav.t = obs.t
            return

        self._solve_fail_streak = 0
        self.last_valid_epoch = ep
        self.last_valid_time = obs.t
        if self.per_epoch_n:
            # WP13n piece 4: this epoch's map REPLACES the key dict --
            # every CP-visible (s,f) got a fresh key this epoch (tc/'s
            # `_carry_prev_amb` clears every `amb_key` each epoch).
            self.amb_keys = dict(new_amb)
            for _key in new_amb:
                self._amb_last_seen[_key] = ep
        else:
            self.amb_keys.update(new_amb)
            # WP13n piece 2: stamp the birth epoch of each newly-committed
            # ambiguity (tc/'s `amb_init_epoch`, set at seed time; recorded
            # here at COMMIT time instead so an abandoned graph never leaves
            # a ghost stamp -- same reason `amb_keys` itself merges here).
            # (per-epoch mode stamps FRESH seeds at seed time instead --
            # a continuing key keeps its original age for ar_wait_new.)
            for _key in new_amb:
                self._amb_init_epoch[_key] = ep
                self._amb_last_seen[_key] = ep

        # WP13g (TASK_M7.md): compute THIS epoch's own pre-FDE main DDPR
        # residual RMS + worst-satellite residual UNCONDITIONALLY (matches
        # postfit.py's `_compute_postfit_diagnostics`, which computes
        # `main_res_pre_fde` regardless of `cfg.fde_enable` -- our own
        # `_log_main_ddpr` is otherwise only ever invoked from inside
        # `_apply_fde`, which never runs when FDE is off, the winning
        # config per WP13d/f). Kept as `last_main_ddpr_rms`/
        # `last_main_ddpr_worst` -- the LITERAL, PR-only reference
        # quantity, used only for the periodic diagnostic print below,
        # unchanged semantics from WP13d.
        factors_all = (self.smoother.getFactors() if self.smoother
                       else self.isam.getFactorsUnsafe())
        self._globalize_epoch_ddpr_tags(nf_before, graph.size())
        self._log_main_ddpr(factors_all, nf_before, nf_before + graph.size(), estimate)

        # WP13g sanity-gating signal: `_main_ddpr_res`/`_main_ddpr_worst_res`
        # (disclosed adaptation -- see `_compute_main_dd_res`'s docstring
        # for why the LITERAL PR-only quantity above is near-always ~0 on
        # this port's factor graph and can't gate anything).
        self._main_ddpr_res, self._main_ddpr_worst_res = self._compute_main_dd_res(
            nf_before, graph.size(), estimate, key_x, new_amb)
        self._update_cp_hold()

        # WP13s literal: tc/'s full post-fit quality bookkeeping
        # (persist-bad + reference/observation/pair EWMAs + post-DDPR
        # counters) via the imported SatQualityState -- replaces the
        # reduced WP13r block below.
        if self.tc_literal:
            self._tc_literal_postfit_quality()

        # WP13r: per-sat quarantine update -- tc/'s optimize/stage.py
        # lines ~228-254 (post-solve, PR-only per-sat residual signal):
        # `streak` consecutive epochs above `res_thresh` quarantine the
        # sat for `hold` epochs and regenerate its FLOAT N (amb_gen
        # bump; a COMMITTED integer survives -- tc/ does not clear_hold
        # here -- it just sits out the candidate set until expiry).
        if self.ar_persist_bad_enable and not self.tc_literal:
            _per_sat_pb = self._main_ddpr_per_sat_pr or {}
            _seen_pb = set()
            for _s_pb, _rmax_pb in _per_sat_pb.items():
                _s_pb = int(_s_pb)
                _seen_pb.add(_s_pb)
                if float(_rmax_pb) > self.ar_persist_bad_res_thresh:
                    _st_pb = self._persist_bad_streak.get(_s_pb, 0) + 1
                    self._persist_bad_streak[_s_pb] = _st_pb
                    if _st_pb >= self.ar_persist_bad_streak:
                        if self._persist_bad_hold.get(_s_pb, 0) <= 0:
                            self.persist_bad_hold_count += 1
                        self._persist_bad_hold[_s_pb] = max(
                            int(self._persist_bad_hold.get(_s_pb, 0)),
                            self.ar_persist_bad_hold)
                        for _f_pb in range(self.nav.nf):
                            _key_pb = (_s_pb, _f_pb)
                            if (_key_pb in self.amb_keys
                                    and _key_pb not in self._committed):
                                self._release_ambiguity(
                                    _s_pb, _f_pb, reason='persist_bad')
                                self.persist_bad_release_count += 1
                else:
                    self._persist_bad_streak[_s_pb] = 0
            for _s_pb in list(self._persist_bad_streak.keys()):
                if _s_pb not in _seen_pb:
                    self._persist_bad_streak[_s_pb] = 0

        if self.fde_enable:
            estimate = self._apply_fde(graph, key_x, nv, nf_before, estimate)
            self.current_estimate = estimate

        if self.per_epoch_n:
            # WP13n piece 4: refresh tc/'s `prev_amb_tc` analogue from
            # the (post-FDE) estimate -- next epoch's continuity priors
            # + BetweenN chain read (key, value) from here. An (s,f)
            # whose key was released/rejected simply drops out and
            # re-seeds fresh next epoch, exactly like tc/.
            self._prev_amb_values = {
                _key: (_k, float(estimate.atDouble(_k)))
                for _key, _k in self.amb_keys.items()
                if estimate.exists(_k)}

        self._write_back_tc(estimate, key_x, cp_amb)
        self.nav.smode = 5

        if self.nav.armode > 0:
            self._do_ar(obs, rs, vs, dts, sat, el, iu)

        _sanity_fired = False
        if self.sanity_enable:
            pose_tc_now = estimate.atPose3(key_x)
            fired = self._ddpr_sanity_check(
                nf_before, graph.size(), pose_tc_now, nav_pred, key_x,
                self._last_nb)
            _sanity_fired = bool(fired)
            if fired and os.environ.get('WP13G_DBG_SANITY'):
                _, tow_dbg2 = time2gpst(obs.t)
                print(f"DBG_SANITY_FIRE ep={ep} tow={tow_dbg2:.1f} "
                      f"main_res={self._main_ddpr_res:.2f} "
                      f"pos_ecef={self.nav.x[0:3]}")

        # WP13s literal: tc/'s postprocess Stage D-4 (fix-streak update;
        # on FLT, release the single most suspicious held). tc/ skips
        # this entirely on a sanity-fired epoch (postprocess.run returns
        # the sanity result before `_update_streaks_and_post_hooks`).
        if self.tc_literal and not _sanity_fired:
            self._tc_literal_postprocess(
                getattr(self, '_last_candidates', []))

        if os.environ.get('WP13F_DBG_TC'):
            _, tow_dbg = time2gpst(obs.t)
            bias = estimate.atConstantBias(key_b)
            ba = np.array(bias.accelerometer())
            bg = np.array(bias.gyroscope())
            vel_dbg = np.array(estimate.atVector(key_v))
            print(f"DBG_TC ep={ep} tow={tow_dbg:.1f} smode={self.nav.smode} "
                  f"pos_ecef={self.nav.x[0:3]} vel_enu={vel_dbg} "
                  f"|vel|={np.linalg.norm(vel_dbg):.3f} "
                  f"bias_acc={ba} |bias_acc|={np.linalg.norm(ba):.4f} "
                  f"bias_gyro_deg_s={np.degrees(bg)} nv={nv} "
                  f"fde_reject={self.last_fde_reject} "
                  f"main_ddpr_res={self._main_ddpr_res:.2f} "
                  f"sanity_bad={self._ddpr_bad_count}")

        self.nav.t = obs.t
        self.epoch2 += 1
        self.epoch_time += 1.0
