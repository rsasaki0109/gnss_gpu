#!/usr/bin/env python3
"""WP13a: run the standalone RTK-on-FGO (GtsamRtk) over one Tokyo PPC run.

Usage:
  .venv/Scripts/python.exe wp13a_run_standalone_rtk.py <run> [--lag L] [--huber-pr H]

  <run> is 1, 2, or 3 (Tokyo PPC dataset run{N}).

Writes:
  results/wp13a/run{N}.pos   RTKLIB-column-order .pos (week tow x y z lat lon
                             height Q ns sdx sdy sdz age ratio), Q=4 fix /
                             Q=5 float (matches nav.smode convention, and the
                             gnss_gpu scorer's default fix_statuses={4}).
  results/wp13a/run{N}.npz   per-epoch tow, ecef (sol_xyz), smode, err3d
                             (nearest-reference-row 3D error, diagnostic only
                             -- official scoring uses score_vs_inuex35.py's
                             own tow-tolerant reference lookup).

Environment variables (read by GtsamRtk.__init__, see gtsam_rtk_standalone.py):
  AR_MODE (default 3 = fix-and-hold), SIG_PR/SIG_CP/SIG_DYN/SIG_AMB, LAG,
  HUBER_PR. MAX_EP caps the number of synced epochs processed (debug only;
  default = process to end of rover.obs).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import gtsam

_HERE = Path(__file__).resolve().parent
_TC_SRC = _HERE / "tc" / "src"
if str(_TC_SRC) not in sys.path:
    # Only used for the generic RINEX-signal-autodetect helper (boilerplate
    # header-driven signal picking, not a TC/IMU/NLOS feature) so the data
    # loading mirrors tc/examples/run_imu_gnss_tc.py exactly.
    sys.path.insert(0, str(_TC_SRC))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cssrlib.rinex as rn
import cssrlib.gnss as gn
from cssrlib.gnss import uTYP, ecef2pos

from gnss_fgo.utils.sig_autodetect import auto_detect_signals  # noqa: E402
from gnss_fgo.utils.geometry import load_imu_csv  # noqa: E402

from gtsam_rtk_standalone import GtsamRtk, GtsamRtkTc  # noqa: E402

_DATA_ROOT = Path(os.environ.get(
    "PPC_DATA_ROOT",
    r"C:\Users\rsasa\Workspace\old\gnss_gpu\datasets\PPC-Dataset-data\tokyo",
))


def load_reference(reffile: Path):
    tows = []
    ecefs = []
    with open(reffile) as f:
        for row in csv.DictReader(f):
            tows.append(float(row["GPS TOW (s)"]))
            ecefs.append([
                float(row["ECEF X (m)"]),
                float(row["ECEF Y (m)"]),
                float(row["ECEF Z (m)"]),
            ])
    return np.asarray(tows, dtype=np.float64), np.asarray(ecefs, dtype=np.float64)


def write_pos_file(path: Path, rows, header_note: str) -> None:
    """RTKLIB xyz-ecef column order, Q at index 8 (0-based) -- see
    experiments/score_vs_inuex35.py:_pos_quality_flag / wp4_run_local_fgo_full
    .write_pos_file for why this exact column layout is required for the
    campaign scorer to parse Q back out correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write(f"% {header_note}\n")
        fh.write(
            "%  GPST_week   tow(s)      x-ecef(m)        y-ecef(m)        z-ecef(m)"
            "   lat(deg)   lon(deg)  height(m)   Q  ns   sdx    sdy    sdz   age  ratio\n"
        )
        for week, tow, xyz, q, ns in rows:
            lat, lon, hgt = ecef2pos(xyz)
            fh.write(
                f"{int(week):4d} {float(tow):14.4f} "
                f"{xyz[0]:16.4f} {xyz[1]:16.4f} {xyz[2]:16.4f}  "
                f"{np.degrees(lat):10.6f} {np.degrees(lon):11.6f} {hgt:9.3f} "
                f"{int(q)}   {int(ns):2d}  0.000  0.000  0.000  0.00  0.0\n"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=int, choices=(1, 2, 3))
    ap.add_argument("--lag", type=float, default=0.0, help="IncrementalFixedLagSmoother lag [s], 0=ISAM2")
    ap.add_argument("--huber-pr", type=float, default=0.0)
    ap.add_argument("--huber-cp", type=float, default=0.0,
                     help="WP13f (TASK_M6.md): Huber M-estimator threshold "
                          "for GtsamRtkTc's Phase-2 Arm DD-CP factor (0=off). "
                          "See gtsam_rtk_standalone.py's HUBER_CP docstring.")
    ap.add_argument("--ar-mode", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=_HERE / "results" / "wp13a")
    ap.add_argument("--max-ep", type=int, default=0, help="0 = process to end of rover.obs")
    ap.add_argument("--maxage", type=float, default=30.0, help="max age [s] of base obs for sync")
    ap.add_argument("--nf", type=int, default=3, help="number of frequency bands (matches tc/examples)")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=2000,
                     help="write partial .pos/.npz every N synced epochs "
                          "(0=disable), so an external kill loses at most "
                          "one checkpoint interval of progress instead of "
                          "the whole run (output is only otherwise written "
                          "at the very end)")
    ap.add_argument("--resume", action="store_true",
                     help="WP13c: if <out-dir>/run{N}.npz already exists, "
                          "load its checkpointed rows, fast-forward the "
                          "sync generator past them (no rtk.process() "
                          "calls, cheap I/O only), and continue appending "
                          "from there instead of starting over -- lets a "
                          "full run survive this environment's ~55-80min "
                          "background-process kill across several restarts. "
                          "The GTSAM graph itself restarts fresh at the "
                          "resume point (a brief re-convergence, like any "
                          "RTK engine restart) -- see WP13C_REPORT.md.")
    # WP13c (TASK_M3.md): AR acceptance gate + hold-release knobs, read by
    # GtsamRtk.__init__ from the matching env vars (see gtsam_rtk_standalone.py).
    ap.add_argument("--hold-ratio", type=float, default=0.0)
    ap.add_argument("--hold-confirm-n", type=int, default=1)
    ap.add_argument("--hold-valpos-thres", type=float, default=0.0)
    ap.add_argument("--hold-streak-tol", type=float, default=0.05)
    ap.add_argument("--min-committed-for-fix", type=int, default=0)
    ap.add_argument("--release-k", type=int, default=0)
    ap.add_argument("--release-thres", type=float, default=1.0)
    ap.add_argument("--parmode", type=int, default=0,
                     help="0 = leave nav.parmode at its cssrlib default (2, PAR)")
    ap.add_argument("--valpos-thres", type=float, default=0.0,
                     help="WP13a's whole-epoch VALPOS_THRES gate (disclosed "
                          "negative at 3-4; default off, kept for reference)")
    ap.add_argument("--sig-dyn", type=float, default=5.0,
                     help="TASK_M3.md: do not re-sweep dynamics -- fixed at "
                          "WP13a's own established winning value (5.0) unless "
                          "SIG_DYN is already set in the environment")
    # WP13d (TASK_M4.md): GICI-style FDE knobs, read by GtsamRtk.__init__
    # from the matching env vars (see gtsam_rtk_standalone.py). Defaults
    # mirror inuex35's own TcConfig defaults.
    ap.add_argument("--fde-enable", type=int, default=1)
    ap.add_argument("--fde-pr", type=float, default=4.0)
    ap.add_argument("--fde-cp", type=float, default=0.5)
    ap.add_argument("--fde-max-frac", type=float, default=0.5)
    ap.add_argument("--fde-max-iter", type=int, default=1)
    ap.add_argument("--fde-median-sub", type=int, default=0)
    ap.add_argument("--fde-remove-cp", type=int, default=1,
                     help="1=literal reference behavior (structurally "
                          "remove rejected DD-CP factors via "
                          "removeFactorIndices); 0=disclosed adaptation, "
                          "only reset the ambiguity (cycle slip), never "
                          "structurally remove a CP factor -- see "
                          "WP13D_REPORT.md for why 1 destabilizes this "
                          "GNSS-only (no IMU) smoother at --lag 1.0")
    # WP13e (TASK_M5.md): IMU tight coupling (GtsamRtkTc) instead of the
    # GNSS-only Point3 float (GtsamRtk). Loads imu.csv from the same run
    # directory. See gtsam_rtk_standalone.py's GtsamRtkTc docstring.
    ap.add_argument("--imu-tc", action="store_true",
                     help="use GtsamRtkTc (Pose3+Vel+Bias, CombinedImuFactor "
                          "+ Arm DD factors) instead of GtsamRtk (Point3)")
    ap.add_argument("--phase1-epochs", type=int, default=100,
                     help="GtsamRtkTc: committed GNSS-only Phase-1 epochs "
                          "before transitioning to IMU tight coupling "
                          "(WP13e milestone sweep: 40->23.3m, 100->21.5m, "
                          "150->153m AllRMS on run2's first 3000ep -- "
                          "non-monotonic, sensitive to this stop-and-go "
                          "dataset's own local conditions; 100 is the best "
                          "found, not swept exhaustively)")
    ap.add_argument("--lever-arm", type=str, default="0.31,0,0.55",
                     help="GtsamRtkTc: body-FLU lever arm x,y,z [m] "
                          "(default matches tc/examples/run_imu_gnss_tc.py)")
    # WP13g (TASK_M7.md): DDPR-sanity / pose-innovation check knobs, read
    # by GtsamRtkTc.__init__ from the matching env vars (see
    # gtsam_rtk_standalone.py's "WP13g" __init__ block docstring). Defaults
    # are the reference's own (config.py / the tokyo_mode2_satbad_cponly
    # preset -- see that block's docstring for which preset value each one
    # comes from).
    ap.add_argument("--sanity-enable", type=int, default=1,
                     help="1=on (default), 0=off (falls back to WP13c's "
                          "RELEASE_K-only recovery, which has failed 4x on "
                          "this dataset's worst excursions per WP13F_REPORT.md)")
    ap.add_argument("--sanity-pose-replace-thresh", type=float, default=5.0)
    ap.add_argument("--main-ddpr-res-catastrophic", type=float, default=15.0)
    ap.add_argument("--ddpr-sanity-persist", type=int, default=3)
    ap.add_argument("--main-ddpr-res-thresh", type=float, default=3.0)
    ap.add_argument("--ddpr-fast-worst-sat-min", type=float, default=10.0)
    ap.add_argument("--ddpr-fast-min-persist", type=int, default=2,
                     help="WP13g disclosed adaptation (not in postfit.py): "
                          "min consecutive bad epochs before the "
                          "catastrophic fast path may act (1=literal "
                          "reference; see gtsam_rtk_standalone.py's "
                          "_ddpr_sanity_check docstring for why 2 is the "
                          "default here)")
    ap.add_argument("--sanity-do-reset", type=int, default=1,
                     help="1=literal reference (a sanity fire ALSO wipes "
                          "all tracked ambiguities); 0=report-layer only "
                          "(WP13g found the full wipe regresses the "
                          "already-solved canyon -- see WP13G_REPORT.md)")
    # WP13h (TASK_M8.md): NHC + ZUPT/ZARU + Doppler motion-constraint
    # backbone knobs, read by GtsamRtkTc.__init__ from the matching env
    # vars (see gtsam_rtk_standalone.py's "WP13h" __init__ block).
    # Defaults are tc/'s own TcConfig defaults / this task's own spec.
    ap.add_argument("--nhc-enable", type=int, default=1)
    ap.add_argument("--nhc-min-speed", type=float, default=0.0)
    ap.add_argument("--nhc-sigma-lat", type=float, default=0.3)
    ap.add_argument("--nhc-sigma-vert", type=float, default=0.2)
    ap.add_argument("--nhc-lever", type=str, default="0,0,0")
    ap.add_argument("--zupt-enable", type=int, default=1)
    ap.add_argument("--zupt-max-acc-std", type=float, default=0.55)
    ap.add_argument("--zupt-max-gyro-std", type=float, default=0.030)
    ap.add_argument("--zupt-max-gyro-median", type=float, default=0.020)
    ap.add_argument("--zupt-min-samples", type=int, default=5)
    ap.add_argument("--zupt-sigma-zero-velocity", type=float, default=0.5)
    ap.add_argument("--zupt-sigma-zero-rotation", type=float, default=0.010)
    ap.add_argument("--doppler-vel-sigma", type=float, default=0.5,
                     help="0=off")
    ap.add_argument("--doppler-max-res", type=float, default=2.0)
    # WP13i (TASK_M9.md): post-fit DD-PR cross-check in the AR accept gate
    # (tc/'s `ar_ddpr_xvalidate_thresh`/`_delta_thresh`), read by
    # GtsamRtkTc._do_ar via GtsamRtk.__init__ from the matching env vars.
    ap.add_argument("--ar-ddpr-xvalidate-thresh", type=float, default=10.0,
                     help="reject (report float, keep held ambiguities) a "
                          "candidate LAMBDA fix whose OWN DD-PR residual "
                          "(evaluated at the fix's LAMBDA-refined position, "
                          "ambiguity-independent) exceeds this many metres; "
                          "0=off. Default matches tc/'s config.py.")
    ap.add_argument("--ar-ddpr-xvalidate-delta-thresh", type=float, default=0.0,
                     help="also reject if the DD-PR residual at the fix's "
                          "position worsens by more than this many metres "
                          "vs. this epoch's pre-AR float DD-PR residual; "
                          "0=off (default, matches tc/'s config.py)")
    # WP13j (TASK_M10.md): PLATEAU ray-traced NLOS prior, read by
    # GtsamRtk.__init__/_do_ar/_build_dd_factors_arm from the matching env
    # vars (see gtsam_rtk_standalone.py's WP13j module comment). NLOS_RUN
    # is always derived from the `run` positional (below), not a separate
    # flag -- the mask file is 1:1 with the dataset run.
    ap.add_argument("--nlos-enable", type=int, default=0,
                     help="1=load the PLATEAU ray-traced NLOS mask for "
                          "this run (0 cost when 0; all 3 layers below "
                          "become no-ops)")
    ap.add_argument("--nlos-ar-exclude", type=int, default=0,
                     help="layer (a): exclude NLOS-flagged sats from the "
                          "LAMBDA AR candidate set this epoch")
    ap.add_argument("--nlos-commit-reject", type=int, default=0,
                     help="layer (c): refuse to hold (or report Q=4 for) a "
                          "candidate fix that depends on NLOS sats")
    ap.add_argument("--nlos-meas-mode", type=int, default=0,
                     help="layer (b): 0=off, 1=hard-exclude a DD-PR/DD-CP "
                          "factor touching an NLOS sat, 2=inflate its "
                          "sigma by --nlos-meas-sigma-mult")
    ap.add_argument("--nlos-meas-sigma-mult", type=float, default=5.0)
    ap.add_argument("--nlos-report-frac", type=float, default=0.5,
                     help="layer (c) whole-epoch report reject: fraction "
                          "of active candidates that must be NLOS-flagged "
                          "(0.0=ANY NLOS-flagged active candidate voids "
                          "the report; default 0.5=strict majority)")
    # WP13k (TASK_M11.md): partial/subset AR (tc/'s actual
    # `_run_lambda_attempts`/`_try_subset_ar` chain, see
    # gtsam_rtk_standalone.py's WP13k __init__ block docstring for why
    # this -- not cssrlib's own exclmax=1-capped parmode=2 PAR -- is the
    # real fix-rate lever).
    ap.add_argument("--ar-rtklib-mode", type=int, default=0,
                     help="0=default: plain resamb_lambda(sat, nav.parmode, "
                          "nav.par_P0) (this port's pre-WP13k primary AR "
                          "call, PAR/parmode=2); 1=RTKLIB-faithful full-ILS "
                          "+ 1-sat round-robin exclusion (resamb_lambda_"
                          "rtklib, tc/'s own literal rtklib_mode=1 default). "
                          "WP13k's '>40x slower' measurement was FALSIFIED "
                          "by WP13m: the hang was one cold-start NaN "
                          "covariance spinning mlambda._estimILS past its "
                          "LOOPMAX (now guarded in gtsam_rtk_standalone.py, "
                          "no cssrlib edit); guarded full-ILS runs at "
                          "~24-30 ep/s, fully tractable -- see "
                          "WP13M_REPORT.md. Default stays 0 only so WP13l's "
                          "shipped config reproduces without flags.")
    ap.add_argument("--ar-arfilter", type=int, default=1)
    ap.add_argument("--ar-minfixsats", type=int, default=4)
    ap.add_argument("--par-p0", type=float, default=0.995)
    ap.add_argument("--subset-ar-enable", type=int, default=1)
    ap.add_argument("--subset-ar-max-drop", type=int, default=2)
    ap.add_argument("--subset-ar-min-nb", type=int, default=4)
    ap.add_argument("--subset-ar-max-candidates", type=int, default=5)
    ap.add_argument("--subset-ar-hold-ratio", type=float, default=1.0,
                     help="separate (lower) ratio-report bar for subset-AR"
                          "-sourced epochs -- PAR-mode's own s1/s0 is not "
                          "the same metric as a full-ILS ratio and is "
                          "structurally << --hold-ratio; see "
                          "gtsam_rtk_standalone.py's WP13k notes")
    # WP13k priority 2: sat_badness EWMA (tc/'s `preprocess/sat_quality.py`,
    # reduced scope -- see gtsam_rtk_standalone.py's __init__ docstring).
    ap.add_argument("--sat-badness-enable", type=int, default=0)
    ap.add_argument("--sat-badness-ddpr-thresh", type=float, default=1.0)
    ap.add_argument("--sat-badness-sigma-scale-cp", type=float, default=1.5)
    ap.add_argument("--sat-badness-ewma-alpha", type=float, default=0.3)
    ap.add_argument("--sat-badness-decay", type=float, default=0.9)
    # WP13k priority 3: cp_hold FSM (reduced scope -- see
    # gtsam_rtk_standalone.py's __init__ docstring).
    ap.add_argument("--cp-hold-enable", type=int, default=0)
    ap.add_argument("--cp-hold-trigger-thresh", type=float, default=3.0)
    ap.add_argument("--cp-hold-epochs", type=int, default=5)
    # WP13l: tc/'s downstream purity stack replacing the ratio gate for
    # PAR-sourced fixes (see gtsam_rtk_standalone.py __init__'s WP13l
    # docstring). Defaults are tc/'s own config.py defaults.
    ap.add_argument("--lambda-corr-max", type=float, default=0.0)
    ap.add_argument("--lambda-corr-hard-max", type=float, default=1.0)
    ap.add_argument("--low-nb-fix-reject-nb-max", type=int, default=6)
    ap.add_argument("--weak-fix-nb-max", type=int, default=2)
    ap.add_argument("--weak-fix-lambda-corr-max", type=float, default=0.08)
    ap.add_argument("--weak-fix-main-ddpr-res-max", type=float, default=0.8)
    ap.add_argument("--ar-context-main-ddpr-max", type=float, default=1.2)
    ap.add_argument("--ar-context-worst-sat-max", type=float, default=4.0)
    ap.add_argument("--ar-context-nb-max", type=int, default=6)
    ap.add_argument("--ar-context-reject-during-cp-hold", type=int, default=1)
    ap.add_argument("--ar-context-reject-during-ddpr-bad", type=int, default=1)
    ap.add_argument("--par-ratio-min", type=float, default=0.0,
                     help="disclosed non-tc/ addition, default 0=off; see gtsam_rtk_standalone.py _do_ar")
    # WP13m (TASK_M13.md): this port never overrode cssrlib's Nav-class
    # elmin/cnr_min defaults (15deg/25dBHz) the way tc/'s runner.py does
    # (cfg.elmin_deg=25/cnr_min_dbhz=30 under the tokyo_mode2_satbad_cponly
    # preset) -- see gtsam_rtk_standalone.py __init__'s WP13m docstring.
    ap.add_argument("--elmin-deg", type=float, default=15.0,
                     help="AR/obs elevation mask [deg]; cssrlib default 15, tc/ preset 25")
    ap.add_argument("--cnr-min-dbhz", type=float, default=25.0,
                     help="CNR floor [dBHz]; cssrlib default 25, tc/ preset 30")
    ap.add_argument("--lambda-diag-csv", type=str, default="",
                     help="WP13m: dump per-mlambda-call n/cond(Qb)/dt to this CSV, no behavior change")
    args = ap.parse_args()

    if args.lambda_diag_csv:
        import wp13m_lambda_diag as _diag
        _diag.install()

    os.environ["LAG"] = str(args.lag)
    os.environ["HUBER_PR"] = str(args.huber_pr)
    os.environ["HUBER_CP"] = str(args.huber_cp)
    os.environ["AR_MODE"] = str(args.ar_mode)
    os.environ.setdefault("SIG_DYN", str(args.sig_dyn))
    os.environ["HOLD_RATIO"] = str(args.hold_ratio)
    os.environ["HOLD_CONFIRM_N"] = str(args.hold_confirm_n)
    os.environ["HOLD_VALPOS_THRES"] = str(args.hold_valpos_thres)
    os.environ["HOLD_STREAK_TOL"] = str(args.hold_streak_tol)
    os.environ["MIN_COMMITTED_FOR_FIX"] = str(args.min_committed_for_fix)
    os.environ["RELEASE_K"] = str(args.release_k)
    os.environ["RELEASE_THRES"] = str(args.release_thres)
    if args.parmode:
        os.environ["PARMODE"] = str(args.parmode)
    os.environ["VALPOS_THRES"] = str(args.valpos_thres)
    os.environ["FDE_ENABLE"] = str(args.fde_enable)
    os.environ["FDE_PR"] = str(args.fde_pr)
    os.environ["FDE_CP"] = str(args.fde_cp)
    os.environ["FDE_MAX_FRAC"] = str(args.fde_max_frac)
    os.environ["FDE_MAX_ITER"] = str(args.fde_max_iter)
    os.environ["FDE_MEDIAN_SUB"] = str(args.fde_median_sub)
    os.environ["FDE_REMOVE_CP"] = str(args.fde_remove_cp)
    os.environ["PHASE1_EPOCHS"] = str(args.phase1_epochs)
    os.environ["LEVER_ARM"] = str(args.lever_arm)
    os.environ["SANITY_ENABLE"] = str(args.sanity_enable)
    os.environ["SANITY_POSE_REPLACE_THRESH"] = str(args.sanity_pose_replace_thresh)
    os.environ["MAIN_DDPR_RES_CATASTROPHIC"] = str(args.main_ddpr_res_catastrophic)
    os.environ["DDPR_SANITY_PERSIST"] = str(args.ddpr_sanity_persist)
    os.environ["MAIN_DDPR_RES_THRESH"] = str(args.main_ddpr_res_thresh)
    os.environ["DDPR_FAST_WORST_SAT_MIN"] = str(args.ddpr_fast_worst_sat_min)
    os.environ["DDPR_FAST_MIN_PERSIST"] = str(args.ddpr_fast_min_persist)
    os.environ["SANITY_DO_RESET"] = str(args.sanity_do_reset)
    os.environ["NHC_ENABLE"] = str(args.nhc_enable)
    os.environ["NHC_MIN_SPEED"] = str(args.nhc_min_speed)
    os.environ["NHC_SIGMA_LAT"] = str(args.nhc_sigma_lat)
    os.environ["NHC_SIGMA_VERT"] = str(args.nhc_sigma_vert)
    os.environ["NHC_LEVER"] = str(args.nhc_lever)
    os.environ["ZUPT_ENABLE"] = str(args.zupt_enable)
    os.environ["ZUPT_MAX_ACC_STD"] = str(args.zupt_max_acc_std)
    os.environ["ZUPT_MAX_GYRO_STD"] = str(args.zupt_max_gyro_std)
    os.environ["ZUPT_MAX_GYRO_MEDIAN"] = str(args.zupt_max_gyro_median)
    os.environ["ZUPT_MIN_SAMPLES"] = str(args.zupt_min_samples)
    os.environ["ZUPT_SIGMA_ZERO_VELOCITY"] = str(args.zupt_sigma_zero_velocity)
    os.environ["ZUPT_SIGMA_ZERO_ROTATION"] = str(args.zupt_sigma_zero_rotation)
    os.environ["DOPPLER_VEL_SIGMA"] = str(args.doppler_vel_sigma)
    os.environ["DOPPLER_MAX_RES"] = str(args.doppler_max_res)
    os.environ["AR_DDPR_XVALIDATE_THRESH"] = str(args.ar_ddpr_xvalidate_thresh)
    os.environ["AR_DDPR_XVALIDATE_DELTA_THRESH"] = str(args.ar_ddpr_xvalidate_delta_thresh)
    os.environ["NLOS_ENABLE"] = str(args.nlos_enable)
    os.environ["NLOS_RUN"] = str(args.run)
    os.environ["NLOS_AR_EXCLUDE"] = str(args.nlos_ar_exclude)
    os.environ["NLOS_COMMIT_REJECT"] = str(args.nlos_commit_reject)
    os.environ["NLOS_MEAS_MODE"] = str(args.nlos_meas_mode)
    os.environ["NLOS_MEAS_SIGMA_MULT"] = str(args.nlos_meas_sigma_mult)
    os.environ["NLOS_REPORT_FRAC"] = str(args.nlos_report_frac)
    os.environ["AR_RTKLIB_MODE"] = str(args.ar_rtklib_mode)
    os.environ["AR_ARFILTER"] = str(args.ar_arfilter)
    os.environ["AR_MINFIXSATS"] = str(args.ar_minfixsats)
    os.environ["PAR_P0"] = str(args.par_p0)
    os.environ["SUBSET_AR_ENABLE"] = str(args.subset_ar_enable)
    os.environ["SUBSET_AR_MAX_DROP"] = str(args.subset_ar_max_drop)
    os.environ["SUBSET_AR_MIN_NB"] = str(args.subset_ar_min_nb)
    os.environ["SUBSET_AR_MAX_CANDIDATES"] = str(args.subset_ar_max_candidates)
    os.environ["SUBSET_AR_HOLD_RATIO"] = str(args.subset_ar_hold_ratio)
    os.environ["SAT_BADNESS_ENABLE"] = str(args.sat_badness_enable)
    os.environ["SAT_BADNESS_DDPR_THRESH"] = str(args.sat_badness_ddpr_thresh)
    os.environ["SAT_BADNESS_SIGMA_SCALE_CP"] = str(args.sat_badness_sigma_scale_cp)
    os.environ["SAT_BADNESS_EWMA_ALPHA"] = str(args.sat_badness_ewma_alpha)
    os.environ["SAT_BADNESS_DECAY"] = str(args.sat_badness_decay)
    os.environ["CP_HOLD_ENABLE"] = str(args.cp_hold_enable)
    os.environ["CP_HOLD_TRIGGER_THRESH"] = str(args.cp_hold_trigger_thresh)
    os.environ["CP_HOLD_EPOCHS"] = str(args.cp_hold_epochs)
    os.environ["LAMBDA_CORR_MAX"] = str(args.lambda_corr_max)
    os.environ["LAMBDA_CORR_HARD_MAX"] = str(args.lambda_corr_hard_max)
    os.environ["LOW_NB_FIX_REJECT_NB_MAX"] = str(args.low_nb_fix_reject_nb_max)
    os.environ["WEAK_FIX_NB_MAX"] = str(args.weak_fix_nb_max)
    os.environ["WEAK_FIX_LAMBDA_CORR_MAX"] = str(args.weak_fix_lambda_corr_max)
    os.environ["WEAK_FIX_MAIN_DDPR_RES_MAX"] = str(args.weak_fix_main_ddpr_res_max)
    os.environ["AR_CONTEXT_MAIN_DDPR_MAX"] = str(args.ar_context_main_ddpr_max)
    os.environ["AR_CONTEXT_WORST_SAT_MAX"] = str(args.ar_context_worst_sat_max)
    os.environ["AR_CONTEXT_NB_MAX"] = str(args.ar_context_nb_max)
    os.environ["AR_CONTEXT_REJECT_DURING_CP_HOLD"] = str(args.ar_context_reject_during_cp_hold)
    os.environ["AR_CONTEXT_REJECT_DURING_DDPR_BAD"] = str(args.ar_context_reject_during_ddpr_bad)
    os.environ["PAR_RATIO_MIN"] = str(args.par_ratio_min)
    os.environ["ELMIN_DEG"] = str(args.elmin_deg)
    os.environ["CNR_MIN_DBHZ"] = str(args.cnr_min_dbhz)

    run_dir = _DATA_ROOT / f"run{args.run}"
    obsfile = run_dir / "rover.obs"
    basefile = run_dir / "base.obs"
    navfile = run_dir / "base.nav"
    reffile = run_dir / "reference.csv"
    imufile = run_dir / "imu.csv"
    required = [obsfile, basefile, navfile, reffile]
    if args.imu_tc:
        required.append(imufile)
    for p in required:
        if not p.exists():
            print(f"ERROR: missing input file {p}")
            return 1

    dec = rn.rnxdec()
    decb = rn.rnxdec()
    dec.decode_obsh(str(obsfile))
    decb.decode_obsh(str(basefile))
    sigs, sigsb = auto_detect_signals(
        dec.sig_map, decb.sig_map, max_freq=args.nf,
        required=(uTYP.C, uTYP.L, uTYP.S),
    )
    # WP13h (TASK_M8.md): register rover Doppler (uTYP.D) on the same
    # picked bands, transcribed verbatim from tc/examples/
    # run_imu_gnss_tc.py (lines ~125-136) -- base has no Doppler
    # (detslp_dop/doppler_velocity_ls are rover-only), and without this
    # `dec.setSignals(sigs)` never includes the D observation codes, so
    # `obs.D` decodes with 0 columns and `_add_doppler_vel_prior` can
    # never fire (found via a WP13h capped run2 3000ep smoke test:
    # dop=0 the entire run even with --doppler-vel-sigma 0.5).
    rov_picks_by_sys = {}
    for s in sigs:
        rov_picks_by_sys.setdefault(s.sys, set()).add(int(s.sig) // 100)
    for sys_id, bands in rov_picks_by_sys.items():
        rov_typ_d = {int(s.sig) // 100: s
                     for s in dec.sig_map.get(sys_id, {}).values()
                     if s.typ == uTYP.D}
        for band in bands:
            if band in rov_typ_d:
                sigs.append(rov_typ_d[band])
    dec.setSignals(sigs)
    decb.setSignals(sigsb)

    nav = gn.Nav(nf=args.nf)
    dec.decode_nav(str(navfile), nav)

    base_ecef = np.array(list(decb.pos))
    if np.linalg.norm(base_ecef) == 0:
        print("ERROR: base position not found in RINEX header")
        return 1

    ref_tows, ref_ecefs = load_reference(reffile)
    print(f"Loaded {len(ref_tows)} reference epochs")

    nav.rb = base_ecef.tolist()
    nav.pmode = 1
    nav.ephopt = 0
    pos0 = np.array(dec.pos) if np.linalg.norm(dec.pos) > 0 else base_ecef.copy()

    nep = args.max_ep if args.max_ep > 0 else 10**9
    sync_gen = rn.sync_obs_hold(dec, decb, maxage=args.maxage)

    pos_rows = []
    diag = {"ne": [], "tow": [], "week": [], "ecef": [], "smode": [], "err3d": [], "nsat_dd": [],
            "fde_reject": [], "ar_nb": [], "ar_ratio": [], "ar_raw_nb": [], "ar_phase": [],
            # WP13p: held-pool + churn per-epoch diagnostics (report-only)
            "ar_ncand": [], "ar_ncommit": [], "n_committed": [], "resets_cum": []}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos_path = args.out_dir / f"run{args.run}.pos"
    npz_path = args.out_dir / f"run{args.run}.npz"

    # WP13g Part 2 (TASK_M7.md "also fix"): `rtk` is assigned later (after
    # the --resume fast-forward block below), but `_write_outputs` can be
    # called BEFORE that (the "--resume: sync generator exhausted" early
    # return) -- pre-bind so the closure below never sees an unbound name.
    rtk = None

    def _capture_phase2_checkpoint():
        """WP13g Part 2: snapshot enough Phase-2 NavState (pose/vel/bias +
        IMU sample cursor + the committed epoch's own week/tow) to let
        --resume seed a NEW GtsamRtkTc DIRECTLY in phase=2 (via
        `_restore_phase2_from_checkpoint`), bypassing the stationary-
        assumption Phase-1 bootstrap entirely -- WP13F_REPORT.md's
        disclosed bug: re-running that bootstrap mid-drive (on data that
        is actually moving) produced ~9500m-scale artifacts at the resume
        boundary. Returns None when there is nothing restorable yet
        (GNSS-only GtsamRtk, or GtsamRtkTc still in its Phase-1 bootstrap)."""
        if (rtk is None or not args.imu_tc
                or getattr(rtk, "phase", 1) != 2
                or rtk.current_estimate is None
                or rtk.last_valid_epoch is None):
            return None
        ep0 = rtk.last_valid_epoch
        try:
            pose = rtk.current_estimate.atPose3(rtk.PX(ep0))
            vel = np.array(rtk.current_estimate.atVector(rtk.PV(ep0)))
            bias = rtk.current_estimate.atConstantBias(rtk.PB(ep0))
        except RuntimeError:
            return None
        week_c, tow_c = gn.time2gpst(rtk.last_valid_time)
        return {
            "ckpt_phase2_valid": 1,
            "ckpt_pose_R": np.array(pose.rotation().matrix()),
            "ckpt_pose_t": np.array(pose.translation()),
            "ckpt_vel_enu": vel,
            "ckpt_bias_acc": np.array(bias.accelerometer()),
            "ckpt_bias_gyro": np.array(bias.gyroscope()),
            "ckpt_imu_idx": int(rtk.imu_idx),
            "ckpt_week": int(week_c),
            "ckpt_tow": float(tow_c),
        }

    def _write_outputs(note_suffix: str) -> None:
        write_pos_file(
            pos_path, pos_rows,
            f"WP13a/c/d standalone RTK-on-FGO (joint-marginal-cov AR + gated "
            f"hold + GICI FDE), tokyo run{args.run}, AR_MODE={args.ar_mode} "
            f"LAG={args.lag} HUBER_PR={args.huber_pr} HOLD_RATIO={args.hold_ratio} "
            f"HOLD_CONFIRM_N={args.hold_confirm_n} "
            f"HOLD_VALPOS_THRES={args.hold_valpos_thres} "
            f"RELEASE_K={args.release_k} FDE_ENABLE={args.fde_enable} "
            f"FDE_PR={args.fde_pr} FDE_CP={args.fde_cp} "
            f"HUBER_CP={args.huber_cp} SANITY_ENABLE={args.sanity_enable} "
            f"SANITY_POSE_REPLACE_THRESH={args.sanity_pose_replace_thresh} "
            f"MAIN_DDPR_RES_CATASTROPHIC={args.main_ddpr_res_catastrophic} "
            f"DDPR_SANITY_PERSIST={args.ddpr_sanity_persist} "
            f"MAIN_DDPR_RES_THRESH={args.main_ddpr_res_thresh} "
            f"NHC_ENABLE={args.nhc_enable} ZUPT_ENABLE={args.zupt_enable} "
            f"DOPPLER_VEL_SIGMA={args.doppler_vel_sigma} "
            f"AR_DDPR_XVALIDATE_THRESH={args.ar_ddpr_xvalidate_thresh} "
            f"AR_DDPR_XVALIDATE_DELTA_THRESH={args.ar_ddpr_xvalidate_delta_thresh}"
            f"{note_suffix}")
        ckpt = _capture_phase2_checkpoint()
        savez_kwargs = dict(
            ne=np.asarray(diag["ne"]),
            tow=np.asarray(diag["tow"]),
            week=np.asarray(diag["week"]),
            sol_xyz=np.asarray(diag["ecef"]) if diag["ecef"] else np.zeros((0, 3)),
            smode=np.asarray(diag["smode"]),
            err3d=np.asarray(diag["err3d"]),
            nsat_dd=np.asarray(diag["nsat_dd"]),
            fde_reject=np.asarray(diag["fde_reject"]),
            ar_nb=np.asarray(diag["ar_nb"]),
            ar_ratio=np.asarray(diag["ar_ratio"]),
            ar_raw_nb=np.asarray(diag["ar_raw_nb"]),
            ar_phase=np.asarray(diag["ar_phase"]),
            ar_ncand=np.asarray(diag["ar_ncand"]),
            ar_ncommit=np.asarray(diag["ar_ncommit"]),
            n_committed=np.asarray(diag["n_committed"]),
            resets_cum=np.asarray(diag["resets_cum"]),
            base_xyz=base_ecef,
            phase=int(getattr(rtk, "phase", 1)) if rtk is not None else 1,
        )
        if ckpt is not None:
            savez_kwargs.update(ckpt)
        else:
            savez_kwargs["ckpt_phase2_valid"] = 0
        np.savez(npz_path, **savez_kwargs)

    # WP13c --resume: fast-forward past already-checkpointed synced epochs
    # (cheap I/O only, no rtk.process()) and continue appending. The GTSAM
    # graph itself always starts fresh at the resume point -- see the
    # --resume help text / WP13C_REPORT.md for why a true graph-level
    # resume (serializing ISAM2/IFLS state) was out of scope here.
    start_ne = 0
    resume_phase2_state = None
    if args.resume and npz_path.exists():
        old = np.load(npz_path)
        old_ne = np.asarray(old["ne"]).reshape(-1)
        # WP13g Part 2: pick up a persisted Phase-2 NavState (if this
        # checkpoint has one -- pre-WP13g checkpoints and Phase-1-only
        # checkpoints won't) so the NEW GtsamRtkTc below can be seeded
        # DIRECTLY in phase=2 via `_restore_phase2_from_checkpoint`,
        # instead of starting at phase=1 and re-running the stationary-
        # assumption bootstrap on data that's actually moving.
        old_phase = int(old["phase"]) if "phase" in old.files else 1
        if (args.imu_tc and old_phase == 2
                and "ckpt_phase2_valid" in old.files
                and int(old["ckpt_phase2_valid"]) == 1):
            resume_phase2_state = {
                "pose_R": np.asarray(old["ckpt_pose_R"]),
                "pose_t": np.asarray(old["ckpt_pose_t"]),
                "vel_enu": np.asarray(old["ckpt_vel_enu"]),
                "bias_acc": np.asarray(old["ckpt_bias_acc"]),
                "bias_gyro": np.asarray(old["ckpt_bias_gyro"]),
                "imu_idx": int(old["ckpt_imu_idx"]),
                "week": int(old["ckpt_week"]),
                "tow": float(old["ckpt_tow"]),
            }
        if old_ne.size > 0:
            for key, col in (("ne", "ne"), ("tow", "tow"), ("week", "week"),
                             ("smode", "smode"), ("err3d", "err3d"), ("nsat_dd", "nsat_dd")):
                diag[key] = list(np.asarray(old[col]).reshape(-1))
            # fde_reject is new in WP13d -- older (pre-WP13d) checkpoints
            # won't have it; backfill with zeros so lengths stay aligned.
            if "fde_reject" in old.files:
                diag["fde_reject"] = list(np.asarray(old["fde_reject"]).reshape(-1))
            else:
                diag["fde_reject"] = [0] * old_ne.size
            # WP13k: ar_nb/ar_ratio/ar_raw_nb/ar_phase are new (tc/-diff
            # comparison diagnostics) -- older checkpoints won't have them;
            # backfill with zeros so lengths stay aligned.
            for k2 in ("ar_nb", "ar_raw_nb", "ar_phase",
                       # WP13p: new columns -- older checkpoints backfill 0
                       "ar_ncand", "ar_ncommit", "n_committed", "resets_cum"):
                diag[k2] = (list(np.asarray(old[k2]).reshape(-1)) if k2 in old.files
                            else [0] * old_ne.size)
            diag["ar_ratio"] = (list(np.asarray(old["ar_ratio"]).reshape(-1))
                                 if "ar_ratio" in old.files else [0.0] * old_ne.size)
            old_ecef = np.asarray(old["sol_xyz"])
            diag["ecef"] = [row.copy() for row in old_ecef]
            for i in range(old_ne.size):
                q = 4 if int(diag["smode"][i]) == 4 else 5
                ns = int(diag["nsat_dd"][i])
                pos_rows.append((int(diag["week"][i]), float(diag["tow"][i]),
                                  diag["ecef"][i], q, ns))
            start_ne = int(old_ne.max()) + 1
            pos0 = old_ecef[-1].copy()
            print(f"--resume: loaded {old_ne.size} rows from {npz_path}, "
                  f"fast-forwarding to synced epoch {start_ne}, pos0={pos0}")
            for _ in range(start_ne):
                try:
                    next(sync_gen)
                except StopIteration:
                    print("--resume: sync generator exhausted during fast-forward "
                          "(checkpoint already covers the full run)")
                    _write_outputs(" [RESUMED, already complete]")
                    return 0
        else:
            print(f"--resume: {npz_path} has 0 rows, starting from scratch")

    if args.imu_tc:
        imu_data = load_imu_csv(str(imufile))
        print(f"Loaded {len(imu_data)} IMU samples from {imufile}")
        rtk = GtsamRtkTc(nav, imu_data, pos0)
    else:
        rtk = GtsamRtk(nav, pos0)

    # WP13g Part 2 (TASK_M7.md "also fix"): if the checkpoint we just
    # fast-forwarded past was already in Phase 2, seed this brand-new
    # GtsamRtkTc DIRECTLY in phase=2 from the persisted NavState --
    # bypassing `_transition_to_phase2`'s stationary-assumption bootstrap
    # (accel-tilt pitch/roll, bootstrap-displacement heading) entirely.
    # WP13F_REPORT.md's disclosed bug: without this, --resume always
    # re-ran that bootstrap on data that is actually moving, producing
    # ~9500m-scale artifacts right at the resume boundary.
    if resume_phase2_state is not None:
        st = resume_phase2_state
        pose0 = gtsam.Pose3(
            gtsam.Rot3(st["pose_R"]), gtsam.Point3(*st["pose_t"]))
        bias0 = gtsam.imuBias.ConstantBias(st["bias_acc"], st["bias_gyro"])
        rtk._restore_phase2_from_checkpoint(
            pose0, st["vel_enu"], bias0, st["week"], st["tow"], st["imu_idx"])
        print(f"--resume: restored Phase-2 NavState directly (phase={rtk.phase}, "
              f"no Phase-1 bootstrap re-run) -- pose_t={st['pose_t']} "
              f"vel_enu={st['vel_enu']} imu_idx={st['imu_idx']} "
              f"week={st['week']} tow={st['tow']:.1f}")
    print(f"Base: {base_ecef}")
    print(f"AR_MODE={args.ar_mode} LAG={args.lag} HUBER_PR={args.huber_pr} nf={args.nf} "
          f"HOLD_RATIO={args.hold_ratio} HOLD_CONFIRM_N={args.hold_confirm_n} "
          f"HOLD_VALPOS_THRES={args.hold_valpos_thres} RELEASE_K={args.release_k} "
          f"start_ne={start_ne} IMU_TC={args.imu_tc}"
          + (f" PHASE1_EPOCHS={args.phase1_epochs} LEVER_ARM={args.lever_arm} "
             f"HUBER_CP={args.huber_cp} SANITY_ENABLE={args.sanity_enable} "
             f"SANITY_POSE_REPLACE_THRESH={args.sanity_pose_replace_thresh} "
             f"MAIN_DDPR_RES_CATASTROPHIC={args.main_ddpr_res_catastrophic} "
             f"DDPR_SANITY_PERSIST={args.ddpr_sanity_persist} "
             f"MAIN_DDPR_RES_THRESH={args.main_ddpr_res_thresh} "
             f"DDPR_FAST_WORST_SAT_MIN={args.ddpr_fast_worst_sat_min} "
             f"NHC_ENABLE={args.nhc_enable} ZUPT_ENABLE={args.zupt_enable} "
             f"DOPPLER_VEL_SIGMA={args.doppler_vel_sigma} "
             f"AR_DDPR_XVALIDATE_THRESH={args.ar_ddpr_xvalidate_thresh} "
             f"AR_DDPR_XVALIDATE_DELTA_THRESH={args.ar_ddpr_xvalidate_delta_thresh}"
             if args.imu_tc else ""))

    t_start = time.time()
    n_fix = sum(1 for s in diag["smode"] if int(s) == 4)
    n_float = sum(1 for s in diag["smode"] if int(s) != 4)
    n_processed = len(pos_rows)
    ne = start_ne - 1
    first_epoch = True
    for ne in range(start_ne, nep):
        try:
            obs, obsb, dt_sync = next(sync_gen)
        except StopIteration:
            break
        if first_epoch:
            nav.t = obs.t
            first_epoch = False

        prev_valid = rtk.last_valid_epoch
        rtk.process(obs, obsb=obsb)
        updated = rtk.last_valid_epoch != prev_valid

        week, tow = gn.time2gpst(obs.t)

        if updated:
            ecef = nav.x[0:3].copy()
            smode = int(nav.smode)
            if smode == 4:
                n_fix += 1
                q = 4
            else:
                n_float += 1
                q = 5
            ns = int(np.sum(nav.vsat[:, 0]))
            pos_rows.append((week, tow, ecef, q, ns))
            ri = int(np.argmin(np.abs(ref_tows - tow))) if len(ref_tows) else -1
            err3d = (float(np.linalg.norm(ecef - ref_ecefs[ri]))
                     if ri >= 0 and abs(ref_tows[ri] - tow) < 0.5 else float("nan"))
            diag["ne"].append(ne)
            diag["tow"].append(tow)
            diag["week"].append(week)
            diag["ecef"].append(ecef)
            diag["smode"].append(smode)
            diag["err3d"].append(err3d)
            diag["nsat_dd"].append(ns)
            diag["fde_reject"].append(int(rtk.last_fde_reject))
            _s0 = getattr(rtk, '_last_s0', 0.0)
            _s1 = getattr(rtk, '_last_s1', 0.0)
            diag["ar_nb"].append(int(getattr(rtk, '_last_nb', 0)))
            diag["ar_ratio"].append(float(_s1 / _s0) if _s0 > 0 else 0.0)
            diag["ar_raw_nb"].append(int(getattr(rtk, '_last_resamb_raw_nb', -1)))
            diag["ar_phase"].append(int(getattr(rtk, 'phase', 1)))
            diag["ar_ncand"].append(int(getattr(rtk, '_last_ar_ncand', 0)))
            diag["ar_ncommit"].append(int(getattr(rtk, '_last_ar_ncommit', 0)))
            diag["n_committed"].append(len(getattr(rtk, '_committed', {})))
            diag["resets_cum"].append(
                int(getattr(rtk, 'slip_reset_count', 0))
                + int(getattr(rtk, 'mw_reset_count', 0))
                + int(getattr(rtk, 'cmc_reset_count', 0))
                + int(getattr(rtk, 'outage_reset_count', 0)))
            n_processed += 1

        if (ne + 1) % args.log_every == 0:
            elapsed = time.time() - t_start
            n_new = ne + 1 - start_ne
            phase_str = f" phase={rtk.phase}" if hasattr(rtk, "phase") else ""
            print(f"  ep {ne+1:6d}  tow={tow:10.1f}  fix={n_fix:5d} flt={n_float:5d}"
                  f"  committed={len(rtk._committed):4d} streak={len(rtk._streak):4d}"
                  f"  fde_rej_total={rtk.fde_total_rejected:6d} fde_ep={rtk.fde_epochs_rejected:5d}"
                  f" fde_safeguard={rtk.fde_epochs_safeguard_skipped:4d}"
                  f" main_ddpr_rms={rtk.last_main_ddpr_rms:6.2f}"
                  f" sanity_fire={getattr(rtk, 'sanity_fire_count', 0):4d}"
                  f"(fast={getattr(rtk, 'sanity_fast_count', 0)}"
                  f",replace={getattr(rtk, 'sanity_replace_count', 0)})"
                  f" nhc={getattr(rtk, '_nhc_fire_count', 0)}"
                  f" zupt={getattr(rtk, '_zupt_fire_count', 0)}"
                  f" dop={getattr(rtk, '_doppler_fire_count', 0)}"
                  f" xval_rej={getattr(rtk, '_ar_ddpr_xvalidate_reject_count', 0)}"
                  f" subset_ar={getattr(rtk, 'subset_ar_used_count', 0)}"
                  f"/{getattr(rtk, 'subset_ar_attempt_count', 0)}"
                  f" cp_hold={getattr(rtk, 'cp_hold_trigger_count', 0)}t"
                  f"/{getattr(rtk, 'cp_hold_active_epochs', 0)}ep"
                  f" nlos_hit={getattr(rtk, 'nlos_lookup_hit', 0)}"
                  f"/miss={getattr(rtk, 'nlos_lookup_miss', 0)}"
                  f" nlos_ar_exc={getattr(rtk, 'nlos_ar_excluded_epochs', 0)}"
                  f" nlos_commit_rej={getattr(rtk, 'nlos_commit_rejected', 0)}"
                  f" nlos_report_rej={getattr(rtk, 'nlos_report_rejected', 0)}"
                  f" par_gate_rej={getattr(rtk, 'par_gate_reject_count', 0)}"
                  f"{phase_str}"
                  f"  ({elapsed:.1f}s, {n_new/max(elapsed,1e-6):.1f} ep/s)")

        if args.checkpoint_every > 0 and (ne + 1) % args.checkpoint_every == 0:
            _write_outputs(f" [CHECKPOINT at ep {ne+1}, partial]")
            print(f"  checkpoint: wrote {len(pos_rows)} rows through ep {ne+1}")

    elapsed = time.time() - t_start
    print(f"Done: {ne+1} synced epochs, {n_processed} committed updates "
          f"({n_fix} fix, {n_float} float) in {elapsed:.1f}s")
    print(f"FDE: total_rejected={rtk.fde_total_rejected} epochs_rejected={rtk.fde_epochs_rejected} "
          f"epochs_safeguard_skipped={rtk.fde_epochs_safeguard_skipped} "
          f"mean_main_ddpr_rms={(rtk.main_ddpr_rms_sum/rtk.main_ddpr_rms_n if rtk.main_ddpr_rms_n else 0.0):.3f}")
    print(f"SANITY: fire_count={getattr(rtk, 'sanity_fire_count', 0)} "
          f"fast_count={getattr(rtk, 'sanity_fast_count', 0)} "
          f"replace_count={getattr(rtk, 'sanity_replace_count', 0)}")
    print(f"AR_DDPR_XVALIDATE: reject_count="
          f"{getattr(rtk, '_ar_ddpr_xvalidate_reject_count', 0)}")
    print(f"WP13L PAR_GATE: reject_count={getattr(rtk, 'par_gate_reject_count', 0)} "
          f"by_reason={getattr(rtk, 'par_gate_reject_by_reason', {})}")
    print(f"WP13N: slip_reset_enable={getattr(rtk, 'slip_reset_enable', 0)} "
          f"slip_resets={getattr(rtk, 'slip_reset_count', 0)} "
          f"mw_resets={getattr(rtk, 'mw_reset_count', 0)} "
          f"cmc_resets={getattr(rtk, 'cmc_reset_count', 0)} "
          f"outage_resets={getattr(rtk, 'outage_reset_count', 0)} "
          f"ar_wait_new={getattr(rtk, 'ar_wait_new', 0)} "
          f"per_sat_gate_drops={getattr(rtk, 'per_sat_gate_drop_count', 0)} "
          f"varerr_enable={getattr(rtk, 'varerr_enable', 0)} "
          f"per_epoch_n={getattr(rtk, 'per_epoch_n', 0)} "
          f"n_between={getattr(rtk, 'n_between_count', 0)} "
          f"n_cont_seed={getattr(rtk, 'n_cont_seed_count', 0)} "
          f"n_fresh_seed={getattr(rtk, 'n_fresh_seed_count', 0)} "
          f"solve_resets={getattr(rtk, 'solve_reset_count', 0)}")
    print(f"WP13O: thresar={getattr(getattr(rtk, 'nav', None), 'thresar', None)} "
          f"ar_context_pr_only={getattr(rtk, 'ar_context_pr_only', 0)} "
          f"per_sat_gate_pr_only={getattr(rtk, 'per_sat_gate_pr_only', 0)} "
          f"hold_ratio={getattr(rtk, 'hold_ratio', None)}")
    print(f"WP13P: ar_tier_mode={getattr(rtk, 'ar_tier_mode', 0)} "
          f"tier_elmin_deg={getattr(rtk, 'ar_tier_elmin_deg', 0)} "
          f"tier_cnr_min={getattr(rtk, 'ar_tier_cnr_min', 0)} "
          f"tier_held_exempt={getattr(rtk, 'ar_tier_held_exempt', 1)} "
          f"tier_vsat_drops={getattr(rtk, 'ar_tier_drop_count', 0)} "
          f"tier_cp_excl={getattr(rtk, 'ar_tier_cp_excl_count', 0)} "
          f"small_nb_max_p2={getattr(rtk, 'small_nb_max_p2', 0)} "
          f"small_nb_ratio_p2={getattr(rtk, 'small_nb_ratio_p2', 0)} "
          f"small_nb_rejects={getattr(rtk, 'small_nb_bar_reject_count', 0)} "
          f"slip_marginal_confirm_n={getattr(rtk, 'slip_marginal_confirm_n', 0)} "
          f"fde_release_ref={getattr(rtk, 'fde_release_ref', 0)}")
    print(f"WP13Q: release_seed_held={getattr(rtk, 'release_seed_held', 0)} "
          f"release_seed_sigma={getattr(rtk, 'release_seed_sigma', 0)} "
          f"seed_stored={getattr(rtk, 'release_seed_stored', 0)} "
          f"seed_used={getattr(rtk, 'release_seed_used', 0)} "
          f"seed_expired={getattr(rtk, 'release_seed_expired', 0)}")
    print(f"WP13R: cond_hold={getattr(rtk, 'cond_hold', 0)} "
          f"cond_epochs={getattr(rtk, 'cond_hold_epochs', 0)} "
          f"cond_key_epochs={getattr(rtk, 'cond_hold_key_epochs', 0)} "
          f"recov_cp_hold_epochs={getattr(rtk, 'recov_cp_hold_epochs', 0)} "
          f"recov_triggers={getattr(rtk, 'recov_cp_hold_trigger_count', 0)} "
          f"recov_cp_suppressed={getattr(rtk, 'recov_cp_hold_cp_suppressed_epochs', 0)} "
          f"sanity_skip_multipath={getattr(rtk, 'sanity_skip_multipath_count', 0)} "
          f"persist_bad_enable={getattr(rtk, 'ar_persist_bad_enable', 0)} "
          f"persist_bad_holds={getattr(rtk, 'persist_bad_hold_count', 0)} "
          f"persist_bad_releases={getattr(rtk, 'persist_bad_release_count', 0)}")
    print(f"WP13S: tc_literal={getattr(rtk, 'tc_literal', 0)} "
          f"lit_commits={getattr(rtk, 'lit_commit_count', 0)} "
          f"lit_ar_skip_hold={getattr(rtk, 'lit_ar_skip_hold_count', 0)} "
          f"cp_pr_rejects={getattr(rtk, 'cp_pr_reject_count', 0)} "
          f"rejc_wipes={getattr(rtk, 'rejc_wipe_count', 0)} "
          f"dirty_resets={getattr(rtk, 'dirty_reset_count', 0)} "
          f"flt_releases={getattr(rtk, 'flt_release_count', 0)} "
          f"pim_breaks={getattr(rtk, 'pim_break_count', 0)} "
          f"n_committed_end={len(getattr(rtk, '_committed', {}) or {})}")
    print(f"WP13K SUBSET_AR: rtklib_mode={args.ar_rtklib_mode} "
          f"subset_enable={args.subset_ar_enable} max_drop={args.subset_ar_max_drop} "
          f"used={getattr(rtk, 'subset_ar_used_count', 0)} "
          f"attempted={getattr(rtk, 'subset_ar_attempt_count', 0)} "
          f"sat_badness_enable={args.sat_badness_enable}")
    print(f"WP15: cuda_lambda={getattr(rtk, 'cuda_lambda', 0)} "
          f"gpu_active={int(getattr(rtk, '_cuda_mlambda_batch', None) is not None)} "
          f"batches={getattr(rtk, 'cuda_lambda_batch_count', 0)} "
          f"combos={getattr(rtk, 'cuda_lambda_combo_count', 0)} "
          f"cpu_fallbacks={getattr(rtk, 'cuda_lambda_fallback_count', 0)} "
          f"cascade_s={getattr(rtk, 'wp15_cascade_seconds', 0.0):.1f}")
    print(f"NLOS: enable={args.nlos_enable} ar_exclude={args.nlos_ar_exclude} "
          f"commit_reject={args.nlos_commit_reject} meas_mode={args.nlos_meas_mode} "
          f"mask_rows={getattr(rtk, '_nlos_mask_n_rows', 0)} "
          f"mask_nlos_rows={getattr(rtk, '_nlos_mask_n_nlos', 0)} "
          f"lookup_hit={getattr(rtk, 'nlos_lookup_hit', 0)} "
          f"lookup_miss={getattr(rtk, 'nlos_lookup_miss', 0)} "
          f"ar_excluded_epochs={getattr(rtk, 'nlos_ar_excluded_epochs', 0)} "
          f"ar_excluded_sat_epochs={getattr(rtk, 'nlos_ar_excluded_sat_epochs', 0)} "
          f"commit_rejected={getattr(rtk, 'nlos_commit_rejected', 0)} "
          f"report_rejected={getattr(rtk, 'nlos_report_rejected', 0)} "
          f"meas_pr_affected={getattr(rtk, 'nlos_meas_pr_affected', 0)} "
          f"meas_cp_affected={getattr(rtk, 'nlos_meas_cp_affected', 0)}")

    _write_outputs("")
    print(f"Wrote {pos_path} ({len(pos_rows)} rows) and {npz_path}")

    if args.lambda_diag_csv:
        import wp13m_lambda_diag as _diag
        _diag.dump_csv(args.lambda_diag_csv)
        print(f"WP13M_LAMBDA_DIAG: wrote {len(_diag.records())} mlambda-call "
              f"records to {args.lambda_diag_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
