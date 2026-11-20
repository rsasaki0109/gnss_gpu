#!/usr/bin/env python3
"""WP16 (TASK_M22): run the RB-FGO-PF Stage-0 prototype over one Tokyo
PPC run window.

The SHARED filter runs the WP13r Q4 stack (T3 env + RELEASE_SEED_HELD/
REF_ONLY) with fix-and-hold NEUTRALIZED (HOLD_RATIO ~ inf -> the graph
stays a pure-float fixed-lag FGO; see rbpf_fgo.py's module docstring for
why the PF's evidence must not be conditioned on single-hypothesis
holds). Subset-AR is disabled for throughput (its output only ever fed
the commit path, which is off; the PF runs its own mlambda top-K).

Usage:
  .venv/Scripts/python.exe wp16_run_rbpf.py <run> --max-ep 3000 \
      --out-dir results/wp16/<probe> [--rb-n 64 --rb-k 12 --rb-beta 0.5]

Writes run{N}.pos (Q=4 on PF fix), run{N}.npz (per-epoch PF diagnostics:
gamma, nb, n_basins, ess, map_err, ...) and trace.csv (per-epoch basin
mass traces inside --trace-tow windows).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TC_SRC = _HERE / "tc" / "src"
for p in (str(_TC_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _set_shared_filter_env(args):
    """WP13r Q4 config (results/wp13r/probe.sh + wp13a runner CLI
    mapping), with the fix-and-hold machinery neutralized and subset-AR
    off. Set BEFORE importing gtsam_rtk_standalone consumers."""
    env = {
        # runner CLI-equivalent block (probe.sh values)
        'LAG': '1.0', 'HUBER_PR': '0.0', 'HUBER_CP': '1.0', 'AR_MODE': '3',
        'SIG_DYN': '5.0',
        # WP17: HOLD_RATIO (phase-1 scope) back to the probe's 5. WP16
        # set 1e9 GLOBALLY, which the WP13o O3 probe already measured to
        # damage ONLY the phase-1 stationary bootstrap (ep0-200 RMS
        # 1.75->4.05; the PF never runs in phase 1) -- measured here as
        # ne[0,200) float RMS 1.9-5.7 vs WP13r's 0.04-2.5, the entire
        # residual AllRMS gap. The PF's pure-float requirement is
        # phase-2 only and stays enforced via HOLD_RATIO_P2=1e9 (the
        # phase-2 override applied at the transition/graph re-seed).
        'HOLD_RATIO': '5',
        'HOLD_CONFIRM_N': '1', 'HOLD_VALPOS_THRES': '0.0',
        'HOLD_STREAK_TOL': '0.05', 'MIN_COMMITTED_FOR_FIX': '0',
        'RELEASE_K': '0', 'RELEASE_THRES': '1.0', 'VALPOS_THRES': '0.0',
        'FDE_ENABLE': '1', 'FDE_PR': '4.0', 'FDE_CP': '0.5',
        'FDE_MAX_FRAC': '0.5', 'FDE_MAX_ITER': '1', 'FDE_MEDIAN_SUB': '0',
        'FDE_REMOVE_CP': '1',
        'PHASE1_EPOCHS': '200', 'LEVER_ARM': '0.31,0,0.55',
        'SANITY_ENABLE': str(args.sanity_enable),
        'SANITY_POSE_REPLACE_THRESH': '5.0',
        'MAIN_DDPR_RES_CATASTROPHIC': '15.0', 'DDPR_SANITY_PERSIST': '3',
        'MAIN_DDPR_RES_THRESH': '3.0', 'DDPR_FAST_WORST_SAT_MIN': '10.0',
        'DDPR_FAST_MIN_PERSIST': '2', 'SANITY_DO_RESET': '1',
        'NHC_ENABLE': '1', 'NHC_MIN_SPEED': '0.0', 'NHC_SIGMA_LAT': '0.3',
        'NHC_SIGMA_VERT': '0.2', 'NHC_LEVER': '0,0,0',
        'ZUPT_ENABLE': '1', 'ZUPT_MAX_ACC_STD': '0.55',
        'ZUPT_MAX_GYRO_STD': '0.030', 'ZUPT_MAX_GYRO_MEDIAN': '0.020',
        'ZUPT_MIN_SAMPLES': '5', 'ZUPT_SIGMA_ZERO_VELOCITY': '0.5',
        'ZUPT_SIGMA_ZERO_ROTATION': '0.010',
        'DOPPLER_VEL_SIGMA': '0.5', 'DOPPLER_MAX_RES': '2.0',
        'AR_DDPR_XVALIDATE_THRESH': '10.0',
        'AR_DDPR_XVALIDATE_DELTA_THRESH': '0.0',
        'NLOS_ENABLE': '0', 'NLOS_RUN': str(args.run),
        'NLOS_AR_EXCLUDE': '0', 'NLOS_COMMIT_REJECT': '0',
        'NLOS_MEAS_MODE': '0', 'NLOS_MEAS_SIGMA_MULT': '5.0',
        'NLOS_REPORT_FRAC': '0.5',
        'AR_RTKLIB_MODE': '1', 'AR_ARFILTER': '1', 'AR_MINFIXSATS': '4',
        'PAR_P0': '0.995',
        'SUBSET_AR_ENABLE': '0',        # probe: 1  -> off (throughput; PF
        'SUBSET_AR_MAX_DROP': '2',      #   does its own top-K mlambda)
        'SUBSET_AR_MIN_NB': '4', 'SUBSET_AR_MAX_CANDIDATES': '5',
        'SUBSET_AR_HOLD_RATIO': '1.0',
        'SAT_BADNESS_ENABLE': '1', 'SAT_BADNESS_DDPR_THRESH': '1.0',
        'SAT_BADNESS_SIGMA_SCALE_CP': '1.5', 'SAT_BADNESS_EWMA_ALPHA': '0.3',
        'SAT_BADNESS_DECAY': '0.9',
        'CP_HOLD_ENABLE': '0', 'CP_HOLD_TRIGGER_THRESH': '3.0',
        'CP_HOLD_EPOCHS': '5',
        'LAMBDA_CORR_MAX': '0.0', 'LAMBDA_CORR_HARD_MAX': '0.0',
        'LOW_NB_FIX_REJECT_NB_MAX': '6', 'WEAK_FIX_NB_MAX': '2',
        'WEAK_FIX_LAMBDA_CORR_MAX': '0.08',
        'WEAK_FIX_MAIN_DDPR_RES_MAX': '0.8',
        'AR_CONTEXT_MAIN_DDPR_MAX': '1.2', 'AR_CONTEXT_WORST_SAT_MAX': '4.0',
        'AR_CONTEXT_NB_MAX': '6', 'AR_CONTEXT_REJECT_DURING_CP_HOLD': '1',
        'AR_CONTEXT_REJECT_DURING_DDPR_BAD': '1',
        'PAR_RATIO_MIN': '5.0', 'ELMIN_DEG': '15.0', 'CNR_MIN_DBHZ': '25.0',
        # probe.sh env block (B + O11 + T3)
        'SLIP_RESET_ENABLE': '1', 'SLIP_RESET_MW_THRESH': '1.0',
        'SLIP_RESET_CMC_THRESH': '3.0', 'AR_WAIT_NEW': '3',
        'PER_SAT_GATE_ENABLE': '1', 'VARERR_ENABLE': '1',
        'SOLVE_RESET_AFTER': '3', 'PER_SAT_GATE_PR_ONLY': '1',
        'AR_CONTEXT_PR_ONLY': '1', 'THRESAR_P2': '3.0',
        'HOLD_RATIO_P2': '1e9',         # probe: 3  -> commit path OFF in p2
        'LAMBDA_CORR_HARD_MAX_P2': '1.0', 'SUBSET_AR_USE_PAR': '1',
        'HOLD_CONFIRM_N_P2': '2', 'MIN_COMMITTED_FOR_FIX_P2': '1',
        'FDE_RELEASE_REF': '1',
        # Q4 delta
        'RELEASE_SEED_HELD': '1', 'RELEASE_SEED_REF_ONLY': '1',
    }
    for k, v in env.items():
        os.environ[k] = v
    # WP16 PF knobs
    os.environ['RBPF_N'] = str(args.rb_n)
    os.environ['RBPF_K'] = str(args.rb_k)
    os.environ['RBPF_BETA'] = str(args.rb_beta)
    os.environ['RBPF_GAMMA'] = str(args.rb_gamma)
    os.environ['RBPF_MIN_FIX_NB'] = str(args.rb_min_nb)
    if args.rb_seed is not None:
        os.environ['RBPF_SEED'] = str(args.rb_seed)
    os.environ['RBPF_BASE_AR'] = str(args.run_base_ar)


def _override_bands(sigs, sigsb, sig_map_rov, sig_map_base, spec, nf,
                    required):
    """Re-pick the bands of the systems named in `spec` ('C:2,6,5;G:1,2,5',
    RINEX system letters) in the given order, keeping only bands that both
    receivers provide for every required observation type ('R:-' drops
    the system)."""
    from cssrlib.gnss import char2sys
    from gnss_fgo.utils.sig_autodetect import _group_by_band
    for part in spec.split(";"):
        if not part.strip():
            continue
        letter, bands = part.split(":")
        sys_id = char2sys(letter.strip())
        if bands.strip() == "-":     # drop the system
            sigs = [x for x in sigs if x.sys != sys_id]
            sigsb = [x for x in sigsb if x.sys != sys_id]
            print(f"WP45: system {letter} dropped")
            continue
        order = [int(b) for b in bands.split(",")]
        if sys_id not in sig_map_rov or sys_id not in sig_map_base:
            continue
        rov = {t: _group_by_band(sig_map_rov[sys_id], t) for t in required}
        base = {t: _group_by_band(sig_map_base[sys_id], t) for t in required}
        ok = [b for b in order
              if all(b in rov[t] and b in base[t] for t in required)][:nf]
        if len(ok) < nf:
            raise SystemExit(f"--band-priority {part}: only bands {ok} are "
                             f"common to rover and base")
        sigs = [x for x in sigs if x.sys != sys_id]
        sigsb = [x for x in sigsb if x.sys != sys_id]
        for b in ok:
            for t in required:
                sigs.append(rov[t][b])
                sigsb.append(base[t][b])
        print(f"WP45: system {letter} bands {ok}")
    return sigs, sigsb


def _rule_fix(rule, out, cols, ecef, flt_ecef, guard_m=5.0, flt_max=0.0):
    """WP45 causal FIX label for this epoch (fix_rule_loro.apply on one
    row): PF agreement counts, the non-empty agreement with its mass, or
    the shared filter's own LAMBDA fix, behind the PF output guards
    (reported position within guard_m of the code-only DDPR position and
    the float / DDPR / fix vote 6 m / 3.5 m). WP56: flt_max > 0 also vetoes
    the label while the reported position is more than flt_max from the
    shared graph's float."""
    t, f, t2, f2, m, b = rule[:6]
    # optional 7th field (WP48): veto every label while the PF's code-only
    # DDPR quality exceeds Q (fix_rule_loro.py --guards `qmax`)
    if len(rule) > 6 and rule[6] < 99:
        q_ = cols["rb_ddpr_qual"][-1]
        if np.isfinite(q_) and q_ > rule[6]:
            return False
    if flt_max > 0:
        d_ = float(np.linalg.norm(np.asarray(ecef, dtype=float)
                                  - np.asarray(flt_ecef, dtype=float)))
        if d_ > flt_max:
            return False
    base_fix = cols["base_smode"][-1] == 4 and cols["base_nb"][-1] >= b
    pf = False
    if out is not None and out.get("ok"):
        nb_at = out.get("nb_at") or [0] * 6
        nb_ne = out.get("nb_at_ne") or [0] * 6
        ne_mass = float(out.get("ne_mass", float("nan")))
        pf = (nb_at[int(t)] >= f
              or (nb_ne[int(t2)] >= f2 and ne_mass >= m))
    if not (pf or base_fix):
        return False
    ddpr = cols["ddpr_xyz"][-1]
    if np.all(np.isfinite(ddpr)):
        d_fix = float(np.linalg.norm(np.asarray(ecef) - ddpr))
        d_flt = float(np.linalg.norm(np.asarray(flt_ecef) - ddpr))
        if d_fix > guard_m or (d_flt > 6.0 and d_fix > 3.5):
            return False
    return True


# WP44: `rb_reason` codes (EpochProblem.reason)
RB_REASONS = ("", "no_joint_marginal", "nonfinite_pos", "no_pairs",
              "nonfinite_marginal", "no_rows")


def _parse_trace_windows(spec):
    wins = []
    for part in (spec or '').split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(':')
        wins.append((float(a), float(b)))
    return wins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=int, choices=(1, 2, 3))
    ap.add_argument("--max-ep", type=int, default=3000)
    ap.add_argument("--out-dir", type=Path, default=_HERE / "results" / "wp16" / "probe")
    ap.add_argument("--rb-n", type=int, default=64)
    ap.add_argument("--rb-k", type=int, default=12)
    ap.add_argument("--rb-beta", type=float, default=0.5)
    ap.add_argument("--rb-gamma", type=float, default=0.99)
    ap.add_argument("--rb-min-nb", type=int, default=5)
    ap.add_argument("--rb-seed", type=int, default=None,
                    help="PF RNG seed; when omitted, preserve RBPF_SEED or "
                         "use RbpfConfig's documented default")
    ap.add_argument("--run-base-ar", type=int, default=1,
                    help="1=also run the base _do_ar (ratio diagnostics for "
                         "comparison); 0=skip it (throughput), keeping only "
                         "tc/'s per-sat vsat gate")
    ap.add_argument("--sanity-enable", type=int, default=0,
                    help="WP13r run2 ship uses 0; run1 recovery configs use 1")
    ap.add_argument("--trace-tow", type=str, default="",
                    help="comma list a:b of tow windows to dump per-epoch "
                         "basin-mass traces for (trace.csv)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=1500,
                    help="WP18: write partial .pos/.npz (+ phase-2 NavState) "
                         "every N synced epochs (0=disable) so an external "
                         "kill loses at most one interval -- ported from "
                         "wp13a_run_standalone_rtk.py")
    ap.add_argument("--resume", action="store_true",
                    help="WP18: if <out-dir>/run{N}.npz exists, load its "
                         "checkpointed rows, fast-forward the sync generator "
                         "past them (no rtk.process()), seed Phase 2 directly "
                         "from the persisted NavState "
                         "(_restore_phase2_from_checkpoint, WP13g), and "
                         "continue appending. The graph AND the PF restart "
                         "fresh at the boundary (brief re-convergence, like "
                         "any RTK engine restart); the graph re-seed arms the "
                         "PF's re-entry spawner automatically.")
    ap.add_argument("--dr-fill", type=int, default=0,
                    help="WP43: output an IMU dead-reckoned position (smode 6, "
                         "written as float) on rover epochs the graph abandons "
                         "(last reported position + predicted antenna motion)")
    ap.add_argument("--dr-max-s", type=float, default=1e9,
                    help="WP43: longest gap [s] since the last committed epoch "
                         "that --dr-fill bridges")
    ap.add_argument("--gap-fill", type=int, default=0,
                    help="WP59: inside a rover observation gap (no synced "
                         "epoch for more than 1.5 x --gap-dt), output IMU "
                         "dead-reckoned positions at the rover rate, using "
                         "only IMU samples up to each output time. Written to "
                         "the npz gap_* arrays; the per-epoch rows are "
                         "unchanged")
    ap.add_argument("--gap-dt", type=float, default=0.2,
                    help="WP59: rover epoch interval [s] for --gap-fill")
    ap.add_argument("--maxage", type=float, default=30.0)
    ap.add_argument("--nf", type=int, default=3)
    ap.add_argument("--fix-rule", default="",
                    help="WP45 causal FIX label t_idx,F,t2_idx,F2,M,B[,Q] "
                         "(the fix_rule_loro.py grammar; Q vetoes labels while "
                         "the DDPR quality exceeds it): the .pos status and the "
                         "npz `rule_fix` column use it; positions and the "
                         "PF are unchanged. Empty = the PF's own fix flag")
    ap.add_argument("--fix-hyst", default="",
                    help="WP50 hysteresis label L,G,t_idx,F,t2_idx,F2,M,B[,Q]: "
                         "FIX once --fix-rule held L consecutive epochs, then "
                         "while this continuation rule holds (bridging G "
                         "failed epochs)")
    ap.add_argument("--fix-flt-max", type=float, default=0.0,
                    help="WP56: veto the --fix-rule / --fix-hyst label while "
                         "the reported position is more than this [m] from "
                         "the shared graph's float (the forward integer "
                         "carry is not vetoed). 0 = off")
    ap.add_argument("--carry", type=int, default=0,
                    help="WP50 forward integer carry (needs RBPF_CARRY=1): "
                         "an epoch the label leaves unfixed is labelled FIX "
                         "at the carried-integer position while the last "
                         "FIX epoch's integer set still fits the DD carrier")
    ap.add_argument("--af-select", type=int, default=0,
                    help="WP55: on epochs the label leaves unfixed, report "
                         "whichever of the reported position, graph float and "
                         "MAP / mixture tracked beliefs best fits the DD "
                         "carrier modulo whole cycles, averaged over the last "
                         "N epochs (af_select.py; needs RBPF_CARRY=1). 0 = off")
    ap.add_argument("--af-margin", type=float, default=0.1,
                    help="WP55: fit margin a candidate needs over the "
                         "reported position")
    ap.add_argument("--reverse", type=int, default=0,
                    help="WP51: backward pass. Runs the filter on a "
                         "time-mirrored copy of the run (reverse_time.py); "
                         "tow, truth and outputs stay in true time, rows "
                         "in processing (reverse) order")
    ap.add_argument("--band-priority", default="",
                    help="WP45: per-system RINEX band order overriding the "
                         "auto-detect L1>L2>L5 priority, e.g. 'C:2,6,5' picks "
                         "BeiDou B1I+B3I(+B2a): B1C (band 1) exists on BDS-3 "
                         "only, so f0&f1 = B1C+B1I keeps ~23%% of the tracked "
                         "BeiDou satellites on Tokyo run1")
    ap.add_argument("--env", action="append", default=[],
                    help="extra KEY=VAL env overrides (e.g. the WP13r run1 "
                         "recovery flag set), applied after the base config")
    args = ap.parse_args()

    _set_shared_filter_env(args)
    for kv in args.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v

    # imports AFTER env (module init reads nothing, but instance ctors do)
    import cssrlib.rinex as rn
    import cssrlib.gnss as gn
    from cssrlib.gnss import uTYP
    from gnss_fgo.utils.sig_autodetect import auto_detect_signals
    from gnss_fgo.utils.geometry import load_imu_csv
    from wp13a_run_standalone_rtk import load_reference, write_pos_file, _DATA_ROOT
    from rbpf_fgo import (RbpfConfig, RbpfFgo, RbpfGtsamRtkTc,
                          WitnessGtsamRtkTc)

    run_dir = _DATA_ROOT / f"run{args.run}"
    obsfile, basefile = run_dir / "rover.obs", run_dir / "base.obs"
    navfile, reffile = run_dir / "base.nav", run_dir / "reference.csv"
    imufile = run_dir / "imu.csv"
    for p in (obsfile, basefile, navfile, reffile, imufile):
        if not p.exists():
            print(f"ERROR: missing {p}")
            return 1

    dec, decb = rn.rnxdec(), rn.rnxdec()
    dec.decode_obsh(str(obsfile))
    decb.decode_obsh(str(basefile))
    sigs, sigsb = auto_detect_signals(
        dec.sig_map, decb.sig_map, max_freq=args.nf,
        required=(uTYP.C, uTYP.L, uTYP.S))
    if args.band_priority:
        sigs, sigsb = _override_bands(sigs, sigsb, dec.sig_map, decb.sig_map,
                                      args.band_priority, args.nf,
                                      (uTYP.C, uTYP.L, uTYP.S))
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
        print("ERROR: base position not in RINEX header")
        return 1
    ref_tows, ref_ecefs = load_reference(reffile)
    print(f"Loaded {len(ref_tows)} reference epochs")
    nav.rb = base_ecef.tolist()
    nav.pmode = 1
    nav.ephopt = 0
    pos0 = np.array(dec.pos) if np.linalg.norm(dec.pos) > 0 else base_ecef.copy()
    witness_nav = (copy.deepcopy(nav)
                   if int(os.environ.get('RBPF_WITNESS', '0')) else None)

    imu_data = load_imu_csv(str(imufile))
    print(f"Loaded {len(imu_data)} IMU samples")

    trace_windows = _parse_trace_windows(args.trace_tow)
    trace_rows = []
    row_dump, row_epochs = [], []   # WP45 RBPF_DUMP_ROWS diagnostic
    gap_rows = []                   # WP59 --gap-fill output

    mirror = None
    if args.reverse:
        # WP51: feed the filter a time-mirrored world (reverse_time.py)
        import reverse_time as rtime
        os.environ['RBPF_REVERSE'] = '1'
        os.environ.setdefault('PHASE1_STUCK_N', '5')
        # WP57: take the IMU bias from a parked window, not the run's end
        os.environ.setdefault('IMU_STATIC_SEARCH', '1')
        if not any(kv.startswith('PHASE1_EPOCHS=') for kv in args.env):
            # a 20 s GNSS-only bootstrap (forward: 40 s); runs mostly end
            # parked, so the backward PF starts 20 s earlier
            os.environ['PHASE1_EPOCHS'] = '100'
        rov_l, bas_l = rtime.read_all(dec), rtime.read_all(decb)
        (w0, tw0), (w1, tw1) = gn.time2gpst(rov_l[0].t), gn.time2gpst(rov_l[-1].t)
        if w0 != w1:
            print("ERROR: --reverse needs the run inside one GPS week")
            return 1
        mirror = rtime.Mirror(rov_l[0].t, rov_l[-1].t, tw0, tw1)
        dec, decb = (rtime.ListDecoder(rtime.mirror_obs(rov_l, mirror)),
                     rtime.ListDecoder(rtime.mirror_obs(bas_l, mirror)))
        del rov_l, bas_l
        imu_data = rtime.mirror_imu(imu_data, mirror)
        rtime.patch_satposs(mirror)
        print(f"--reverse: mirrored about tow {mirror.mtow / 2:.2f}")

    def _true_tow(tow_):
        return float(mirror.tow(tow_)) if mirror is not None else tow_

    sync_gen = rn.sync_obs_hold(dec, decb, maxage=args.maxage)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pos_path = args.out_dir / f"run{args.run}.pos"
    npz_path = args.out_dir / f"run{args.run}.npz"
    trace_path = args.out_dir / "trace.csv"

    _FCOLS = ("rb_ddpr_dpos", "rb_ddpr_qual", "rb_fix_dpr", "rb_min_info",
              "rb_rep_dpr", "rb_witness_chi", "rb_loo_max_m",
              "rb_loo_med_m", "rb_loo_n", "rb_delay_5s_m",
              "rb_delay_20s_m", "rb_delay_corr_cos",
              "rb_delay_cross_cos", "rb_delayed_rejuv_trigger",
              "rb_shadow_promote", "rb_shadow_n")
    cols = {k: [] for k in (
        "ne", "tow", "week", "ecef", "witness_xyz", "witness_err3d",
        "smode", "err3d", "nsat_dd",
        "flt_err3d", "flt_xyz", "map_err3d", "ar_ratio", "ar_raw_nb", "ar_phase",
        "gamma", "second_gamma", "gamma_pair", "rb_nb", "rb_nbasins", "rb_ess",
        "rb_ok", "rb_npairs", "rb_nheld", "rb_spawn",
        "trk_map_xyz", "trk_mix_xyz", "trk_map_mass",
        "trk_var_xyz", "trk_var_mass", "rb_nb_at", "ddpr_xyz",
        "base_smode", "base_nb", "rb_reason", "rb_nb_at_ne", "rb_afv_n", "rb_afv_frac",
        "rb_ne_mass", "rule_fix", "hyst_fix", "carry_fwd", "pos_src", "pre_af_xyz", "rb_wl_n", "rb_wl_ratio", "rb_wl_rms",
        "wl_xyz", "wlpf_ok", "wlpf_nb_at", "wlpf_nb_at_ne", "wlpf_ne_mass",
        "wlpf_npairs", "wlpf_xyz") + _FCOLS}
    fix_rule = (tuple(float(x) for x in args.fix_rule.split(","))
                if args.fix_rule else None)
    hyst_rule, hyst = None, None
    if args.fix_hyst:
        from integer_carry import HysteresisLabel
        _h = [float(x) for x in args.fix_hyst.split(",")]
        hyst = HysteresisLabel(entry=int(_h[0]), bridge=int(_h[1]))
        hyst_rule = tuple(_h[2:])
    carrier = None
    if args.carry:
        from integer_carry import IntegerCarry
        carrier = IntegerCarry()
    # WP52: 1 = hold the carried integer set in the shared graph on carried
    # epochs, 2 = also on labelled FIX epochs (needs --carry 1)
    carry_fb = int(os.environ.get('RBPF_CARRY_FB', '0'))
    af_sel = None
    if args.af_select > 0:
        from af_select import CausalSelector
        af_sel = CausalSelector(past=args.af_select, margin=args.af_margin)
    pos_rows = []
    rtk = None

    def _capture_phase2_checkpoint():
        """WP13g Part 2 pattern (ported from wp13a_run_standalone_rtk.py):
        persist enough Phase-2 NavState for --resume to bypass the
        Phase-1 stationary bootstrap entirely."""
        if (rtk is None or getattr(rtk, "phase", 1) != 2
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

    def _flush_trace():
        if not trace_rows:
            return
        new_file = not trace_path.exists()
        with trace_path.open("a") as fh:
            if new_file:
                fh.write("ne,tow,gamma,ess,n_basins,rank,mass,nb,empty,err3d\n")
            for r in trace_rows:
                fh.write(",".join(
                    f"{x:.6f}" if isinstance(x, float) else str(x)
                    for x in r) + "\n")
        trace_rows.clear()

    def _write_outputs():
        write_pos_file(pos_path, pos_rows,
                       f"WP16/17/18 RB-FGO-PF, tokyo run{args.run}, "
                       f"N={args.rb_n} K={args.rb_k} beta={args.rb_beta} "
                       f"gamma={args.rb_gamma} min_nb={args.rb_min_nb} "
                       f"base=Q4-floatified")
        savez_kwargs = dict(
            ne=np.asarray(cols["ne"]), tow=np.asarray(cols["tow"]),
            week=np.asarray(cols["week"]),
            sol_xyz=(np.asarray(cols["ecef"]) if cols["ecef"]
                     else np.zeros((0, 3))),
            witness_xyz=(np.asarray(cols["witness_xyz"])
                         if cols["witness_xyz"] else np.zeros((0, 3))),
            witness_err3d=np.asarray(cols["witness_err3d"]),
            smode=np.asarray(cols["smode"]), err3d=np.asarray(cols["err3d"]),
            nsat_dd=np.asarray(cols["nsat_dd"]),
            flt_err3d=np.asarray(cols["flt_err3d"]),
            flt_xyz=(np.asarray(cols["flt_xyz"]) if cols["flt_xyz"]
                     else np.zeros((0, 3))),
            map_err3d=np.asarray(cols["map_err3d"]),
            ar_ratio=np.asarray(cols["ar_ratio"]),
            ar_raw_nb=np.asarray(cols["ar_raw_nb"]),
            ar_phase=np.asarray(cols["ar_phase"]),
            gamma=np.asarray(cols["gamma"]),
            second_gamma=np.asarray(cols["second_gamma"]),
            gamma_pair=np.asarray(cols["gamma_pair"]),
            rb_nb=np.asarray(cols["rb_nb"]),
            rb_nbasins=np.asarray(cols["rb_nbasins"]),
            rb_ess=np.asarray(cols["rb_ess"]),
            rb_ok=np.asarray(cols["rb_ok"]),
            rb_npairs=np.asarray(cols["rb_npairs"]),
            rb_nheld=np.asarray(cols["rb_nheld"]),
            rb_spawn=np.asarray(cols["rb_spawn"]),
            trk_map_xyz=(np.asarray(cols["trk_map_xyz"]) if cols["trk_map_xyz"]
                         else np.zeros((0, 3))),
            trk_mix_xyz=(np.asarray(cols["trk_mix_xyz"]) if cols["trk_mix_xyz"]
                         else np.zeros((0, 3))),
            trk_map_mass=np.asarray(cols["trk_map_mass"]),
            trk_var_xyz=np.asarray(cols["trk_var_xyz"], dtype=float),
            trk_var_mass=np.asarray(cols["trk_var_mass"], dtype=float),
            rb_nb_at=np.asarray(cols["rb_nb_at"], dtype=int),
            ddpr_xyz=np.asarray(cols["ddpr_xyz"], dtype=float),
            base_smode=np.asarray(cols["base_smode"], dtype=int),
            base_nb=np.asarray(cols["base_nb"], dtype=int),
            rb_reason=np.asarray(cols["rb_reason"], dtype=int),
            rb_nb_at_ne=np.asarray(cols["rb_nb_at_ne"], dtype=int),
            rb_ne_mass=np.asarray(cols["rb_ne_mass"], dtype=float),
            rb_afv_n=np.asarray(cols["rb_afv_n"], dtype=int),
            rb_afv_frac=np.asarray(cols["rb_afv_frac"], dtype=float),
            rule_fix=np.asarray(cols["rule_fix"], dtype=int),
            hyst_fix=np.asarray(cols["hyst_fix"], dtype=int),
            carry_fwd=np.asarray(cols["carry_fwd"], dtype=int),
            pos_src=np.asarray(cols["pos_src"], dtype=int),
            pre_af_xyz=np.asarray(cols["pre_af_xyz"], dtype=float),
            rb_wl_n=np.asarray(cols["rb_wl_n"], dtype=int),
            rb_wl_ratio=np.asarray(cols["rb_wl_ratio"], dtype=float),
            rb_wl_rms=np.asarray(cols["rb_wl_rms"], dtype=float),
            wl_xyz=np.asarray(cols["wl_xyz"], dtype=float),
            wlpf_ok=np.asarray(cols["wlpf_ok"], dtype=int),
            wlpf_nb_at=np.asarray(cols["wlpf_nb_at"], dtype=int),
            wlpf_nb_at_ne=np.asarray(cols["wlpf_nb_at_ne"], dtype=int),
            wlpf_ne_mass=np.asarray(cols["wlpf_ne_mass"], dtype=float),
            wlpf_npairs=np.asarray(cols["wlpf_npairs"], dtype=int),
            wlpf_xyz=np.asarray(cols["wlpf_xyz"], dtype=float),
            base_xyz=base_ecef,
            rb_n=args.rb_n, rb_k=args.rb_k, rb_beta=args.rb_beta,
            rb_gamma=args.rb_gamma, rb_min_nb=args.rb_min_nb,
            rb_seed=rtk.rbpf.cfg.seed, reverse=int(args.reverse),
        )
        for k in _FCOLS:
            savez_kwargs[k] = np.asarray(cols[k])
        if args.gap_fill:
            # WP59: [true tow, anchor tow, xyz (anchored on the last
            # reported position), IMU delta from the anchor's antenna]
            savez_kwargs["gap_rows"] = (np.asarray(gap_rows, dtype=float)
                                        if gap_rows else np.zeros((0, 8)))
        ckpt = _capture_phase2_checkpoint()
        if ckpt is not None:
            savez_kwargs.update(ckpt)
        else:
            savez_kwargs["ckpt_phase2_valid"] = 0
        np.savez(npz_path, **savez_kwargs)
        _flush_trace()

    # WP18 --resume: fast-forward past checkpointed synced epochs and seed
    # phase 2 directly from the persisted NavState (wp13a/WP13g pattern).
    start_ne = 0
    resume_phase2_state = None
    if args.resume and npz_path.exists():
        old = np.load(npz_path)
        old_ne = np.asarray(old["ne"]).reshape(-1)
        if ("ckpt_phase2_valid" in old.files
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
            for k in cols:
                if k in ("ecef", "witness_xyz", "flt_xyz", "trk_map_xyz",
                         "trk_mix_xyz", "pre_af_xyz"):
                    source = "sol_xyz" if k == "ecef" else k
                    if source in old.files:
                        cols[k] = [row.copy() for row in np.asarray(old[source])]
                    else:
                        cols[k] = [np.full(3, np.nan) for _ in old_ne]
                elif k in ("rb_nb_at", "ddpr_xyz", "rb_nb_at_ne"):
                    arr_ = np.asarray(old[k]) if k in old.files else None
                    if arr_ is not None and arr_.ndim == 2 and arr_.shape[0] == old_ne.size:
                        cols[k] = list(arr_)
                    else:   # pre-WP44 checkpoint
                        cols[k] = [np.zeros(6, dtype=int) if k != "ddpr_xyz"
                                   else np.full(3, np.nan) for _ in old_ne]
                elif k in ("trk_var_xyz", "trk_var_mass"):
                    nv_ = 1 + len([v for v in os.environ.get(
                        "RBPF_TRACK_VARIANTS", "").split(",") if v.strip()])
                    arr_ = np.asarray(old[k]) if k in old.files else None
                    if arr_ is not None and arr_.ndim >= 2 and arr_.shape[0] == old_ne.size:
                        cols[k] = list(arr_)
                    else:   # pre-WP43 checkpoint
                        shape_ = (nv_, 3) if k == "trk_var_xyz" else (nv_,)
                        cols[k] = [np.full(shape_, np.nan) for _ in old_ne]
                elif k in old.files:
                    cols[k] = list(np.asarray(old[k]).reshape(-1))
                elif k == "pos_src":    # pre-WP55 checkpoint
                    cols[k] = [0] * old_ne.size
                else:   # pre-WP18 checkpoint: nan-backfill new diagnostics
                    cols[k] = [float("nan")] * old_ne.size
            for i in range(old_ne.size):
                pos_rows.append((int(cols["week"][i]), float(cols["tow"][i]),
                                 cols["ecef"][i],
                                 4 if int(cols["smode"][i]) == 4 else 5,
                                 int(cols["nsat_dd"][i])))
            start_ne = int(old_ne.max()) + 1
            pos0 = np.asarray(cols["ecef"][-1], dtype=float).copy()
            print(f"--resume: loaded {old_ne.size} rows from {npz_path}, "
                  f"fast-forwarding to synced epoch {start_ne}")
            for _ in range(start_ne):
                try:
                    next(sync_gen)
                except StopIteration:
                    print("--resume: sync generator exhausted during "
                          "fast-forward (checkpoint already complete)")
                    return 0
        else:
            print(f"--resume: {npz_path} has 0 rows, starting from scratch")

    rtk = RbpfGtsamRtkTc(nav, imu_data, pos0)
    witness = (WitnessGtsamRtkTc(witness_nav, imu_data, pos0)
               if witness_nav is not None else None)
    witness_rbpf = RbpfFgo(RbpfConfig()) if witness is not None else None
    # WP62 diagnostic: per-basin graph branches copied from the witness
    # graph (basin_branch_probe.py); observational, off unless asked.
    branch_probe = None
    if witness is not None and int(os.environ.get('RBPF_BRANCH_PROBE', '0')):
        from basin_branch_probe import BasinBranchProbe
        branch_probe = BasinBranchProbe(
            witness, args.out_dir / f"run{args.run}_branch.csv")
    print(f"RBPF: N={args.rb_n} K={args.rb_k} beta={args.rb_beta} "
          f"gamma*={args.rb_gamma} min_nb={args.rb_min_nb} "
          f"base_ar={args.run_base_ar} seed={rtk.rbpf.cfg.seed}")
    if resume_phase2_state is not None:
        import gtsam
        st = resume_phase2_state
        pose0 = gtsam.Pose3(
            gtsam.Rot3(st["pose_R"]), gtsam.Point3(*st["pose_t"]))
        bias0 = gtsam.imuBias.ConstantBias(st["bias_acc"], st["bias_gyro"])
        rtk._restore_phase2_from_checkpoint(
            pose0, st["vel_enu"], bias0, st["week"], st["tow"],
            st["imu_idx"])
        print(f"--resume: restored Phase-2 NavState (phase={rtk.phase}, "
              f"imu_idx={st['imu_idx']} week={st['week']} "
              f"tow={st['tow']:.1f})")

    last_rep = None       # WP43 --dr-fill anchor: (ecef, antenna, tow)
    last_sync_tow = None  # WP59 --gap-fill: previous synced epoch (filter time)
    t_start = time.time()
    n_fix = sum(1 for s in cols["smode"] if int(s) == 4)
    n_flt = sum(1 for s in cols["smode"] if int(s) != 4)
    first_epoch = True
    ne = start_ne - 1
    nep = args.max_ep if args.max_ep > 0 else 10 ** 9
    for ne in range(start_ne, nep):
        try:
            obs, obsb, dt_sync = next(sync_gen)
        except StopIteration:
            break
        if first_epoch:
            nav.t = obs.t
            first_epoch = False
        prev_valid = rtk.last_valid_epoch
        rtk._rbpf_last = None
        rtk._rbpf_witness_prob = None
        rtk._rbpf_witness_pos = None
        rtk._rbpf_witness_P = None
        if len(ref_tows):
            _, _tow0 = gn.time2gpst(obs.t)
            _tow0 = _true_tow(_tow0)
            _ri0 = int(np.argmin(np.abs(ref_tows - _tow0)))
            rtk._rbpf_truth_ecef = (ref_ecefs[_ri0]
                                    if abs(ref_tows[_ri0] - _tow0) < 0.5
                                    else None)
        if witness is not None:
            # GtsamRtkTc preprocessing annotates its observation objects in
            # place.  Keep the diagnostic branch observational only: sharing
            # these objects changes the production filter even when every
            # witness gate is disabled.
            witness.process(copy.deepcopy(obs), obsb=copy.deepcopy(obsb))
            if witness.last_valid_epoch is not None:
                rtk._rbpf_witness_pos = witness.nav.x[0:3].copy()
                rtk._rbpf_witness_P = witness.nav.P[0:3, 0:3].copy()
            if (getattr(witness, 'phase', 1) == 2
                    and witness._rbpf_ctx is not None):
                rtk._rbpf_witness_prob = witness_rbpf._extract(
                    witness, witness._rbpf_ctx)
        if args.gap_fill:
            # WP59: rover gap -> IMU dead reckoning at the rover rate from
            # the last committed state, before this epoch is processed
            _wk, _tw = gn.time2gpst(obs.t)
            if (last_sync_tow is not None and last_rep is not None
                    and _tw - last_sync_tow > 1.5 * args.gap_dt):
                _k = 1
                while last_sync_tow + _k * args.gap_dt < _tw - 0.5 * args.gap_dt:
                    _tg = last_sync_tow + _k * args.gap_dt
                    _k += 1
                    _got = rtk.predict_antenna_at(gn.gpst2time(_wk, _tg))
                    if _got is None:
                        continue
                    _d = np.asarray(_got[0], dtype=float) - last_rep[1]
                    if not np.all(np.isfinite(_d)):
                        continue
                    gap_rows.append(np.concatenate(
                        [[_true_tow(_tg), _true_tow(last_rep[2])],
                         last_rep[0] + _d, _d]))
            last_sync_tow = _tw
        rtk.process(obs, obsb=obsb)
        updated = rtk.last_valid_epoch != prev_valid
        week, tow = gn.time2gpst(obs.t)
        tow_t = _true_tow(tow)      # true time (== tow unless --reverse)
        if branch_probe is not None and updated:
            branch_probe.after_epoch(rtk, ne, tow_t)
        dr_ecef = None
        # WP59 P2_IMU_ONLY: a committed epoch without GNSS rows is reported
        # like a DR epoch, from the last reported position plus the graph's
        # (NHC-constrained) antenna motion since then
        imu_only = updated and bool(getattr(rtk, '_last_imu_only', False))
        if imu_only and last_rep is not None and tow - last_rep[2] <= args.dr_max_s:
            try:
                _ant = rtk._antenna_ecef(rtk.current_estimate.atPose3(
                    rtk.PX(rtk.last_valid_epoch)))
                dr_ecef = last_rep[0] + (_ant - last_rep[1])
                if not np.all(np.isfinite(dr_ecef)):
                    dr_ecef = None
            except RuntimeError:
                dr_ecef = None
        if not updated:
            if not (args.dr_fill and last_rep is not None
                    and tow - last_rep[2] <= args.dr_max_s):
                continue
            got = rtk.predict_antenna_at(obs.t)
            if got is None:
                continue
            dr_ecef = last_rep[0] + (got[0] - last_rep[1])
            if not np.all(np.isfinite(dr_ecef)):
                continue

        out = rtk._rbpf_last if updated else None
        flt_ecef = nav.x[0:3].copy()
        if dr_ecef is not None:
            ecef, smode = dr_ecef, 6
        elif out is not None and out['pos'] is not None:
            ecef = np.asarray(out['pos'], dtype=float)
            smode = 4 if out['fix'] else 5
        else:
            ecef, smode = flt_ecef, 5
        if (updated and not imu_only and (args.dr_fill or args.gap_fill)
                and getattr(rtk, 'phase', 1) == 2
                and rtk.current_estimate is not None):
            try:
                last_rep = (np.asarray(ecef, dtype=float).copy(),
                            rtk._antenna_ecef(rtk.current_estimate.atPose3(
                                rtk.PX(rtk.last_valid_epoch))),
                            tow)
            except RuntimeError:
                last_rep = None
        if smode == 4:
            n_fix += 1
        else:
            n_flt += 1

        ri = int(np.argmin(np.abs(ref_tows - tow_t))) if len(ref_tows) else -1
        have_ref = ri >= 0 and abs(ref_tows[ri] - tow_t) < 0.5

        def _err(p):
            return (float(np.linalg.norm(np.asarray(p) - ref_ecefs[ri]))
                    if (have_ref and p is not None) else float("nan"))

        map_err = float("nan")
        if out is not None and out['top']:
            for tb in out['top']:
                if not tb['empty'] and tb['pos'] is not None:
                    map_err = _err(tb['pos'])
                    break

        cols["ne"].append(ne)
        if out is not None and out.get("rows") is not None:
            # WP45 diagnostic rows (RBPF_DUMP_ROWS=1), keyed by npz row
            ri_ = len(cols["ne"]) - 1
            row_dump.append(np.column_stack(
                [np.full(len(out["rows"]), ri_, dtype=float), out["rows"]]))
            fx_ = out.get("fix_xyz")
            row_epochs.append(np.concatenate(
                [[ri_], out["rows_pos0"],
                 fx_ if fx_ is not None else np.full(3, np.nan)]))
        cols["tow"].append(tow_t)
        cols["week"].append(week)
        cols["ecef"].append(ecef)
        witness_ecef = (np.asarray(rtk._rbpf_witness_pos, dtype=float).copy()
                        if rtk._rbpf_witness_pos is not None
                        else np.full(3, np.nan))
        cols["witness_xyz"].append(witness_ecef)
        cols["witness_err3d"].append(_err(witness_ecef))
        cols["smode"].append(smode)
        cols["err3d"].append(_err(ecef))
        cols["nsat_dd"].append(int(np.sum(nav.vsat[:, 0])))
        cols["flt_err3d"].append(_err(flt_ecef))
        cols["flt_xyz"].append(np.asarray(flt_ecef, dtype=float).copy()
                               if flt_ecef is not None else np.full(3, np.nan))
        cols["map_err3d"].append(map_err)
        for k_ in ("trk_map", "trk_mix"):
            v_ = out.get(k_) if out is not None else None
            cols[k_ + "_xyz"].append(np.asarray(v_, dtype=float).copy()
                                     if v_ is not None else np.full(3, np.nan))
        cols["trk_map_mass"].append(float(out.get("trk_map_mass", float("nan")))
                                    if out is not None else float("nan"))
        # WP44 diagnostic: the shared filter's own report this epoch
        # (phase 1 runs a fix-and-hold AR before the PF starts)
        cols["base_smode"].append(int(getattr(nav, "smode", 0)) if updated else 0)
        cols["base_nb"].append(int(getattr(rtk, "_last_nb", 0) or 0) if updated else 0)
        # WP44 diagnostic: why the PF had no problem this epoch
        # (index into RB_REASONS; -1 = PF not stepped)
        cols["rb_reason"].append(
            RB_REASONS.index(out["reason"]) if out is not None
            and out.get("reason") in RB_REASONS else -1)
        from rbpf_fgo import NB_AT_THRESH
        cols["rb_nb_at"].append(np.asarray(
            out["nb_at"] if out is not None and "nb_at" in out
            else [0] * len(NB_AT_THRESH), dtype=int))
        cols["rb_nb_at_ne"].append(np.asarray(
            out["nb_at_ne"] if out is not None and "nb_at_ne" in out
            else [0] * len(NB_AT_THRESH), dtype=int))
        cols["rb_ne_mass"].append(float(out.get("ne_mass", float("nan")))
                                  if out is not None else float("nan"))
        # WP45 ambiguity-function label feature (RBPF_AFV=1)
        cols["rb_afv_n"].append(int(out.get("afv_n", 0)) if out is not None else 0)
        cols["rb_afv_frac"].append(float(out.get("afv_frac", float("nan")))
                                   if out is not None else float("nan"))
        # WP49 wide-lane diagnostic (RBPF_WL=1)
        cols["rb_wl_n"].append(int(out.get("wl_n", 0)) if out is not None else 0)
        cols["rb_wl_ratio"].append(float(out.get("wl_ratio", float("nan")))
                                   if out is not None else float("nan"))
        cols["rb_wl_rms"].append(float(out.get("wl_rms", float("nan")))
                                 if out is not None else float("nan"))
        wl_ = out.get("wl_pos") if out is not None else None
        cols["wl_xyz"].append(np.asarray(wl_, dtype=float).copy()
                              if wl_ is not None else np.full(3, np.nan))
        # WP49 wide-lane RB-PF (RBPF_WLPF=1)
        wo_ = getattr(rtk, "_rbpf_wl_last", None)
        cols["wlpf_ok"].append(int(bool(wo_ is not None and wo_.get("ok"))))
        cols["wlpf_nb_at"].append(np.asarray(
            wo_["nb_at"] if wo_ is not None and wo_.get("nb_at") is not None
            else [0] * len(NB_AT_THRESH), dtype=int))
        cols["wlpf_nb_at_ne"].append(np.asarray(
            wo_["nb_at_ne"] if wo_ is not None and wo_.get("nb_at_ne") is not None
            else [0] * len(NB_AT_THRESH), dtype=int))
        cols["wlpf_ne_mass"].append(float(wo_.get("ne_mass", float("nan")))
                                    if wo_ is not None else float("nan"))
        cols["wlpf_npairs"].append(int(wo_.get("npairs", 0)) if wo_ is not None else 0)
        wp_ = wo_.get("pos") if wo_ is not None else None
        cols["wlpf_xyz"].append(np.asarray(wp_, dtype=float).copy()
                                if wp_ is not None else np.full(3, np.nan))
        dp_ = out.get("ddpr_xyz") if out is not None else None
        cols["ddpr_xyz"].append(np.asarray(dp_, dtype=float).copy()
                                if dp_ is not None else np.full(3, np.nan))
        nvar_ = len(rtk.rbpf.cfg.track_variants)
        tv_ = out.get("trk_var") if out is not None else None
        cols["trk_var_xyz"].append(np.array(
            [np.full(3, np.nan) if tv_ is None or tv_[v][0] is None
             else np.asarray(tv_[v][0], dtype=float) for v in range(nvar_)]))
        cols["trk_var_mass"].append(np.array(
            [float("nan") if tv_ is None else tv_[v][1]
             for v in range(nvar_)]))
        _s0 = getattr(rtk, '_last_s0', 0.0)
        _s1 = getattr(rtk, '_last_s1', 0.0)
        cols["ar_ratio"].append(float(_s1 / _s0) if _s0 > 0 else 0.0)
        cols["ar_raw_nb"].append(int(getattr(rtk, '_last_resamb_raw_nb', -1)))
        cols["ar_phase"].append(int(getattr(rtk, 'phase', 1)))
        if out is not None:
            cols["gamma"].append(out['gamma'])
            cols["second_gamma"].append(out['second_gamma'])
            cols["gamma_pair"].append(out.get('gamma_pair', 0.0))
            cols["rb_nb"].append(out['nb'])
            cols["rb_nbasins"].append(out['n_basins'])
            cols["rb_ess"].append(out['ess'])
            cols["rb_ok"].append(int(out['ok']))
            cols["rb_npairs"].append(out['npairs'])
            cols["rb_nheld"].append(out.get('nheld', 0))
            cols["rb_spawn"].append(out.get('spawn', 0))
        else:
            cols["gamma"].append(0.0)
            cols["second_gamma"].append(0.0)
            cols["gamma_pair"].append(0.0)
            cols["rb_nb"].append(0)
            cols["rb_nbasins"].append(0)
            cols["rb_ess"].append(float("nan"))
            cols["rb_ok"].append(0)
            cols["rb_npairs"].append(0)
            cols["rb_nheld"].append(0)
            cols["rb_spawn"].append(0)
        for k, ok_ in (("rb_ddpr_dpos", "ddpr_dpos"),
                       ("rb_ddpr_qual", "ddpr_qual"),
                       ("rb_fix_dpr", "fix_dpr"),
                       ("rb_min_info", "min_info"),
                       ("rb_rep_dpr", "rep_dpr"),
                       ("rb_witness_chi", "witness_chi"),
                       ("rb_loo_max_m", "loo_max_m"),
                       ("rb_loo_med_m", "loo_med_m"),
                       ("rb_loo_n", "loo_n"),
                       ("rb_delay_5s_m", "delay_5s_m"),
                       ("rb_delay_20s_m", "delay_20s_m"),
                       ("rb_delay_corr_cos", "delay_corr_cos"),
                       ("rb_delay_cross_cos", "delay_cross_cos"),
                       ("rb_delayed_rejuv_trigger", "delayed_rejuv_trigger"),
                       ("rb_shadow_promote", "shadow_promote"),
                       ("rb_shadow_n", "shadow_n")):
            cols[k].append(float(out.get(ok_, float("nan")))
                           if out is not None else float("nan"))
        rule_ok = (_rule_fix(fix_rule, out, cols, ecef, flt_ecef,
                             flt_max=args.fix_flt_max)
                   if fix_rule is not None and smode != 6 else False)
        hyst_ok = rule_ok
        if hyst is not None:
            cont_ok = (_rule_fix(hyst_rule, out, cols, ecef, flt_ecef,
                                 flt_max=args.fix_flt_max)
                       if smode != 6 else False)
            hyst_ok = hyst.update(tow, rule_ok, cont_ok)
            rule_ok = hyst_ok
        carried = False
        if carrier is not None:
            rows_c = out.get("carry_rows") if out is not None else None
            p0_c = out.get("rows_pos0") if out is not None else None
            if rows_c is None or p0_c is None or smode == 6:
                carrier.reset()
                carrier.t_prev = tow
            elif hyst_ok:
                carrier.init(rows_c, p0_c, ecef, tow)
            else:
                x_c = carrier.step(rows_c, p0_c, tow)
                if x_c is not None:
                    carried = True
                    rule_ok = True
                    ecef = np.asarray(x_c, dtype=float)
                    cols["ecef"][-1] = ecef
                    cols["err3d"][-1] = _err(ecef)
            if carry_fb:
                # WP52: hold the carried integers in the shared graph
                fb_on = carried or (carry_fb >= 2 and hyst_ok)
                rtk.rbpf.carry_feedback(rtk, carrier.state if fb_on else None)
        cols["pre_af_xyz"].append(np.asarray(ecef, dtype=float).copy())
        pos_src = 0
        if af_sel is not None:
            # WP55: causal ambiguity-fit choice of the reported position
            cand = [ecef, flt_ecef, cols["trk_map_xyz"][-1],
                    cols["trk_mix_xyz"][-1]]
            pos_src = af_sel.step(
                out.get("carry_rows") if out is not None else None,
                out.get("rows_pos0") if out is not None else None,
                cand, not rule_ok)
            if pos_src:
                ecef = np.asarray(cand[pos_src], dtype=float)
                cols["ecef"][-1] = ecef
                cols["err3d"][-1] = _err(ecef)
        cols["pos_src"].append(pos_src)
        cols["hyst_fix"].append(int(hyst_ok))
        cols["carry_fwd"].append(int(carried))
        cols["rule_fix"].append(int(rule_ok))
        q_ = (4 if rule_ok else 5) if fix_rule is not None else (
            4 if smode == 4 else 5)
        pos_rows.append((week, tow_t, ecef, q_, int(np.sum(nav.vsat[:, 0]))))

        if out is not None and any(a <= tow <= b for a, b in trace_windows):
            for rank, tb in enumerate(out['top']):
                trace_rows.append((
                    ne, tow, out['gamma'], out['ess'], out['n_basins'],
                    rank, tb['mass'], tb['nb'], int(tb['empty']),
                    _err(tb['pos'])))

        if (ne + 1) % args.log_every == 0:
            el_s = time.time() - t_start
            pf = rtk.rbpf
            print(f"  ep {ne+1:6d} tow={tow:10.1f} fix={n_fix:5d} "
                  f"flt={n_flt:5d} phase={getattr(rtk, 'phase', 1)} "
                  f"pf_steps={pf.n_steps} pf_meas={pf.n_meas_updates} "
                  f"pf_skip={pf.n_skipped} resamp={pf.n_resamples} "
                  f"pf_s={pf.step_seconds:.1f}s "
                  f"({(ne+1-start_ne)/max(el_s,1e-6):.2f} ep/s)", flush=True)

        if (args.checkpoint_every > 0
                and (ne + 1) % args.checkpoint_every == 0):
            _write_outputs()
            print(f"  checkpoint: wrote {len(pos_rows)} rows through "
                  f"ep {ne+1}", flush=True)

    elapsed = time.time() - t_start
    if branch_probe is not None:
        branch_probe.close()
        print(f"WP62 branch probe: {branch_probe.n_events} events -> "
              f"{args.out_dir / f'run{args.run}_branch.csv'}")
    pf = rtk.rbpf
    print(f"Done: {ne+1} synced epochs, {len(pos_rows)} committed "
          f"({n_fix} fix, {n_flt} float) in {elapsed:.1f}s "
          f"= {(ne+1-start_ne)/max(elapsed,1e-6):.2f} ep/s total; "
          f"PF {pf.step_seconds:.1f}s over {pf.n_steps} steps "
          f"({1e3*pf.step_seconds/max(pf.n_steps,1):.1f} ms/step)")
    if pf.cfg.fb_enable or pf.cfg.spawn_enable or pf.cfg.use_cuda:
        print(f"WP17: fb_commits={pf.fb_commit_count} "
              f"fb_rel_mass={pf.fb_release_mass} "
              f"fb_rel_unseen={pf.fb_release_unseen} "
              f"spawn_events={pf.spawn_events} "
              f"gpu_topk={pf.gpu_topk_calls} "
              f"gpu_fallback={pf.gpu_topk_fallbacks}")
    if carry_fb:
        print(f"WP52: carry_fb={carry_fb} epochs={pf.carry_fb_epochs} "
              f"commits={pf.carry_fb_commits} skipped={pf.carry_fb_skipped}")
    print(f"WP18+: fb_rel_ddpr={pf.fb_release_ddpr} "
          f"fb_rel_support={pf.fb_release_support} "
          f"fb_dual_gated={pf.fb_dual_gated} "
          f"fb_witness_gated={pf.fb_witness_gated} "
          f"fb_commit_gated={pf.fb_commit_gated} "
          f"fix_info_rejects={pf.fix_info_rejects} "
          f"fix_vote_rejects={pf.fix_vote_rejects} "
          f"chal_spawns={pf.chal_spawn_events} "
          f"delayed_rejuv={pf.delayed_rejuv_events} "
          f"shadow_promotions={pf.shadow_promotions} "
          f"(FIX_MIN_INFO={pf.cfg.fix_min_info} "
          f"FIX_VOTE_DD={pf.cfg.fix_vote_dd}/{pf.cfg.fix_vote_dpr} "
          f"FB_COMMIT_MAX_DPR={pf.cfg.fb_commit_max_dpr} "
          f"FB_DDPR_REL_D={pf.cfg.fb_ddpr_rel_d}/{pf.cfg.fb_ddpr_rel_m} "
          f"CHAL={pf.cfg.chal_d}/{pf.cfg.chal_m} "
          f"DDPR_QUAL_MAX={pf.cfg.ddpr_qual_max})")

    if pf.cfg.keep:
        print(f"WP44: keep=1 maxout_s={pf.cfg.keep_maxout_s} "
              f"drop={sorted(pf.cfg.keep_drop)}")
    if pf.cfg.track:
        print(f"WP43: track_evals={pf.track_evals} "
              f"({pf.track_evals / max(pf.n_meas_updates, 1):.1f}/epoch) "
              f"Q={pf.cfg.track_q} floor={pf.cfg.track_floor} "
              f"inherit={pf.cfg.track_inherit} empty={pf.cfg.track_empty} "
              f"out={pf.cfg.track_out}")
    _write_outputs()
    if row_dump:
        rows_path = npz_path.with_name(npz_path.stem + "_rows.npz")
        np.savez_compressed(rows_path, rows=np.concatenate(row_dump),
                            epochs=np.asarray(row_epochs))
        print(f"WP45: wrote {len(row_epochs)} epochs of DD rows to {rows_path}")
    print(f"Wrote {pos_path} and {npz_path}")

    print(f"Motion factors (phase-2 epochs): Doppler velocity "
          f"{getattr(rtk, '_doppler_fire_count', 0)} (low-sat "
          f"{getattr(rtk, '_doppler_lowsat_count', 0)}), IMU-only epochs "
          f"{getattr(rtk, 'imu_only_count', 0)}, NLOS-online inside "
          f"{getattr(rtk, 'nlos_online_inside', 0)}, NLOS PR rows "
          f"{getattr(rtk, 'nlos_meas_pr_affected', 0)}, NHC "
          f"{getattr(rtk, '_nhc_fire_count', 0)}, ZUPT "
          f"{getattr(rtk, '_zupt_fire_count', 0)}, phase-2 epochs "
          f"{getattr(rtk, 'epoch2', 0)}")
    # window summary (gate quick-look; official numbers via analyze script)
    e = np.asarray(cols["err3d"])
    sm = np.asarray(cols["smode"])
    ph = np.asarray(cols["ar_phase"])
    okm = np.isfinite(e)
    p2 = (ph == 2) & okm
    fx = p2 & (sm == 4)
    ef = e[fx]
    if p2.sum():
        print(f"WP16 SUMMARY p2={p2.sum()} fix={fx.sum()} "
              f"fix%={100*fx.sum()/max(p2.sum(),1):.1f} "
              f"false%(>0.5m)={100*float(np.mean(ef>0.5)) if ef.size else 0:.2f} "
              f"med_fix_err={100*float(np.median(ef)) if ef.size else float('nan'):.1f}cm "
              f"<50cm(all)={100*float(np.mean(e[okm]<0.5)):.1f}% "
              f"AllRMS={float(np.sqrt(np.mean(e[okm]**2))):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
