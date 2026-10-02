"""Command-line interface for ``exp_ppc_ctrbpf_fgo``.

``_build_arg_parser`` is the argument-parser construction moved verbatim out
of ``exp_ppc_ctrbpf_fgo.main`` (which re-exports it and the path defaults).
No behaviour change.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from ppc_ctrbpf_rtkdiag import _RTKDIAG_POLICIES

RESULTS_DIR = Path(__file__).resolve().parent / "results"
_DEFAULT_DATA_ROOT = Path("datasets/PPC-Dataset-data")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CT-RBPF-FGO PPC port (Phase 0 scaffolding)")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-prefix", type=str, default="ppc_ctrbpf_fgo")
    parser.add_argument(
        "--write-internal-diagnostics",
        action="store_true",
        help="Write per-epoch PF ESS/spread/stage-delta diagnostics to results",
    )
    parser.add_argument(
        "--pf-mode-policy",
        choices=("off", "diagnostic", "emit"),
        default="off",
        help=(
            "Weighted particle-mode policy: diagnostic records modes without "
            "trajectory changes; emit replaces accepted PF weighted means"
        ),
    )
    parser.add_argument("--pf-mode-voxel-size-m", type=float, default=2.0)
    parser.add_argument("--pf-mode-min-core-cell-mass", type=float, default=1.0e-4)
    parser.add_argument("--pf-mode-min-core-cell-particles", type=int, default=3)
    parser.add_argument("--pf-mode-min-mass", type=float, default=0.01)
    parser.add_argument("--pf-mode-assignment-radius-m", type=float, default=6.0)
    parser.add_argument("--pf-mode-max-modes", type=int, default=8)
    parser.add_argument("--pf-mode-max-particles", type=int, default=8192)
    parser.add_argument("--pf-mode-select-min-mass", type=float, default=0.20)
    parser.add_argument("--pf-mode-select-min-score-ratio", type=float, default=1.5)
    parser.add_argument(
        "--pf-mode-allow-single-mode",
        action="store_true",
        help="Permit mode emission for a single detected mode (default: multimodal only)",
    )
    parser.add_argument("--pf-mode-min-epoch", type=int, default=10)
    parser.add_argument("--pf-mode-prediction-sigma-m", type=float, default=5.0)
    parser.add_argument("--pf-mode-max-prediction-distance-m", type=float, default=20.0)
    parser.add_argument("--pf-mode-min-mean-distance-m", type=float, default=0.5)
    parser.add_argument("--pf-mode-max-mean-distance-m", type=float, default=20.0)
    parser.add_argument(
        "--enable-pf-ffbsi-smoother",
        action="store_true",
        help=(
            "Enable mode-conditioned systematic-ancestor fixed-lag smoothing; "
            "it smooths stored post-Doppler filtering particles without a second "
            "Doppler update (the separate offline backward replay reverses both "
            "satellite velocity and Doppler sign)"
        ),
    )
    parser.add_argument("--pf-ffbsi-lag-epochs", type=int, default=25)
    parser.add_argument("--pf-ffbsi-paths", type=int, default=32)
    parser.add_argument("--pf-ffbsi-seed", type=int, default=20260713)
    parser.add_argument(
        "--pf-ffbsi-mode",
        choices=("marginal", "genealogy"),
        default="marginal",
    )
    parser.add_argument("--pf-ffbsi-max-std-m", type=float, default=5.0)
    parser.add_argument("--pf-ffbsi-max-correction-m", type=float, default=10.0)
    parser.add_argument("--pf-ffbsi-min-unique-particles", type=int, default=2)
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=RESULTS_DIR / "libgnss_ctrbpf_pos",
    )
    parser.add_argument("--n-particles", type=int, default=50_000)
    parser.add_argument("--sigma-pr", type=float, default=8.0)
    parser.add_argument("--pr-ess-guard-min-ratio", type=float, default=0.0,
                        help="If >0, temper each PR log-likelihood increment so PF ESS ratio stays above this target")
    parser.add_argument("--pr-ess-guard-max-iters", type=int, default=12,
                        help="Binary-search iterations for --pr-ess-guard-min-ratio")
    parser.add_argument("--pr-gmm-statuses", type=str, default="1,3",
                        help="Hybrid Status values where PR-GMM likelihood is used (default '1,3'; empty means all)")
    parser.add_argument("--pr-gmm-w-los", type=float, default=0.7,
                        help="LOS mixture weight for PR-GMM likelihood (default 0.7)")
    parser.add_argument("--pr-gmm-mu-nlos-m", type=float, default=15.0,
                        help="Positive NLOS pseudorange bias mean for PR-GMM [m] (default 15)")
    parser.add_argument("--pr-gmm-sigma-nlos-m", type=float, default=30.0,
                        help="NLOS pseudorange sigma for PR-GMM [m] (default 30)")
    parser.add_argument("--pr-gmm-hybrid-loose-sigma-m", type=float, default=5.0,
                        help="Hybrid PU sigma on PR-GMM statuses; <=0 keeps global hybrid sigma (default 5)")
    parser.add_argument("--pr-gmm-clock-quantile", type=float, default=0.35,
                        help="Clock-bias correction residual quantile when PR-GMM is active (default 0.35)")
    parser.add_argument("--pr-weight-mode", choices=("raw", "unit", "cn0-relative"), default="raw",
                        help="PF pseudorange likelihood weight transform. raw preserves legacy C/N0-as-weight behavior")
    parser.add_argument("--pr-weight-ref-cn0", type=float, default=45.0,
                        help="Reference C/N0 for --pr-weight-mode cn0-relative (default 45)")
    parser.add_argument("--pr-weight-min", type=float, default=0.25,
                        help="Minimum transformed PR likelihood weight when clipping is active (default 0.25)")
    parser.add_argument("--pr-weight-max", type=float, default=1.5,
                        help="Maximum transformed PR likelihood weight when clipping is active (default 1.5)")
    parser.add_argument("--pr-systems", type=str, default="G,E,J",
                        help="Constellations used for undifferenced PR/WLS/PF updates; data load and DD can still include --systems/--dd-systems (default G,E,J)")
    parser.add_argument("--pr-min-elevation-deg", type=float, default=-90.0,
                        help="Drop PR satellites below this elevation angle from the current PF estimate; default off")
    parser.add_argument("--pr-auto-elevation-gate", action="store_true",
                        help="Auto-enable a high PR elevation gate when WLS postfit residual is poor and --pr-min-elevation-deg is off")
    parser.add_argument("--pr-auto-elevation-deg", type=float, default=30.0,
                        help="Elevation gate selected by --pr-auto-elevation-gate when triggered (default 30)")
    parser.add_argument("--pr-auto-elevation-postfit-rms-m", type=float, default=6.0,
                        help="Median centered WLS postfit RMS threshold for --pr-auto-elevation-gate (default 6m)")
    parser.add_argument("--pr-auto-elevation-min-sat-fraction", type=float, default=0.7,
                        help="Reject auto elevation gate if median satellites fall below this fraction of ungated WLS; <=0 disables")
    parser.add_argument("--pr-auto-elevation-max-pdop-ratio", type=float, default=2.0,
                        help="Reject auto elevation gate if median PDOP grows above this ratio; <=0 disables")
    parser.add_argument("--pr-auto-robust-profile", action="store_true",
                        help="Auto-select robust PR settings for high-WLS-residual or low-satellite segments")
    parser.add_argument("--pr-auto-robust-low-sat-threshold", type=int, default=8,
                        help="Satellite-count threshold for --pr-auto-robust-profile low-sat detection (default 8)")
    parser.add_argument("--pr-auto-robust-low-sat-epochs", type=int, default=3,
                        help="Minimum low-satellite epochs to trigger --pr-auto-robust-profile (default 3)")
    parser.add_argument("--pr-auto-robust-prefit-gate-m", type=float, default=20.0,
                        help="PR prefit gate selected by --pr-auto-robust-profile when global --pr-prefit-gate-m is off")
    parser.add_argument("--pr-auto-robust-sigma-pr", type=float, default=50.0,
                        help="PR sigma selected by --pr-auto-robust-profile when global --sigma-pr is the default")
    parser.add_argument("--pr-atmosphere-model", choices=("off", "fixed", "model"), default="off",
                        help="PR atmosphere correction: off, fixed zenith/sin(el), or Saastamoinen+Klobuchar model")
    parser.add_argument("--pr-atmosphere-scale", type=float, default=1.0,
                        help="Scale factor for the Saastamoinen+Klobuchar PR atmosphere model (default 1.0)")
    parser.add_argument("--pr-atmosphere-extra-zenith-m", type=float, default=0.0,
                        help="Extra zenith delay mapped by 1/sin(elevation) after the PR atmosphere model (default 0)")
    parser.add_argument("--pr-slant-delay-zenith-m", type=float, default=0.0,
                        help="Subtract this zenith delay divided by sin(elevation) from PR updates; default off")
    parser.add_argument("--pr-prefit-gate-m", type=float, default=0.0,
                        help="Drop satellites whose robust clock-centered PR prefit residual exceeds this [m]; <=0 disables")
    parser.add_argument("--pr-prefit-gate-min-sats", type=int, default=6,
                        help="Minimum satellites kept by the PR prefit gate (default 6)")
    parser.add_argument("--pr-prefit-gate-keep-best", type=int, default=0,
                        help="If >0, keep at most this many smallest-prefit satellites after gating")
    parser.add_argument("--pr-prefit-ref", choices=("pf", "hybrid"), default="pf",
                        help="Reference position for PR prefit residuals (default pf; hybrid uses libgnss++ position when available)")
    parser.add_argument("--pr-prefit-per-system", action="store_true",
                        help="Estimate PR prefit clock/ISB separately per constellation before residual gating")
    parser.add_argument("--pr-skip-statuses", type=str, default="",
                        help="Comma-separated hybrid Status values where undifferenced PR update is skipped")
    parser.add_argument("--defer-epoch-resample", action="store_true",
                        help="Accumulate PR/DD/Doppler/PU likelihoods within an epoch and resample only after emission")
    parser.add_argument("--enable-reservoir-stein", action="store_true",
                        help="Use epoch-end weighted reservoir + Stein rejuvenation instead of standard PF resampling")
    parser.add_argument("--reservoir-stein-size", type=int, default=2048,
                        help="Reservoir size for --enable-reservoir-stein; <=0 uses all particles")
    parser.add_argument("--reservoir-stein-elite-fraction", type=float, default=0.25,
                        help="Fraction of reservoir slots reserved for top-weight particles")
    parser.add_argument("--reservoir-stein-steps", type=int, default=1,
                        help="Number of Stein transport steps per reservoir resample")
    parser.add_argument("--reservoir-stein-step-size", type=float, default=0.05,
                        help="Stein transport step size")
    parser.add_argument("--reservoir-stein-repulsion-scale", type=float, default=1.0,
                        help="Scale for the SVGD repulsion term")
    parser.add_argument("--reservoir-stein-guide-sigma-m", type=float, default=2.0,
                        help="Position sigma for the weighted-mean guide gradient [m]")
    parser.add_argument("--reservoir-stein-guide-sigma-cb-m", type=float, default=50.0,
                        help="Clock-bias sigma for the weighted-mean guide gradient [m]")
    parser.add_argument("--reservoir-stein-seed", type=int, default=20260512,
                        help="Base RNG seed for reservoir selection and expansion")
    parser.add_argument("--sigma-pos", type=float, default=2.0)
    parser.add_argument("--sigma-cb", type=float, default=50.0)
    parser.add_argument("--spread-pos-init", type=float, default=50.0)
    parser.add_argument("--spread-cb-init", type=float, default=500.0)
    parser.add_argument("--sigma-doppler-mps", type=float, default=0.5)
    parser.add_argument("--doppler-systems", type=str, default="G,E,J",
                        help="Constellations used for Doppler-KF updates; default excludes C until BDS Doppler/clock handling is validated")
    parser.add_argument("--doppler-prefit-gate-mps", type=float, default=0.0,
                        help="Drop Doppler rows whose centered WLS residual exceeds this [m/s]; <=0 disables")
    parser.add_argument("--doppler-prefit-gate-min-sats", type=int, default=6,
                        help="Minimum Doppler rows kept by --doppler-prefit-gate-mps")
    parser.add_argument("--velocity-init-sigma", type=float, default=1.0)
    parser.add_argument("--velocity-process-noise", type=float, default=1.0)
    parser.add_argument("--position-update-sigma-m", type=float, default=30.0)
    parser.add_argument("--position-update-min-epoch", type=int, default=0,
                        help="Skip WLS position_update before this local epoch index; <=0 disables")
    parser.add_argument("--position-update-min-pr-sats", type=int, default=0,
                        help="Skip WLS position_update when the epoch uses fewer PR satellites than this; <=0 disables")
    parser.add_argument("--position-update-max-wls-rms-m", type=float, default=0.0,
                        help="Skip WLS position_update when centered WLS postfit RMS exceeds this [m]; <=0 disables")
    parser.add_argument("--position-update-max-wls-pdop", type=float, default=0.0,
                        help="Skip WLS position_update when WLS PDOP exceeds this; <=0 disables")
    parser.add_argument("--position-update-max-wls-to-pf-m", type=float, default=0.0,
                        help="Skip WLS position_update when |WLS-PF| before PU exceeds this [m]; <=0 disables")
    parser.add_argument("--disable-correct-clock-bias", action="store_true")
    parser.add_argument("--systems", type=str, default="G,R,E,C,J")
    parser.add_argument(
        "--methods",
        type=str,
        default="pf,pf+pu,rbpf,rbpf+pu",
        help=(
            "Comma-separated subset of {pf, pf+pu, rbpf, rbpf+pu, pf+dd, "
            "rbpf+dd, rbpf+dd+pu, rbpf+dd+gate, rbpf+dd+ar+gate, rbpf+dd+cp+gate, rbpf+dd+gate+pu, "
            "rbpf+dd+pu+tdcp, rbpf+dd+gate+pu+tdcp, "
            "rbpf+dd+pu+bridge, rbpf+dd+gate+pu+bridge, "
            "rbpf+dd+dopq+pu, rbpf+dd+dopq+pu+bridge, "
            "rbpf+dd+gate+pu+ddpr, rbpf+dd+gate+pu+bridge+ddpr, "
            "rbpf+dd+gate+rtkdiag_pf, rbpf+dd+gate+rtkdiag_pf+bridge, "
            "pf+hybrid, rbpf+dd+hybrid, rbpf+dd+gate+hybrid, "
            "rbpf+dd+gate+hybrid+gmm, "
            "rbpf+dd+gate+hybrid+rtkdiag_pf, "
            "rbpf+dd+gate+phase7, rbpf+dd+gate+phase7+gmm, "
            "rbpf+dd+gate+phase4, "
            "rbpf+dd+gate+hybrid+phase4, "
            "rbpf+dd+gate+hybrid+tdcp, rbpf+dd+gate+hybrid+tdcp+phase4, "
            "rbpf+dd+gate+hybrid+zupt, rbpf+dd+gate+hybrid+zupt+tdcp, "
            "rbpf+dd+gate+hybrid+imu_tc, rbpf+dd+gate+hybrid+zupt+imu_tc, "
            "rbpf+dd+gate+hybrid+ins_tc, rbpf+dd+gate+hybrid+gmm+ins_tc, "
            "rbpf+dd+gate+hybrid+zupt+ins_tc, "
            "rbpf+dd+gate+hybrid+rtkdiag_pf+imu_tc, "
            "rbpf+dd+gate+hybrid+rtkdiag_pf+ins_tc, "
            "embedded, embedded_ctpf_fgo, embedded_pfemit, "
            "gpu_ctpf_fgo, gpu_ctpf_fgo_pfemit}"
        ),
    )
    parser.add_argument("--dd-sigma-cycles", type=float, default=0.05,
                        help="DD carrier AFV sigma in cycles (default 0.05)")
    parser.add_argument("--dd-min-pairs", type=int, default=4,
                        help="Min common rover/base sats to attempt DD (default 4)")
    parser.add_argument("--dd-min-pairs-update", type=int, default=3,
                        help="Min DD pairs required to apply pf.update_dd_carrier_afv (default 3)")
    parser.add_argument(
        "--cp-mupf-dd-pr-sigma-m", type=float, default=5.0, dest="cp_mupf_dd_pr_sigma_m",
        help=(
            "WP23a MUPF stage (i): DD-pseudorange sigma [m] (default 5.0, "
            "inherited from --fgo-dd-pr-sigma-m's tc_fgo/rbpf_fgo default)."
        ),
    )
    parser.add_argument(
        "--cp-mupf-pr-stage-target-ess-ratio", type=float, default=0.10,
        dest="cp_mupf_pr_stage_target_ess_ratio",
        help="WP23a MUPF stage (i) target post-tempering ESS/N (default 0.10).",
    )
    parser.add_argument(
        "--cp-mupf-dd-cp-sigma-sequence-cycles", type=str, default="2.0,0.5,0.05",
        dest="cp_mupf_dd_cp_sigma_sequence_cycles",
        help=(
            "WP23a MUPF stage (ii): comma-separated coarse-to-fine DD-carrier "
            "AFV sigma sequence [cycles] (default 2.0,0.5,0.05 -- reused from "
            "the validated gnss_gpu.pf_smoother_config.MupfConfig default)."
        ),
    )
    parser.add_argument(
        "--cp-mupf-cp-stage-target-ess-ratio", type=float, default=0.10,
        dest="cp_mupf_cp_stage_target_ess_ratio",
        help="WP23a MUPF stage (ii) target post-tempering ESS/N per sigma step (default 0.10).",
    )
    parser.add_argument(
        "--cp-mupf-stage-max-iters", type=int, default=20, dest="cp_mupf_stage_max_iters",
        help="MUPF per-stage annealing bisection iterations (default 20).",
    )
    parser.add_argument(
        "--cp-mupf-stage-max-tempering-steps", type=int, default=64,
        dest="cp_mupf_stage_max_tempering_steps",
        help=(
            "WP23b maximum annealed-SMC beta increments per staged likelihood "
            "(default 64; exhausting the limit is a hard error)."
        ),
    )
    parser.add_argument(
        "--cp-mupf-min-pairs", type=int, default=3, dest="cp_mupf_min_pairs",
        help="WP23a MUPF: min DD pairs required to attempt a given stage's update (default 3).",
    )
    parser.add_argument(
        "--cp-mupf-cp-n-groups", type=int, default=1, dest="cp_mupf_cp_n_groups",
        help=(
            "WP23a MUPF stage (ii): split DD-CP pairs round-robin into this "
            "many sequential sub-updates per sigma step (default 1 = off; "
            "the spec's documented fallback if a single CP update is still "
            "too sharp)."
        ),
    )
    parser.add_argument(
        "--disable-cp-mupf-slip-gate", action="store_true",
        help="WP23a: disable the epoch-to-epoch DD-carrier phase continuity (cycle-slip proxy) gate.",
    )
    parser.add_argument(
        "--cp-mupf-slip-max-delta-cycles", type=float, default=2.0,
        dest="cp_mupf_slip_max_delta_cycles",
        help="WP23a cycle-slip proxy: max tolerated |delta DD-carrier cycles| between consecutive epochs (default 2.0).",
    )
    parser.add_argument(
        "--cp-mupf-slip-max-dt-s", type=float, default=2.0, dest="cp_mupf_slip_max_dt_s",
        help="WP23a cycle-slip proxy: max epoch gap [s] over which continuity is checked (default 2.0).",
    )
    parser.add_argument(
        "--cp-mupf-resample-before-stage", action="store_true",
        help=(
            "WP23a diagnostic: resample-if-needed before, not just after, "
            "each MUPF stage's update (root-cause of the DD-PR stage's "
            "measured alpha=0.0 inertness -- see WP23A_REPORT.md)."
        ),
    )
    parser.add_argument("--dd-systems", type=str, default="G,E,J,C",
                        help="Constellations used for DD (GLONASS skipped, default G,E,J,C)")
    parser.add_argument("--dd-base-interp", action="store_true",
                        help="Interpolate base RINEX between epochs when rover TOW falls between two")
    parser.add_argument("--dd-min-elevation-deg", type=float, default=-90.0,
                        help="Drop DD satellites below this elevation angle in degrees (default off)")
    parser.add_argument("--dd-min-snr", type=float, default=0.0,
                        help="Drop DD satellites with SNR/CN0 below this value (default off)")
    parser.add_argument("--dd-keep-best", type=int, default=0,
                        help="Keep only the best N DD satellites by elevation then SNR per epoch; <=0 disables")
    parser.add_argument("--dd-pr-pair-residual-max-m", type=float, default=0.0,
                        help="Gate FGO DD-pseudorange pairs by residual at DD selection position; <=0 disables")
    parser.add_argument("--dd-pr-epoch-median-residual-max-m", type=float, default=0.0,
                        help="Drop an FGO DD-pseudorange epoch if kept-pair median residual exceeds this; <=0 disables")
    parser.add_argument("--dd-pr-gate-min-pairs", type=int, default=3,
                        help="Minimum DD-pseudorange pairs after residual gating (default 3)")
    parser.add_argument("--enable-dd-pr-ls-anchor", action="store_true",
                        help="Solve a gated DD-pseudorange LS anchor and use it as FGO initial/prior on selected statuses")
    parser.add_argument("--dd-pr-ls-anchor-min-pairs", type=int, default=3,
                        help="Minimum DD-pseudorange pairs for LS anchor (default 3)")
    parser.add_argument("--dd-pr-ls-anchor-dd-sigma-m", type=float, default=2.0,
                        help="DD-pseudorange sigma for LS anchor solve [m] (default 2)")
    parser.add_argument("--dd-pr-ls-anchor-solve-prior-sigma-m", type=float, default=100.0,
                        help="Seed prior sigma inside DD-PR LS anchor solve [m] (default 100)")
    parser.add_argument("--dd-pr-ls-anchor-prior-sigma-m", type=float, default=3.0,
                        help="FGO prior sigma assigned to accepted DD-PR LS anchors [m] (default 3)")
    parser.add_argument("--dd-pr-ls-anchor-max-shift-m", type=float, default=100.0,
                        help="Reject DD-PR LS anchors that move farther than this from the seed [m] (default 100)")
    parser.add_argument("--dd-pr-ls-anchor-max-postfit-rms-m", type=float, default=5.0,
                        help="Reject DD-PR LS anchors with final DD-PR RMS above this [m] (default 5)")
    parser.add_argument("--dd-pr-ls-anchor-statuses", type=str, default="1,3",
                        help="Hybrid statuses where DD-PR LS anchors may be used; empty means all (default 1,3)")
    parser.add_argument("--dd-pr-ls-anchor-no-initial", action="store_true",
                        help="Use accepted DD-PR LS anchors as FGO priors only, not initial positions")
    parser.add_argument("--dd-pr-ls-anchor-mode",
                        choices=("prior", "initial", "diagnostic", "pf", "pu", "pf-pu", "prior+pf", "initial+pf"),
                        default="prior",
                        help=("How accepted DD-PR LS anchors enter FGO: prior=initial plus "
                              "position prior, initial=initial value only, diagnostic=stats only; "
                              "pf/pu modes apply a realtime PF position_update"))
    # Phase 2: region-aware gate on RBPF velocity-KF (Doppler) update.
    # Mirrors the AAA gate knobs from internal_docs/plan.md §10.1.
    parser.add_argument("--rbpf-velocity-kf-gate-min-dd-pairs", type=int, default=None,
                        help="Skip Doppler KF update unless DD pair count >= N (default off)")
    parser.add_argument("--rbpf-velocity-kf-gate-min-ess-ratio", type=float, default=None,
                        help="Skip Doppler KF update unless ESS / n_particles >= ratio (default off)")
    parser.add_argument("--rbpf-velocity-kf-gate-max-spread-m", type=float, default=None,
                        help="Skip Doppler KF update if PF position spread exceeds this [m] (default off)")
    parser.add_argument("--rbpf-velocity-kf-gate-max-doppler-wls-rms-mps", type=float, default=0.0,
                        help="Skip Doppler KF update when centered Doppler-WLS RMS exceeds this [m/s]; <=0 disables")
    parser.add_argument("--rbpf-velocity-kf-gate-max-doppler-wls-speed-mps", type=float, default=0.0,
                        help="Skip Doppler KF update when centered Doppler-WLS speed exceeds this [m/s]; <=0 disables")
    # Phase 6: libgnss++ hybrid position update (uses .pos files from
    # experiments/results/libgnss_rtk_pos_v5/ — the 50.91% baseline).
    parser.add_argument("--hybrid-pos-dir", type=Path, default=None,
                        help="Directory of libgnss++ .pos files used as hybrid PU baseline "
                             "(expects {city}_{run}_full.pos)")
    parser.add_argument("--hybrid-pos-suffix", type=str, default="_full.pos",
                        help="Suffix used to find pos files (default _full.pos)")
    parser.add_argument("--hybrid-sigma-m", type=float, default=1.0,
                        help="Sigma [m] for the hybrid position_update soft constraint (default 1.0)")
    parser.add_argument("--hybrid-recenter-max-shift-m", type=float, default=0.0,
                        help="Recenter PF cloud to the single hybrid anchor before hybrid PU when shift <= this [m]; 0 disables")
    parser.add_argument("--hybrid-emit-pf-statuses", type=str, default="",
                        help="Comma-separated hybrid Status values where PF may be emitted when hybrid_emit_pf_estimate is enabled; empty means all")
    parser.add_argument("--hybrid-vguide-max-dt-s", type=float, default=0.5,
                        help="Max gap [s] between consecutive hybrid samples for finite-diff velocity (default 0.5)")
    # Phase 10i: PF rescue using RTK diagnostics from gnss_solve
    parser.add_argument("--rtkdiag-candidate-pos-dir", type=Path, default=None,
                        help="Directory of relaxed RTK candidate .pos files for rtkdiag_pf "
                             "(expects {city}_{run}_full.pos)")
    parser.add_argument("--rtkdiag-candidate-diag-dir", type=Path, default=None,
                        help="Directory of relaxed RTK diagnostics CSV files for rtkdiag_pf "
                             "(expects {city}_{run}_full.csv)")
    parser.add_argument("--rtkdiag-candidate-pos-dirs", type=str, default="",
                        help="Comma-separated candidate .pos directories for multi-candidate rtkdiag_pf")
    parser.add_argument("--rtkdiag-candidate-diag-dirs", type=str, default="",
                        help="Comma-separated diagnostics CSV directories for multi-candidate rtkdiag_pf")
    parser.add_argument("--rtkdiag-candidate-labels", type=str, default="",
                        help="Optional comma-separated labels for multi-candidate rtkdiag_pf")
    parser.add_argument("--rtkdiag-candidate-block-labels", type=str, default="",
                        help="Comma-separated candidate labels to block for all runs")
    parser.add_argument("--rtkdiag-candidate-block-labels-by-run", type=str, default="",
                        help=(
                            "Semicolon-separated run-specific candidate label blocks, "
                            "e.g. 'nagoya/run2=r20g15+r15g15'"
                        ))
    parser.add_argument("--rtkdiag-candidate-pos-suffix", type=str, default="_full.pos",
                        help="Suffix used for RTK diagnostic candidate pos files (default _full.pos)")
    parser.add_argument("--rtkdiag-candidate-diag-suffix", type=str, default="_full.csv",
                        help="Suffix used for RTK diagnostic CSV files (default _full.csv)")
    parser.add_argument("--rtkdiag-candidate-sigma-m", type=float, default=0.02,
                        help="PF position_update sigma for diagnostics-passing RTK candidate [m] (default 0.02)")
    parser.add_argument("--rtkdiag-candidate-ratio-min", type=float, default=1.5,
                        help="Minimum candidate final_ratio for rtkdiag_pf gate (default 1.5)")
    parser.add_argument("--rtkdiag-candidate-residual-rms-max", type=float, default=1.8,
                        help="Maximum candidate final_residual_rms for rtkdiag_pf gate (default 1.8)")
    parser.add_argument("--rtkdiag-candidate-main-status5-residual-rms-max", type=float, default=0.3,
                        help="Accept status=5 (Float) candidates in main gate when residual_rms<=this [m] (default 0.3); 0 disables status=5")
    parser.add_argument("--rtkdiag-candidate-rms-prefilter-k", type=int, default=0,
                        help="Pre-filter gated candidates to top-K by residual_rms before selector ranking (0=disable). Sim showed K=3-7 captures +12pp upper bound; K>=20 degrades.")
    parser.add_argument("--rtkdiag-candidate-cluster-vote-radius-m", type=float, default=0.5,
                        help="cluster_vote select_mode: spatial cluster radius (m). Default 0.5 matches 50cm PPC threshold.")
    parser.add_argument("--rtkdiag-candidate-ranker-score-path", type=str, default="",
                        help="ranker select_mode: path to predictions CSV (run_id,tow,label,p_pass) from train_selector_ranker.py")
    parser.add_argument("--rtkdiag-candidate-ranker-stickiness", type=float, default=0.0,
                        help="ranker select_mode: stickiness s in [0,1]. If previous epoch's picked label has p_pass>=s*max_p_pass at current epoch, keep it (sequence smoothing). 0=disabled (max p_pass per epoch independently).")
    parser.add_argument("--spp-nlos-mask-path", type=str, default="",
                        help="SPP G: PLATEAU NLOS mask CSV (tow,epoch_idx,prn,is_los); is_los=0 → soft-downweight. Format: literal path or template with {city}/{run} (e.g. /tmp/{city}_{run}_per_epoch_nlos.csv).")
    parser.add_argument("--spp-nlos-k-weak", type=float, default=1.0,
                        help="SPP G: weak NLOS down-weight factor (weight /= k). 1.0 = off, 3-5 typical.")
    parser.add_argument("--spp-nlos-k-strong", type=float, default=1.0,
                        help="SPP G: strong NLOS down-weight factor for confirmed NLOS (1.0 = same as weak). Requires --spp-nlos-strong-mask-path.")
    parser.add_argument("--spp-nlos-strong-mask-path", type=str, default="",
                        help="SPP G: optional separate CSV listing strong-NLOS PRNs (same format as --spp-nlos-mask-path).")
    parser.add_argument("--pf-nlos-mask-path", type=str, default="",
                        help="PF: PLATEAU NLOS mask CSV (tow,epoch_idx,prn,is_los); is_los=0 -> soft-downweight at PF update. Supports {city}/{run} template.")
    parser.add_argument("--pf-nlos-k-weak", type=float, default=3.0,
                        help="PF: weak NLOS down-weight factor (weight /= k). Default off when --pf-nlos-mask-path is unset.")
    parser.add_argument("--pf-nlos-k-strong", type=float, default=3.0,
                        help="PF: strong NLOS down-weight factor for confirmed NLOS. Requires --pf-nlos-strong-mask-path.")
    parser.add_argument("--pf-nlos-strong-mask-path", type=str, default="",
                        help="PF: optional separate CSV listing strong-NLOS PRNs (same format as --pf-nlos-mask-path).")
    parser.add_argument(
        "--pf-nlos-preset",
        choices=("off", "soft-k3"),
        default="off",
        help="Bundle PF NLOS mask path template (plateau_nlos_phase33) and k_weak/k_strong=3.",
    )
    parser.add_argument("--spp-irls", type=str, default="off",
                        choices=("off", "cauchy", "huber"),
                        help="SPP B: post-WLS IRLS refinement weight function. off=disabled.")
    parser.add_argument("--spp-irls-c", type=float, default=15.0,
                        help="SPP B: IRLS scale parameter [m] (Cauchy c, Huber threshold).")
    parser.add_argument("--spp-irls-shift-cap-m", type=float, default=50.0,
                        help="SPP B: max allowed |IRLS shift| from raw WLS pos (m). Beyond → discard refinement.")
    parser.add_argument("--rtkdiag-candidate-bridge-enable", action="store_true",
                        help="Add synthetic pf_bridge candidate (last good anchor + PF velocity * Δt) when Δt<=bridge_max_s")
    parser.add_argument("--rtkdiag-candidate-bridge-max-s", type=float, default=6.0,
                        help="Max Δt [s] for pf_bridge synthetic candidate (default 6.0, 2位 nagoya2 setting)")
    parser.add_argument("--rtkdiag-candidate-bridge-residual-rms-m", type=float, default=0.5,
                        help="Calibrated residual_rms [m] for pf_bridge candidate (default 0.5)")
    parser.add_argument("--rtkdiag-candidate-bridge-anchor-mode", type=str, default="last_emit",
                        choices=("last_emit", "last_fix4"),
                        help="Bridge anchor source: last_emit (any emit, inherits cluster bias) or last_fix4 (trusted Fix only)")
    parser.add_argument("--rtkdiag-candidate-bridge-fix4-min-ratio", type=float, default=3.0,
                        help="Min ratio to qualify as trusted Fix=4 anchor (default 3.0)")
    parser.add_argument("--rtkdiag-candidate-bridge-fix4-max-residual", type=float, default=0.1,
                        help="Max residual_rms to qualify as trusted Fix=4 anchor (default 0.1)")
    parser.add_argument("--rtkdiag-candidate-max-to-hybrid-m", type=float, default=1.0,
                        help="Skip RTK diagnostic candidate when |candidate-hybrid| exceeds this [m] (default 1.0); <=0 disables")
    parser.add_argument("--rtkdiag-candidate-emit-max-diff-m", type=float, default=0.4,
                        help="Emit PF only if |PF - candidate| <= this [m] after candidate PU (default 0.4)")
    parser.add_argument("--rtkdiag-candidate-recenter-max-shift-m", type=float, default=10000.0,
                        help="Recenter PF cloud to candidate before candidate PU when shift <= this [m] (default 10000)")
    parser.add_argument("--rtkdiag-candidate-soft-top-k", type=int, default=1,
                        help="Use top-K gated RTK candidates as a Gaussian-mixture PF update; 1 keeps single-candidate PU")
    parser.add_argument("--rtkdiag-candidate-soft-weight-eps", type=float, default=0.01,
                        help="Epsilon for inverse-sort-key weights in --rtkdiag-candidate-soft-top-k mixture")
    parser.add_argument("--rtkdiag-candidate-proposal-cloud", action="store_true",
                        help="Replace particles with a candidate-centered proposal cloud instead of applying candidate log-likelihood weights")
    parser.add_argument("--rtkdiag-candidate-proposal-spread-m", type=float, default=0.25,
                        help="Position spread [m] for --rtkdiag-candidate-proposal-cloud")
    parser.add_argument("--rtkdiag-candidate-select-mode",
                        choices=("residual", "ratio", "score", "maxabs", "nrows", "hybrid_anchor", "wavg3", "wavg5", "consensus3", "consensus5", "cluster_vote", "ranker", "ranker_gici_cluster_override",
                                 "rms_per_row", "score_per_row", "score_per_row2", "score_per_row3", "rms_minus_alpha_rows", "log_combined",
                                 "composite_3axis_n2", "composite_3axis_t2", "composite_3axis_n1",
                                 "composite_n2_v2", "composite_n3_v2", "composite_n1_v2", "composite_t2_v2",
                                 "composite_t3_v2", "composite_t3_v4", "composite_t2_v3",
                                 "composite_n1_v3", "composite_n2_v3", "composite_n2_v4", "composite_n3_v3", "composite_n3_v4",
                                 "temporal_n2_v1", "temporal_n2_v2", "temporal_n2_v3", "temporal_n2_v4", "temporal_n2_v5", "temporal_n2_v6", "temporal_n2_v7", "temporal_n2_v8", "temporal_n2_v9", "temporal_n2_v10",
                                 "temporal_n2_v11_a001", "temporal_n2_v12_a01", "temporal_n2_v13_a1", "temporal_n2_v14_a3", "temporal_n2_v15_a10",
                                 "temporal_hybdelta_t3_v1", "temporal_hybdelta_t3_v2", "temporal_hybdelta_t3_v3", "temporal_hybdelta_t3_v4", "temporal_hybdelta_t3_v5", "temporal_hybdelta_t3_v6", "temporal_hybdelta_t3_v7", "temporal_hybdelta_t3_v8",
                                 "temporal_hybdelta_n2_v1", "temporal_hybdelta_n3_v1", "temporal_hybdelta_n3_v2", "temporal_hybdelta_n3_v3", "temporal_hybdelta_n3_v4", "temporal_hybdelta_n3_v5", "temporal_hybdelta_n3_v6"),
                        default="residual",
                        help="How to choose among multiple gated RTK candidates (default residual). "
                             "hybrid_anchor picks the candidate closest to the hybrid floor; "
                             "useful when hybrid is reliable. ranker_gici_cluster_override applies "
                             "the supervised ranker, then re-picks within the xd_gici family toward a "
                             "tight low-RMS cluster when the pick is a high-risk GICI variant "
                             "(plan.md Phase 43/71)")
    parser.add_argument("--rtkdiag-candidate-emit-mode",
                        choices=("pf", "candidate-on-drift", "candidate"),
                        default="pf",
                        help=(
                            "Output policy for diagnostics-passing RTK candidate epochs: "
                            "pf=emit PF only when close to candidate and otherwise keep hybrid; "
                            "candidate-on-drift=emit PF when close, otherwise emit candidate; "
                            "candidate=always emit selected candidate (default pf)"
                        ))
    parser.add_argument("--rtkdiag-candidate-min-epoch", type=int, default=0,
                        help="Skip RTK diagnostic candidate PU/emission before this local epoch index")
    parser.add_argument("--rtkdiag-candidate-require-any-diag-fields", type=str, default="",
                        help="Comma-separated candidate diagnostic boolean fields; at least one must be true")
    parser.add_argument("--rtkdiag-candidate-require-all-diag-fields", type=str, default="",
                        help="Comma-separated candidate diagnostic boolean fields; all must be true")
    parser.add_argument("--rtkdiag-candidate-min-diag-fields", type=str, default="",
                        help="Comma-separated candidate diagnostic lower bounds, e.g. dd_pr_kept=5")
    parser.add_argument("--rtkdiag-candidate-max-diag-fields", type=str, default="",
                        help="Comma-separated candidate diagnostic upper bounds, e.g. dd_pr_shift_m=50")
    parser.add_argument("--rtkdiag-candidate-fallback-mode",
                        choices=("pf", "hybrid", "hybrid-last-good", "wls", "quality-wls", "last-good", "wls-last-good", "quality-wls-last-good"),
                        default="hybrid",
                        help="Fallback when rtkdiag candidate is unavailable/rejected (default hybrid; matches canonical 73.76%% behavior)")
    parser.add_argument("--rtkdiag-candidate-fallback-max-wls-rms-m", type=float, default=0.0,
                        help="Maximum WLS postfit RMS for quality WLS rtkdiag fallback; <=0 disables")
    parser.add_argument("--rtkdiag-candidate-fallback-max-wls-pdop", type=float, default=0.0,
                        help="Maximum WLS PDOP for quality WLS rtkdiag fallback; <=0 disables")
    parser.add_argument("--rtkdiag-candidate-fallback-max-wls-to-pf-m", type=float, default=0.0,
                        help="Maximum |WLS-PF| for quality WLS rtkdiag fallback; <=0 disables")
    parser.add_argument("--rtkdiag-candidate-fallback-max-hold-age-s", type=float, default=5.0,
                        help="Maximum age for last-good rtkdiag fallback; <=0 disables age check")
    parser.add_argument("--rtkdiag-candidate-force-emit-mode",
                        choices=("pf", "candidate-on-drift", "candidate"),
                        default="",
                        help="Override any run-index policy's emit-mode rewrite; useful for PF-emission experiments")
    parser.add_argument(
        "--rtkdiag-candidate-label-factors",
        type=str,
        default="",
        help=(
            "Comma-separated label=factor sort-key multipliers applied after "
            "the built-in selector label penalties. Values below 1 prefer a "
            "candidate label; values above 1 penalize it."
        ),
    )
    parser.add_argument("--rtkdiag-candidate-run-index-policy",
                        choices=tuple(["none"] + sorted(_RTKDIAG_POLICIES)),
                        default="none",
                        help=(
                            "Experimental per-run-index RTKDiag policy. phase10o uses "
                            "run1=nrows/rms1.4, run2=maxabs/rms5.0, run3=score/rms5.0, "
                            "all with candidate emit. phase10p uses the same family split "
                            "but run2/run3 rms5.0 -> rms6.0 for the r30 candidate set "
                            "phase10r uses run1=nrows/rms1.4, run2=score/ratio1.7/rms6.0, "
                            "run3=score/rms7.0 for the r15 no-hold candidate set. "
                            "phase11h follows phase10r but filters gate15 candidates on "
                            "nagoya/run2. phase11i follows phase11h but also filters "
                            "r30/r30g on tokyo/run2, nagoya/run1, and nagoya/run2. "
                            "phase11l follows phase11i but filters r20g10 on "
                            "nagoya/run1 and nagoya/run2. "
                            "phase11n follows phase11l but additionally filters "
                            "r15g10 and r25g10 on all nagoya runs. "
                            "phase11x extends phase11n with city x run gate values "
                            "from the offline gate sweep on Phase 11v candidates "
                            "(tokyo/run1 ratio2.5; tokyo/run2 rms10; tokyo/run3 ratio1.3 rms10; "
                            "nagoya/run1 rms1.0; nagoya/run2 ratio1.5 rms7; nagoya/run3 ratio1.7 rms10). "
                            "phase11y extends phase11x by widening rms_max to 20 on the "
                            "heavy-NLOS runs (tokyo/run3, nagoya/run2, nagoya/run3 with ratio_min "
                            "dropped to 1.0 on tokyo/run3 and nagoya/run2) "
                            "(default none)"
                        ))
    # Phase 4: post-process FGO + LAMBDA partial fix
    parser.add_argument("--fgo-window-size", type=int, default=30,
                        help="FGO window size in epochs (default 30)")
    parser.add_argument("--fgo-window-stride", type=int, default=15,
                        help="FGO window stride in epochs (default 15, 50%% overlap)")
    parser.add_argument("--fgo-lambda-ratio", type=float, default=3.0,
                        help="LAMBDA ratio test threshold (default 3.0)")
    parser.add_argument("--fgo-lambda-min-epochs", type=int, default=10,
                        help="Min epochs with DD obs in window to run LAMBDA (default 10)")
    parser.add_argument("--fgo-lambda-max-epoch-gap", type=int, default=6,
                        help="Max epoch-index gap still considered one ambiguity track (default 6 for PPC 5Hz rover / 1Hz DD)")
    parser.add_argument("--fgo-min-fixed-to-apply", type=int, default=3,
                        help="Min ratio-passed integer fixes per window to apply FGO output (default 3)")
    parser.add_argument("--fgo-prior-sigma-m", type=float, default=0.5,
                        help="FGO prior sigma [m] for initial positions (default 0.5)")
    parser.add_argument("--fgo-dd-sigma-cycles", type=float, default=0.20,
                        help="FGO DD carrier float sigma [cycles] (default 0.20)")
    parser.add_argument("--fgo-dd-pr-sigma-m", type=float, default=5.0,
                        help="FGO DD pseudorange sigma [m] for absolute-position factors (default 5.0)")
    parser.add_argument("--fgo-apply-hybrid-statuses", type=str, default="1,3",
                        help="Comma-separated hybrid Status values where Phase 4 may overwrite "
                             "the hybrid passthrough; default '1,3' protects Status=4 cm-class "
                             "epochs. Use empty string '' to disable the gate (apply everywhere).")
    parser.add_argument("--fgo-anchor-sigma-m", type=float, default=0.05,
                        help="Per-epoch FGO prior sigma [m] applied to non-rewritten Status epochs "
                             "(D2: tight anchor, default 0.05). Set <=0 to fall back to legacy "
                             "endpoint-only priors at --fgo-prior-sigma-m.")
    parser.add_argument("--fgo-loose-sigma-m", type=float, default=5.0,
                        help="Per-epoch FGO prior sigma [m] applied to rewritten Status epochs "
                             "(D2: loose, lets DD drive the solve, default 5.0).")
    parser.add_argument("--enable-ct-spline-motion-prior", action="store_true",
                        help="Fit a cubic continuous-time spline and use its motion as the FGO between-factor prior")
    parser.add_argument("--ct-spline-smoothing-m", type=float, default=0.5,
                        help="Approximate position smoothing scale [m] for --enable-ct-spline-motion-prior")
    parser.add_argument("--ct-motion-sigma-m", type=float, default=0.25,
                        help="FGO between-factor sigma [m] for CT spline motion prior")
    parser.add_argument("--ct-motion-min-epochs", type=int, default=6,
                        help="Minimum valid epochs required to fit the CT spline motion prior")
    parser.add_argument("--fgo-min-correction-m", type=float, default=0.5,
                        help="Minimum |FGO - hybrid| disagreement [m] required to overwrite the "
                             "hybrid passthrough at a given epoch (D2b: filters out small noisy "
                             "rewrites that cost cm-class passes; default 0.5, set 0 to disable).")
    parser.add_argument("--fgo-apply-window-all", action="store_true",
                        help="Apply every unprotected epoch in a fixed FGO window instead of only epochs with an accepted integer fix")
    # Phase 8: TDCP-anchored hybrid smoother
    parser.add_argument("--tdcp-sigma-mps", type=float, default=0.05,
                        help="TDCP velocity sigma [m/s] used as the smoother process noise (default 0.05)")
    parser.add_argument("--tdcp-postfit-max-m", type=float, default=1.0,
                        help="Reject TDCP velocity if postfit RMS exceeds this [m] (default 1.0)")
    parser.add_argument("--tdcp-min-sats", type=int, default=5,
                        help="Min satellites tracked through both epochs for TDCP (default 5)")
    parser.add_argument("--tdcp-obs-anchor-sigma-m", type=float, default=0.05,
                        help="Smoother observation sigma [m] for protected hybrid epochs "
                             "(Status NOT in --fgo-apply-hybrid-statuses; default 0.05)")
    parser.add_argument("--tdcp-obs-loose-sigma-m", type=float, default=5.0,
                        help="Smoother observation sigma [m] for rewritable hybrid epochs (default 5.0)")
    parser.add_argument("--tdcp-obs-missing-sigma-m", type=float, default=1000.0,
                        help="Smoother observation sigma [m] when hybrid is missing entirely (default 1000)")
    parser.add_argument("--low-sat-bridge-min-pr-sats", type=int, default=11,
                        help="PR satellite count below which low-sat bridge treats an epoch as weak (default 11)")
    parser.add_argument("--low-sat-bridge-min-span-epochs", type=int, default=3,
                        help="Minimum consecutive weak-PR epochs to bridge (default 3)")
    parser.add_argument("--low-sat-bridge-max-span-epochs", type=int, default=25,
                        help="Maximum consecutive weak-PR epochs to bridge (default 25)")
    parser.add_argument("--low-sat-bridge-max-gap-s", type=float, default=10.0,
                        help="Maximum anchor-to-anchor time gap for low-sat bridging [s] (default 10)")
    parser.add_argument("--low-sat-bridge-startup-max-wls-pdop", type=float, default=0.5,
                        help="Back-extrapolate initial epochs while WLS PDOP exceeds this; <=0 disables startup bridge")
    parser.add_argument("--low-sat-bridge-startup-min-epochs", type=int, default=2,
                        help="Minimum weak-WLS startup epochs required to apply startup bridge (default 2)")
    parser.add_argument("--low-sat-bridge-startup-max-epochs", type=int, default=10,
                        help="Maximum initial weak-WLS epochs eligible for startup bridge (default 10)")
    # Phase 9a: ZUPT (zero-velocity update) using PPC IMU
    parser.add_argument("--zupt-acc-norm-low-mps2", type=float, default=9.6,
                        help="Lower bound on accel norm [m/s^2] for static detection (default 9.6)")
    parser.add_argument("--zupt-acc-norm-high-mps2", type=float, default=9.95,
                        help="Upper bound on accel norm [m/s^2] for static detection (default 9.95)")
    parser.add_argument("--zupt-gyro-norm-max-dps", type=float, default=1.5,
                        help="Max gyro norm [deg/s] for static detection (default 1.5)")
    parser.add_argument("--zupt-apply-hybrid-statuses", type=str, default="1,3",
                        help="Hybrid Status values where ZUPT may overwrite (default '1,3', protects 4)")
    parser.add_argument("--zupt-min-consecutive", type=int, default=5,
                        help="Min consecutive static epochs (incl. current) required before ZUPT rewrites (default 5)")
    parser.add_argument("--zupt-max-anchor-drift-m", type=float, default=0.5,
                        help="Max |base - anchor| disagreement [m] tolerated before ZUPT skips (default 0.5)")
    # Phase 9b: tight-coupled IMU. In-loop pre-integration; per-particle
    # position pseudo-observation; emission switch on Status=1/3.
    parser.add_argument("--imu-tc-emit-pf-hybrid-statuses", type=str, default="1,3",
                        help="Hybrid Status values where IMU-TC emits the PF estimate (default '1,3', protects 4)")
    parser.add_argument("--imu-tc-pos-sigma-base-m", type=float, default=0.5,
                        help="IMU position pseudo-obs sigma at t=0s after anchor reset (default 0.5)")
    parser.add_argument("--imu-tc-pos-sigma-per-s", type=float, default=0.5,
                        help="IMU sigma growth per s of dead-reckoning (default 0.5)")
    parser.add_argument("--imu-tc-max-dr-seconds", type=float, default=5.0,
                        help="Max IMU dead-reckoning duration before falling back to hybrid (default 5.0)")
    parser.add_argument("--imu-tc-max-disagreement-m", type=float, default=30.0,
                        help="Skip IMU PU when |IMU-pred - hybrid| > this [m] (default 30)")
    parser.add_argument("--imu-tc-emit-max-diff-m", type=float, default=20.0,
                        help="Bound PF emission to hybrid +- this [m]; else keep hybrid (default 20)")
    parser.add_argument("--imu-tc-hybrid-loose-sigma-m", type=float, default=5.0,
                        help="Hybrid PU sigma on m-class Status epochs when IMU-TC is on (default 5.0; 0 disables)")
    # Phase 9c: full 15-state INS-GNSS EKF tight coupling.
    parser.add_argument("--ins-tc-emit-pf-hybrid-statuses", type=str, default="1,3",
                        help="Hybrid Status values where INS-TC emits the PF estimate (default '1,3', protects 4)")
    parser.add_argument("--ins-tc-obs-status-4-sigma-m", type=float, default=0.05,
                        help="INS position-observation horizontal sigma for Status=4 hybrid epochs (default 0.05)")
    parser.add_argument("--ins-tc-obs-status-3-sigma-m", type=float, default=0.0,
                        help="INS loose observation sigma for Status=3 hybrid epochs; 0 disables (default 0)")
    parser.add_argument("--ins-tc-max-dr-seconds", type=float, default=10.0,
                        help="Max INS dead-reckoning time since a position observation (default 10)")
    parser.add_argument("--ins-tc-max-disagreement-m", type=float, default=30.0,
                        help="Skip INS PF PU when |INS - hybrid| > this [m] (default 30)")
    parser.add_argument("--ins-tc-emit-max-diff-m", type=float, default=1.0,
                        help="Bound PF emission to hybrid +- this [m]; else keep hybrid (default 1)")
    parser.add_argument("--ins-tc-pf-pu-floor-sigma-m", type=float, default=0.1,
                        help="Lower bound on sigma fed to pf.position_update from INS (default 0.1)")
    parser.add_argument("--ins-tc-pf-pu-ceiling-sigma-m", type=float, default=5.0,
                        help="Upper bound on sigma fed to pf.position_update from INS (default 5.0)")
    parser.add_argument("--ins-tc-disable-particle-imu-predict", action="store_true",
                        help="Disable per-particle PF strapdown IMU prediction")
    parser.add_argument("--ins-tc-particle-imu-sigma-pos-m", type=float, default=0.02,
                        help="Per IMU-step PF position noise for strapdown predict [m] (default 0.02)")
    parser.add_argument("--ins-tc-particle-imu-sigma-acc-mps2", type=float, default=0.10,
                        help="PF strapdown accelerometer noise [m/s^2] (default 0.10)")
    parser.add_argument("--ins-tc-particle-imu-sigma-gyro-rps", type=float, default=0.005,
                        help="PF strapdown gyro noise [rad/s] (default 0.005)")
    parser.add_argument("--ins-tc-particle-imu-acc-bias-rw", type=float, default=1.0e-4,
                        help="PF accel-bias random walk [m/s^2/sqrt(s)] (default 1e-4)")
    parser.add_argument("--ins-tc-particle-imu-gyro-bias-rw", type=float, default=1.0e-5,
                        help="PF gyro-bias random walk [rad/s/sqrt(s)] (default 1e-5)")
    parser.add_argument("--ins-tc-particle-imu-att-spread-rad", type=float, default=math.radians(2.0),
                        help="Initial per-particle attitude spread [rad] (default 2deg)")
    parser.add_argument("--ins-tc-particle-imu-acc-bias-spread", type=float, default=0.05,
                        help="Initial accel-bias particle spread [m/s^2] (default 0.05)")
    parser.add_argument("--ins-tc-particle-imu-gyro-bias-spread-rps", type=float, default=math.radians(0.1),
                        help="Initial gyro-bias particle spread [rad/s] (default 0.1deg/s)")
    parser.add_argument("--ins-tc-particle-imu-velocity-spread-mps", type=float, default=0.5,
                        help="Initial PF inertial velocity spread [m/s] (default 0.5)")
    parser.add_argument("--ins-tc-enable-recenter-status4", action="store_true",
                        help="Enable PF cloud recentering on trusted Status=4 INS/GNSS anchors")
    parser.add_argument("--ins-tc-recenter-max-shift-m", type=float, default=5000.0,
                        help="Skip Status=4 PF recenter if required shift exceeds this [m] (default 5000)")
    parser.add_argument("--ins-tc-disable-motion-predict", action="store_true",
                        help="Disable INS velocity as the PF predict motion guide")
    parser.add_argument("--ins-tc-predict-sigma-pos-m", type=float, default=0.2,
                        help="PF predict position noise [m] when using INS velocity guide (default 0.2)")
    parser.add_argument("--ins-tc-predict-velocity-alpha", type=float, default=1.0,
                        help="Blend factor for INS velocity in PF predict (default 1.0)")
    parser.add_argument("--ins-tc-align-acc-low", type=float, default=9.6,
                        help="Lower accel-norm bound for INS static alignment (default 9.6)")
    parser.add_argument("--ins-tc-align-acc-high", type=float, default=9.95,
                        help="Upper accel-norm bound for INS static alignment (default 9.95)")
    parser.add_argument("--ins-tc-align-gyro-max-dps", type=float, default=1.5,
                        help="Max gyro norm [deg/s] for INS static alignment (default 1.5)")
    parser.add_argument("--ins-tc-align-min-samples", type=int, default=50,
                        help="Consecutive static IMU samples required for INS alignment (default 50)")
    parser.add_argument("--ins-tc-yaw-init-min-speed-mps", type=float, default=1.0,
                        help="Planar ENU speed required before yaw is initialized from hybrid velocity (default 1.0)")
    parser.add_argument("--ins-tc-quality-gate-enabled", action="store_true",
                        help="Enable rolling fix-rate gate on ins_tc PF emit. Suppresses ins_tc emission "
                             "in epochs where the previous K hybrid statuses had a fix rate >= threshold "
                             "(default off)")
    parser.add_argument("--ins-tc-quality-gate-window-epochs", type=int, default=30,
                        help="Window length for the ins_tc quality-gate fix-rate average (default 30)")
    parser.add_argument("--ins-tc-quality-gate-max-fix-rate", type=float, default=0.5,
                        help="Maximum recent fix rate at which ins_tc PF emit is allowed; above this it "
                             "is suppressed (default 0.5)")
    parser.add_argument("--ins-tc-quality-gate-pu-skip", action="store_true",
                        help="When set (and quality gate enabled), also skip ins_tc PF position_update "
                             "in high-fix-rate epochs, not just PF emit. Reduces PF state contamination "
                             "in good-GNSS runs (default off)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Cap epochs per run (smoke / debug). None = full run.")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--imu", choices=("off", "preint"), default="off",
        help=(
            "WP22a ablation switch: 'off' (default) = baseline predict step "
            "unchanged (no IMU guide, i.e. reproduces the pre-WP22a "
            "RBPF-velKF+DD+gate+hybrid numbers). 'preint' = wire in the "
            "WP21b IMU-preintegration predict-step guide "
            "(ImuPreintPfGuide, heading-uncertainty sigma_pos + "
            "set_velocity_covariance Sigma_v feeding) using the PPC 100 Hz "
            "IMU, applied uniformly to every selected --methods variant."
        ),
    )
    parser.add_argument("--imu-preint-sigma-accel", type=float, default=0.05,
                        help="ImuPreintPfGuide sigma_accel_mps2_sqrthz (WP21b default).")
    parser.add_argument("--imu-preint-sigma-gyro", type=float, default=0.005,
                        help="ImuPreintPfGuide sigma_gyro_radps_sqrthz (WP21b default).")
    parser.add_argument("--imu-preint-sigma-pos-floor", type=float, default=0.05,
                        help="Small numerical-stability floor on sigma_pos (WP21b default, not a tuning knob).")
    parser.add_argument("--imu-preint-sigma-pos-scale", type=float, default=1.0)
    parser.add_argument("--imu-preint-velocity-blend-alpha", type=float, default=0.3)
    parser.add_argument("--imu-preint-sigma-spp-pos-m", type=float, default=30.0,
                        help="Documented raw-SPP horizontal sigma [m] used for the heading-uncertainty term.")
    parser.add_argument("--imu-preint-min-heading-fix-disp-m", type=float, default=2.0)
    parser.add_argument(
        "--enable-epoch-tempering", action="store_true",
        help=(
            "WP22b item 2: adaptive likelihood tempering. Scales this "
            "epoch's total log-likelihood increment by a bisected "
            "temperature beta in (0,1] targeting "
            "--epoch-tempering-target-ess-ratio ESS/N (default 0.10), "
            "applied uniformly to every selected --methods variant. "
            "Forces end-of-epoch deferred resampling internally."
        ),
    )
    parser.add_argument("--epoch-tempering-target-ess-ratio", type=float, default=0.10,
                        help="Target post-tempering ESS/N (default 0.10).")
    parser.add_argument("--epoch-tempering-max-iters", type=int, default=20,
                        help="Bisection iterations for the tempering solve (default 20).")
    parser.add_argument(
        "--enable-cn0-gmm", action="store_true",
        help=(
            "WP22b item 3: drive the PR-GMM LOS/NLOS mixture weight w_los "
            "per satellite from measured C/N0 + elevation (see "
            "_cn0_elevation_w_los), instead of the fixed --pr-gmm-w-los "
            "scalar. Implies enable_pr_gmm=True. Applied uniformly to "
            "every selected --methods variant."
        ),
    )
    parser.add_argument("--cn0-gmm-baseline-dbhz0", type=float, default=30.0,
                        help="Clear-sky C/N0 baseline at 0deg elevation [dB-Hz] (default 30).")
    parser.add_argument("--cn0-gmm-baseline-dbhz90", type=float, default=45.0,
                        help="Clear-sky C/N0 baseline at 90deg elevation [dB-Hz] (default 45).")
    parser.add_argument("--cn0-gmm-gap-mid-db", type=float, default=14.0,
                        help="C/N0 deficit [dB] below baseline at which w_los=0.5; default 14.0 is the "
                             "mean of the two validated UrbanNav site gaps (16.5/11.9 dB).")
    parser.add_argument("--cn0-gmm-logistic-scale-db", type=float, default=4.0,
                        help="Logistic transition width [dB] (default 4.0, sharp -- matches the validated "
                             "0.94-0.985 AUC separation).")
    parser.add_argument("--cn0-gmm-w-los-min", type=float, default=0.05)
    parser.add_argument("--cn0-gmm-w-los-max", type=float, default=0.97)
    parser.add_argument("--cn0-gmm-n-buckets", type=int, default=5,
                        help="Number of per-satellite w_los buckets (kernel calls) per epoch (default 5).")
    parser.add_argument(
        "--enable-particle-nlos", action="store_true",
        help=(
            "WP22b item 4: enable the native per-particle NLOS threshold "
            "gate (undifferenced PR + DD-carrier-AFV kernels; each "
            "particle computes its own residual from its own hypothesized "
            "state, so the same scalar threshold rejects a different "
            "satellite subset per particle). Applied uniformly to every "
            "selected --methods variant. No CUDA changes -- this wires an "
            "existing kernel feature that was never plumbed through this "
            "runner's _build_pf before WP22b."
        ),
    )
    parser.add_argument("--particle-nlos-undiff-pr-threshold-m", type=float, default=30.0)
    parser.add_argument("--particle-nlos-dd-carrier-threshold-cycles", type=float, default=0.5)
    parser.add_argument("--particle-nlos-huber", action="store_true")
    parser.add_argument("--particle-nlos-huber-undiff-pr-k", type=float, default=1.5)
    parser.add_argument("--particle-nlos-huber-dd-carrier-k", type=float, default=1.5)
    parser.add_argument(
        "--runs",
        type=str,
        default="all",
        help="Comma-separated run filter, e.g. 'tokyo/run1,nagoya/run3'. 'all' = 6 runs.",
    )
    return parser
