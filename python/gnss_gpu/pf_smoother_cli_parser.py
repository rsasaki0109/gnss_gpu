"""Argparse construction for PF smoother evaluation CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_pf_smoother_arg_parser(default_sigma_pos: float) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PF + optional forward-backward smoother (gnssplusplus stack)")
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Apply a named CLI preset before parsing; later flags override preset values",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available CLI presets and exit",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=str, default="Odaiba")
    parser.add_argument("--n-particles", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-pos", type=float, default=float(default_sigma_pos))
    parser.add_argument("--sigma-pr", type=float, default=3.0)
    parser.add_argument(
        "--position-update-sigma",
        type=float,
        default=3.0,
        help="SPP soft constraint (m); use negative to disable",
    )
    parser.add_argument(
        "--predict-guide",
        choices=("spp", "tdcp", "tdcp_adaptive", "imu", "imu_spp_blend"),
        default="spp",
    )
    parser.add_argument("--smoother", action="store_true", help="Enable forward-backward smooth")
    parser.add_argument("--compare-both", action="store_true", help="Run with and without smoother")
    parser.add_argument("--max-epochs", type=int, default=0, help="Limit valid epochs (0 = no limit)")
    parser.add_argument(
        "--skip-valid-epochs",
        type=int,
        default=0,
        help="Process (burn-in) this many valid epochs before recording metrics; "
        "total processed = skip + max-epochs when max-epochs > 0",
    )
    parser.add_argument("--urban-rover", type=str, default="trimble")
    parser.add_argument(
        "--smoother-skip-widelane-dd-pseudorange",
        action="store_true",
        help="In the backward smoother pass, replay undifferenced PR instead of wide-lane-derived DD PR",
    )
    parser.add_argument(
        "--smoother-position-update-sigma",
        type=float,
        default=None,
        help="Override SPP position-update sigma used only by the backward smoother pass",
    )
    parser.add_argument(
        "--smoother-widelane-forward-guard",
        action="store_true",
        help="For epochs where wide-lane DD PR was used in the forward pass, keep the forward estimate after smoothing",
    )
    parser.add_argument(
        "--smoother-widelane-forward-guard-min-shift-m",
        type=float,
        default=None,
        help="Only apply --smoother-widelane-forward-guard when smoothed-vs-forward 3D shift is at least this value",
    )
    parser.add_argument(
        "--stop-segment-constant",
        action="store_true",
        help="Replace IMU stop-detected segments with a robust constant position after smoothing",
    )
    parser.add_argument(
        "--stop-segment-min-epochs",
        type=int,
        default=5,
        help="Minimum contiguous stop epochs required for --stop-segment-constant",
    )
    parser.add_argument(
        "--stop-segment-source",
        choices=(
            "smoothed",
            "forward",
            "combined",
            "smoothed_density",
            "forward_density",
            "combined_density",
            "smoothed_auto",
            "forward_auto",
            "combined_auto",
            "smoothed_auto_tail",
            "forward_auto_tail",
            "combined_auto_tail",
        ),
        default="smoothed",
        help="Position source used to estimate the constant stop-segment center",
    )
    parser.add_argument(
        "--stop-segment-max-radius-m",
        type=float,
        default=None,
        help="Skip stop segments whose robust 90th-percentile radius exceeds this value",
    )
    parser.add_argument(
        "--stop-segment-blend",
        type=float,
        default=1.0,
        help="Blend factor toward the constant stop-segment center",
    )
    parser.add_argument(
        "--stop-segment-density-neighbors",
        type=int,
        default=200,
        help="Nearest-neighbor count used by *_density stop-segment sources",
    )
    parser.add_argument(
        "--stop-segment-static-gnss",
        action="store_true",
        help="Refine IMU stop-detected segments as static GNSS batches after smoothing",
    )
    parser.add_argument(
        "--stop-segment-static-min-observations",
        type=int,
        default=40,
        help="Minimum GNSS residual count required for static stop-segment refinement",
    )
    parser.add_argument(
        "--stop-segment-static-prior-sigma-m",
        type=float,
        default=20.0,
        help="Position prior sigma for static stop-segment refinement",
    )
    parser.add_argument(
        "--stop-segment-static-pr-sigma-m",
        type=float,
        default=8.0,
        help="Undifferenced pseudorange sigma for static stop-segment refinement",
    )
    parser.add_argument(
        "--stop-segment-static-dd-pr-sigma-m",
        type=float,
        default=4.0,
        help="Double-difference pseudorange sigma for static stop-segment refinement",
    )
    parser.add_argument(
        "--stop-segment-static-dd-cp-sigma-cycles",
        type=float,
        default=0.50,
        help="Modulo double-difference carrier sigma for static stop-segment refinement",
    )
    parser.add_argument(
        "--stop-segment-static-max-update-m",
        type=float,
        default=25.0,
        help="Reject static stop-segment refinement whose update exceeds this distance",
    )
    parser.add_argument(
        "--stop-segment-static-blend",
        type=float,
        default=1.0,
        help="Blend factor toward the static GNSS stop-segment solution",
    )
    parser.add_argument(
        "--sigma-pos-tdcp",
        type=float,
        default=None,
        help="When TDCP velocity is accepted, use this predict sigma_pos (m); "
        "omit to use --sigma-pos for all epochs",
    )
    parser.add_argument(
        "--sigma-pos-tdcp-tight",
        type=float,
        default=None,
        help="If set and TDCP postfit RMS < --tdcp-tight-rms-max, use this sigma_pos",
    )
    parser.add_argument(
        "--tdcp-tight-rms-max",
        type=float,
        default=1.0e9,
        help="postfit RMS threshold (m) for --sigma-pos-tdcp-tight (default: disabled)",
    )
    parser.add_argument(
        "--tdcp-rms-threshold",
        type=float,
        default=3.0,
        help="Postfit RMS threshold (m) for tdcp_adaptive mode; "
        "epochs with RMS >= threshold fall back to Doppler/random-walk predict",
    )
    parser.add_argument(
        "--tdcp-elevation-weight",
        action="store_true",
        help="WLS row weight sin(el)^2 when measurements expose elevation (TDCP guide only)",
    )
    parser.add_argument(
        "--tdcp-el-sin-floor",
        type=float,
        default=0.1,
        help="Floor on sin(elevation) when --tdcp-elevation-weight is set",
    )
    parser.add_argument(
        "--residual-downweight",
        action="store_true",
        help="Downweight satellites with large SPP residuals (Cauchy-like)",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=15.0,
        help="Residual threshold (m) for Cauchy downweighting",
    )
    parser.add_argument(
        "--pr-accel-downweight",
        action="store_true",
        help="Downweight satellites with large pseudorange acceleration (multipath indicator)",
    )
    parser.add_argument(
        "--pr-accel-threshold",
        type=float,
        default=5.0,
        help="PR acceleration threshold (m) for Cauchy downweighting",
    )
    parser.add_argument("--gmm", action="store_true", help="Use GMM likelihood (LOS+NLOS mixture)")
    parser.add_argument("--gmm-w-los", type=float, default=0.7, help="GMM LOS weight")
    parser.add_argument("--gmm-mu-nlos", type=float, default=15.0, help="GMM NLOS mean bias (m)")
    parser.add_argument("--gmm-sigma-nlos", type=float, default=30.0, help="GMM NLOS sigma (m)")
    parser.add_argument(
        "--doppler-position-update",
        action="store_true",
        help="Apply a second position_update using Doppler-predicted position (prev_estimate + velocity*dt)",
    )
    parser.add_argument(
        "--doppler-pu-sigma",
        type=float,
        default=5.0,
        help="Sigma (m) for Doppler-predicted position_update constraint",
    )
    parser.add_argument(
        "--doppler-per-particle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply legacy sampled per-particle Doppler velocity update (default: disabled)",
    )
    parser.add_argument(
        "--doppler-sigma-mps",
        type=float,
        default=0.5,
        help="Doppler range-rate sigma for per-particle velocity update (m/s)",
    )
    parser.add_argument(
        "--doppler-velocity-update-gain",
        type=float,
        default=0.25,
        help="Blend toward each particle's Doppler WLS velocity solution",
    )
    parser.add_argument(
        "--doppler-max-velocity-update-mps",
        type=float,
        default=10.0,
        help="Cap per-epoch Doppler velocity correction magnitude (m/s)",
    )
    parser.add_argument(
        "--doppler-min-sats",
        type=int,
        default=4,
        help="Minimum Doppler rows required for per-particle velocity update",
    )
    parser.add_argument(
        "--rbpf-velocity-kf",
        action="store_true",
        help="Enable proper RBPF velocity marginalization with a per-particle KF state",
    )
    parser.add_argument(
        "--rbpf-velocity-init-sigma",
        type=float,
        default=2.0,
        help="Initial proper-RBPF velocity KF sigma per axis (m/s)",
    )
    parser.add_argument(
        "--rbpf-velocity-process-noise",
        type=float,
        default=1.0,
        help="Proper-RBPF velocity process noise Q_v (m^2/s^3)",
    )
    parser.add_argument(
        "--rbpf-doppler-sigma",
        type=float,
        default=None,
        help="Doppler range-rate sigma for proper RBPF velocity KF (m/s); defaults to --doppler-sigma-mps",
    )
    parser.add_argument(
        "--rbpf-velocity-kf-gate-min-dd-pairs",
        type=int,
        default=None,
        help="Only apply RBPF Doppler KF when DD carrier kept-pair count is at least this value",
    )
    parser.add_argument(
        "--rbpf-velocity-kf-gate-min-ess-ratio",
        type=float,
        default=None,
        help="Only apply RBPF Doppler KF when pre-update ESS ratio is at least this value",
    )
    parser.add_argument(
        "--rbpf-velocity-kf-gate-max-spread-m",
        type=float,
        default=None,
        help="Only apply RBPF Doppler KF when particle spread is at or below this value",
    )
    parser.add_argument(
        "--pf-sigma-vel",
        type=float,
        default=0.0,
        help="Per-axis particle velocity process noise for predict (m/s)",
    )
    parser.add_argument(
        "--pf-velocity-guide-alpha",
        type=float,
        default=1.0,
        help="Blend factor toward the shared velocity guide before propagation",
    )
    parser.add_argument(
        "--pf-init-spread-vel",
        type=float,
        default=0.0,
        help="Initial per-axis particle velocity spread (m/s)",
    )
    parser.add_argument(
        "--imu-tight-coupling",
        action="store_true",
        help="Apply IMU dead-reckoning position_update after SPP in each epoch",
    )
    parser.add_argument(
        "--imu-stop-sigma-pos",
        type=float,
        default=None,
        help="When IMU detects stop (speed<0.01 m/s), use this predict sigma_pos (m); "
        "omit to use --sigma-pos for stop epochs",
    )
    parser.add_argument(
        "--tdcp-position-update",
        action="store_true",
        help="Apply TDCP displacement as tight position_update (carrier-phase constraint)",
    )
    parser.add_argument("--tdcp-pu-sigma", type=float, default=0.5,
                        help="Sigma for TDCP displacement position_update (m)")
    parser.add_argument("--tdcp-pu-rms-max", type=float, default=3.0,
                        help="Max TDCP postfit RMS to apply displacement PU (m)")
    parser.add_argument(
        "--tdcp-pu-spp-max-diff-mps",
        type=float,
        default=6.0,
        help="Max TDCP/SPP finite-difference velocity disagreement for TDCP PU; <=0 disables this guard",
    )
    parser.add_argument(
        "--tdcp-pu-gate-dd-carrier-max-pairs",
        type=int,
        default=None,
        help="Only apply TDCP PU when DD carrier kept-pair count is at or below this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-dd-carrier-min-pairs",
        type=int,
        default=None,
        help="Only apply TDCP PU when DD carrier kept-pair count is at or above this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-dd-pseudorange-max-pairs",
        type=int,
        default=None,
        help="Only apply TDCP PU when DD pseudorange kept-pair count is at or below this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-min-spread-m",
        type=float,
        default=None,
        help="Only apply TDCP PU when particle spread is at or above this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-max-spread-m",
        type=float,
        default=None,
        help="Only apply TDCP PU when particle spread is at or below this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-min-ess-ratio",
        type=float,
        default=None,
        help="Only apply TDCP PU when ESS ratio is at or above this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-max-ess-ratio",
        type=float,
        default=None,
        help="Only apply TDCP PU when ESS ratio is at or below this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-dd-pr-max-raw-median-m",
        type=float,
        default=None,
        help="Only apply TDCP PU when raw DD pseudorange median residual is at or below this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-dd-cp-max-raw-afv-median-cycles",
        type=float,
        default=None,
        help="Only apply TDCP PU when raw DD carrier AFV median residual is at or below this value",
    )
    parser.add_argument(
        "--tdcp-pu-gate-logic",
        choices=("any", "all"),
        default="any",
        help="Combine active TDCP PU gate conditions with any/or or all/and",
    )
    parser.add_argument(
        "--tdcp-pu-gate-stop-mode",
        choices=("any", "stopped", "moving"),
        default="any",
        help="Restrict TDCP PU to IMU stop-detected epochs, moving epochs, or either",
    )
    parser.add_argument("--mupf", action="store_true",
                        help="Multiple Update PF: carrier phase AFV update after pseudorange")
    parser.add_argument("--mupf-sigma-cycles", type=float, default=0.05,
                        help="Carrier phase AFV sigma in cycles (default 0.05 ≈ 1cm)")
    parser.add_argument("--mupf-snr-min", type=float, default=25.0,
                        help="Min C/N0 (dB-Hz) for carrier phase in MUPF")
    parser.add_argument("--mupf-elev-min", type=float, default=0.15,
                        help="Min elevation (rad) for carrier phase in MUPF (~8.6 deg)")
    parser.add_argument("--dd-pseudorange", action="store_true",
                        help="Use DD pseudorange as the primary weight update (requires base station RINEX)")
    parser.add_argument("--dd-pseudorange-sigma", type=float, default=0.75,
                        help="DD pseudorange sigma in meters (default 0.75)")
    parser.add_argument("--dd-pseudorange-base-interp", action="store_true",
                        help="Linearly interpolate 1 Hz base pseudorange to rover epoch before DD formation")
    parser.add_argument("--dd-pseudorange-gate-residual-m", type=float, default=None,
                        help="Drop DD pseudorange pairs whose abs residual exceeds this threshold (m)")
    parser.add_argument("--dd-pseudorange-gate-adaptive-floor-m", type=float, default=None,
                        help="Adaptive DD pseudorange pair gate floor in meters")
    parser.add_argument("--dd-pseudorange-gate-adaptive-mad-mult", type=float, default=None,
                        help="Adaptive DD pseudorange pair gate uses median + k*MAD with this k")
    parser.add_argument("--dd-pseudorange-gate-epoch-median-m", type=float, default=None,
                        help="Skip a DD pseudorange epoch when kept-pair median abs residual exceeds this threshold (m)")
    parser.add_argument("--dd-pseudorange-gate-ess-min-scale", type=float, default=1.0,
                        help="ESS-linked lower multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-ess-max-scale", type=float, default=1.0,
                        help="ESS-linked upper multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-spread-min-scale", type=float, default=1.0,
                        help="Spread-linked lower multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-spread-max-scale", type=float, default=1.0,
                        help="Spread-linked upper multiplier for DD pseudorange gate thresholds")
    parser.add_argument("--dd-pseudorange-gate-low-spread-m", type=float, default=1.5,
                        help="Particle spread below this is treated as tightly converged for DD pseudorange gating")
    parser.add_argument("--dd-pseudorange-gate-high-spread-m", type=float, default=8.0,
                        help="Particle spread above this is treated as diffuse for DD pseudorange gating")
    parser.add_argument("--per-particle-nlos-gate", action="store_true",
                        help="Reject large-residual observations independently for each particle inside PF kernels")
    parser.add_argument("--per-particle-nlos-dd-pr-threshold-m", type=float, default=10.0,
                        help="Per-particle DD pseudorange abs residual threshold in meters")
    parser.add_argument("--per-particle-nlos-dd-carrier-threshold-cycles", type=float, default=0.5,
                        help="Per-particle DD carrier AFV abs residual threshold in cycles")
    parser.add_argument("--per-particle-nlos-undiff-pr-threshold-m", type=float, default=30.0,
                        help="Per-particle undifferenced pseudorange abs residual threshold in meters")
    parser.add_argument("--per-particle-huber", action="store_true",
                        help="Use per-particle Huber soft cost in SPP PR, DD PR, and DD carrier AFV kernels")
    parser.add_argument("--per-particle-huber-dd-pr-k", type=float, default=1.5,
                        help="Huber k for per-particle DD pseudorange residuals")
    parser.add_argument("--per-particle-huber-dd-carrier-k", type=float, default=1.5,
                        help="Huber k for per-particle DD carrier AFV residuals")
    parser.add_argument("--per-particle-huber-undiff-pr-k", type=float, default=1.5,
                        help="Huber k for per-particle undifferenced pseudorange residuals")
    parser.add_argument("--widelane", action="store_true",
                        help="Replace DD pseudorange with ratio-tested L1-L2 wide-lane fixed DD pseudorange when available")
    parser.add_argument("--widelane-min-fix-rate", type=float, default=0.3,
                        help="Minimum per-epoch fixed/candidate wide-lane DD pair rate before replacing DD pseudorange")
    parser.add_argument("--widelane-ratio-threshold", type=float, default=3.0,
                        help="Minimum LAMBDA ratio for accepting a wide-lane DD integer fix")
    parser.add_argument("--widelane-dd-sigma", type=float, default=0.1,
                        help="DD pseudorange sigma in meters for fixed wide-lane DD rows")
    parser.add_argument("--widelane-gate-min-fixed-pairs", type=int, default=None,
                        help="Require at least this many fixed wide-lane DD pairs before using a WL epoch")
    parser.add_argument("--widelane-gate-min-fix-rate", type=float, default=None,
                        help="Require this fixed/candidate pair rate before using a WL epoch")
    parser.add_argument("--widelane-gate-min-spread-m", type=float, default=None,
                        help="Require PF spread at least this large before using a WL epoch")
    parser.add_argument("--widelane-gate-max-epoch-median-residual-m", type=float, default=None,
                        help="Reject a WL epoch when kept-pair median residual to the PF estimate exceeds this threshold")
    parser.add_argument("--widelane-gate-max-pair-residual-m", type=float, default=None,
                        help="Drop WL DD pairs whose residual to the PF estimate exceeds this threshold")
    parser.add_argument("--mupf-dd", action="store_true",
                        help="Use Double-Differenced carrier phase AFV (requires base station RINEX)")
    parser.add_argument("--mupf-dd-sigma-cycles", type=float, default=0.05,
                        help="DD carrier phase AFV sigma in cycles (default 0.05)")
    parser.add_argument("--mupf-dd-base-interp", action="store_true",
                        help="Linearly interpolate 1 Hz base carrier phase to rover epoch before DD formation")
    parser.add_argument("--mupf-dd-gate-afv-cycles", type=float, default=None,
                        help="Drop DD carrier pairs whose abs AFV exceeds this threshold (cycles)")
    parser.add_argument("--mupf-dd-gate-adaptive-floor-cycles", type=float, default=None,
                        help="Adaptive DD carrier pair gate floor in cycles")
    parser.add_argument("--mupf-dd-gate-adaptive-mad-mult", type=float, default=None,
                        help="Adaptive DD carrier pair gate uses median + k*MAD with this k")
    parser.add_argument("--mupf-dd-gate-epoch-median-cycles", type=float, default=None,
                        help="Skip a DD carrier epoch when kept-pair median abs AFV exceeds this threshold (cycles)")
    parser.add_argument("--mupf-dd-gate-low-ess-epoch-median-cycles", type=float, default=None,
                        help="Contextual DD carrier epoch-median AFV limit (cycles) used under low-ESS conditions")
    parser.add_argument("--mupf-dd-gate-low-ess-max-ratio", type=float, default=None,
                        help="Enable contextual DD carrier epoch-median gate when ESS ratio is at or below this threshold")
    parser.add_argument("--mupf-dd-gate-low-ess-max-spread-m", type=float, default=None,
                        help="Require PF spread to stay at or below this threshold when applying the contextual DD carrier epoch-median gate")
    parser.add_argument("--mupf-dd-gate-low-ess-require-no-dd-pr", action="store_true",
                        help="Require DD pseudorange to be absent before applying the contextual DD carrier epoch-median gate")
    parser.add_argument("--mupf-dd-gate-ess-min-scale", type=float, default=1.0,
                        help="ESS-linked lower multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-ess-max-scale", type=float, default=1.0,
                        help="ESS-linked upper multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-spread-min-scale", type=float, default=1.0,
                        help="Spread-linked lower multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-spread-max-scale", type=float, default=1.0,
                        help="Spread-linked upper multiplier for DD carrier gate thresholds")
    parser.add_argument("--mupf-dd-gate-low-spread-m", type=float, default=1.5,
                        help="Particle spread below this is treated as tightly converged for DD carrier gating")
    parser.add_argument("--mupf-dd-gate-high-spread-m", type=float, default=8.0,
                        help="Particle spread above this is treated as diffuse for DD carrier gating")
    parser.add_argument("--mupf-dd-sigma-support-low-pairs", type=int, default=None,
                        help="Relax DD carrier sigma when kept DD pair count is at or below this threshold")
    parser.add_argument("--mupf-dd-sigma-support-high-pairs", type=int, default=None,
                        help="Return DD carrier sigma to baseline when kept DD pair count reaches this threshold")
    parser.add_argument("--mupf-dd-sigma-support-max-scale", type=float, default=1.0,
                        help="Maximum DD carrier sigma multiplier from sparse-support scaling")
    parser.add_argument("--mupf-dd-sigma-afv-good-cycles", type=float, default=None,
                        help="DD carrier raw abs AFV median at or below this keeps sigma at baseline")
    parser.add_argument("--mupf-dd-sigma-afv-bad-cycles", type=float, default=None,
                        help="DD carrier raw abs AFV median at or above this relaxes sigma toward max scale")
    parser.add_argument("--mupf-dd-sigma-afv-max-scale", type=float, default=1.0,
                        help="Maximum DD carrier sigma multiplier from raw AFV scaling")
    parser.add_argument("--mupf-dd-sigma-ess-low-ratio", type=float, default=None,
                        help="ESS ratio at or below this relaxes DD carrier sigma toward the ESS max scale")
    parser.add_argument("--mupf-dd-sigma-ess-high-ratio", type=float, default=None,
                        help="ESS ratio at or above this keeps DD carrier sigma at baseline")
    parser.add_argument("--mupf-dd-sigma-ess-max-scale", type=float, default=1.0,
                        help="Maximum DD carrier sigma multiplier from ESS-linked scaling")
    parser.add_argument("--mupf-dd-sigma-max-scale", type=float, default=None,
                        help="Optional clip on the combined DD carrier sigma multiplier")
    parser.add_argument("--carrier-anchor", action="store_true",
                        help="Seed per-satellite carrier biases from good DD epochs and reuse them as pseudorange-like updates when DD is weak")
    parser.add_argument("--carrier-anchor-sigma-m", type=float, default=0.25,
                        help="Sigma in meters for carrier-bias-conditioned anchored updates")
    parser.add_argument("--carrier-anchor-min-sats", type=int, default=4,
                        help="Minimum anchored carrier satellites required to apply the carrier-anchor update")
    parser.add_argument("--carrier-anchor-max-age-s", type=float, default=3.0,
                        help="Maximum age in seconds for a stored carrier bias before it is ignored")
    parser.add_argument("--carrier-anchor-max-residual-m", type=float, default=0.75,
                        help="Maximum anchored carrier residual in meters before a satellite is rejected")
    parser.add_argument("--carrier-anchor-max-continuity-residual-m", type=float, default=0.50,
                        help="Maximum inter-epoch carrier continuity residual in meters before a satellite is treated as slipped")
    parser.add_argument("--carrier-anchor-min-stable-epochs", type=int, default=1,
                        help="Minimum stable epochs required before a stored carrier bias can be reused")
    parser.add_argument("--carrier-anchor-blend-alpha", type=float, default=0.5,
                        help="EMA blending factor when refreshing a stored carrier bias on a trusted DD epoch")
    parser.add_argument("--carrier-anchor-reanchor-jump-cycles", type=float, default=4.0,
                        help="If refreshed carrier bias jumps by more than this many cycles, replace it instead of blending")
    parser.add_argument("--carrier-anchor-seed-dd-min-pairs", type=int, default=3,
                        help="Minimum kept DD carrier pairs required to trust an epoch for carrier-anchor seeding")
    parser.add_argument("--mupf-dd-fallback-undiff", action="store_true",
                        help="When DD carrier is unavailable, replay a same-band undifferenced carrier AFV update")
    parser.add_argument("--mupf-dd-fallback-sigma-cycles", type=float, default=0.10,
                        help="Sigma for undifferenced carrier AFV fallback used when DD carrier is unavailable")
    parser.add_argument("--mupf-dd-fallback-min-sats", type=int, default=4,
                        help="Minimum same-band carrier satellites required for undifferenced AFV fallback")
    parser.add_argument("--mupf-dd-fallback-prefer-tracked", action="store_true",
                        help="When carrier-anchor tracker is available, prefer tracker-consistent satellites for undifferenced AFV fallback")
    parser.add_argument("--mupf-dd-fallback-tracked-min-stable-epochs", type=int, default=1,
                        help="Minimum tracker stable epochs required before a satellite is eligible for tracked undiff AFV fallback")
    parser.add_argument("--mupf-dd-fallback-tracked-min-sats", type=int, default=None,
                        help="Minimum tracker-consistent satellites required before treating hybrid undiff fallback as tracked-assisted")
    parser.add_argument("--mupf-dd-fallback-tracked-continuity-good-m", type=float, default=None,
                        help="Tracked fallback continuity median at or below this tightens fallback sigma toward the tracked min scale")
    parser.add_argument("--mupf-dd-fallback-tracked-continuity-bad-m", type=float, default=None,
                        help="Tracked fallback continuity median at or above this relaxes fallback sigma toward the tracked max scale")
    parser.add_argument("--mupf-dd-fallback-tracked-sigma-min-scale", type=float, default=1.0,
                        help="Minimum multiplier for tracked undiff fallback sigma when continuity is very good")
    parser.add_argument("--mupf-dd-fallback-tracked-sigma-max-scale", type=float, default=1.0,
                        help="Maximum multiplier for tracked undiff fallback sigma when continuity degrades")
    parser.add_argument("--mupf-dd-fallback-weak-dd-max-pairs", type=int, default=None,
                        help="If set, try undiff carrier fallback before DD carrier update when kept DD carrier pairs are at or below this threshold")
    parser.add_argument("--mupf-dd-fallback-weak-dd-max-ess-ratio", type=float, default=None,
                        help="If set, weak-DD fallback replacement also requires PF ESS ratio to be at or below this threshold")
    parser.add_argument("--mupf-dd-fallback-weak-dd-min-raw-afv-median-cycles", type=float, default=None,
                        help="When weak-DD fallback replacement is enabled, require raw abs AFV median to be at or above this threshold")
    parser.add_argument("--mupf-dd-fallback-weak-dd-require-no-dd-pr", action="store_true",
                        help="Require DD pseudorange to be absent before replacing a weak DD carrier update with undiff fallback")
    parser.add_argument("--mupf-dd-skip-low-support-ess-ratio", type=float, default=None,
                        help="Skip DD carrier update when ESS ratio is at or below this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-max-pairs", type=int, default=None,
                        help="Skip DD carrier update when kept DD carrier pairs are at or below this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-max-spread-m", type=float, default=None,
                        help="Skip DD carrier update only when PF spread is at or below this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-min-raw-afv-median-cycles", type=float, default=None,
                        help="Skip DD carrier update when raw abs AFV median is at or above this threshold")
    parser.add_argument("--mupf-dd-skip-low-support-require-no-dd-pr", action="store_true",
                        help="Require DD pseudorange to be absent when applying DD carrier low-support skip")
    parser.add_argument(
        "--epoch-diagnostics-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-epoch diagnostics (tail analysis)",
    )
    parser.add_argument(
        "--epoch-diagnostics-top-k",
        type=int,
        default=0,
        help="Print the worst K aligned epochs by 2D error using per-epoch diagnostics",
    )
    parser.add_argument(
        "--smoother-tail-guard-ess-max-ratio",
        type=float,
        default=None,
        help="If set, fallback smoothed epochs to forward when ESS ratio is at or below this threshold",
    )
    parser.add_argument(
        "--smoother-tail-guard-dd-carrier-max-pairs",
        type=int,
        default=None,
        help="If set, require DD carrier kept-pair count to be at or below this threshold for smoother tail guard",
    )
    parser.add_argument(
        "--smoother-tail-guard-dd-pseudorange-max-pairs",
        type=int,
        default=None,
        help="If set, require DD pseudorange kept-pair count to be at or below this threshold for smoother tail guard",
    )
    parser.add_argument(
        "--smoother-tail-guard-min-shift-m",
        type=float,
        default=None,
        help="If set, require smoothed-vs-forward 3D shift to be at or above this threshold for smoother tail guard",
    )
    parser.add_argument(
        "--smoother-tail-guard-expand-epochs",
        type=int,
        default=None,
        help="Expand smoother tail guard to moving epochs within this many epochs of an initially guarded moving epoch",
    )
    parser.add_argument(
        "--smoother-tail-guard-expand-min-shift-m",
        type=float,
        default=None,
        help="If set, require this smoothed-vs-forward 3D shift for expanded smoother tail guard epochs",
    )
    parser.add_argument(
        "--smoother-tail-guard-expand-dd-pseudorange-max-pairs",
        type=int,
        default=None,
        help="If set, require expanded smoother tail guard epochs to have DD pseudorange kept-pair count at or below this threshold",
    )
    parser.add_argument(
        "--fgo-local-window",
        type=str,
        default=None,
        help="Apply local FGO post-process to a weak-DD window: 'auto' or inclusive N:M aligned-epoch indices",
    )
    parser.add_argument(
        "--fgo-local-window-min-epochs",
        type=int,
        default=100,
        help="Minimum weak-DD run length for --fgo-local-window auto",
    )
    parser.add_argument(
        "--fgo-local-dd-max-pairs",
        type=int,
        default=4,
        help="DD carrier kept-pair threshold used by --fgo-local-window auto",
    )
    parser.add_argument(
        "--fgo-local-prior-sigma-m",
        type=float,
        default=0.5,
        help="Endpoint prior sigma for the local FGO solve",
    )
    parser.add_argument(
        "--fgo-local-motion-sigma-m",
        type=float,
        default=1.0,
        help="Between-factor sigma for local FGO motion deltas",
    )
    parser.add_argument(
        "--fgo-local-dd-huber-k",
        type=float,
        default=1.5,
        help="Huber k for local FGO DD carrier factors",
    )
    parser.add_argument(
        "--fgo-local-pr-huber-k",
        type=float,
        default=1.5,
        help="Huber k for local FGO DD/undiff pseudorange factors",
    )
    parser.add_argument(
        "--fgo-local-dd-sigma-cycles",
        type=float,
        default=0.20,
        help="DD carrier sigma in cycles for local FGO",
    )
    parser.add_argument(
        "--fgo-local-pr-sigma-m",
        type=float,
        default=5.0,
        help="DD and undifferenced pseudorange sigma in meters for local FGO",
    )
    parser.add_argument(
        "--fgo-local-max-iterations",
        type=int,
        default=50,
        help="Maximum GTSAM LM iterations for local FGO",
    )
    parser.add_argument(
        "--fgo-local-lambda",
        action="store_true",
        help="Enable strict ratio-tested integer ambiguity fixing inside the local FGO window",
    )
    parser.add_argument(
        "--fgo-local-lambda-ratio-threshold",
        type=float,
        default=3.0,
        help="Minimum second/best ILS residual ratio for local FGO ambiguity fixes",
    )
    parser.add_argument(
        "--fgo-local-lambda-sigma-cycles",
        type=float,
        default=0.05,
        help="Carrier sigma in cycles for ratio-tested fixed DD ambiguity factors",
    )
    parser.add_argument(
        "--fgo-local-lambda-min-epochs",
        type=int,
        default=20,
        help="Minimum continuous epochs per DD pair before local FGO LAMBDA fixing",
    )
    parser.add_argument(
        "--fgo-local-motion-source",
        choices=("predict", "tdcp", "prefer_tdcp"),
        default="predict",
        help="Motion delta source for local FGO between factors",
    )
    parser.add_argument(
        "--fgo-local-tdcp-rms-max-m",
        type=float,
        default=3.0,
        help="Maximum TDCP postfit RMS for local FGO TDCP motion deltas",
    )
    parser.add_argument(
        "--fgo-local-tdcp-spp-max-diff-mps",
        type=float,
        default=6.0,
        help="Maximum TDCP/SPP finite-difference velocity disagreement for local FGO TDCP motion",
    )
    parser.add_argument(
        "--fgo-local-two-step",
        action="store_true",
        help="Run a coarse motion+undifferenced-PR local FGO stage before the DD local FGO stage",
    )
    parser.add_argument(
        "--fgo-local-stage1-prior-sigma-m",
        type=float,
        default=None,
        help="Endpoint prior sigma for the first coarse local FGO stage",
    )
    parser.add_argument(
        "--fgo-local-stage1-motion-sigma-m",
        type=float,
        default=None,
        help="Between-factor sigma for the first coarse local FGO stage",
    )
    parser.add_argument(
        "--fgo-local-stage1-pr-sigma-m",
        type=float,
        default=None,
        help="Undifferenced pseudorange sigma for the first coarse local FGO stage",
    )
    return parser
