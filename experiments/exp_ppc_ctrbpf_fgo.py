#!/usr/bin/env python3
# ruff: noqa: E402
"""CT-RBPF-FGO PPC port — Phase 0 scaffolding.

Implements the PPC-specific particle filter pipeline planned in
``internal_docs/ctrbpf_fgo_ppc_design.md``. Phase 0 only wires:

  - PPCDatasetLoader -> PR/Doppler/sat_velocity per epoch
  - WLS init -> ParticleFilterDevice predict/update loop
  - Optional RBPF velocity-KF Doppler update (--enable-rbpf-velocity-kf)
  - Optional WLS position-update soft constraint (--enable-position-update)
  - Honest score_ppc2024 over the FULL reference.csv denominator
    (missing PF epochs are filled with [0,0,0]; pattern from
     experiments/exp_ppc_libgnss_hybrid.py)

Subsequent phases will add region-aware gates, local FGO bridge, LAMBDA,
and B-spline post-smoothing per the design doc.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from exp_urbannav_baseline import run_wls  # noqa: E402
from gnss_gpu.io.nav_rinex import read_gps_klobuchar_from_nav_header
from gnss_gpu.io.ppc import PPCDatasetLoader
from gnss_gpu.ppc_score import score_ppc2024
from gnss_gpu.range_model import rotate_satellites_sagnac
from gnss_gpu.spp import _iono_klobuchar, _tropo_saastamoinen

RESULTS_DIR = _SCRIPT_DIR / "results"
_DEFAULT_DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")

_FULL_RUNS = (
    ("tokyo", "run1"),
    ("tokyo", "run2"),
    ("tokyo", "run3"),
    ("nagoya", "run1"),
    ("nagoya", "run2"),
    ("nagoya", "run3"),
)

_GPS_L1_WAVELENGTH_M = 0.19029367279836488

# Per-system PR weight scale: GPS=1, GLO=0.7, others=0.5. Empirically
# matches what gnss_solve does in libgnss++.
_SYSTEM_WEIGHT_SCALE = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.5, 4: 0.7}

_SYS_ID_TO_CHAR = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}

_WGS84_A = 6_378_137.0
_WGS84_E2 = 6.694379990141316e-3


@dataclass(frozen=True)
class CTRBPFConfig:
    n_particles: int = 50_000
    sigma_pr: float = 8.0
    pr_ess_guard_min_ratio: float = 0.0
    pr_ess_guard_max_iters: int = 12
    enable_pr_gmm: bool = False
    pr_gmm_statuses: tuple[int, ...] = (1, 3)
    pr_gmm_w_los: float = 0.7
    pr_gmm_mu_nlos_m: float = 15.0
    pr_gmm_sigma_nlos_m: float = 30.0
    pr_gmm_hybrid_loose_sigma_m: float = 5.0
    pr_gmm_clock_quantile: float = 0.35
    pr_weight_mode: str = "raw"
    pr_weight_ref_cn0: float = 45.0
    pr_weight_min: float = 0.25
    pr_weight_max: float = 1.5
    pr_systems: tuple[str, ...] = ("G", "E", "J")
    pr_min_elevation_deg: float = -90.0
    pr_atmosphere_model: str = "off"
    pr_atmosphere_scale: float = 1.0
    pr_atmosphere_extra_zenith_m: float = 0.0
    pr_slant_delay_zenith_m: float = 0.0
    pr_iono_alpha: tuple[float, ...] = ()
    pr_iono_beta: tuple[float, ...] = ()
    pr_prefit_gate_m: float = 0.0
    pr_prefit_gate_min_sats: int = 6
    pr_prefit_gate_keep_best: int = 0
    pr_prefit_ref: str = "pf"
    pr_prefit_per_system: bool = False
    pr_skip_statuses: tuple[int, ...] = ()
    defer_epoch_resample: bool = False
    enable_reservoir_stein: bool = False
    reservoir_stein_size: int = 2048
    reservoir_stein_elite_fraction: float = 0.25
    reservoir_stein_steps: int = 1
    reservoir_stein_step_size: float = 0.05
    reservoir_stein_repulsion_scale: float = 1.0
    reservoir_stein_guide_sigma_m: float = 2.0
    reservoir_stein_guide_sigma_cb_m: float = 50.0
    reservoir_stein_seed: int = 20260512
    # Phase 10i: RTK-diagnostic candidate as a PF pseudo-observation.
    # The v5 libgnss++ hybrid remains the passthrough floor. A relaxed RTK
    # candidate is injected into the PF only when its diagnostics pass the
    # residual gate, and only those epochs emit the PF estimate.
    enable_rtkdiag_pf_rescue: bool = False
    rtkdiag_candidate_sigma_m: float = 0.02
    rtkdiag_candidate_ratio_min: float = 1.5
    rtkdiag_candidate_residual_rms_max: float = 1.8
    rtkdiag_candidate_main_status5_residual_rms_max: float = 0.3
    # Pre-filter gated candidates to top-K by final_residual_rms (lowest is best)
    # BEFORE applying the selector ranking. 0 disables. Sweet spot K=3-7 (sim
    # showed K=5 captures +12pp upper bound; K=20 collapses to baseline).
    rtkdiag_candidate_rms_prefilter_k: int = 0
    # cluster_vote select_mode parameters: cluster gated candidates by
    # spatial proximity, pick largest cluster, within it pick lowest rms.
    # Designed to bypass the rms-only ranking trap on cluster-biased runs
    # (n/r2 etc.) where the wrong cluster has slightly lower rms than oracle.
    rtkdiag_candidate_cluster_vote_radius_m: float = 0.5
    # ranker select_mode: path to predictions CSV from train_selector_ranker.py
    # Columns: run_id, tow, label, p_pass. When select_mode == "ranker", pick
    # the gated candidate with highest p_pass per epoch. Supervised LightGBM
    # ranker trained on path-weighted oracle labels (Path F).
    rtkdiag_candidate_ranker_score_path: str = ""
    rtkdiag_candidate_ranker_stickiness: float = 0.0
    rtkdiag_candidate_bridge_enable: bool = False
    rtkdiag_candidate_bridge_max_s: float = 6.0
    rtkdiag_candidate_bridge_residual_rms_m: float = 0.5
    rtkdiag_candidate_bridge_anchor_mode: str = "last_emit"  # "last_emit" | "last_fix4"
    rtkdiag_candidate_bridge_fix4_min_ratio: float = 3.0
    rtkdiag_candidate_bridge_fix4_max_residual: float = 0.1
    rtkdiag_candidate_max_to_hybrid_m: float = 1.0
    rtkdiag_candidate_emit_max_diff_m: float = 0.4
    rtkdiag_candidate_recenter_max_shift_m: float = 10000.0
    rtkdiag_candidate_soft_top_k: int = 1
    rtkdiag_candidate_soft_weight_eps: float = 0.01
    rtkdiag_candidate_proposal_cloud: bool = False
    rtkdiag_candidate_proposal_spread_m: float = 0.25
    rtkdiag_candidate_select_mode: str = "residual"
    rtkdiag_candidate_emit_mode: str = "pf"
    rtkdiag_candidate_min_epoch: int = 0
    rtkdiag_candidate_require_any_diag_fields: tuple[str, ...] = ()
    rtkdiag_candidate_require_all_diag_fields: tuple[str, ...] = ()
    rtkdiag_candidate_min_diag_fields: tuple[tuple[str, float], ...] = ()
    rtkdiag_candidate_max_diag_fields: tuple[tuple[str, float], ...] = ()
    rtkdiag_candidate_fallback_mode: str = "hybrid"
    rtkdiag_candidate_fallback_max_wls_rms_m: float = 0.0
    rtkdiag_candidate_fallback_max_wls_pdop: float = 0.0
    rtkdiag_candidate_fallback_max_wls_to_pf_m: float = 0.0
    rtkdiag_candidate_fallback_max_hold_age_s: float = 5.0
    rtkdiag_candidate_local_ungate_windows: tuple[tuple[int, int, tuple[str, ...]], ...] = ()
    rtkdiag_candidate_local_ungate_tow_windows: tuple[tuple[float, float, tuple[str, ...]], ...] = ()
    rtkdiag_candidate_label_factors: tuple[tuple[str, float], ...] = ()
    rtkdiag_candidate_float_labels: tuple[str, ...] = ()
    rtkdiag_candidate_float_residual_rms_max: float = 0.0
    rtkdiag_candidate_float_abs_max: float = 0.0
    rtkdiag_candidate_float_min_sats: int = 0
    rtkdiag_candidate_status5_labels: tuple[str, ...] = ()
    rtkdiag_candidate_status5_tow_windows: tuple[tuple[float, float, tuple[str, ...]], ...] = ()
    rtkdiag_candidate_status5_max_dt_s: float = 0.0
    rtkdiag_candidate_status5_residual_rms_max: float = 0.0
    rtkdiag_candidate_status5_min_sats: int = 0
    sigma_pos: float = 2.0
    sigma_cb: float = 50.0
    spread_pos_init: float = 50.0
    spread_cb_init: float = 500.0
    sigma_doppler_mps: float = 0.5
    doppler_systems: tuple[str, ...] = ("G", "E", "J")
    doppler_prefit_gate_mps: float = 0.0
    doppler_prefit_gate_min_sats: int = 6
    velocity_init_sigma: float = 1.0
    velocity_process_noise: float = 1.0
    enable_rbpf_velocity_kf: bool = False
    enable_position_update: bool = False
    position_update_sigma_m: float = 30.0
    position_update_min_epoch: int = 0
    position_update_min_pr_sats: int = 0
    position_update_max_wls_rms_m: float = 0.0
    position_update_max_wls_pdop: float = 0.0
    position_update_max_wls_to_pf_m: float = 0.0
    enable_correct_clock_bias: bool = True
    enable_dd_carrier_afv: bool = False
    dd_sigma_cycles: float = 0.05
    dd_min_pairs: int = 4
    dd_min_pairs_update: int = 3
    dd_systems: tuple[str, ...] = ("G", "E", "J", "C")
    dd_base_interp: bool = False
    dd_min_elevation_deg: float = -90.0
    dd_min_snr: float = 0.0
    dd_keep_best: int = 0
    dd_pr_pair_residual_max_m: float = 0.0
    dd_pr_epoch_median_residual_max_m: float = 0.0
    dd_pr_gate_min_pairs: int = 3
    enable_dd_pr_ls_anchor: bool = False
    dd_pr_ls_anchor_min_pairs: int = 3
    dd_pr_ls_anchor_dd_sigma_m: float = 2.0
    dd_pr_ls_anchor_solve_prior_sigma_m: float = 100.0
    dd_pr_ls_anchor_prior_sigma_m: float = 3.0
    dd_pr_ls_anchor_max_shift_m: float = 100.0
    dd_pr_ls_anchor_max_postfit_rms_m: float = 5.0
    dd_pr_ls_anchor_statuses: tuple[int, ...] = (1, 3)
    dd_pr_ls_anchor_set_initial: bool = True
    dd_pr_ls_anchor_mode: str = "prior"
    # Phase 2: region-aware gate for the RBPF velocity-KF (Doppler) update.
    # ``None`` disables a gate. DD-pair gate uses 0 if no DD computer is
    # available, so it implicitly skips the KF update unless DD is wired.
    rbpf_kf_gate_min_dd_pairs: int | None = None
    rbpf_kf_gate_min_ess_ratio: float | None = None
    rbpf_kf_gate_max_spread_m: float | None = None
    rbpf_kf_gate_max_doppler_wls_rms_mps: float = 0.0
    rbpf_kf_gate_max_doppler_wls_speed_mps: float = 0.0
    # Phase 6: libgnss++ hybrid (50.91% baseline) position update. When
    # enabled, ``pf.position_update(hybrid_pos, sigma=hybrid_sigma_m)`` is
    # applied per epoch (if a hybrid sample exists for the rover TOW),
    # placing the cloud within 1 m of the hybrid baseline before the
    # Doppler-KF/DD-AFV updates so fractional DD residuals become meaningful.
    enable_hybrid_pu: bool = False
    hybrid_sigma_m: float = 1.0
    hybrid_recenter_max_shift_m: float = 0.0
    # Phase 7: derive a per-epoch velocity guide from hybrid pos finite
    # differences and feed it to ``pf.predict(velocity=...)`` so the cloud
    # can independently track the trajectory. Without this, the cloud is
    # stationary while the truth moves and hybrid PU at sigma=1m cannot
    # rescue particles that are several meters from the moving baseline.
    enable_hybrid_velocity_guide: bool = False
    # Phase 7 output mode. Default ``passthrough`` emits the hybrid pos
    # directly when available (Phase 6 MVP); ``pf`` emits the PF estimate
    # so the run can show whether DD-AFV / Doppler-KF correction beats
    # plain hybrid.
    hybrid_emit_pf_estimate: bool = False
    # Optional status gate for ``hybrid_emit_pf_estimate``. Empty means emit
    # PF at every hybrid epoch. For PPC, Status=4 anchors are often cm-class,
    # so GPU/PF diagnostics should usually emit PF only on weak statuses.
    hybrid_emit_pf_statuses: tuple[int, ...] = ()
    # Phase 4: post-process FGO + LAMBDA partial fix. After the PF loop
    # completes, slide a window over the trajectory and call
    # ``solve_local_fgo_with_lambda`` per window using cached DD-carrier
    # observations. Where LAMBDA accepts at least ``fgo_min_fixed_to_apply``
    # integer fixes via the ratio test, the FGO positions replace the PF /
    # hybrid output for that window. This is fixed-lag (not strictly
    # realtime) but the latency = window_size * dt is bounded.
    enable_fgo_lambda: bool = False
    fgo_window_size: int = 30
    fgo_window_stride: int = 15
    fgo_lambda_ratio: float = 3.0
    fgo_lambda_min_epochs: int = 10
    fgo_lambda_max_epoch_gap: int = 6
    fgo_min_fixed_to_apply: int = 3
    fgo_prior_sigma_m: float = 0.5
    fgo_dd_sigma_cycles: float = 0.20
    fgo_dd_pr_sigma_m: float = 5.0
    # C2: per-epoch gate. If non-empty, FGO output is only written back to
    # ``positions[i]`` when the hybrid Status at ``times[i]`` is one of these
    # values; epochs with other Status values keep the hybrid passthrough.
    # Empty tuple = apply Phase 4 to every epoch (legacy behavior).
    # Default ``(1, 3)`` skips Status=4 (cm-class libgnss++ output) and only
    # rewrites Status=1/3 (m-class).
    fgo_apply_hybrid_statuses: tuple[int, ...] = (1, 3)
    # D2: per-epoch prior sigmas inside the FGO solve. Status values that
    # are NOT in ``fgo_apply_hybrid_statuses`` (e.g. Status=4 = cm-class
    # hybrid) get the tight ``fgo_anchor_sigma_m`` so the FGO treats them
    # as cm-class anchors. Status values that ARE in the apply set
    # (m-class hybrid) get ``fgo_loose_sigma_m`` so DD carrier + DD PR
    # drive the local solve. Set ``fgo_anchor_sigma_m`` to a non-positive
    # value to disable per-epoch priors entirely (fall back to legacy
    # endpoint-only priors at ``fgo_prior_sigma_m``).
    fgo_anchor_sigma_m: float = 0.05
    fgo_loose_sigma_m: float = 5.0
    # Continuous-time trajectory prior for the FGO window. This fits a cubic
    # smoothing spline to the PF/anchor trajectory and feeds its inter-epoch
    # displacement as the FGO motion factor. It is a post-loop CT layer, not
    # yet an in-loop spline state.
    enable_ct_spline_motion_prior: bool = False
    ct_spline_smoothing_m: float = 0.5
    ct_motion_sigma_m: float = 0.25
    ct_motion_min_epochs: int = 6
    # D2b "minimum-correction gate": skip rewrites where FGO output is
    # within ``fgo_min_correction_m`` of the hybrid passthrough. Empirical
    # evidence (tokyo/run2 first 2000 ep) showed Phase 4 nudges many
    # cm-class hybrid passes (~5cm) across the 0.5m PPC threshold; small
    # rewrites cost more pass than they recover. Set to 0.0 to disable.
    fgo_min_correction_m: float = 0.5
    fgo_apply_fixed_epochs_only: bool = True
    # Phase 8: TDCP-anchored hybrid smoother. After the PF loop, run a
    # per-coordinate forward+backward Kalman smoother over the trajectory,
    # using the rover-side TDCP velocity as the motion model and the hybrid
    # passthrough as the observation. Each epoch's observation sigma is
    # derived from its hybrid Status (anchor sigma for "good" Status, loose
    # sigma for "rewritable" Status, huge sigma when hybrid is missing).
    enable_tdcp_smoother: bool = False
    tdcp_sigma_mps: float = 0.05
    tdcp_postfit_max_m: float = 1.0
    tdcp_min_sats: int = 5
    tdcp_obs_anchor_sigma_m: float = 0.05
    tdcp_obs_loose_sigma_m: float = 5.0
    tdcp_obs_missing_sigma_m: float = 1000.0
    enable_low_sat_bridge: bool = False
    low_sat_bridge_min_pr_sats: int = 11
    low_sat_bridge_min_span_epochs: int = 3
    low_sat_bridge_max_span_epochs: int = 25
    low_sat_bridge_max_gap_s: float = 10.0
    low_sat_bridge_startup_max_wls_pdop: float = 0.5
    low_sat_bridge_startup_min_epochs: int = 2
    low_sat_bridge_startup_max_epochs: int = 10
    # Phase 9a: ZUPT (zero-velocity update) using PPC IMU. Per rover epoch
    # we look at the specific-force / angular-rate norms over the IMU
    # samples that fall in [t_i, t_{i+1}). When the accel norm is close
    # to gravity AND the gyro norm is small, the vehicle is stopped and
    # the rover position must equal the last position. We use this to
    # damp hybrid jitter on Status=1/3 epochs that sit inside a stop.
    enable_zupt: bool = False
    zupt_acc_norm_low_mps2: float = 9.78
    zupt_acc_norm_high_mps2: float = 9.85
    zupt_gyro_norm_max_dps: float = 0.3
    # Number of consecutive static epochs required (including the current
    # one) before ZUPT actually rewrites. Single-epoch static glitches are
    # discarded; 5 epochs at 5 Hz = 1 s of confirmed stop.
    zupt_min_consecutive: int = 5
    # Maximum |base - anchor| disagreement [m] tolerated before ZUPT
    # rewrites. If the current passthrough already differs from the static
    # anchor by more than this, the anchor is probably stale and we skip
    # the rewrite (we cannot tell at runtime whether base drifted or the
    # vehicle moved).
    zupt_max_anchor_drift_m: float = 0.5
    # Only ZUPT-rewrite epochs whose hybrid Status is in this set; the
    # Status=4 cm-class epochs stay fixed to their hybrid value.
    zupt_apply_hybrid_statuses: tuple[int, ...] = (1, 3)
    # Phase 9b: tight-coupled IMU. Unlike Phase 9a (post-process ZUPT) or
    # Phase 4/8 (post-process FGO/TDCP), IMU evidence enters the PF *during*
    # the loop as a per-particle position pseudo-observation
    # (``pf.position_update(imu_predicted_pos, sigma=imu_sigma)``). The
    # IMU prediction is derived from a sliding pre-integration window
    # anchored at the most recent Status=4 (cm-class) hybrid epoch:
    #   - body-frame accel is rotated into ENU using a yaw derived from the
    #     anchor's recent velocity (course-over-ground, pitch/roll = 0),
    #   - gravity (9.81 m/s^2) is removed from the body-z axis,
    #   - acceleration is double-integrated to give Δp_imu since anchor.
    # Output emission is gated on hybrid Status: Status=4 epochs always emit
    # the hybrid passthrough (cm-class anchor); Status in
    # ``imu_tc_emit_pf_hybrid_statuses`` (default (1, 3)) emit the PF
    # estimate so the IMU likelihood actually surfaces in the PPC score.
    enable_imu_tc: bool = False
    imu_tc_emit_pf_hybrid_statuses: tuple[int, ...] = (1, 3)
    # Position pseudo-observation sigma at t=0s after the anchor reset.
    imu_tc_pos_sigma_base_m: float = 0.5
    # Sigma growth per second of dead-reckoning (linear; reflects
    # accel-bias drift). At 5s elapsed, default sigma = 0.5 + 5*0.5 = 3.0 m.
    imu_tc_pos_sigma_per_s: float = 0.5
    # Maximum dead-reckoning duration [s]. Past this, IMU drift dominates
    # and we skip the PU + emission switch (epoch falls back to hybrid).
    imu_tc_max_dr_seconds: float = 5.0
    # If |IMU-predicted pos - hybrid pos| exceeds this when both are
    # available, the IMU prediction is presumed wrong (yaw drift, axis
    # error) and we skip the PU + emission switch. Conservative default
    # 30 m: PPC NLOS hybrid jumps can exceed 20 m, so this is loose.
    imu_tc_max_disagreement_m: float = 30.0
    # When the emission switch fires we still bound the PF estimate to the
    # hybrid ± this distance; if the PF wandered farther we keep hybrid.
    # Protects against PF cloud collapse drift in long NLOS gaps.
    imu_tc_emit_max_diff_m: float = 20.0
    # Hybrid PU sigma override on Status values that ARE in
    # ``imu_tc_emit_pf_hybrid_statuses``. When set > 0, the hybrid pos
    # update on those (m-class) epochs uses this sigma instead of
    # ``hybrid_sigma_m`` (default 1.0). Looser hybrid PU lets the IMU
    # pre-integration drive the cloud through NLOS without being clamped
    # to the m-class hybrid baseline. Set to 0 to keep the global sigma.
    imu_tc_hybrid_loose_sigma_m: float = 5.0
    # Static-detection thresholds for the *anchor*: when the vehicle is
    # IMU-static at a Status=4 epoch, we set the anchor velocity to zero
    # (instead of a noisy hybrid finite difference). Reuses the Phase 9a
    # ZUPT thresholds so a single `--zupt-*` knob set covers both.
    imu_tc_anchor_static_acc_low_mps2: float = 9.6
    imu_tc_anchor_static_acc_high_mps2: float = 9.95
    imu_tc_anchor_static_gyro_max_dps: float = 1.5
    # Phase 9c: full 15-state INS-GNSS EKF. This coexists with Phase 9b
    # but is enabled through separate method labels so the yaw-only
    # pre-integration baseline remains reproducible.
    enable_ins_tc: bool = False
    ins_tc_emit_pf_hybrid_statuses: tuple[int, ...] = (1, 3)
    ins_tc_obs_status_4_sigma_m: float = 0.05
    ins_tc_obs_status_3_sigma_m: float = 0.0
    ins_tc_max_dr_seconds: float = 10.0
    ins_tc_max_disagreement_m: float = 30.0
    ins_tc_emit_max_diff_m: float = 1.0
    ins_tc_pf_pu_floor_sigma_m: float = 0.1
    ins_tc_pf_pu_ceiling_sigma_m: float = 5.0
    ins_tc_use_particle_imu_predict: bool = True
    ins_tc_particle_imu_sigma_pos_m: float = 0.02
    ins_tc_particle_imu_sigma_acc_mps2: float = 0.10
    ins_tc_particle_imu_sigma_gyro_rps: float = 0.005
    ins_tc_particle_imu_acc_bias_rw: float = 1.0e-4
    ins_tc_particle_imu_gyro_bias_rw: float = 1.0e-5
    ins_tc_particle_imu_att_spread_rad: float = math.radians(2.0)
    ins_tc_particle_imu_acc_bias_spread: float = 0.05
    ins_tc_particle_imu_gyro_bias_spread_rps: float = math.radians(0.1)
    ins_tc_particle_imu_velocity_spread_mps: float = 0.5
    ins_tc_recenter_status4: bool = False
    ins_tc_recenter_max_shift_m: float = 5000.0
    ins_tc_use_motion_predict: bool = True
    ins_tc_predict_sigma_pos_m: float = 0.2
    ins_tc_predict_velocity_alpha: float = 1.0
    ins_tc_align_acc_low: float = 9.6
    ins_tc_align_acc_high: float = 9.95
    ins_tc_align_gyro_max_dps: float = 1.5
    ins_tc_align_min_samples: int = 50
    ins_tc_yaw_init_min_speed_mps: float = 1.0
    # GNSS-quality based gate on the ins_tc emit decision. When enabled, the
    # rolling fix rate over the previous ``ins_tc_quality_gate_window_epochs``
    # epochs is computed; when fix rate >= ``ins_tc_quality_gate_max_fix_rate``
    # the ins_tc PF-emit is suppressed (defer to GNSS / hybrid). Designed to
    # reduce ins_tc regression on high-baseline runs (tokyo/run2 -7.06pp,
    # nagoya/run1 -2.59pp) where GNSS Fix solutions are already accurate.
    ins_tc_quality_gate_enabled: bool = False
    ins_tc_quality_gate_window_epochs: int = 30
    ins_tc_quality_gate_max_fix_rate: float = 0.5
    # When true (and ins_tc_quality_gate_enabled), the rolling-fix-rate gate
    # also suppresses ins_tc PF position_update (PU), not just emit. Default
    # off so the existing emit-only gate behaviour is preserved.
    ins_tc_quality_gate_pu_skip: bool = False
    systems: tuple[str, ...] = ("G", "R", "E", "C", "J")
    method_label: str = "PF-PR"


@dataclass
class _PPCMeasurement:
    """Light-weight measurement object compatible with DDCarrierComputer.

    Only the attributes consumed by ``compute_dd`` (in rover_obs_path mode)
    are populated: ``system_id``, ``prn``, ``satellite_ecef``, ``elevation``,
    ``snr``. ``carrier_phase`` is read directly from the rover RINEX, not
    from this object.
    """

    system_id: int
    prn: int
    satellite_ecef: np.ndarray
    elevation: float
    snr: float


def _ecef_to_llh(x: float, y: float, z: float) -> tuple[float, float, float]:
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(6):
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - N
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * N / max(N + h, 1.0)))
    sin_lat = math.sin(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    alt = p / max(math.cos(lat), 1e-12) - N
    return lat, lon, alt


def _elevation_rad(rx_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    lat, lon, _ = _ecef_to_llh(float(rx_ecef[0]), float(rx_ecef[1]), float(rx_ecef[2]))
    dx = sat_ecef - rx_ecef
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    e = -sin_lon * dx[0] + cos_lon * dx[1]
    n = -sin_lat * cos_lon * dx[0] - sin_lat * sin_lon * dx[1] + cos_lat * dx[2]
    u = cos_lat * cos_lon * dx[0] + cos_lat * sin_lon * dx[1] + sin_lat * dx[2]
    return math.atan2(float(u), math.sqrt(float(e) * float(e) + float(n) * float(n)))


def _build_dd_measurements(
    sat_ecef: np.ndarray,
    system_ids: np.ndarray,
    sat_id_strs: list[str],
    weights: np.ndarray,
    rover_pos: np.ndarray,
    allowed_systems: tuple[str, ...],
    min_elevation_deg: float = -90.0,
    min_snr: float = 0.0,
    keep_best: int = 0,
) -> list[_PPCMeasurement]:
    """Build measurement objects from per-epoch arrays for compute_dd()."""
    out: list[_PPCMeasurement] = []
    sids = np.asarray(system_ids, dtype=np.int32)
    min_elev_rad = math.radians(float(min_elevation_deg))
    for k in range(int(sat_ecef.shape[0])):
        sys_char = _SYS_ID_TO_CHAR.get(int(sids[k]))
        if sys_char is None or sys_char not in allowed_systems:
            continue
        snr = float(weights[k]) if k < len(weights) else 1.0
        if np.isfinite(float(min_snr)) and snr < float(min_snr):
            continue
        sat_id_str = sat_id_strs[k] if k < len(sat_id_strs) else ""
        prn_str = sat_id_str[1:].lstrip("0") if sat_id_str else ""
        try:
            prn = int(prn_str) if prn_str else 0
        except ValueError:
            prn = 0
        if prn <= 0:
            continue
        sat_pos = np.asarray(sat_ecef[k], dtype=np.float64)
        if sat_pos.size != 3 or not np.all(np.isfinite(sat_pos)):
            continue
        elev = _elevation_rad(rover_pos, sat_pos)
        if np.isfinite(min_elev_rad) and elev < min_elev_rad:
            continue
        out.append(
            _PPCMeasurement(
                system_id=int(sids[k]),
                prn=prn,
                satellite_ecef=sat_pos,
                elevation=float(elev),
                snr=snr,
            )
        )
    if int(keep_best) > 0 and len(out) > int(keep_best):
        out = sorted(
            out,
            key=lambda m: (
                float(m.elevation),
                float(m.snr),
            ),
            reverse=True,
        )[: int(keep_best)]
    return out


@dataclass
class _DDStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    pairs_total: int = 0


@dataclass
class _PRObsStats:
    epochs_gaussian: int = 0
    epochs_gmm: int = 0
    deferred_resample_epochs: int = 0
    prefit_epochs: int = 0
    prefit_sats_kept: int = 0
    prefit_sats_dropped: int = 0
    epochs_skipped: int = 0


@dataclass
class _ReservoirSteinStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    reservoir_size_sum: int = 0
    ess_before_sum: float = 0.0
    bandwidth_sum: float = 0.0


@dataclass
class _RBPFGateStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    skipped_min_dd_pairs: int = 0
    skipped_min_ess_ratio: int = 0
    skipped_max_spread: int = 0
    skipped_doppler_wls_rms: int = 0
    skipped_doppler_wls_speed: int = 0


@dataclass
class _HybridStats:
    epochs_attempted: int = 0
    epochs_applied: int = 0
    epochs_lookup_missing: int = 0
    recenter_applied: int = 0
    recenter_skipped: int = 0
    recenter_shift_sum_m: float = 0.0


@dataclass
class _RTKDiagPFStats:
    epochs_evaluated: int = 0
    gate_pass: int = 0
    candidate_missing: int = 0
    candidate_options_total: int = 0
    pu_applied: int = 0
    recenter_applied: int = 0
    recenter_skipped: int = 0
    emit_pf_estimate: int = 0
    emit_candidate: int = 0
    emit_skipped_pf_drift: int = 0
    skipped_min_epoch: int = 0
    skipped_diag_policy: int = 0
    skipped_hybrid_distance: int = 0
    fallback_pf: int = 0
    fallback_wls: int = 0
    fallback_hybrid: int = 0
    fallback_last_good: int = 0
    selected_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class _FGOStats:
    windows_attempted: int = 0
    windows_solved: int = 0
    windows_applied: int = 0
    n_fixed_total: int = 0
    n_fixed_observations_total: int = 0
    epochs_replaced: int = 0
    lambda_tracks_total: int = 0
    lambda_segments_total: int = 0
    lambda_candidates_total: int = 0
    lambda_ratio_rejected_total: int = 0
    lambda_best_ratio: float = 0.0
    lambda_ratio_median_sum: float = 0.0
    lambda_ratio_p90_sum: float = 0.0
    lambda_segment_n_epochs_median_sum: float = 0.0
    lambda_segment_n_epochs_max: int = 0
    lambda_segment_variance_median_sum: float = 0.0
    lambda_segment_abs_frac_median_sum: float = 0.0
    lambda_segment_abs_frac_p90_sum: float = 0.0
    postfit_fixed_count_total: int = 0
    postfit_fixed_abs_cycles_median_sum: float = 0.0
    postfit_fixed_abs_cycles_p90_sum: float = 0.0
    postfit_float_count_total: int = 0
    postfit_float_afv_abs_cycles_median_sum: float = 0.0
    postfit_float_afv_abs_cycles_p90_sum: float = 0.0
    postfit_dd_pr_count_total: int = 0
    postfit_dd_pr_abs_m_median_sum: float = 0.0
    postfit_dd_pr_abs_m_p90_sum: float = 0.0
    lambda_diag_windows: int = 0
    dd_pr_ls_anchor_attempted: int = 0
    dd_pr_ls_anchor_accepted: int = 0
    dd_pr_ls_anchor_status_skipped: int = 0
    dd_pr_ls_anchor_rejected_postfit: int = 0
    dd_pr_ls_anchor_rejected_solve: int = 0
    dd_pr_ls_anchor_shift_sum_m: float = 0.0
    dd_pr_ls_anchor_postfit_sum_m: float = 0.0
    dd_pr_ls_anchor_gt_count: int = 0
    dd_pr_ls_anchor_gt_error_sum_m: float = 0.0
    dd_pr_ls_anchor_seed_error_sum_m: float = 0.0
    dd_pr_ls_anchor_improved: int = 0
    dd_pr_ls_anchor_pass_05m: int = 0
    dd_pr_ls_anchor_pass_5m: int = 0


@dataclass
class _TDCPSmootherStats:
    pairs_attempted: int = 0
    pairs_accepted: int = 0
    pairs_rejected_min_sats: int = 0
    pairs_rejected_postfit: int = 0


@dataclass
class _LowSatBridgeStats:
    spans_applied: int = 0
    epochs_rewritten: int = 0
    startup_spans_applied: int = 0
    startup_epochs_rewritten: int = 0


@dataclass
class _ZUPTStats:
    epochs_evaluated: int = 0
    epochs_static: int = 0
    epochs_rewritten: int = 0
    epochs_no_imu: int = 0


@dataclass
class _IMUTCStats:
    """Per-run counters for Phase 9b tight-coupled IMU."""

    epochs_evaluated: int = 0  # rover epochs reached after the first anchor
    anchor_resets: int = 0  # Status=4 epochs that re-set the anchor
    anchor_resets_static: int = 0  # subset where the anchor was IMU-static
    pu_applied: int = 0  # epochs where pf.position_update(imu_pred) fired
    pu_skipped_no_imu: int = 0
    pu_skipped_no_anchor: int = 0
    pu_skipped_dr_too_long: int = 0
    pu_skipped_disagreement: int = 0
    emit_pf_estimate: int = 0  # epochs where output[i] = pf.estimate (vs hybrid)
    emit_skipped_pf_drift: int = 0  # PF wandered too far from hybrid; kept hybrid
    avg_dr_seconds: float = 0.0
    _dr_seconds_sum: float = 0.0
    _dr_seconds_n: int = 0

    def record_dr_seconds(self, dr: float) -> None:
        if not (dr is None or dr != dr):  # noqa: SIM114 — NaN safe
            self._dr_seconds_sum += float(dr)
            self._dr_seconds_n += 1
            self.avg_dr_seconds = self._dr_seconds_sum / max(self._dr_seconds_n, 1)


@dataclass
class _INSTCStats:
    """Per-run counters for Phase 9c INS-GNSS tight coupling."""

    aligned_at_epoch: int = -1
    yaw_initialized_at_epoch: int = -1
    epochs_evaluated: int = 0
    pu_applied: int = 0
    pu_skipped_not_aligned: int = 0
    pu_skipped_no_yaw: int = 0
    pu_skipped_dr_too_long: int = 0
    pu_skipped_disagreement: int = 0
    particle_imu_initialized: int = 0
    particle_imu_predict_used: int = 0
    recenter_applied: int = 0
    recenter_skipped: int = 0
    recenter_shift_sum_m: float = 0.0
    motion_predict_used: int = 0
    obs_status_4_used: int = 0
    obs_status_3_used: int = 0
    emit_pf_estimate: int = 0
    emit_skipped_pf_drift: int = 0
    final_acc_bias_norm: float = 0.0
    final_gyro_bias_norm_dps: float = 0.0
    final_pos_sigma_m: float = 0.0


def _build_imu_per_epoch_stats(
    imu: dict[str, np.ndarray],
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per rover-epoch accel norm, gyro norm and sample count.

    Each rover epoch ``i`` aggregates the IMU samples whose timestamp falls in
    ``[times[i] - dt/2, times[i] + dt/2)`` where ``dt`` is the median rover
    epoch spacing. Epochs with zero IMU samples have NaN stats.
    """
    n = int(times.size)
    acc_norm_per = np.full(n, np.nan, dtype=np.float64)
    gyro_norm_per = np.full(n, np.nan, dtype=np.float64)
    n_samples = np.zeros(n, dtype=np.int32)
    if not imu or "time" not in imu or imu["time"].size == 0:
        return acc_norm_per, gyro_norm_per, n_samples

    t_imu = np.asarray(imu["time"], dtype=np.float64)
    ax = np.asarray(imu.get("acc_x", np.full_like(t_imu, np.nan)), dtype=np.float64)
    ay = np.asarray(imu.get("acc_y", np.full_like(t_imu, np.nan)), dtype=np.float64)
    az = np.asarray(imu.get("acc_z", np.full_like(t_imu, np.nan)), dtype=np.float64)
    gx = np.asarray(imu.get("gyro_x", np.full_like(t_imu, np.nan)), dtype=np.float64)
    gy = np.asarray(imu.get("gyro_y", np.full_like(t_imu, np.nan)), dtype=np.float64)
    gz = np.asarray(imu.get("gyro_z", np.full_like(t_imu, np.nan)), dtype=np.float64)
    acc_norm_imu = np.sqrt(ax * ax + ay * ay + az * az)
    gyro_norm_imu = np.sqrt(gx * gx + gy * gy + gz * gz)

    if n >= 2:
        dt_med = float(np.median(np.diff(times)))
    else:
        dt_med = 0.2
    half = 0.5 * dt_med

    for i in range(n):
        t = float(times[i])
        lo = t - half
        hi = t + half
        # Binary search for sample range.
        i0 = int(np.searchsorted(t_imu, lo, side="left"))
        i1 = int(np.searchsorted(t_imu, hi, side="left"))
        if i1 <= i0:
            continue
        a_slice = acc_norm_imu[i0:i1]
        g_slice = gyro_norm_imu[i0:i1]
        finite_a = np.isfinite(a_slice)
        finite_g = np.isfinite(g_slice)
        if finite_a.any():
            acc_norm_per[i] = float(np.mean(a_slice[finite_a]))
        if finite_g.any():
            gyro_norm_per[i] = float(np.mean(g_slice[finite_g]))
        n_samples[i] = int(min(int(np.sum(finite_a)), int(np.sum(finite_g))))
    return acc_norm_per, gyro_norm_per, n_samples


def _apply_zupt(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    imu: dict[str, np.ndarray] | None,
    config: CTRBPFConfig,
    hybrid_status: dict[float, int] | None,
    stats: _ZUPTStats,
) -> np.ndarray:
    """Hold position constant on rover epochs where IMU reports static.

    ``positions[i] := positions[i-1]`` when:
      - ``imu`` provides finite accel / gyro norms for epoch ``i``,
      - the accel norm is in ``[zupt_acc_norm_low_mps2, zupt_acc_norm_high_mps2]``,
      - the gyro norm is below ``zupt_gyro_norm_max_dps``,
      - the hybrid Status at this epoch is in ``zupt_apply_hybrid_statuses``
        (so cm-class Status=4 anchors are never moved).

    ``positions`` is treated as the hybrid passthrough output; this
    function returns a modified copy.
    """
    if imu is None or "time" not in imu or imu["time"].size == 0:
        return positions

    n = int(positions.shape[0])
    if n < 2:
        return positions

    acc_per, gyro_per, _ = _build_imu_per_epoch_stats(imu, times)

    rewrite_set = set(int(s) for s in config.zupt_apply_hybrid_statuses)
    out = np.array(positions, dtype=np.float64, copy=True)

    # ZUPT can only copy from a past anchor position if the vehicle has
    # been continuously static since that anchor. Track:
    #   - last_static_anchor: pos at the most recent Status=4 epoch where
    #     the vehicle was also IMU-static
    #   - static_streak: consecutive static epochs since last motion
    last_static_anchor: np.ndarray | None = None
    static_streak: int = 0
    min_streak = max(1, int(config.zupt_min_consecutive))
    max_drift = float(config.zupt_max_anchor_drift_m)

    for i in range(n):
        anchor_here = False
        if hybrid_status is not None and config.zupt_apply_hybrid_statuses:
            t_key = round(float(times[i]), 1)
            st = hybrid_status.get(t_key)
            if st is not None and st not in rewrite_set:
                anchor_here = True

        a = acc_per[i] if i < acc_per.size else float("nan")
        g = gyro_per[i] if i < gyro_per.size else float("nan")
        is_static = (
            np.isfinite(a)
            and np.isfinite(g)
            and float(config.zupt_acc_norm_low_mps2) <= a <= float(config.zupt_acc_norm_high_mps2)
            and g <= float(config.zupt_gyro_norm_max_dps)
        )

        if i > 0:
            stats.epochs_evaluated += 1
            if not (np.isfinite(a) and np.isfinite(g)):
                stats.epochs_no_imu += 1
            elif is_static:
                stats.epochs_static += 1

        if not is_static:
            last_static_anchor = None
            static_streak = 0
            continue

        static_streak += 1

        if anchor_here:
            last_static_anchor = np.asarray(out[i], dtype=np.float64).copy()
            continue

        if (
            last_static_anchor is None
            or static_streak < min_streak
            or i == 0
        ):
            continue
        # Skip the rewrite if base drifted away from the anchor by more
        # than the allowed tolerance -- the anchor is probably stale.
        drift = float(np.linalg.norm(out[i] - last_static_anchor))
        if drift > max_drift:
            continue
        out[i] = last_static_anchor
        stats.epochs_rewritten += 1

    return out


# ---------------------------------------------------------------------------
# Phase 9b: tight-coupled IMU helpers
# ---------------------------------------------------------------------------


def _ecef_to_enu_rotation(lat: float, lon: float) -> np.ndarray:
    """3x3 rotation matrix that maps an ECEF delta to local ENU at (lat, lon)."""
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so,             co,            0.0],
            [-sl * co, -sl * so,  cl],
            [ cl * co,  cl * so,  sl],
        ],
        dtype=np.float64,
    )


def _ecef_velocity_to_enu(pos_ecef: np.ndarray, vel_ecef: np.ndarray) -> np.ndarray:
    lat, lon, _ = _ecef_to_llh(float(pos_ecef[0]), float(pos_ecef[1]), float(pos_ecef[2]))
    R = _ecef_to_enu_rotation(lat, lon)
    return R @ np.asarray(vel_ecef, dtype=np.float64)


def _enu_delta_to_ecef(pos_ecef: np.ndarray, enu: np.ndarray) -> np.ndarray:
    lat, lon, _ = _ecef_to_llh(float(pos_ecef[0]), float(pos_ecef[1]), float(pos_ecef[2]))
    R = _ecef_to_enu_rotation(lat, lon)
    return R.T @ np.asarray(enu, dtype=np.float64)


def _ecef_to_enu_at_origin(
    pos_ecef: np.ndarray,
    origin_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R @ (
        np.asarray(pos_ecef, dtype=np.float64).reshape(3)
        - np.asarray(origin_ecef, dtype=np.float64).reshape(3)
    )


def _ecef_velocity_to_enu_at_origin(
    vel_ecef: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R @ np.asarray(vel_ecef, dtype=np.float64).reshape(3)


def _enu_velocity_to_ecef_at_origin(
    vel_enu: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R.T @ np.asarray(vel_enu, dtype=np.float64).reshape(3)


def _gravity_ecef_at_origin(origin_lat: float, origin_lon: float) -> np.ndarray:
    R = _ecef_to_enu_rotation(float(origin_lat), float(origin_lon))
    return R.T @ np.array([0.0, 0.0, -9.81], dtype=np.float64)


def _slice_imu_samples(
    imu: dict[str, np.ndarray],
    imu_t: np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Return INS EKF IMU rows [t, ax, ay, az, gx, gy, gz] in [start, end]."""
    if not math.isfinite(t_start) or not math.isfinite(t_end) or t_end < t_start:
        return np.empty((0, 7), dtype=np.float64)
    i0 = int(np.searchsorted(imu_t, float(t_start), side="left"))
    i1 = int(np.searchsorted(imu_t, float(t_end), side="right"))
    if i1 <= i0:
        return np.empty((0, 7), dtype=np.float64)
    return np.column_stack(
        (
            np.asarray(imu_t[i0:i1], dtype=np.float64),
            np.asarray(imu["acc_x"][i0:i1], dtype=np.float64),
            np.asarray(imu["acc_y"][i0:i1], dtype=np.float64),
            np.asarray(imu["acc_z"][i0:i1], dtype=np.float64),
            np.asarray(imu["gyro_x"][i0:i1], dtype=np.float64),
            np.asarray(imu["gyro_y"][i0:i1], dtype=np.float64),
            np.asarray(imu["gyro_z"][i0:i1], dtype=np.float64),
        )
    )


@dataclass
class _IMUAnchor:
    """State of the IMU dead-reckoning anchor.

    The anchor is the most recent rover epoch where the position is trusted
    (Status=4 hybrid, or first epoch). Pre-integration accumulates body-frame
    accelerometer samples since this time and projects them into ENU using
    the anchor's heading (yaw). The first ``valid_yaw`` is set as soon as we
    have a usable course-over-ground estimate; until then the IMU PU is
    skipped.
    """

    pos_ecef: np.ndarray
    time: float
    yaw_enu_rad: float  # CCW from East in ENU; vehicle forward = +yaw direction
    valid_yaw: bool
    velocity_enu: np.ndarray  # (east, north, up) m/s, used as initial dr velocity


def _heading_from_velocity_enu(vel_enu: np.ndarray, min_speed_mps: float = 0.7) -> tuple[float, bool]:
    """Yaw (CCW from East) derived from ENU velocity.

    Returns (yaw_rad, valid). ``valid`` is False when the planar speed is
    below ``min_speed_mps`` (heading cannot be reliably inferred).
    """
    east = float(vel_enu[0]) if vel_enu.size > 0 else 0.0
    north = float(vel_enu[1]) if vel_enu.size > 1 else 0.0
    speed = math.hypot(east, north)
    if not math.isfinite(speed) or speed < float(min_speed_mps):
        return 0.0, False
    return math.atan2(north, east), True


def _build_imu_segment_index(imu: dict[str, np.ndarray] | None) -> np.ndarray | None:
    """Sorted IMU timestamp array used for binary search; ``None`` if no IMU."""
    if imu is None or "time" not in imu:
        return None
    t = np.asarray(imu["time"], dtype=np.float64)
    if t.size == 0:
        return None
    return t


def _integrate_imu_between(
    imu: dict[str, np.ndarray],
    imu_t: np.ndarray,
    t_start: float,
    t_end: float,
    yaw_enu_rad: float,
    initial_velocity_enu: np.ndarray,
    *,
    gravity_mps2: float = 9.81,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pre-integrate body-frame IMU between two rover epochs.

    Returns ``(delta_pos_enu, end_velocity_enu, n_samples)``. The body frame
    convention assumed: x=forward, y=left, z=up (right-handed, +z aligned
    with gravity reaction). Pitch and roll are taken as zero so accel_z is
    just the vertical specific force. Yaw rotates the horizontal body axes
    into ENU (east, north).

    Returns zero deltas (and propagated initial velocity * dt) when no
    samples fall in [t_start, t_end).
    """
    dt_total = float(t_end - t_start)
    if not math.isfinite(dt_total) or dt_total <= 0.0:
        return np.zeros(3, dtype=np.float64), np.asarray(initial_velocity_enu, dtype=np.float64).copy(), 0

    i0 = int(np.searchsorted(imu_t, t_start, side="left"))
    i1 = int(np.searchsorted(imu_t, t_end, side="left"))
    if i1 <= i0:
        # No IMU samples in window -- propagate at constant velocity.
        v = np.asarray(initial_velocity_enu, dtype=np.float64).copy()
        dp = v * dt_total
        return dp, v, 0

    ax = np.asarray(imu["acc_x"][i0:i1], dtype=np.float64)
    ay = np.asarray(imu["acc_y"][i0:i1], dtype=np.float64)
    az = np.asarray(imu["acc_z"][i0:i1], dtype=np.float64)
    ts = np.asarray(imu_t[i0:i1], dtype=np.float64)

    # Replace NaNs with zeros (rare, treat as no acceleration sample).
    ax = np.where(np.isfinite(ax), ax, 0.0)
    ay = np.where(np.isfinite(ay), ay, 0.0)
    az = np.where(np.isfinite(az), az, 0.0)

    # Body -> ENU (yaw only; pitch/roll = 0). x=forward, y=left, z=up.
    cy = math.cos(yaw_enu_rad)
    sy = math.sin(yaw_enu_rad)
    east_acc = ax * cy - ay * sy
    north_acc = ax * sy + ay * cy
    up_acc = az - float(gravity_mps2)

    # Per-sample dt: use the gaps between successive timestamps, plus an
    # opening dt from t_start to ts[0] and a closing dt from ts[-1] to t_end.
    n = ts.size
    dts = np.empty(n, dtype=np.float64)
    if n == 1:
        dts[0] = dt_total
    else:
        # Each sample is "valid" for the time slice ending at the next sample.
        diffs = np.diff(ts)
        dts[:-1] = diffs
        # Last sample carries from ts[-1] to t_end.
        dts[-1] = max(0.0, t_end - float(ts[-1]))
        # Adjust the first sample to also cover t_start..ts[0] so we don't
        # lose the opening fraction.
        opening = max(0.0, float(ts[0]) - t_start)
        dts[0] = dts[0] + opening
    # Clamp negative dts (out-of-order samples) to zero.
    dts = np.where(dts > 0.0, dts, 0.0)

    # Trapezoidal-ish integration: assume the sample's accel applies for its
    # owned dt. v_end = v_start + sum(a * dt); p_end = p_start + sum(v_mid * dt).
    v_east = float(initial_velocity_enu[0])
    v_north = float(initial_velocity_enu[1])
    v_up = float(initial_velocity_enu[2])
    dp_east = 0.0
    dp_north = 0.0
    dp_up = 0.0
    for k in range(n):
        dt_k = float(dts[k])
        if dt_k <= 0.0:
            continue
        # Mid-point velocity for position step (trapezoidal rule).
        ve_mid = v_east + 0.5 * float(east_acc[k]) * dt_k
        vn_mid = v_north + 0.5 * float(north_acc[k]) * dt_k
        vu_mid = v_up + 0.5 * float(up_acc[k]) * dt_k
        dp_east += ve_mid * dt_k
        dp_north += vn_mid * dt_k
        dp_up += vu_mid * dt_k
        v_east += float(east_acc[k]) * dt_k
        v_north += float(north_acc[k]) * dt_k
        v_up += float(up_acc[k]) * dt_k

    delta = np.array([dp_east, dp_north, dp_up], dtype=np.float64)
    v_end = np.array([v_east, v_north, v_up], dtype=np.float64)
    return delta, v_end, int(n)


@dataclass
class _TDCPMeasurement:
    """Light-weight measurement object compatible with the TDCP module.

    The TDCP solver pulls ``system_id``, ``prn``, ``satellite_ecef``,
    ``carrier_phase`` (cycles), ``satellite_velocity`` (m/s), and
    ``clock_drift`` (s/s). The PPC loader supplies all of these when
    ``include_sat_velocity=True``.
    """

    system_id: int
    prn: int
    satellite_ecef: np.ndarray
    carrier_phase: float
    satellite_velocity: np.ndarray
    clock_drift: float
    elevation: float = 0.0


def _build_tdcp_measurements(
    sat_ecef: np.ndarray,
    system_ids: np.ndarray,
    sat_id_strs: list[str],
    carrier_phase: np.ndarray,
    sat_velocity: np.ndarray,
    clock_drift: np.ndarray,
    rover_pos: np.ndarray,
) -> list[_TDCPMeasurement]:
    out: list[_TDCPMeasurement] = []
    sids = np.asarray(system_ids, dtype=np.int32)
    for k in range(int(sat_ecef.shape[0])):
        sys_char = _SYS_ID_TO_CHAR.get(int(sids[k]))
        if sys_char is None:
            continue
        sat_id_str = sat_id_strs[k] if k < len(sat_id_strs) else ""
        prn_str = sat_id_str[1:].lstrip("0") if sat_id_str else ""
        try:
            prn = int(prn_str) if prn_str else 0
        except ValueError:
            prn = 0
        if prn <= 0:
            continue
        sat_pos = np.asarray(sat_ecef[k], dtype=np.float64)
        if sat_pos.size != 3 or not np.all(np.isfinite(sat_pos)):
            continue
        cp = float(carrier_phase[k]) if k < len(carrier_phase) else float("nan")
        if not np.isfinite(cp) or abs(cp) < 1e3:
            continue
        sv = (
            np.asarray(sat_velocity[k], dtype=np.float64)
            if k < len(sat_velocity)
            else np.zeros(3, dtype=np.float64)
        )
        if sv.size != 3 or not np.all(np.isfinite(sv)):
            continue
        cd = float(clock_drift[k]) if k < len(clock_drift) else 0.0
        if not np.isfinite(cd):
            cd = 0.0
        elev = _elevation_rad(rover_pos, sat_pos)
        out.append(
            _TDCPMeasurement(
                system_id=int(sids[k]),
                prn=prn,
                satellite_ecef=sat_pos,
                carrier_phase=cp,
                satellite_velocity=sv,
                clock_drift=cd,
                elevation=float(elev),
            )
        )
    return out


def _apply_tdcp_smoother(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    data: dict,
    config: CTRBPFConfig,
    hybrid_pos: dict[float, np.ndarray] | None,
    hybrid_status: dict[float, int] | None,
    stats: _TDCPSmootherStats,
) -> np.ndarray:
    """Forward+backward Kalman smoother over ``positions`` using TDCP velocity.

    State per coordinate is scalar (position only). The motion model is
    ``x[t+1] = x[t] + v_tdcp[t] * dt[t]`` and the observation is the
    hybrid passthrough (``hybrid_pos[t]``) with sigma derived from the
    hybrid Status field. Where TDCP is rejected we fall through to a
    very loose motion model (sigma = 10 m) so the observation alone
    drives the state. Where hybrid is missing we use a huge observation
    sigma so the prediction propagates unchanged.

    Decoupled per-coordinate scalar Kalman/RTS — fine for trajectory
    smoothing on the 0.5 m PPC threshold question; cross-axis
    correlations from satellite geometry are second-order at this scale.
    """
    from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics

    n = int(positions.shape[0])
    if n < 2:
        return positions

    sat_ecef = data["sat_ecef"]
    system_ids = data.get("system_ids")
    used_prns = data.get("used_prns") or [[] for _ in range(n)]
    carrier_phase = data.get("carrier_phase")
    sat_velocity = data.get("sat_velocity")
    clock_drift = data.get("clock_drift")
    if (
        carrier_phase is None
        or sat_velocity is None
        or clock_drift is None
        or system_ids is None
    ):
        return positions

    # TDCP velocity per consecutive pair (between epoch i and i+1).
    tdcp_v = [None] * (n - 1)
    for i in range(n - 1):
        stats.pairs_attempted += 1
        dt_i = float(times[i + 1] - times[i])
        if not np.isfinite(dt_i) or dt_i <= 0.0:
            continue
        prev_meas = _build_tdcp_measurements(
            np.asarray(sat_ecef[i], dtype=np.float64),
            np.asarray(system_ids[i], dtype=np.int32),
            list(used_prns[i]) if i < len(used_prns) else [],
            np.asarray(carrier_phase[i], dtype=np.float64),
            np.asarray(sat_velocity[i], dtype=np.float64),
            np.asarray(clock_drift[i], dtype=np.float64),
            np.asarray(positions[i], dtype=np.float64),
        )
        cur_meas = _build_tdcp_measurements(
            np.asarray(sat_ecef[i + 1], dtype=np.float64),
            np.asarray(system_ids[i + 1], dtype=np.int32),
            list(used_prns[i + 1]) if (i + 1) < len(used_prns) else [],
            np.asarray(carrier_phase[i + 1], dtype=np.float64),
            np.asarray(sat_velocity[i + 1], dtype=np.float64),
            np.asarray(clock_drift[i + 1], dtype=np.float64),
            np.asarray(positions[i + 1], dtype=np.float64),
        )
        if len(prev_meas) < int(config.tdcp_min_sats) or len(cur_meas) < int(config.tdcp_min_sats):
            stats.pairs_rejected_min_sats += 1
            continue
        v, postfit = estimate_velocity_from_tdcp_with_metrics(
            np.asarray(positions[i], dtype=np.float64),
            prev_meas,
            cur_meas,
            dt_i,
            min_sats=int(config.tdcp_min_sats),
            max_postfit_rms_m=float(config.tdcp_postfit_max_m),
        )
        if v is None:
            stats.pairs_rejected_postfit += 1
            continue
        tdcp_v[i] = (v, dt_i, float(postfit))
        stats.pairs_accepted += 1

    # Per-epoch observation sigma from hybrid Status. For pure-PF TDCP variants
    # there is no hybrid anchor; use the PF trajectory itself as a loose
    # observation so TDCP can bridge weak-PR gaps without pinning bad epochs.
    obs_sigma = np.full(n, float(config.tdcp_obs_missing_sigma_m), dtype=np.float64)
    if hybrid_pos is not None:
        anchor_set = set(int(s) for s in config.fgo_apply_hybrid_statuses)
        for i in range(n):
            t_key = round(float(times[i]), 1)
            if t_key not in hybrid_pos:
                continue
            st = hybrid_status.get(t_key) if hybrid_status is not None else None
            if st is None:
                obs_sigma[i] = float(config.tdcp_obs_loose_sigma_m)
            elif st in anchor_set:
                obs_sigma[i] = float(config.tdcp_obs_loose_sigma_m)
            else:
                obs_sigma[i] = float(config.tdcp_obs_anchor_sigma_m)
    else:
        obs_sigma[:] = float(config.tdcp_obs_loose_sigma_m)

    # Anchor mask: Status=4 (or any status NOT in fgo_apply_hybrid_statuses)
    # are HARD-PINNED to the hybrid passthrough. The Kalman smoother only
    # operates on non-anchor epochs, bridging them via TDCP velocity from
    # the surrounding anchors. This protects cm-class hybrid passes from
    # being corrupted by TDCP postfit noise.
    anchor_mask = np.zeros(n, dtype=bool)
    if hybrid_pos is not None and hybrid_status is not None:
        rewrite_set = set(int(s) for s in config.fgo_apply_hybrid_statuses)
        for i in range(n):
            t_key = round(float(times[i]), 1)
            st = hybrid_status.get(t_key)
            if st is not None and st not in rewrite_set:
                anchor_mask[i] = True

    # Per-coordinate scalar Kalman forward + RTS backward.
    sigma_v = float(config.tdcp_sigma_mps)
    out = np.array(positions, dtype=np.float64, copy=True)
    for axis in range(3):
        x = np.zeros(n, dtype=np.float64)
        P = np.zeros(n, dtype=np.float64)
        # Initialise: anchors get the hybrid value at near-zero variance,
        # non-anchors take the passthrough as initial state with the
        # status-derived obs sigma.
        x[0] = positions[0, axis]
        P[0] = (1e-3) ** 2 if anchor_mask[0] else float(obs_sigma[0]) ** 2

        for i in range(1, n):
            tv = tdcp_v[i - 1]
            if tv is not None:
                v_axis = float(tv[0][axis])
                dt_i = tv[1]
                Q = (sigma_v * dt_i) ** 2
                x_pred = x[i - 1] + v_axis * dt_i
            else:
                # No TDCP: predict by inheriting last state, big process noise.
                Q = 100.0
                x_pred = x[i - 1]
            P_pred = P[i - 1] + Q
            if anchor_mask[i]:
                # Hard pin to hybrid value.
                x[i] = positions[i, axis]
                P[i] = (1e-3) ** 2
            else:
                R = obs_sigma[i] ** 2
                K = P_pred / (P_pred + R)
                z = positions[i, axis]
                x[i] = x_pred + K * (z - x_pred)
                P[i] = (1.0 - K) * P_pred

        # Backward (RTS smoother) — keep anchors hard-pinned.
        xs = np.array(x, copy=True)
        Ps = np.array(P, copy=True)
        for i in range(n - 2, -1, -1):
            if anchor_mask[i]:
                xs[i] = positions[i, axis]
                Ps[i] = (1e-3) ** 2
                continue
            tv = tdcp_v[i]
            if tv is not None:
                v_axis = float(tv[0][axis])
                dt_i = tv[1]
                Q = (sigma_v * dt_i) ** 2
                x_pred = x[i] + v_axis * dt_i
            else:
                Q = 100.0
                x_pred = x[i]
            P_pred = P[i] + Q
            if P_pred > 0.0:
                G = P[i] / P_pred
                xs[i] = x[i] + G * (xs[i + 1] - x_pred)
                Ps[i] = P[i] + G * G * (Ps[i + 1] - P_pred)
        out[:, axis] = xs

    # Hard pin anchors in the final output (defensive; the loop above
    # already keeps them at the hybrid value).
    for i in range(n):
        if anchor_mask[i]:
            out[i] = positions[i]

    return out


def _apply_low_sat_bridge(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    pr_used_counts: np.ndarray,
    config: CTRBPFConfig,
    stats: _LowSatBridgeStats,
) -> np.ndarray:
    """Linearly bridge short low-PR spans between stronger PF anchors."""
    out = np.asarray(positions, dtype=np.float64).copy()
    counts = np.asarray(pr_used_counts, dtype=np.float64).reshape(-1)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    n = int(out.shape[0])
    if n < 3 or counts.size != n or t.size != n:
        return out

    min_sats = int(config.low_sat_bridge_min_pr_sats)
    min_span = max(1, int(config.low_sat_bridge_min_span_epochs))
    max_span = max(min_span, int(config.low_sat_bridge_max_span_epochs))
    max_gap_s = float(config.low_sat_bridge_max_gap_s)
    if min_sats <= 0 or max_gap_s <= 0.0:
        return out

    i = 0
    while i < n:
        if not np.isfinite(counts[i]) or counts[i] >= min_sats:
            i += 1
            continue
        start = i
        while i < n and np.isfinite(counts[i]) and counts[i] < min_sats:
            i += 1
        end = i - 1
        span_len = end - start + 1
        lo = start - 1
        hi = i
        if (
            span_len < min_span
            or span_len > max_span
            or lo < 0
            or hi >= n
            or not np.isfinite(t[hi] - t[lo])
            or (t[hi] - t[lo]) <= 0.0
            or (t[hi] - t[lo]) > max_gap_s
            or not np.all(np.isfinite(out[lo, :3]))
            or not np.all(np.isfinite(out[hi, :3]))
            or counts[lo] < min_sats
            or counts[hi] < min_sats
        ):
            continue
        for j in range(start, hi):
            u = float((t[j] - t[lo]) / (t[hi] - t[lo]))
            out[j, :3] = (1.0 - u) * out[lo, :3] + u * out[hi, :3]
        stats.spans_applied += 1
        stats.epochs_rewritten += span_len
    return out


def _apply_startup_wls_bridge(
    *,
    positions: np.ndarray,
    times: np.ndarray,
    wls_quality: dict[str, np.ndarray] | None,
    config: CTRBPFConfig,
    stats: _LowSatBridgeStats,
) -> np.ndarray:
    """Back-extrapolate over weak-WLS startup epochs from the first good anchors."""
    if wls_quality is None or "pdop" not in wls_quality:
        return positions
    out = np.asarray(positions, dtype=np.float64).copy()
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    pdop = np.asarray(wls_quality["pdop"], dtype=np.float64).reshape(-1)
    n = int(out.shape[0])
    if n < 4 or t.size != n or pdop.size != n:
        return out

    threshold = float(config.low_sat_bridge_startup_max_wls_pdop)
    min_epochs = max(1, int(config.low_sat_bridge_startup_min_epochs))
    max_epochs = max(min_epochs, int(config.low_sat_bridge_startup_max_epochs))
    if threshold <= 0.0:
        return out

    span = 0
    while (
        span < n
        and span < max_epochs
        and np.isfinite(pdop[span])
        and pdop[span] > threshold
    ):
        span += 1
    if span < min_epochs or span + 1 >= n:
        return out
    if (
        not np.all(np.isfinite(out[span, :3]))
        or not np.all(np.isfinite(out[span + 1, :3]))
        or not np.isfinite(t[span + 1] - t[span])
        or (t[span + 1] - t[span]) <= 0.0
    ):
        return out

    velocity = (out[span + 1, :3] - out[span, :3]) / (t[span + 1] - t[span])
    for j in range(span):
        out[j, :3] = out[span, :3] - velocity * (t[span] - t[j])
    stats.startup_spans_applied += 1
    stats.startup_epochs_rewritten += span
    return out


def _build_hybrid_velocity_guide(
    hybrid_pos: dict[float, np.ndarray],
    times: np.ndarray,
    max_dt_s: float = 0.5,
) -> dict[float, np.ndarray]:
    """Per-epoch ECEF velocity guide derived from hybrid pos finite-diffs.

    For rover TOW ``t``, the velocity is ``(hp[t] - hp[t_prev]) / (t - t_prev)``
    where ``t_prev`` is the most recent hybrid sample within ``max_dt_s``.
    Epochs without a usable predecessor receive zero velocity (so PF stays
    stationary that step rather than extrapolating an unreliable guide).
    """
    out: dict[float, np.ndarray] = {}
    if not hybrid_pos:
        return out
    sorted_keys = np.array(sorted(hybrid_pos.keys()), dtype=np.float64)
    for t in times:
        t_key = round(float(t), 1)
        hp_now = hybrid_pos.get(t_key)
        if hp_now is None:
            continue
        idx = int(np.searchsorted(sorted_keys, t_key))
        # find the most recent strictly-prior key
        prev_key = None
        for j in range(idx - 1, -1, -1):
            cand = float(sorted_keys[j])
            if cand >= t_key:
                continue
            if (t_key - cand) > max_dt_s:
                break
            prev_key = cand
            break
        if prev_key is None:
            continue
        hp_prev = hybrid_pos.get(prev_key)
        if hp_prev is None:
            continue
        dt = t_key - prev_key
        if dt <= 0.0:
            continue
        v = (hp_now - hp_prev) / dt
        if not np.all(np.isfinite(v)):
            continue
        out[t_key] = v
    return out


def _load_hybrid_pos_file(
    path: Path,
) -> tuple[dict[float, np.ndarray], dict[float, int]]:
    """Parse a libgnss++ ``.pos`` file into TOW-keyed ECEF + Status lookups.

    File format (whitespace-separated, '%' comments):
        GPS_Week  GPS_TOW  X  Y  Z  Lat  Lon  Height  Status  ...

    Returns ``(positions, statuses)``. Rows with non-finite coords or
    status==0 (no fix) are dropped. Keys are rounded to 0.1 s to match
    the rover-epoch TOW grid used elsewhere.

    Note: libgnss++ Status semantics are inverted from RTKLIB; on the
    Phase 6 baseline pos files Status=4 is the high-quality (cm-class)
    bucket, while Status in {1, 3} are m-class regions where Phase 4
    FGO+LAMBDA is more likely to help than to hurt.
    """
    positions: dict[float, np.ndarray] = {}
    statuses: dict[float, int] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                tow = round(float(parts[1]), 1)
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                status = int(float(parts[8]))
            except (ValueError, IndexError):
                continue
            if status == 0:
                continue
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                continue
            positions[tow] = np.array([x, y, z], dtype=np.float64)
            statuses[tow] = int(status)
    return positions, statuses


def _load_rtk_diag_file(path: Path) -> dict[float, dict[str, str]]:
    """Parse gnss_solve --diagnostics-csv output keyed by rounded TOW."""
    rows: dict[float, dict[str, str]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tow = round(float(row["tow"]), 1)
            except (KeyError, TypeError, ValueError):
                continue
            rows[tow] = row
    return rows


def _diag_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _diag_bool(row: dict[str, str], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", "", "nan"}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return bool(text)


def _rtkdiag_candidate_diag_policy_gate(
    row: dict[str, str],
    *,
    require_any_fields: tuple[str, ...],
    require_all_fields: tuple[str, ...],
    min_fields: tuple[tuple[str, float], ...],
    max_fields: tuple[tuple[str, float], ...],
) -> bool:
    if require_any_fields and not any(_diag_bool(row, key) for key in require_any_fields):
        return False
    if require_all_fields and not all(_diag_bool(row, key) for key in require_all_fields):
        return False
    for key, threshold in min_fields:
        value = _diag_float(row, key)
        if not np.isfinite(value) or value < float(threshold):
            return False
    for key, threshold in max_fields:
        value = _diag_float(row, key)
        if not np.isfinite(value) or value > float(threshold):
            return False
    return True


def _rtkdiag_candidate_gate(
    row: dict[str, str] | None,
    *,
    ratio_min: float,
    residual_rms_max: float,
    status5_residual_rms_max: float = 0.3,
) -> bool:
    """Main candidate gate.

    - status=4 (Fix): ratio>=ratio_min AND residual_rms<=residual_rms_max
    - status=5 (Float): residual_rms<=status5_residual_rms_max (ratio irrelevant);
      set status5_residual_rms_max=0.0 to disable status=5 acceptance.
    """
    if row is None:
        return False
    try:
        output_added = int(row.get("output_added", "0")) == 1
        status = int(row.get("final_status", "0"))
    except ValueError:
        return False
    if not output_added:
        return False
    residual_rms = _diag_float(row, "final_residual_rms")
    if status == 4:
        return (
            _diag_float(row, "final_ratio") >= float(ratio_min)
            and residual_rms <= float(residual_rms_max)
        )
    if status == 5 and float(status5_residual_rms_max) > 0.0:
        return residual_rms <= float(status5_residual_rms_max)
    return False


def _rtkdiag_candidate_float_gate(
    row: dict[str, str] | None,
    *,
    label: str,
    allowed_labels: tuple[str, ...],
    residual_rms_max: float,
    residual_abs_max: float,
    min_sats: int,
) -> bool:
    if row is None or not allowed_labels or label not in set(allowed_labels):
        return False
    try:
        output_added = int(row.get("output_added", "0")) == 1
        final_status = int(row.get("final_status", "0")) == 3
    except ValueError:
        return False
    return (
        output_added
        and final_status
        and _diag_float(row, "final_residual_rms") <= float(residual_rms_max)
        and _diag_float(row, "final_residual_abs_max") <= float(residual_abs_max)
        and _diag_float(row, "final_sats") >= float(min_sats)
    )


def _rtkdiag_candidate_status5_gate(
    row: dict[str, str] | None,
    *,
    label: str,
    allowed_labels: tuple[str, ...],
    residual_rms_max: float,
    min_sats: int,
) -> bool:
    if row is None or not allowed_labels or label not in set(allowed_labels):
        return False
    try:
        final_status = int(row.get("final_status", "0")) == 5
    except ValueError:
        return False
    return (
        final_status
        and _diag_float(row, "final_residual_rms") <= float(residual_rms_max)
        and _diag_float(row, "final_sats") >= float(min_sats)
    )


def _rtkdiag_nearest_candidate_row(
    candidate_pos: dict[float, np.ndarray],
    candidate_diag: dict[float, dict[str, str]],
    t_key: float,
    *,
    max_dt_s: float,
) -> tuple[float, np.ndarray | None, dict[str, str] | None]:
    max_steps = max(0, int(round(float(max_dt_s) * 10.0)))
    for step in range(max_steps + 1):
        offsets = (0.0,) if step == 0 else (-0.1 * step, 0.1 * step)
        for offset in offsets:
            cand_t_key = round(float(t_key) + float(offset), 1)
            cand = candidate_pos.get(cand_t_key)
            diag_row = candidate_diag.get(cand_t_key)
            if cand is not None and diag_row is not None:
                return cand_t_key, cand, diag_row
    return float(t_key), None, None


def _rtkdiag_candidate_sort_key(
    row: dict[str, str],
    *,
    mode: str,
) -> tuple[float, float]:
    """Rank gated RTK diagnostic candidates; smaller tuple is better."""
    ratio = _diag_float(row, "final_ratio")
    residual = _diag_float(row, "final_residual_rms")
    update_rows = _diag_float(row, "final_update_rows")
    if mode == "ratio":
        return (-ratio, residual)
    if mode == "score":
        return (residual / max(ratio, 1.0e-6), residual)
    if mode == "maxabs":
        return (_diag_float(row, "final_residual_abs_max"), residual)
    if mode == "nrows":
        return (-update_rows, residual)
    if mode == "rms_per_row":
        return (residual / max(update_rows, 1.0), residual)
    if mode == "score_per_row":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0), residual)
    if mode == "score_per_row2":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0) ** 2, residual)
    if mode == "score_per_row3":
        return ((residual / max(ratio, 1.0e-6)) / max(update_rows, 1.0) ** 3, residual)
    if mode == "rms_minus_alpha_rows":
        return (residual - 0.1 * update_rows, residual)
    if mode == "log_combined":
        import math as _m
        return (_m.log(residual + 1.0e-3) - 0.5 * _m.log(max(update_rows, 1.0)), residual)
    if mode == "composite_3axis_n2":
        # 3-axis sim BEST for n/r2: residual / (ratio^0.5 * rows^1.5 * abs_max^0.5)
        # sim 39.92% vs score 39.12 (+0.74pp); also n/r3 sim 59.04% vs 58.85 (+0.19pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.5
                       * max(update_rows, 1.0) ** 1.5
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_3axis_t2":
        # 3-axis sim BEST for t/r2: residual / (ratio^0.5 * rows^2.0)
        # sim 84.78% vs score_per_row 84.53 (+0.25pp).
        return (
            residual / (max(ratio, 1.0e-6) ** 0.5
                       * max(update_rows, 1.0) ** 2.0),
            residual,
        )
    if mode == "composite_3axis_n1":
        # 3-axis sim BEST for n/r1: residual / (rows^0.5 * abs_max^0.5)
        # sim 64.25% vs rms_per_row 63.89 (+0.37pp). Note: a=0 (no ratio dependence).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(update_rows, 1.0) ** 0.5
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n2_v2":
        # Fine sim BEST for n/r2: residual / (ratio^0.4 * rows^1.0 * abs_max^0.7)
        # sim 40.14% vs composite_3axis_n2 39.92 (+0.22pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.4
                       * max(update_rows, 1.0) ** 1.0
                       * max(abs_max, 1.0e-3) ** 0.7),
            residual,
        )
    if mode == "composite_n3_v2":
        # Fine sim BEST for n/r3: residual / (ratio^0.2 * rows^0.5 * abs_max^0.5)
        # sim 59.10% vs composite_3axis_n2 59.04 (+0.05pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.5
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n1_v2":
        # Fine sim BEST for n/r1: residual / (rows^0.5 * abs_max^0.3)
        # sim 64.31% vs composite_3axis_n1 64.25 (+0.06pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(update_rows, 1.0) ** 0.5
                       * max(abs_max, 1.0e-3) ** 0.3),
            residual,
        )
    if mode == "composite_t2_v2":
        # Fine sim BEST for t/r2: residual / (ratio^0.2 * rows^2.0 * abs_max^0.5)
        # sim 84.86% vs composite_3axis_t2 84.78 (+0.08pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 2.0
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_t3_v2":
        # Phase 11dm t/r3 re-sweep after 11dl pool:
        # residual / (ratio^1.3 * rows^1.5 * abs_max^-0.5).
        # Negative abs exponent intentionally penalizes tiny abs_max candidates
        # that became traps in the expanded 11dl pool.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 1.3
                       * max(update_rows, 1.0) ** 1.5
                       * max(abs_max, 1.0e-3) ** -0.5),
            residual,
        )
    if mode == "composite_t3_v4":
        # Phase 11eb t/r3 re-sweep after 11ea blocks:
        # residual / (ratio^1.5 * rows^1.5 * abs_max^-0.7).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 1.5
                       * max(update_rows, 1.0) ** 1.5
                       * max(abs_max, 1.0e-3) ** -0.7),
            residual,
        )
    if mode == "composite_t2_v3":
        # Phase 11ed t/r2 re-sweep after 11ec pool:
        # residual / (ratio^0.1 * rows^1.0 * abs_max^0.5), sim 85.099%.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.1
                       * max(update_rows, 1.0) ** 1.0
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n1_v3":
        # Ultra-fine sim BEST for n/r1: residual / (rows^0.7 * abs_max^0.3)
        # sim 64.33% vs composite_n1_v2 64.31 (+0.02pp). a=0 (no ratio).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(update_rows, 1.0) ** 0.7
                       * max(abs_max, 1.0e-3) ** 0.3),
            residual,
        )
    if mode == "composite_n2_v3":
        # Ultra-fine sim BEST for n/r2: residual / (ratio^0.3 * rows^0.7 * abs_max^0.8)
        # sim 40.21% vs composite_n2_v2 40.14 (+0.07pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.3
                       * max(update_rows, 1.0) ** 0.7
                       * max(abs_max, 1.0e-3) ** 0.8),
            residual,
        )
    if mode == "composite_n2_v4":
        # Phase 11dm n/r2 re-sweep after 11dl pool:
        # residual / (ratio^0.2 * rows^0.3 * abs_max^0.8), sim 40.73%.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.3
                       * max(abs_max, 1.0e-3) ** 0.8),
            residual,
        )
    if mode in {"temporal_n2_v1", "temporal_n2_v2", "temporal_n2_v3"}:
        # Stateless fallback for diagnostics; the PF loop adds the temporal
        # previous-position penalty on top of this same base key.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.3
                       * max(abs_max, 1.0e-3) ** 0.8),
            residual,
        )
    if mode in {"temporal_hybdelta_t3_v1", "temporal_hybdelta_t3_v2", "temporal_hybdelta_t3_v3"}:
        return _rtkdiag_candidate_sort_key(row, mode="composite_t3_v2")
    if mode == "temporal_hybdelta_t3_v4":
        return _rtkdiag_candidate_sort_key(row, mode="composite_t3_v4")
    if mode == "temporal_hybdelta_n2_v1":
        return _rtkdiag_candidate_sort_key(row, mode="composite_n2_v4")
    if mode == "temporal_hybdelta_n3_v1":
        return _rtkdiag_candidate_sort_key(row, mode="composite_n3_v3")
    if mode == "temporal_hybdelta_n3_v2":
        return _rtkdiag_candidate_sort_key(row, mode="composite_n3_v4")
    if mode == "composite_n3_v3":
        # Ultra-fine sim BEST for n/r3: residual / (ratio^0.2 * rows^0.7 * abs_max^0.5)
        # sim 59.15% vs composite_n3_v2 59.10 (+0.05pp).
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 0.7
                       * max(abs_max, 1.0e-3) ** 0.5),
            residual,
        )
    if mode == "composite_n3_v4":
        # Phase 11ec n/r3 re-sweep after 11eb pool:
        # residual / (ratio^0.2 * rows^1.0 * abs_max^0.7), sim 62.02%.
        abs_max = _diag_float(row, "final_residual_abs_max")
        return (
            residual / (max(ratio, 1.0e-6) ** 0.2
                       * max(update_rows, 1.0) ** 1.0
                       * max(abs_max, 1.0e-3) ** 0.7),
            residual,
        )
    # Conservative default: prefer the tightest measurement update residual.
    return (residual, -ratio)


def _rtkdiag_local_ungate_labels(
    windows: tuple[tuple[int, int, tuple[str, ...]], ...],
    epoch_idx: int,
) -> tuple[str, ...] | None:
    for start_idx, end_idx, labels in windows:
        if int(start_idx) <= int(epoch_idx) <= int(end_idx):
            return tuple(labels)
    return None


def _rtkdiag_local_ungate_labels_for_tow(
    windows: tuple[tuple[float, float, tuple[str, ...]], ...],
    tow: float,
) -> tuple[str, ...] | None:
    for start_tow, end_tow, labels in windows:
        if float(start_tow) <= float(tow) <= float(end_tow):
            return tuple(labels)
    return None


def _rtkdiag_fixed_output_ok(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    try:
        return int(row.get("output_added", "0")) == 1 and int(row.get("final_status", "0")) == 4
    except ValueError:
        return False


def _load_full_reference(path: Path) -> list[tuple[float, np.ndarray]]:
    """Load every reference.csv row in order (TOW seconds, ECEF)."""
    rows: list[tuple[float, np.ndarray]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 1)
            ecef = np.array(
                [float(row[5]), float(row[6]), float(row[7])],
                dtype=np.float64,
            )
            rows.append((tow, ecef))
    return rows


def _reference_position_map(rows: list[tuple[float, np.ndarray]]) -> dict[float, np.ndarray]:
    """Return reference ECEF positions keyed by rounded TOW for diagnostics."""
    return {round(float(tow), 1): np.asarray(ecef, dtype=np.float64) for tow, ecef in rows}


_RANKER_PREDICTIONS_CACHE: dict[str, dict[tuple[str, float, str], float]] = {}


def _load_ranker_predictions(path: str) -> dict[tuple[str, float, str], float]:
    """Load ranker predictions CSV into {(run_id, tow, label): p_pass}. Cached."""
    if not path:
        return {}
    if path in _RANKER_PREDICTIONS_CACHE:
        return _RANKER_PREDICTIONS_CACHE[path]
    lookup: dict[tuple[str, float, str], float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                run_id = str(row["run_id"])
                tow = round(float(row["tow"]), 1)
                label = str(row["label"])
                p = float(row["p_pass"])
                lookup[(run_id, tow, label)] = p
            except (ValueError, KeyError):
                continue
    _RANKER_PREDICTIONS_CACHE[path] = lookup
    return lookup


def _ranker_lookup_for_run(path: str, city: str, run: str) -> dict[tuple[float, str], float]:
    """Slice ranker predictions for a single run into {(tow, label): p_pass}."""
    if not path:
        return {}
    full = _load_ranker_predictions(path)
    run_id = f"{city}_{run}"
    return {(t, lbl): p for (rid, t, lbl), p in full.items() if rid == run_id}


_NLOS_MASK_CACHE: dict[str, dict[int, set[str]]] = {}


def _load_nlos_mask_csv(path: str) -> dict[int, set[str]]:
    """Load PLATEAU NLOS mask CSV → {epoch_idx: {prn1, prn2, ...}} of NLOS PRNs.

    Expected columns: tow, epoch_idx, prn, is_los (1=LOS, 0=NLOS).
    Only rows with is_los=0 contribute to the returned set.
    """
    if not path:
        return {}
    if path in _NLOS_MASK_CACHE:
        return _NLOS_MASK_CACHE[path]
    out: dict[int, set[str]] = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    is_los = int(row["is_los"])
                except (KeyError, ValueError):
                    continue
                if is_los != 0:
                    continue
                try:
                    epoch_idx = int(row["epoch_idx"])
                    prn = str(row["prn"]).strip()
                except (KeyError, ValueError):
                    continue
                if not prn:
                    continue
                out.setdefault(epoch_idx, set()).add(prn)
    except FileNotFoundError:
        pass
    _NLOS_MASK_CACHE[path] = out
    return out


def _filter_data_by_systems(data: dict, systems: tuple[str, ...]) -> dict:
    """Return a shallow data copy with per-satellite arrays masked by system."""
    allowed = {str(s).strip() for s in systems if str(s).strip()}
    if not allowed:
        return data
    allowed_ids = {sid for sid, ch in _SYS_ID_TO_CHAR.items() if ch in allowed}
    out = dict(data)
    n_epochs = int(data["n_epochs"])
    sat_counts: list[int] = []

    def mask_for_epoch(i: int) -> np.ndarray:
        sids = np.asarray(data["system_ids"][i], dtype=np.int32)
        return np.array([int(sid) in allowed_ids for sid in sids], dtype=bool)

    masked_keys = {
        "sat_ecef",
        "pseudoranges",
        "weights",
        "system_ids",
        "carrier_phase",
        "doppler_hz",
        "sat_velocity",
        "clock_drift",
    }
    for key in masked_keys:
        if key not in data:
            continue
        vals = []
        for i in range(n_epochs):
            arr = np.asarray(data[key][i])
            vals.append(arr[mask_for_epoch(i)])
        out[key] = vals

    for key in ("used_prns", "carrier_codes", "doppler_codes"):
        if key not in data:
            continue
        vals = []
        for i in range(n_epochs):
            mask = mask_for_epoch(i)
            seq = list(data[key][i])
            vals.append([v for v, keep in zip(seq, mask) if bool(keep)])
        out[key] = vals

    for i in range(n_epochs):
        sat_counts.append(int(np.count_nonzero(mask_for_epoch(i))))
    out["satellite_counts"] = np.asarray(sat_counts, dtype=np.int32)
    out["n_satellites"] = int(np.median(out["satellite_counts"])) if sat_counts else 0
    out["constellations"] = tuple(
        sorted({sat_id[0] for sats in out.get("used_prns", []) for sat_id in sats if sat_id})
    )
    return out


def _filter_data_by_elevation(
    data: dict,
    receiver_positions: np.ndarray,
    min_elevation_deg: float,
    *,
    min_sats: int = 5,
) -> dict:
    """Return a data copy with low-elevation satellites removed per epoch."""
    threshold = float(min_elevation_deg)
    if threshold <= -90.0:
        return data
    positions = np.asarray(receiver_positions, dtype=np.float64)
    out = dict(data)
    n_epochs = int(data["n_epochs"])
    sat_counts: list[int] = []

    def mask_for_epoch(i: int) -> np.ndarray:
        sat = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        if (
            i >= positions.shape[0]
            or positions.shape[1] < 3
            or not np.all(np.isfinite(positions[i, :3]))
        ):
            return np.ones(sat.shape[0], dtype=bool)
        elevations = np.array(
            [
                math.degrees(
                    _sat_elevation_azimuth(positions[i, :3], sat_row)[0]
                )
                for sat_row in sat
            ],
            dtype=np.float64,
        )
        mask = elevations >= threshold
        if int(np.count_nonzero(mask)) >= int(min_sats):
            return mask
        keep_n = min(max(int(min_sats), 1), int(elevations.size))
        keep_idx = np.argsort(elevations)[-keep_n:]
        mask = np.zeros(elevations.size, dtype=bool)
        mask[keep_idx] = True
        return mask

    masked_keys = {
        "sat_ecef",
        "pseudoranges",
        "weights",
        "system_ids",
        "carrier_phase",
        "doppler_hz",
        "sat_velocity",
        "clock_drift",
    }
    for key in masked_keys:
        if key not in data:
            continue
        vals = []
        for i in range(n_epochs):
            arr = np.asarray(data[key][i])
            vals.append(arr[mask_for_epoch(i)])
        out[key] = vals

    for key in ("used_prns", "carrier_codes", "doppler_codes"):
        if key not in data:
            continue
        vals = []
        for i in range(n_epochs):
            mask = mask_for_epoch(i)
            seq = list(data[key][i])
            vals.append([v for v, keep in zip(seq, mask) if bool(keep)])
        out[key] = vals

    for i in range(n_epochs):
        sat_counts.append(int(np.count_nonzero(mask_for_epoch(i))))
    out["satellite_counts"] = np.asarray(sat_counts, dtype=np.int32)
    out["n_satellites"] = int(np.median(out["satellite_counts"])) if sat_counts else 0
    out["constellations"] = tuple(
        sorted({sat_id[0] for sats in out.get("used_prns", []) for sat_id in sats if sat_id})
    )
    return out


def _filter_data_by_pr_prefit(
    data: dict,
    receiver_positions: np.ndarray,
    *,
    gate_m: float,
    clock_quantile: float,
    min_sats: int,
    keep_best: int,
    per_system: bool,
) -> tuple[dict, int, int]:
    """Return a data copy with robust PR prefit outliers removed per epoch."""
    if float(gate_m) <= 0.0:
        return data, 0, 0
    positions = np.asarray(receiver_positions, dtype=np.float64)
    out = dict(data)
    n_epochs = int(data["n_epochs"])
    sat_counts: list[int] = []
    kept_total = 0
    dropped_total = 0

    def mask_for_epoch(i: int) -> np.ndarray:
        sat = np.asarray(data["sat_ecef"][i], dtype=np.float64)
        pr = np.asarray(data["pseudoranges"][i], dtype=np.float64)
        if (
            i >= positions.shape[0]
            or positions.shape[1] < 3
            or not np.all(np.isfinite(positions[i, :3]))
        ):
            return np.ones(pr.shape[0], dtype=bool)
        sat_ref = rotate_satellites_sagnac(positions[i, :3], sat)
        sids = (
            np.asarray(data["system_ids"][i], dtype=np.int32)
            if "system_ids" in data
            else None
        )
        return _pr_prefit_gate_mask(
            sat_ref,
            pr,
            positions[i, :3],
            gate_m=float(gate_m),
            clock_quantile=float(clock_quantile),
            min_sats=int(min_sats),
            keep_best=int(keep_best),
            system_ids=sids,
            per_system=bool(per_system),
        )

    masks = [mask_for_epoch(i) for i in range(n_epochs)]
    masked_keys = {
        "sat_ecef",
        "pseudoranges",
        "weights",
        "system_ids",
        "carrier_phase",
        "doppler_hz",
        "sat_velocity",
        "clock_drift",
    }
    for key in masked_keys:
        if key not in data:
            continue
        vals = []
        for i in range(n_epochs):
            arr = np.asarray(data[key][i])
            vals.append(arr[masks[i]])
        out[key] = vals

    for key in ("used_prns", "carrier_codes", "doppler_codes"):
        if key not in data:
            continue
        vals = []
        for i in range(n_epochs):
            seq = list(data[key][i])
            vals.append([v for v, keep in zip(seq, masks[i]) if bool(keep)])
        out[key] = vals

    for mask in masks:
        kept = int(np.count_nonzero(mask))
        total = int(mask.size)
        sat_counts.append(kept)
        kept_total += kept
        dropped_total += max(total - kept, 0)
    out["satellite_counts"] = np.asarray(sat_counts, dtype=np.int32)
    out["n_satellites"] = int(np.median(out["satellite_counts"])) if sat_counts else 0
    out["constellations"] = tuple(
        sorted({sat_id[0] for sats in out.get("used_prns", []) for sat_id in sats if sat_id})
    )
    return out, kept_total, dropped_total


def _wls_centered_postfit_rms_m(
    data: dict,
    receiver_positions: np.ndarray,
    *,
    per_system: bool = True,
) -> float:
    """Median epoch RMS after centering WLS PR residuals by clock/system bias."""
    positions = np.asarray(receiver_positions, dtype=np.float64)
    values: list[float] = []
    for i, (sat_epoch, pr_epoch) in enumerate(
        zip(data["sat_ecef"], data["pseudoranges"])
    ):
        if (
            i >= positions.shape[0]
            or positions.shape[1] < 3
            or not np.all(np.isfinite(positions[i, :3]))
        ):
            continue
        sat = np.asarray(sat_epoch, dtype=np.float64)
        pr = np.asarray(pr_epoch, dtype=np.float64)
        if sat.shape[0] < 4 or pr.size != sat.shape[0]:
            continue
        sat_rot = rotate_satellites_sagnac(positions[i, :3], sat)
        residuals = pr - np.linalg.norm(sat_rot - positions[i, :3], axis=1)
        finite = np.isfinite(residuals)
        if int(np.count_nonzero(finite)) < 4:
            continue
        centered = residuals.copy()
        if per_system and "system_ids" in data:
            sids = np.asarray(data["system_ids"][i], dtype=np.int32)
            if sids.size == residuals.size:
                for sid in np.unique(sids[finite]):
                    group = finite & (sids == int(sid))
                    if int(np.count_nonzero(group)) > 0:
                        centered[group] -= float(np.median(residuals[group]))
            else:
                centered[finite] -= float(np.median(residuals[finite]))
        else:
            centered[finite] -= float(np.median(residuals[finite]))
        r = centered[finite]
        values.append(float(math.sqrt(float(np.mean(r * r)))))
    return float(np.median(values)) if values else float("nan")


def _wls_epoch_quality_metrics(
    data: dict,
    receiver_positions: np.ndarray,
    *,
    per_system: bool = True,
) -> dict[str, np.ndarray]:
    """Compute per-epoch WLS residual and geometry metrics for diagnostics/gates."""
    positions = np.asarray(receiver_positions, dtype=np.float64)
    sat_ecef = data["sat_ecef"]
    pr_obs = data["pseudoranges"]
    weights = data.get("weights")
    system_ids = data.get("system_ids")
    n_epochs = int(data["n_epochs"])
    n_sat = np.zeros(n_epochs, dtype=np.int32)
    rms = np.full(n_epochs, np.nan, dtype=np.float64)
    absmax = np.full(n_epochs, np.nan, dtype=np.float64)
    pdop = np.full(n_epochs, np.nan, dtype=np.float64)
    cond = np.full(n_epochs, np.nan, dtype=np.float64)

    for i in range(n_epochs):
        if (
            i >= positions.shape[0]
            or positions.shape[1] < 3
            or not np.all(np.isfinite(positions[i, :3]))
            or np.all(positions[i, :3] == 0.0)
        ):
            continue
        pos = positions[i, :3]
        sat = np.asarray(sat_ecef[i], dtype=np.float64)
        pr = np.asarray(pr_obs[i], dtype=np.float64)
        if weights is None:
            w = np.ones(pr.shape, dtype=np.float64)
        else:
            w = np.asarray(weights[i], dtype=np.float64)
        sat_rot = rotate_satellites_sagnac(pos, sat)
        ranges = np.linalg.norm(sat_rot - pos, axis=1)
        finite = (
            np.isfinite(pr)
            & np.isfinite(ranges)
            & np.isfinite(w)
            & (w > 0.0)
            & (ranges > 1.0)
        )
        n_sat[i] = int(np.count_nonzero(finite))
        if n_sat[i] < 4:
            continue

        residuals = pr - ranges
        centered = residuals.copy()
        if per_system and system_ids is not None:
            sids = np.asarray(system_ids[i], dtype=np.int32)
            if sids.size == residuals.size:
                for sid in np.unique(sids[finite]):
                    group = finite & (sids == int(sid))
                    if int(np.count_nonzero(group)) > 0:
                        centered[group] -= float(np.median(residuals[group]))
            else:
                centered[finite] -= float(np.median(residuals[finite]))
        else:
            centered[finite] -= float(np.median(residuals[finite]))

        wf = w[finite]
        cf = centered[finite]
        sw = float(np.sum(wf))
        if sw > 0.0:
            rms[i] = float(math.sqrt(float(np.sum(wf * cf * cf) / sw)))
            absmax[i] = float(np.max(np.abs(cf)))

        sat_f = sat_rot[finite]
        ranges_f = ranges[finite]
        los = (pos[None, :] - sat_f) / ranges_f[:, None]
        if per_system and system_ids is not None:
            sids_all = np.asarray(system_ids[i], dtype=np.int32)
            if sids_all.size == residuals.size:
                sids_f = sids_all[finite]
                unique_sids = sorted(int(s) for s in np.unique(sids_f))
                sid_to_col = {sid: k for k, sid in enumerate(unique_sids)}
                h = np.zeros((n_sat[i], 3 + len(unique_sids)), dtype=np.float64)
                h[:, :3] = los
                for row, sid in enumerate(sids_f):
                    h[row, 3 + sid_to_col[int(sid)]] = 1.0
            else:
                h = np.ones((n_sat[i], 4), dtype=np.float64)
                h[:, :3] = los
        else:
            h = np.ones((n_sat[i], 4), dtype=np.float64)
            h[:, :3] = los
        try:
            normal = h.T @ (wf[:, None] * h)
            cov = np.linalg.pinv(normal)
            pdop[i] = float(math.sqrt(max(float(np.trace(cov[:3, :3])), 0.0)))
            cond[i] = float(np.linalg.cond(normal))
        except np.linalg.LinAlgError:
            continue

    return {
        "n_sat": n_sat,
        "postfit_rms_m": rms,
        "postfit_absmax_m": absmax,
        "pdop": pdop,
        "normal_cond": cond,
    }


def _sat_elevation_azimuth(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
) -> tuple[float, float, float, float, float]:
    lat, lon, alt = _ecef_to_llh(
        float(rx_ecef[0]), float(rx_ecef[1]), float(rx_ecef[2])
    )
    dx = np.asarray(sat_ecef, dtype=np.float64) - np.asarray(rx_ecef, dtype=np.float64)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    e = -sin_lon * dx[0] + cos_lon * dx[1]
    n = -sin_lat * cos_lon * dx[0] - sin_lat * sin_lon * dx[1] + cos_lat * dx[2]
    u = cos_lat * cos_lon * dx[0] + cos_lat * sin_lon * dx[1] + sin_lat * dx[2]
    return math.atan2(float(u), math.hypot(float(e), float(n))), math.atan2(float(e), float(n)), lat, lon, alt


def _pr_atmosphere_delay_m(
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    tow: float,
    *,
    model: str,
    fixed_zenith_m: float,
    model_scale: float,
    extra_zenith_m: float,
    iono_alpha: tuple[float, ...],
    iono_beta: tuple[float, ...],
) -> float:
    mode = str(model).strip().lower()
    if mode == "off" and float(fixed_zenith_m) > 0.0:
        mode = "fixed"
    if mode in {"", "off"}:
        return 0.0
    el, az, lat, lon, alt = _sat_elevation_azimuth(rx_ecef, sat_ecef)
    sin_el = max(math.sin(el), 0.1)
    if mode == "fixed":
        return float(fixed_zenith_m) / sin_el
    if mode == "model":
        alpha = iono_alpha if len(iono_alpha) == 4 else (
            1.1176e-08,
            -7.4506e-09,
            -5.9605e-08,
            1.1921e-07,
        )
        beta = iono_beta if len(iono_beta) == 4 else (
            1.1264e05,
            -3.2768e04,
            -2.6214e05,
            4.5875e05,
        )
        modeled = float(_tropo_saastamoinen(lat, alt, el)) + float(
            _iono_klobuchar(list(alpha), list(beta), lat, lon, az, el, float(tow))
        )
        return float(model_scale) * modeled + float(extra_zenith_m) / sin_el
    return 0.0


def _apply_atmosphere_to_pseudoranges(
    data: dict,
    receiver_positions: np.ndarray,
    *,
    model: str,
    fixed_zenith_m: float,
    model_scale: float,
    extra_zenith_m: float,
    iono_alpha: tuple[float, ...],
    iono_beta: tuple[float, ...],
) -> dict:
    """Return a data copy with modeled PR atmosphere delay removed."""
    mode = str(model).strip().lower()
    if mode == "off" and float(fixed_zenith_m) > 0.0:
        mode = "fixed"
    if mode in {"", "off"}:
        return data
    positions = np.asarray(receiver_positions, dtype=np.float64)
    out = dict(data)
    corrected: list[np.ndarray] = []
    for i, (sat_epoch, pr_epoch) in enumerate(zip(data["sat_ecef"], data["pseudoranges"])):
        sat_arr = np.asarray(sat_epoch, dtype=np.float64)
        pr_arr = np.asarray(pr_epoch, dtype=np.float64)
        if i >= positions.shape[0] or not np.all(np.isfinite(positions[i, :3])):
            corrected.append(pr_arr.copy())
            continue
        rx = positions[i, :3]
        delay = np.array(
            [
                _pr_atmosphere_delay_m(
                    rx,
                    sat_row,
                    float(data["times"][i]),
                    model=mode,
                    fixed_zenith_m=float(fixed_zenith_m),
                    model_scale=float(model_scale),
                    extra_zenith_m=float(extra_zenith_m),
                    iono_alpha=iono_alpha,
                    iono_beta=iono_beta,
                )
                for sat_row in sat_arr
            ],
            dtype=np.float64,
        )
        corrected.append(pr_arr - delay)
    out["pseudoranges"] = corrected
    return out


def _apply_slant_delay_to_pseudoranges(
    data: dict,
    receiver_positions: np.ndarray,
    zenith_delay_m: float,
) -> dict:
    """Return a data copy with a simple elevation-mapped PR delay removed."""
    return _apply_atmosphere_to_pseudoranges(
        data,
        receiver_positions,
        model="fixed",
        fixed_zenith_m=float(zenith_delay_m),
        model_scale=1.0,
        extra_zenith_m=0.0,
        iono_alpha=(),
        iono_beta=(),
    )


def _scale_weights_per_system(weights: np.ndarray, system_ids: np.ndarray) -> np.ndarray:
    out = np.asarray(weights, dtype=np.float64).copy()
    sids = np.asarray(system_ids, dtype=np.int32)
    for sys_id, scale in _SYSTEM_WEIGHT_SCALE.items():
        mask = sids == sys_id
        if mask.any() and scale != 1.0:
            out[mask] *= scale
    return out


def _pr_likelihood_weights(weights: np.ndarray, config: CTRBPFConfig) -> np.ndarray:
    """Convert PPC C/N0-like values into PF likelihood multipliers."""
    w = np.asarray(weights, dtype=np.float64)
    mode = str(config.pr_weight_mode).strip().lower()
    if mode == "raw":
        return w.astype(np.float64, copy=True)
    elif mode == "unit":
        out = np.ones_like(w, dtype=np.float64)
    elif mode in {"cn0-relative", "relative"}:
        ref = max(float(config.pr_weight_ref_cn0), 1.0)
        out = w / ref
    else:
        raise ValueError(f"unsupported pr_weight_mode: {config.pr_weight_mode}")

    lo = float(config.pr_weight_min)
    hi = float(config.pr_weight_max)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        out = np.clip(out, lo, hi)
    return out.astype(np.float64, copy=False)


def _pr_prefit_gate_mask(
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    ref_pos_ecef: np.ndarray,
    *,
    gate_m: float,
    clock_quantile: float,
    min_sats: int,
    keep_best: int,
    system_ids: np.ndarray | None = None,
    per_system: bool = False,
) -> np.ndarray:
    """Gate pseudoranges by robust prefit residual around a reference position."""
    n_sat = int(len(pseudoranges))
    mask = np.ones(n_sat, dtype=bool)
    if n_sat < 4 or float(gate_m) <= 0.0:
        return mask

    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()
    ref = np.asarray(ref_pos_ecef, dtype=np.float64).ravel()
    ranges = np.linalg.norm(sat - ref[:3], axis=1)
    residuals = pr - ranges
    finite = np.isfinite(residuals)
    if int(finite.sum()) < 4:
        return mask

    q = float(np.clip(float(clock_quantile), 0.0, 1.0))
    abs_prefit = np.full(n_sat, np.inf, dtype=np.float64)
    if per_system and system_ids is not None:
        sids = np.asarray(system_ids, dtype=np.int32).ravel()
        if sids.size == n_sat:
            for sid in np.unique(sids[finite]):
                group = finite & (sids == int(sid))
                if int(group.sum()) == 0:
                    continue
                cb = float(np.quantile(residuals[group], q))
                abs_prefit[group] = np.abs(residuals[group] - cb)
        else:
            cb = float(np.quantile(residuals[finite], q))
            abs_prefit[finite] = np.abs(residuals[finite] - cb)
    else:
        cb = float(np.quantile(residuals[finite], q))
        abs_prefit[finite] = np.abs(residuals[finite] - cb)
    mask = finite & (abs_prefit <= float(gate_m))

    min_keep = max(4, min(int(min_sats), n_sat))
    finite_order = np.argsort(np.where(finite, abs_prefit, np.inf))
    if int(mask.sum()) < min_keep:
        mask = np.zeros(n_sat, dtype=bool)
        mask[finite_order[:min_keep]] = True
    elif int(keep_best) > 0 and int(mask.sum()) > int(keep_best):
        gated_order = np.array([idx for idx in finite_order if mask[idx]], dtype=np.int32)
        mask = np.zeros(n_sat, dtype=bool)
        mask[gated_order[: int(keep_best)]] = True
    return mask


def _build_pf(config: CTRBPFConfig):
    from gnss_gpu import ParticleFilterDevice

    pf = ParticleFilterDevice(
        n_particles=config.n_particles,
        sigma_pos=config.sigma_pos,
        sigma_cb=config.sigma_cb,
        sigma_pr=config.sigma_pr,
        resampling="megopolis",
        seed=42,
    )
    return pf


def _ess_ratio_from_log_weights(log_weights: np.ndarray) -> float:
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    finite = np.isfinite(lw)
    if int(np.count_nonzero(finite)) == 0:
        return float("nan")
    values = lw[finite]
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    sw = float(np.sum(weights))
    sw2 = float(np.sum(weights * weights))
    if sw <= 0.0 or sw2 <= 0.0:
        return float("nan")
    ess = (sw * sw) / sw2
    return float(ess / max(int(lw.size), 1))


def _apply_pr_ess_guard(
    pf,
    pre_pr_log_weights: np.ndarray,
    *,
    min_ratio: float,
    max_iters: int,
) -> tuple[float, float, float]:
    """Temper the latest PR log-likelihood increment to preserve PF ESS."""
    target = float(min_ratio)
    post_pr_log_weights = np.asarray(pf.get_log_weights(), dtype=np.float64)
    post_ratio = _ess_ratio_from_log_weights(post_pr_log_weights)
    if target <= 0.0 or not np.isfinite(post_ratio) or post_ratio >= target:
        return 1.0, post_ratio, post_ratio

    pre = np.asarray(pre_pr_log_weights, dtype=np.float64)
    if pre.shape != post_pr_log_weights.shape:
        return 1.0, post_ratio, post_ratio
    delta = post_pr_log_weights - pre
    if not np.all(np.isfinite(delta)):
        return 1.0, post_ratio, post_ratio

    pre_ratio = _ess_ratio_from_log_weights(pre)
    if not np.isfinite(pre_ratio) or pre_ratio < target:
        pf.set_log_weights(pre)
        return 0.0, post_ratio, pre_ratio

    lo = 0.0
    hi = 1.0
    best_alpha = 0.0
    best_ratio = pre_ratio
    for _ in range(max(1, int(max_iters))):
        mid = 0.5 * (lo + hi)
        candidate = pre + mid * delta
        ratio = _ess_ratio_from_log_weights(candidate)
        if np.isfinite(ratio) and ratio >= target:
            best_alpha = mid
            best_ratio = ratio
            lo = mid
        else:
            hi = mid

    pf.set_log_weights(pre + best_alpha * delta)
    final_ratio = _ess_ratio_from_log_weights(np.asarray(pf.get_log_weights(), dtype=np.float64))
    if np.isfinite(final_ratio):
        best_ratio = final_ratio
    return float(best_alpha), float(post_ratio), float(best_ratio)


def _capture_pf_internal_state(
    pf,
    row: dict[str, object],
    prefix: str,
    *,
    n_particles: int,
    reference_ecef: np.ndarray | None = None,
) -> np.ndarray:
    """Attach scalar PF state diagnostics to ``row`` and return estimate."""
    est = np.asarray(pf.estimate(), dtype=np.float64)
    ess = float(pf.get_ess())
    spread = float(pf.get_position_spread(center=est[:3]))
    row[f"{prefix}_x"] = float(est[0])
    row[f"{prefix}_y"] = float(est[1])
    row[f"{prefix}_z"] = float(est[2])
    row[f"{prefix}_cb_m"] = float(est[3])
    row[f"{prefix}_ess"] = ess
    row[f"{prefix}_ess_ratio"] = ess / max(int(n_particles), 1)
    row[f"{prefix}_spread_m"] = spread
    if reference_ecef is not None:
        ref = np.asarray(reference_ecef, dtype=np.float64).reshape(-1)
        if ref.size >= 3 and np.all(np.isfinite(ref[:3])):
            delta = est[:3] - ref[:3]
            row[f"{prefix}_to_ref_m"] = float(np.linalg.norm(delta))
            lat, lon, _ = _ecef_to_llh(float(ref[0]), float(ref[1]), float(ref[2]))
            enu = _ecef_to_enu_rotation(lat, lon) @ delta
            row[f"{prefix}_err_e_m"] = float(enu[0])
            row[f"{prefix}_err_n_m"] = float(enu[1])
            row[f"{prefix}_err_u_m"] = float(enu[2])
    return est


def _diag_distance(
    lhs: np.ndarray | None,
    rhs: np.ndarray | None,
) -> float | str:
    if lhs is None or rhs is None:
        return ""
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3:
        return ""
    if not np.all(np.isfinite(a[:3])) or not np.all(np.isfinite(b[:3])):
        return ""
    return float(np.linalg.norm(a[:3] - b[:3]))


def _diag_enu_error(
    lhs: np.ndarray | None,
    rhs: np.ndarray | None,
) -> tuple[float | str, float | str, float | str]:
    if lhs is None or rhs is None:
        return "", "", ""
    a = np.asarray(lhs, dtype=np.float64).reshape(-1)
    b = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if a.size < 3 or b.size < 3:
        return "", "", ""
    if not np.all(np.isfinite(a[:3])) or not np.all(np.isfinite(b[:3])):
        return "", "", ""
    delta = a[:3] - b[:3]
    lat, lon, _ = _ecef_to_llh(float(b[0]), float(b[1]), float(b[2]))
    enu = _ecef_to_enu_rotation(lat, lon) @ delta
    return float(enu[0]), float(enu[1]), float(enu[2])


def _weighted_mean_from_log_weights(values: np.ndarray, log_weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    lw = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if arr.ndim != 2 or arr.shape[0] != lw.size:
        return np.full(arr.shape[1] if arr.ndim == 2 else 1, np.nan, dtype=np.float64)
    finite = np.isfinite(lw) & np.all(np.isfinite(arr), axis=1)
    if int(np.count_nonzero(finite)) == 0:
        return np.full(arr.shape[1], np.nan, dtype=np.float64)
    lw_f = lw[finite]
    weights = np.exp(lw_f - float(np.max(lw_f)))
    sw = float(np.sum(weights))
    if sw <= 0.0:
        return np.full(arr.shape[1], np.nan, dtype=np.float64)
    return np.sum(arr[finite] * weights[:, None], axis=0) / sw


def _capture_pf_velocity_state(pf, row: dict[str, object], prefix: str) -> np.ndarray:
    states = np.asarray(pf.get_particle_states(), dtype=np.float64)
    log_weights = np.asarray(pf.get_log_weights(), dtype=np.float64)
    vel = _weighted_mean_from_log_weights(states[:, 4:7], log_weights)
    if vel.size >= 3 and np.all(np.isfinite(vel[:3])):
        row[f"{prefix}_vx_mps"] = float(vel[0])
        row[f"{prefix}_vy_mps"] = float(vel[1])
        row[f"{prefix}_vz_mps"] = float(vel[2])
        row[f"{prefix}_speed_mps"] = float(np.linalg.norm(vel[:3]))
    return vel


def _reference_velocity_for_epoch(
    times: np.ndarray,
    reference_pos: dict[float, np.ndarray] | None,
    idx: int,
) -> np.ndarray | None:
    if reference_pos is None:
        return None
    t_now = float(times[idx])
    p_now = reference_pos.get(round(t_now, 1))
    if p_now is None or not np.all(np.isfinite(p_now)):
        return None

    prev_vel = None
    if idx > 0:
        t_prev = float(times[idx - 1])
        p_prev = reference_pos.get(round(t_prev, 1))
        dt_prev = t_now - t_prev
        if p_prev is not None and dt_prev > 0.0 and np.all(np.isfinite(p_prev)):
            prev_vel = (np.asarray(p_now, dtype=np.float64) - np.asarray(p_prev, dtype=np.float64)) / dt_prev

    next_vel = None
    if idx + 1 < len(times):
        t_next = float(times[idx + 1])
        p_next = reference_pos.get(round(t_next, 1))
        dt_next = t_next - t_now
        if p_next is not None and dt_next > 0.0 and np.all(np.isfinite(p_next)):
            next_vel = (np.asarray(p_next, dtype=np.float64) - np.asarray(p_now, dtype=np.float64)) / dt_next

    if prev_vel is not None and next_vel is not None:
        return 0.5 * (prev_vel + next_vel)
    return prev_vel if prev_vel is not None else next_vel


def _doppler_centered_residual_rms(
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray,
    doppler_hz: np.ndarray,
    weights: np.ndarray,
    rx_pos: np.ndarray,
    rx_vel: np.ndarray,
    *,
    doppler_sign: float,
    wavelength_m: float,
) -> float:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    sv = np.asarray(sat_vel, dtype=np.float64).reshape(-1, 3)
    dop = np.asarray(doppler_hz, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    pos = np.asarray(rx_pos, dtype=np.float64).reshape(-1)
    vel = np.asarray(rx_vel, dtype=np.float64).reshape(-1)
    if sat.shape != sv.shape or sat.shape[0] != dop.size or dop.size != w.size:
        return float("nan")
    if pos.size < 3 or vel.size < 3 or not np.all(np.isfinite(pos[:3])) or not np.all(np.isfinite(vel[:3])):
        return float("nan")
    rows_h: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_w: list[float] = []
    for s in range(dop.size):
        ww = max(float(w[s]), 0.0)
        if ww <= 0.0 or not np.isfinite(ww) or not np.isfinite(dop[s]):
            continue
        los_vec = sat[s] - pos[:3]
        rng = float(np.linalg.norm(los_vec))
        if rng < 1.0 or not np.isfinite(rng):
            continue
        los = los_vec / rng
        rows_h.append(-los)
        rows_y.append(float(doppler_sign) * float(dop[s]) * float(wavelength_m) - float(np.dot(sv[s], los)))
        rows_w.append(ww)
    if len(rows_y) < 4:
        return float("nan")
    h = np.asarray(rows_h, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    ww = np.asarray(rows_w, dtype=np.float64)
    sw = float(np.sum(ww))
    if sw <= 0.0:
        return float("nan")
    h_center = h - np.sum(h * ww[:, None], axis=0) / sw
    y_center = y - float(np.sum(y * ww) / sw)
    residual = y_center - h_center @ vel[:3]
    return float(math.sqrt(float(np.sum(ww * residual * residual) / sw)))


def _doppler_centered_wls_velocity(
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray,
    doppler_hz: np.ndarray,
    weights: np.ndarray,
    rx_pos: np.ndarray,
    *,
    doppler_sign: float,
    wavelength_m: float,
) -> tuple[np.ndarray | None, float]:
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    sv = np.asarray(sat_vel, dtype=np.float64).reshape(-1, 3)
    dop = np.asarray(doppler_hz, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    pos = np.asarray(rx_pos, dtype=np.float64).reshape(-1)
    rows_h: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_w: list[float] = []
    for s in range(dop.size):
        ww = max(float(w[s]), 0.0)
        if ww <= 0.0 or not np.isfinite(ww) or not np.isfinite(dop[s]):
            continue
        los_vec = sat[s] - pos[:3]
        rng = float(np.linalg.norm(los_vec))
        if rng < 1.0 or not np.isfinite(rng):
            continue
        los = los_vec / rng
        rows_h.append(-los)
        rows_y.append(float(doppler_sign) * float(dop[s]) * float(wavelength_m) - float(np.dot(sv[s], los)))
        rows_w.append(ww)
    if len(rows_y) < 4:
        return None, float("nan")
    h = np.asarray(rows_h, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    ww = np.asarray(rows_w, dtype=np.float64)
    sw = float(np.sum(ww))
    if sw <= 0.0:
        return None, float("nan")
    h_center = h - np.sum(h * ww[:, None], axis=0) / sw
    y_center = y - float(np.sum(y * ww) / sw)
    try:
        vel, *_ = np.linalg.lstsq(h_center * np.sqrt(ww[:, None]), y_center * np.sqrt(ww), rcond=None)
    except np.linalg.LinAlgError:
        return None, float("nan")
    rms = _doppler_centered_residual_rms(
        sat,
        sv,
        dop,
        ww,
        pos[:3],
        vel,
        doppler_sign=doppler_sign,
        wavelength_m=wavelength_m,
    )
    return vel, rms


def _doppler_prefit_gate_mask(
    sat_ecef: np.ndarray,
    sat_vel: np.ndarray,
    doppler_hz: np.ndarray,
    weights: np.ndarray,
    rx_pos: np.ndarray,
    *,
    gate_mps: float,
    min_sats: int,
    doppler_sign: float,
    wavelength_m: float,
) -> np.ndarray:
    n = int(np.asarray(doppler_hz).reshape(-1).size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if float(gate_mps) <= 0.0:
        return np.ones(n, dtype=bool)
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    sv = np.asarray(sat_vel, dtype=np.float64).reshape(-1, 3)
    dop = np.asarray(doppler_hz, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    pos = np.asarray(rx_pos, dtype=np.float64).reshape(-1)
    vel, _rms = _doppler_centered_wls_velocity(
        sat,
        sv,
        dop,
        w,
        pos[:3],
        doppler_sign=doppler_sign,
        wavelength_m=wavelength_m,
    )
    if vel is None or not np.all(np.isfinite(vel[:3])):
        return np.ones(n, dtype=bool)

    rows_h: list[np.ndarray] = []
    rows_y: list[float] = []
    rows_w: list[float] = []
    valid_indices: list[int] = []
    for s in range(n):
        ww = max(float(w[s]), 0.0)
        if ww <= 0.0 or not np.isfinite(ww) or not np.isfinite(dop[s]):
            continue
        los_vec = sat[s] - pos[:3]
        rng = float(np.linalg.norm(los_vec))
        if rng < 1.0 or not np.isfinite(rng):
            continue
        los = los_vec / rng
        rows_h.append(-los)
        rows_y.append(float(doppler_sign) * float(dop[s]) * float(wavelength_m) - float(np.dot(sv[s], los)))
        rows_w.append(ww)
        valid_indices.append(s)
    if len(valid_indices) < 4:
        return np.ones(n, dtype=bool)

    h = np.asarray(rows_h, dtype=np.float64)
    y = np.asarray(rows_y, dtype=np.float64)
    ww = np.asarray(rows_w, dtype=np.float64)
    sw = float(np.sum(ww))
    if sw <= 0.0:
        return np.ones(n, dtype=bool)
    h_center = h - np.sum(h * ww[:, None], axis=0) / sw
    y_center = y - float(np.sum(y * ww) / sw)
    residual_abs = np.abs(y_center - h_center @ vel[:3])
    keep_valid = residual_abs <= float(gate_mps)
    min_keep = max(4, min(int(min_sats), len(valid_indices)))
    if int(np.count_nonzero(keep_valid)) < min_keep:
        order = np.argsort(residual_abs)
        keep_valid = np.zeros(len(valid_indices), dtype=bool)
        keep_valid[order[:min_keep]] = True
    out = np.zeros(n, dtype=bool)
    for local_idx, original_idx in enumerate(valid_indices):
        out[int(original_idx)] = bool(keep_valid[local_idx])
    return out


def _resample_deferred(config: CTRBPFConfig) -> bool:
    return bool(config.defer_epoch_resample) or bool(config.enable_reservoir_stein)


def _reservoir_stein_resample_if_needed(
    pf,
    config: CTRBPFConfig,
    stats: _ReservoirSteinStats,
    epoch_index: int,
) -> bool:
    ess = float(pf.get_ess())
    if ess >= float(pf.ess_threshold) * int(pf.n_particles):
        return False

    stats.epochs_attempted += 1
    states = np.asarray(pf.get_particle_states(), dtype=np.float64)
    log_weights = np.asarray(pf.get_log_weights(), dtype=np.float64)
    guide = np.asarray(pf.estimate(), dtype=np.float64)

    sigma_pos = max(float(config.reservoir_stein_guide_sigma_m), 1.0e-6)
    sigma_cb = max(float(config.reservoir_stein_guide_sigma_cb_m), 1.0e-6)
    grad = np.empty((states.shape[0], 4), dtype=np.float64)
    grad[:, :3] = (guide[:3] - states[:, :3]) / (sigma_pos * sigma_pos)
    grad[:, 3] = (guide[3] - states[:, 3]) / (sigma_cb * sigma_cb)

    from gnss_gpu.reservoir_stein import ReservoirSteinConfig, reservoir_stein_update

    reservoir_size = int(config.reservoir_stein_size)
    if reservoir_size <= 0:
        reservoir_size = int(pf.n_particles)
    reservoir_size = max(1, min(reservoir_size, int(pf.n_particles)))
    result = reservoir_stein_update(
        states[:, :4],
        log_weights,
        grad,
        ReservoirSteinConfig(
            reservoir_size=reservoir_size,
            elite_fraction=float(config.reservoir_stein_elite_fraction),
            stein_steps=int(config.reservoir_stein_steps),
            stein_step_size=float(config.reservoir_stein_step_size),
            repulsion_scale=float(config.reservoir_stein_repulsion_scale),
            seed=int(config.reservoir_stein_seed) + int(epoch_index),
        ),
    )

    reservoir_states = states[np.asarray(result.source_indices, dtype=np.int64)].copy()
    reservoir_states[:, :4] = np.asarray(result.particles, dtype=np.float64)
    if reservoir_states.shape[0] == int(pf.n_particles):
        new_states = reservoir_states
    else:
        rng = np.random.default_rng(int(config.reservoir_stein_seed) + 1_000_003 + int(epoch_index))
        probs = np.asarray(result.weights, dtype=np.float64)
        probs = probs / max(float(np.sum(probs)), np.finfo(np.float64).tiny)
        idx = rng.choice(reservoir_states.shape[0], size=int(pf.n_particles), replace=True, p=probs)
        new_states = reservoir_states[idx]
    pf.set_particle_states(new_states)

    stats.epochs_applied += 1
    stats.reservoir_size_sum += int(reservoir_states.shape[0])
    stats.ess_before_sum += float(result.ess_before)
    if result.bandwidths:
        stats.bandwidth_sum += float(result.bandwidths[-1])
    return True


def _apply_rtkdiag_candidate_proposal_cloud(
    pf,
    refs_ecef: np.ndarray,
    mixture_weights: np.ndarray | None,
    spread_m: float,
    seed: int,
) -> None:
    refs = np.asarray(refs_ecef, dtype=np.float64).reshape(-1, 3)
    refs = refs[np.all(np.isfinite(refs), axis=1)]
    if refs.shape[0] == 0:
        return
    if mixture_weights is None:
        weights = np.full(refs.shape[0], 1.0 / refs.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(mixture_weights, dtype=np.float64).reshape(-1)
        if weights.size != refs.shape[0]:
            weights = np.ones(refs.shape[0], dtype=np.float64)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
        total = float(np.sum(weights))
        if total <= 0.0:
            weights = np.full(refs.shape[0], 1.0 / refs.shape[0], dtype=np.float64)
        else:
            weights = weights / total

    states = np.asarray(pf.get_particle_states(), dtype=np.float64)
    log_weights = np.asarray(pf.get_log_weights(), dtype=np.float64)
    state_weights = np.exp(log_weights - float(np.max(log_weights)))
    state_total = float(np.sum(state_weights))
    if state_total <= 0.0 or not np.all(np.isfinite(state_weights)):
        state_weights = None
    else:
        state_weights = state_weights / state_total

    rng = np.random.default_rng(int(seed))
    src_idx = rng.choice(states.shape[0], size=int(pf.n_particles), replace=True, p=state_weights)
    ref_idx = rng.choice(refs.shape[0], size=int(pf.n_particles), replace=True, p=weights)
    new_states = states[src_idx].copy()
    sigma = max(float(spread_m), 1.0e-6)
    new_states[:, :3] = refs[ref_idx] + rng.normal(0.0, sigma, size=(int(pf.n_particles), 3))
    pf.set_particle_states(new_states)


def _ct_spline_motion_prior(
    positions: np.ndarray,
    times: np.ndarray,
    config: CTRBPFConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit a cubic CT trajectory and return inter-epoch motion factors."""
    pos = np.asarray(positions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if pos.ndim != 2 or pos.shape[1] < 3 or t.size != pos.shape[0]:
        return None, None
    valid = (
        np.isfinite(t)
        & np.isfinite(pos[:, :3]).all(axis=1)
        & (np.linalg.norm(pos[:, :3], axis=1) > 1.0)
    )
    if int(np.count_nonzero(valid)) < max(4, int(config.ct_motion_min_epochs)):
        return None, None

    t0 = float(t[valid][0])
    tt = t - t0
    tv = tt[valid]
    pv = pos[valid, :3]
    unique_t, unique_idx = np.unique(tv, return_index=True)
    if unique_t.size < max(4, int(config.ct_motion_min_epochs)):
        return None, None
    pv = pv[unique_idx]

    smooth_sigma = max(float(config.ct_spline_smoothing_m), 0.0)
    smooth_s = (smooth_sigma * smooth_sigma) * float(unique_t.size)
    try:
        from scipy.interpolate import UnivariateSpline  # type: ignore

        ct_pos = np.empty((pos.shape[0], 3), dtype=np.float64)
        for axis in range(3):
            spline = UnivariateSpline(unique_t, pv[:, axis], k=3, s=smooth_s)
            ct_pos[:, axis] = np.asarray(spline(tt), dtype=np.float64)
    except Exception:
        # Fallback: centered moving average on the observed trajectory. This is
        # not a cubic spline, but still supplies a smooth CT-like motion prior.
        ct_pos = pos[:, :3].copy()
        radius = max(1, int(round(max(float(config.ct_spline_smoothing_m), 0.5) * 2.0)))
        for i in range(pos.shape[0]):
            lo = max(0, i - radius)
            hi = min(pos.shape[0], i + radius + 1)
            mask = valid[lo:hi]
            if np.any(mask):
                ct_pos[i] = np.mean(pos[lo:hi, :3][mask], axis=0)

    if not np.isfinite(ct_pos).all():
        return None, None
    deltas = np.diff(ct_pos, axis=0)
    edge_valid = valid[:-1] & valid[1:] & np.isfinite(deltas).all(axis=1)
    deltas = np.where(edge_valid[:, np.newaxis], deltas, np.nan)
    sigmas = np.full(deltas.shape[0], float(config.ct_motion_sigma_m), dtype=np.float64)
    sigmas[~edge_valid] = np.nan
    return deltas, sigmas


def _run_ctrbpf_on_segment(
    data: dict,
    wls_positions: np.ndarray,
    config: CTRBPFConfig,
    wls_quality: dict[str, np.ndarray] | None = None,
    dd_computer=None,
    dd_pr_computer=None,
    hybrid_pos: dict[float, np.ndarray] | None = None,
    hybrid_velocity: dict[float, np.ndarray] | None = None,
    hybrid_status: dict[float, int] | None = None,
    reference_pos: dict[float, np.ndarray] | None = None,
    rtkdiag_candidates: list[
        tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]
    ] | None = None,
    imu: dict[str, np.ndarray] | None = None,
    collect_internal_diagnostics: bool = False,
    ranker_score_lookup: dict[tuple[float, str], float] | None = None,
    ranker_stickiness: float = 0.0,
) -> tuple[
    np.ndarray,
    float,
    _PRObsStats,
    _ReservoirSteinStats,
    _DDStats,
    _RBPFGateStats,
    _HybridStats,
    _RTKDiagPFStats,
    _FGOStats,
    _TDCPSmootherStats,
    _LowSatBridgeStats,
    _ZUPTStats,
    _IMUTCStats,
    _INSTCStats,
    list[dict[str, object]],
]:
    """Run the PF on the loaded PPC segment and return (positions, ms/epoch).

    ``positions`` is shape ``(n_epochs, 3)``, aligned with ``data['times']``.
    """
    n_epochs = int(data["n_epochs"])
    sat_ecef = data["sat_ecef"]
    pseudoranges = data["pseudoranges"]
    weights = data["weights"]
    system_ids = data.get("system_ids")
    sat_velocity = data.get("sat_velocity")
    doppler_hz = data.get("doppler_hz")
    used_prns = data.get("used_prns") or [[] for _ in range(n_epochs)]
    times = np.asarray(data["times"], dtype=np.float64)
    pr_obs_stats = _PRObsStats()
    reservoir_stein_stats = _ReservoirSteinStats()
    dd_stats = _DDStats()
    gate_stats = _RBPFGateStats()
    hybrid_stats = _HybridStats()
    rtkdiag_pf_stats = _RTKDiagPFStats()
    gate_active = config.enable_rbpf_velocity_kf and (
        config.rbpf_kf_gate_min_dd_pairs is not None
        or config.rbpf_kf_gate_min_ess_ratio is not None
        or config.rbpf_kf_gate_max_spread_m is not None
        or float(config.rbpf_kf_gate_max_doppler_wls_rms_mps) > 0.0
        or float(config.rbpf_kf_gate_max_doppler_wls_speed_mps) > 0.0
    )
    need_dd_compute = (
        config.enable_dd_carrier_afv
        or (gate_active and config.rbpf_kf_gate_min_dd_pairs is not None)
        or config.enable_fgo_lambda
    )
    use_hybrid = config.enable_hybrid_pu and hybrid_pos is not None
    use_rtkdiag_pf = (
        bool(config.enable_rtkdiag_pf_rescue)
        and rtkdiag_candidates is not None
        and len(rtkdiag_candidates) > 0
    )
    use_vguide = (
        config.enable_hybrid_velocity_guide
        and hybrid_velocity is not None
        and len(hybrid_velocity) > 0
    )
    fgo_dd_cache: list = [None] * n_epochs  # holds DDCarrierEpoch | None
    fgo_dd_pr_cache: list = [None] * n_epochs  # holds DDPseudorangeEpoch | None
    fgo_dd_pr_ls_anchor_pos = np.full((n_epochs, 3), np.nan, dtype=np.float64)
    fgo_dd_pr_ls_anchor_sigma = np.full(n_epochs, np.nan, dtype=np.float64)
    fgo_stats = _FGOStats()
    tdcp_stats = _TDCPSmootherStats()
    low_sat_bridge_stats = _LowSatBridgeStats()
    zupt_stats = _ZUPTStats()
    imu_tc_stats = _IMUTCStats()
    ins_tc_stats = _INSTCStats()
    defer_resample = _resample_deferred(config)
    internal_diagnostics: list[dict[str, object]] = []

    # Phase 9b: tight-coupled IMU state (in-loop pre-integration since the
    # last Status=4 hybrid anchor). Initialized lazily on the first epoch
    # we have a usable position + heading.
    if config.enable_imu_tc and config.enable_ins_tc:
        raise ValueError("enable_imu_tc and enable_ins_tc are mutually exclusive")
    use_imu_tc = config.enable_imu_tc and imu is not None
    use_ins_tc = config.enable_ins_tc and imu is not None
    imu_t = _build_imu_segment_index(imu) if (use_imu_tc or use_ins_tc) else None
    if use_imu_tc and imu_t is None:
        use_imu_tc = False  # IMU file present but empty
    if use_ins_tc and imu_t is None:
        use_ins_tc = False
    imu_tc_anchor: _IMUAnchor | None = None
    imu_tc_anchor_acc: np.ndarray | None = None
    imu_tc_anchor_gyro: np.ndarray | None = None
    imu_tc_emit_set = set(int(s) for s in config.imu_tc_emit_pf_hybrid_statuses)
    imu_tc_anchor_set = set(int(s) for s in config.zupt_apply_hybrid_statuses)
    if use_imu_tc and hybrid_status is not None:
        # Pre-compute per-epoch IMU stats so the anchor-static check matches
        # the Phase 9a ZUPT logic (re-using the same windowing).
        imu_tc_anchor_acc, imu_tc_anchor_gyro, _ = _build_imu_per_epoch_stats(imu, times)

    ins_ekf = None
    ins_origin_ecef: np.ndarray | None = None
    ins_origin_lat: float | None = None
    ins_origin_lon: float | None = None
    ins_last_obs_t: float | None = None
    ins_particle_imu_initialized = False
    ins_particle_imu_last_t: float | None = None
    ins_particle_imu_last_accel: np.ndarray | None = None
    ins_particle_imu_last_gyro_dps: np.ndarray | None = None
    ins_gravity_ecef: np.ndarray | None = None
    ins_particle_imu_available = bool(config.ins_tc_use_particle_imu_predict)
    ins_tc_emit_set = set(int(s) for s in config.ins_tc_emit_pf_hybrid_statuses)
    if use_ins_tc:
        from gnss_gpu.ins_ekf import INSEKF, INSConfig

        ins_ekf = INSEKF(
            INSConfig(
                static_acc_low=float(config.ins_tc_align_acc_low),
                static_acc_high=float(config.ins_tc_align_acc_high),
                static_gyro_max_dps=float(config.ins_tc_align_gyro_max_dps),
                align_min_static_samples=int(config.ins_tc_align_min_samples),
                yaw_init_min_speed_mps=float(config.ins_tc_yaw_init_min_speed_mps),
            )
        )

    positions = np.zeros((n_epochs, 3), dtype=np.float64)
    pr_used_counts = np.zeros(n_epochs, dtype=np.int32)
    init_tow = round(float(times[0]), 1)
    init_ref = reference_pos.get(init_tow) if reference_pos is not None else None
    wls_init_pos = np.asarray(wls_positions[0, :3], dtype=np.float64)
    wls_init_err_e, wls_init_err_n, wls_init_err_u = _diag_enu_error(wls_init_pos, init_ref)
    init_pos = np.asarray(wls_init_pos, dtype=np.float64)
    init_cb = float(wls_positions[0, 3])
    init_spread = float(config.spread_pos_init)
    init_source = "wls"
    hybrid_init_pos: np.ndarray | None = None
    hybrid_init_available = False
    if use_hybrid:
        hybrid_init_pos = hybrid_pos.get(init_tow)
        if hybrid_init_pos is not None and np.all(np.isfinite(hybrid_init_pos)):
            init_pos = np.asarray(hybrid_init_pos, dtype=np.float64)
            init_source = "hybrid"
            hybrid_init_available = True
            # Hybrid is already cm-class for "fix" status epochs and m-class
            # for float, so a 5m spread captures both regimes.
            init_spread = max(5.0, float(config.hybrid_sigma_m) * 5.0)
    init_err_e, init_err_n, init_err_u = _diag_enu_error(init_pos, init_ref)
    hybrid_init_err_e, hybrid_init_err_n, hybrid_init_err_u = _diag_enu_error(hybrid_init_pos, init_ref)
    init_diag = {
        "pf_init_source": init_source,
        "pf_init_spread_pos_m": float(init_spread),
        "pf_init_spread_cb_m": float(config.spread_cb_init),
        "pf_init_velocity_sigma_m": (
            float(config.velocity_init_sigma) if config.enable_rbpf_velocity_kf else 0.0
        ),
        "pf_init_clock_bias_m": float(init_cb),
        "pf_init_x": float(init_pos[0]),
        "pf_init_y": float(init_pos[1]),
        "pf_init_z": float(init_pos[2]),
        "pf_init_to_ref_m": _diag_distance(init_pos, init_ref),
        "pf_init_err_e_m": init_err_e,
        "pf_init_err_n_m": init_err_n,
        "pf_init_err_u_m": init_err_u,
        "wls_init_clock_bias_m": float(init_cb),
        "wls_init_to_ref_m": _diag_distance(wls_init_pos, init_ref),
        "wls_init_err_e_m": wls_init_err_e,
        "wls_init_err_n_m": wls_init_err_n,
        "wls_init_err_u_m": wls_init_err_u,
        "hybrid_init_available": bool(hybrid_init_available),
        "hybrid_init_to_ref_m": _diag_distance(hybrid_init_pos, init_ref),
        "hybrid_init_err_e_m": hybrid_init_err_e,
        "hybrid_init_err_n_m": hybrid_init_err_n,
        "hybrid_init_err_u_m": hybrid_init_err_u,
    }

    pf = _build_pf(config)
    pf.initialize(
        init_pos,
        clock_bias=init_cb,
        spread_pos=init_spread,
        spread_cb=config.spread_cb_init,
        velocity_init_sigma=config.velocity_init_sigma if config.enable_rbpf_velocity_kf else 0.0,
    )
    rtkdiag_last_good_pos: np.ndarray | None = None
    rtkdiag_last_good_t: float | None = None
    rtkdiag_last_fix4_pos: np.ndarray | None = None
    rtkdiag_last_fix4_t: float | None = None
    rtkdiag_temporal_prev: np.ndarray | None = None
    rtkdiag_temporal_prev_hybrid: np.ndarray | None = None

    # Rolling window of hybrid statuses, used by the ins_tc quality gate.
    # Each entry is the int hybrid_status for that epoch (or 0 when unknown).
    from collections import deque as _deque
    ins_tc_quality_window: _deque[int] = _deque(
        maxlen=max(1, int(config.ins_tc_quality_gate_window_epochs))
    )
    ins_tc_quality_gate_skip_count = 0

    def _wls_fallback_ok(idx: int, pf_pos: np.ndarray) -> bool:
        if idx < 0 or idx >= int(wls_positions.shape[0]):
            return False
        wls_pos = np.asarray(wls_positions[idx, :3], dtype=np.float64)
        if not np.all(np.isfinite(wls_pos)) or np.all(wls_pos == 0.0):
            return False
        if wls_quality is not None:
            max_rms = float(config.rtkdiag_candidate_fallback_max_wls_rms_m)
            if max_rms > 0.0:
                rms = float("nan")
                if idx < len(wls_quality.get("postfit_rms_m", ())):
                    rms = float(wls_quality["postfit_rms_m"][idx])
                if not np.isfinite(rms) or rms > max_rms:
                    return False
            max_pdop = float(config.rtkdiag_candidate_fallback_max_wls_pdop)
            if max_pdop > 0.0:
                pdop = float("nan")
                if idx < len(wls_quality.get("pdop", ())):
                    pdop = float(wls_quality["pdop"][idx])
                if not np.isfinite(pdop) or pdop > max_pdop:
                    return False
        max_to_pf = float(config.rtkdiag_candidate_fallback_max_wls_to_pf_m)
        if max_to_pf > 0.0:
            pf_arr = np.asarray(pf_pos, dtype=np.float64).reshape(-1)[:3]
            if (
                not np.all(np.isfinite(pf_arr))
                or float(np.linalg.norm(wls_pos - pf_arr)) > max_to_pf
            ):
                return False
        return True

    def _select_rtkdiag_fallback(
        idx: int,
        tow: float,
        pf_pos: np.ndarray,
        *,
        suffix: str,
    ) -> tuple[np.ndarray, str]:
        nonlocal rtkdiag_last_good_pos, rtkdiag_last_good_t
        mode = str(config.rtkdiag_candidate_fallback_mode).strip().lower()
        pf_arr = np.asarray(pf_pos, dtype=np.float64).reshape(-1)[:3]
        if mode in {"hybrid", "hybrid-last-good"} and hp_prefetched_valid:
            rtkdiag_pf_stats.fallback_hybrid += 1
            return np.asarray(hp_prefetched, dtype=np.float64), (
                f"rtkdiag_fallback_hybrid{suffix}"
            )
        if mode in {"wls", "quality-wls", "wls-last-good", "quality-wls-last-good"}:
            if mode == "wls" or _wls_fallback_ok(idx, pf_arr):
                wls_pos = np.asarray(wls_positions[idx, :3], dtype=np.float64)
                if np.all(np.isfinite(wls_pos)) and not np.all(wls_pos == 0.0):
                    rtkdiag_pf_stats.fallback_wls += 1
                    return wls_pos, f"rtkdiag_fallback_wls{suffix}"
        if mode in {"last-good", "wls-last-good", "quality-wls-last-good", "hybrid-last-good"}:
            max_age = float(config.rtkdiag_candidate_fallback_max_hold_age_s)
            age_ok = (
                rtkdiag_last_good_pos is not None
                and rtkdiag_last_good_t is not None
                and np.all(np.isfinite(rtkdiag_last_good_pos))
                and (max_age <= 0.0 or float(tow) - float(rtkdiag_last_good_t) <= max_age)
            )
            if age_ok:
                rtkdiag_pf_stats.fallback_last_good += 1
                return np.asarray(rtkdiag_last_good_pos, dtype=np.float64), (
                    f"rtkdiag_fallback_last_good{suffix}"
                )
        rtkdiag_pf_stats.fallback_pf += 1
        return pf_arr, f"pf{suffix}"

    t0 = time.perf_counter()
    _prev_ranker_label: str | None = None
    for i in range(n_epochs):
        t_now = float(times[i])
        t_key = round(t_now, 1)
        hp_prefetched = hybrid_pos.get(t_key) if use_hybrid else None
        hp_prefetched_valid = (
            hp_prefetched is not None
            and np.all(np.isfinite(hp_prefetched))
            and not np.all(np.asarray(hp_prefetched, dtype=np.float64) == 0.0)
        )

        if i == 0:
            dt = float(data.get("dt", 0.2))
        else:
            dt = float(times[i] - times[i - 1])
            if not np.isfinite(dt) or dt <= 0.0:
                dt = float(data.get("dt", 0.2))
        st_diag = hybrid_status.get(t_key) if hybrid_status is not None else None
        ref_diag_epoch = reference_pos.get(t_key) if reference_pos is not None else None
        pf_diag_row: dict[str, object] | None = None
        if collect_internal_diagnostics:
            pf_diag_row = {
                "epoch": int(i),
                "tow": float(t_now),
                "dt": float(dt),
                "hybrid_status": int(st_diag) if st_diag is not None else "",
                "hybrid_available": bool(hp_prefetched_valid),
                "hybrid_to_ref_m": _diag_distance(hp_prefetched, ref_diag_epoch),
            }
            pf_diag_row.update(init_diag)
            if ref_diag_epoch is not None and np.all(np.isfinite(ref_diag_epoch[:3])):
                pf_diag_row["ref_x"] = float(ref_diag_epoch[0])
                pf_diag_row["ref_y"] = float(ref_diag_epoch[1])
                pf_diag_row["ref_z"] = float(ref_diag_epoch[2])
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_epoch_start",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        ins_epoch_prepared = False
        if use_ins_tc and ins_ekf is not None:
            if ins_origin_ecef is None and hp_prefetched_valid:
                ins_origin_ecef = np.asarray(hp_prefetched, dtype=np.float64).copy()
                lat, lon, _ = _ecef_to_llh(
                    float(ins_origin_ecef[0]),
                    float(ins_origin_ecef[1]),
                    float(ins_origin_ecef[2]),
                )
                ins_origin_lat = lat
                ins_origin_lon = lon
                ins_gravity_ecef = _gravity_ecef_at_origin(lat, lon)
                ins_ekf.initialize_position(np.zeros(3, dtype=np.float64))
                ins_last_obs_t = t_now

            if (
                ins_origin_ecef is not None
                and ins_origin_lat is not None
                and ins_origin_lon is not None
            ):
                ins_tc_stats.epochs_evaluated += 1
                if i > 0 and imu_t is not None and imu is not None:
                    imu_window = _slice_imu_samples(
                        imu,
                        imu_t,
                        float(times[i - 1]),
                        t_now,
                    )
                else:
                    imu_window = np.empty((0, 7), dtype=np.float64)

                if not ins_ekf.aligned:
                    for sample in imu_window:
                        ins_ekf.feed_imu_for_alignment(sample[0], sample[1:4], sample[4:7])
                    if ins_ekf.aligned:
                        if ins_tc_stats.aligned_at_epoch < 0:
                            ins_tc_stats.aligned_at_epoch = i
                        if hp_prefetched_valid:
                            ins_ekf.initialize_position(
                                _ecef_to_enu_at_origin(
                                    np.asarray(hp_prefetched, dtype=np.float64),
                                    ins_origin_ecef,
                                    ins_origin_lat,
                                    ins_origin_lon,
                                )
                            )
                            ins_last_obs_t = t_now

                if ins_ekf.aligned and imu_window.size > 0:
                    ins_ekf.propagate(imu_window)

                if ins_ekf.aligned and not ins_ekf.yaw_initialized:
                    v_guide_now = hybrid_velocity.get(t_key) if hybrid_velocity else None
                    if v_guide_now is not None and np.all(np.isfinite(v_guide_now)):
                        v_enu = _ecef_velocity_to_enu_at_origin(
                            np.asarray(v_guide_now, dtype=np.float64),
                            ins_origin_lat,
                            ins_origin_lon,
                        )
                        if ins_ekf.initialize_yaw_from_velocity(v_enu):
                            if ins_tc_stats.yaw_initialized_at_epoch < 0:
                                ins_tc_stats.yaw_initialized_at_epoch = i

                ins_epoch_prepared = True

        pf_predict_done = False
        if (
            use_ins_tc
            and ins_particle_imu_available
            and ins_ekf is not None
            and ins_ekf.aligned
            and ins_ekf.yaw_initialized
            and ins_origin_lat is not None
            and ins_origin_lon is not None
            and ins_gravity_ecef is not None
        ):
            if not ins_particle_imu_initialized:
                q_body_to_ecef = ins_ekf.attitude_quat_body_to_ecef(
                    ins_origin_lat,
                    ins_origin_lon,
                )
                v_body_ecef = ins_ekf.velocity_ecef(ins_origin_lat, ins_origin_lon)
                try:
                    pf.set_inertial_state(
                        q_body_to_ecef,
                        ins_ekf.accel_bias_body(),
                        ins_ekf.gyro_bias_body_radps(),
                        v_body_ecef,
                        attitude_spread_rad=float(config.ins_tc_particle_imu_att_spread_rad),
                        accel_bias_spread=float(config.ins_tc_particle_imu_acc_bias_spread),
                        gyro_bias_spread=float(config.ins_tc_particle_imu_gyro_bias_spread_rps),
                        velocity_spread=float(config.ins_tc_particle_imu_velocity_spread_mps),
                    )
                    ins_particle_imu_initialized = True
                    ins_tc_stats.particle_imu_initialized += 1
                    ins_particle_imu_last_t = t_now
                    if imu_window.size > 0:
                        last_sample = imu_window[-1]
                        ins_particle_imu_last_accel = np.asarray(last_sample[1:4], dtype=np.float64)
                        ins_particle_imu_last_gyro_dps = np.asarray(last_sample[4:7], dtype=np.float64)
                except RuntimeError:
                    ins_particle_imu_available = False
            elif ins_particle_imu_initialized:
                if imu_window.size > 0:
                    for sample in imu_window:
                        sample_t = float(sample[0])
                        sample_accel = np.asarray(sample[1:4], dtype=np.float64)
                        sample_gyro_dps = np.asarray(sample[4:7], dtype=np.float64)
                        if (
                            ins_particle_imu_last_t is not None
                            and ins_particle_imu_last_accel is not None
                            and ins_particle_imu_last_gyro_dps is not None
                            and sample_t > ins_particle_imu_last_t
                        ):
                            dt_imu = sample_t - float(ins_particle_imu_last_t)
                            pf.predict_imu(
                                ins_particle_imu_last_accel,
                                ins_particle_imu_last_gyro_dps * (math.pi / 180.0),
                                ins_gravity_ecef,
                                dt_imu,
                                sigma_pos=float(config.ins_tc_particle_imu_sigma_pos_m),
                                sigma_acc=float(config.ins_tc_particle_imu_sigma_acc_mps2),
                                sigma_gyro=float(config.ins_tc_particle_imu_sigma_gyro_rps),
                                sigma_acc_bias_rw=float(config.ins_tc_particle_imu_acc_bias_rw),
                                sigma_gyro_bias_rw=float(config.ins_tc_particle_imu_gyro_bias_rw),
                            )
                            ins_tc_stats.particle_imu_predict_used += 1
                            pf_predict_done = True
                        ins_particle_imu_last_t = sample_t
                        ins_particle_imu_last_accel = sample_accel
                        ins_particle_imu_last_gyro_dps = sample_gyro_dps
                if (
                    ins_particle_imu_last_t is not None
                    and ins_particle_imu_last_accel is not None
                    and ins_particle_imu_last_gyro_dps is not None
                    and t_now > ins_particle_imu_last_t
                ):
                    pf.predict_imu(
                        ins_particle_imu_last_accel,
                        ins_particle_imu_last_gyro_dps * (math.pi / 180.0),
                        ins_gravity_ecef,
                        t_now - float(ins_particle_imu_last_t),
                        sigma_pos=float(config.ins_tc_particle_imu_sigma_pos_m),
                        sigma_acc=float(config.ins_tc_particle_imu_sigma_acc_mps2),
                        sigma_gyro=float(config.ins_tc_particle_imu_sigma_gyro_rps),
                        sigma_acc_bias_rw=float(config.ins_tc_particle_imu_acc_bias_rw),
                        sigma_gyro_bias_rw=float(config.ins_tc_particle_imu_gyro_bias_rw),
                    )
                    ins_tc_stats.particle_imu_predict_used += 1
                    pf_predict_done = True
                    ins_particle_imu_last_t = t_now

        v_guide = None
        v_guide_from_ins = False
        if (
            not pf_predict_done
            and
            use_ins_tc
            and bool(config.ins_tc_use_motion_predict)
            and ins_ekf is not None
            and ins_ekf.aligned
            and ins_ekf.yaw_initialized
            and ins_origin_lat is not None
            and ins_origin_lon is not None
        ):
            v_ins_ecef = _enu_velocity_to_ecef_at_origin(
                ins_ekf.velocity_enu(),
                ins_origin_lat,
                ins_origin_lon,
            )
            if np.all(np.isfinite(v_ins_ecef)):
                v_guide = v_ins_ecef
                v_guide_from_ins = True
                ins_tc_stats.motion_predict_used += 1
        if use_vguide:
            if v_guide is None:
                v_guide = hybrid_velocity.get(round(float(times[i]), 1))
        if pf_predict_done:
            pass
        elif v_guide is not None and np.all(np.isfinite(v_guide)):
            pf.predict(
                velocity=np.asarray(v_guide, dtype=np.float64),
                dt=dt,
                sigma_pos=(
                    float(config.ins_tc_predict_sigma_pos_m)
                    if v_guide_from_ins
                    else None
                ),
                velocity_guide_alpha=(
                    float(config.ins_tc_predict_velocity_alpha)
                    if v_guide_from_ins
                    else None
                ),
                rbpf_velocity_kf=config.enable_rbpf_velocity_kf,
                velocity_process_noise=config.velocity_process_noise,
            )
        else:
            pf.predict(
                dt=dt,
                rbpf_velocity_kf=config.enable_rbpf_velocity_kf,
                velocity_process_noise=config.velocity_process_noise,
            )
        if pf_diag_row is not None:
            pf_diag_row["predict_source"] = (
                "particle_imu"
                if pf_predict_done
                else ("ins_motion" if v_guide_from_ins else ("hybrid_velocity" if v_guide is not None else "process"))
            )
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_predict",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        sat_i = np.asarray(sat_ecef[i], dtype=np.float64)
        pr_i = np.asarray(pseudoranges[i], dtype=np.float64)
        w_i = np.asarray(weights[i], dtype=np.float64)
        sids_i_full = None if system_ids is None else np.asarray(system_ids[i], dtype=np.int32)
        if config.pr_systems and sids_i_full is not None:
            allowed_pr_ids = {
                sid for sid, ch in _SYS_ID_TO_CHAR.items() if ch in set(config.pr_systems)
            }
            pr_sys_mask = np.array([int(sid) in allowed_pr_ids for sid in sids_i_full], dtype=bool)
            sat_i = sat_i[pr_sys_mask]
            pr_i = pr_i[pr_sys_mask]
            w_i = w_i[pr_sys_mask]
            sids_i_full = sids_i_full[pr_sys_mask]
        if system_ids is not None:
            w_i = _scale_weights_per_system(w_i, sids_i_full)
        w_i = _pr_likelihood_weights(w_i, config)

        finite = (
            np.all(np.isfinite(sat_i), axis=1)
            & np.isfinite(pr_i)
            & np.isfinite(w_i)
            & (pr_i > 1e6)
        )
        if int(finite.sum()) < 4:
            pr_used_counts[i] = int(finite.sum())
            est = np.asarray(pf.estimate(), dtype=np.float64)
            if use_rtkdiag_pf:
                positions[i], _min_sat_source = _select_rtkdiag_fallback(
                    i,
                    t_now,
                    est[:3],
                    suffix="_min_sats",
                )
            else:
                positions[i] = est[:3]
                _min_sat_source = "pf_min_sats"
            if pf_diag_row is not None:
                pf_diag_row["n_sat_raw"] = int(len(pr_i))
                pf_diag_row["n_sat_used_pr"] = int(finite.sum())
                pf_diag_row["pr_update_mode"] = "skip_min_sats"
                pf_diag_row["emitted_source"] = _min_sat_source
                pf_diag_row["emit_to_pf_estimate_m"] = _diag_distance(
                    positions[i],
                    est[:3],
                )
                pf_diag_row["emit_to_hybrid_m"] = _diag_distance(positions[i], hp_prefetched)
                pf_diag_row["emit_to_ref_m"] = _diag_distance(positions[i], ref_diag_epoch)
                internal_diagnostics.append(pf_diag_row)
            if _min_sat_source.startswith("rtkdiag_fallback_wls") and np.all(
                np.isfinite(positions[i])
            ):
                rtkdiag_last_good_pos = np.asarray(positions[i], dtype=np.float64).copy()
                rtkdiag_last_good_t = t_now
            continue
        sat_i = sat_i[finite]
        pr_i = pr_i[finite]
        w_i = w_i[finite]
        sids_i = None if sids_i_full is None else np.asarray(sids_i_full)[finite]
        pr_sagnac_ref = np.asarray(pf.estimate(), dtype=np.float64)[:3]
        if (
            float(config.pr_min_elevation_deg) > -90.0
            and np.all(np.isfinite(pr_sagnac_ref))
        ):
            min_elev_rad = math.radians(float(config.pr_min_elevation_deg))
            elev_mask = np.array(
                [_elevation_rad(pr_sagnac_ref, sat_row) >= min_elev_rad for sat_row in sat_i],
                dtype=bool,
            )
            if int(np.count_nonzero(elev_mask)) >= 4:
                if pf_diag_row is not None:
                    pf_diag_row["pr_elevation_gate_deg"] = float(config.pr_min_elevation_deg)
                    pf_diag_row["pr_elevation_kept_count"] = int(np.count_nonzero(elev_mask))
                    pf_diag_row["pr_elevation_dropped_count"] = int(elev_mask.size - np.count_nonzero(elev_mask))
                sat_i = sat_i[elev_mask]
                pr_i = pr_i[elev_mask]
                w_i = w_i[elev_mask]
                if sids_i is not None:
                    sids_i = sids_i[elev_mask]
            elif pf_diag_row is not None:
                pf_diag_row["pr_elevation_gate_deg"] = float(config.pr_min_elevation_deg)
                pf_diag_row["pr_elevation_kept_count"] = int(np.count_nonzero(elev_mask))
                pf_diag_row["pr_elevation_dropped_count"] = int(elev_mask.size - np.count_nonzero(elev_mask))
        pr_used_counts[i] = int(len(pr_i))
        atmosphere_model = str(config.pr_atmosphere_model).strip().lower()
        if atmosphere_model == "off" and float(config.pr_slant_delay_zenith_m) > 0.0:
            atmosphere_model = "fixed"
        if atmosphere_model not in {"", "off"} and np.all(np.isfinite(pr_sagnac_ref)):
            atmosphere_delay = np.array(
                [
                    _pr_atmosphere_delay_m(
                        pr_sagnac_ref,
                        sat_row,
                        t_now,
                        model=atmosphere_model,
                        fixed_zenith_m=float(config.pr_slant_delay_zenith_m),
                        model_scale=float(config.pr_atmosphere_scale),
                        extra_zenith_m=float(config.pr_atmosphere_extra_zenith_m),
                        iono_alpha=tuple(config.pr_iono_alpha),
                        iono_beta=tuple(config.pr_iono_beta),
                    )
                    for sat_row in sat_i
                ],
                dtype=np.float64,
            )
            pr_i = pr_i - atmosphere_delay
            if pf_diag_row is not None:
                pf_diag_row["pr_atmosphere_model"] = atmosphere_model
                pf_diag_row["pr_atmosphere_scale"] = float(config.pr_atmosphere_scale)
                pf_diag_row["pr_atmosphere_extra_zenith_m"] = float(
                    config.pr_atmosphere_extra_zenith_m
                )
                pf_diag_row["pr_slant_delay_zenith_m"] = float(
                    config.pr_slant_delay_zenith_m
                )
                pf_diag_row["pr_atmosphere_delay_median_m"] = float(
                    np.median(atmosphere_delay)
                )
        if np.all(np.isfinite(pr_sagnac_ref)):
            sat_i = rotate_satellites_sagnac(pr_sagnac_ref, sat_i)
            if pf_diag_row is not None:
                pf_diag_row["pr_sagnac_applied"] = True
        elif pf_diag_row is not None:
            pf_diag_row["pr_sagnac_applied"] = False
        st_obs = hybrid_status.get(t_key) if hybrid_status is not None else None
        skip_pr_here = (
            st_obs is not None
            and int(st_obs) in {int(s) for s in config.pr_skip_statuses}
        )

        clock_quantile = (
            float(config.pr_gmm_clock_quantile)
            if bool(config.enable_pr_gmm)
            else 0.5
        )
        if (not skip_pr_here) and float(config.pr_prefit_gate_m) > 0.0:
            ref_mode = str(config.pr_prefit_ref).strip().lower()
            if ref_mode == "hybrid" and hp_prefetched_valid:
                prefit_ref = np.asarray(hp_prefetched, dtype=np.float64)
            else:
                prefit_ref = np.asarray(pf.estimate(), dtype=np.float64)[:3]
            gate_mask = _pr_prefit_gate_mask(
                sat_i,
                pr_i,
                prefit_ref,
                gate_m=float(config.pr_prefit_gate_m),
                clock_quantile=clock_quantile,
                min_sats=int(config.pr_prefit_gate_min_sats),
                keep_best=int(config.pr_prefit_gate_keep_best),
                system_ids=sids_i,
                per_system=bool(config.pr_prefit_per_system),
            )
            if int(gate_mask.sum()) >= 4 and int(gate_mask.sum()) < int(gate_mask.size):
                pr_obs_stats.prefit_epochs += 1
                pr_obs_stats.prefit_sats_kept += int(gate_mask.sum())
                pr_obs_stats.prefit_sats_dropped += int(gate_mask.size - gate_mask.sum())
                sat_i = sat_i[gate_mask]
                pr_i = pr_i[gate_mask]
                w_i = w_i[gate_mask]
                if sids_i is not None:
                    sids_i = sids_i[gate_mask]
                pr_used_counts[i] = int(len(pr_i))

        if (not skip_pr_here) and config.enable_correct_clock_bias and i % 5 == 0:
            pf.correct_clock_bias(sat_i, pr_i, quantile=clock_quantile)

        use_pr_gmm_here = bool(config.enable_pr_gmm)
        if use_pr_gmm_here and st_obs is not None and config.pr_gmm_statuses:
            use_pr_gmm_here = int(st_obs) in {int(s) for s in config.pr_gmm_statuses}
        pr_ess_guard_enabled = (
            (not skip_pr_here)
            and float(config.pr_ess_guard_min_ratio) > 0.0
        )
        pre_pr_log_weights = (
            np.asarray(pf.get_log_weights(), dtype=np.float64)
            if pr_ess_guard_enabled
            else None
        )
        if pf_diag_row is not None:
            pf_diag_row["pr_ess_guard_enabled"] = bool(pr_ess_guard_enabled)
            if pre_pr_log_weights is not None:
                pf_diag_row["pr_ess_guard_pre_ratio"] = _ess_ratio_from_log_weights(pre_pr_log_weights)
        if skip_pr_here:
            pr_obs_stats.epochs_skipped += 1
            if pf_diag_row is not None:
                pf_diag_row["pr_update_mode"] = "skipped_status"
        elif use_pr_gmm_here:
            pf.update_gmm(
                sat_i,
                pr_i,
                weights=w_i,
                sigma_pr=float(config.sigma_pr),
                w_los=float(config.pr_gmm_w_los),
                mu_nlos=float(config.pr_gmm_mu_nlos_m),
                sigma_nlos=float(config.pr_gmm_sigma_nlos_m),
                resample=(not defer_resample) and not pr_ess_guard_enabled,
            )
            pr_obs_stats.epochs_gmm += 1
            if pf_diag_row is not None:
                pf_diag_row["pr_update_mode"] = "gmm"
        else:
            pf.update(
                sat_i,
                pr_i,
                weights=w_i,
                resample=(not defer_resample) and not pr_ess_guard_enabled,
            )
            pr_obs_stats.epochs_gaussian += 1
            if pf_diag_row is not None:
                pf_diag_row["pr_update_mode"] = "gaussian"
        if pr_ess_guard_enabled and pre_pr_log_weights is not None:
            alpha, post_ratio, final_ratio = _apply_pr_ess_guard(
                pf,
                pre_pr_log_weights,
                min_ratio=float(config.pr_ess_guard_min_ratio),
                max_iters=int(config.pr_ess_guard_max_iters),
            )
            if pf_diag_row is not None:
                pf_diag_row["pr_ess_guard_alpha"] = alpha
                pf_diag_row["pr_ess_guard_post_ratio"] = post_ratio
                pf_diag_row["pr_ess_guard_final_ratio"] = final_ratio
            if not defer_resample:
                _ = pf.resample_if_needed()
        if pf_diag_row is not None:
            pf_diag_row["n_sat_raw"] = int(len(sids_i_full)) if sids_i_full is not None else int(len(pr_i))
            pf_diag_row["n_sat_used_pr"] = int(len(pr_i))
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_pr",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        # Phase 1+2: compute DD once (cached) so both the Doppler-KF gate and
        # the AFV update can read its pair count.
        dd_result = None
        if need_dd_compute and dd_computer is not None:
            rover_pos_now = np.asarray(pf.estimate(), dtype=np.float64)[:3]
            dd_select_pos = (
                np.asarray(hp_prefetched, dtype=np.float64)[:3]
                if hp_prefetched_valid
                else rover_pos_now
            )
            sat_full = np.asarray(sat_ecef[i], dtype=np.float64)
            sids_full = (
                np.asarray(system_ids[i], dtype=np.int32)
                if system_ids is not None
                else np.zeros(int(sat_full.shape[0]), dtype=np.int32)
            )
            sat_id_strs = list(used_prns[i]) if i < len(used_prns) else []
            w_full_meas = np.asarray(weights[i], dtype=np.float64)
            measurements = _build_dd_measurements(
                sat_full,
                sids_full,
                sat_id_strs,
                w_full_meas,
                dd_select_pos,
                config.dd_systems,
                min_elevation_deg=float(config.dd_min_elevation_deg),
                min_snr=float(config.dd_min_snr),
                keep_best=int(config.dd_keep_best),
            )
            if len(measurements) >= 2:
                dd_result = dd_computer.compute_dd(
                    float(times[i]),
                    measurements,
                    rover_position_approx=dd_select_pos,
                    min_common_sats=config.dd_min_pairs,
                )
                # Phase 4 cache: save the DD carrier observation for the
                # post-process FGO + LAMBDA pass after the loop.
                if dd_result is not None and int(getattr(dd_result, "n_dd", 0)) > 0:
                    if config.enable_fgo_lambda:
                        from gnss_gpu.local_fgo import DDCarrierEpoch

                        fgo_dd_cache[i] = DDCarrierEpoch.from_result(dd_result)
                    # Phase 4 v2 / realtime DD-PR: cache DD pseudorange for
                    # FGO and optionally turn the same robust LS anchor into a
                    # PF soft position update. DD carrier alone is relative;
                    # DD-PR supplies an absolute anchor when PR/WLS is biased.
                    if dd_pr_computer is not None and (
                        config.enable_fgo_lambda or config.enable_dd_pr_ls_anchor
                    ):
                        try:
                            dd_pr_result = dd_pr_computer.compute_dd(
                                float(times[i]),
                                measurements,
                                rover_position_approx=dd_select_pos,
                                min_common_sats=config.dd_min_pairs,
                                rover_weights=[float(m.snr) for m in measurements],
                            )
                        except Exception:
                            dd_pr_result = None
                        if (
                            dd_pr_result is not None
                            and (
                                float(config.dd_pr_pair_residual_max_m) > 0.0
                                or float(config.dd_pr_epoch_median_residual_max_m) > 0.0
                            )
                        ):
                            from gnss_gpu.dd_quality import gate_dd_pseudorange

                            dd_pr_result, _dd_pr_gate_stats = gate_dd_pseudorange(
                                dd_pr_result,
                                dd_select_pos,
                                pair_residual_max_m=(
                                    float(config.dd_pr_pair_residual_max_m)
                                    if float(config.dd_pr_pair_residual_max_m) > 0.0
                                    else None
                                ),
                                epoch_median_residual_max_m=(
                                    float(config.dd_pr_epoch_median_residual_max_m)
                                    if float(config.dd_pr_epoch_median_residual_max_m) > 0.0
                                    else None
                                ),
                                min_pairs=max(1, int(config.dd_pr_gate_min_pairs)),
                            )
                        if (
                            dd_pr_result is not None
                            and int(getattr(dd_pr_result, "n_dd", 0)) > 0
                        ):
                            if bool(config.enable_dd_pr_ls_anchor):
                                status_ok = True
                                if config.dd_pr_ls_anchor_statuses and st_obs is not None:
                                    status_ok = int(st_obs) in {
                                        int(s) for s in config.dd_pr_ls_anchor_statuses
                                    }
                                if not status_ok:
                                    fgo_stats.dd_pr_ls_anchor_status_skipped += 1
                                else:
                                    fgo_stats.dd_pr_ls_anchor_attempted += 1
                                    try:
                                        from gnss_gpu.gsdc_dgnss import (
                                            DDWLSConfig,
                                            dd_pseudorange_position_update,
                                        )

                                        anchor_pos, anchor_diag = dd_pseudorange_position_update(
                                            dd_select_pos,
                                            dd_pr_result,
                                            DDWLSConfig(
                                                min_dd_pairs=int(
                                                    config.dd_pr_ls_anchor_min_pairs
                                                ),
                                                dd_sigma_m=float(
                                                    config.dd_pr_ls_anchor_dd_sigma_m
                                                ),
                                                prior_sigma_m=float(
                                                    config.dd_pr_ls_anchor_solve_prior_sigma_m
                                                ),
                                                max_shift_m=float(
                                                    config.dd_pr_ls_anchor_max_shift_m
                                                ),
                                                max_iter=8,
                                            ),
                                        )
                                    except Exception:
                                        anchor_pos = None
                                        anchor_diag = {"accepted": False}
                                    accepted = bool(anchor_diag.get("accepted", False))
                                    postfit_rms = float(anchor_diag.get("final_rms_m", float("inf")))
                                    if accepted and (
                                        not np.isfinite(postfit_rms)
                                        or postfit_rms
                                        > float(config.dd_pr_ls_anchor_max_postfit_rms_m)
                                    ):
                                        accepted = False
                                        fgo_stats.dd_pr_ls_anchor_rejected_postfit += 1
                                    if accepted and anchor_pos is not None and np.all(
                                        np.isfinite(anchor_pos)
                                    ):
                                        anchor_ecef = np.asarray(anchor_pos, dtype=np.float64)[:3]
                                        fgo_dd_pr_ls_anchor_pos[i, :] = anchor_ecef
                                        fgo_dd_pr_ls_anchor_sigma[i] = float(
                                            config.dd_pr_ls_anchor_prior_sigma_m
                                        )
                                        fgo_stats.dd_pr_ls_anchor_accepted += 1
                                        fgo_stats.dd_pr_ls_anchor_shift_sum_m += float(
                                            anchor_diag.get("shift_m", 0.0)
                                        )
                                        fgo_stats.dd_pr_ls_anchor_postfit_sum_m += postfit_rms
                                        if reference_pos is not None:
                                            ref_ecef = reference_pos.get(round(float(times[i]), 1))
                                            if ref_ecef is not None and np.all(np.isfinite(ref_ecef)):
                                                seed_ecef = np.asarray(dd_select_pos, dtype=np.float64)[:3]
                                                anchor_err = float(np.linalg.norm(anchor_ecef - ref_ecef[:3]))
                                                seed_err = float(np.linalg.norm(seed_ecef - ref_ecef[:3]))
                                                if np.isfinite(anchor_err) and np.isfinite(seed_err):
                                                    fgo_stats.dd_pr_ls_anchor_gt_count += 1
                                                    fgo_stats.dd_pr_ls_anchor_gt_error_sum_m += anchor_err
                                                    fgo_stats.dd_pr_ls_anchor_seed_error_sum_m += seed_err
                                                    if anchor_err < seed_err:
                                                        fgo_stats.dd_pr_ls_anchor_improved += 1
                                                    if anchor_err <= 0.5:
                                                        fgo_stats.dd_pr_ls_anchor_pass_05m += 1
                                                    if anchor_err <= 5.0:
                                                        fgo_stats.dd_pr_ls_anchor_pass_5m += 1
                                    elif not bool(anchor_diag.get("accepted", False)):
                                        fgo_stats.dd_pr_ls_anchor_rejected_solve += 1
                            if config.enable_fgo_lambda:
                                from gnss_gpu.local_fgo import DDPseudorangeEpoch

                                fgo_dd_pr_cache[i] = DDPseudorangeEpoch.from_result(dd_pr_result)
        if pf_diag_row is not None:
            pf_diag_row["dd_carrier_n"] = (
                int(getattr(dd_result, "n_dd", 0)) if dd_result is not None else 0
            )
            pf_diag_row["dd_pr_ls_anchor_available"] = bool(
                np.all(np.isfinite(fgo_dd_pr_ls_anchor_pos[i, :]))
            )
            pf_diag_row["dd_pr_ls_anchor_sigma_m"] = (
                float(fgo_dd_pr_ls_anchor_sigma[i])
                if np.isfinite(fgo_dd_pr_ls_anchor_sigma[i])
                else ""
            )
            pf_diag_row["dd_pr_ls_anchor_to_pf_after_pr_m"] = _diag_distance(
                fgo_dd_pr_ls_anchor_pos[i, :],
                np.asarray(
                    [
                        pf_diag_row.get("pf_after_pr_x", float("nan")),
                        pf_diag_row.get("pf_after_pr_y", float("nan")),
                        pf_diag_row.get("pf_after_pr_z", float("nan")),
                    ],
                    dtype=np.float64,
                ),
            )
            ref_diag = reference_pos.get(t_key) if reference_pos is not None else None
            pf_diag_row["dd_pr_ls_anchor_to_ref_m"] = _diag_distance(
                fgo_dd_pr_ls_anchor_pos[i, :],
                ref_diag,
            )

        doppler_update_applied = False
        doppler_gate_skip_reason = ""
        if config.enable_rbpf_velocity_kf and sat_velocity is not None and doppler_hz is not None:
            sv_full = np.asarray(sat_velocity[i], dtype=np.float64)
            dop_full = np.asarray(doppler_hz[i], dtype=np.float64)
            sat_full = np.asarray(sat_ecef[i], dtype=np.float64)
            w_full = np.asarray(weights[i], dtype=np.float64)
            if system_ids is not None:
                sids_doppler_full = np.asarray(system_ids[i], dtype=np.int32)
                w_full = _scale_weights_per_system(w_full, sids_doppler_full)
            else:
                sids_doppler_full = None
            dop_finite = (
                np.isfinite(dop_full)
                & np.all(np.isfinite(sv_full), axis=1)
                & np.all(np.isfinite(sat_full), axis=1)
                & np.isfinite(w_full)
            )
            doppler_raw_finite_count = int(np.count_nonzero(dop_finite))
            if config.doppler_systems and sids_doppler_full is not None:
                allowed_doppler_ids = {
                    sid for sid, ch in _SYS_ID_TO_CHAR.items() if ch in set(config.doppler_systems)
                }
                doppler_sys_mask = np.array(
                    [int(sid) in allowed_doppler_ids for sid in sids_doppler_full],
                    dtype=bool,
                )
                dop_finite &= doppler_sys_mask
            if pf_diag_row is not None:
                pf_diag_row["doppler_raw_finite_count"] = int(doppler_raw_finite_count)
                pf_diag_row["doppler_system_filtered_count"] = int(np.count_nonzero(dop_finite))
                pf_diag_row["doppler_systems"] = ",".join(str(s) for s in config.doppler_systems)
            if int(dop_finite.sum()) >= 4 and float(config.doppler_prefit_gate_mps) > 0.0:
                pf_pos_for_doppler_gate = np.asarray(pf.estimate(), dtype=np.float64)[:3]
                doppler_prefit_mask = _doppler_prefit_gate_mask(
                    sat_full[dop_finite],
                    sv_full[dop_finite],
                    dop_full[dop_finite],
                    w_full[dop_finite],
                    pf_pos_for_doppler_gate,
                    gate_mps=float(config.doppler_prefit_gate_mps),
                    min_sats=int(config.doppler_prefit_gate_min_sats),
                    doppler_sign=-1.0,
                    wavelength_m=_GPS_L1_WAVELENGTH_M,
                )
                doppler_prefit_indices = np.flatnonzero(dop_finite)
                dop_finite[doppler_prefit_indices[~doppler_prefit_mask]] = False
                if pf_diag_row is not None:
                    pf_diag_row["doppler_prefit_gate_mps"] = float(config.doppler_prefit_gate_mps)
                    pf_diag_row["doppler_prefit_kept_count"] = int(np.count_nonzero(doppler_prefit_mask))
                    pf_diag_row["doppler_prefit_dropped_count"] = int(doppler_prefit_mask.size - np.count_nonzero(doppler_prefit_mask))
            if int(dop_finite.sum()) >= 4:
                doppler_gate_wls_rms_mps = float("nan")
                doppler_gate_wls_speed_mps = float("nan")
                if (
                    float(config.rbpf_kf_gate_max_doppler_wls_rms_mps) > 0.0
                    or float(config.rbpf_kf_gate_max_doppler_wls_speed_mps) > 0.0
                ):
                    pf_pos_for_doppler_quality = np.asarray(
                        pf.estimate(), dtype=np.float64
                    )[:3]
                    wls_vel_for_gate, wls_rms_for_gate = _doppler_centered_wls_velocity(
                        sat_full[dop_finite],
                        sv_full[dop_finite],
                        dop_full[dop_finite],
                        w_full[dop_finite],
                        pf_pos_for_doppler_quality,
                        doppler_sign=-1.0,
                        wavelength_m=_GPS_L1_WAVELENGTH_M,
                    )
                    doppler_gate_wls_rms_mps = float(wls_rms_for_gate)
                    if wls_vel_for_gate is not None and np.all(
                        np.isfinite(wls_vel_for_gate[:3])
                    ):
                        doppler_gate_wls_speed_mps = float(
                            np.linalg.norm(wls_vel_for_gate[:3])
                        )
                if pf_diag_row is not None:
                    pf_vel_before_doppler = _capture_pf_velocity_state(
                        pf,
                        pf_diag_row,
                        "pf_before_doppler_vel",
                    )
                    pf_pos_before_doppler = np.asarray(pf.estimate(), dtype=np.float64)[:3]
                    ref_vel_diag = _reference_velocity_for_epoch(times, reference_pos, i)
                    if ref_vel_diag is not None and np.all(np.isfinite(ref_vel_diag[:3])):
                        pf_diag_row["ref_velocity_speed_mps"] = float(np.linalg.norm(ref_vel_diag[:3]))
                    for label, sign in (("current", -1.0), ("flipped", 1.0)):
                        pf_diag_row[f"doppler_{label}_pfvel_rms_mps"] = _doppler_centered_residual_rms(
                            sat_full[dop_finite],
                            sv_full[dop_finite],
                            dop_full[dop_finite],
                            w_full[dop_finite],
                            pf_pos_before_doppler,
                            pf_vel_before_doppler,
                            doppler_sign=sign,
                            wavelength_m=_GPS_L1_WAVELENGTH_M,
                        )
                        if ref_vel_diag is not None:
                            pf_diag_row[f"doppler_{label}_refvel_rms_mps"] = _doppler_centered_residual_rms(
                                sat_full[dop_finite],
                                sv_full[dop_finite],
                                dop_full[dop_finite],
                                w_full[dop_finite],
                                pf_pos_before_doppler,
                                ref_vel_diag,
                                doppler_sign=sign,
                                wavelength_m=_GPS_L1_WAVELENGTH_M,
                            )
                        wls_vel, wls_rms = _doppler_centered_wls_velocity(
                            sat_full[dop_finite],
                            sv_full[dop_finite],
                            dop_full[dop_finite],
                            w_full[dop_finite],
                            pf_pos_before_doppler,
                            doppler_sign=sign,
                            wavelength_m=_GPS_L1_WAVELENGTH_M,
                        )
                        pf_diag_row[f"doppler_{label}_wls_rms_mps"] = wls_rms
                        if wls_vel is not None and np.all(np.isfinite(wls_vel[:3])):
                            pf_diag_row[f"doppler_{label}_wls_speed_mps"] = float(np.linalg.norm(wls_vel[:3]))
                            if ref_vel_diag is not None and np.all(np.isfinite(ref_vel_diag[:3])):
                                pf_diag_row[f"doppler_{label}_wls_to_refvel_mps"] = float(
                                    np.linalg.norm(wls_vel[:3] - ref_vel_diag[:3])
                                )
                if gate_active:
                    gate_stats.epochs_attempted += 1
                gate_skipped = False
                if config.rbpf_kf_gate_min_dd_pairs is not None:
                    n_dd_now = int(getattr(dd_result, "n_dd", 0)) if dd_result is not None else 0
                    if n_dd_now < int(config.rbpf_kf_gate_min_dd_pairs):
                        gate_stats.skipped_min_dd_pairs += 1
                        gate_skipped = True
                        doppler_gate_skip_reason = "min_dd_pairs"
                if not gate_skipped and config.rbpf_kf_gate_min_ess_ratio is not None:
                    ess = float(pf.get_ess())
                    ess_ratio = ess / max(int(config.n_particles), 1)
                    if ess_ratio < float(config.rbpf_kf_gate_min_ess_ratio):
                        gate_stats.skipped_min_ess_ratio += 1
                        gate_skipped = True
                        doppler_gate_skip_reason = "min_ess"
                if not gate_skipped and config.rbpf_kf_gate_max_spread_m is not None:
                    spread = float(pf.get_position_spread())
                    if spread > float(config.rbpf_kf_gate_max_spread_m):
                        gate_stats.skipped_max_spread += 1
                        gate_skipped = True
                        doppler_gate_skip_reason = "max_spread"
                if (
                    not gate_skipped
                    and float(config.rbpf_kf_gate_max_doppler_wls_rms_mps) > 0.0
                    and (
                        not np.isfinite(doppler_gate_wls_rms_mps)
                        or doppler_gate_wls_rms_mps
                        > float(config.rbpf_kf_gate_max_doppler_wls_rms_mps)
                    )
                ):
                    gate_stats.skipped_doppler_wls_rms += 1
                    gate_skipped = True
                    doppler_gate_skip_reason = "doppler_wls_rms"
                if (
                    not gate_skipped
                    and float(config.rbpf_kf_gate_max_doppler_wls_speed_mps) > 0.0
                    and (
                        not np.isfinite(doppler_gate_wls_speed_mps)
                        or doppler_gate_wls_speed_mps
                        > float(config.rbpf_kf_gate_max_doppler_wls_speed_mps)
                    )
                ):
                    gate_stats.skipped_doppler_wls_speed += 1
                    gate_skipped = True
                    doppler_gate_skip_reason = "doppler_wls_speed"
                if not gate_skipped:
                    pf.update_doppler_kf(
                        sat_full[dop_finite],
                        sv_full[dop_finite],
                        dop_full[dop_finite],
                        weights=w_full[dop_finite],
                        wavelength=_GPS_L1_WAVELENGTH_M,
                        sigma_mps=config.sigma_doppler_mps,
                        resample=not defer_resample,
                    )
                    doppler_update_applied = True
                    if gate_active:
                        gate_stats.epochs_applied += 1
                if pf_diag_row is not None:
                    _capture_pf_velocity_state(
                        pf,
                        pf_diag_row,
                        "pf_after_doppler_vel",
                    )
        if pf_diag_row is not None:
            pf_diag_row["doppler_update_applied"] = bool(doppler_update_applied)
            pf_diag_row["doppler_gate_skip_reason"] = doppler_gate_skip_reason
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_doppler",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        dd_carrier_update_applied = False
        if config.enable_dd_carrier_afv and dd_computer is not None:
            dd_stats.epochs_attempted += 1
            if (
                dd_result is not None
                and int(getattr(dd_result, "n_dd", 0)) >= int(config.dd_min_pairs_update)
            ):
                pf.update_dd_carrier_afv(
                    dd_result,
                    sigma_cycles=float(config.dd_sigma_cycles),
                    resample=not defer_resample,
                )
                dd_stats.epochs_applied += 1
                dd_stats.pairs_total += int(dd_result.n_dd)
                dd_carrier_update_applied = True
        if pf_diag_row is not None:
            pf_diag_row["dd_carrier_update_applied"] = bool(dd_carrier_update_applied)
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_dd_carrier",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        dd_pr_ls_anchor_update_applied = False
        anchor_mode_for_pf = str(config.dd_pr_ls_anchor_mode).strip().lower()
        if (
            config.enable_dd_pr_ls_anchor
            and anchor_mode_for_pf in {"pf", "pu", "pf-pu", "prior+pf", "initial+pf"}
            and np.all(np.isfinite(fgo_dd_pr_ls_anchor_pos[i, :]))
            and np.isfinite(fgo_dd_pr_ls_anchor_sigma[i])
            and float(fgo_dd_pr_ls_anchor_sigma[i]) > 0.0
        ):
            pf.position_update(
                np.asarray(fgo_dd_pr_ls_anchor_pos[i, :], dtype=np.float64),
                sigma_pos=float(fgo_dd_pr_ls_anchor_sigma[i]),
            )
            dd_pr_ls_anchor_update_applied = True
        if pf_diag_row is not None:
            pf_diag_row["dd_pr_ls_anchor_update_applied"] = bool(
                dd_pr_ls_anchor_update_applied
            )
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_dd_pr_ls_anchor",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        if config.enable_position_update:
            ref = wls_positions[i, :3]
            pr_sat_count_for_pu = int(len(pr_i))
            wls_postfit_rms_m = float("nan")
            wls_postfit_absmax_m = float("nan")
            wls_pdop = float("nan")
            wls_normal_cond = float("nan")
            if wls_quality is not None:
                if i < len(wls_quality.get("postfit_rms_m", ())):
                    wls_postfit_rms_m = float(wls_quality["postfit_rms_m"][i])
                if i < len(wls_quality.get("postfit_absmax_m", ())):
                    wls_postfit_absmax_m = float(wls_quality["postfit_absmax_m"][i])
                if i < len(wls_quality.get("pdop", ())):
                    wls_pdop = float(wls_quality["pdop"][i])
                if i < len(wls_quality.get("normal_cond", ())):
                    wls_normal_cond = float(wls_quality["normal_cond"][i])
            pu_skip_min_epoch = int(i) < int(config.position_update_min_epoch)
            pu_skip_min_sats = (
                int(config.position_update_min_pr_sats) > 0
                and pr_sat_count_for_pu < int(config.position_update_min_pr_sats)
            )
            pu_skip_wls_rms = (
                float(config.position_update_max_wls_rms_m) > 0.0
                and (
                    not np.isfinite(wls_postfit_rms_m)
                    or wls_postfit_rms_m
                    > float(config.position_update_max_wls_rms_m)
                )
            )
            pu_skip_wls_pdop = (
                float(config.position_update_max_wls_pdop) > 0.0
                and (
                    not np.isfinite(wls_pdop)
                    or wls_pdop > float(config.position_update_max_wls_pdop)
                )
            )
            pf_ref_before_pu = np.asarray(pf.estimate(), dtype=np.float64)[:3]
            wls_to_pf_before_pu_m = (
                float(np.linalg.norm(ref - pf_ref_before_pu))
                if np.all(np.isfinite(ref)) and np.all(np.isfinite(pf_ref_before_pu))
                else float("nan")
            )
            pu_skip_wls_to_pf = (
                float(config.position_update_max_wls_to_pf_m) > 0.0
                and (
                    not np.isfinite(wls_to_pf_before_pu_m)
                    or wls_to_pf_before_pu_m
                    > float(config.position_update_max_wls_to_pf_m)
                )
            )
            wls_pu_err_e, wls_pu_err_n, wls_pu_err_u = _diag_enu_error(ref, ref_diag_epoch)
            if pf_diag_row is not None:
                pf_diag_row["position_update_min_epoch"] = int(
                    config.position_update_min_epoch
                )
                pf_diag_row["position_update_skipped_min_epoch"] = bool(
                    pu_skip_min_epoch
                )
                pf_diag_row["position_update_min_pr_sats"] = int(
                    config.position_update_min_pr_sats
                )
                pf_diag_row["position_update_pr_sats"] = pr_sat_count_for_pu
                pf_diag_row["position_update_skipped_min_sats"] = bool(
                    pu_skip_min_sats
                )
                pf_diag_row["position_update_max_wls_rms_m"] = float(
                    config.position_update_max_wls_rms_m
                )
                pf_diag_row["position_update_max_wls_pdop"] = float(
                    config.position_update_max_wls_pdop
                )
                pf_diag_row["position_update_max_wls_to_pf_m"] = float(
                    config.position_update_max_wls_to_pf_m
                )
                pf_diag_row["position_update_wls_postfit_rms_m"] = wls_postfit_rms_m
                pf_diag_row["position_update_wls_postfit_absmax_m"] = (
                    wls_postfit_absmax_m
                )
                pf_diag_row["position_update_wls_pdop"] = wls_pdop
                pf_diag_row["position_update_wls_normal_cond"] = wls_normal_cond
                pf_diag_row["position_update_wls_to_pf_before_m"] = (
                    wls_to_pf_before_pu_m
                )
                pf_diag_row["position_update_wls_to_ref_m"] = _diag_distance(
                    ref,
                    ref_diag_epoch,
                )
                pf_diag_row["position_update_wls_err_e_m"] = wls_pu_err_e
                pf_diag_row["position_update_wls_err_n_m"] = wls_pu_err_n
                pf_diag_row["position_update_wls_err_u_m"] = wls_pu_err_u
                pf_diag_row["position_update_skipped_wls_rms"] = bool(
                    pu_skip_wls_rms
                )
                pf_diag_row["position_update_skipped_wls_pdop"] = bool(
                    pu_skip_wls_pdop
                )
                pf_diag_row["position_update_skipped_wls_to_pf"] = bool(
                    pu_skip_wls_to_pf
                )
            if (
                not pu_skip_min_epoch
                and not pu_skip_min_sats
                and not pu_skip_wls_rms
                and not pu_skip_wls_pdop
                and not pu_skip_wls_to_pf
                and np.all(np.isfinite(ref))
                and not np.all(ref == 0.0)
            ):
                pf.position_update(ref, sigma_pos=config.position_update_sigma_m)
        if pf_diag_row is not None:
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_position_update",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        # Phase 6: hybrid (libgnss++ 50.91%) blend. The runtime PF cannot
        # currently follow the trajectory without a velocity guide (predict
        # step keeps particles stationary, hybrid PU at sigma~1m cannot
        # rescue a cloud that is several meters from the moving hybrid
        # baseline). MVP behavior: emit the hybrid position directly when
        # available, and only fall through to the PF estimate when hybrid
        # is missing for that TOW. This gives us the hybrid baseline as the
        # floor and lets the PF cover hybrid-gap epochs. A future revision
        # will add a velocity guide so the PF can independently correct
        # hybrid (e.g., DD-carrier-driven cm-pull), but until then the PF
        # alone diverges over hundreds of meters.
        hp = None
        if use_hybrid:
            hybrid_stats.epochs_attempted += 1
            hp = hp_prefetched
            if hp is None:
                hybrid_stats.epochs_lookup_missing += 1
            elif np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                # Phase 9b: relax the hybrid PU sigma on m-class Status
                # epochs so IMU pre-integration can dominate. Without this
                # the tight 1m hybrid PU clamps the cloud to the (wrong-by-
                # several-meters) hybrid passthrough and the IMU rescue
                # never surfaces in pf.estimate().
                hybrid_sigma_now = float(config.hybrid_sigma_m)
                st_now_pu = hybrid_status.get(round(float(times[i]), 1)) if hybrid_status is not None else None
                if (
                    config.enable_pr_gmm
                    and float(config.pr_gmm_hybrid_loose_sigma_m) > 0.0
                ):
                    gmm_status_match = True
                    if st_now_pu is not None and config.pr_gmm_statuses:
                        gmm_status_match = int(st_now_pu) in {int(s) for s in config.pr_gmm_statuses}
                    if gmm_status_match:
                        hybrid_sigma_now = float(config.pr_gmm_hybrid_loose_sigma_m)
                if (
                    config.enable_imu_tc
                    and float(config.imu_tc_hybrid_loose_sigma_m) > 0.0
                    and hybrid_status is not None
                ):
                    if (
                        st_now_pu is not None
                        and int(st_now_pu) in {int(s) for s in config.imu_tc_emit_pf_hybrid_statuses}
                    ):
                        hybrid_sigma_now = float(config.imu_tc_hybrid_loose_sigma_m)
                if float(config.hybrid_recenter_max_shift_m) > 0.0:
                    shift_norm, recentered = pf.recenter_position(
                        hp,
                        max_shift_m=float(config.hybrid_recenter_max_shift_m),
                    )
                    if recentered:
                        hybrid_stats.recenter_applied += 1
                        hybrid_stats.recenter_shift_sum_m += float(shift_norm)
                    else:
                        hybrid_stats.recenter_skipped += 1
                pf.position_update(hp, sigma_pos=hybrid_sigma_now)
                hybrid_stats.epochs_applied += 1
        if pf_diag_row is not None:
            pf_diag_row["hybrid_pu_applied"] = bool(
                use_hybrid
                and hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(np.asarray(hp, dtype=np.float64) == 0.0)
            )
            pf_diag_row["pf_before_hybrid_to_hybrid_m"] = _diag_distance(
                np.asarray(
                    [
                        pf_diag_row.get("pf_after_position_update_x", float("nan")),
                        pf_diag_row.get("pf_after_position_update_y", float("nan")),
                        pf_diag_row.get("pf_after_position_update_z", float("nan")),
                    ],
                    dtype=np.float64,
                ),
                hp,
            )
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_hybrid",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        rtkdiag_pf_emit_here = False
        rtkdiag_pf_ref: np.ndarray | None = None
        rtkdiag_selected_label = ""
        rtkdiag_gated_options_epoch = 0
        rtkdiag_candidate_epoch_ready = True
        if use_rtkdiag_pf:
            rtkdiag_pf_stats.epochs_evaluated += 1
            rtkdiag_min_epoch = max(int(config.rtkdiag_candidate_min_epoch), 0)
            rtkdiag_candidate_epoch_ready = int(i) >= rtkdiag_min_epoch
            if not rtkdiag_candidate_epoch_ready:
                rtkdiag_pf_stats.skipped_min_epoch += 1
            select_mode = str(config.rtkdiag_candidate_select_mode)
            is_fusion = select_mode in {"wavg3", "wavg5"}
            is_consensus = select_mode in {"consensus3", "consensus5"}
            is_cluster_vote = select_mode == "cluster_vote"
            is_ranker = select_mode == "ranker"
            temporal_prevdist_alpha = {
                "temporal_n2_v1": 0.001,
                "temporal_n2_v2": 0.0006,
                "temporal_n2_v3": 0.00062,
                "temporal_n2_v4": 0.00062,
                "temporal_n2_v5": 0.00062,
                "temporal_n2_v6": 0.00062,
                "temporal_n2_v7": 0.00062,
                "temporal_n2_v8": 0.00062,
                "temporal_n2_v9": 0.00062,
                "temporal_n2_v10": 0.00062,
                "temporal_n2_v11_a001": 0.001,
                "temporal_n2_v12_a01": 0.01,
                "temporal_n2_v13_a1": 0.1,
                "temporal_n2_v14_a3": 0.3,
                "temporal_n2_v15_a10": 1.0,
            }.get(select_mode)
            is_temporal_prevdist = temporal_prevdist_alpha is not None
            temporal_hybdelta_base = {
                "temporal_hybdelta_t3_v1": "composite_t3_v2",
                "temporal_hybdelta_t3_v2": "composite_t3_v2",
                "temporal_hybdelta_t3_v3": "composite_t3_v2",
                "temporal_hybdelta_t3_v4": "composite_t3_v4",
                "temporal_hybdelta_t3_v5": "composite_t3_v4",
                "temporal_hybdelta_t3_v6": "composite_t3_v4",
                "temporal_hybdelta_t3_v7": "composite_t3_v4",
                "temporal_hybdelta_t3_v8": "composite_t3_v4",
                "temporal_hybdelta_n2_v1": "composite_n2_v4",
                "temporal_hybdelta_n3_v1": "composite_n3_v3",
                "temporal_hybdelta_n3_v2": "composite_n3_v4",
                "temporal_hybdelta_n3_v3": "composite_n3_v4",
                "temporal_hybdelta_n3_v4": "composite_n3_v4",
                "temporal_hybdelta_n3_v5": "composite_n3_v4",
                "temporal_hybdelta_n3_v6": "composite_n3_v4",
            }.get(select_mode)
            temporal_hybdelta_alpha = {
                "temporal_hybdelta_t3_v1": 0.0003,
                "temporal_hybdelta_t3_v2": 0.0002,
                "temporal_hybdelta_t3_v3": 0.00022,
                "temporal_hybdelta_t3_v4": 0.0002,
                "temporal_hybdelta_t3_v5": 0.0002,
                "temporal_hybdelta_t3_v6": 0.0002,
                "temporal_hybdelta_t3_v7": 0.0002,
                "temporal_hybdelta_t3_v8": 0.0002,
                "temporal_hybdelta_n2_v1": 0.0003,
                "temporal_hybdelta_n3_v1": 0.0003,
                "temporal_hybdelta_n3_v2": 0.0006,
                "temporal_hybdelta_n3_v3": 0.0006,
                "temporal_hybdelta_n3_v4": 0.0006,
                "temporal_hybdelta_n3_v5": 0.0006,
                "temporal_hybdelta_n3_v6": 0.0006,
            }.get(select_mode)
            label_penalty_factors = {
                "temporal_hybdelta_t3_v5": {
                    "rtkout5minobs3": 1.06,
                    "mlc1r10": 1.03,
                },
                "temporal_hybdelta_t3_v6": {
                    "rtkout5minobs3": 1.06,
                    "mlc1r10": 1.03,
                    "c1p1hr": 1.10,
                },
                "temporal_hybdelta_t3_v7": {
                    "rtkout5minobs3": 1.06,
                    "mlc1r10": 1.03,
                    "c1p1hr": 1.10,
                    "r20ga": 3.00,
                    "psig1": 1.50,
                    "r15ga": 1.20,
                },
                "temporal_hybdelta_t3_v8": {
                    "rtkout5minobs3": 1.06,
                    "mlc1r10": 1.03,
                    "c1p1hr": 1.10,
                    "r20ga": 3.00,
                    "psig1": 1.50,
                    "r15ga": 1.20,
                    "r25g10": 1.50,
                    "r20g10": 1.50,
                    "r15g10": 1.10,
                },
                "temporal_n2_v4": {
                    "mlc1oGc0001": 1.06,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                },
                "temporal_n2_v5": {
                    "mlc1oGc0001": 1.06,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                    "csig005_em10": 1.06,
                    "mlc1oG": 1.06,
                },
                "temporal_n2_v6": {
                    "mlc1oGc0001": 1.06,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                    "csig005_em10": 1.06,
                    "mlc1oG": 1.06,
                    "oGc005": 1.10,
                    "psig3": 1.20,
                },
                "temporal_n2_v7": {
                    "mlc1oGc0001": 1.06,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                    "csig005_em10": 1.06,
                    "mlc1oG": 1.06,
                    "oGc005": 1.10,
                    "psig3": 1.20,
                    "r15": 1.06,
                    "r15g": 1.01,
                },
                "temporal_n2_v8": {
                    "mlc1oGc0001": 1.06,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                    "csig005_em10": 1.06,
                    "mlc1oG": 1.06,
                    "oGc005": 1.10,
                    "psig3": 1.20,
                    "r15": 1.06,
                    "r15g": 1.01,
                    "csig05_psig1": 1.01,
                    "rtkout5oG": 1.03,
                },
                "temporal_n2_v9": {
                    "mlc1oGc0001": 1.06,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                    "csig005_em10": 1.06,
                    "mlc1oG": 1.06,
                    "oGc005": 1.10,
                    "psig3": 1.20,
                    "r15": 1.06,
                    "r15g": 1.0403,
                    "csig05_psig1": 1.01,
                    "rtkout5oG": 1.03,
                    "csig05": 1.01,
                    "r25g": 1.01,
                },
                "temporal_n2_v10": {
                    "mlc1oGc0001": 1.0706,
                    "mlc1r10oG": 1.10,
                    "rtkout3": 1.06,
                    "csig005_em10": 1.06,
                    "mlc1oG": 1.06,
                    "oGc005": 1.10,
                    "psig3": 1.20,
                    "r15": 1.06,
                    "r15g": 1.0403,
                    "csig05_psig1": 1.01,
                    "rtkout5oG": 1.03,
                    "csig05": 1.01,
                    "r25g": 1.01,
                    "n2loose3": 1.06,
                    "r25": 1.01,
                },
                "temporal_n2_v11_a001": {},
                "temporal_n2_v12_a01": {},
                "temporal_n2_v13_a1": {},
                "temporal_n2_v14_a3": {},
                "temporal_n2_v15_a10": {},
                "temporal_hybdelta_n3_v3": {
                    "rtkout5c005em3": 1.06,
                    "mlc2nobds": 1.50,
                    "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
                },
                "temporal_hybdelta_n3_v4": {
                    "rtkout5c005em3": 1.06,
                    "mlc2nobds": 1.50,
                    "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
                    "mlc1c005p1": 1.50,
                    "n3tight": 1.10,
                },
                "temporal_hybdelta_n3_v5": {
                    "rtkout5c005em3": 1.06,
                    "mlc2nobds": 1.50,
                    "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
                    "mlc1c005p1": 1.50,
                    "n3tight": 1.10,
                    "mlc1oGc005p1": 1.03,
                    "csig05psh": 1.10,
                },
                "temporal_hybdelta_n3_v6": {
                    "rtkout5c005em3": 1.06,
                    "mlc2nobds": 1.50,
                    "xd_n3_loose_hold4_ratio15_gate10_min6": 1.03,
                    "mlc1c005p1": 1.50,
                    "n3tight": 1.10,
                    "mlc1oGc005p1": 1.03,
                    "csig05psh": 1.10,
                    "n3tight2": 1.01,
                },
            }.get(select_mode, {})
            if config.rtkdiag_candidate_label_factors:
                label_penalty_factors = dict(label_penalty_factors)
                label_penalty_factors.update(
                    {
                        str(label): float(factor)
                        for label, factor in config.rtkdiag_candidate_label_factors
                    }
                )
            is_temporal_hybdelta = temporal_hybdelta_base is not None
            collected: list[tuple[str, np.ndarray, dict[str, str], tuple[float, float]]] = []
            gated_options = 0
            local_ungate_labels = _rtkdiag_local_ungate_labels(
                tuple(config.rtkdiag_candidate_local_ungate_windows),
                int(i),
            )
            if local_ungate_labels is None:
                local_ungate_labels = _rtkdiag_local_ungate_labels_for_tow(
                    tuple(config.rtkdiag_candidate_local_ungate_tow_windows),
                    float(t_key),
                )
            for label, candidate_pos, candidate_diag in rtkdiag_candidates or []:
                if not rtkdiag_candidate_epoch_ready:
                    continue
                diag_row = candidate_diag.get(t_key)
                gate_ok = _rtkdiag_candidate_gate(
                    diag_row,
                    ratio_min=float(config.rtkdiag_candidate_ratio_min),
                    residual_rms_max=float(config.rtkdiag_candidate_residual_rms_max),
                    status5_residual_rms_max=float(
                        config.rtkdiag_candidate_main_status5_residual_rms_max
                    ),
                )
                float_gate_ok = _rtkdiag_candidate_float_gate(
                    diag_row,
                    label=label,
                    allowed_labels=tuple(config.rtkdiag_candidate_float_labels),
                    residual_rms_max=float(config.rtkdiag_candidate_float_residual_rms_max),
                    residual_abs_max=float(config.rtkdiag_candidate_float_abs_max),
                    min_sats=int(config.rtkdiag_candidate_float_min_sats),
                )
                local_ungate_ok = (
                    local_ungate_labels is not None
                    and _rtkdiag_fixed_output_ok(diag_row)
                    and (not local_ungate_labels or label in local_ungate_labels)
                )
                cand = candidate_pos.get(t_key)
                status5_gate_ok = False
                if not gate_ok and not float_gate_ok and not local_ungate_ok:
                    status5_window_labels = _rtkdiag_local_ungate_labels_for_tow(
                        tuple(config.rtkdiag_candidate_status5_tow_windows),
                        float(t_key),
                    )
                    if (
                        status5_window_labels is not None
                        and (not status5_window_labels or label in status5_window_labels)
                    ):
                        _cand_t_key, status5_cand, status5_diag_row = (
                            _rtkdiag_nearest_candidate_row(
                                candidate_pos,
                                candidate_diag,
                                t_key,
                                max_dt_s=float(
                                    config.rtkdiag_candidate_status5_max_dt_s
                                ),
                            )
                        )
                        status5_gate_ok = _rtkdiag_candidate_status5_gate(
                            status5_diag_row,
                            label=label,
                            allowed_labels=tuple(
                                config.rtkdiag_candidate_status5_labels
                            ),
                            residual_rms_max=float(
                                config.rtkdiag_candidate_status5_residual_rms_max
                            ),
                            min_sats=int(config.rtkdiag_candidate_status5_min_sats),
                        )
                        if status5_gate_ok:
                            diag_row = status5_diag_row
                            cand = status5_cand
                if (
                    not gate_ok
                    and not float_gate_ok
                    and not local_ungate_ok
                    and not status5_gate_ok
                ):
                    continue
                if not _rtkdiag_candidate_diag_policy_gate(
                    diag_row,
                    require_any_fields=tuple(
                        config.rtkdiag_candidate_require_any_diag_fields
                    ),
                    require_all_fields=tuple(
                        config.rtkdiag_candidate_require_all_diag_fields
                    ),
                    min_fields=tuple(config.rtkdiag_candidate_min_diag_fields),
                    max_fields=tuple(config.rtkdiag_candidate_max_diag_fields),
                ):
                    rtkdiag_pf_stats.skipped_diag_policy += 1
                    continue
                gated_options += 1
                if cand is None or not np.all(np.isfinite(cand)) or np.all(cand == 0.0):
                    continue
                max_to_hybrid = float(config.rtkdiag_candidate_max_to_hybrid_m)
                if (
                    max_to_hybrid > 0.0
                    and hp is not None
                    and np.all(np.isfinite(hp))
                    and not np.all(np.asarray(hp, dtype=np.float64) == 0.0)
                    and float(
                        np.linalg.norm(
                            np.asarray(cand, dtype=np.float64)
                            - np.asarray(hp, dtype=np.float64)
                        )
                    )
                    > max_to_hybrid
                ):
                    rtkdiag_pf_stats.skipped_hybrid_distance += 1
                    continue
                if is_fusion or is_consensus:
                    candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode="score")
                elif is_temporal_prevdist:
                    candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode="composite_n2_v4")
                elif temporal_hybdelta_base is not None:
                    candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode=temporal_hybdelta_base)
                elif select_mode == "hybrid_anchor" and hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                    cand_arr = np.asarray(cand, dtype=np.float64)
                    dist_to_hyb = float(np.linalg.norm(cand_arr - np.asarray(hp, dtype=np.float64)))
                    candidate_key = (dist_to_hyb, _diag_float(diag_row, "final_residual_rms"))
                elif select_mode == "hybrid_anchor":
                    candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode="residual")
                else:
                    candidate_key = _rtkdiag_candidate_sort_key(diag_row, mode=select_mode)
                if label_penalty_factors:
                    factor = float(label_penalty_factors.get(label, 1.0))
                    if factor != 1.0:
                        candidate_key = (float(candidate_key[0]) * factor, float(candidate_key[1]))
                collected.append((label, np.asarray(cand, dtype=np.float64), diag_row, candidate_key))

            # Velocity/IMU bridge candidate (2位再現): when last good anchor is
            # recent (<= bridge_max_s), propose anchor + PF_velocity * Δt as a
            # synthetic candidate. Helps n/r2 where 19 gici candidates cluster
            # within 0.38m and selector can't distinguish truth among them.
            # anchor_mode: "last_emit" = any emit (cluster bias inherited);
            #              "last_fix4" = last trusted status=4 high-ratio Fix only
            _bridge_anchor_mode = str(config.rtkdiag_candidate_bridge_anchor_mode)
            if _bridge_anchor_mode == "last_fix4":
                _bridge_anchor_pos = rtkdiag_last_fix4_pos
                _bridge_anchor_t = rtkdiag_last_fix4_t
            else:
                _bridge_anchor_pos = rtkdiag_last_good_pos
                _bridge_anchor_t = rtkdiag_last_good_t
            if (
                rtkdiag_candidate_epoch_ready
                and bool(config.rtkdiag_candidate_bridge_enable)
                and _bridge_anchor_pos is not None
                and _bridge_anchor_t is not None
                and np.all(np.isfinite(_bridge_anchor_pos))
            ):
                bridge_dt = float(t_key) - float(_bridge_anchor_t)
                if 0.0 < bridge_dt <= float(config.rtkdiag_candidate_bridge_max_s):
                    # Get PF velocity from particle states[:, 4:7] (vx, vy, vz)
                    try:
                        _pstates = np.asarray(pf.get_particle_states(), dtype=np.float64)
                        _logw = np.asarray(pf.get_log_weights(), dtype=np.float64)
                        bridge_vel = _weighted_mean_from_log_weights(_pstates[:, 4:7], _logw)
                    except (AttributeError, IndexError, ValueError):
                        bridge_vel = None
                    if bridge_vel is not None and bridge_vel.size >= 3 and np.all(np.isfinite(bridge_vel[:3])):
                        bridge_pos = (
                            np.asarray(_bridge_anchor_pos, dtype=np.float64)
                            + bridge_vel[:3] * bridge_dt
                        )
                        if np.all(np.isfinite(bridge_pos)):
                            bridge_residual = float(
                                config.rtkdiag_candidate_bridge_residual_rms_m
                            )
                            bridge_diag = {
                                "tow": str(t_key),
                                "output_added": "1",
                                "final_status": "4",
                                "final_ratio": "10.0",
                                "final_residual_rms": str(bridge_residual),
                                "final_residual_abs_max": str(bridge_residual),
                                "final_sats": "20",
                                "final_update_rows": "10",
                                "final_pdop": "1.0",
                                "final_baseline_m": "0.0",
                            }
                            # Bridge sort key: penalize by Δt^2 so close-anchor bridges
                            # rank high but far ones drop off.
                            bridge_key = (bridge_residual * (1.0 + bridge_dt ** 2), bridge_residual)
                            collected.append(
                                ("pf_bridge",
                                 np.asarray(bridge_pos, dtype=np.float64),
                                 bridge_diag,
                                 bridge_key)
                            )
                            gated_options += 1

            # Pre-filter to top-K by residual_rms before applying selector
            # ranking. Drops cluster-biased high-rms candidates that current
            # composite formulas inadvertently rank above the truth. Sim shows
            # K=3-7 captures most of the +12pp ranking headroom; K>=20 collapses.
            rms_prefilter_k = int(config.rtkdiag_candidate_rms_prefilter_k)
            if rms_prefilter_k > 0 and len(collected) > rms_prefilter_k:
                collected = sorted(
                    collected,
                    key=lambda c: _diag_float(c[2], "final_residual_rms"),
                )[:rms_prefilter_k]
                gated_options = len(collected)

            if gated_options > 0:
                rtkdiag_gated_options_epoch = int(gated_options)
                rtkdiag_pf_stats.gate_pass += 1
                rtkdiag_pf_stats.candidate_options_total += int(gated_options)
                hp_valid_for_temporal = hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp, dtype=np.float64) == 0.0)
                if not collected:
                    rtkdiag_pf_stats.candidate_missing += 1
                    if (is_temporal_prevdist or is_temporal_hybdelta) and hp_valid_for_temporal:
                        rtkdiag_temporal_prev = np.asarray(hp, dtype=np.float64)
                        rtkdiag_temporal_prev_hybrid = np.asarray(hp, dtype=np.float64)
                else:
                    if is_fusion:
                        n_fuse = 3 if select_mode == "wavg3" else 5
                        sorted_top = sorted(collected, key=lambda c: c[3])[:n_fuse]
                        eps = 0.01
                        raw_w = np.array([1.0 / (max(c[3][0], 0.0) + eps) for c in sorted_top], dtype=np.float64)
                        if raw_w.sum() < 1e-9 or not np.all(np.isfinite(raw_w)):
                            raw_w = np.ones(len(sorted_top), dtype=np.float64)
                        raw_w /= raw_w.sum()
                        fused_pos = np.sum(np.stack([w * c[1] for w, c in zip(raw_w, sorted_top)], axis=0), axis=0)
                        label = sorted_top[0][0] + "+wavg"
                        selected_pos = fused_pos
                        _selected_diag = sorted_top[0][2]
                        _selected_key = sorted_top[0][3]
                    elif is_ranker:
                        # Supervised LightGBM ranker (Path F): pick gated
                        # candidate with highest p_pass from precomputed
                        # lookup (run_id, tow, label) -> p_pass.
                        # Falls back to lowest-rms when lookup is missing or
                        # the epoch is not covered by predictions.
                        # Optional stickiness: if previous epoch's pick has
                        # p_pass >= stickiness * max_p, keep that label
                        # (sequence smoothing for oscillation-prone epochs).
                        best_score = -float("inf")
                        best_cand = None
                        tow_key = round(float(t_key), 1) if ranker_score_lookup else None
                        scores_this_epoch: dict[str, tuple[float, tuple]] = {}
                        if ranker_score_lookup and tow_key is not None:
                            for cand in collected:
                                p = ranker_score_lookup.get((tow_key, cand[0]))
                                if p is None:
                                    continue
                                scores_this_epoch[cand[0]] = (float(p), cand)
                                if p > best_score:
                                    best_score = float(p)
                                    best_cand = cand
                        if (
                            best_cand is not None
                            and ranker_stickiness > 0.0
                            and _prev_ranker_label is not None
                            and _prev_ranker_label in scores_this_epoch
                        ):
                            prev_p, prev_cand = scores_this_epoch[_prev_ranker_label]
                            if prev_p >= ranker_stickiness * best_score:
                                best_cand = prev_cand
                                best_score = prev_p
                        if best_cand is None:
                            best_cand = min(
                                collected,
                                key=lambda c: _diag_float(c[2], "final_residual_rms"),
                            )
                        label = best_cand[0] + "+rnk"
                        selected_pos = best_cand[1]
                        _selected_diag = best_cand[2]
                        _selected_key = best_cand[3]
                        _prev_ranker_label = best_cand[0]
                    elif is_cluster_vote:
                        # Greedy single-pass spatial clustering at 50cm radius.
                        # Pick largest cluster (tie-break: lowest rms member).
                        # Within winning cluster, pick lowest-rms candidate.
                        radius = float(config.rtkdiag_candidate_cluster_vote_radius_m)
                        clusters: list[dict] = []
                        for cand in collected:
                            cpos = np.asarray(cand[1], dtype=np.float64)
                            assigned = False
                            for cl in clusters:
                                if float(np.linalg.norm(cpos - cl["centroid"])) <= radius:
                                    cl["members"].append(cand)
                                    n_now = len(cl["members"])
                                    cl["centroid"] = (cl["centroid"] * (n_now - 1) + cpos) / n_now
                                    assigned = True
                                    break
                            if not assigned:
                                clusters.append({"centroid": cpos.copy(), "members": [cand]})

                        def _cluster_rank(cl: dict) -> tuple[int, float]:
                            rms_min = min(
                                _diag_float(m[2], "final_residual_rms") for m in cl["members"]
                            )
                            return (-len(cl["members"]), rms_min)
                        clusters.sort(key=_cluster_rank)
                        best_cluster = clusters[0]
                        best_cand = min(
                            best_cluster["members"],
                            key=lambda m: _diag_float(m[2], "final_residual_rms"),
                        )
                        label = best_cand[0] + "+clv"
                        selected_pos = best_cand[1]
                        _selected_diag = best_cand[2]
                        _selected_key = best_cand[3]
                    elif is_consensus:
                        n_cons = 3 if select_mode == "consensus3" else 5
                        sorted_top = sorted(collected, key=lambda c: c[3])[:n_cons]
                        cand_positions = np.stack([c[1] for c in sorted_top], axis=0)
                        median_pos = np.median(cand_positions, axis=0)
                        # Pick candidate nearest to median (robust selector)
                        dists = np.linalg.norm(cand_positions - median_pos, axis=1)
                        best_idx = int(np.argmin(dists))
                        label = sorted_top[best_idx][0] + "+cons"
                        selected_pos = sorted_top[best_idx][1]
                        _selected_diag = sorted_top[best_idx][2]
                        _selected_key = sorted_top[best_idx][3]
                    elif is_temporal_prevdist and rtkdiag_temporal_prev is not None:
                        alpha = float(temporal_prevdist_alpha)
                        best_cand = min(
                            collected,
                            key=lambda c, _prev=rtkdiag_temporal_prev: (
                                c[3][0] + alpha * float(np.linalg.norm(c[1] - _prev)),
                                c[3][1],
                            ),
                        )
                        label, selected_pos, _selected_diag, _selected_key = best_cand
                    elif (
                        is_temporal_hybdelta
                        and rtkdiag_temporal_prev is not None
                        and rtkdiag_temporal_prev_hybrid is not None
                        and hp_valid_for_temporal
                    ):
                        alpha = float(temporal_hybdelta_alpha)
                        predicted_pos = (
                            rtkdiag_temporal_prev
                            + (np.asarray(hp, dtype=np.float64) - rtkdiag_temporal_prev_hybrid)
                        )
                        best_cand = min(
                            collected,
                            key=lambda c, _pred=predicted_pos: (
                                c[3][0] + alpha * float(np.linalg.norm(c[1] - _pred)),
                                c[3][1],
                            ),
                        )
                        label, selected_pos, _selected_diag, _selected_key = best_cand
                    else:
                        best_cand = min(collected, key=lambda c: c[3])
                        label, selected_pos, _selected_diag, _selected_key = best_cand
                    rtkdiag_pf_ref = selected_pos
                    rtkdiag_selected_label = str(label)
                    # Track last high-confidence Fix=4 anchor for bridge (independent of cluster bias).
                    if _selected_diag is not None and label != "pf_bridge":
                        try:
                            _sel_status = int(_selected_diag.get("final_status", "0"))
                            _sel_ratio = _diag_float(_selected_diag, "final_ratio")
                            _sel_residual = _diag_float(_selected_diag, "final_residual_rms")
                        except (ValueError, TypeError):
                            _sel_status = 0; _sel_ratio = 0.0; _sel_residual = 1e6
                        if (
                            _sel_status == 4
                            and _sel_ratio >= float(config.rtkdiag_candidate_bridge_fix4_min_ratio)
                            and _sel_residual <= float(config.rtkdiag_candidate_bridge_fix4_max_residual)
                        ):
                            rtkdiag_last_fix4_pos = np.asarray(selected_pos, dtype=np.float64).copy()
                            rtkdiag_last_fix4_t = float(t_key)
                    if is_temporal_prevdist or is_temporal_hybdelta:
                        rtkdiag_temporal_prev = np.asarray(selected_pos, dtype=np.float64)
                        if hp_valid_for_temporal:
                            rtkdiag_temporal_prev_hybrid = np.asarray(hp, dtype=np.float64)
                    rtkdiag_pf_stats.selected_counts[label] = (
                        rtkdiag_pf_stats.selected_counts.get(label, 0) + 1
                    )
                    soft_top_k = max(int(config.rtkdiag_candidate_soft_top_k), 1)
                    pf_update_refs = np.asarray(rtkdiag_pf_ref, dtype=np.float64).reshape(1, 3)
                    pf_update_weights = None
                    if soft_top_k > 1 and len(collected) > 1:
                        refs_list = [np.asarray(rtkdiag_pf_ref, dtype=np.float64)]
                        keys_list = [_selected_key]
                        for _cand_label, cand_pos, _cand_diag, cand_key in sorted(collected, key=lambda c: c[3]):
                            if len(refs_list) >= soft_top_k:
                                break
                            cand_arr = np.asarray(cand_pos, dtype=np.float64)
                            if not np.all(np.isfinite(cand_arr)):
                                continue
                            if any(float(np.linalg.norm(cand_arr - ref)) < 1.0e-6 for ref in refs_list):
                                continue
                            refs_list.append(cand_arr)
                            keys_list.append(cand_key)
                        if len(refs_list) > 1:
                            eps = max(float(config.rtkdiag_candidate_soft_weight_eps), 1.0e-9)
                            raw_w = np.asarray(
                                [
                                    1.0 / (max(float(key[0]), 0.0) + eps)
                                    if np.isfinite(float(key[0]))
                                    else 0.0
                                    for key in keys_list
                                ],
                                dtype=np.float64,
                            )
                            if float(np.sum(raw_w)) <= 0.0 or not np.all(np.isfinite(raw_w)):
                                raw_w = np.ones(len(refs_list), dtype=np.float64)
                            raw_w /= float(np.sum(raw_w))
                            pf_update_refs = np.stack(refs_list, axis=0)
                            pf_update_weights = raw_w
                    if config.rtkdiag_candidate_proposal_cloud:
                        _apply_rtkdiag_candidate_proposal_cloud(
                            pf,
                            pf_update_refs,
                            pf_update_weights,
                            spread_m=float(config.rtkdiag_candidate_proposal_spread_m),
                            seed=int(config.reservoir_stein_seed) + 2_000_003 + int(i),
                        )
                    else:
                        recenter_max = float(config.rtkdiag_candidate_recenter_max_shift_m)
                        if recenter_max > 0.0:
                            recenter_ref = rtkdiag_pf_ref
                            if pf_update_weights is not None and pf_update_refs.shape[0] > 1:
                                recenter_ref = np.sum(
                                    pf_update_refs * pf_update_weights[:, np.newaxis],
                                    axis=0,
                                )
                            shift_norm, recentered = pf.recenter_position(
                                recenter_ref,
                                max_shift_m=recenter_max,
                            )
                            if recentered:
                                rtkdiag_pf_stats.recenter_applied += 1
                            else:
                                rtkdiag_pf_stats.recenter_skipped += 1
                        sigma_candidate = max(float(config.rtkdiag_candidate_sigma_m), 0.01)
                        if pf_update_weights is not None and pf_update_refs.shape[0] > 1:
                            pf.position_mixture_update(
                                pf_update_refs,
                                sigma_pos=sigma_candidate,
                                mixture_weights=pf_update_weights,
                            )
                        else:
                            pf.position_update(
                                rtkdiag_pf_ref,
                                sigma_pos=sigma_candidate,
                            )
                    rtkdiag_pf_stats.pu_applied += 1
                    rtkdiag_pf_emit_here = True
            elif (is_temporal_prevdist or is_temporal_hybdelta) and hp is not None and np.all(np.isfinite(hp)) and not np.all(np.asarray(hp, dtype=np.float64) == 0.0):
                rtkdiag_temporal_prev = np.asarray(hp, dtype=np.float64)
                rtkdiag_temporal_prev_hybrid = np.asarray(hp, dtype=np.float64)
        if pf_diag_row is not None:
            pf_diag_row["rtkdiag_gated_options"] = int(rtkdiag_gated_options_epoch)
            pf_diag_row["rtkdiag_selected_label"] = rtkdiag_selected_label
            pf_diag_row["rtkdiag_candidate_epoch_ready"] = bool(
                rtkdiag_candidate_epoch_ready
            )
            pf_diag_row["rtkdiag_candidate_min_epoch"] = int(
                config.rtkdiag_candidate_min_epoch
            )
            pf_diag_row["rtkdiag_candidate_available"] = rtkdiag_pf_ref is not None
            pf_diag_row["rtkdiag_candidate_to_hybrid_m"] = _diag_distance(
                rtkdiag_pf_ref,
                hp,
            )
            pf_diag_row["rtkdiag_candidate_to_ref_m"] = _diag_distance(
                rtkdiag_pf_ref,
                ref_diag_epoch,
            )
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_after_rtkdiag",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        # Phase 9b: tight-coupled IMU. After hybrid PU but BEFORE estimate /
        # emission, integrate IMU since the anchor and apply a per-particle
        # position pseudo-observation. This is the "tight" piece: every
        # particle gets an IMU log-likelihood, not a global rewrite. We
        # also evaluate whether to emit the PF estimate (vs hybrid) on
        # this epoch based on the hybrid Status.
        imu_tc_emit_pf_here = False
        ins_tc_emit_pf_here = False
        if use_imu_tc:
            t_now = float(times[i])
            st_now = (
                hybrid_status.get(round(t_now, 1)) if hybrid_status is not None else None
            )
            anchor_eligible_here = (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(hp == 0.0)
                and (
                    st_now is None
                    or imu_tc_anchor_set
                    and st_now not in {int(s) for s in config.imu_tc_emit_pf_hybrid_statuses}
                )
            )
            # Lazily seed the anchor on the very first epoch with a usable
            # position so the loop has something to integrate against.
            if imu_tc_anchor is None:
                seed_pos = None
                if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                    seed_pos = np.asarray(hp, dtype=np.float64)
                elif np.all(np.isfinite(positions[i])) and not np.all(positions[i] == 0.0):
                    seed_pos = np.asarray(positions[i], dtype=np.float64)
                if seed_pos is not None:
                    imu_tc_anchor = _IMUAnchor(
                        pos_ecef=seed_pos.copy(),
                        time=t_now,
                        yaw_enu_rad=0.0,
                        valid_yaw=False,
                        velocity_enu=np.zeros(3, dtype=np.float64),
                    )
                    imu_tc_stats.anchor_resets += 1

            if imu_tc_anchor is not None:
                imu_tc_stats.epochs_evaluated += 1

                # Refresh yaw from a recent velocity when we don't have one yet.
                if not imu_tc_anchor.valid_yaw:
                    v_guide_now = (
                        hybrid_velocity.get(round(t_now, 1)) if hybrid_velocity else None
                    )
                    if v_guide_now is not None and np.all(np.isfinite(v_guide_now)):
                        v_enu = _ecef_velocity_to_enu(imu_tc_anchor.pos_ecef, v_guide_now)
                        yaw, valid = _heading_from_velocity_enu(v_enu)
                        if valid:
                            imu_tc_anchor = _IMUAnchor(
                                pos_ecef=imu_tc_anchor.pos_ecef,
                                time=imu_tc_anchor.time,
                                yaw_enu_rad=yaw,
                                valid_yaw=True,
                                velocity_enu=v_enu,
                            )

                dr_seconds = t_now - imu_tc_anchor.time
                imu_tc_stats.record_dr_seconds(dr_seconds)

                run_pu = imu_tc_anchor.valid_yaw
                if not run_pu:
                    imu_tc_stats.pu_skipped_no_anchor += 1
                if run_pu and dr_seconds > float(config.imu_tc_max_dr_seconds):
                    imu_tc_stats.pu_skipped_dr_too_long += 1
                    run_pu = False

                imu_pred_pos = None
                if run_pu and imu_t is not None and imu is not None:
                    delta_enu, _vel_enu_end, n_samp = _integrate_imu_between(
                        imu,
                        imu_t,
                        imu_tc_anchor.time,
                        t_now,
                        imu_tc_anchor.yaw_enu_rad,
                        imu_tc_anchor.velocity_enu,
                    )
                    if n_samp == 0:
                        imu_tc_stats.pu_skipped_no_imu += 1
                        run_pu = False
                    else:
                        delta_ecef = _enu_delta_to_ecef(imu_tc_anchor.pos_ecef, delta_enu)
                        imu_pred_pos = imu_tc_anchor.pos_ecef + delta_ecef
                        if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                            disagreement = float(np.linalg.norm(imu_pred_pos - hp))
                            if disagreement > float(config.imu_tc_max_disagreement_m):
                                imu_tc_stats.pu_skipped_disagreement += 1
                                run_pu = False

                if run_pu and imu_pred_pos is not None:
                    sigma_imu = (
                        float(config.imu_tc_pos_sigma_base_m)
                        + float(config.imu_tc_pos_sigma_per_s) * float(max(dr_seconds, 0.0))
                    )
                    pf.position_update(imu_pred_pos, sigma_pos=max(sigma_imu, 0.05))
                    imu_tc_stats.pu_applied += 1

                # Decide emission policy for this epoch (post-IMU-PU).
                if (
                    st_now is not None
                    and int(st_now) in imu_tc_emit_set
                ):
                    imu_tc_emit_pf_here = True

                # Anchor-reset on Status=4 (or anything NOT in the emit-PF set).
                if (
                    anchor_eligible_here
                    and st_now is not None
                    and int(st_now) not in imu_tc_emit_set
                ):
                    # Derive anchor velocity from hybrid velocity guide when
                    # available; otherwise fall back to whatever the previous
                    # anchor velocity was (will get refreshed lazily).
                    v_anchor_enu = imu_tc_anchor.velocity_enu
                    yaw_anchor = imu_tc_anchor.yaw_enu_rad
                    valid_yaw_anchor = imu_tc_anchor.valid_yaw
                    v_guide_now = (
                        hybrid_velocity.get(round(t_now, 1)) if hybrid_velocity else None
                    )
                    if v_guide_now is not None and np.all(np.isfinite(v_guide_now)):
                        v_anchor_enu = _ecef_velocity_to_enu(
                            np.asarray(hp, dtype=np.float64), v_guide_now
                        )
                        yaw_new, valid_new = _heading_from_velocity_enu(v_anchor_enu)
                        if valid_new:
                            yaw_anchor = yaw_new
                            valid_yaw_anchor = True
                    # Static check: if IMU says vehicle is stopped at this
                    # anchor, force zero velocity (matches Phase 9a logic).
                    is_static_anchor = False
                    if imu_tc_anchor_acc is not None and imu_tc_anchor_gyro is not None and i < imu_tc_anchor_acc.size:
                        a = imu_tc_anchor_acc[i]
                        g = imu_tc_anchor_gyro[i]
                        if (
                            np.isfinite(a)
                            and np.isfinite(g)
                            and float(config.imu_tc_anchor_static_acc_low_mps2) <= a <= float(config.imu_tc_anchor_static_acc_high_mps2)
                            and g <= float(config.imu_tc_anchor_static_gyro_max_dps)
                        ):
                            is_static_anchor = True
                            v_anchor_enu = np.zeros(3, dtype=np.float64)
                    imu_tc_anchor = _IMUAnchor(
                        pos_ecef=np.asarray(hp, dtype=np.float64).copy(),
                        time=t_now,
                        yaw_enu_rad=yaw_anchor,
                        valid_yaw=valid_yaw_anchor,
                        velocity_enu=v_anchor_enu,
                    )
                    imu_tc_stats.anchor_resets += 1
                    if is_static_anchor:
                        imu_tc_stats.anchor_resets_static += 1

        # Phase 9c: full INS-GNSS EKF. The EKF owns attitude and IMU bias
        # estimation in local ENU; the PF still receives only a position
        # pseudo-observation through the existing GPU position_update path.
        if use_ins_tc and ins_ekf is not None:
            t_now = float(times[i])
            t_key = round(t_now, 1)
            st_now = hybrid_status.get(t_key) if hybrid_status is not None else None
            hp_valid = (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(np.asarray(hp, dtype=np.float64) == 0.0)
            )

            if (
                ins_origin_ecef is not None
                and ins_origin_lat is not None
                and ins_origin_lon is not None
            ):
                if ins_ekf.aligned and hp_valid:
                    p_meas_enu = _ecef_to_enu_at_origin(
                        np.asarray(hp, dtype=np.float64),
                        ins_origin_ecef,
                        ins_origin_lat,
                        ins_origin_lon,
                    )
                    if st_now == 4 and float(config.ins_tc_obs_status_4_sigma_m) > 0.0:
                        sigma = float(config.ins_tc_obs_status_4_sigma_m)
                        ins_ekf.update_position_enu(p_meas_enu, (sigma, sigma, sigma * 2.0))
                        ins_last_obs_t = t_now
                        ins_tc_stats.obs_status_4_used += 1
                        if bool(config.ins_tc_recenter_status4):
                            shift_norm, recentered = pf.recenter_position(
                                np.asarray(hp, dtype=np.float64),
                                max_shift_m=float(config.ins_tc_recenter_max_shift_m),
                            )
                            if recentered:
                                ins_tc_stats.recenter_applied += 1
                                ins_tc_stats.recenter_shift_sum_m += float(shift_norm)
                            else:
                                ins_tc_stats.recenter_skipped += 1
                    elif st_now == 3 and float(config.ins_tc_obs_status_3_sigma_m) > 0.0:
                        sigma = float(config.ins_tc_obs_status_3_sigma_m)
                        ins_ekf.update_position_enu(p_meas_enu, (sigma, sigma, sigma * 2.0))
                        ins_last_obs_t = t_now
                        ins_tc_stats.obs_status_3_used += 1

                ins_valid_for_emit = False
                if not ins_ekf.aligned:
                    ins_tc_stats.pu_skipped_not_aligned += 1
                elif not ins_ekf.yaw_initialized:
                    ins_tc_stats.pu_skipped_no_yaw += 1
                else:
                    dr_seconds = (
                        t_now - ins_last_obs_t
                        if ins_last_obs_t is not None
                        else float("inf")
                    )
                    if dr_seconds > float(config.ins_tc_max_dr_seconds):
                        ins_tc_stats.pu_skipped_dr_too_long += 1
                    else:
                        p_ins_ecef = ins_ekf.position_ecef(
                            ins_origin_ecef,
                            ins_origin_lat,
                            ins_origin_lon,
                        )
                        if hp_valid and float(np.linalg.norm(p_ins_ecef - hp)) > float(config.ins_tc_max_disagreement_m):
                            ins_tc_stats.pu_skipped_disagreement += 1
                        elif (
                            bool(config.ins_tc_quality_gate_enabled)
                            and bool(config.ins_tc_quality_gate_pu_skip)
                            and len(ins_tc_quality_window) > 0
                            and (
                                sum(1 for s in ins_tc_quality_window if int(s) == 4)
                                / max(1, len(ins_tc_quality_window))
                            )
                            >= float(config.ins_tc_quality_gate_max_fix_rate)
                        ):
                            ins_tc_stats.pu_skipped_disagreement += 1  # accounted under disagreement bucket
                            ins_tc_quality_gate_skip_count += 1
                        else:
                            sigma_ins = max(
                                float(config.ins_tc_pf_pu_floor_sigma_m),
                                min(
                                    float(config.ins_tc_pf_pu_ceiling_sigma_m),
                                    float(ins_ekf.position_sigma_m()),
                                ),
                            )
                            pf.position_update(p_ins_ecef, sigma_pos=sigma_ins)
                            if not defer_resample:
                                pf.resample_if_needed()
                            ins_tc_stats.pu_applied += 1
                            ins_valid_for_emit = True

                if (
                    ins_valid_for_emit
                    and st_now is not None
                    and int(st_now) in ins_tc_emit_set
                ):
                    ins_tc_emit_pf_here = True

        # GNSS quality gate: when enabled and recent fix rate is high,
        # suppress ins_tc PF-emit (defer to GNSS / hybrid). This keeps the
        # gain on low-baseline runs (canyon) while reducing the regression
        # on high-baseline runs (open sky). The window holds the previous
        # K epochs' hybrid statuses (4=Fix, 3=Float, 1=uncertain, 0=unknown).
        if use_ins_tc and bool(config.ins_tc_quality_gate_enabled):
            if (
                ins_tc_emit_pf_here
                and len(ins_tc_quality_window) > 0
            ):
                fix_count = sum(1 for s in ins_tc_quality_window if int(s) == 4)
                fix_rate = fix_count / max(1, len(ins_tc_quality_window))
                if fix_rate >= float(config.ins_tc_quality_gate_max_fix_rate):
                    ins_tc_emit_pf_here = False
                    ins_tc_quality_gate_skip_count += 1
        # Append CURRENT epoch's hybrid status to the rolling window so the
        # NEXT epoch's gate sees this one.
        if use_ins_tc:
            _st_for_window = (
                hybrid_status.get(round(float(times[i]), 1))
                if hybrid_status is not None
                else None
            )
            ins_tc_quality_window.append(
                int(_st_for_window) if _st_for_window is not None else 0
            )
        if pf_diag_row is not None:
            pf_diag_row["imu_tc_emit_pf_here"] = bool(imu_tc_emit_pf_here)
            pf_diag_row["ins_tc_emit_pf_here"] = bool(ins_tc_emit_pf_here)
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_before_emit",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )

        resampled_before_emit = False
        if config.enable_reservoir_stein:
            resampled_before_emit = _reservoir_stein_resample_if_needed(
                pf, config, reservoir_stein_stats, i
            )
            if resampled_before_emit:
                pr_obs_stats.deferred_resample_epochs += 1

        est = np.asarray(pf.estimate(), dtype=np.float64)
        emitted_source = "pf"
        if ins_tc_emit_pf_here:
            if (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(hp == 0.0)
                and float(np.linalg.norm(est[:3] - hp)) > float(config.ins_tc_emit_max_diff_m)
            ):
                positions[i] = np.asarray(hp, dtype=np.float64)
                ins_tc_stats.emit_skipped_pf_drift += 1
                emitted_source = "hybrid_ins_pf_drift"
            else:
                positions[i] = est[:3]
                ins_tc_stats.emit_pf_estimate += 1
                emitted_source = "pf_ins_tc"
        elif imu_tc_emit_pf_here:
            # Phase 9b: emit PF estimate on Status in emit_pf set, as long as
            # the PF didn't wander too far from hybrid (cloud-collapse guard).
            if (
                hp is not None
                and np.all(np.isfinite(hp))
                and not np.all(hp == 0.0)
                and float(np.linalg.norm(est[:3] - hp)) > float(config.imu_tc_emit_max_diff_m)
            ):
                positions[i] = np.asarray(hp, dtype=np.float64)
                imu_tc_stats.emit_skipped_pf_drift += 1
                emitted_source = "hybrid_imu_pf_drift"
            else:
                positions[i] = est[:3]
                imu_tc_stats.emit_pf_estimate += 1
                emitted_source = "pf_imu_tc"
        elif rtkdiag_pf_emit_here and rtkdiag_pf_ref is not None:
            # Phase 10i: a trusted relaxed-RTK candidate is a PF
            # pseudo-observation. The conservative default emits the PF only
            # when the weighted cloud follows the candidate; newer modes let
            # diagnostics-passing candidates fill the epochs the PF guard would
            # otherwise throw back to the hybrid floor.
            emit_mode = str(config.rtkdiag_candidate_emit_mode)
            candidate_delta_m = float(np.linalg.norm(est[:3] - rtkdiag_pf_ref))
            pf_close_to_candidate = candidate_delta_m <= float(
                config.rtkdiag_candidate_emit_max_diff_m
            )
            if emit_mode == "candidate":
                positions[i] = np.asarray(rtkdiag_pf_ref, dtype=np.float64)
                rtkdiag_pf_stats.emit_candidate += 1
                emitted_source = "rtkdiag_candidate"
            elif pf_close_to_candidate:
                positions[i] = est[:3]
                rtkdiag_pf_stats.emit_pf_estimate += 1
                emitted_source = "pf_rtkdiag"
            elif emit_mode == "candidate-on-drift":
                positions[i] = np.asarray(rtkdiag_pf_ref, dtype=np.float64)
                rtkdiag_pf_stats.emit_candidate += 1
                rtkdiag_pf_stats.emit_skipped_pf_drift += 1
                emitted_source = "rtkdiag_candidate_on_drift"
            else:
                if hp is not None and np.all(np.isfinite(hp)) and not np.all(hp == 0.0):
                    positions[i] = np.asarray(hp, dtype=np.float64)
                    emitted_source = "hybrid_rtkdiag_pf_drift"
                else:
                    positions[i] = est[:3]
                    emitted_source = "pf_rtkdiag_no_hybrid"
                rtkdiag_pf_stats.emit_skipped_pf_drift += 1
        elif use_rtkdiag_pf:
            positions[i], emitted_source = _select_rtkdiag_fallback(
                i,
                t_now,
                est[:3],
                suffix="_rtkdiag",
            )
        elif (
            use_hybrid
            and not config.hybrid_emit_pf_estimate
            and hp is not None
            and np.all(np.isfinite(hp))
            and not np.all(hp == 0.0)
        ):
            # Phase 6 passthrough: trust hybrid as the floor.
            positions[i] = np.asarray(hp, dtype=np.float64)
            emitted_source = "hybrid"
        elif (
            use_hybrid
            and config.hybrid_emit_pf_estimate
            and config.hybrid_emit_pf_statuses
            and hp is not None
            and np.all(np.isfinite(hp))
            and not np.all(hp == 0.0)
            and hybrid_status is not None
        ):
            st_emit = hybrid_status.get(round(float(times[i]), 1))
            if st_emit is not None and int(st_emit) not in {
                int(s) for s in config.hybrid_emit_pf_statuses
            }:
                positions[i] = np.asarray(hp, dtype=np.float64)
                emitted_source = "hybrid_status_gate"
            else:
                positions[i] = est[:3]
                emitted_source = "pf_hybrid_emit"
        else:
            # Phase 7 / non-hybrid: emit the PF's own weighted-mean estimate
            # so any DD-AFV / Doppler-KF correction shows up in the score.
            positions[i] = est[:3]
            emitted_source = "pf"

        if (
            emitted_source.startswith("rtkdiag_candidate")
            or emitted_source.startswith("rtkdiag_fallback_wls")
            or emitted_source.startswith("rtkdiag_fallback_hybrid")
        ):
            if np.all(np.isfinite(positions[i])):
                rtkdiag_last_good_pos = np.asarray(positions[i], dtype=np.float64).copy()
                rtkdiag_last_good_t = t_now

        if defer_resample and not config.enable_reservoir_stein:
            did_resample = pf.resample_if_needed()
            if did_resample:
                pr_obs_stats.deferred_resample_epochs += 1
        else:
            did_resample = bool(resampled_before_emit)
        if pf_diag_row is not None:
            pf_diag_row["resampled_before_emit"] = bool(resampled_before_emit)
            pf_diag_row["resampled_epoch_end"] = bool(did_resample)
            pf_diag_row["emitted_source"] = emitted_source
            pf_diag_row["emit_to_pf_estimate_m"] = _diag_distance(positions[i], est[:3])
            pf_diag_row["emit_to_hybrid_m"] = _diag_distance(positions[i], hp)
            pf_diag_row["emit_to_rtkdiag_candidate_m"] = _diag_distance(
                positions[i],
                rtkdiag_pf_ref,
            )
            pf_diag_row["emit_to_ref_m"] = _diag_distance(positions[i], ref_diag_epoch)
            _capture_pf_internal_state(
                pf,
                pf_diag_row,
                "pf_epoch_end",
                n_particles=int(config.n_particles),
                reference_ecef=ref_diag_epoch,
            )
            internal_diagnostics.append(pf_diag_row)

    elapsed = (time.perf_counter() - t0) * 1000.0
    ms_per_epoch = elapsed / max(n_epochs, 1)

    # Phase 9a: ZUPT before any smoother / FGO so jitter on stops is
    # damped first.
    if config.enable_zupt:
        positions = _apply_zupt(
            positions=positions,
            times=times,
            imu=imu,
            config=config,
            hybrid_status=hybrid_status,
            stats=zupt_stats,
        )

    # Phase 8: TDCP-anchored hybrid smoother. Runs BEFORE Phase 4 so
    # Phase 4 can operate on the smoothed trajectory. With TDCP off this
    # is a no-op.
    if config.enable_tdcp_smoother:
        positions = _apply_tdcp_smoother(
            positions=positions,
            times=times,
            data=data,
            config=config,
            hybrid_pos=hybrid_pos,
            hybrid_status=hybrid_status,
            stats=tdcp_stats,
        )

    if config.enable_low_sat_bridge:
        positions_before_bridge = positions.copy()
        positions = _apply_startup_wls_bridge(
            positions=positions,
            times=times,
            wls_quality=wls_quality,
            config=config,
            stats=low_sat_bridge_stats,
        )
        positions = _apply_low_sat_bridge(
            positions=positions,
            times=times,
            pr_used_counts=pr_used_counts,
            config=config,
            stats=low_sat_bridge_stats,
        )
        if collect_internal_diagnostics and internal_diagnostics:
            for row in internal_diagnostics:
                idx_raw = row.get("epoch")
                try:
                    idx = int(idx_raw)
                except (TypeError, ValueError):
                    continue
                if idx < 0 or idx >= int(positions.shape[0]):
                    continue
                before = np.asarray(positions_before_bridge[idx], dtype=np.float64)
                after = np.asarray(positions[idx], dtype=np.float64)
                changed = (
                    np.all(np.isfinite(before))
                    and np.all(np.isfinite(after))
                    and float(np.linalg.norm(after - before)) > 1.0e-9
                )
                row["low_sat_bridge_rewritten"] = bool(changed)
                if changed:
                    row["emit_to_ref_pre_bridge_m"] = row.get("emit_to_ref_m", "")
                    tow_key = round(float(times[idx]), 1)
                    ref_bridge = (
                        reference_pos.get(tow_key)
                        if reference_pos is not None
                        else None
                    )
                    row["emit_to_ref_m"] = _diag_distance(after, ref_bridge)
                    row["emitted_source"] = f"{row.get('emitted_source', '')}+bridge"

    # Phase 4: post-process FGO + LAMBDA partial fix over sliding windows.
    if config.enable_fgo_lambda and any(c is not None for c in fgo_dd_cache):
        # Build per-epoch indices that are eligible for FGO replacement
        # based on the hybrid Status gate (C2). When the gate is empty or
        # no hybrid status was loaded, every epoch is eligible.
        protect_indices: set[int] = set()
        if hybrid_status is not None:
            allowed: set[int] | None = (
                set(int(s) for s in config.fgo_apply_hybrid_statuses)
                if config.fgo_apply_hybrid_statuses
                else None
            )
            for i in range(n_epochs):
                st = hybrid_status.get(round(float(times[i]), 1))
                if st is None:
                    continue
                if allowed is not None and st not in allowed:
                    # Protect epoch (keep hybrid passthrough) when it has a
                    # status NOT in the allowed-rewrite set.
                    protect_indices.add(i)
                elif allowed is None and int(st) == 4:
                    # Default guard: when no apply list is configured, never
                    # let FGO overwrite Status=4 cm-class hybrid epochs.
                    protect_indices.add(i)

        # D2: build per-epoch prior sigma array. Status values NOT in the
        # apply set get the tight anchor sigma; Status values in the apply
        # set get the loose sigma; missing-status epochs default to loose.
        prior_sigmas_arr: np.ndarray | None = None
        if (
            hybrid_status is not None
            and float(config.fgo_anchor_sigma_m) > 0.0
            and config.fgo_apply_hybrid_statuses
        ):
            allowed = set(int(s) for s in config.fgo_apply_hybrid_statuses)
            prior_sigmas_arr = np.full(n_epochs, float(config.fgo_loose_sigma_m), dtype=np.float64)
            for i in range(n_epochs):
                st = hybrid_status.get(round(float(times[i]), 1))
                if st is not None and st not in allowed:
                    prior_sigmas_arr[i] = float(config.fgo_anchor_sigma_m)

        ct_motion_deltas = None
        ct_motion_sigmas = None
        if config.enable_ct_spline_motion_prior:
            ct_motion_deltas, ct_motion_sigmas = _ct_spline_motion_prior(
                positions, times, config
            )

        positions = _apply_fgo_lambda(
            positions=positions,
            dd_cache=fgo_dd_cache,
            dd_pr_cache=fgo_dd_pr_cache,
            config=config,
            stats=fgo_stats,
            protect_indices=protect_indices,
            prior_sigmas=prior_sigmas_arr,
            anchor_positions=fgo_dd_pr_ls_anchor_pos,
            anchor_sigmas=fgo_dd_pr_ls_anchor_sigma,
            motion_deltas=ct_motion_deltas,
            motion_sigmas=ct_motion_sigmas,
        )

    if ins_ekf is not None:
        ins_tc_stats.final_acc_bias_norm = float(np.linalg.norm(ins_ekf.b_a))
        ins_tc_stats.final_gyro_bias_norm_dps = float(
            np.linalg.norm(ins_ekf.b_g) * 180.0 / math.pi
        )
        ins_tc_stats.final_pos_sigma_m = float(ins_ekf.position_sigma_m())

    return (
        positions,
        ms_per_epoch,
        pr_obs_stats,
        reservoir_stein_stats,
        dd_stats,
        gate_stats,
        hybrid_stats,
        rtkdiag_pf_stats,
        fgo_stats,
        tdcp_stats,
        low_sat_bridge_stats,
        zupt_stats,
        imu_tc_stats,
        ins_tc_stats,
        internal_diagnostics,
    )


def _apply_fgo_lambda(
    *,
    positions: np.ndarray,
    dd_cache: list,
    dd_pr_cache: list | None,
    config: CTRBPFConfig,
    stats: _FGOStats,
    protect_indices: set[int] | None = None,
    prior_sigmas: np.ndarray | None = None,
    anchor_positions: np.ndarray | None = None,
    anchor_sigmas: np.ndarray | None = None,
    motion_deltas: np.ndarray | None = None,
    motion_sigmas: np.ndarray | None = None,
) -> np.ndarray:
    """Slide a window over the trajectory and run solve_local_fgo_with_lambda.

    Replaces ``positions[start:end+1]`` with the FGO output whenever LAMBDA
    accepts at least ``fgo_min_fixed_to_apply`` integer fixes via the ratio
    test. The replacement is per-window; overlapping windows write the most
    recent solve, so a stride < window_size effectively re-solves earlier
    epochs with potentially better integer support.
    """
    from gnss_gpu.local_fgo import (
        LambdaFixConfig,
        LocalFgoConfig,
        LocalFgoProblem,
        LocalFgoWindow,
        solve_local_fgo_with_lambda,
    )

    n = int(positions.shape[0])
    win_size = max(2, int(config.fgo_window_size))
    stride = max(1, int(config.fgo_window_stride))
    if win_size > n:
        return positions

    base_cfg = LocalFgoConfig(
        prior_sigma_m=float(config.fgo_prior_sigma_m),
        dd_sigma_cycles=float(config.fgo_dd_sigma_cycles),
        dd_pr_sigma_m=float(config.fgo_dd_pr_sigma_m),
    )
    lam_cfg = LambdaFixConfig(
        ratio_threshold=float(config.fgo_lambda_ratio),
        min_epochs=int(config.fgo_lambda_min_epochs),
        max_epoch_gap=int(config.fgo_lambda_max_epoch_gap),
    )

    out = positions.copy()
    # Bug #2 fix: keep an immutable copy of the original (hybrid passthrough)
    # positions. Each window must use the original for both the soft prior
    # anchor and the ``min_correction_m`` comparison; otherwise overlapping
    # windows leak state via ``out`` and the result becomes stride/order
    # dependent (small sub-threshold moves can accumulate, and the prior
    # anchor drifts between windows).
    original = positions.copy()
    start = 0
    while start + win_size <= n:
        end = start + win_size - 1
        window_dd = dd_cache[start : end + 1]
        n_dd_epochs = sum(1 for d in window_dd if d is not None)
        if n_dd_epochs < int(config.fgo_lambda_min_epochs):
            start += stride
            continue

        stats.windows_attempted += 1
        slice_init = np.asarray(out[start : end + 1], dtype=np.float64).copy()
        slice_orig = np.asarray(original[start : end + 1], dtype=np.float64).copy()
        window_dd_pr = (
            dd_pr_cache[start : end + 1] if dd_pr_cache is not None else None
        )
        slice_prior_sigmas = (
            np.asarray(prior_sigmas[start : end + 1], dtype=np.float64).copy()
            if prior_sigmas is not None
            else None
        )
        slice_prior_positions = slice_orig.copy()
        if anchor_positions is not None and anchor_sigmas is not None:
            anchor_mode = str(config.dd_pr_ls_anchor_mode).strip().lower()
            slice_anchor_pos = np.asarray(
                anchor_positions[start : end + 1], dtype=np.float64
            ).reshape(win_size, 3)
            slice_anchor_sigmas = np.asarray(
                anchor_sigmas[start : end + 1], dtype=np.float64
            ).reshape(win_size)
            anchor_mask = (
                np.all(np.isfinite(slice_anchor_pos), axis=1)
                & np.isfinite(slice_anchor_sigmas)
                & (slice_anchor_sigmas > 0.0)
            )
            if np.any(anchor_mask):
                if (
                    anchor_mode in {"prior", "initial"}
                    and bool(config.dd_pr_ls_anchor_set_initial)
                ):
                    slice_init[anchor_mask] = slice_anchor_pos[anchor_mask]
                if anchor_mode == "prior":
                    slice_prior_positions[anchor_mask] = slice_anchor_pos[anchor_mask]
                    if slice_prior_sigmas is None:
                        slice_prior_sigmas = np.full(
                            win_size, float(config.fgo_prior_sigma_m), dtype=np.float64
                        )
                    slice_prior_sigmas[anchor_mask] = np.minimum(
                        slice_prior_sigmas[anchor_mask],
                        slice_anchor_sigmas[anchor_mask],
                    )
        slice_motion_deltas = (
            np.asarray(motion_deltas[start:end], dtype=np.float64).copy()
            if motion_deltas is not None
            else None
        )
        slice_motion_sigmas = (
            np.asarray(motion_sigmas[start:end], dtype=np.float64).copy()
            if motion_sigmas is not None
            else None
        )
        problem = LocalFgoProblem(
            initial_positions_ecef=slice_init,
            window=LocalFgoWindow(0, win_size - 1),
            motion_deltas_ecef=slice_motion_deltas,
            motion_sigmas_m=slice_motion_sigmas,
            dd_carrier=window_dd,
            dd_pseudorange=window_dd_pr,
            prior_positions_ecef=slice_prior_positions,
            prior_sigmas_m=slice_prior_sigmas,
        )
        try:
            result, summary = solve_local_fgo_with_lambda(problem, base_cfg, lam_cfg)
        except Exception:
            start += stride
            continue
        stats.windows_solved += 1
        n_fixed = int(summary.get("n_fixed", 0))
        n_fixed_obs = int(summary.get("n_fixed_observations", 0))
        stats.n_fixed_total += n_fixed
        stats.n_fixed_observations_total += n_fixed_obs
        for iteration_info in summary.get("iterations", []) or []:
            stats.lambda_diag_windows += 1
            stats.lambda_tracks_total += int(iteration_info.get("n_tracks", 0))
            stats.lambda_segments_total += int(iteration_info.get("n_segments", 0))
            stats.lambda_candidates_total += int(iteration_info.get("n_candidates", 0))
            stats.lambda_ratio_rejected_total += int(
                iteration_info.get("n_ratio_rejected", 0)
            )
            best_ratio = float(iteration_info.get("best_ratio", 0.0))
            if np.isfinite(best_ratio):
                stats.lambda_best_ratio = max(float(stats.lambda_best_ratio), best_ratio)
            elif best_ratio == float("inf"):
                stats.lambda_best_ratio = float("inf")
            stats.lambda_ratio_median_sum += float(iteration_info.get("ratio_median", 0.0))
            stats.lambda_ratio_p90_sum += float(iteration_info.get("ratio_p90", 0.0))
            stats.lambda_segment_n_epochs_median_sum += float(
                iteration_info.get("segment_n_epochs_median", 0.0)
            )
            stats.lambda_segment_n_epochs_max = max(
                int(stats.lambda_segment_n_epochs_max),
                int(iteration_info.get("segment_n_epochs_max", 0)),
            )
            stats.lambda_segment_variance_median_sum += float(
                iteration_info.get("segment_variance_median", 0.0)
            )
            stats.lambda_segment_abs_frac_median_sum += float(
                iteration_info.get("segment_abs_frac_median", 0.0)
            )
            stats.lambda_segment_abs_frac_p90_sum += float(
                iteration_info.get("segment_abs_frac_p90", 0.0)
            )
        postfit = summary.get("postfit", {}) or {}
        stats.postfit_fixed_count_total += int(postfit.get("carrier_fixed_count", 0))
        stats.postfit_fixed_abs_cycles_median_sum += float(
            postfit.get("carrier_fixed_abs_cycles_median", 0.0)
        )
        stats.postfit_fixed_abs_cycles_p90_sum += float(
            postfit.get("carrier_fixed_abs_cycles_p90", 0.0)
        )
        stats.postfit_float_count_total += int(postfit.get("carrier_float_count", 0))
        stats.postfit_float_afv_abs_cycles_median_sum += float(
            postfit.get("carrier_float_afv_abs_cycles_median", 0.0)
        )
        stats.postfit_float_afv_abs_cycles_p90_sum += float(
            postfit.get("carrier_float_afv_abs_cycles_p90", 0.0)
        )
        stats.postfit_dd_pr_count_total += int(postfit.get("dd_pr_count", 0))
        stats.postfit_dd_pr_abs_m_median_sum += float(
            postfit.get("dd_pr_abs_m_median", 0.0)
        )
        stats.postfit_dd_pr_abs_m_p90_sum += float(
            postfit.get("dd_pr_abs_m_p90", 0.0)
        )
        if n_fixed >= int(config.fgo_min_fixed_to_apply):
            new_positions = np.asarray(result.positions_ecef, dtype=np.float64)
            if (
                new_positions.shape == slice_init.shape
                and np.all(np.isfinite(new_positions))
            ):
                # Bug #1 fix: per-epoch fix mask. Restrict the rewrite to
                # epochs that actually had a DD carrier ambiguity integer-
                # fixed via LAMBDA (window-relative indices in
                # ``summary["fixed_epochs"]``). Float-only or weakly-
                # constrained epochs in the same window kept the hybrid
                # passthrough.
                fixed_rel_epochs = set(int(e) for e in summary.get("fixed_epochs", []) or [])
                min_corr = float(config.fgo_min_correction_m)
                replaced = 0
                for rel_i in range(win_size):
                    abs_i = start + rel_i
                    if protect_indices is not None and abs_i in protect_indices:
                        continue
                    if (
                        bool(config.fgo_apply_fixed_epochs_only)
                        and fixed_rel_epochs
                        and rel_i not in fixed_rel_epochs
                    ):
                        continue
                    if min_corr > 0.0:
                        delta = float(np.linalg.norm(new_positions[rel_i] - slice_orig[rel_i]))
                        if delta < min_corr:
                            continue
                    out[abs_i] = new_positions[rel_i]
                    replaced += 1
                if replaced > 0:
                    stats.windows_applied += 1
                    stats.epochs_replaced += replaced
        start += stride

    return out


def _write_pos_file(
    path: Path,
    times: np.ndarray,
    positions: np.ndarray,
    status: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("% CT-RBPF-FGO PPC port (gnss_gpu)\n")
        fh.write("% GPST                tow         x-ecef(m)        y-ecef(m)        z-ecef(m)   Q  ns   sdx    sdy    sdz   age  ratio\n")
        for tow, pos in zip(times, positions, strict=True):
            fh.write(
                f"0 {float(tow):14.4f} "
                f"{pos[0]:16.4f} {pos[1]:16.4f} {pos[2]:16.4f}  "
                f"   1   0  0.000  0.000  0.000  0.00  0.0   {status}\n"
            )


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parse_path_list(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def _parse_label_list(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_label_factor_list(raw: str) -> tuple[tuple[str, float], ...]:
    factors: list[tuple[str, float]] = []
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"invalid label factor spec {spec!r}; expected label=factor"
            )
        label, factor_raw = (part.strip() for part in spec.split("=", 1))
        if not label:
            raise ValueError(
                f"invalid label factor spec {spec!r}; expected label=factor"
            )
        factors.append((label, float(factor_raw)))
    return tuple(factors)


def _parse_diag_threshold_list(raw: str) -> tuple[tuple[str, float], ...]:
    thresholds: list[tuple[str, float]] = []
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if "=" in spec:
            key, value_raw = (part.strip() for part in spec.split("=", 1))
        elif ":" in spec:
            key, value_raw = (part.strip() for part in spec.split(":", 1))
        else:
            raise ValueError(
                f"invalid diagnostic threshold {spec!r}; expected field=value"
            )
        if not key:
            raise ValueError(
                f"invalid diagnostic threshold {spec!r}; expected field=value"
            )
        thresholds.append((key, float(value_raw)))
    return tuple(thresholds)


def _parse_run_label_blocks(raw: str) -> dict[tuple[str, str], set[str]]:
    blocks: dict[tuple[str, str], set[str]] = {}
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"invalid run label block spec {spec!r}; expected city/run=label+label"
            )
        run_key, labels_raw = spec.split("=", 1)
        if "/" not in run_key:
            raise ValueError(
                f"invalid run label block key {run_key!r}; expected city/run"
            )
        city, run = (part.strip() for part in run_key.split("/", 1))
        labels = {p.strip() for p in labels_raw.replace(",", "+").split("+") if p.strip()}
        if not city or not run or not labels:
            raise ValueError(
                f"invalid run label block spec {spec!r}; expected city/run=label+label"
            )
        blocks[(city, run)] = set(labels)
    return blocks


def _emission_count(
    full_ref_rows: list[tuple[float, np.ndarray]],
    times_used: np.ndarray,
) -> int:
    pos_map = {round(float(t), 1): True for t in times_used}
    return int(sum(1 for tow, _ in full_ref_rows if pos_map.get(round(float(tow), 1))))


def _config_variants(args: argparse.Namespace) -> list[CTRBPFConfig]:
    variants: list[CTRBPFConfig] = []
    dd_systems = tuple(s.strip() for s in args.dd_systems.split(",") if s.strip())
    base = dict(
        n_particles=args.n_particles,
        sigma_pr=args.sigma_pr,
        pr_ess_guard_min_ratio=args.pr_ess_guard_min_ratio,
        pr_ess_guard_max_iters=args.pr_ess_guard_max_iters,
        pr_gmm_statuses=tuple(
            int(s.strip()) for s in args.pr_gmm_statuses.split(",") if s.strip()
        ),
        pr_gmm_w_los=args.pr_gmm_w_los,
        pr_gmm_mu_nlos_m=args.pr_gmm_mu_nlos_m,
        pr_gmm_sigma_nlos_m=args.pr_gmm_sigma_nlos_m,
        pr_gmm_hybrid_loose_sigma_m=args.pr_gmm_hybrid_loose_sigma_m,
        pr_gmm_clock_quantile=args.pr_gmm_clock_quantile,
        pr_weight_mode=args.pr_weight_mode,
        pr_weight_ref_cn0=args.pr_weight_ref_cn0,
        pr_weight_min=args.pr_weight_min,
        pr_weight_max=args.pr_weight_max,
        pr_systems=tuple(s.strip() for s in args.pr_systems.split(",") if s.strip()),
        pr_min_elevation_deg=args.pr_min_elevation_deg,
        pr_atmosphere_model=args.pr_atmosphere_model,
        pr_atmosphere_scale=args.pr_atmosphere_scale,
        pr_atmosphere_extra_zenith_m=args.pr_atmosphere_extra_zenith_m,
        pr_slant_delay_zenith_m=args.pr_slant_delay_zenith_m,
        pr_prefit_gate_m=args.pr_prefit_gate_m,
        pr_prefit_gate_min_sats=args.pr_prefit_gate_min_sats,
        pr_prefit_gate_keep_best=args.pr_prefit_gate_keep_best,
        pr_prefit_ref=args.pr_prefit_ref,
        pr_prefit_per_system=bool(args.pr_prefit_per_system),
        pr_skip_statuses=tuple(
            int(s.strip()) for s in args.pr_skip_statuses.split(",") if s.strip()
        ),
        defer_epoch_resample=bool(args.defer_epoch_resample),
        enable_reservoir_stein=bool(args.enable_reservoir_stein),
        reservoir_stein_size=args.reservoir_stein_size,
        reservoir_stein_elite_fraction=args.reservoir_stein_elite_fraction,
        reservoir_stein_steps=args.reservoir_stein_steps,
        reservoir_stein_step_size=args.reservoir_stein_step_size,
        reservoir_stein_repulsion_scale=args.reservoir_stein_repulsion_scale,
        reservoir_stein_guide_sigma_m=args.reservoir_stein_guide_sigma_m,
        reservoir_stein_guide_sigma_cb_m=args.reservoir_stein_guide_sigma_cb_m,
        reservoir_stein_seed=args.reservoir_stein_seed,
        rtkdiag_candidate_sigma_m=args.rtkdiag_candidate_sigma_m,
        rtkdiag_candidate_ratio_min=args.rtkdiag_candidate_ratio_min,
        rtkdiag_candidate_residual_rms_max=args.rtkdiag_candidate_residual_rms_max,
        rtkdiag_candidate_main_status5_residual_rms_max=args.rtkdiag_candidate_main_status5_residual_rms_max,
        rtkdiag_candidate_rms_prefilter_k=args.rtkdiag_candidate_rms_prefilter_k,
        rtkdiag_candidate_cluster_vote_radius_m=args.rtkdiag_candidate_cluster_vote_radius_m,
        rtkdiag_candidate_ranker_score_path=args.rtkdiag_candidate_ranker_score_path,
        rtkdiag_candidate_ranker_stickiness=args.rtkdiag_candidate_ranker_stickiness,
        rtkdiag_candidate_bridge_enable=args.rtkdiag_candidate_bridge_enable,
        rtkdiag_candidate_bridge_max_s=args.rtkdiag_candidate_bridge_max_s,
        rtkdiag_candidate_bridge_residual_rms_m=args.rtkdiag_candidate_bridge_residual_rms_m,
        rtkdiag_candidate_bridge_anchor_mode=args.rtkdiag_candidate_bridge_anchor_mode,
        rtkdiag_candidate_bridge_fix4_min_ratio=args.rtkdiag_candidate_bridge_fix4_min_ratio,
        rtkdiag_candidate_bridge_fix4_max_residual=args.rtkdiag_candidate_bridge_fix4_max_residual,
        rtkdiag_candidate_max_to_hybrid_m=args.rtkdiag_candidate_max_to_hybrid_m,
        rtkdiag_candidate_emit_max_diff_m=args.rtkdiag_candidate_emit_max_diff_m,
        rtkdiag_candidate_recenter_max_shift_m=args.rtkdiag_candidate_recenter_max_shift_m,
        rtkdiag_candidate_soft_top_k=args.rtkdiag_candidate_soft_top_k,
        rtkdiag_candidate_soft_weight_eps=args.rtkdiag_candidate_soft_weight_eps,
        rtkdiag_candidate_proposal_cloud=bool(args.rtkdiag_candidate_proposal_cloud),
        rtkdiag_candidate_proposal_spread_m=args.rtkdiag_candidate_proposal_spread_m,
        rtkdiag_candidate_select_mode=args.rtkdiag_candidate_select_mode,
        rtkdiag_candidate_emit_mode=args.rtkdiag_candidate_emit_mode,
        rtkdiag_candidate_min_epoch=args.rtkdiag_candidate_min_epoch,
        rtkdiag_candidate_require_any_diag_fields=tuple(
            _parse_label_list(args.rtkdiag_candidate_require_any_diag_fields)
        ),
        rtkdiag_candidate_require_all_diag_fields=tuple(
            _parse_label_list(args.rtkdiag_candidate_require_all_diag_fields)
        ),
        rtkdiag_candidate_min_diag_fields=_parse_diag_threshold_list(
            args.rtkdiag_candidate_min_diag_fields
        ),
        rtkdiag_candidate_max_diag_fields=_parse_diag_threshold_list(
            args.rtkdiag_candidate_max_diag_fields
        ),
        rtkdiag_candidate_fallback_mode=args.rtkdiag_candidate_fallback_mode,
        rtkdiag_candidate_fallback_max_wls_rms_m=(
            args.rtkdiag_candidate_fallback_max_wls_rms_m
        ),
        rtkdiag_candidate_fallback_max_wls_pdop=(
            args.rtkdiag_candidate_fallback_max_wls_pdop
        ),
        rtkdiag_candidate_fallback_max_wls_to_pf_m=(
            args.rtkdiag_candidate_fallback_max_wls_to_pf_m
        ),
        rtkdiag_candidate_fallback_max_hold_age_s=(
            args.rtkdiag_candidate_fallback_max_hold_age_s
        ),
        rtkdiag_candidate_label_factors=_parse_label_factor_list(
            args.rtkdiag_candidate_label_factors
        ),
        sigma_pos=args.sigma_pos,
        sigma_cb=args.sigma_cb,
        spread_pos_init=args.spread_pos_init,
        spread_cb_init=args.spread_cb_init,
        sigma_doppler_mps=args.sigma_doppler_mps,
        doppler_systems=tuple(s.strip() for s in args.doppler_systems.split(",") if s.strip()),
        doppler_prefit_gate_mps=args.doppler_prefit_gate_mps,
        doppler_prefit_gate_min_sats=args.doppler_prefit_gate_min_sats,
        velocity_init_sigma=args.velocity_init_sigma,
        velocity_process_noise=args.velocity_process_noise,
        position_update_sigma_m=args.position_update_sigma_m,
        position_update_min_epoch=args.position_update_min_epoch,
        position_update_min_pr_sats=args.position_update_min_pr_sats,
        position_update_max_wls_rms_m=args.position_update_max_wls_rms_m,
        position_update_max_wls_pdop=args.position_update_max_wls_pdop,
        position_update_max_wls_to_pf_m=args.position_update_max_wls_to_pf_m,
        enable_correct_clock_bias=not args.disable_correct_clock_bias,
        dd_sigma_cycles=args.dd_sigma_cycles,
        dd_min_pairs=args.dd_min_pairs,
        dd_min_pairs_update=args.dd_min_pairs_update,
        dd_systems=dd_systems,
        dd_base_interp=bool(args.dd_base_interp),
        dd_min_elevation_deg=args.dd_min_elevation_deg,
        dd_min_snr=args.dd_min_snr,
        dd_keep_best=args.dd_keep_best,
        dd_pr_pair_residual_max_m=args.dd_pr_pair_residual_max_m,
        dd_pr_epoch_median_residual_max_m=args.dd_pr_epoch_median_residual_max_m,
        dd_pr_gate_min_pairs=args.dd_pr_gate_min_pairs,
        enable_dd_pr_ls_anchor=bool(args.enable_dd_pr_ls_anchor),
        dd_pr_ls_anchor_min_pairs=args.dd_pr_ls_anchor_min_pairs,
        dd_pr_ls_anchor_dd_sigma_m=args.dd_pr_ls_anchor_dd_sigma_m,
        dd_pr_ls_anchor_solve_prior_sigma_m=args.dd_pr_ls_anchor_solve_prior_sigma_m,
        dd_pr_ls_anchor_prior_sigma_m=args.dd_pr_ls_anchor_prior_sigma_m,
        dd_pr_ls_anchor_max_shift_m=args.dd_pr_ls_anchor_max_shift_m,
        dd_pr_ls_anchor_max_postfit_rms_m=args.dd_pr_ls_anchor_max_postfit_rms_m,
        dd_pr_ls_anchor_statuses=tuple(
            int(s.strip()) for s in args.dd_pr_ls_anchor_statuses.split(",") if s.strip()
        ),
        dd_pr_ls_anchor_set_initial=not bool(args.dd_pr_ls_anchor_no_initial),
        dd_pr_ls_anchor_mode=str(args.dd_pr_ls_anchor_mode),
        rbpf_kf_gate_max_doppler_wls_rms_mps=args.rbpf_velocity_kf_gate_max_doppler_wls_rms_mps,
        rbpf_kf_gate_max_doppler_wls_speed_mps=args.rbpf_velocity_kf_gate_max_doppler_wls_speed_mps,
        hybrid_sigma_m=args.hybrid_sigma_m,
        hybrid_recenter_max_shift_m=args.hybrid_recenter_max_shift_m,
        hybrid_emit_pf_statuses=tuple(
            int(s.strip()) for s in args.hybrid_emit_pf_statuses.split(",") if s.strip()
        ),
        fgo_window_size=args.fgo_window_size,
        fgo_window_stride=args.fgo_window_stride,
        fgo_lambda_ratio=args.fgo_lambda_ratio,
        fgo_lambda_min_epochs=args.fgo_lambda_min_epochs,
        fgo_lambda_max_epoch_gap=args.fgo_lambda_max_epoch_gap,
        fgo_min_fixed_to_apply=args.fgo_min_fixed_to_apply,
        fgo_prior_sigma_m=args.fgo_prior_sigma_m,
        fgo_dd_sigma_cycles=args.fgo_dd_sigma_cycles,
        fgo_dd_pr_sigma_m=args.fgo_dd_pr_sigma_m,
        fgo_apply_hybrid_statuses=tuple(
            int(s.strip()) for s in args.fgo_apply_hybrid_statuses.split(",") if s.strip()
        ),
        fgo_anchor_sigma_m=args.fgo_anchor_sigma_m,
        fgo_loose_sigma_m=args.fgo_loose_sigma_m,
        enable_ct_spline_motion_prior=bool(args.enable_ct_spline_motion_prior),
        ct_spline_smoothing_m=args.ct_spline_smoothing_m,
        ct_motion_sigma_m=args.ct_motion_sigma_m,
        ct_motion_min_epochs=args.ct_motion_min_epochs,
        fgo_min_correction_m=args.fgo_min_correction_m,
        fgo_apply_fixed_epochs_only=not bool(args.fgo_apply_window_all),
        tdcp_sigma_mps=args.tdcp_sigma_mps,
        tdcp_postfit_max_m=args.tdcp_postfit_max_m,
        tdcp_min_sats=args.tdcp_min_sats,
        tdcp_obs_anchor_sigma_m=args.tdcp_obs_anchor_sigma_m,
        tdcp_obs_loose_sigma_m=args.tdcp_obs_loose_sigma_m,
        tdcp_obs_missing_sigma_m=args.tdcp_obs_missing_sigma_m,
        low_sat_bridge_min_pr_sats=args.low_sat_bridge_min_pr_sats,
        low_sat_bridge_min_span_epochs=args.low_sat_bridge_min_span_epochs,
        low_sat_bridge_max_span_epochs=args.low_sat_bridge_max_span_epochs,
        low_sat_bridge_max_gap_s=args.low_sat_bridge_max_gap_s,
        low_sat_bridge_startup_max_wls_pdop=args.low_sat_bridge_startup_max_wls_pdop,
        low_sat_bridge_startup_min_epochs=args.low_sat_bridge_startup_min_epochs,
        low_sat_bridge_startup_max_epochs=args.low_sat_bridge_startup_max_epochs,
        zupt_acc_norm_low_mps2=args.zupt_acc_norm_low_mps2,
        zupt_acc_norm_high_mps2=args.zupt_acc_norm_high_mps2,
        zupt_gyro_norm_max_dps=args.zupt_gyro_norm_max_dps,
        zupt_apply_hybrid_statuses=tuple(
            int(s.strip()) for s in args.zupt_apply_hybrid_statuses.split(",") if s.strip()
        ),
        zupt_min_consecutive=args.zupt_min_consecutive,
        zupt_max_anchor_drift_m=args.zupt_max_anchor_drift_m,
        imu_tc_emit_pf_hybrid_statuses=tuple(
            int(s.strip()) for s in args.imu_tc_emit_pf_hybrid_statuses.split(",") if s.strip()
        ),
        imu_tc_pos_sigma_base_m=args.imu_tc_pos_sigma_base_m,
        imu_tc_pos_sigma_per_s=args.imu_tc_pos_sigma_per_s,
        imu_tc_max_dr_seconds=args.imu_tc_max_dr_seconds,
        imu_tc_max_disagreement_m=args.imu_tc_max_disagreement_m,
        imu_tc_emit_max_diff_m=args.imu_tc_emit_max_diff_m,
        imu_tc_hybrid_loose_sigma_m=args.imu_tc_hybrid_loose_sigma_m,
        imu_tc_anchor_static_acc_low_mps2=args.zupt_acc_norm_low_mps2,
        imu_tc_anchor_static_acc_high_mps2=args.zupt_acc_norm_high_mps2,
        imu_tc_anchor_static_gyro_max_dps=args.zupt_gyro_norm_max_dps,
        ins_tc_emit_pf_hybrid_statuses=tuple(
            int(s.strip()) for s in args.ins_tc_emit_pf_hybrid_statuses.split(",") if s.strip()
        ),
        ins_tc_obs_status_4_sigma_m=args.ins_tc_obs_status_4_sigma_m,
        ins_tc_obs_status_3_sigma_m=args.ins_tc_obs_status_3_sigma_m,
        ins_tc_max_dr_seconds=args.ins_tc_max_dr_seconds,
        ins_tc_max_disagreement_m=args.ins_tc_max_disagreement_m,
        ins_tc_emit_max_diff_m=args.ins_tc_emit_max_diff_m,
        ins_tc_pf_pu_floor_sigma_m=args.ins_tc_pf_pu_floor_sigma_m,
        ins_tc_pf_pu_ceiling_sigma_m=args.ins_tc_pf_pu_ceiling_sigma_m,
        ins_tc_use_particle_imu_predict=not args.ins_tc_disable_particle_imu_predict,
        ins_tc_particle_imu_sigma_pos_m=args.ins_tc_particle_imu_sigma_pos_m,
        ins_tc_particle_imu_sigma_acc_mps2=args.ins_tc_particle_imu_sigma_acc_mps2,
        ins_tc_particle_imu_sigma_gyro_rps=args.ins_tc_particle_imu_sigma_gyro_rps,
        ins_tc_particle_imu_acc_bias_rw=args.ins_tc_particle_imu_acc_bias_rw,
        ins_tc_particle_imu_gyro_bias_rw=args.ins_tc_particle_imu_gyro_bias_rw,
        ins_tc_particle_imu_att_spread_rad=args.ins_tc_particle_imu_att_spread_rad,
        ins_tc_particle_imu_acc_bias_spread=args.ins_tc_particle_imu_acc_bias_spread,
        ins_tc_particle_imu_gyro_bias_spread_rps=args.ins_tc_particle_imu_gyro_bias_spread_rps,
        ins_tc_particle_imu_velocity_spread_mps=args.ins_tc_particle_imu_velocity_spread_mps,
        ins_tc_recenter_status4=bool(args.ins_tc_enable_recenter_status4),
        ins_tc_recenter_max_shift_m=args.ins_tc_recenter_max_shift_m,
        ins_tc_use_motion_predict=not args.ins_tc_disable_motion_predict,
        ins_tc_predict_sigma_pos_m=args.ins_tc_predict_sigma_pos_m,
        ins_tc_predict_velocity_alpha=args.ins_tc_predict_velocity_alpha,
        ins_tc_align_acc_low=args.ins_tc_align_acc_low,
        ins_tc_align_acc_high=args.ins_tc_align_acc_high,
        ins_tc_align_gyro_max_dps=args.ins_tc_align_gyro_max_dps,
        ins_tc_align_min_samples=args.ins_tc_align_min_samples,
        ins_tc_yaw_init_min_speed_mps=args.ins_tc_yaw_init_min_speed_mps,
        ins_tc_quality_gate_enabled=bool(args.ins_tc_quality_gate_enabled),
        ins_tc_quality_gate_window_epochs=int(args.ins_tc_quality_gate_window_epochs),
        ins_tc_quality_gate_max_fix_rate=float(args.ins_tc_quality_gate_max_fix_rate),
        ins_tc_quality_gate_pu_skip=bool(args.ins_tc_quality_gate_pu_skip),
        # NOTE: rbpf_kf_gate_* defaults stay None in `base` so that bare
        # `rbpf` / `rbpf+dd` variants run without a gate (true baseline).
        # Only the `rbpf+dd+gate*` variants below opt in via `aaa_gate`.
        # `enable_hybrid_pu` also stays False here; only `*+hybrid` opts in.
        systems=tuple(args.systems.split(",")),
    )
    if "pf" in args.methods:
        variants.append(CTRBPFConfig(**base, method_label="PF-PR"))
    if "pf+pu" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_position_update=True,
            method_label="PF-PR+PU",
        ))
    if "rbpf" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            method_label="RBPF-velKF",
        ))
    if "rbpf+pu" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_position_update=True,
            method_label="RBPF-velKF+PU",
        ))
    if "pf+dd" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_dd_carrier_afv=True,
            method_label="PF-PR+DD",
        ))
    if "rbpf+dd" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            method_label="RBPF-velKF+DD",
        ))
    if "rbpf+dd+pu" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            method_label="RBPF-velKF+DD+PU",
        ))
    if "rbpf+dd+pu+tdcp" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+PU+TDCP",
        ))
    if "rbpf+dd+pu+bridge" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            enable_low_sat_bridge=True,
            method_label="RBPF-velKF+DD+PU+bridge",
        ))
    # Phase 2 variants: same as rbpf+dd / rbpf+dd+pu but force the
    # AAA-style region-aware gate defaults so a single CLI flag is enough
    # to opt in. Per-knob CLI overrides still apply via args.*.
    aaa_gate = dict(
        rbpf_kf_gate_min_dd_pairs=(
            args.rbpf_velocity_kf_gate_min_dd_pairs
            if args.rbpf_velocity_kf_gate_min_dd_pairs is not None
            else 15
        ),
        rbpf_kf_gate_min_ess_ratio=(
            args.rbpf_velocity_kf_gate_min_ess_ratio
            if args.rbpf_velocity_kf_gate_min_ess_ratio is not None
            else 0.02
        ),
        rbpf_kf_gate_max_spread_m=args.rbpf_velocity_kf_gate_max_spread_m,
    )
    if "rbpf+dd+gate" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            method_label="RBPF-velKF+DD+gate",
        ))
    if "rbpf+dd+gate+pu" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            method_label="RBPF-velKF+DD+gate+PU",
        ))
    if "rbpf+dd+gate+pu+tdcp" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+gate+PU+TDCP",
        ))
    if "rbpf+dd+gate+pu+bridge" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            enable_low_sat_bridge=True,
            method_label="RBPF-velKF+DD+gate+PU+bridge",
        ))
    dopq_gate = dict(
        rbpf_kf_gate_min_dd_pairs=None,
        rbpf_kf_gate_min_ess_ratio=None,
        rbpf_kf_gate_max_spread_m=args.rbpf_velocity_kf_gate_max_spread_m,
        rbpf_kf_gate_max_doppler_wls_rms_mps=(
            args.rbpf_velocity_kf_gate_max_doppler_wls_rms_mps
            if args.rbpf_velocity_kf_gate_max_doppler_wls_rms_mps > 0.0
            else 1.0
        ),
        rbpf_kf_gate_max_doppler_wls_speed_mps=(
            args.rbpf_velocity_kf_gate_max_doppler_wls_speed_mps
            if args.rbpf_velocity_kf_gate_max_doppler_wls_speed_mps > 0.0
            else 20.0
        ),
    )
    if "rbpf+dd+dopq+pu" in args.methods:
        variant_kwargs = {**base, **dopq_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            method_label="RBPF-velKF+DD+dopq+PU",
        ))
    if "rbpf+dd+dopq+pu+bridge" in args.methods:
        variant_kwargs = {**base, **dopq_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            enable_low_sat_bridge=True,
            method_label="RBPF-velKF+DD+dopq+PU+bridge",
        ))
    if "rbpf+dd+gate+pu+ddpr" in args.methods:
        variant_kwargs = {
            **base,
            **aaa_gate,
            "enable_dd_pr_ls_anchor": True,
            "dd_pr_ls_anchor_mode": "pf-pu",
            "dd_pr_ls_anchor_statuses": (),
        }
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            method_label="RBPF-velKF+DD+gate+PU+DDPR",
        ))
    if "rbpf+dd+gate+pu+bridge+ddpr" in args.methods:
        variant_kwargs = {
            **base,
            **aaa_gate,
            "enable_dd_pr_ls_anchor": True,
            "dd_pr_ls_anchor_mode": "pf-pu",
            "dd_pr_ls_anchor_statuses": (),
        }
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_position_update=True,
            enable_low_sat_bridge=True,
            method_label="RBPF-velKF+DD+gate+PU+bridge+DDPR",
        ))
    if "rbpf+dd+gate+rtkdiag_pf" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_rtkdiag_pf_rescue=True,
            method_label="RBPF-velKF+DD+gate+rtkdiag_pf",
        ))
    if "rbpf+dd+gate+rtkdiag_pf+bridge" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_rtkdiag_pf_rescue=True,
            enable_low_sat_bridge=True,
            method_label="RBPF-velKF+DD+gate+rtkdiag_pf+bridge",
        ))
    # Phase 6 variants: layer hybrid (libgnss++ 50.91% baseline) PU on top.
    if "pf+hybrid" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_hybrid_pu=True,
            method_label="PF-PR+hybrid",
        ))
    if "rbpf+dd+hybrid" in args.methods:
        variants.append(CTRBPFConfig(
            **base,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            method_label="RBPF-velKF+DD+hybrid",
        ))
    if "rbpf+dd+gate+hybrid" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            method_label="RBPF-velKF+DD+gate+hybrid",
        ))
    # Phase 10a: switch Status=1/3 pseudorange likelihood from Gaussian to
    # LOS/NLOS positive-bias GMM, while keeping Status=4 on the sharp model.
    if "rbpf+dd+gate+hybrid+gmm" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_pr_gmm=True,
            method_label="RBPF-velKF+DD+gate+hybrid+gmm",
        ))
    # Phase 10i: keep v5 hybrid as the floor, but inject the relaxed RTK
    # candidate into the PF on diagnostics-passing epochs and emit PF only
    # there. This is the PF version of the Phase 10h dual-profile chooser.
    if "rbpf+dd+gate+hybrid+rtkdiag_pf" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_rtkdiag_pf_rescue=True,
            method_label="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf",
        ))
    # Phase 19ba: rtkdiag_pf + TDCP smoother. Cluster-vote / K=3 prefilter
    # picks the per-epoch candidate; TDCP fwd+bwd Kalman smooths the emitted
    # trajectory using carrier-phase delta velocity. Designed to interpolate
    # deep-canyon spans (1-5min between fix=4 anchors) where no candidate
    # is within 50cm but adjacent anchors are.
    if "rbpf+dd+gate+hybrid+rtkdiag_pf+tdcp" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_rtkdiag_pf_rescue=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf+tdcp",
        ))
    # Phase 11cq: rtkdiag_pf + post-process FGO (with bug #3 fixed in motion-delta).
    if "rbpf+dd+gate+hybrid+rtkdiag_pf+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_rtkdiag_pf_rescue=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf+phase4",
        ))
    # Phase 11ex: rtkdiag_pf + imu_tc combo (TURING gici-open architectural ref:
    # rtk_imu_tc estimator). rtkdiag_pf still drives candidate selection; imu_tc
    # adds per-particle IMU pre-integration and Status=1/3 PF emission. Goal:
    # IMU helps prune bad candidates by pulling particles toward IMU-predicted
    # trajectory, so wrong candidate PU has less effect on final estimate.
    if "rbpf+dd+gate+hybrid+rtkdiag_pf+imu_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_rtkdiag_pf_rescue=True,
            enable_imu_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf+imu_tc",
        ))
    # Phase 11ex variant: rtkdiag_pf + ins_tc combo (15-state INS+GNSS EKF).
    if "rbpf+dd+gate+hybrid+rtkdiag_pf+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_rtkdiag_pf_rescue=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+rtkdiag_pf+ins_tc",
        ))
    # Phase 7: full stack (vguide + hybrid PU + DD AFV + RBPF gate) emitting
    # the PF's own estimate. Goal: show DD-AFV cm-correction beat the hybrid
    # baseline floor (Phase 6 passthrough).
    if "rbpf+dd+gate+phase7" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            hybrid_emit_pf_estimate=True,
            method_label="RBPF-velKF+DD+gate+phase7",
        ))
    if "rbpf+dd+gate+phase7+gmm" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            hybrid_emit_pf_estimate=True,
            enable_pr_gmm=True,
            method_label="RBPF-velKF+DD+gate+phase7+gmm",
        ))
    # Phase 9a: hybrid passthrough + ZUPT (IMU stop detection).
    if "rbpf+dd+gate+hybrid+zupt" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_zupt=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt",
        ))
    # Phase 9b: tight-coupled IMU (in-loop pre-integration; per-particle
    # position pseudo-observation; emission switches to PF estimate on
    # Status=1/3 epochs). Requires hybrid for the Status=4 anchor reset.
    if "rbpf+dd+gate+hybrid+imu_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_imu_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+imu_tc",
        ))
    # Phase 9a + Phase 9b stacked. ZUPT damps stops, IMU-TC handles motion
    # NLOS rescue. Same anchor IMU-static thresholds for both.
    if "rbpf+dd+gate+hybrid+zupt+imu_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_zupt=True,
            enable_imu_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt+imu_tc",
        ))
    # Phase 9c: full INS-GNSS EKF (15-state; online accel/gyro bias).
    if "rbpf+dd+gate+hybrid+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+ins_tc",
        ))
    # Phase 10a stacked with per-particle INS/IMU propagation.
    if "rbpf+dd+gate+hybrid+gmm+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_pr_gmm=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+gmm+ins_tc",
        ))
    # Phase 9a + Phase 9c stacked.
    if "rbpf+dd+gate+hybrid+zupt+ins_tc" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_zupt=True,
            enable_ins_tc=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt+ins_tc",
        ))
    # Phase 9a + Phase 8 stacked.
    if "rbpf+dd+gate+hybrid+zupt+tdcp" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_zupt=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+gate+hybrid+zupt+tdcp",
        ))
    # Phase 8: hybrid passthrough + TDCP-anchored Kalman smoother.
    if "rbpf+dd+gate+hybrid+tdcp" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_tdcp_smoother=True,
            method_label="RBPF-velKF+DD+gate+hybrid+tdcp",
        ))
    # Phase 8 + Phase 4 stacked.
    if "rbpf+dd+gate+hybrid+tdcp+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_tdcp_smoother=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+hybrid+tdcp+phase4",
        ))
    # Phase 4: hybrid passthrough + post-process FGO + LAMBDA partial fix.
    # The PF supplies cached DD carrier observations (no hybrid bias on the
    # FGO solve), and where LAMBDA accepts integer fixes we replace the
    # hybrid passthrough with the cm-pulled FGO trajectory.
    if "rbpf+dd+gate+hybrid+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+hybrid+phase4",
        ))
    # Phase 4 without hybrid: pure PF + FGO + LAMBDA. Useful to see whether
    # LAMBDA alone (independent of hybrid) is enough to cm-pull the
    # trajectory.
    if "rbpf+dd+gate+phase4" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_fgo_lambda=True,
            method_label="RBPF-velKF+DD+gate+phase4",
        ))
    # Embedded target: one RTK/hybrid anchor stream, CT/RBPF updates, fixed-lag
    # FGO/LAMBDA, and conservative single-stream emission. No RTKDiag
    # candidate selector.
    if "embedded" in args.methods or "embedded_ctpf_fgo" in args.methods:
        variant_kwargs = {**base, **aaa_gate, "fgo_lambda_min_epochs": 3}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_fgo_lambda=True,
            method_label="EMBEDDED-RTK-CTPF-FGO",
        ))
    if "embedded_pfemit" in args.methods:
        variant_kwargs = {**base, **aaa_gate, "fgo_lambda_min_epochs": 3}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            hybrid_emit_pf_estimate=True,
            enable_fgo_lambda=True,
            method_label="EMBEDDED-CTPF-FGO-PFEMIT",
        ))
    if "gpu_ctpf_fgo" in args.methods:
        variant_kwargs = {
            **base,
            **aaa_gate,
            "enable_ct_spline_motion_prior": True,
            "fgo_lambda_min_epochs": 3,
            # PPC base/rover common-sat coverage is often below the old
            # AAA-style 15-DD-pair gate, which disabled Doppler KF for whole
            # runs. Keep the gate active for diagnostics, but do not require
            # DD pairs before applying Doppler.
            "rbpf_kf_gate_min_dd_pairs": (
                args.rbpf_velocity_kf_gate_min_dd_pairs
                if args.rbpf_velocity_kf_gate_min_dd_pairs is not None
                else 0
            ),
            "hybrid_recenter_max_shift_m": (
                10000.0
                if float(base.get("hybrid_recenter_max_shift_m", 0.0)) <= 0.0
                else float(base["hybrid_recenter_max_shift_m"])
            ),
        }
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            enable_fgo_lambda=True,
            method_label="GPU-CTPF-FGO",
        ))
    if "gpu_ctpf_fgo_pfemit" in args.methods:
        variant_kwargs = {
            **base,
            **aaa_gate,
            "enable_ct_spline_motion_prior": True,
            "fgo_lambda_min_epochs": 3,
            "hybrid_emit_pf_statuses": (
                tuple(
                    int(s.strip()) for s in args.hybrid_emit_pf_statuses.split(",") if s.strip()
                )
                if args.hybrid_emit_pf_statuses.strip()
                else (1, 3)
            ),
            "rbpf_kf_gate_min_dd_pairs": (
                args.rbpf_velocity_kf_gate_min_dd_pairs
                if args.rbpf_velocity_kf_gate_min_dd_pairs is not None
                else 0
            ),
            "hybrid_recenter_max_shift_m": (
                10000.0
                if float(base.get("hybrid_recenter_max_shift_m", 0.0)) <= 0.0
                else float(base["hybrid_recenter_max_shift_m"])
            ),
        }
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=True,
            enable_hybrid_pu=True,
            enable_hybrid_velocity_guide=True,
            hybrid_emit_pf_estimate=True,
            enable_fgo_lambda=True,
            method_label="GPU-CTPF-FGO-PFEMIT",
        ))
    if not variants:
        raise ValueError(f"no valid methods: {args.methods}")
    return variants


_RTKDIAG_POLICIES = {
    "phase10o", "phase10p", "phase10r",
    "phase11h", "phase11i", "phase11l", "phase11n",
    "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di", "phase11dk", "phase11dl", "phase11dm", "phase11dn", "phase11do", "phase11dp", "phase11dq", "phase11dr", "phase11ds", "phase11dt", "phase11du", "phase11dv", "phase11dw", "phase11dx", "phase11dy", "phase11dz", "phase11ea", "phase11eb", "phase11ec", "phase11ed", "phase11ee", "phase11ef", "phase11eg", "phase11eh", "phase11ei", "phase11ej", "phase11ek", "phase11el", "phase11em", "phase11en", "phase11eo", "phase11ep", "phase11eq", "phase11er", "phase11er_mrescue", "phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue",
}


_NAGOYA_RUN2_PHASE11EQ_LABELS = {
    "fgo_v14_snr38",
    "full_ratio15_lock3_trustedseed_rtkout3oGem3",
    "dev_demo5_trusted_o3",
    "n2_nobds",
    "fgo_v1",
    "full_ratio15_lock3_trustedseed_rtkout3mlc1",
    "full_ratio15_lock3_trustedseed_rtkout5",
}

_NAGOYA_RUN2_PHASE11ER_EXT_LABELS = {
    *_NAGOYA_RUN2_PHASE11EQ_LABELS,
    "libgnss_ext_subset",
}


def _apply_rtkdiag_run_index_policy(
    variant: CTRBPFConfig,
    *,
    run: str,
    policy: str,
    city: str | None = None,
) -> CTRBPFConfig:
    if (
        policy not in _RTKDIAG_POLICIES
        or not bool(variant.enable_rtkdiag_pf_rescue)
    ):
        return variant
    if policy in {"phase11eq", "phase11er", "phase11er_mrescue", "phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"}:
        if city == "nagoya" and run == "run2":
            max_to_hybrid_m = 60.0 if policy in {"phase11er_mrescue", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"} else 10.0
            float_labels = ("libgnss_ext_subset",) if policy in {"phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"} else ()
            status5_labels = ("fgo_v14_snr38",) if policy == "phase11er_ext_s5_mrescue" else ()
            return replace(
                variant,
                rtkdiag_candidate_select_mode="residual",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_max_to_hybrid_m=max_to_hybrid_m,
                rtkdiag_candidate_emit_mode="candidate",
                rtkdiag_candidate_fallback_mode="hybrid",
                rtkdiag_candidate_float_labels=float_labels,
                rtkdiag_candidate_float_residual_rms_max=1.0 if float_labels else 0.0,
                rtkdiag_candidate_float_abs_max=3.0 if float_labels else 0.0,
                rtkdiag_candidate_float_min_sats=8 if float_labels else 0,
                rtkdiag_candidate_status5_labels=status5_labels,
                rtkdiag_candidate_status5_tow_windows=(
                    (556116.0, 556122.4, status5_labels),
                ) if status5_labels else (),
                rtkdiag_candidate_status5_max_dt_s=0.2 if status5_labels else 0.0,
                rtkdiag_candidate_status5_residual_rms_max=1.0 if status5_labels else 0.0,
                rtkdiag_candidate_status5_min_sats=3 if status5_labels else 0,
            )
        return replace(
            variant,
            rtkdiag_candidate_emit_mode="candidate",
            rtkdiag_candidate_fallback_mode="hybrid",
        )
    if policy == "phase11ep":
        if city == "tokyo" and run == "run1":
            variant = replace(
                variant,
                rtkdiag_candidate_local_ungate_tow_windows=(
                    (188028.6, 188054.0, ()),
                    (188131.8, 188137.4, ()),
                    (188151.6, 188164.4, ()),
                    (188259.0, 188268.2, ()),
                    (188403.0, 188417.4, ()),
                    (188432.2, 188443.2, ()),
                    (188554.6, 188560.6, ()),
                    (189207.2, 189216.6, ()),
                ),
            )
        elif city == "tokyo" and run == "run2":
            variant = replace(
                variant,
                rtkdiag_candidate_local_ungate_tow_windows=((178248.4, 178255.4, ()),),
            )
        elif city == "nagoya" and run == "run1":
            variant = replace(
                variant,
                rtkdiag_candidate_local_ungate_tow_windows=(
                    (551063.2, 551076.2, ()),
                    (551106.6, 551113.8, ()),
                    (551315.0, 551349.8, ()),
                ),
            )
        elif city == "nagoya" and run == "run2":
            variant = replace(
                variant,
                rtkdiag_candidate_label_factors=(
                    (
                        "xd_fixedicb_raw_n2_icbsweep_5734_5773_l1p8_l2p0",
                        0.8,
                    ),
                ),
            )
        policy = "phase11eo"
    if policy == "phase11eo":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v8",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v10",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11en"
    if policy == "phase11en":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v7",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v9",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v6",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11em"
    if policy == "phase11em":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v8",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11el"
    if policy == "phase11el":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v7",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ek"
    if policy == "phase11ek":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v6",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v5",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ej"
    if policy == "phase11ej":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v6",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v5",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v4",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ei"
    if policy == "phase11ei":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v5",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11eh"
    if policy == "phase11eh":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v4",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v3",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11eg"
    if policy == "phase11eg":
        policy = "phase11ef"
    if policy == "phase11ef":
        policy = "phase11ee"
    if policy == "phase11ee":
        policy = "phase11ed"
    if policy == "phase11ed":
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="composite_t2_v3",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ec"
    if policy == "phase11ec":
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v2",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11eb"
    if policy == "phase11eb":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v4",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11ea"
    if policy == "phase11ea":
        policy = "phase11dz"
    if policy == "phase11dz":
        policy = "phase11dy"
    if policy == "phase11dy":
        policy = "phase11dx"
    if policy == "phase11dx":
        policy = "phase11dw"
    if policy == "phase11dw":
        policy = "phase11dv"
    if policy == "phase11dv":
        policy = "phase11du"
    if policy == "phase11du":
        policy = "phase11dt"
    if policy == "phase11dt":
        policy = "phase11ds"
    if policy == "phase11ds":
        policy = "phase11dr"
    if policy == "phase11dr":
        policy = "phase11dq"
    if policy == "phase11dq":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v3",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v3",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11dp"
    if policy == "phase11dp":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v2",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v2",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11do"
    if policy == "phase11do":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_t3_v1",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n2_v1",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_hybdelta_n3_v1",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11dn"
    if policy == "phase11dn":
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="temporal_n2_v1",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11dm"
    if policy == "phase11dm":
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="composite_t3_v2",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="composite_n2_v4",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        policy = "phase11di"
    if policy in {"phase11dk", "phase11dl"}:
        # Phase 11dk/dl keep Phase 11di selector/gate settings and only change
        # the per-run candidate allow-list in _filter_rtkdiag_candidates_by_policy.
        policy = "phase11di"
    if policy in {"phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        # Phase 11aa: extends Phase 11z by switching tokyo/run1 selector
        # to hybrid_anchor (consensus selector showed +0.20pp on this run /
        # +0.05pp aggregate). Other runs unchanged from Phase 11z.
        # Phase 11ab: same gates as 11aa, but r15ga (--glonass-ar autocal,
        # ratio 1.5) is added to the candidate pool — restricted via
        # blocked_labels to {(tokyo, run3), (nagoya, run1)} where offline
        # simulation showed positive aggregate (+88m / +0.19pp).
        # Phase 11ac: extends 11ab with two more selectively-eligible
        # candidates: r20ga (ratio2.0 + glonass-ar) for {(tokyo, run3),
        # (nagoya, run2)} and em10 (--elevation-mask-deg 10) for {(tokyo,
        # run3)}. Combined offline gain over 11ab is +81m / +0.18pp.
        # Phase 11ad: extends 11ac with two more candidates restricted to
        # {(tokyo, run3)}: psig1 (--pseudorange-sigma 1.0) and holdrlx
        # (--min-hold-count 3 --hold-ratio-threshold 1.5). Offline shows
        # +146m on tokyo/run3 alone (additive).
        if city == "tokyo" and run == "run1":
            # Phase 11bw: switch t/r1 to residual mode (selector sweep predicted +164m
            # vs score in small candidate pool; check if PF realises the gain).
            if policy in {"phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=2.5,
                    rtkdiag_candidate_residual_rms_max=1.4,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11bu: switch t/r1 to score mode (confirmed +3.34pp / +344m on t/r1 alone)
            # — hybrid_anchor was underusing high-precision combos.
            if policy in {"phase11bu", "phase11bv"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=2.5,
                    rtkdiag_candidate_residual_rms_max=1.4,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="hybrid_anchor",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            # Phase 11dc/dd: composite_t2_v2 = residual / (ratio^0.2 * rows^2.0 * abs_max^0.5).
            # Fine sim BEST = 84.86% vs db 84.78 (+0.08pp). dd: ultra-fine confirmed same optimum.
            if policy in {"phase11dc", "phase11dd", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_t2_v2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11db: composite_3axis_t2 = residual / (ratio^0.5 * rows^2.0).
            # 3-axis sim BEST = 84.78% vs score_per_row 84.53 (+0.25pp).
            if policy == "phase11db":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_t2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cv: switch t/r2 score → score_per_row.
            if policy in {"phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cc: try t/r2 score → residual (mirror t/r1 pattern).
            if policy == "phase11cc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=10.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            # Phase 11cu/cv: switch t/r3 score → score_per_row (sweep +0.12pp / +19.6m).
            if policy in {"phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cb: tighten ratio_min 1.0 → 1.7 for t/r3 score (rms_max kept 50).
            if policy == "phase11cb":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11ca: tighten rms_max 50 → 5 for t/r3 score (regressed -1.40pp).
            if policy == "phase11ca":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=5.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            # Phase 11dd: composite_n1_v3 = residual / (rows^0.7 * abs_max^0.3).
            # Ultra-fine sim BEST = 64.33% vs dc 64.31 (+0.02pp).
            if policy in {"phase11dd", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n1_v3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dc: composite_n1_v2 = residual / (rows^0.5 * abs_max^0.3).
            # Fine sim BEST = 64.31% vs db 64.25 (+0.06pp).
            if policy == "phase11dc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n1_v2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11db: composite_3axis_n1 = residual / (rows^0.5 * abs_max^0.5) (no ratio).
            # 3-axis sim BEST = 64.25% vs rms_per_row 63.89 (+0.37pp).
            if policy == "phase11db":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_n1",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cv: switch n/r1 nrows → rms_per_row (sweep +0.14pp / +6.2m).
            if policy in {"phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="rms_per_row",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11bz: try n/r1 nrows → ratio (max ratio = highest ambiguity confidence).
            if policy == "phase11bz":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="ratio",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11by: try n/r1 nrows → residual (mirror t/r1 success).
            # Tight rms_max=1.0 → residual mode should pick the truly-best rms candidate.
            if policy == "phase11by":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11bv: try switching n/r1 from nrows to score (mirror t/r1 success).
            if policy == "phase11bv":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=1.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            # Phase 11dh: dd composite_n2_v3 + emit_mode="pf" + recenter=2.0m.
            # Test: if candidate > 2m from PF, skip recenter, drift_skip → emit hybrid.
            # Goal: filter outlier candidates by hybrid agreement.
            if policy == "phase11dh":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n2_v3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="pf",
                    rtkdiag_candidate_recenter_max_shift_m=2.0,
                )
            # Phase 11dd: composite_n2_v3 = residual / (ratio^0.3 * rows^0.7 * abs_max^0.8).
            # Ultra-fine sim BEST = 40.21% vs dc 40.14 (+0.07pp).
            if policy in {"phase11dd", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n2_v3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dc: composite_n2_v2 = residual / (ratio^0.4 * rows^1.0 * abs_max^0.7).
            # Fine sim BEST = 40.14% vs db 39.92 (+0.22pp).
            if policy == "phase11dc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n2_v2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11da/db: composite_3axis_n2 = residual / (ratio^0.5 * rows^1.5 * abs_max^0.5).
            # 3-axis sim BEST = 39.92% vs cy 39.18% (+0.74pp).
            if policy in {"phase11da", "phase11db"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_n2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cv/cz: switch n/r2 score → score_per_row2 (alpha sweep b=2、sim 39.43% vs 39.18%、+0.25pp).
            if policy in {"phase11cv", "phase11cz"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row2",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cy: switch n/r2 score → score_per_row (filter fix 後 PF +0.06pp 確認).
            if policy in {"phase11cu", "phase11cy"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cp: try ratio mode on n/r2 (untested).
            if policy == "phase11cp":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="ratio",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11co: try consensus5 (median-anchored) on n/r2.
            if policy == "phase11co":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="consensus5",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cn: try wavg3 fusion mode on n/r2 (top 3 candidates weighted avg).
            if policy == "phase11cn":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="wavg3",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cj: relax sigma_m 0.02 → 1.0 on n/r2 (PF takes candidate weakly).
            if policy == "phase11cj":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                    rtkdiag_candidate_sigma_m=1.0,
                )
            # Phase 11ci: relax emit_max_diff_m 0.4 → 2.0 on n/r2 (no effect).
            if policy == "phase11ci":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                    rtkdiag_candidate_emit_max_diff_m=2.0,
                )
            # Phase 11bx: try n/r2 score → residual (regressed -4.49pp).
            if policy == "phase11bx":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.0,
                    rtkdiag_candidate_residual_rms_max=50.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            # Phase 11cv: switch n/r3 score → score_per_row3 (alpha sweep +1.06pp at b=3).
            if policy == "phase11cv":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row3",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dd: n/r3 composite_n3_v3 = residual / (ratio^0.2 * rows^0.7 * abs_max^0.5).
            # Ultra-fine sim BEST = 59.15% vs dc 59.10 (+0.05pp).
            if policy in {"phase11dd", "phase11dh", "phase11di"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n3_v3",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11dc: n/r3 composite_n3_v2 = residual / (ratio^0.2 * rows^0.5 * abs_max^0.5).
            # Fine sim BEST = 59.10% vs db 59.04 (+0.05pp).
            if policy == "phase11dc":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_n3_v2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11da/db: n/r3 composite_3axis_n2 (3-axis sim 59.04% vs cy 58.85%、+0.19pp).
            if policy in {"phase11da", "phase11db"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="composite_3axis_n2",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cx/cy: n/r3 score → score_per_row3 (alpha sweep +1.06pp at b=3、PF +0.25pp 確認).
            if policy in {"phase11cx", "phase11cy", "phase11cz"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row3",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cu/cw: switch n/r3 score → score_per_row (sweep +0.81pp / +27m).
            if policy in {"phase11cu", "phase11cw"}:
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="score_per_row",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            # Phase 11cd: try n/r3 score → residual (last selector permutation).
            if policy == "phase11cd":
                return replace(
                    variant,
                    rtkdiag_candidate_select_mode="residual",
                    rtkdiag_candidate_ratio_min=1.7,
                    rtkdiag_candidate_residual_rms_max=30.0,
                    rtkdiag_candidate_emit_mode="candidate",
                )
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11z logic if city unknown.
    if policy == "phase11z":
        # Phase 11z: extends Phase 11y by pushing rms_max to 50 on
        # tokyo/run3 + nagoya/run2 and rms_max=30 on nagoya/run3 (extended2
        # gate sweep showed plateau between rms=50 and rms=100, with most
        # gain on nagoya/run2 +0.79pp).
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=50.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=30.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11y/phase11x logic if city unknown.
    if policy == "phase11y":
        # Phase 11y: extends Phase 11x by widening rms_max to 20 on the
        # heavy-NLOS runs (tokyo/run3, nagoya/run2, nagoya/run3) and
        # dropping ratio_min to 1.0 where the extended gate sweep showed
        # selector improvement. Other runs unchanged from Phase 11x.
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.0,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=20.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through to phase11x logic if city unknown.
    if policy == "phase11x":
        # Phase 11x: city x run gate from offline gate sweep on Phase 11v
        # candidate pool. Selector mode is unchanged from phase11n.
        if city == "tokyo" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=2.5,
                rtkdiag_candidate_residual_rms_max=1.4,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "tokyo" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.3,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run1":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="nrows",
                rtkdiag_candidate_ratio_min=1.5,
                rtkdiag_candidate_residual_rms_max=1.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run2":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.5,
                rtkdiag_candidate_residual_rms_max=7.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        if city == "nagoya" and run == "run3":
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=10.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        # Fall through: city unknown, behave like phase11n.
    if run == "run1":
        return replace(
            variant,
            rtkdiag_candidate_select_mode="nrows",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=1.4,
            rtkdiag_candidate_emit_mode="candidate",
        )
    if run == "run2":
        if policy in {"phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa"}:
            return replace(
                variant,
                rtkdiag_candidate_select_mode="score",
                rtkdiag_candidate_ratio_min=1.7,
                rtkdiag_candidate_residual_rms_max=6.0,
                rtkdiag_candidate_emit_mode="candidate",
            )
        return replace(
            variant,
            rtkdiag_candidate_select_mode="maxabs",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=6.0 if policy == "phase10p" else 5.0,
            rtkdiag_candidate_emit_mode="candidate",
        )
    if run == "run3":
        return replace(
            variant,
            rtkdiag_candidate_select_mode="score",
            rtkdiag_candidate_ratio_min=1.5,
            rtkdiag_candidate_residual_rms_max=7.0 if policy in {"phase10r", "phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa"} else (
                6.0 if policy == "phase10p" else 5.0
            ),
            rtkdiag_candidate_emit_mode="candidate",
        )
    return variant


def _filter_rtkdiag_candidates_by_policy(
    candidates: list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]],
    *,
    city: str,
    run: str,
    policy: str,
    blocked_labels: set[str] | None = None,
) -> list[tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]]:
    if policy in {"phase11eq", "phase11er", "phase11er_mrescue", "phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"}:
        # Phase 11er is the conservative n/r2 rescue found from the
        # lowcase audit: residual-select across a small candidate stack,
        # candidate emit, hybrid fallback, and a 10 m hybrid-distance gate.
        # phase11er_mrescue keeps the same whitelist but widens the gate to
        # 60 m in _apply_rtkdiag_run_index_policy for 3 m outlier reduction.
        # phase11er_ext variants add one libgnss++ extended/subset AR candidate.
        # phase11er_ext_float_mrescue also allows tightly gated FLOAT output
        # from that extended candidate only. phase11er_ext_s5_mrescue adds a
        # narrow, audited fgo_v14 status-5 bridge over the n/r2 tail gap.
        # On every other run, block all candidates so the rtkdiag variant is a
        # hybrid passthrough and cannot regress unrelated runs.
        if city == "nagoya" and run == "run2":
            allowed = (
                _NAGOYA_RUN2_PHASE11ER_EXT_LABELS
                if policy in {"phase11er_ext", "phase11er_ext_mrescue", "phase11er_ext_float_mrescue", "phase11er_ext_s5_mrescue"}
                else _NAGOYA_RUN2_PHASE11EQ_LABELS
            )
            extra_blocked = set(blocked_labels or set())
            extra_blocked.update(
                label
                for label, _, _ in candidates
                if label not in allowed
            )
            return _filter_rtkdiag_candidates_by_policy(
                candidates,
                city=city,
                run=run,
                policy="phase11ep",
                blocked_labels=extra_blocked,
            )
        return []
    if policy == "phase11eo":
        # Phase 11eo changes only t/r3 and n/r2 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11en.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11en",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11en":
        # Phase 11en changes only t/r3, n/r2, and n/r3 selector label
        # penalties. Candidate pool and block rules are exactly Phase 11em.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11em",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11em":
        # Phase 11em changes only n/r2 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11el.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11el",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11el":
        # Phase 11el changes only n/r2 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11ek.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ek",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ek":
        # Phase 11ek changes only n/r2 and n/r3 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11ej.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ej",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ej":
        # Phase 11ej changes only t/r3, n/r2, and n/r3 selector label
        # penalties. Candidate pool and block rules are exactly Phase 11ei.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ei",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ei":
        # Phase 11ei changes only t/r3 selector label penalties. Candidate
        # pool and block rules are exactly Phase 11eh.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11eh",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11eh":
        # Phase 11eh changes only n/r2 and n/r3 selector label penalties.
        # Candidate pool and block rules are exactly Phase 11eg.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11eg",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11eg":
        # Phase 11eg: Phase 11ef plus n/r3-only micro-add of
        # csig01_psig1, em5oG, and mlc2nobds. Combo replay on nagoya/run3
        # gave +2.494m.
        extra_blocked = set(blocked_labels or set())
        if (city, run) != ("nagoya", "run3"):
            extra_blocked.update({"csig01_psig1", "em5oG", "mlc2nobds"})
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ef",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ef":
        # Phase 11ef: Phase 11ee plus t/r3-only micro-add of
        # rtkout5minobs3. Single-add replay on tokyo/run3 gave +1.817m.
        extra_blocked = set(blocked_labels or set())
        if (city, run) != ("tokyo", "run3"):
            extra_blocked.add("rtkout5minobs3")
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ee",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ee":
        # Phase 11ee: Phase 11ed base plus n/r2-only micro-add of
        # csig005_em10 and onlyG_r05. Both labels are blocked elsewhere to
        # avoid widening the pool on runs that were not replay-positive.
        extra_blocked = set(blocked_labels or set())
        if (city, run) != ("nagoya", "run2"):
            extra_blocked.update({"csig005_em10", "onlyG_r05"})
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ed",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ed":
        # Phase 11ed changes only tokyo/run2 selector parameters; candidate
        # blocks are exactly the Phase 11ec pool.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ec",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ec":
        # Phase 11ec changes only nagoya/run3 selector parameters; candidate
        # blocks are exactly the Phase 11eb pool.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11eb",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11eb":
        # Phase 11eb changes only tokyo/run3 selector parameters; candidate
        # blocks are exactly the Phase 11ea pool.
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ea",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11ea":
        # Phase 11ea: tenth selected-loss block pass on top of 11dz.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r25g20"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dz",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dz":
        # Phase 11dz: ninth selected-loss block pass on top of 11dy.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"csig005_holdvrlx", "r20"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dy",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dy":
        # Phase 11dy: eighth selected-loss block pass on top of 11dx.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r20ga"},
            ("nagoya", "run3"): {"r30"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dx",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dx":
        # Phase 11dx: seventh selected-loss block pass on top of 11dw.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run1"): {"c005p1"},
            ("nagoya", "run2"): {"em5mlc2oG"},
            ("nagoya", "run3"): {"mlc1c005r10", "r20", "r20g40", "r30g"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dw",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dw":
        # Phase 11dw: sixth selected-loss block pass on top of 11dv.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run1"): {"r20g10", "r20g15"},
            ("nagoya", "run2"): {"onlyG"},
            ("nagoya", "run3"): {"r15g", "r15g20", "r25g20"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dv",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dv":
        # Phase 11dv: fifth selected-loss block pass on top of 11du.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r20g40", "ratio12oG", "nobds"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11du",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11du":
        # Phase 11du: fourth selected-loss block pass on top of 11dt.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run2"): {"r15g20", "r20g", "csig1"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dt",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dt":
        # Phase 11dt: third selected-loss block pass on top of 11ds.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run3"): {"oGr05", "psig2"},
            ("nagoya", "run2"): {"mlc1c005"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11ds",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11ds":
        # Phase 11ds: second selected-loss block pass on top of 11dr.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run1"): {"csig05hvr", "r25g15"},
            ("tokyo", "run2"): {"r15nh"},
            ("tokyo", "run3"): {"mlc1oGc005p1", "c005ga", "r05", "oGc01p1"},
            ("nagoya", "run1"): {"rtkout10", "csig05_em10"},
            ("nagoya", "run2"): {"n2loose"},
            ("nagoya", "run3"): {"r15nh", "r20g", "csig05", "mlc1"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dr",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dr":
        # Phase 11dr: keep Phase 11dq temporal selectors, but block labels
        # that single-run replay showed were locally replaceable by better
        # candidates.
        blocked_by_run: dict[tuple[str, str], set[str]] = {
            ("tokyo", "run1"): {"oGp1hr", "csig05psh"},
            ("nagoya", "run1"): {"c005hr", "mlc1c005p1", "oGc01"},
            ("nagoya", "run2"): {"rtkout5", "rtkout5c005", "oGr05", "n2loose2"},
            ("tokyo", "run3"): {"csig01", "mlc1oG", "csig05ps"},
            ("nagoya", "run3"): {"r15g15", "em3mlc1oG", "psig1", "csig05hr", "oGp1"},
        }
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(blocked_by_run.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dq",
            blocked_labels=extra_blocked,
        )
    if policy in {"phase11do", "phase11dp", "phase11dq"}:
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dn",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11dn":
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dm",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11dm":
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dl",
            blocked_labels=blocked_labels,
        )
    if policy == "phase11dl":
        # 11dk + final run-local positives from discovered diag dirs and
        # post-11dk all-known replay.  Offline combo: +10.410m / +0.022471pp.
        allowed: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run1"): {"xd_r25_nohold", "csig05_em10"},
            ("nagoya", "run2"): {"csig05_psig1", "em5mlc2oG"},
            ("nagoya", "run3"): {"xd_n3_loose_hold4_ratio15_gate10_min6"},
            ("tokyo", "run1"): {"xd_ratio4", "xd_r2_nohold", "xd_r25_nohold", "r10c005p1"},
            ("tokyo", "run2"): {"xd_ratio3_gate10_min6", "em5c005p1"},
            ("tokyo", "run3"): {"csig01_holdvrlx"},
        }
        surgical_labels = set().union(*allowed.values())
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(surgical_labels - allowed.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11dk",
            blocked_labels=extra_blocked,
        )
    if policy == "phase11dk":
        # 11di true-base + run-local positive extras discovered by
        # sim_ppc_phase_csv_addcand.py --allowed-pairs (delta_pass_m > 1m).
        allowed: dict[tuple[str, str], set[str]] = {
            ("nagoya", "run1"): {"ratio12", "csig05_psig1_holdvrlx"},
            ("nagoya", "run2"): {"mlc1oGc0001", "mlc1r10oG", "rtkout5oG", "psig3", "csig005_holdvrlx", "ratio12oG"},
            ("nagoya", "run3"): {"mlc1c005r10em3", "mlc1", "csig05_holdrlx_em10", "r10", "r08"},
            ("tokyo", "run1"): {"oGc005p1hr", "c005p1hr", "oGc005p2", "mlc1r10c005p1", "em3oG", "oGc005hr", "csig005_holdvrlx"},
            ("tokyo", "run2"): {"oGc005p05", "mlc1oGp1"},
            ("tokyo", "run3"): {"csig05_r10", "csig01_holdrlx"},
        }
        surgical_labels = set().union(*allowed.values())
        extra_blocked = set(blocked_labels or set())
        extra_blocked.update(surgical_labels - allowed.get((city, run), set()))
        return _filter_rtkdiag_candidates_by_policy(
            candidates,
            city=city,
            run=run,
            policy="phase11di",
            blocked_labels=extra_blocked,
        )
    effective_blocked_labels = set(blocked_labels or set())
    # Phase 11ce: experimentally block mlc1oG on n/r2 (currently dominant
    # selection ~1729 epochs, ~42% of selected). Test if this drives selection
    # toward better candidates.
    if policy == "phase11ce" and city == "nagoya" and run == "run2":
        effective_blocked_labels.add("mlc1oG")
    # Phase 11cf: 5 new candidates restricted to n/r2 only (regressed -0.86pp on n/r2).
    if policy == "phase11cf" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"c005em10", "mlc1c005r10oG", "GE", "psig01", "nopostflt"})
    # Phase 11cg: 5 different candidates targeting n/r2 (regressed -0.30pp).
    if policy == "phase11cg" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"c005hvrlx", "mlc1oGc0001", "GEonly", "noarfilt", "c005minar3"})
    # Phase 11ch: open phase11cf candidates GLOBALLY (block only on n/r2 where they regressed).
    if policy == "phase11ch" and (city, run) == ("nagoya", "run2"):
        effective_blocked_labels.update({"c005em10", "mlc1c005r10oG", "GE", "psig01", "nopostflt"})
    # Phase 11ck: modeauto + modestatic (regressed -7.70pp on n/r2).
    if policy == "phase11ck" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"modeauto", "modestatic"})
    # Phase 11cl: psig005 + oGc005hr + r12oG candidates restricted to n/r2.
    if policy == "phase11cl" and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.update({"psig005", "oGc005hr", "r12oG"})
    # Phase 11dg: surgical addition of 3 NEW candidates with strict per-run allowance
    # based on per-epoch oracle pick frequency (sim_ppc_oracle_label_freq.py):
    # - xr25_glonassar: allowed only on tokyo/run1 (8.6%) and tokyo/run2 (10.1%)
    # - xmlc1psig005: allowed only on tokyo/run2 (12.2%)
    # - xcsig005_em10: allowed only on nagoya/run2 (9.0%)
    if policy == "phase11dg":
        if (city, run) not in {("tokyo", "run1"), ("tokyo", "run2")}:
            effective_blocked_labels.add("xr25_glonassar")
        if (city, run) != ("tokyo", "run2"):
            effective_blocked_labels.add("xmlc1psig005")
        if (city, run) != ("nagoya", "run2"):
            effective_blocked_labels.add("xcsig005_em10")
    # Phase 11di: narrower surgical version after replaying additions against
    # the real Phase 11dd per-run pools. 11dg let several oracle-frequent
    # labels into runs where the 11dd composite selector actually regressed.
    if policy == "phase11di":
        if (city, run) != ("tokyo", "run1"):
            effective_blocked_labels.update({"xr25_glonassar", "xcsig005_em10"})
        if (city, run) != ("nagoya", "run1"):
            effective_blocked_labels.add("xr17_glonassar")
        if (city, run) != ("tokyo", "run3"):
            effective_blocked_labels.add("xpsig05")
        effective_blocked_labels.update({"xmlc1psig005", "xnobds_holdrlx"})
    if policy in {"phase11h", "phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and city == "nagoya" and run == "run2":
        # Phase 11cm: skip this 11h-era block on n/r2 to test if old reasoning still holds.
        # Phase 11cu/cv/cw: re-apply block (the cm-cp experiment showed regression).
        effective_blocked_labels.update({"r15g15", "r20g15", "r25g15", "r30g15"})
    if policy in {"phase11i", "phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (
        (city, run) in {("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run2")}
    ):
        effective_blocked_labels.update({"r30", "r30g"})
    if policy in {"phase11l", "phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (
        (city, run) in {("nagoya", "run1"), ("nagoya", "run2")}
    ):
        effective_blocked_labels.add("r20g10")
    if policy in {"phase11n", "phase11x", "phase11y", "phase11z", "phase11aa", "phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and city == "nagoya":
        effective_blocked_labels.update({"r15g10", "r25g10"})
    # Phase 11ab: r15ga (--glonass-ar autocal ratio 1.5) only helps tokyo/run3
    # and nagoya/run1 in offline simulation; block on the other 4 runs to
    # avoid the -30/-19/-5/-2 m/run regressions seen in the sim.
    if policy in {"phase11ab", "phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("r15ga")
    # Phase 11ac: r20ga only helps tokyo/run3 (+12.6m) and nagoya/run2
    # (+47.7m) on top of 11ab base. Block elsewhere.
    if policy in {"phase11ac", "phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("r20ga")
    # Phase 11ac: em10 (elev mask 10°) only helps tokyo/run3 (+21m). Block
    # elsewhere. Phase 11af also allows em10 on nagoya/run1 (+2.0m offline).
    if policy in {"phase11ac", "phase11ad", "phase11ae"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("em10")
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("em10")
    # Phase 11ad: psig1 (--pseudorange-sigma 1.0) gives +125m only on
    # tokyo/run3. Phase 11af also allows psig1 on nagoya/run3 (+4.1m offline).
    if policy in {"phase11ad", "phase11ae"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig1")
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("psig1")
    # Phase 11ad: holdrlx (relaxed hold ambiguity) gives +72m only on
    # tokyo/run3.
    if policy in {"phase11ad", "phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("holdrlx")
    # Phase 11ae: r12ga (--ratio 1.2 + --glonass-ar autocal) gives +103m
    # additive on tokyo/run3 only. Phase 11af keeps it (no harm).
    if policy in {"phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("r12ga")
    # Phase 11ae: psig2 (--pseudorange-sigma 2.0) gives +44.5m additive on
    # tokyo/run3 only. Phase 11af/ag keeps it (no harm).
    if policy in {"phase11ae", "phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig2")
    # Phase 11af: psig1hr (psig1 + holdrlx combined config) gives +90.8m
    # additive on tokyo/run3 only. Negative on nagoya runs.
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("psig1hr")
    # Phase 11af: nobds (--no-beidou) gives +29.7m on nagoya/run2 (largest),
    # +9.8m on tokyo/run3, +4.2m on tokyo/run1. Negative on nagoya/run1, run3.
    if policy in {"phase11af", "phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}:
        effective_blocked_labels.add("nobds")
    # Phase 11ag: csig05 (--carrier-phase-sigma 0.0005) gives massive gains:
    # tokyo/run3 +95.8m, nagoya/run2 +95.8m, tokyo/run1 +9.8m. Negative on
    # nagoya/run1/run3 originally; Phase 11am also enables nagoya/run3 (+2.6m
    # additive after the other csig05* variants).
    if policy in {"phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1")}:
        effective_blocked_labels.add("csig05")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("tokyo", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05")
    # Phase 11ag: noglo (--no-glonass) gives +40.5m on tokyo/run1 only.
    if policy in {"phase11ag", "phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("noglo")
    # Phase 11ag: csig1 (--carrier-phase-sigma 0.001) gives +0.6m on
    # nagoya/run1. Phase 11ah extends to {nagoya/run1, nagoya/run2, tokyo/run3}
    # (additive after csig05).
    if policy == "phase11ag" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("csig1")
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run1"), ("nagoya", "run2"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig1")
    # Phase 11ah: rout30 (--rtk-update-outlier-threshold 30) gives +3.6m
    # on nagoya/run3 only.
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("rout30")
    # Phase 11ah: rout20 (--rtk-update-outlier-threshold 20) gives +2.2m
    # on nagoya/run1 only.
    if policy in {"phase11ah", "phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rout20")
    # Phase 11ai: csig05hr (csig05 + holdrlx combined) gives massive gains:
    # tokyo/run1 +509.8m, tokyo/run3 +147.1m, nagoya/run3 +128.3m. Block on
    # nagoya/run2 (-7.7m) and nagoya/run1 (+0.4m, near-zero).
    if policy in {"phase11ai", "phase11aj", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05hr")
    # Phase 11ai: csig05ps (csig05 + psig1 combined) gives +22.5m on
    # tokyo/run3 and +3.4m on nagoya/run1. Phase 11aj also enables on
    # nagoya/run3 (+11.7m additive).
    # Phase 11ai/ak: csig05ps (csig05 + psig1) for {tokyo/run3, nagoya/run1}.
    # Phase 11aj added nagoya/run3 (-0.12pp PF on that run, so removed in 11ak).
    if policy in {"phase11ai", "phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("csig05ps")
    if policy == "phase11aj" and (city, run) not in {("tokyo", "run3"), ("nagoya", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05ps")
    # Phase 11aj/ak: csig01 (--carrier-phase-sigma 0.0001). Phase 11ak narrows
    # to {tokyo/run3, nagoya/run2} only — nagoya/run3 was -0.12pp in PF.
    if policy == "phase11aj" and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig01")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("csig01")
    # Phase 11aj: rout100 hurt nagoya/run1 by -1.49pp in PF — block in phase11ak.
    if policy == "phase11aj" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rout100")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rout100")
    # Phase 11aj: csig05nb (csig05 + nobds) +0.8m offline but -0.02pp PF.
    # Block entirely in phase11ak.
    if policy == "phase11aj" and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("csig05nb")
    if policy in {"phase11ak", "phase11al", "phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("csig05nb")
    # Phase 11ak: csig05hvr (csig05 + holdvrlx very loose hold) gives +53.2m
    # on tokyo/run1 only. Phase 11am also enables tokyo/run3 (+29.4m offline).
    if policy in {"phase11ak", "phase11al"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("csig05hvr")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig05hvr")
    # Phase 11ak: csig05psh (csig05 + psig1 + holdrlx triple) gives +44m on
    # tokyo/run3 and +21.5m on nagoya/run3. Phase 11am also enables tokyo/run1 (+4.1m).
    if policy in {"phase11ak", "phase11al"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("csig05psh")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3"), ("tokyo", "run1")}:
        effective_blocked_labels.add("csig05psh")
    # Phase 11ak: csig05em (csig05 + em10) gives +2.5m on nagoya/run1.
    if policy == "phase11ak" and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("csig05em")
    if policy == "phase11al":
        effective_blocked_labels.add("csig05em")
    # Phase 11am: csig05em allowed only on tokyo/run3 (+20.5m offline). Block elsewhere.
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("csig05em")
    # Phase 11ak: csig01hr (csig01 + holdrlx) gives +42.8m on nagoya/run2.
    # Phase 11am widens to {tokyo/run1 +7.2m, tokyo/run3 +23.1m, nagoya/run2 +42.8m}.
    if policy in {"phase11ak", "phase11al"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("csig01hr")
    if policy in {"phase11am", "phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run2"), ("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("csig01hr")
    # Phase 11an: c5p1hvr (csig05+psig1+holdvrlx 4-knob) gives +104.6m on
    # tokyo/run1 and +28.4m on nagoya/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("c5p1hvr")
    # Phase 11an: c1p1hr (csig01+psig1+holdrlx triple) gives +7.3m on tokyo/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("c1p1hr")
    # Phase 11an: c5hrem (csig05+holdrlx+em10 triple) gives +18.2m on tokyo/run3 offline.
    if policy in {"phase11an", "phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("c5hrem")
    # Phase 11ao: csig005 (--carrier-phase-sigma 0.00005, super-tight) gives
    # +36.1m on nagoya/run2 (the lowest run). Marginal elsewhere.
    if policy in {"phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("csig005")
    # Phase 11ao: c5nbhr (csig05+nobds+holdrlx triple). Marginal +9.8m on
    # tokyo/run1 and +6.8m on nagoya/run1. Negative on others.
    if policy in {"phase11ao", "phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c5nbhr")
    # Phase 11ap: c005hr (csig005+holdrlx) marginal +2.6m on tokyo/run1, +9.5m on nagoya/run1.
    if policy in {"phase11ap", "phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c005hr")
    # Phase 11aq: r05 (--ratio 0.5) marginal +5.0/+3.9/+2.3m on tokyo/run1, tokyo/run3, nagoya/run1.
    if policy in {"phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("r05")
    # Phase 11aq: c005ga (csig005+glonassar) marginal +1.8/+3.3m on tokyo/run1, tokyo/run3.
    if policy in {"phase11aq", "phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("c005ga")
    # Phase 11ar: onlyG (--no-glonass --no-beidou, only G+E+J). +9.4m tokyo/run1
    # and **+20.2m on nagoya/run2** (the lowest-scoring run). Negative on nagoya/run3.
    if policy in {"phase11ar", "phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("nagoya", "run2")}:
        effective_blocked_labels.add("onlyG")
    # Phase 11as: oGc05 (onlyG + csig05) +14.3m on tokyo/run1.
    if policy in {"phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGc05")
    # Phase 11as: oGc005 (onlyG + csig005) +10.4m on nagoya/run2.
    if policy in {"phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("oGc005")
    # Phase 11as: oGp1 (onlyG + psig1) +2.1/+3.0m on nagoya/run1, nagoya/run3.
    if policy in {"phase11as", "phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run1"), ("nagoya", "run3")}:
        effective_blocked_labels.add("oGp1")
    # Phase 11at: oGp1hr (onlyG + psig1 + holdrlx) +15.4m on tokyo/run1.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGp1hr")
    # Phase 11at: oGp1c05 (onlyG + psig1 + csig05) +9.2m nagoya/run3, +2.0m nagoya/run1.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run3"), ("nagoya", "run1")}:
        effective_blocked_labels.add("oGp1c05")
    # Phase 11at: oGr05 (onlyG + ratio 0.5) marginal +2.2/+3.7m on tokyo/run3, nagoya/run2.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("oGr05")
    # Phase 11at: oGc01 (onlyG + csig01) marginal +3.1m on nagoya/run1.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("oGc01")
    # Phase 11at: oGem10 (onlyG + em10) marginal +1.5m/+1.1m on tokyo/run3, nagoya/run3.
    if policy in {"phase11at", "phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run3")}:
        effective_blocked_labels.add("oGem10")
    # Phase 11au: oGc005p1 (onlyG + csig005 + psig1) +37m tokyo/run2, +6.6m tokyo/run1, +1.2m nagoya/run1.
    if policy in {"phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run1"), ("nagoya", "run1")}:
        effective_blocked_labels.add("oGc005p1")
    # Phase 11au: c005p1 (csig005 + psig1, no onlyG) +13.9m tokyo/run1, +15.1m tokyo/run2, +1.5m nagoya/run1.
    if policy in {"phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1")}:
        effective_blocked_labels.add("c005p1")
    # Phase 11au: oGc01p1 (onlyG + csig01 + psig1) +21.4m tokyo/run2, +3.4m tokyo/run3.
    if policy in {"phase11au", "phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run3")}:
        effective_blocked_labels.add("oGc01p1")
    # Phase 11aw: oGc00005p1 (onlyG + csig 0.00005 + psig 1) +16.7m tokyo/run1.
    if policy in {"phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("oGc00005p1")
    # Phase 11aw: oGc0001p1 (onlyG + csig 0.0001 + psig 1) +16.3m tokyo/run2, +5.4m tokyo/run1.
    if policy in {"phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run1")}:
        effective_blocked_labels.add("oGc0001p1")
    # Phase 11aw: nobdsc005p1 (no-beidou + csig005 + psig1) +2.7m nagoya/run3.
    if policy in {"phase11aw", "phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("nobdsc005p1")
    # Phase 11ay: em5 (--elevation-mask-deg 5) +13.8m nagoya/run1, +8.0m tokyo/run2.
    if policy in {"phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run1"), ("tokyo", "run2")}:
        effective_blocked_labels.add("em5")
    # Phase 11ay: mlc2oG (--min-lock-count 2 + onlyG) +7.5m tokyo/run1, +1.3m tokyo/run3.
    if policy in {"phase11ay", "phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run1"), ("tokyo", "run3")}:
        effective_blocked_labels.add("mlc2oG")
    # Phase 11ba: mlc1oG (--min-lock-count 1 + onlyG) +11.3m t/r2, +9.3m t/r3, +3.6m n/r2.
    if policy in {"phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run2"), ("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("mlc1oG")
    # Phase 11ba: em3 (--elev-mask-deg 3) +5.4m nagoya/run1.
    if policy in {"phase11ba", "phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("em3")
    # Phase 11bc: mlc1oGc005p1 (mlc1+onlyG+csig005+psig1) positive 5 runs, n/r2 only loser.
    if policy in {"phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run1"), ("nagoya", "run3"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("mlc1oGc005p1")
    # Phase 11bc: em3mlc1oG (em3+mlc1+onlyG) n/r3 +12m, t/r3 +6m, n/r1 +1m.
    if policy in {"phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run3"), ("nagoya", "run3"), ("nagoya", "run1"),
    }:
        effective_blocked_labels.add("em3mlc1oG")
    # Phase 11bc: mlc1oGc005 (mlc1+onlyG+csig005) n/r2 +7m, t/r2 +3m, t/r3 +5m.
    if policy in {"phase11bc", "phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("nagoya", "run2"), ("tokyo", "run2"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("mlc1oGc005")
    # Phase 11be: mlc1c005p1 (mlc1+csig005+psig1, no onlyG) +5m n/r1, +4m n/r3, +2m n/r2.
    # Phase 11bf: drop n/r2 (lost -3.36m PF, displaced existing winners).
    _mlc1c005p1_runs = {("nagoya", "run1"), ("nagoya", "run3"), ("nagoya", "run2")}
    if policy in {"phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        _mlc1c005p1_runs = {("nagoya", "run1"), ("nagoya", "run3")}
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in _mlc1c005p1_runs:
        effective_blocked_labels.add("mlc1c005p1")
    # Phase 11be: mlc1oGc005em3 (mlc1+onlyG+csig005+em3) n/r2 +5m, t/r3 +4m, t/r1 +1m.
    # Phase 11bf: drop n/r2 (suspected over-trust contributor to -3.36m loss).
    _mlc1oGc005em3_runs = {("nagoya", "run2"), ("tokyo", "run3"), ("tokyo", "run1")}
    if policy in {"phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        _mlc1oGc005em3_runs = {("tokyo", "run3"), ("tokyo", "run1")}
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in _mlc1oGc005em3_runs:
        effective_blocked_labels.add("mlc1oGc005em3")
    # Phase 11be: mlc1oGc005r12 (mlc1+onlyG+csig005+ratio 1.2) t/r3 +5m, t/r1 +1m.
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run3"), ("tokyo", "run1"),
    }:
        effective_blocked_labels.add("mlc1oGc005r12")
    # Phase 11be: mlc1nobds (mlc1+no-beidou) t/r1 +5m.
    if policy in {"phase11be", "phase11bf", "phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("mlc1nobds")
    # Phase 11bh: mlc1c005r10 (mlc1+csig005+ratio 1.0) +14m n/r3, +3m t/r1, +1.5m t/r2.
    if policy in {"phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("nagoya", "run3"), ("tokyo", "run1"), ("tokyo", "run2"),
    }:
        effective_blocked_labels.add("mlc1c005r10")
    # Phase 11bh: mlc1r10 (mlc1+ratio 1.0) +8m n/r1, +6.5m t/r3.
    if policy in {"phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("nagoya", "run1"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("mlc1r10")
    # Phase 11bh: mlc1c005 (mlc1+csig005) +4.2m n/r2 (rare positive).
    if policy in {"phase11bh", "phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("mlc1c005")
    # Phase 11bk: rtkout5 (--rtk-update-outlier-threshold 5) HUGE on tokyo/run1 (+74m, +0.72pp).
    # Also tokyo/run2 +6.94m, nagoya/run2 +2.91m. Other runs negative.
    if policy in {"phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("tokyo", "run2"), ("nagoya", "run2"),
    }:
        effective_blocked_labels.add("rtkout5")
    # Phase 11bk: rtkout10 (--rtk-update-outlier-threshold 10) +3.95m nagoya/run1.
    if policy in {"phase11bk", "phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rtkout10")
    # Phase 11bl: rtkout3 (--rtk-update-outlier-threshold 3) HUGE on tokyo/run1 (+167m, +1.62pp).
    # Marginal positive on nagoya/run2 (+1.5m). Other runs negative.
    if policy in {"phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("nagoya", "run2"),
    }:
        effective_blocked_labels.add("rtkout3")
    # Phase 11bl: rtkout7 marginal positive on nagoya/run1 (+4.2m).
    if policy in {"phase11bl", "phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run1"):
        effective_blocked_labels.add("rtkout7")
    # Phase 11bm: rtkout1 (--rtk-update-outlier-threshold 1) HUGE on tokyo/run1 (+175.5m, +1.70pp)
    # and tokyo/run3 (+14.4m). Other runs negative.
    if policy in {"phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run1"), ("tokyo", "run3"),
    }:
        effective_blocked_labels.add("rtkout1")
    # Phase 11bm: rtkout5c005 (rtkout5 + carrier-phase-sigma 0.0005) +15.0m nagoya/run2.
    if policy in {"phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("rtkout5c005")
    # Phase 11bm: rtkout5em3 (rtkout5 + elevation-mask-deg 3) +13.1m t/r2, +4.2m n/r1.
    if policy in {"phase11bm", "phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {
        ("tokyo", "run2"), ("nagoya", "run1"),
    }:
        effective_blocked_labels.add("rtkout5em3")
    # Phase 11bn: rtkout2/rtkout4/rtkout3c005 are not winners — block from all runs.
    # Were leaked into pool by phase 11bm dirs/labels but no per-run target; caused
    # -113m on t/r3 and -67m on n/r2 displacement in 11bm.
    if policy in {"phase11bn", "phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout2")
        effective_blocked_labels.add("rtkout4")
        effective_blocked_labels.add("rtkout3c005")
    # Phase 11bo: rtkout3oG (rtkout3 + no-glonass + no-beidou) +61.5m on tokyo/run1.
    # Other runs negative.
    if policy in {"phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3oG")
    # Phase 11bo: rtkout3em3 / rtkout3minobs3 are non-winners on every run; block globally.
    if policy in {"phase11bo", "phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout3em3")
        effective_blocked_labels.add("rtkout3minobs3")
    # Phase 11bp: rtkout1c005oG (rtkout1 + carrier-phase-sigma 0.0005 + no-glonass/beidou)
    # +125.1m on tokyo/run1 (best variant in phase 11bp sweep). Other runs negative.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout1c005oG")
    # Phase 11bp: rtkout5oGc005 (rtkout5 + no-glonass/beidou + csig005) +35.3m on tokyo/run3
    # (rare positive on this run). Other runs negative.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run3"):
        effective_blocked_labels.add("rtkout5oGc005")
    # Phase 11bp: rtkout1c005 +2.8m on n/r2 (marginal); also strong on t/r1 (+111m) but rtkout1c005oG
    # is even better there, so target n/r2 only.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run2"):
        effective_blocked_labels.add("rtkout1c005")
    # Phase 11bp: non-winner combos (rtkout1oG, rtkout1em3, rtkout1minobs3, rtkout10minobs3) — block globally.
    if policy in {"phase11bp", "phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout1oG")
        effective_blocked_labels.add("rtkout1em3")
        effective_blocked_labels.add("rtkout1minobs3")
        effective_blocked_labels.add("rtkout10minobs3")
    # Phase 11bq: rtkout3c005oG (rtkout3 + csig005 + no-glonass/beidou) +404m on tokyo/run1 (HUGE).
    # Other runs marginally negative. Best single variant of session.
    if policy in {"phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3c005oG")
    # Phase 11bq: rtkout5c005em3 (rtkout5 + csig005 + em3) +18m on n/r3 (first n/r3 winner) and
    # +4.5m on n/r2.
    if policy in {"phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("nagoya", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("rtkout5c005em3")
    # Phase 11bq: non-winner combos — block globally.
    if policy in {"phase11bq", "phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout1c005em3")
        effective_blocked_labels.add("rtkout1oGc005em3")
        effective_blocked_labels.add("rtkout3c005em3")
        effective_blocked_labels.add("rtkout1minobs4")
        effective_blocked_labels.add("rtkout1minobs6")
    # Phase 11br: rtkout3oGem3 (rtkout3 + no-glonass/beidou + em3) +73.5m on tokyo/run1 (offline,
    # vs phase11bp base — additive on top of phase11bq's rtkout3c005oG depends on epoch overlap).
    if policy in {"phase11br", "phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3oGem3")
    # Block rtkout3oGem3 globally for phase11bq (kept as phase11br-only target).
    if policy == "phase11bq":
        effective_blocked_labels.add("rtkout3oGem3")
    # Phase 11bt: rtkout3mlc1c005 (rtkout3 + mlc1 + csig005) +278.1m on tokyo/run1 (offline winner).
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("tokyo", "run1"):
        effective_blocked_labels.add("rtkout3mlc1c005")
    # Phase 11bt: rtkout5mlc1c005oG (rtkout5 + mlc1 + csig005 + no-glonass/beidou)
    # +26.6m on t/r3 and +11.5m on n/r2 — winner on TWO runs.
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) not in {("tokyo", "run3"), ("nagoya", "run2")}:
        effective_blocked_labels.add("rtkout5mlc1c005oG")
    # Phase 11bt: rtkout3mlc1 +17.1m on n/r3 (third n/r3 winner of session).
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"} and (city, run) != ("nagoya", "run3"):
        effective_blocked_labels.add("rtkout3mlc1")
    # Phase 11bt: non-winner combos — block globally.
    if policy in {"phase11bt", "phase11bu", "phase11bv", "phase11bw", "phase11bx", "phase11by", "phase11bz", "phase11ca", "phase11cb", "phase11cc", "phase11cd", "phase11ce", "phase11cf", "phase11cg", "phase11ch", "phase11ci", "phase11cj", "phase11ck", "phase11cl", "phase11cm", "phase11cn", "phase11co", "phase11cp", "phase11cu", "phase11cv", "phase11cw", "phase11cx", "phase11cy", "phase11cz", "phase11da", "phase11db", "phase11dc", "phase11dd", "phase11dg", "phase11dh", "phase11di"}:
        effective_blocked_labels.add("rtkout1mlc1")
        effective_blocked_labels.add("rtkout5mlc1")
        effective_blocked_labels.add("rtkout3mlc1oG")
        effective_blocked_labels.add("rtkout1mlc1c005")
    if not effective_blocked_labels:
        return candidates
    return [
        candidate for candidate in candidates
        if candidate[0] not in effective_blocked_labels
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-RBPF-FGO PPC port (Phase 0 scaffolding)")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-prefix", type=str, default="ppc_ctrbpf_fgo")
    parser.add_argument(
        "--write-internal-diagnostics",
        action="store_true",
        help="Write per-epoch PF ESS/spread/stage-delta diagnostics to results",
    )
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
            "rbpf+dd, rbpf+dd+pu, rbpf+dd+gate, rbpf+dd+gate+pu, "
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
                        choices=("residual", "ratio", "score", "maxabs", "nrows", "hybrid_anchor", "wavg3", "wavg5", "consensus3", "consensus5", "cluster_vote", "ranker",
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
                             "useful when hybrid is reliable")
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
        "--runs",
        type=str,
        default="all",
        help="Comma-separated run filter, e.g. 'tokyo/run1,nagoya/run3'. 'all' = 6 runs.",
    )
    args = parser.parse_args()

    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    args.systems_tuple = tuple(s.strip() for s in args.systems.split(",") if s.strip())

    if args.runs == "all":
        runs = _FULL_RUNS
    else:
        wanted = {r.strip() for r in args.runs.split(",") if r.strip()}
        runs = tuple((c, r) for c, r in _FULL_RUNS if f"{c}/{r}" in wanted)
        if not runs:
            raise SystemExit(f"no matching runs: {args.runs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.pos_dir.mkdir(parents=True, exist_ok=True)

    variants = _config_variants(args)
    rtkdiag_candidate_pos_dirs = (
        _parse_path_list(args.rtkdiag_candidate_pos_dirs)
        if str(args.rtkdiag_candidate_pos_dirs).strip()
        else ([args.rtkdiag_candidate_pos_dir] if args.rtkdiag_candidate_pos_dir else [])
    )
    rtkdiag_candidate_diag_dirs = (
        _parse_path_list(args.rtkdiag_candidate_diag_dirs)
        if str(args.rtkdiag_candidate_diag_dirs).strip()
        else ([args.rtkdiag_candidate_diag_dir] if args.rtkdiag_candidate_diag_dir else [])
    )
    rtkdiag_candidate_labels = _parse_label_list(args.rtkdiag_candidate_labels)
    if not rtkdiag_candidate_labels:
        rtkdiag_candidate_labels = [
            p.name or f"candidate{i}"
            for i, p in enumerate(rtkdiag_candidate_pos_dirs)
        ]
    rtkdiag_candidate_block_labels = set(
        _parse_label_list(args.rtkdiag_candidate_block_labels)
    )
    rtkdiag_candidate_block_labels_by_run = _parse_run_label_blocks(
        args.rtkdiag_candidate_block_labels_by_run
    )

    any_dd = any(v.enable_dd_carrier_afv for v in variants)
    any_dd_pr = any(v.enable_dd_pr_ls_anchor for v in variants)
    any_dd_for_gate = any(
        (v.enable_rbpf_velocity_kf and v.rbpf_kf_gate_min_dd_pairs is not None)
        for v in variants
    )
    any_gate = any(
        v.enable_rbpf_velocity_kf
        and (
            v.rbpf_kf_gate_min_dd_pairs is not None
            or v.rbpf_kf_gate_min_ess_ratio is not None
            or v.rbpf_kf_gate_max_spread_m is not None
        )
        for v in variants
    )
    any_hybrid = any(v.enable_hybrid_pu for v in variants)
    if any_hybrid and args.hybrid_pos_dir is None:
        raise SystemExit(
            "--hybrid-pos-dir is required when any *+hybrid method is selected"
        )
    any_rtkdiag_pf = any(v.enable_rtkdiag_pf_rescue for v in variants)
    if any_rtkdiag_pf and (
        not rtkdiag_candidate_pos_dirs
        or not rtkdiag_candidate_diag_dirs
        or len(rtkdiag_candidate_pos_dirs) != len(rtkdiag_candidate_diag_dirs)
        or len(rtkdiag_candidate_labels) != len(rtkdiag_candidate_pos_dirs)
    ):
        raise SystemExit(
            "*+rtkdiag_pf requires matching candidate pos/diag dirs and labels "
            "(use either singular --rtkdiag-candidate-pos-dir/diag-dir or "
            "comma-separated --rtkdiag-candidate-pos-dirs/diag-dirs)"
        )

    print("=" * 72)
    print("  CT-RBPF-FGO PPC port — Phase 6 (hybrid blend)")
    print(f"  Methods: {[v.method_label for v in variants]}")
    print(f"  Particles: {args.n_particles}, Systems: {args.systems_tuple}")
    if any_dd or any_dd_for_gate or any_dd_pr:
        print(
        f"  DD: sigma_cycles={args.dd_sigma_cycles}, "
        f"min_pairs={args.dd_min_pairs}/{args.dd_min_pairs_update}, "
        f"systems={args.dd_systems}, base_interp={bool(args.dd_base_interp)}, "
        f"min_el={args.dd_min_elevation_deg}, min_snr={args.dd_min_snr}, "
        f"keep_best={args.dd_keep_best}, "
        f"dd_pr_gate={args.dd_pr_pair_residual_max_m}/"
        f"{args.dd_pr_epoch_median_residual_max_m}m"
    )
    if any_gate:
        print(
            f"  RBPF gate: min_dd_pairs={args.rbpf_velocity_kf_gate_min_dd_pairs}, "
            f"min_ess_ratio={args.rbpf_velocity_kf_gate_min_ess_ratio}, "
            f"max_spread_m={args.rbpf_velocity_kf_gate_max_spread_m} "
            "(per-variant overrides via method labels)"
        )
    if any_hybrid:
        print(
            f"  Hybrid PU: dir={args.hybrid_pos_dir}, "
            f"suffix={args.hybrid_pos_suffix}, sigma_m={args.hybrid_sigma_m}"
        )
    if any_rtkdiag_pf:
        policy_note = (
            " base"
            if str(args.rtkdiag_candidate_run_index_policy) != "none"
            else ""
        )
        print(
            f"  RTKDiag PF{policy_note}: candidates={rtkdiag_candidate_labels}, "
            f"sigma={args.rtkdiag_candidate_sigma_m}, "
            f"ratio>={args.rtkdiag_candidate_ratio_min}, "
            f"rms<={args.rtkdiag_candidate_residual_rms_max}, "
            f"select={args.rtkdiag_candidate_select_mode}, "
            f"emit={args.rtkdiag_candidate_emit_mode}, "
            f"min_epoch={args.rtkdiag_candidate_min_epoch}, "
            f"diag_any={args.rtkdiag_candidate_require_any_diag_fields or '-'}, "
            f"diag_min={args.rtkdiag_candidate_min_diag_fields or '-'}, "
            f"diag_max={args.rtkdiag_candidate_max_diag_fields or '-'}, "
            f"fallback={args.rtkdiag_candidate_fallback_mode}, "
            f"run_policy={args.rtkdiag_candidate_run_index_policy}"
        )
    print("=" * 72)

    rows: list[dict[str, object]] = []
    internal_diag_rows: list[dict[str, object]] = []
    agg_pass: dict[str, float] = {v.method_label: 0.0 for v in variants}
    agg_total: dict[str, float] = {v.method_label: 0.0 for v in variants}
    per_run_pcts: dict[str, list[float]] = {v.method_label: [] for v in variants}

    for city, run in runs:
        run_dir = args.data_root / city / run
        if not run_dir.is_dir():
            print(f"  [skip] {city}/{run}: missing {run_dir}")
            continue
        print(f"\n[{city}/{run}] loading PPC data ...", flush=True)
        loader = PPCDatasetLoader(run_dir)
        data = loader.load_experiment_data(
            max_epochs=args.max_epochs,
            start_epoch=args.start_epoch,
            systems=args.systems_tuple,
            include_sat_velocity=any(
                v.enable_rbpf_velocity_kf or v.enable_tdcp_smoother
                for v in variants
            ),
        )
        iono_alpha_run, iono_beta_run = read_gps_klobuchar_from_nav_header(
            run_dir / "base.nav"
        )
        iono_alpha_tuple = tuple(float(x) for x in (iono_alpha_run or ()))
        iono_beta_tuple = tuple(float(x) for x in (iono_beta_run or ()))
        full_ref = _load_full_reference(run_dir / "reference.csv")
        reference_pos_run = _reference_position_map(full_ref)
        n_emit = _emission_count(full_ref, np.asarray(data["times"], dtype=np.float64))
        print(
            f"  data: {data['n_epochs']} usable / {len(full_ref)} ref epochs "
            f"({100.0 * n_emit / max(len(full_ref), 1):.1f}% coverage), "
            f"constellations={data['constellations']}",
            flush=True,
        )

        dd_computer = None
        dd_pr_computer = None
        any_fgo = any(v.enable_fgo_lambda for v in variants)
        if any_dd or any_dd_for_gate or any_fgo or any_dd_pr:
            from gnss_gpu.dd_carrier import DDCarrierComputer

            base_obs_path = run_dir / "base.obs"
            rover_obs_path = run_dir / "rover.obs"
            if not base_obs_path.is_file() or not rover_obs_path.is_file():
                print(
                    f"  [DD] WARNING: missing rover.obs/base.obs in {run_dir}, "
                    "DD updates skipped for this run"
                )
            else:
                dd_systems_run = tuple(
                    s.strip() for s in args.dd_systems.split(",") if s.strip()
                )
                dd_computer = DDCarrierComputer(
                    base_obs_path,
                    rover_obs_path=rover_obs_path,
                    base_position=np.asarray(data["base_ecef"], dtype=np.float64),
                    allowed_systems=dd_systems_run,
                    interpolate_base_epochs=bool(args.dd_base_interp),
                )
                if any_fgo or any_dd_pr:
                    from gnss_gpu.dd_pseudorange import DDPseudorangeComputer

                    dd_pr_computer = DDPseudorangeComputer(
                        base_obs_path,
                        rover_obs_path=rover_obs_path,
                        base_position=np.asarray(data["base_ecef"], dtype=np.float64),
                        allowed_systems=dd_systems_run,
                        interpolate_base_epochs=bool(args.dd_base_interp),
                    )
                    print(
                        f"  [DD-PR] Loaded for absolute anchor: "
                        f"base_systems={dd_systems_run}"
                    )

        hybrid_pos_run: dict[float, np.ndarray] | None = None
        hybrid_velocity_run: dict[float, np.ndarray] | None = None
        hybrid_status_run: dict[float, int] | None = None
        rtkdiag_candidates_run: list[
            tuple[str, dict[float, np.ndarray], dict[float, dict[str, str]]]
        ] = []
        imu_run: dict[str, np.ndarray] | None = None
        if any(v.enable_zupt or v.enable_imu_tc or v.enable_ins_tc for v in variants):
            try:
                imu_run = loader.load_imu()
            except FileNotFoundError as exc:
                print(
                    f"  [IMU] WARNING: {exc}, ZUPT / Phase 9b/9c IMU-TC "
                    "disabled for this run"
                )
                imu_run = None
        if (any_hybrid or any_fgo) and args.hybrid_pos_dir is not None:
            hybrid_path = args.hybrid_pos_dir / f"{city}_{run}{args.hybrid_pos_suffix}"
            if not hybrid_path.is_file():
                print(
                    f"  [hybrid] WARNING: missing {hybrid_path}, hybrid PU disabled for this run"
                )
            else:
                hybrid_pos_run, hybrid_status_run = _load_hybrid_pos_file(hybrid_path)
                from collections import Counter
                status_dist = Counter(hybrid_status_run.values())
                print(
                    f"  [hybrid] {hybrid_path.name}: "
                    f"{len(hybrid_pos_run)} usable rows, status={dict(status_dist)}"
                )
                hybrid_velocity_run = _build_hybrid_velocity_guide(
                    hybrid_pos_run,
                    np.asarray(data["times"], dtype=np.float64),
                    max_dt_s=float(args.hybrid_vguide_max_dt_s),
                )
                print(
                    f"  [hybrid] velocity guide: {len(hybrid_velocity_run)} epochs "
                    f"(max_dt={args.hybrid_vguide_max_dt_s}s)"
                )
        if any_rtkdiag_pf:
            for label, pos_dir, diag_dir in zip(
                rtkdiag_candidate_labels,
                rtkdiag_candidate_pos_dirs,
                rtkdiag_candidate_diag_dirs,
                strict=True,
            ):
                candidate_pos_path = pos_dir / f"{city}_{run}{args.rtkdiag_candidate_pos_suffix}"
                candidate_diag_path = diag_dir / f"{city}_{run}{args.rtkdiag_candidate_diag_suffix}"
                if not candidate_pos_path.is_file() or not candidate_diag_path.is_file():
                    print(
                        f"  [rtkdiag:{label}] WARNING: missing {candidate_pos_path} or "
                        f"{candidate_diag_path}; candidate skipped for this run"
                    )
                    continue
                candidate_pos_run, _candidate_status = _load_hybrid_pos_file(candidate_pos_path)
                candidate_diag_run = _load_rtk_diag_file(candidate_diag_path)
                rtkdiag_candidates_run.append((label, candidate_pos_run, candidate_diag_run))
                print(
                    f"  [rtkdiag:{label}] {candidate_pos_path.name}: "
                    f"{len(candidate_pos_run)} candidate rows, "
                    f"{len(candidate_diag_run)} diagnostic rows"
                )
        rtkdiag_candidates_run_policy = _filter_rtkdiag_candidates_by_policy(
            rtkdiag_candidates_run,
            city=city,
            run=run,
            policy=str(args.rtkdiag_candidate_run_index_policy),
            blocked_labels=(
                rtkdiag_candidate_block_labels
                | rtkdiag_candidate_block_labels_by_run.get((city, run), set())
            ),
        )
        if len(rtkdiag_candidates_run_policy) != len(rtkdiag_candidates_run):
            kept_labels = {label for label, _, _ in rtkdiag_candidates_run_policy}
            removed_labels = [
                label for label, _, _ in rtkdiag_candidates_run
                if label not in kept_labels
            ]
            print(
                f"  [rtkdiag-policy:{args.rtkdiag_candidate_run_index_policy}] "
                f"filtered candidates for {city}/{run}: "
                f"removed={removed_labels}",
                flush=True,
            )

        pr_systems_run = tuple(s.strip() for s in args.pr_systems.split(",") if s.strip())
        wls_data = _filter_data_by_systems(data, pr_systems_run)
        if wls_data is not data:
            print(
                f"  PR/WLS systems: {pr_systems_run} "
                f"(median sats={wls_data['n_satellites']}; loaded={data['constellations']})"
            )
        spp_kwargs: dict = {}
        if args.spp_nlos_mask_path:
            nlos_path_run = args.spp_nlos_mask_path.format(city=city, run=run)
            nlos_set_run = _load_nlos_mask_csv(nlos_path_run)
            if nlos_set_run:
                spp_kwargs["nlos_set_per_epoch"] = nlos_set_run
                spp_kwargs["nlos_k_weak"] = float(args.spp_nlos_k_weak)
                spp_kwargs["nlos_k_strong"] = float(args.spp_nlos_k_strong)
                n_nlos_epochs = sum(1 for s in nlos_set_run.values() if s)
                n_nlos_obs = sum(len(s) for s in nlos_set_run.values())
                print(
                    f"  SPP NLOS mask: {nlos_path_run} → "
                    f"{n_nlos_obs} obs across {n_nlos_epochs} epochs "
                    f"(k_weak={args.spp_nlos_k_weak}, k_strong={args.spp_nlos_k_strong})"
                )
            else:
                print(f"  SPP NLOS mask: {nlos_path_run} → empty/missing, skipping")
        if args.spp_nlos_strong_mask_path:
            strong_path_run = args.spp_nlos_strong_mask_path.format(city=city, run=run)
            strong_set_run = _load_nlos_mask_csv(strong_path_run)
            if strong_set_run:
                spp_kwargs["nlos_strong_set_per_epoch"] = strong_set_run
        if args.spp_irls != "off":
            spp_kwargs["irls_mode"] = str(args.spp_irls)
            spp_kwargs["irls_c"] = float(args.spp_irls_c)
            spp_kwargs["irls_threshold_m"] = float(args.spp_irls_c)
            print(
                f"  SPP IRLS: {args.spp_irls} c={args.spp_irls_c:.1f}m "
                f"shift_cap={args.spp_irls_shift_cap_m:.1f}m"
            )
        wls_atmosphere_model = str(args.pr_atmosphere_model).strip().lower()
        if wls_atmosphere_model == "off" and float(args.pr_slant_delay_zenith_m) > 0.0:
            wls_atmosphere_model = "fixed"
        selected_pr_min_elevation_deg = float(args.pr_min_elevation_deg)
        selected_pr_prefit_gate_m = float(args.pr_prefit_gate_m)
        selected_sigma_pr = float(args.sigma_pr)
        selected_pr_auto_robust_profile = False
        auto_elevation_postfit_rms_m = float("nan")
        auto_elevation_gated_sat_fraction = float("nan")
        auto_elevation_gated_pdop_ratio = float("nan")
        auto_elevation_selected = False
        if (
            bool(args.pr_auto_elevation_gate)
            and selected_pr_min_elevation_deg <= -90.0
        ):
            auto_wls_data = wls_data
            auto_seed_positions, auto_ms = run_wls(auto_wls_data)
            if wls_atmosphere_model not in {"", "off"}:
                auto_wls_data = _apply_atmosphere_to_pseudoranges(
                    auto_wls_data,
                    auto_seed_positions,
                    model=wls_atmosphere_model,
                    fixed_zenith_m=float(args.pr_slant_delay_zenith_m),
                    model_scale=float(args.pr_atmosphere_scale),
                    extra_zenith_m=float(args.pr_atmosphere_extra_zenith_m),
                    iono_alpha=iono_alpha_tuple,
                    iono_beta=iono_beta_tuple,
                )
            if float(args.pr_prefit_gate_m) > 0.0:
                auto_wls_data, _auto_kept, _auto_dropped = _filter_data_by_pr_prefit(
                    auto_wls_data,
                    auto_seed_positions,
                    gate_m=float(args.pr_prefit_gate_m),
                    clock_quantile=0.5,
                    min_sats=int(args.pr_prefit_gate_min_sats),
                    keep_best=int(args.pr_prefit_gate_keep_best),
                    per_system=bool(args.pr_prefit_per_system),
                )
            auto_positions, auto_wls_ms = run_wls(auto_wls_data)
            auto_ms += auto_wls_ms
            auto_postfit_rms = _wls_centered_postfit_rms_m(
                auto_wls_data,
                auto_positions,
            )
            auto_elevation_postfit_rms_m = float(auto_postfit_rms)
            auto_select_ok = (
                np.isfinite(auto_postfit_rms)
                and auto_postfit_rms
                > float(args.pr_auto_elevation_postfit_rms_m)
            )
            auto_guard_msg = ""
            if auto_select_ok:
                gated_wls_data = _filter_data_by_elevation(
                    auto_wls_data,
                    auto_positions,
                    float(args.pr_auto_elevation_deg),
                )
                gated_positions, gated_wls_ms = run_wls(gated_wls_data)
                auto_ms += gated_wls_ms
                base_quality = _wls_epoch_quality_metrics(auto_wls_data, auto_positions)
                gated_quality = _wls_epoch_quality_metrics(gated_wls_data, gated_positions)
                base_sat = float(np.nanmedian(base_quality["n_sat"]))
                gated_sat = float(np.nanmedian(gated_quality["n_sat"]))
                base_pdop = float(np.nanmedian(base_quality["pdop"]))
                gated_pdop = float(np.nanmedian(gated_quality["pdop"]))
                sat_fraction = (
                    gated_sat / base_sat
                    if np.isfinite(base_sat) and base_sat > 0.0
                    else float("nan")
                )
                pdop_ratio = (
                    gated_pdop / base_pdop
                    if np.isfinite(base_pdop) and base_pdop > 0.0
                    else float("nan")
                )
                auto_elevation_gated_sat_fraction = float(sat_fraction)
                auto_elevation_gated_pdop_ratio = float(pdop_ratio)
                min_sat_fraction = float(args.pr_auto_elevation_min_sat_fraction)
                max_pdop_ratio = float(args.pr_auto_elevation_max_pdop_ratio)
                sat_guard_ok = (
                    min_sat_fraction <= 0.0
                    or (np.isfinite(sat_fraction) and sat_fraction >= min_sat_fraction)
                )
                pdop_guard_ok = (
                    max_pdop_ratio <= 0.0
                    or (np.isfinite(pdop_ratio) and pdop_ratio <= max_pdop_ratio)
                )
                auto_select_ok = bool(sat_guard_ok and pdop_guard_ok)
                auto_guard_msg = (
                    f", gated_sats={gated_sat:.0f}/{base_sat:.0f} "
                    f"({sat_fraction:.2f}), gated_pdop={gated_pdop:.2f}/"
                    f"{base_pdop:.2f} ({pdop_ratio:.2f})"
                )
            if auto_select_ok:
                selected_pr_min_elevation_deg = float(args.pr_auto_elevation_deg)
                auto_elevation_selected = True
            print(
                f"  PR/WLS auto elevation: postfit_rms={auto_postfit_rms:.2f}m, "
                f"threshold={float(args.pr_auto_elevation_postfit_rms_m):.2f}m, "
                f"selected={selected_pr_min_elevation_deg:.1f} deg "
                f"{auto_guard_msg} "
                f"({auto_ms:.2f} ms/epoch probe)"
            )
        low_sat_threshold = int(args.pr_auto_robust_low_sat_threshold)
        low_sat_epochs = 0
        if bool(args.pr_auto_robust_profile) and low_sat_threshold > 0:
            sat_counts_for_profile = np.asarray(
                wls_data.get("satellite_counts", []),
                dtype=np.float64,
            )
            if sat_counts_for_profile.size > 0:
                low_sat_epochs = int(
                    np.count_nonzero(sat_counts_for_profile < low_sat_threshold)
                )
            selected_pr_auto_robust_profile = (
                selected_pr_min_elevation_deg > -90.0
                or low_sat_epochs >= int(args.pr_auto_robust_low_sat_epochs)
            )
            if selected_pr_auto_robust_profile:
                if selected_pr_prefit_gate_m <= 0.0:
                    selected_pr_prefit_gate_m = float(
                        args.pr_auto_robust_prefit_gate_m
                    )
                if abs(float(args.sigma_pr) - 8.0) < 1.0e-9:
                    selected_sigma_pr = float(args.pr_auto_robust_sigma_pr)
            print(
                "  PR auto robust profile: "
                f"selected={bool(selected_pr_auto_robust_profile)}, "
                f"low_sat_epochs={low_sat_epochs} below {low_sat_threshold}, "
                f"prefit={selected_pr_prefit_gate_m:.1f}m, "
                f"sigma_pr={selected_sigma_pr:.1f}"
            )
        if wls_atmosphere_model not in {"", "off"}:
            wls_seed_positions, wls_seed_ms = run_wls(wls_data)
            if selected_pr_min_elevation_deg > -90.0:
                wls_data = _filter_data_by_elevation(
                    wls_data,
                    wls_seed_positions,
                    selected_pr_min_elevation_deg,
                )
                print(
                    f"  PR/WLS elevation gate: >= {selected_pr_min_elevation_deg:.1f} deg "
                    f"(median sats={wls_data['n_satellites']})"
                )
                wls_seed_positions, wls_gate_ms = run_wls(wls_data)
                wls_seed_ms += wls_gate_ms
            wls_data = _apply_atmosphere_to_pseudoranges(
                wls_data,
                wls_seed_positions,
                model=wls_atmosphere_model,
                fixed_zenith_m=float(args.pr_slant_delay_zenith_m),
                model_scale=float(args.pr_atmosphere_scale),
                extra_zenith_m=float(args.pr_atmosphere_extra_zenith_m),
                iono_alpha=iono_alpha_tuple,
                iono_beta=iono_beta_tuple,
            )
            if float(selected_pr_prefit_gate_m) > 0.0:
                wls_data, prefit_kept, prefit_dropped = _filter_data_by_pr_prefit(
                    wls_data,
                    wls_seed_positions,
                    gate_m=float(selected_pr_prefit_gate_m),
                    clock_quantile=0.5,
                    min_sats=int(args.pr_prefit_gate_min_sats),
                    keep_best=int(args.pr_prefit_gate_keep_best),
                    per_system=bool(args.pr_prefit_per_system),
                )
                print(
                    f"  PR/WLS prefit gate: <= {float(selected_pr_prefit_gate_m):.1f} m "
                    f"(median sats={wls_data['n_satellites']}, "
                    f"dropped={prefit_dropped}/{prefit_kept + prefit_dropped})"
                )
            wls_positions, wls_ms = run_wls(wls_data, **spp_kwargs)
            wls_ms += wls_seed_ms
        else:
            wls_positions, wls_ms = run_wls(wls_data, **spp_kwargs)
            if selected_pr_min_elevation_deg > -90.0:
                wls_data = _filter_data_by_elevation(
                    wls_data,
                    wls_positions,
                    selected_pr_min_elevation_deg,
                )
                print(
                    f"  PR/WLS elevation gate: >= {selected_pr_min_elevation_deg:.1f} deg "
                    f"(median sats={wls_data['n_satellites']})"
                )
                wls_positions, wls_gate_ms = run_wls(wls_data, **spp_kwargs)
                wls_ms += wls_gate_ms
            if float(selected_pr_prefit_gate_m) > 0.0:
                wls_data, prefit_kept, prefit_dropped = _filter_data_by_pr_prefit(
                    wls_data,
                    wls_positions,
                    gate_m=float(selected_pr_prefit_gate_m),
                    clock_quantile=0.5,
                    min_sats=int(args.pr_prefit_gate_min_sats),
                    keep_best=int(args.pr_prefit_gate_keep_best),
                    per_system=bool(args.pr_prefit_per_system),
                )
                print(
                    f"  PR/WLS prefit gate: <= {float(selected_pr_prefit_gate_m):.1f} m "
                    f"(median sats={wls_data['n_satellites']}, "
                    f"dropped={prefit_dropped}/{prefit_kept + prefit_dropped})"
                )
                wls_positions, wls_prefit_ms = run_wls(wls_data, **spp_kwargs)
                wls_ms += wls_prefit_ms
        wls_quality = _wls_epoch_quality_metrics(wls_data, wls_positions)
        finite_wls_rms = wls_quality["postfit_rms_m"][
            np.isfinite(wls_quality["postfit_rms_m"])
        ]
        finite_wls_pdop = wls_quality["pdop"][np.isfinite(wls_quality["pdop"])]
        if finite_wls_rms.size > 0:
            wls_quality_msg = (
                f", postfit_rms med/p90="
                f"{np.median(finite_wls_rms):.2f}/{np.percentile(finite_wls_rms, 90):.2f}m"
            )
            if finite_wls_pdop.size > 0:
                wls_quality_msg += (
                    f", PDOP med/p90="
                    f"{np.median(finite_wls_pdop):.2f}/{np.percentile(finite_wls_pdop, 90):.2f}"
                )
        else:
            wls_quality_msg = ""
        print(f"  WLS init done ({wls_ms:.2f} ms/epoch{wls_quality_msg})", flush=True)

        ranker_lookup_run: dict[tuple[float, str], float] | None = None
        if args.rtkdiag_candidate_ranker_score_path:
            ranker_lookup_run = _ranker_lookup_for_run(
                args.rtkdiag_candidate_ranker_score_path, city, run,
            )
            print(
                f"  ranker predictions loaded for {city}/{run}: {len(ranker_lookup_run)} entries",
                flush=True,
            )

        for configured_variant in variants:
            user_select_mode = configured_variant.rtkdiag_candidate_select_mode
            variant = _apply_rtkdiag_run_index_policy(
                configured_variant,
                run=run,
                policy=str(args.rtkdiag_candidate_run_index_policy),
                city=city,
            )
            # Preserve explicit user select_mode through per-run policy overrides.
            # Phase 11ep+ presets unconditionally rewrite select_mode (residual /
            # temporal_n2_v10 etc.). Bypass when user passed something non-default.
            if user_select_mode != "residual" and variant.rtkdiag_candidate_select_mode != user_select_mode:
                variant = replace(variant, rtkdiag_candidate_select_mode=user_select_mode)
            variant = replace(
                variant,
                sigma_pr=selected_sigma_pr,
                pr_iono_alpha=iono_alpha_tuple,
                pr_iono_beta=iono_beta_tuple,
                pr_min_elevation_deg=selected_pr_min_elevation_deg,
                pr_prefit_gate_m=selected_pr_prefit_gate_m,
            )
            if str(args.rtkdiag_candidate_force_emit_mode).strip():
                variant = replace(
                    variant,
                    rtkdiag_candidate_emit_mode=str(args.rtkdiag_candidate_force_emit_mode),
                )
            print(f"  [{variant.method_label}] running ...", flush=True)
            need_dd_for_variant = variant.enable_dd_carrier_afv or variant.enable_dd_pr_ls_anchor or (
                variant.enable_rbpf_velocity_kf
                and variant.rbpf_kf_gate_min_dd_pairs is not None
            )
            dd_for_variant = dd_computer if need_dd_for_variant else None
            dd_pr_for_variant = (
                dd_pr_computer
                if (variant.enable_fgo_lambda or variant.enable_dd_pr_ls_anchor)
                else None
            )
            hybrid_for_variant = hybrid_pos_run if variant.enable_hybrid_pu else None
            # Phase 9b also consumes the hybrid velocity guide -- it derives
            # the IMU anchor yaw from the rover's course-over-ground at the
            # most recent Status=4 anchor. The PF predict step only uses it
            # when ``enable_hybrid_velocity_guide=True``; for IMU-TC alone we
            # pass the dict but the predict path stays on its constant-velocity
            # default (variant.enable_hybrid_velocity_guide=False).
            hybrid_v_for_variant = (
                hybrid_velocity_run
                if (
                    variant.enable_hybrid_velocity_guide
                    or variant.enable_imu_tc
                    or variant.enable_ins_tc
                )
                else None
            )
            hybrid_status_for_variant = (
                hybrid_status_run
                if (
                    variant.enable_fgo_lambda
                    or variant.enable_tdcp_smoother
                    or variant.enable_zupt
                    or variant.enable_imu_tc
                    or variant.enable_ins_tc
                    or variant.enable_pr_gmm
                    or variant.pr_skip_statuses
                    or (
                        float(variant.pr_prefit_gate_m) > 0.0
                        and str(variant.pr_prefit_ref).strip().lower() == "hybrid"
                    )
                )
                else None
            )
            imu_for_variant = (
                imu_run
                if (variant.enable_zupt or variant.enable_imu_tc or variant.enable_ins_tc)
                else None
            )
            rtkdiag_candidates_for_variant = (
                rtkdiag_candidates_run_policy if variant.enable_rtkdiag_pf_rescue else None
            )
            rtkdiag_candidate_labels_for_variant = (
                [label for label, _, _ in rtkdiag_candidates_for_variant]
                if rtkdiag_candidates_for_variant is not None
                else rtkdiag_candidate_labels
            )
            positions, ms_per_epoch, pr_obs_stats, reservoir_stein_stats, dd_stats, gate_stats, hybrid_stats, rtkdiag_pf_stats, fgo_stats, tdcp_stats, low_sat_bridge_stats, zupt_stats, imu_tc_stats, ins_tc_stats, variant_internal_diag_rows = _run_ctrbpf_on_segment(
                data, wls_positions, variant,
                wls_quality=wls_quality,
                dd_computer=dd_for_variant,
                dd_pr_computer=dd_pr_for_variant,
                hybrid_pos=hybrid_for_variant,
                hybrid_velocity=hybrid_v_for_variant,
                hybrid_status=hybrid_status_for_variant,
                reference_pos=reference_pos_run,
                rtkdiag_candidates=rtkdiag_candidates_for_variant,
                imu=imu_for_variant,
                collect_internal_diagnostics=bool(args.write_internal_diagnostics),
                ranker_score_lookup=ranker_lookup_run,
                ranker_stickiness=float(args.rtkdiag_candidate_ranker_stickiness),
            )
            if args.write_internal_diagnostics:
                for diag in variant_internal_diag_rows:
                    diag["city"] = city
                    diag["run"] = run
                    diag["method"] = variant.method_label
                internal_diag_rows.extend(variant_internal_diag_rows)
            honest_est = np.asarray(
                [p for p in _aligned_positions(full_ref, data["times"], positions)],
                dtype=np.float64,
            )
            honest_ref = np.array([t for _, t in full_ref], dtype=np.float64)
            score = score_ppc2024(honest_est, honest_ref)
            score_1m = score_ppc2024(honest_est, honest_ref, threshold_m=1.0)
            score_3m = score_ppc2024(honest_est, honest_ref, threshold_m=3.0)
            segment_est: list[np.ndarray] = []
            segment_ref: list[np.ndarray] = []
            for t_obs, pos_obs in zip(data["times"], positions, strict=True):
                ref_obs = reference_pos_run.get(round(float(t_obs), 1))
                if (
                    ref_obs is not None
                    and np.all(np.isfinite(ref_obs[:3]))
                    and np.all(np.isfinite(pos_obs[:3]))
                ):
                    segment_est.append(np.asarray(pos_obs[:3], dtype=np.float64))
                    segment_ref.append(np.asarray(ref_obs[:3], dtype=np.float64))
            segment_est_arr = np.asarray(segment_est, dtype=np.float64)
            segment_ref_arr = np.asarray(segment_ref, dtype=np.float64)
            segment_score = (
                score_ppc2024(segment_est_arr, segment_ref_arr)
                if segment_est
                else None
            )
            segment_score_1m = (
                score_ppc2024(segment_est_arr, segment_ref_arr, threshold_m=1.0)
                if segment_est
                else None
            )
            segment_score_3m = (
                score_ppc2024(segment_est_arr, segment_ref_arr, threshold_m=3.0)
                if segment_est
                else None
            )
            pos_path = args.pos_dir / f"{city}_{run}_{variant.method_label}.pos"
            _write_pos_file(pos_path, np.asarray(data["times"]), positions)

            variant_gate_active = variant.enable_rbpf_velocity_kf and (
                variant.rbpf_kf_gate_min_dd_pairs is not None
                or variant.rbpf_kf_gate_min_ess_ratio is not None
                or variant.rbpf_kf_gate_max_spread_m is not None
                or float(variant.rbpf_kf_gate_max_doppler_wls_rms_mps) > 0.0
                or float(variant.rbpf_kf_gate_max_doppler_wls_speed_mps) > 0.0
            )
            row = {
                "city": city,
                "run": run,
                "method": variant.method_label,
                "n_particles": variant.n_particles,
                "sigma_pr": float(variant.sigma_pr),
                "start_epoch": int(args.start_epoch),
                "max_epochs": int(args.max_epochs) if args.max_epochs is not None else "",
                "n_ref_epochs": len(full_ref),
                "n_pf_epochs": int(data["n_epochs"]),
                "coverage_pct": float(100.0 * n_emit / max(len(full_ref), 1)),
                "honest_ppc_pct": float(score.score_pct),
                "honest_pass_m": float(score.pass_distance_m),
                "honest_total_m": float(score.total_distance_m),
                "honest_ppc_1m_pct": float(score_1m.score_pct),
                "honest_pass_1m_m": float(score_1m.pass_distance_m),
                "honest_ppc_3m_pct": float(score_3m.score_pct),
                "honest_pass_3m_m": float(score_3m.pass_distance_m),
                "segment_ppc_pct": float(segment_score.score_pct) if segment_score is not None else "",
                "segment_pass_m": float(segment_score.pass_distance_m) if segment_score is not None else "",
                "segment_total_m": float(segment_score.total_distance_m) if segment_score is not None else "",
                "segment_ppc_1m_pct": float(segment_score_1m.score_pct) if segment_score_1m is not None else "",
                "segment_pass_1m_m": float(segment_score_1m.pass_distance_m) if segment_score_1m is not None else "",
                "segment_ppc_3m_pct": float(segment_score_3m.score_pct) if segment_score_3m is not None else "",
                "segment_pass_3m_m": float(segment_score_3m.pass_distance_m) if segment_score_3m is not None else "",
                "segment_fail_3m_epochs": (
                    int(np.count_nonzero(np.isfinite(segment_score_3m.errors_3d) & (segment_score_3m.errors_3d > 3.0)))
                    if segment_score_3m is not None
                    else 0
                ),
                "segment_epoch_pass_pct": float(segment_score.epoch_pass_pct) if segment_score is not None else "",
                "segment_n_epochs": int(segment_score.n_epochs) if segment_score is not None else 0,
                "ms_per_epoch": float(ms_per_epoch),
                "rbpf_velocity_kf": int(variant.enable_rbpf_velocity_kf),
                "position_update": int(variant.enable_position_update),
                "defer_epoch_resample": int(variant.defer_epoch_resample),
                "deferred_resample_epochs": int(pr_obs_stats.deferred_resample_epochs),
                "reservoir_stein": int(variant.enable_reservoir_stein),
                "reservoir_stein_epochs": int(reservoir_stein_stats.epochs_applied),
                "reservoir_stein_size": int(variant.reservoir_stein_size),
                "reservoir_stein_avg_selected": (
                    float(reservoir_stein_stats.reservoir_size_sum)
                    / max(int(reservoir_stein_stats.epochs_applied), 1)
                ),
                "reservoir_stein_avg_ess_before": (
                    float(reservoir_stein_stats.ess_before_sum)
                    / max(int(reservoir_stein_stats.epochs_applied), 1)
                ),
                "reservoir_stein_avg_bandwidth": (
                    float(reservoir_stein_stats.bandwidth_sum)
                    / max(int(reservoir_stein_stats.epochs_applied), 1)
                ),
                "pr_weight_mode": str(variant.pr_weight_mode),
                "pr_weight_ref_cn0": float(variant.pr_weight_ref_cn0),
                "pr_weight_min": float(variant.pr_weight_min),
                "pr_weight_max": float(variant.pr_weight_max),
                "pr_systems": ",".join(str(s) for s in variant.pr_systems),
                "pr_min_elevation_deg": float(variant.pr_min_elevation_deg),
                "pr_auto_elevation_gate": int(bool(args.pr_auto_elevation_gate)),
                "pr_auto_elevation_selected": int(bool(auto_elevation_selected)),
                "pr_auto_elevation_postfit_rms_m": float(
                    auto_elevation_postfit_rms_m
                ),
                "pr_auto_elevation_min_sat_fraction": float(
                    args.pr_auto_elevation_min_sat_fraction
                ),
                "pr_auto_elevation_gated_sat_fraction": float(
                    auto_elevation_gated_sat_fraction
                ),
                "pr_auto_elevation_max_pdop_ratio": float(
                    args.pr_auto_elevation_max_pdop_ratio
                ),
                "pr_auto_elevation_gated_pdop_ratio": float(
                    auto_elevation_gated_pdop_ratio
                ),
                "pr_auto_robust_profile": int(bool(args.pr_auto_robust_profile)),
                "pr_auto_robust_selected": int(
                    bool(selected_pr_auto_robust_profile)
                ),
                "pr_auto_robust_low_sat_epochs": int(low_sat_epochs),
                "pr_auto_robust_low_sat_threshold": int(low_sat_threshold),
                "pr_atmosphere_model": str(variant.pr_atmosphere_model),
                "pr_atmosphere_scale": float(variant.pr_atmosphere_scale),
                "pr_atmosphere_extra_zenith_m": float(
                    variant.pr_atmosphere_extra_zenith_m
                ),
                "pr_slant_delay_zenith_m": float(variant.pr_slant_delay_zenith_m),
                "pr_prefit_gate_m": float(variant.pr_prefit_gate_m),
                "pr_prefit_gate_min_sats": int(variant.pr_prefit_gate_min_sats),
                "pr_prefit_gate_keep_best": int(variant.pr_prefit_gate_keep_best),
                "pr_prefit_ref": str(variant.pr_prefit_ref),
                "pr_prefit_per_system": int(variant.pr_prefit_per_system),
                "pr_prefit_epochs": int(pr_obs_stats.prefit_epochs),
                "pr_prefit_sats_kept": int(pr_obs_stats.prefit_sats_kept),
                "pr_prefit_sats_dropped": int(pr_obs_stats.prefit_sats_dropped),
                "pr_skip_statuses": ",".join(str(int(s)) for s in variant.pr_skip_statuses),
                "pr_skipped_epochs": int(pr_obs_stats.epochs_skipped),
                "pr_gmm": int(variant.enable_pr_gmm),
                "pr_gmm_epochs": int(pr_obs_stats.epochs_gmm),
                "pr_gaussian_epochs": int(pr_obs_stats.epochs_gaussian),
                "pr_gmm_w_los": float(variant.pr_gmm_w_los),
                "pr_gmm_mu_nlos_m": float(variant.pr_gmm_mu_nlos_m),
                "pr_gmm_sigma_nlos_m": float(variant.pr_gmm_sigma_nlos_m),
                "pr_gmm_hybrid_loose_sigma_m": float(variant.pr_gmm_hybrid_loose_sigma_m),
                "pr_gmm_clock_quantile": float(variant.pr_gmm_clock_quantile),
                "doppler_systems": ",".join(str(s) for s in variant.doppler_systems),
                "doppler_prefit_gate_mps": float(variant.doppler_prefit_gate_mps),
                "doppler_prefit_gate_min_sats": int(variant.doppler_prefit_gate_min_sats),
                "dd_carrier_afv": int(variant.enable_dd_carrier_afv),
                "dd_min_elevation_deg": float(variant.dd_min_elevation_deg),
                "dd_min_snr": float(variant.dd_min_snr),
                "dd_keep_best": int(variant.dd_keep_best),
                "dd_pr_pair_residual_max_m": float(variant.dd_pr_pair_residual_max_m),
                "dd_pr_epoch_median_residual_max_m": float(
                    variant.dd_pr_epoch_median_residual_max_m
                ),
                "dd_pr_gate_min_pairs": int(variant.dd_pr_gate_min_pairs),
                "dd_pr_ls_anchor": int(variant.enable_dd_pr_ls_anchor),
                "dd_pr_ls_anchor_mode": str(variant.dd_pr_ls_anchor_mode),
                "dd_pr_ls_anchor_attempted": int(
                    fgo_stats.dd_pr_ls_anchor_attempted
                ),
                "dd_pr_ls_anchor_accepted": int(
                    fgo_stats.dd_pr_ls_anchor_accepted
                ),
                "dd_pr_ls_anchor_status_skipped": int(
                    fgo_stats.dd_pr_ls_anchor_status_skipped
                ),
                "dd_pr_ls_anchor_rejected_postfit": int(
                    fgo_stats.dd_pr_ls_anchor_rejected_postfit
                ),
                "dd_pr_ls_anchor_rejected_solve": int(
                    fgo_stats.dd_pr_ls_anchor_rejected_solve
                ),
                "dd_pr_ls_anchor_avg_shift_m": (
                    float(fgo_stats.dd_pr_ls_anchor_shift_sum_m)
                    / max(int(fgo_stats.dd_pr_ls_anchor_accepted), 1)
                ),
                "dd_pr_ls_anchor_avg_postfit_m": (
                    float(fgo_stats.dd_pr_ls_anchor_postfit_sum_m)
                    / max(int(fgo_stats.dd_pr_ls_anchor_accepted), 1)
                ),
                "dd_pr_ls_anchor_gt_count": int(fgo_stats.dd_pr_ls_anchor_gt_count),
                "dd_pr_ls_anchor_avg_gt_error_m": (
                    float(fgo_stats.dd_pr_ls_anchor_gt_error_sum_m)
                    / max(int(fgo_stats.dd_pr_ls_anchor_gt_count), 1)
                ),
                "dd_pr_ls_anchor_avg_seed_error_m": (
                    float(fgo_stats.dd_pr_ls_anchor_seed_error_sum_m)
                    / max(int(fgo_stats.dd_pr_ls_anchor_gt_count), 1)
                ),
                "dd_pr_ls_anchor_improved_pct": (
                    100.0
                    * float(fgo_stats.dd_pr_ls_anchor_improved)
                    / max(int(fgo_stats.dd_pr_ls_anchor_gt_count), 1)
                ),
                "dd_pr_ls_anchor_pass_05m_pct": (
                    100.0
                    * float(fgo_stats.dd_pr_ls_anchor_pass_05m)
                    / max(int(fgo_stats.dd_pr_ls_anchor_gt_count), 1)
                ),
                "dd_pr_ls_anchor_pass_5m_pct": (
                    100.0
                    * float(fgo_stats.dd_pr_ls_anchor_pass_5m)
                    / max(int(fgo_stats.dd_pr_ls_anchor_gt_count), 1)
                ),
                "dd_epochs_attempted": int(dd_stats.epochs_attempted),
                "dd_epochs_applied": int(dd_stats.epochs_applied),
                "dd_pairs_total": int(dd_stats.pairs_total),
                "rbpf_kf_gate_active": int(variant_gate_active),
                "rbpf_kf_gate_min_dd_pairs": (
                    -1 if variant.rbpf_kf_gate_min_dd_pairs is None
                    else int(variant.rbpf_kf_gate_min_dd_pairs)
                ),
                "rbpf_kf_gate_min_ess_ratio": (
                    -1.0 if variant.rbpf_kf_gate_min_ess_ratio is None
                    else float(variant.rbpf_kf_gate_min_ess_ratio)
                ),
                "rbpf_kf_gate_max_spread_m": (
                    -1.0 if variant.rbpf_kf_gate_max_spread_m is None
                    else float(variant.rbpf_kf_gate_max_spread_m)
                ),
                "rbpf_kf_gate_max_doppler_wls_rms_mps": float(
                    variant.rbpf_kf_gate_max_doppler_wls_rms_mps
                ),
                "rbpf_kf_gate_max_doppler_wls_speed_mps": float(
                    variant.rbpf_kf_gate_max_doppler_wls_speed_mps
                ),
                "rbpf_kf_attempted": int(gate_stats.epochs_attempted),
                "rbpf_kf_applied": int(gate_stats.epochs_applied),
                "rbpf_kf_skip_min_dd_pairs": int(gate_stats.skipped_min_dd_pairs),
                "rbpf_kf_skip_min_ess_ratio": int(gate_stats.skipped_min_ess_ratio),
                "rbpf_kf_skip_max_spread": int(gate_stats.skipped_max_spread),
                "rbpf_kf_skip_doppler_wls_rms": int(
                    gate_stats.skipped_doppler_wls_rms
                ),
                "rbpf_kf_skip_doppler_wls_speed": int(
                    gate_stats.skipped_doppler_wls_speed
                ),
                "hybrid_pu": int(variant.enable_hybrid_pu),
                "hybrid_sigma_m": float(variant.hybrid_sigma_m),
                "position_update_min_epoch": int(
                    variant.position_update_min_epoch
                ),
                "position_update_min_pr_sats": int(
                    variant.position_update_min_pr_sats
                ),
                "position_update_max_wls_rms_m": float(
                    variant.position_update_max_wls_rms_m
                ),
                "position_update_max_wls_pdop": float(
                    variant.position_update_max_wls_pdop
                ),
                "position_update_max_wls_to_pf_m": float(
                    variant.position_update_max_wls_to_pf_m
                ),
                "hybrid_recenter_max_shift_m": float(variant.hybrid_recenter_max_shift_m),
                "hybrid_emit_pf_statuses": ",".join(
                    str(int(s)) for s in variant.hybrid_emit_pf_statuses
                ),
                "hybrid_attempted": int(hybrid_stats.epochs_attempted),
                "hybrid_applied": int(hybrid_stats.epochs_applied),
                "hybrid_lookup_missing": int(hybrid_stats.epochs_lookup_missing),
                "hybrid_recenter_applied": int(hybrid_stats.recenter_applied),
                "hybrid_recenter_skipped": int(hybrid_stats.recenter_skipped),
                "hybrid_recenter_avg_shift_m": (
                    float(hybrid_stats.recenter_shift_sum_m)
                    / max(int(hybrid_stats.recenter_applied), 1)
                ),
                "rtkdiag_pf": int(variant.enable_rtkdiag_pf_rescue),
                "rtkdiag_candidate_sigma_m": float(variant.rtkdiag_candidate_sigma_m),
                "rtkdiag_candidate_ratio_min": float(variant.rtkdiag_candidate_ratio_min),
                "rtkdiag_candidate_residual_rms_max": float(
                    variant.rtkdiag_candidate_residual_rms_max
                ),
                "rtkdiag_candidate_max_to_hybrid_m": float(
                    variant.rtkdiag_candidate_max_to_hybrid_m
                ),
                "rtkdiag_candidate_emit_max_diff_m": float(
                    variant.rtkdiag_candidate_emit_max_diff_m
                ),
                "rtkdiag_candidate_recenter_max_shift_m": float(
                    variant.rtkdiag_candidate_recenter_max_shift_m
                ),
                "rtkdiag_candidate_soft_top_k": int(
                    variant.rtkdiag_candidate_soft_top_k
                ),
                "rtkdiag_candidate_soft_weight_eps": float(
                    variant.rtkdiag_candidate_soft_weight_eps
                ),
                "rtkdiag_candidate_proposal_cloud": int(
                    variant.rtkdiag_candidate_proposal_cloud
                ),
                "rtkdiag_candidate_proposal_spread_m": float(
                    variant.rtkdiag_candidate_proposal_spread_m
                ),
                "rtkdiag_candidate_select_mode": str(variant.rtkdiag_candidate_select_mode),
                "rtkdiag_candidate_emit_mode": str(variant.rtkdiag_candidate_emit_mode),
                "rtkdiag_candidate_min_epoch": int(
                    variant.rtkdiag_candidate_min_epoch
                ),
                "rtkdiag_candidate_require_any_diag_fields": ",".join(
                    variant.rtkdiag_candidate_require_any_diag_fields
                ),
                "rtkdiag_candidate_require_all_diag_fields": ",".join(
                    variant.rtkdiag_candidate_require_all_diag_fields
                ),
                "rtkdiag_candidate_min_diag_fields": str(
                    variant.rtkdiag_candidate_min_diag_fields
                ),
                "rtkdiag_candidate_max_diag_fields": str(
                    variant.rtkdiag_candidate_max_diag_fields
                ),
                "rtkdiag_candidate_fallback_mode": str(
                    variant.rtkdiag_candidate_fallback_mode
                ),
                "rtkdiag_candidate_fallback_max_wls_rms_m": float(
                    variant.rtkdiag_candidate_fallback_max_wls_rms_m
                ),
                "rtkdiag_candidate_fallback_max_wls_pdop": float(
                    variant.rtkdiag_candidate_fallback_max_wls_pdop
                ),
                "rtkdiag_candidate_fallback_max_wls_to_pf_m": float(
                    variant.rtkdiag_candidate_fallback_max_wls_to_pf_m
                ),
                "rtkdiag_candidate_fallback_max_hold_age_s": float(
                    variant.rtkdiag_candidate_fallback_max_hold_age_s
                ),
                "rtkdiag_candidate_float_labels": ",".join(
                    variant.rtkdiag_candidate_float_labels
                ),
                "rtkdiag_candidate_float_residual_rms_max": float(
                    variant.rtkdiag_candidate_float_residual_rms_max
                ),
                "rtkdiag_candidate_float_abs_max": float(
                    variant.rtkdiag_candidate_float_abs_max
                ),
                "rtkdiag_candidate_float_min_sats": int(
                    variant.rtkdiag_candidate_float_min_sats
                ),
                "rtkdiag_candidate_status5_labels": ",".join(
                    variant.rtkdiag_candidate_status5_labels
                ),
                "rtkdiag_candidate_status5_tow_windows": str(
                    variant.rtkdiag_candidate_status5_tow_windows
                ),
                "rtkdiag_candidate_status5_max_dt_s": float(
                    variant.rtkdiag_candidate_status5_max_dt_s
                ),
                "rtkdiag_candidate_status5_residual_rms_max": float(
                    variant.rtkdiag_candidate_status5_residual_rms_max
                ),
                "rtkdiag_candidate_status5_min_sats": int(
                    variant.rtkdiag_candidate_status5_min_sats
                ),
                "rtkdiag_candidate_run_index_policy": str(
                    args.rtkdiag_candidate_run_index_policy
                ),
                "rtkdiag_candidate_block_labels": ",".join(
                    sorted(rtkdiag_candidate_block_labels)
                ),
                "rtkdiag_candidate_block_labels_by_run": str(
                    args.rtkdiag_candidate_block_labels_by_run
                ),
                "rtkdiag_candidate_labels": ",".join(rtkdiag_candidate_labels_for_variant),
                "rtkdiag_pf_evaluated": int(rtkdiag_pf_stats.epochs_evaluated),
                "rtkdiag_pf_gate_pass": int(rtkdiag_pf_stats.gate_pass),
                "rtkdiag_pf_candidate_options_total": int(
                    rtkdiag_pf_stats.candidate_options_total
                ),
                "rtkdiag_pf_candidate_missing": int(rtkdiag_pf_stats.candidate_missing),
                "rtkdiag_pf_pu_applied": int(rtkdiag_pf_stats.pu_applied),
                "rtkdiag_pf_recenter_applied": int(rtkdiag_pf_stats.recenter_applied),
                "rtkdiag_pf_recenter_skipped": int(rtkdiag_pf_stats.recenter_skipped),
                "rtkdiag_pf_emit_pf_estimate": int(rtkdiag_pf_stats.emit_pf_estimate),
                "rtkdiag_pf_emit_candidate": int(rtkdiag_pf_stats.emit_candidate),
                "rtkdiag_pf_skipped_min_epoch": int(
                    rtkdiag_pf_stats.skipped_min_epoch
                ),
                "rtkdiag_pf_skipped_diag_policy": int(
                    rtkdiag_pf_stats.skipped_diag_policy
                ),
                "rtkdiag_pf_skipped_hybrid_distance": int(
                    rtkdiag_pf_stats.skipped_hybrid_distance
                ),
                "rtkdiag_pf_fallback_pf": int(rtkdiag_pf_stats.fallback_pf),
                "rtkdiag_pf_fallback_wls": int(rtkdiag_pf_stats.fallback_wls),
                "rtkdiag_pf_fallback_hybrid": int(
                    rtkdiag_pf_stats.fallback_hybrid
                ),
                "rtkdiag_pf_fallback_last_good": int(
                    rtkdiag_pf_stats.fallback_last_good
                ),
                "rtkdiag_pf_emit_skipped_pf_drift": int(
                    rtkdiag_pf_stats.emit_skipped_pf_drift
                ),
                "rtkdiag_pf_selected_counts": ",".join(
                    f"{k}:{v}" for k, v in sorted(rtkdiag_pf_stats.selected_counts.items())
                ),
                "fgo_lambda": int(variant.enable_fgo_lambda),
                "fgo_dd_pr_sigma_m": float(variant.fgo_dd_pr_sigma_m),
                "fgo_lambda_max_epoch_gap": int(variant.fgo_lambda_max_epoch_gap),
                "fgo_apply_fixed_epochs_only": int(
                    variant.fgo_apply_fixed_epochs_only
                ),
                "fgo_windows_attempted": int(fgo_stats.windows_attempted),
                "fgo_windows_solved": int(fgo_stats.windows_solved),
                "fgo_windows_applied": int(fgo_stats.windows_applied),
                "fgo_n_fixed_total": int(fgo_stats.n_fixed_total),
                "fgo_n_fixed_observations_total": int(
                    fgo_stats.n_fixed_observations_total
                ),
                "fgo_epochs_replaced": int(fgo_stats.epochs_replaced),
                "fgo_lambda_tracks_total": int(fgo_stats.lambda_tracks_total),
                "fgo_lambda_segments_total": int(fgo_stats.lambda_segments_total),
                "fgo_lambda_candidates_total": int(fgo_stats.lambda_candidates_total),
                "fgo_lambda_ratio_rejected_total": int(
                    fgo_stats.lambda_ratio_rejected_total
                ),
                "fgo_lambda_best_ratio": float(fgo_stats.lambda_best_ratio),
                "fgo_lambda_ratio_median_avg": (
                    float(fgo_stats.lambda_ratio_median_sum)
                    / max(int(fgo_stats.lambda_diag_windows), 1)
                ),
                "fgo_lambda_ratio_p90_avg": (
                    float(fgo_stats.lambda_ratio_p90_sum)
                    / max(int(fgo_stats.lambda_diag_windows), 1)
                ),
                "fgo_lambda_segment_n_epochs_median_avg": (
                    float(fgo_stats.lambda_segment_n_epochs_median_sum)
                    / max(int(fgo_stats.lambda_diag_windows), 1)
                ),
                "fgo_lambda_segment_n_epochs_max": int(
                    fgo_stats.lambda_segment_n_epochs_max
                ),
                "fgo_lambda_segment_variance_median_avg": (
                    float(fgo_stats.lambda_segment_variance_median_sum)
                    / max(int(fgo_stats.lambda_diag_windows), 1)
                ),
                "fgo_lambda_segment_abs_frac_median_avg": (
                    float(fgo_stats.lambda_segment_abs_frac_median_sum)
                    / max(int(fgo_stats.lambda_diag_windows), 1)
                ),
                "fgo_lambda_segment_abs_frac_p90_avg": (
                    float(fgo_stats.lambda_segment_abs_frac_p90_sum)
                    / max(int(fgo_stats.lambda_diag_windows), 1)
                ),
                "fgo_postfit_fixed_count_total": int(
                    fgo_stats.postfit_fixed_count_total
                ),
                "fgo_postfit_fixed_abs_cycles_median_avg": (
                    float(fgo_stats.postfit_fixed_abs_cycles_median_sum)
                    / max(int(fgo_stats.windows_solved), 1)
                ),
                "fgo_postfit_fixed_abs_cycles_p90_avg": (
                    float(fgo_stats.postfit_fixed_abs_cycles_p90_sum)
                    / max(int(fgo_stats.windows_solved), 1)
                ),
                "fgo_postfit_float_count_total": int(
                    fgo_stats.postfit_float_count_total
                ),
                "fgo_postfit_float_afv_abs_cycles_median_avg": (
                    float(fgo_stats.postfit_float_afv_abs_cycles_median_sum)
                    / max(int(fgo_stats.windows_solved), 1)
                ),
                "fgo_postfit_float_afv_abs_cycles_p90_avg": (
                    float(fgo_stats.postfit_float_afv_abs_cycles_p90_sum)
                    / max(int(fgo_stats.windows_solved), 1)
                ),
                "fgo_postfit_dd_pr_count_total": int(
                    fgo_stats.postfit_dd_pr_count_total
                ),
                "fgo_postfit_dd_pr_abs_m_median_avg": (
                    float(fgo_stats.postfit_dd_pr_abs_m_median_sum)
                    / max(int(fgo_stats.windows_solved), 1)
                ),
                "fgo_postfit_dd_pr_abs_m_p90_avg": (
                    float(fgo_stats.postfit_dd_pr_abs_m_p90_sum)
                    / max(int(fgo_stats.windows_solved), 1)
                ),
                "ct_spline_motion_prior": int(variant.enable_ct_spline_motion_prior),
                "ct_spline_smoothing_m": float(variant.ct_spline_smoothing_m),
                "ct_motion_sigma_m": float(variant.ct_motion_sigma_m),
                "tdcp_smoother": int(variant.enable_tdcp_smoother),
                "tdcp_pairs_attempted": int(tdcp_stats.pairs_attempted),
                "tdcp_pairs_accepted": int(tdcp_stats.pairs_accepted),
                "tdcp_pairs_rejected_min_sats": int(tdcp_stats.pairs_rejected_min_sats),
                "tdcp_pairs_rejected_postfit": int(tdcp_stats.pairs_rejected_postfit),
                "low_sat_bridge": int(variant.enable_low_sat_bridge),
                "low_sat_bridge_min_pr_sats": int(variant.low_sat_bridge_min_pr_sats),
                "low_sat_bridge_spans": int(low_sat_bridge_stats.spans_applied),
                "low_sat_bridge_epochs": int(low_sat_bridge_stats.epochs_rewritten),
                "low_sat_bridge_startup_spans": int(
                    low_sat_bridge_stats.startup_spans_applied
                ),
                "low_sat_bridge_startup_epochs": int(
                    low_sat_bridge_stats.startup_epochs_rewritten
                ),
                "zupt": int(variant.enable_zupt),
                "zupt_evaluated": int(zupt_stats.epochs_evaluated),
                "zupt_static": int(zupt_stats.epochs_static),
                "zupt_rewritten": int(zupt_stats.epochs_rewritten),
                "zupt_no_imu": int(zupt_stats.epochs_no_imu),
                "imu_tc": int(variant.enable_imu_tc),
                "imu_tc_evaluated": int(imu_tc_stats.epochs_evaluated),
                "imu_tc_anchor_resets": int(imu_tc_stats.anchor_resets),
                "imu_tc_anchor_resets_static": int(imu_tc_stats.anchor_resets_static),
                "imu_tc_pu_applied": int(imu_tc_stats.pu_applied),
                "imu_tc_pu_skipped_no_imu": int(imu_tc_stats.pu_skipped_no_imu),
                "imu_tc_pu_skipped_no_anchor": int(imu_tc_stats.pu_skipped_no_anchor),
                "imu_tc_pu_skipped_dr_too_long": int(imu_tc_stats.pu_skipped_dr_too_long),
                "imu_tc_pu_skipped_disagreement": int(imu_tc_stats.pu_skipped_disagreement),
                "imu_tc_emit_pf_estimate": int(imu_tc_stats.emit_pf_estimate),
                "imu_tc_emit_skipped_pf_drift": int(imu_tc_stats.emit_skipped_pf_drift),
                "imu_tc_avg_dr_seconds": float(imu_tc_stats.avg_dr_seconds),
                "ins_tc": int(variant.enable_ins_tc),
                "ins_tc_aligned_at_epoch": int(ins_tc_stats.aligned_at_epoch),
                "ins_tc_yaw_initialized_at_epoch": int(ins_tc_stats.yaw_initialized_at_epoch),
                "ins_tc_evaluated": int(ins_tc_stats.epochs_evaluated),
                "ins_tc_pu_applied": int(ins_tc_stats.pu_applied),
                "ins_tc_pu_skipped_not_aligned": int(ins_tc_stats.pu_skipped_not_aligned),
                "ins_tc_pu_skipped_no_yaw": int(ins_tc_stats.pu_skipped_no_yaw),
                "ins_tc_pu_skipped_dr_too_long": int(ins_tc_stats.pu_skipped_dr_too_long),
                "ins_tc_pu_skipped_disagreement": int(ins_tc_stats.pu_skipped_disagreement),
                "ins_tc_particle_imu_initialized": int(ins_tc_stats.particle_imu_initialized),
                "ins_tc_particle_imu_predict_used": int(ins_tc_stats.particle_imu_predict_used),
                "ins_tc_recenter_applied": int(ins_tc_stats.recenter_applied),
                "ins_tc_recenter_skipped": int(ins_tc_stats.recenter_skipped),
                "ins_tc_recenter_avg_shift_m": (
                    float(ins_tc_stats.recenter_shift_sum_m) / max(int(ins_tc_stats.recenter_applied), 1)
                ),
                "ins_tc_motion_predict_used": int(ins_tc_stats.motion_predict_used),
                "ins_tc_obs_status_4_used": int(ins_tc_stats.obs_status_4_used),
                "ins_tc_obs_status_3_used": int(ins_tc_stats.obs_status_3_used),
                "ins_tc_emit_pf_estimate": int(ins_tc_stats.emit_pf_estimate),
                "ins_tc_emit_skipped_pf_drift": int(ins_tc_stats.emit_skipped_pf_drift),
                "ins_tc_final_acc_bias_norm": float(ins_tc_stats.final_acc_bias_norm),
                "ins_tc_final_gyro_bias_norm_dps": float(ins_tc_stats.final_gyro_bias_norm_dps),
                "ins_tc_final_pos_sigma_m": float(ins_tc_stats.final_pos_sigma_m),
            }
            rows.append(row)
            agg_pass[variant.method_label] += row["honest_pass_m"]
            agg_total[variant.method_label] += row["honest_total_m"]
            run_total = float(row["honest_total_m"])
            if run_total > 0.0:
                per_run_pcts[variant.method_label].append(
                    100.0 * float(row["honest_pass_m"]) / run_total
                )
            dd_msg = ""
            if variant.enable_dd_carrier_afv:
                applied = dd_stats.epochs_applied
                attempted = max(dd_stats.epochs_attempted, 1)
                avg_pairs = dd_stats.pairs_total / max(applied, 1)
                dd_msg = (
                    f", DD applied {applied}/{dd_stats.epochs_attempted} "
                    f"({100.0 * applied / attempted:.1f}%, avg pairs={avg_pairs:.1f})"
                )
            pr_msg = ""
            if variant.enable_pr_gmm:
                pr_total = max(pr_obs_stats.epochs_gmm + pr_obs_stats.epochs_gaussian, 1)
                pr_msg = (
                    f", PR-GMM {pr_obs_stats.epochs_gmm}/{pr_total} "
                    f"(w_los={variant.pr_gmm_w_los:.2f}, "
                    f"mu={variant.pr_gmm_mu_nlos_m:.1f}m, "
                    f"hloose={variant.pr_gmm_hybrid_loose_sigma_m:.1f}m, "
                    f"cbq={variant.pr_gmm_clock_quantile:.2f})"
                )
            defer_msg = ""
            if variant.defer_epoch_resample:
                defer_msg = (
                    f", defer-resample {pr_obs_stats.deferred_resample_epochs}"
                )
            if variant.enable_reservoir_stein:
                avg_ess = (
                    reservoir_stein_stats.ess_before_sum
                    / max(reservoir_stein_stats.epochs_applied, 1)
                )
                defer_msg = (
                    f", reservoir-stein {reservoir_stein_stats.epochs_applied}ep "
                    f"(avg ESS={avg_ess:.0f})"
                )
            pr_weight_msg = ""
            if str(variant.pr_weight_mode) != "raw":
                pr_weight_msg = f", prw={variant.pr_weight_mode}"
            prefit_msg = ""
            if float(variant.pr_prefit_gate_m) > 0.0:
                avg_drop = (
                    pr_obs_stats.prefit_sats_dropped
                    / max(pr_obs_stats.prefit_epochs, 1)
                )
                prefit_msg = (
                    f", PR-prefit {pr_obs_stats.prefit_epochs}ep "
                    f"(drop {avg_drop:.1f}/ep, ref={variant.pr_prefit_ref})"
                )
            pr_skip_msg = ""
            if variant.pr_skip_statuses:
                pr_skip_msg = f", PR-skip {pr_obs_stats.epochs_skipped}"
            gate_msg = ""
            if variant_gate_active and gate_stats.epochs_attempted > 0:
                gate_msg = (
                    f", KF gate {gate_stats.epochs_applied}/"
                    f"{gate_stats.epochs_attempted} "
                    f"(skip dd={gate_stats.skipped_min_dd_pairs}, "
                    f"ess={gate_stats.skipped_min_ess_ratio}, "
                    f"spread={gate_stats.skipped_max_spread}, "
                    f"dop_rms={gate_stats.skipped_doppler_wls_rms}, "
                    f"dop_speed={gate_stats.skipped_doppler_wls_speed})"
                )
            hybrid_msg = ""
            if variant.enable_hybrid_pu and hybrid_stats.epochs_attempted > 0:
                hybrid_msg = (
                    f", hybrid {hybrid_stats.epochs_applied}/"
                    f"{hybrid_stats.epochs_attempted} "
                    f"(missing {hybrid_stats.epochs_lookup_missing})"
                )
            rtkdiag_pf_msg = ""
            if variant.enable_rtkdiag_pf_rescue and rtkdiag_pf_stats.epochs_evaluated > 0:
                selected_counts = ",".join(
                    f"{k}:{v}" for k, v in sorted(rtkdiag_pf_stats.selected_counts.items())
                )
                rtkdiag_pf_msg = (
                    f", RTKDiag-PF gate={rtkdiag_pf_stats.gate_pass}/"
                    f"{rtkdiag_pf_stats.epochs_evaluated} "
                    f"opts={rtkdiag_pf_stats.candidate_options_total} "
                    f"pu={rtkdiag_pf_stats.pu_applied} "
                    f"emit_pf={rtkdiag_pf_stats.emit_pf_estimate} "
                    f"emit_cand={rtkdiag_pf_stats.emit_candidate} "
                    f"(miss={rtkdiag_pf_stats.candidate_missing}, "
                    f"min_skip={rtkdiag_pf_stats.skipped_min_epoch}, "
                    f"diag_skip={rtkdiag_pf_stats.skipped_diag_policy}, "
                    f"hyb_skip={rtkdiag_pf_stats.skipped_hybrid_distance}, "
                    f"fb={rtkdiag_pf_stats.fallback_pf}/"
                    f"{rtkdiag_pf_stats.fallback_wls}/"
                    f"{rtkdiag_pf_stats.fallback_hybrid}/"
                    f"{rtkdiag_pf_stats.fallback_last_good}, "
                    f"drift_skip={rtkdiag_pf_stats.emit_skipped_pf_drift}, "
                    f"rec={rtkdiag_pf_stats.recenter_applied}/"
                    f"{rtkdiag_pf_stats.recenter_skipped}, "
                    f"sel={selected_counts})"
                )
            zupt_msg = ""
            if variant.enable_zupt and zupt_stats.epochs_evaluated > 0:
                zupt_msg = (
                    f", ZUPT static {zupt_stats.epochs_static}/"
                    f"{zupt_stats.epochs_evaluated} "
                    f"(rewritten {zupt_stats.epochs_rewritten}, no_imu {zupt_stats.epochs_no_imu})"
                )
            imu_tc_msg = ""
            if variant.enable_imu_tc and imu_tc_stats.epochs_evaluated > 0:
                imu_tc_msg = (
                    f", IMU-TC pu={imu_tc_stats.pu_applied}/"
                    f"{imu_tc_stats.epochs_evaluated} "
                    f"emit_pf={imu_tc_stats.emit_pf_estimate} "
                    f"(anchors={imu_tc_stats.anchor_resets} "
                    f"static={imu_tc_stats.anchor_resets_static}, "
                    f"skip dr={imu_tc_stats.pu_skipped_dr_too_long} "
                    f"dis={imu_tc_stats.pu_skipped_disagreement} "
                    f"no_anc={imu_tc_stats.pu_skipped_no_anchor}, "
                    f"avg_dr={imu_tc_stats.avg_dr_seconds:.2f}s, "
                    f"pf_drift_skip={imu_tc_stats.emit_skipped_pf_drift})"
                )
            ins_tc_msg = ""
            if variant.enable_ins_tc and ins_tc_stats.epochs_evaluated > 0:
                ins_tc_msg = (
                    f", INS-TC align={ins_tc_stats.aligned_at_epoch} "
                    f"yaw={ins_tc_stats.yaw_initialized_at_epoch} "
                    f"pu={ins_tc_stats.pu_applied}/{ins_tc_stats.epochs_evaluated} "
                    f"emit_pf={ins_tc_stats.emit_pf_estimate} "
                    f"(pimu_init={ins_tc_stats.particle_imu_initialized}, "
                    f"pimu={ins_tc_stats.particle_imu_predict_used}, "
                    f"rec={ins_tc_stats.recenter_applied}/{ins_tc_stats.recenter_skipped}, "
                    f"motion={ins_tc_stats.motion_predict_used}, "
                    f"obs4={ins_tc_stats.obs_status_4_used}, "
                    f"obs3={ins_tc_stats.obs_status_3_used}, "
                    f"skip align={ins_tc_stats.pu_skipped_not_aligned} "
                    f"yaw={ins_tc_stats.pu_skipped_no_yaw} "
                    f"dr={ins_tc_stats.pu_skipped_dr_too_long} "
                    f"dis={ins_tc_stats.pu_skipped_disagreement}, "
                    f"ba={ins_tc_stats.final_acc_bias_norm:.3f}m/s2, "
                    f"bg={ins_tc_stats.final_gyro_bias_norm_dps:.3f}dps, "
                    f"possig={ins_tc_stats.final_pos_sigma_m:.2f}m, "
                    f"pf_drift_skip={ins_tc_stats.emit_skipped_pf_drift})"
                )
            tdcp_msg = ""
            if variant.enable_tdcp_smoother and tdcp_stats.pairs_attempted > 0:
                tdcp_msg = (
                    f", TDCP {tdcp_stats.pairs_accepted}/"
                    f"{tdcp_stats.pairs_attempted} pairs "
                    f"(reject min_sats={tdcp_stats.pairs_rejected_min_sats}, "
                    f"postfit={tdcp_stats.pairs_rejected_postfit})"
                )
            low_sat_bridge_msg = ""
            if variant.enable_low_sat_bridge:
                low_sat_bridge_msg = (
                    f", low-sat bridge {low_sat_bridge_stats.epochs_rewritten}ep/"
                    f"{low_sat_bridge_stats.spans_applied}span"
                    f", startup {low_sat_bridge_stats.startup_epochs_rewritten}ep/"
                    f"{low_sat_bridge_stats.startup_spans_applied}span"
                )
            fgo_msg = ""
            if variant.enable_fgo_lambda and fgo_stats.windows_attempted > 0:
                diag_n = max(int(fgo_stats.lambda_diag_windows), 1)
                fgo_msg = (
                    f", FGO solved {fgo_stats.windows_solved}/"
                    f"{fgo_stats.windows_attempted} applied {fgo_stats.windows_applied} "
                    f"(fixed {fgo_stats.n_fixed_total}, "
                    f"obs {fgo_stats.n_fixed_observations_total}, "
                    f"cand {fgo_stats.lambda_candidates_total}, "
                    f"rej {fgo_stats.lambda_ratio_rejected_total}, "
                    f"bestR {fgo_stats.lambda_best_ratio:.2f}, "
                    f"medR {fgo_stats.lambda_ratio_median_sum / diag_n:.2f}, "
                    f"frac {fgo_stats.lambda_segment_abs_frac_median_sum / diag_n:.2f}, "
                    f"fixRes {fgo_stats.postfit_fixed_abs_cycles_median_sum / max(int(fgo_stats.windows_solved), 1):.3f}cy, "
                    f"ddpr {fgo_stats.postfit_dd_pr_abs_m_median_sum / max(int(fgo_stats.windows_solved), 1):.1f}m, "
                    f"replaced {fgo_stats.epochs_replaced} ep)"
                )
                if variant.enable_dd_pr_ls_anchor:
                    gt_n = max(int(fgo_stats.dd_pr_ls_anchor_gt_count), 1)
                    fgo_msg += (
                        f", DDPR-LS {fgo_stats.dd_pr_ls_anchor_accepted}/"
                        f"{fgo_stats.dd_pr_ls_anchor_attempted} "
                        f"mode={variant.dd_pr_ls_anchor_mode} "
                        f"(shift {fgo_stats.dd_pr_ls_anchor_shift_sum_m / max(int(fgo_stats.dd_pr_ls_anchor_accepted), 1):.1f}m, "
                        f"rms {fgo_stats.dd_pr_ls_anchor_postfit_sum_m / max(int(fgo_stats.dd_pr_ls_anchor_accepted), 1):.1f}m, "
                        f"gt {fgo_stats.dd_pr_ls_anchor_gt_error_sum_m / gt_n:.1f}m, "
                        f"seed {fgo_stats.dd_pr_ls_anchor_seed_error_sum_m / gt_n:.1f}m, "
                        f"imp {100.0 * fgo_stats.dd_pr_ls_anchor_improved / gt_n:.0f}%, "
                        f"<=0.5 {100.0 * fgo_stats.dd_pr_ls_anchor_pass_05m / gt_n:.0f}%)"
                    )
            print(
                f"    PPC honest: {row['honest_ppc_pct']:5.2f}%  "
                f"(pass {row['honest_pass_m']:.0f} / total {row['honest_total_m']:.0f}m, "
                f"{ms_per_epoch:.1f} ms/epoch{pr_msg}{pr_weight_msg}{prefit_msg}{pr_skip_msg}{defer_msg}{dd_msg}{gate_msg}{hybrid_msg}{rtkdiag_pf_msg}{zupt_msg}{imu_tc_msg}{ins_tc_msg}{tdcp_msg}{low_sat_bridge_msg}{fgo_msg})",
                flush=True,
            )

    if not rows:
        raise SystemExit("no rows produced")

    out_csv = RESULTS_DIR / f"{args.results_prefix}_runs.csv"
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    internal_diag_path = RESULTS_DIR / f"{args.results_prefix}_internal_epochs.csv"
    if args.write_internal_diagnostics and internal_diag_rows:
        _write_dict_rows(internal_diag_path, internal_diag_rows)

    print()
    print("=" * 72)
    print("  Honest aggregates (OFFICIAL PPC2024 metric = per-run-averaged;")
    print("  TURING target 85.6% is per-run-averaged; pooled shown for diagnostics)")
    for method in [v.method_label for v in variants]:
        total = agg_total[method]
        if total <= 0:
            continue
        pooled_pct = 100.0 * agg_pass[method] / total
        per_run = per_run_pcts.get(method, [])
        official_pct = sum(per_run) / len(per_run) if per_run else 0.0
        print(
            f"  {method:18s}: OFFICIAL {official_pct:5.2f}%  pooled {pooled_pct:5.2f}%   "
            f"(pass {agg_pass[method]:.0f}m / total {total:.0f}m, n_runs={len(per_run)})"
        )
    print(f"  Saved: {out_csv}")
    if args.write_internal_diagnostics and internal_diag_rows:
        print(f"  Saved internal diagnostics: {internal_diag_path}")
    print("=" * 72)


def _aligned_positions(
    full_ref: list[tuple[float, np.ndarray]],
    times_used: np.ndarray,
    positions: np.ndarray,
) -> list[np.ndarray]:
    """Yield estimated positions aligned to full_ref order, [0,0,0] if missing."""
    pos_map = {round(float(t), 1): p for t, p in zip(times_used, positions, strict=True)}
    out: list[np.ndarray] = []
    for tow, _t in full_ref:
        match = pos_map.get(round(float(tow), 1))
        if match is None or not np.all(np.isfinite(match)):
            out.append(np.zeros(3, dtype=np.float64))
        else:
            out.append(np.asarray(match, dtype=np.float64))
    return out


if __name__ == "__main__":
    main()
