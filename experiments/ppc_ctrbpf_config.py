"""``CTRBPFConfig`` and the named config variants for ``exp_ppc_ctrbpf_fgo``.

Moved verbatim out of ``exp_ppc_ctrbpf_fgo`` (which re-exports every name).
No behaviour change.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from ppc_ctrbpf_io import (
    _parse_diag_threshold_list,
    _parse_label_factor_list,
    _parse_label_list,
)


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
    # WP23a: Suzuki-style (ICRA 2024, "Multiple Update Particle Filter",
    # arXiv:2403.03394) multiple-update schedule. When enabled, this
    # REPLACES the single-shot ``enable_dd_carrier_afv`` DD-carrier update
    # (mutually exclusive at the call site -- see the DD block in
    # ``_run_ctrbpf_on_segment``) with two sequential, independently
    # ESS-tempered stages per epoch: (i) DD-pseudorange (absolute,
    # position-only geometry, least sharp) -> temper to
    # ``cp_mupf_pr_stage_target_ess_ratio`` -> resample-if-needed; (ii)
    # DD-carrier AFV (fractional-cycle, sharpest) -> applied as a
    # coarse-to-fine SIGMA SEQUENCE (each value -> temper to
    # ``cp_mupf_cp_stage_target_ess_ratio`` -> resample-if-needed) rather
    # than a single jump to the tight target sigma. The coarse-to-fine
    # sequence idea is not new to this task: it is exactly the pattern
    # already validated elsewhere in this codebase for the (different,
    # GSDC2023/Trimble) PF-smoother track --
    # ``gnss_gpu.dd_carrier_epoch_update.apply_carrier_epoch_update`` /
    # ``gnss_gpu.pf_smoother_config.MupfConfig.sigma_cycles`` with a
    # validated ``sigma_sequence_cycles=(2.0, 0.5, 0.05)`` default -- reused
    # here rather than reinvented.
    enable_cp_mupf: bool = False
    # Starting sigmas: documented as inherited from tc_fgo/rbpf_fgo's own
    # DD sigmas (``fgo_dd_pr_sigma_m=5.0``, ``fgo_dd_sigma_cycles=0.20``,
    # see below) -- independently tunable copies, not aliases, so changing
    # one does not silently perturb the (unrelated, FGO-postprocess-only)
    # ``enable_fgo_lambda`` feature.
    cp_mupf_dd_pr_sigma_m: float = 5.0
    cp_mupf_pr_stage_target_ess_ratio: float = 0.10
    cp_mupf_dd_cp_sigma_sequence_cycles: tuple[float, ...] = (2.0, 0.5, 0.05)
    cp_mupf_cp_stage_target_ess_ratio: float = 0.10
    cp_mupf_stage_max_iters: int = 20
    cp_mupf_stage_max_tempering_steps: int = 64
    cp_mupf_min_pairs: int = 3
    # >1 splits the (post-slip-gate) DD-CP pairs round-robin into this many
    # sequential sub-updates per sigma stage -- the spec's documented
    # fallback "if a single CP update is still too sharp". Off (1) by
    # default; the coarse-to-fine sigma sequence is tried first.
    cp_mupf_cp_n_groups: int = 1
    # WP23a item 3: cycle-slip proxy. No slip detector exists elsewhere in
    # this codebase's DD machinery for the PPC/non-hybrid path (grepped
    # dd_carrier.py and this file for "slip": none found). Gate CP usage on
    # raw epoch-to-epoch DD-carrier-phase continuity per (ref_sat, sat)
    # pair instead -- see ``_cp_slip_gate``.
    cp_mupf_slip_gate_enabled: bool = True
    cp_mupf_slip_max_delta_cycles: float = 2.0
    cp_mupf_slip_max_dt_s: float = 2.0
    # WP23a diagnostic (default off -- see ``_mupf_stage_update``'s
    # docstring and WP23A_REPORT.md's root-cause section): resample-if-
    # needed *before*, not just after, each MUPF stage's update, so a
    # stage's own tempering guard always starts from a fresh (~ESS/N=1.0)
    # baseline instead of inheriting an already-degenerate ESS/N from
    # earlier untempered updates this same epoch (which otherwise causes
    # ``_apply_pr_ess_guard`` to revert the stage entirely, alpha=0.0).
    cp_mupf_resample_before_stage: bool = False
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
    # Preserve multimodal PF posteriors at emission. ``diagnostic`` computes
    # and records modes without changing the trajectory; ``emit`` replaces a
    # PF-sourced weighted mean only when the selector accepts a reachable mode.
    pf_mode_policy: str = "off"  # off | diagnostic | emit
    pf_mode_voxel_size_m: float = 2.0
    pf_mode_min_core_cell_mass: float = 1.0e-4
    pf_mode_min_core_cell_particles: int = 3
    pf_mode_min_mass: float = 0.01
    pf_mode_assignment_radius_m: float = 6.0
    pf_mode_max_modes: int = 8
    pf_mode_max_particles: int = 8192
    pf_mode_select_min_mass: float = 0.20
    pf_mode_select_min_score_ratio: float = 1.5
    pf_mode_require_multiple_modes: bool = True
    pf_mode_min_epoch: int = 10
    pf_mode_prediction_sigma_m: float = 5.0
    pf_mode_max_prediction_distance_m: float = 20.0
    pf_mode_min_mean_distance_m: float = 0.5
    pf_mode_max_mean_distance_m: float = 20.0
    enable_pf_ffbsi_smoother: bool = False
    pf_ffbsi_lag_epochs: int = 25
    pf_ffbsi_paths: int = 32
    pf_ffbsi_seed: int = 20260713
    pf_ffbsi_mode: str = "marginal"
    pf_ffbsi_max_std_m: float = 5.0
    pf_ffbsi_max_correction_m: float = 10.0
    pf_ffbsi_min_unique_particles: int = 2
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
    # WP22a: WP21b IMU-preintegration predict-step guide
    # (python/gnss_gpu/pf_imu_preint_adapter.py::ImuPreintPfGuide), wired as
    # a standalone switch independent of imu_tc/ins_tc/hybrid-velocity-guide.
    # Between consecutive rover epochs, the buffered 100 Hz PPC IMU segment
    # is preintegrated and closed into (a) an ECEF velocity guide fed to
    # ``pf.predict(velocity=..., rbpf_velocity_kf=True)`` in place of the
    # baseline's guide-less RBPF-velKF predict, (b) a heading-uncertainty-
    # aware ``sigma_pos`` (WP21b item 1: cross-track lever
    # ``|displacement|*sigma_heading``, heading driven by
    # ``gnss_gpu.imu.ComplementaryHeadingFilter`` corrected against this
    # pipeline's own causal ``wls_positions``), and (c) the segment's
    # accel/gyro-derived delta_v covariance fed into every particle's
    # velocity-KF ``Sigma_v`` via ``pf.set_velocity_covariance`` (WP21b item
    # 2). Falls back to the baseline no-guide predict for any epoch with an
    # empty/degenerate IMU segment. Not combined with imu_tc/ins_tc in this
    # ablation (both are alternative predict-step IMU integrations; running
    # them together is untested).
    enable_imu_preint: bool = False
    imu_preint_sigma_accel_mps2_sqrthz: float = 0.05
    imu_preint_sigma_gyro_radps_sqrthz: float = 0.005
    imu_preint_sigma_pos_floor_m: float = 0.05
    imu_preint_sigma_pos_scale: float = 1.0
    imu_preint_velocity_blend_alpha: float = 0.3
    imu_preint_sigma_spp_pos_m: float = 30.0
    imu_preint_min_heading_fix_disp_m: float = 2.0
    # WP22b item 2: adaptive likelihood tempering. A per-epoch temperature
    # beta in (0,1] scales the *total* log-likelihood increment applied that
    # epoch (across whichever of PR/GMM/Doppler-KF/DD-carrier updates ran)
    # so that ESS/N after tempering hits ``epoch_tempering_target_ess_ratio``
    # (bisection; see ``_apply_pr_ess_guard``, reused as-is since it is
    # already a generic log-weight-delta tempering primitive, not PR-
    # specific). Requires end-of-epoch deferred resampling so there is a
    # single well-defined "this epoch's full log-likelihood delta" to temper
    # (see ``_resample_deferred``). Trades statistical efficiency/bias for
    # particle diversity -- tempering does not target the true posterior,
    # only a flattened version of it; see the WP22b report for the caveat.
    enable_epoch_tempering: bool = False
    epoch_tempering_target_ess_ratio: float = 0.10
    epoch_tempering_max_iters: int = 20
    # WP22b item 3: C/N0- and elevation-driven per-satellite GMM mixture
    # weight (w_los), replacing the fixed scalar ``pr_gmm_w_los`` when
    # ``enable_pr_gmm`` is also set. ``pf_device_weight_gmm`` only accepts
    # one scalar w_los per kernel call, so satellites are grouped into
    # ``cn0_gmm_n_buckets`` bins by their computed w_los and one kernel call
    # is issued per non-empty bucket (exact under the independent-
    # observation PF model -- product of per-satellite mixture likelihoods
    # -- up to w_los quantization within a bucket). See
    # ``_cn0_elevation_w_los`` for the calibrated logistic mapping.
    enable_cn0_gmm: bool = False
    cn0_gmm_baseline_dbhz0: float = 30.0
    cn0_gmm_baseline_dbhz90: float = 45.0
    cn0_gmm_gap_mid_db: float = 14.0
    cn0_gmm_logistic_scale_db: float = 4.0
    cn0_gmm_w_los_min: float = 0.05
    cn0_gmm_w_los_max: float = 0.97
    cn0_gmm_n_buckets: int = 5
    # WP22b item 4: particle-wise NLOS deweighting. The native undifferenced
    # and DD-carrier-AFV weight kernels already gate each satellite's
    # contribution per particle (each particle computes its own residual
    # from its own hypothesized state, so the same scalar threshold rejects
    # a different satellite subset per particle -- see
    # ``pfd_weight_kernel``/``pfd_weight_dd_carrier_afv_kernel`` in
    # ``pf_device.cu``). This was never wired into ``_build_pf`` before
    # WP22b; enabling it requires no CUDA changes.
    enable_particle_nlos: bool = False
    particle_nlos_undiff_pr_threshold_m: float = 30.0
    particle_nlos_dd_carrier_threshold_cycles: float = 0.5
    particle_nlos_huber: bool = False
    particle_nlos_huber_undiff_pr_k: float = 1.5
    particle_nlos_huber_dd_carrier_k: float = 1.5
    systems: tuple[str, ...] = ("G", "R", "E", "C", "J")
    method_label: str = "PF-PR"


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
        cp_mupf_dd_pr_sigma_m=args.cp_mupf_dd_pr_sigma_m,
        cp_mupf_pr_stage_target_ess_ratio=args.cp_mupf_pr_stage_target_ess_ratio,
        cp_mupf_dd_cp_sigma_sequence_cycles=tuple(
            float(x) for x in args.cp_mupf_dd_cp_sigma_sequence_cycles.split(",") if x.strip()
        ),
        cp_mupf_cp_stage_target_ess_ratio=args.cp_mupf_cp_stage_target_ess_ratio,
        cp_mupf_stage_max_iters=args.cp_mupf_stage_max_iters,
        cp_mupf_stage_max_tempering_steps=args.cp_mupf_stage_max_tempering_steps,
        cp_mupf_min_pairs=args.cp_mupf_min_pairs,
        cp_mupf_cp_n_groups=args.cp_mupf_cp_n_groups,
        cp_mupf_slip_gate_enabled=not args.disable_cp_mupf_slip_gate,
        cp_mupf_slip_max_delta_cycles=args.cp_mupf_slip_max_delta_cycles,
        cp_mupf_slip_max_dt_s=args.cp_mupf_slip_max_dt_s,
        cp_mupf_resample_before_stage=bool(args.cp_mupf_resample_before_stage),
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
        pf_mode_policy=str(args.pf_mode_policy),
        pf_mode_voxel_size_m=args.pf_mode_voxel_size_m,
        pf_mode_min_core_cell_mass=args.pf_mode_min_core_cell_mass,
        pf_mode_min_core_cell_particles=args.pf_mode_min_core_cell_particles,
        pf_mode_min_mass=args.pf_mode_min_mass,
        pf_mode_assignment_radius_m=args.pf_mode_assignment_radius_m,
        pf_mode_max_modes=args.pf_mode_max_modes,
        pf_mode_max_particles=args.pf_mode_max_particles,
        pf_mode_select_min_mass=args.pf_mode_select_min_mass,
        pf_mode_select_min_score_ratio=args.pf_mode_select_min_score_ratio,
        pf_mode_require_multiple_modes=not bool(args.pf_mode_allow_single_mode),
        pf_mode_min_epoch=args.pf_mode_min_epoch,
        pf_mode_prediction_sigma_m=args.pf_mode_prediction_sigma_m,
        pf_mode_max_prediction_distance_m=args.pf_mode_max_prediction_distance_m,
        pf_mode_min_mean_distance_m=args.pf_mode_min_mean_distance_m,
        pf_mode_max_mean_distance_m=args.pf_mode_max_mean_distance_m,
        enable_pf_ffbsi_smoother=bool(args.enable_pf_ffbsi_smoother),
        pf_ffbsi_lag_epochs=args.pf_ffbsi_lag_epochs,
        pf_ffbsi_paths=args.pf_ffbsi_paths,
        pf_ffbsi_seed=args.pf_ffbsi_seed,
        pf_ffbsi_mode=args.pf_ffbsi_mode,
        pf_ffbsi_max_std_m=args.pf_ffbsi_max_std_m,
        pf_ffbsi_max_correction_m=args.pf_ffbsi_max_correction_m,
        pf_ffbsi_min_unique_particles=args.pf_ffbsi_min_unique_particles,
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
        # WP22a: applied uniformly via --imu {off,preint} to every variant
        # selected by --methods (not gated by method-label string tokens,
        # unlike imu_tc/ins_tc which are opt-in per method combo).
        enable_imu_preint=(args.imu == "preint"),
        imu_preint_sigma_accel_mps2_sqrthz=args.imu_preint_sigma_accel,
        imu_preint_sigma_gyro_radps_sqrthz=args.imu_preint_sigma_gyro,
        imu_preint_sigma_pos_floor_m=args.imu_preint_sigma_pos_floor,
        imu_preint_sigma_pos_scale=args.imu_preint_sigma_pos_scale,
        imu_preint_velocity_blend_alpha=args.imu_preint_velocity_blend_alpha,
        imu_preint_sigma_spp_pos_m=args.imu_preint_sigma_spp_pos_m,
        imu_preint_min_heading_fix_disp_m=args.imu_preint_min_heading_fix_disp_m,
        # WP22b: applied uniformly to every selected variant, same pattern
        # as --imu {off,preint} above, so the ablation grid (item 5) is a
        # single command per {imu arm} x {tempering/GMM/NLOS toggle combo}
        # rather than new method-label branches.
        enable_epoch_tempering=bool(args.enable_epoch_tempering),
        epoch_tempering_target_ess_ratio=args.epoch_tempering_target_ess_ratio,
        epoch_tempering_max_iters=args.epoch_tempering_max_iters,
        enable_cn0_gmm=bool(args.enable_cn0_gmm),
        enable_pr_gmm=bool(args.enable_cn0_gmm),
        cn0_gmm_baseline_dbhz0=args.cn0_gmm_baseline_dbhz0,
        cn0_gmm_baseline_dbhz90=args.cn0_gmm_baseline_dbhz90,
        cn0_gmm_gap_mid_db=args.cn0_gmm_gap_mid_db,
        cn0_gmm_logistic_scale_db=args.cn0_gmm_logistic_scale_db,
        cn0_gmm_w_los_min=args.cn0_gmm_w_los_min,
        cn0_gmm_w_los_max=args.cn0_gmm_w_los_max,
        cn0_gmm_n_buckets=int(args.cn0_gmm_n_buckets),
        enable_particle_nlos=bool(args.enable_particle_nlos),
        particle_nlos_undiff_pr_threshold_m=args.particle_nlos_undiff_pr_threshold_m,
        particle_nlos_dd_carrier_threshold_cycles=args.particle_nlos_dd_carrier_threshold_cycles,
        particle_nlos_huber=bool(args.particle_nlos_huber),
        particle_nlos_huber_undiff_pr_k=args.particle_nlos_huber_undiff_pr_k,
        particle_nlos_huber_dd_carrier_k=args.particle_nlos_huber_dd_carrier_k,
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
    # WP23a: Suzuki-style DD-PR + DD-CP-AFV multiple-update (MUPF) schedule
    # on top of the WP22b winner (rbpf+dd+gate). ``enable_dd_carrier_afv``
    # is explicitly turned off here -- ``enable_cp_mupf`` takes over the
    # DD-carrier weighting (mutually exclusive at the call site) with the
    # two-stage tempered schedule instead of the single-shot call.
    if "rbpf+dd+cp+gate" in args.methods:
        variant_kwargs = {**base, **aaa_gate}
        variants.append(CTRBPFConfig(
            **variant_kwargs,
            enable_rbpf_velocity_kf=True,
            enable_dd_carrier_afv=False,
            enable_cp_mupf=True,
            method_label="RBPF-velKF+DD+CP-MUPF+gate",
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
