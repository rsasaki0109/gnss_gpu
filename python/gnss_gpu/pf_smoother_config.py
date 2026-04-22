"""Configuration objects for PF smoother experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class RunSelectionConfig:
    rover_source: str = "trimble"
    max_epochs: int = 0
    skip_valid_epochs: int = 0


@dataclass(frozen=True)
class ParticleFilterRuntimeConfig:
    n_particles: int
    sigma_pos: float
    sigma_pr: float
    position_update_sigma: float | None
    use_smoother: bool
    resampling: str = "megopolis"
    pf_sigma_vel: float = 0.0
    pf_velocity_guide_alpha: float = 1.0
    pf_init_spread_vel: float = 0.0


@dataclass(frozen=True)
class MotionConfig:
    predict_guide: str
    sigma_pos_tdcp: float | None = None
    sigma_pos_tdcp_tight: float | None = None
    tdcp_tight_rms_max_m: float = 1.0e9
    tdcp_elevation_weight: bool = False
    tdcp_el_sin_floor: float = 0.1
    tdcp_rms_threshold: float = 3.0
    imu_tight_coupling: bool = False
    imu_stop_sigma_pos: float | None = None


@dataclass(frozen=True)
class DopplerConfig:
    position_update: bool = False
    pu_sigma: float = 5.0
    per_particle: bool = False
    sigma_mps: float = 0.5
    velocity_update_gain: float = 0.25
    max_velocity_update_mps: float = 10.0
    min_sats: int = 4
    rbpf_velocity_kf: bool = False
    rbpf_velocity_init_sigma: float = 2.0
    rbpf_velocity_process_noise: float = 1.0
    rbpf_doppler_sigma: float | None = None
    rbpf_gate_min_dd_pairs: int | None = None
    rbpf_gate_min_ess_ratio: float | None = None
    rbpf_gate_max_spread_m: float | None = None


@dataclass(frozen=True)
class TdcpPositionUpdateConfig:
    enabled: bool = False
    sigma: float = 0.5
    rms_max: float = 3.0
    spp_max_diff_mps: float | None = 6.0
    gate_dd_carrier_min_pairs: int | None = None
    gate_dd_carrier_max_pairs: int | None = None
    gate_dd_pseudorange_max_pairs: int | None = None
    gate_min_spread_m: float | None = None
    gate_max_spread_m: float | None = None
    gate_min_ess_ratio: float | None = None
    gate_max_ess_ratio: float | None = None
    gate_dd_pr_max_raw_median_m: float | None = None
    gate_dd_cp_max_raw_afv_median_cycles: float | None = None
    gate_logic: str = "any"
    gate_stop_mode: str = "any"


@dataclass(frozen=True)
class RobustMeasurementConfig:
    residual_downweight: bool = False
    residual_threshold: float = 15.0
    pr_accel_downweight: bool = False
    pr_accel_threshold: float = 5.0
    use_gmm: bool = False
    gmm_w_los: float = 0.7
    gmm_mu_nlos: float = 15.0
    gmm_sigma_nlos: float = 30.0
    per_particle_nlos_gate: bool = False
    per_particle_nlos_dd_pr_threshold_m: float = 10.0
    per_particle_nlos_dd_carrier_threshold_cycles: float = 0.5
    per_particle_nlos_undiff_pr_threshold_m: float = 30.0
    per_particle_huber: bool = False
    per_particle_huber_dd_pr_k: float = 1.5
    per_particle_huber_dd_carrier_k: float = 1.5
    per_particle_huber_undiff_pr_k: float = 1.5


@dataclass(frozen=True)
class MupfConfig:
    enabled: bool = False
    sigma_cycles: float = 0.05
    snr_min: float = 25.0
    elev_min: float = 0.15


@dataclass(frozen=True)
class DDPseudorangeConfig:
    enabled: bool = False
    sigma: float = 0.75
    base_interp: bool = False
    gate_residual_m: float | None = None
    gate_adaptive_floor_m: float | None = None
    gate_adaptive_mad_mult: float | None = None
    gate_epoch_median_m: float | None = None
    gate_ess_min_scale: float = 1.0
    gate_ess_max_scale: float = 1.0
    gate_spread_min_scale: float = 1.0
    gate_spread_max_scale: float = 1.0
    gate_low_spread_m: float = 1.5
    gate_high_spread_m: float = 8.0


@dataclass(frozen=True)
class WidelaneConfig:
    enabled: bool = False
    min_fix_rate: float = 0.3
    ratio_threshold: float = 3.0
    dd_sigma: float = 0.1
    gate_min_fixed_pairs: int | None = None
    gate_min_fix_rate: float | None = None
    gate_min_spread_m: float | None = None
    gate_max_epoch_median_residual_m: float | None = None
    gate_max_pair_residual_m: float | None = None


@dataclass(frozen=True)
class DDCarrierConfig:
    enabled: bool = False
    sigma_cycles: float = 0.05
    base_interp: bool = False
    gate_afv_cycles: float | None = None
    gate_adaptive_floor_cycles: float | None = None
    gate_adaptive_mad_mult: float | None = None
    gate_epoch_median_cycles: float | None = None
    gate_low_ess_epoch_median_cycles: float | None = None
    gate_low_ess_max_ratio: float | None = None
    gate_low_ess_max_spread_m: float | None = None
    gate_low_ess_require_no_dd_pr: bool = False
    gate_ess_min_scale: float = 1.0
    gate_ess_max_scale: float = 1.0
    gate_spread_min_scale: float = 1.0
    gate_spread_max_scale: float = 1.0
    gate_low_spread_m: float = 1.5
    gate_high_spread_m: float = 8.0
    sigma_support_low_pairs: int | None = None
    sigma_support_high_pairs: int | None = None
    sigma_support_max_scale: float = 1.0
    sigma_afv_good_cycles: float | None = None
    sigma_afv_bad_cycles: float | None = None
    sigma_afv_max_scale: float = 1.0
    sigma_ess_low_ratio: float | None = None
    sigma_ess_high_ratio: float | None = None
    sigma_ess_max_scale: float = 1.0
    sigma_max_scale: float | None = None


@dataclass(frozen=True)
class CarrierRescueConfig:
    anchor_enabled: bool = False
    anchor_sigma_m: float = 0.25
    anchor_min_sats: int = 4
    anchor_max_age_s: float = 3.0
    anchor_max_residual_m: float = 0.75
    anchor_max_continuity_residual_m: float = 0.50
    anchor_min_stable_epochs: int = 1
    anchor_blend_alpha: float = 0.5
    anchor_reanchor_jump_cycles: float = 4.0
    anchor_seed_dd_min_pairs: int = 3
    fallback_undiff: bool = False
    fallback_sigma_cycles: float = 0.10
    fallback_min_sats: int = 4
    fallback_prefer_tracked: bool = False
    fallback_tracked_min_stable_epochs: int = 1
    fallback_tracked_min_sats: int | None = None
    fallback_tracked_continuity_good_m: float | None = None
    fallback_tracked_continuity_bad_m: float | None = None
    fallback_tracked_sigma_min_scale: float = 1.0
    fallback_tracked_sigma_max_scale: float = 1.0
    fallback_weak_dd_max_pairs: int | None = None
    fallback_weak_dd_max_ess_ratio: float | None = None
    fallback_weak_dd_min_raw_afv_median_cycles: float | None = None
    fallback_weak_dd_require_no_dd_pr: bool = False
    skip_low_support_ess_ratio: float | None = None
    skip_low_support_max_pairs: int | None = None
    skip_low_support_max_spread_m: float | None = None
    skip_low_support_min_raw_afv_median_cycles: float | None = None
    skip_low_support_require_no_dd_pr: bool = False


@dataclass(frozen=True)
class ObservationConfig:
    robust: RobustMeasurementConfig
    mupf: MupfConfig
    dd_pseudorange: DDPseudorangeConfig
    widelane: WidelaneConfig
    dd_carrier: DDCarrierConfig
    carrier_rescue: CarrierRescueConfig


@dataclass(frozen=True)
class SmootherPostprocessConfig:
    position_update_sigma: float | None = None
    skip_widelane_dd_pseudorange: bool = False
    widelane_forward_guard: bool = False
    widelane_forward_guard_min_shift_m: float | None = None
    stop_segment_constant: bool = False
    stop_segment_min_epochs: int = 5
    stop_segment_source: str = "smoothed"
    stop_segment_max_radius_m: float | None = None
    stop_segment_blend: float = 1.0
    stop_segment_density_neighbors: int = 200
    stop_segment_static_gnss: bool = False
    stop_segment_static_min_observations: int = 40
    stop_segment_static_prior_sigma_m: float = 20.0
    stop_segment_static_pr_sigma_m: float = 8.0
    stop_segment_static_dd_pr_sigma_m: float = 4.0
    stop_segment_static_dd_cp_sigma_cycles: float = 0.50
    stop_segment_static_max_update_m: float | None = 25.0
    stop_segment_static_blend: float = 1.0
    tail_guard_ess_max_ratio: float | None = None
    tail_guard_dd_carrier_max_pairs: int | None = None
    tail_guard_dd_pseudorange_max_pairs: int | None = None
    tail_guard_min_shift_m: float | None = None
    tail_guard_expand_epochs: int | None = None
    tail_guard_expand_min_shift_m: float | None = None
    tail_guard_expand_dd_pseudorange_max_pairs: int | None = None


@dataclass(frozen=True)
class LocalFgoPostprocessConfig:
    window: str | None = None
    window_min_epochs: int = 100
    dd_max_pairs: int = 4
    prior_sigma_m: float = 0.5
    motion_sigma_m: float = 1.0
    dd_huber_k: float = 1.5
    pr_huber_k: float = 1.5
    dd_sigma_cycles: float = 0.20
    pr_sigma_m: float = 5.0
    max_iterations: int = 50
    lambda_enabled: bool = False
    lambda_ratio_threshold: float = 3.0
    lambda_sigma_cycles: float = 0.05
    lambda_min_epochs: int = 20
    motion_source: str = "predict"
    tdcp_rms_max_m: float = 3.0
    tdcp_spp_max_diff_mps: float = 6.0
    two_step: bool = False
    stage1_prior_sigma_m: float | None = None
    stage1_motion_sigma_m: float | None = None
    stage1_pr_sigma_m: float | None = None


@dataclass(frozen=True)
class DiagnosticsConfig:
    collect_epoch_diagnostics: bool = False


@dataclass(frozen=True)
class PfSmootherConfigParts:
    run_selection: RunSelectionConfig
    particle_filter: ParticleFilterRuntimeConfig
    motion: MotionConfig
    doppler: DopplerConfig
    tdcp_position_update: TdcpPositionUpdateConfig
    observations: ObservationConfig
    smoother: SmootherPostprocessConfig
    local_fgo: LocalFgoPostprocessConfig
    diagnostics: DiagnosticsConfig


@dataclass(frozen=True)
class PfSmootherConfig:
    """Runtime configuration for one PF smoother evaluation.

    Dataset handles and run identity stay outside this object so the same
    configuration can be reused across runs and sweeps.
    """

    n_particles: int
    sigma_pos: float
    sigma_pr: float
    position_update_sigma: float | None
    predict_guide: str
    use_smoother: bool
    rover_source: str = "trimble"
    seed: int = 42
    max_epochs: int = 0
    skip_valid_epochs: int = 0
    sigma_pos_tdcp: float | None = None
    sigma_pos_tdcp_tight: float | None = None
    tdcp_tight_rms_max_m: float = 1.0e9
    resampling: str = "megopolis"
    tdcp_elevation_weight: bool = False
    tdcp_el_sin_floor: float = 0.1
    tdcp_rms_threshold: float = 3.0
    residual_downweight: bool = False
    residual_threshold: float = 15.0
    pr_accel_downweight: bool = False
    pr_accel_threshold: float = 5.0
    use_gmm: bool = False
    gmm_w_los: float = 0.7
    gmm_mu_nlos: float = 15.0
    gmm_sigma_nlos: float = 30.0
    doppler_position_update: bool = False
    doppler_pu_sigma: float = 5.0
    doppler_per_particle: bool = False
    doppler_sigma_mps: float = 0.5
    doppler_velocity_update_gain: float = 0.25
    doppler_max_velocity_update_mps: float = 10.0
    doppler_min_sats: int = 4
    rbpf_velocity_kf: bool = False
    rbpf_velocity_init_sigma: float = 2.0
    rbpf_velocity_process_noise: float = 1.0
    rbpf_doppler_sigma: float | None = None
    rbpf_velocity_kf_gate_min_dd_pairs: int | None = None
    rbpf_velocity_kf_gate_min_ess_ratio: float | None = None
    rbpf_velocity_kf_gate_max_spread_m: float | None = None
    pf_sigma_vel: float = 0.0
    pf_velocity_guide_alpha: float = 1.0
    pf_init_spread_vel: float = 0.0
    imu_tight_coupling: bool = False
    imu_stop_sigma_pos: float | None = None
    tdcp_position_update: bool = False
    tdcp_pu_sigma: float = 0.5
    tdcp_pu_rms_max: float = 3.0
    tdcp_pu_spp_max_diff_mps: float | None = 6.0
    tdcp_pu_gate_dd_carrier_min_pairs: int | None = None
    tdcp_pu_gate_dd_carrier_max_pairs: int | None = None
    tdcp_pu_gate_dd_pseudorange_max_pairs: int | None = None
    tdcp_pu_gate_min_spread_m: float | None = None
    tdcp_pu_gate_max_spread_m: float | None = None
    tdcp_pu_gate_min_ess_ratio: float | None = None
    tdcp_pu_gate_max_ess_ratio: float | None = None
    tdcp_pu_gate_dd_pr_max_raw_median_m: float | None = None
    tdcp_pu_gate_dd_cp_max_raw_afv_median_cycles: float | None = None
    tdcp_pu_gate_logic: str = "any"
    tdcp_pu_gate_stop_mode: str = "any"
    mupf: bool = False
    mupf_sigma_cycles: float = 0.05
    mupf_snr_min: float = 25.0
    mupf_elev_min: float = 0.15
    dd_pseudorange: bool = False
    dd_pseudorange_sigma: float = 0.75
    dd_pseudorange_base_interp: bool = False
    dd_pseudorange_gate_residual_m: float | None = None
    dd_pseudorange_gate_adaptive_floor_m: float | None = None
    dd_pseudorange_gate_adaptive_mad_mult: float | None = None
    dd_pseudorange_gate_epoch_median_m: float | None = None
    dd_pseudorange_gate_ess_min_scale: float = 1.0
    dd_pseudorange_gate_ess_max_scale: float = 1.0
    dd_pseudorange_gate_spread_min_scale: float = 1.0
    dd_pseudorange_gate_spread_max_scale: float = 1.0
    dd_pseudorange_gate_low_spread_m: float = 1.5
    dd_pseudorange_gate_high_spread_m: float = 8.0
    per_particle_nlos_gate: bool = False
    per_particle_nlos_dd_pr_threshold_m: float = 10.0
    per_particle_nlos_dd_carrier_threshold_cycles: float = 0.5
    per_particle_nlos_undiff_pr_threshold_m: float = 30.0
    per_particle_huber: bool = False
    per_particle_huber_dd_pr_k: float = 1.5
    per_particle_huber_dd_carrier_k: float = 1.5
    per_particle_huber_undiff_pr_k: float = 1.5
    widelane: bool = False
    widelane_min_fix_rate: float = 0.3
    widelane_ratio_threshold: float = 3.0
    widelane_dd_sigma: float = 0.1
    widelane_gate_min_fixed_pairs: int | None = None
    widelane_gate_min_fix_rate: float | None = None
    widelane_gate_min_spread_m: float | None = None
    widelane_gate_max_epoch_median_residual_m: float | None = None
    widelane_gate_max_pair_residual_m: float | None = None
    mupf_dd: bool = False
    mupf_dd_sigma_cycles: float = 0.05
    mupf_dd_base_interp: bool = False
    mupf_dd_gate_afv_cycles: float | None = None
    mupf_dd_gate_adaptive_floor_cycles: float | None = None
    mupf_dd_gate_adaptive_mad_mult: float | None = None
    mupf_dd_gate_epoch_median_cycles: float | None = None
    mupf_dd_gate_low_ess_epoch_median_cycles: float | None = None
    mupf_dd_gate_low_ess_max_ratio: float | None = None
    mupf_dd_gate_low_ess_max_spread_m: float | None = None
    mupf_dd_gate_low_ess_require_no_dd_pr: bool = False
    mupf_dd_gate_ess_min_scale: float = 1.0
    mupf_dd_gate_ess_max_scale: float = 1.0
    mupf_dd_gate_spread_min_scale: float = 1.0
    mupf_dd_gate_spread_max_scale: float = 1.0
    mupf_dd_gate_low_spread_m: float = 1.5
    mupf_dd_gate_high_spread_m: float = 8.0
    mupf_dd_sigma_support_low_pairs: int | None = None
    mupf_dd_sigma_support_high_pairs: int | None = None
    mupf_dd_sigma_support_max_scale: float = 1.0
    mupf_dd_sigma_afv_good_cycles: float | None = None
    mupf_dd_sigma_afv_bad_cycles: float | None = None
    mupf_dd_sigma_afv_max_scale: float = 1.0
    mupf_dd_sigma_ess_low_ratio: float | None = None
    mupf_dd_sigma_ess_high_ratio: float | None = None
    mupf_dd_sigma_ess_max_scale: float = 1.0
    mupf_dd_sigma_max_scale: float | None = None
    carrier_anchor: bool = False
    carrier_anchor_sigma_m: float = 0.25
    carrier_anchor_min_sats: int = 4
    carrier_anchor_max_age_s: float = 3.0
    carrier_anchor_max_residual_m: float = 0.75
    carrier_anchor_max_continuity_residual_m: float = 0.50
    carrier_anchor_min_stable_epochs: int = 1
    carrier_anchor_blend_alpha: float = 0.5
    carrier_anchor_reanchor_jump_cycles: float = 4.0
    carrier_anchor_seed_dd_min_pairs: int = 3
    mupf_dd_fallback_undiff: bool = False
    mupf_dd_fallback_sigma_cycles: float = 0.10
    mupf_dd_fallback_min_sats: int = 4
    mupf_dd_fallback_prefer_tracked: bool = False
    mupf_dd_fallback_tracked_min_stable_epochs: int = 1
    mupf_dd_fallback_tracked_min_sats: int | None = None
    mupf_dd_fallback_tracked_continuity_good_m: float | None = None
    mupf_dd_fallback_tracked_continuity_bad_m: float | None = None
    mupf_dd_fallback_tracked_sigma_min_scale: float = 1.0
    mupf_dd_fallback_tracked_sigma_max_scale: float = 1.0
    mupf_dd_fallback_weak_dd_max_pairs: int | None = None
    mupf_dd_fallback_weak_dd_max_ess_ratio: float | None = None
    mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles: float | None = None
    mupf_dd_fallback_weak_dd_require_no_dd_pr: bool = False
    mupf_dd_skip_low_support_ess_ratio: float | None = None
    mupf_dd_skip_low_support_max_pairs: int | None = None
    mupf_dd_skip_low_support_max_spread_m: float | None = None
    mupf_dd_skip_low_support_min_raw_afv_median_cycles: float | None = None
    mupf_dd_skip_low_support_require_no_dd_pr: bool = False
    collect_epoch_diagnostics: bool = False
    smoother_position_update_sigma: float | None = None
    smoother_skip_widelane_dd_pseudorange: bool = False
    smoother_widelane_forward_guard: bool = False
    smoother_widelane_forward_guard_min_shift_m: float | None = None
    stop_segment_constant: bool = False
    stop_segment_min_epochs: int = 5
    stop_segment_source: str = "smoothed"
    stop_segment_max_radius_m: float | None = None
    stop_segment_blend: float = 1.0
    stop_segment_density_neighbors: int = 200
    stop_segment_static_gnss: bool = False
    stop_segment_static_min_observations: int = 40
    stop_segment_static_prior_sigma_m: float = 20.0
    stop_segment_static_pr_sigma_m: float = 8.0
    stop_segment_static_dd_pr_sigma_m: float = 4.0
    stop_segment_static_dd_cp_sigma_cycles: float = 0.50
    stop_segment_static_max_update_m: float | None = 25.0
    stop_segment_static_blend: float = 1.0
    smoother_tail_guard_ess_max_ratio: float | None = None
    smoother_tail_guard_dd_carrier_max_pairs: int | None = None
    smoother_tail_guard_dd_pseudorange_max_pairs: int | None = None
    smoother_tail_guard_min_shift_m: float | None = None
    smoother_tail_guard_expand_epochs: int | None = None
    smoother_tail_guard_expand_min_shift_m: float | None = None
    smoother_tail_guard_expand_dd_pseudorange_max_pairs: int | None = None
    fgo_local_window: str | None = None
    fgo_local_window_min_epochs: int = 100
    fgo_local_dd_max_pairs: int = 4
    fgo_local_prior_sigma_m: float = 0.5
    fgo_local_motion_sigma_m: float = 1.0
    fgo_local_dd_huber_k: float = 1.5
    fgo_local_pr_huber_k: float = 1.5
    fgo_local_dd_sigma_cycles: float = 0.20
    fgo_local_pr_sigma_m: float = 5.0
    fgo_local_max_iterations: int = 50
    fgo_local_lambda: bool = False
    fgo_local_lambda_ratio_threshold: float = 3.0
    fgo_local_lambda_sigma_cycles: float = 0.05
    fgo_local_lambda_min_epochs: int = 20
    fgo_local_motion_source: str = "predict"
    fgo_local_tdcp_rms_max_m: float = 3.0
    fgo_local_tdcp_spp_max_diff_mps: float = 6.0
    fgo_local_two_step: bool = False
    fgo_local_stage1_prior_sigma_m: float | None = None
    fgo_local_stage1_motion_sigma_m: float | None = None
    fgo_local_stage1_pr_sigma_m: float | None = None

    @property
    def run_selection(self) -> RunSelectionConfig:
        return RunSelectionConfig(
            rover_source=self.rover_source,
            max_epochs=self.max_epochs,
            skip_valid_epochs=self.skip_valid_epochs,
        )

    @property
    def particle_filter(self) -> ParticleFilterRuntimeConfig:
        return ParticleFilterRuntimeConfig(
            n_particles=self.n_particles,
            sigma_pos=self.sigma_pos,
            sigma_pr=self.sigma_pr,
            position_update_sigma=self.position_update_sigma,
            use_smoother=self.use_smoother,
            resampling=self.resampling,
            pf_sigma_vel=self.pf_sigma_vel,
            pf_velocity_guide_alpha=self.pf_velocity_guide_alpha,
            pf_init_spread_vel=self.pf_init_spread_vel,
        )

    @property
    def motion(self) -> MotionConfig:
        return MotionConfig(
            predict_guide=self.predict_guide,
            sigma_pos_tdcp=self.sigma_pos_tdcp,
            sigma_pos_tdcp_tight=self.sigma_pos_tdcp_tight,
            tdcp_tight_rms_max_m=self.tdcp_tight_rms_max_m,
            tdcp_elevation_weight=self.tdcp_elevation_weight,
            tdcp_el_sin_floor=self.tdcp_el_sin_floor,
            tdcp_rms_threshold=self.tdcp_rms_threshold,
            imu_tight_coupling=self.imu_tight_coupling,
            imu_stop_sigma_pos=self.imu_stop_sigma_pos,
        )

    @property
    def doppler(self) -> DopplerConfig:
        return DopplerConfig(
            position_update=self.doppler_position_update,
            pu_sigma=self.doppler_pu_sigma,
            per_particle=self.doppler_per_particle,
            sigma_mps=self.doppler_sigma_mps,
            velocity_update_gain=self.doppler_velocity_update_gain,
            max_velocity_update_mps=self.doppler_max_velocity_update_mps,
            min_sats=self.doppler_min_sats,
            rbpf_velocity_kf=self.rbpf_velocity_kf,
            rbpf_velocity_init_sigma=self.rbpf_velocity_init_sigma,
            rbpf_velocity_process_noise=self.rbpf_velocity_process_noise,
            rbpf_doppler_sigma=self.rbpf_doppler_sigma,
            rbpf_gate_min_dd_pairs=self.rbpf_velocity_kf_gate_min_dd_pairs,
            rbpf_gate_min_ess_ratio=self.rbpf_velocity_kf_gate_min_ess_ratio,
            rbpf_gate_max_spread_m=self.rbpf_velocity_kf_gate_max_spread_m,
        )

    @property
    def tdcp_position(self) -> TdcpPositionUpdateConfig:
        return TdcpPositionUpdateConfig(
            enabled=self.tdcp_position_update,
            sigma=self.tdcp_pu_sigma,
            rms_max=self.tdcp_pu_rms_max,
            spp_max_diff_mps=self.tdcp_pu_spp_max_diff_mps,
            gate_dd_carrier_min_pairs=self.tdcp_pu_gate_dd_carrier_min_pairs,
            gate_dd_carrier_max_pairs=self.tdcp_pu_gate_dd_carrier_max_pairs,
            gate_dd_pseudorange_max_pairs=self.tdcp_pu_gate_dd_pseudorange_max_pairs,
            gate_min_spread_m=self.tdcp_pu_gate_min_spread_m,
            gate_max_spread_m=self.tdcp_pu_gate_max_spread_m,
            gate_min_ess_ratio=self.tdcp_pu_gate_min_ess_ratio,
            gate_max_ess_ratio=self.tdcp_pu_gate_max_ess_ratio,
            gate_dd_pr_max_raw_median_m=self.tdcp_pu_gate_dd_pr_max_raw_median_m,
            gate_dd_cp_max_raw_afv_median_cycles=(
                self.tdcp_pu_gate_dd_cp_max_raw_afv_median_cycles
            ),
            gate_logic=self.tdcp_pu_gate_logic,
            gate_stop_mode=self.tdcp_pu_gate_stop_mode,
        )

    @property
    def observations(self) -> ObservationConfig:
        return ObservationConfig(
            robust=RobustMeasurementConfig(
                residual_downweight=self.residual_downweight,
                residual_threshold=self.residual_threshold,
                pr_accel_downweight=self.pr_accel_downweight,
                pr_accel_threshold=self.pr_accel_threshold,
                use_gmm=self.use_gmm,
                gmm_w_los=self.gmm_w_los,
                gmm_mu_nlos=self.gmm_mu_nlos,
                gmm_sigma_nlos=self.gmm_sigma_nlos,
                per_particle_nlos_gate=self.per_particle_nlos_gate,
                per_particle_nlos_dd_pr_threshold_m=self.per_particle_nlos_dd_pr_threshold_m,
                per_particle_nlos_dd_carrier_threshold_cycles=(
                    self.per_particle_nlos_dd_carrier_threshold_cycles
                ),
                per_particle_nlos_undiff_pr_threshold_m=(
                    self.per_particle_nlos_undiff_pr_threshold_m
                ),
                per_particle_huber=self.per_particle_huber,
                per_particle_huber_dd_pr_k=self.per_particle_huber_dd_pr_k,
                per_particle_huber_dd_carrier_k=self.per_particle_huber_dd_carrier_k,
                per_particle_huber_undiff_pr_k=self.per_particle_huber_undiff_pr_k,
            ),
            mupf=MupfConfig(
                enabled=self.mupf,
                sigma_cycles=self.mupf_sigma_cycles,
                snr_min=self.mupf_snr_min,
                elev_min=self.mupf_elev_min,
            ),
            dd_pseudorange=DDPseudorangeConfig(
                enabled=self.dd_pseudorange,
                sigma=self.dd_pseudorange_sigma,
                base_interp=self.dd_pseudorange_base_interp,
                gate_residual_m=self.dd_pseudorange_gate_residual_m,
                gate_adaptive_floor_m=self.dd_pseudorange_gate_adaptive_floor_m,
                gate_adaptive_mad_mult=self.dd_pseudorange_gate_adaptive_mad_mult,
                gate_epoch_median_m=self.dd_pseudorange_gate_epoch_median_m,
                gate_ess_min_scale=self.dd_pseudorange_gate_ess_min_scale,
                gate_ess_max_scale=self.dd_pseudorange_gate_ess_max_scale,
                gate_spread_min_scale=self.dd_pseudorange_gate_spread_min_scale,
                gate_spread_max_scale=self.dd_pseudorange_gate_spread_max_scale,
                gate_low_spread_m=self.dd_pseudorange_gate_low_spread_m,
                gate_high_spread_m=self.dd_pseudorange_gate_high_spread_m,
            ),
            widelane=WidelaneConfig(
                enabled=self.widelane,
                min_fix_rate=self.widelane_min_fix_rate,
                ratio_threshold=self.widelane_ratio_threshold,
                dd_sigma=self.widelane_dd_sigma,
                gate_min_fixed_pairs=self.widelane_gate_min_fixed_pairs,
                gate_min_fix_rate=self.widelane_gate_min_fix_rate,
                gate_min_spread_m=self.widelane_gate_min_spread_m,
                gate_max_epoch_median_residual_m=(
                    self.widelane_gate_max_epoch_median_residual_m
                ),
                gate_max_pair_residual_m=self.widelane_gate_max_pair_residual_m,
            ),
            dd_carrier=DDCarrierConfig(
                enabled=self.mupf_dd,
                sigma_cycles=self.mupf_dd_sigma_cycles,
                base_interp=self.mupf_dd_base_interp,
                gate_afv_cycles=self.mupf_dd_gate_afv_cycles,
                gate_adaptive_floor_cycles=self.mupf_dd_gate_adaptive_floor_cycles,
                gate_adaptive_mad_mult=self.mupf_dd_gate_adaptive_mad_mult,
                gate_epoch_median_cycles=self.mupf_dd_gate_epoch_median_cycles,
                gate_low_ess_epoch_median_cycles=(
                    self.mupf_dd_gate_low_ess_epoch_median_cycles
                ),
                gate_low_ess_max_ratio=self.mupf_dd_gate_low_ess_max_ratio,
                gate_low_ess_max_spread_m=self.mupf_dd_gate_low_ess_max_spread_m,
                gate_low_ess_require_no_dd_pr=self.mupf_dd_gate_low_ess_require_no_dd_pr,
                gate_ess_min_scale=self.mupf_dd_gate_ess_min_scale,
                gate_ess_max_scale=self.mupf_dd_gate_ess_max_scale,
                gate_spread_min_scale=self.mupf_dd_gate_spread_min_scale,
                gate_spread_max_scale=self.mupf_dd_gate_spread_max_scale,
                gate_low_spread_m=self.mupf_dd_gate_low_spread_m,
                gate_high_spread_m=self.mupf_dd_gate_high_spread_m,
                sigma_support_low_pairs=self.mupf_dd_sigma_support_low_pairs,
                sigma_support_high_pairs=self.mupf_dd_sigma_support_high_pairs,
                sigma_support_max_scale=self.mupf_dd_sigma_support_max_scale,
                sigma_afv_good_cycles=self.mupf_dd_sigma_afv_good_cycles,
                sigma_afv_bad_cycles=self.mupf_dd_sigma_afv_bad_cycles,
                sigma_afv_max_scale=self.mupf_dd_sigma_afv_max_scale,
                sigma_ess_low_ratio=self.mupf_dd_sigma_ess_low_ratio,
                sigma_ess_high_ratio=self.mupf_dd_sigma_ess_high_ratio,
                sigma_ess_max_scale=self.mupf_dd_sigma_ess_max_scale,
                sigma_max_scale=self.mupf_dd_sigma_max_scale,
            ),
            carrier_rescue=CarrierRescueConfig(
                anchor_enabled=self.carrier_anchor,
                anchor_sigma_m=self.carrier_anchor_sigma_m,
                anchor_min_sats=self.carrier_anchor_min_sats,
                anchor_max_age_s=self.carrier_anchor_max_age_s,
                anchor_max_residual_m=self.carrier_anchor_max_residual_m,
                anchor_max_continuity_residual_m=(
                    self.carrier_anchor_max_continuity_residual_m
                ),
                anchor_min_stable_epochs=self.carrier_anchor_min_stable_epochs,
                anchor_blend_alpha=self.carrier_anchor_blend_alpha,
                anchor_reanchor_jump_cycles=self.carrier_anchor_reanchor_jump_cycles,
                anchor_seed_dd_min_pairs=self.carrier_anchor_seed_dd_min_pairs,
                fallback_undiff=self.mupf_dd_fallback_undiff,
                fallback_sigma_cycles=self.mupf_dd_fallback_sigma_cycles,
                fallback_min_sats=self.mupf_dd_fallback_min_sats,
                fallback_prefer_tracked=self.mupf_dd_fallback_prefer_tracked,
                fallback_tracked_min_stable_epochs=(
                    self.mupf_dd_fallback_tracked_min_stable_epochs
                ),
                fallback_tracked_min_sats=self.mupf_dd_fallback_tracked_min_sats,
                fallback_tracked_continuity_good_m=(
                    self.mupf_dd_fallback_tracked_continuity_good_m
                ),
                fallback_tracked_continuity_bad_m=(
                    self.mupf_dd_fallback_tracked_continuity_bad_m
                ),
                fallback_tracked_sigma_min_scale=(
                    self.mupf_dd_fallback_tracked_sigma_min_scale
                ),
                fallback_tracked_sigma_max_scale=(
                    self.mupf_dd_fallback_tracked_sigma_max_scale
                ),
                fallback_weak_dd_max_pairs=self.mupf_dd_fallback_weak_dd_max_pairs,
                fallback_weak_dd_max_ess_ratio=(
                    self.mupf_dd_fallback_weak_dd_max_ess_ratio
                ),
                fallback_weak_dd_min_raw_afv_median_cycles=(
                    self.mupf_dd_fallback_weak_dd_min_raw_afv_median_cycles
                ),
                fallback_weak_dd_require_no_dd_pr=(
                    self.mupf_dd_fallback_weak_dd_require_no_dd_pr
                ),
                skip_low_support_ess_ratio=self.mupf_dd_skip_low_support_ess_ratio,
                skip_low_support_max_pairs=self.mupf_dd_skip_low_support_max_pairs,
                skip_low_support_max_spread_m=self.mupf_dd_skip_low_support_max_spread_m,
                skip_low_support_min_raw_afv_median_cycles=(
                    self.mupf_dd_skip_low_support_min_raw_afv_median_cycles
                ),
                skip_low_support_require_no_dd_pr=(
                    self.mupf_dd_skip_low_support_require_no_dd_pr
                ),
            ),
        )

    @property
    def smoother(self) -> SmootherPostprocessConfig:
        return SmootherPostprocessConfig(
            position_update_sigma=self.smoother_position_update_sigma,
            skip_widelane_dd_pseudorange=self.smoother_skip_widelane_dd_pseudorange,
            widelane_forward_guard=self.smoother_widelane_forward_guard,
            widelane_forward_guard_min_shift_m=(
                self.smoother_widelane_forward_guard_min_shift_m
            ),
            stop_segment_constant=self.stop_segment_constant,
            stop_segment_min_epochs=self.stop_segment_min_epochs,
            stop_segment_source=self.stop_segment_source,
            stop_segment_max_radius_m=self.stop_segment_max_radius_m,
            stop_segment_blend=self.stop_segment_blend,
            stop_segment_density_neighbors=self.stop_segment_density_neighbors,
            stop_segment_static_gnss=self.stop_segment_static_gnss,
            stop_segment_static_min_observations=(
                self.stop_segment_static_min_observations
            ),
            stop_segment_static_prior_sigma_m=self.stop_segment_static_prior_sigma_m,
            stop_segment_static_pr_sigma_m=self.stop_segment_static_pr_sigma_m,
            stop_segment_static_dd_pr_sigma_m=self.stop_segment_static_dd_pr_sigma_m,
            stop_segment_static_dd_cp_sigma_cycles=(
                self.stop_segment_static_dd_cp_sigma_cycles
            ),
            stop_segment_static_max_update_m=self.stop_segment_static_max_update_m,
            stop_segment_static_blend=self.stop_segment_static_blend,
            tail_guard_ess_max_ratio=self.smoother_tail_guard_ess_max_ratio,
            tail_guard_dd_carrier_max_pairs=self.smoother_tail_guard_dd_carrier_max_pairs,
            tail_guard_dd_pseudorange_max_pairs=(
                self.smoother_tail_guard_dd_pseudorange_max_pairs
            ),
            tail_guard_min_shift_m=self.smoother_tail_guard_min_shift_m,
            tail_guard_expand_epochs=self.smoother_tail_guard_expand_epochs,
            tail_guard_expand_min_shift_m=self.smoother_tail_guard_expand_min_shift_m,
            tail_guard_expand_dd_pseudorange_max_pairs=(
                self.smoother_tail_guard_expand_dd_pseudorange_max_pairs
            ),
        )

    @property
    def local_fgo(self) -> LocalFgoPostprocessConfig:
        return LocalFgoPostprocessConfig(
            window=self.fgo_local_window,
            window_min_epochs=self.fgo_local_window_min_epochs,
            dd_max_pairs=self.fgo_local_dd_max_pairs,
            prior_sigma_m=self.fgo_local_prior_sigma_m,
            motion_sigma_m=self.fgo_local_motion_sigma_m,
            dd_huber_k=self.fgo_local_dd_huber_k,
            pr_huber_k=self.fgo_local_pr_huber_k,
            dd_sigma_cycles=self.fgo_local_dd_sigma_cycles,
            pr_sigma_m=self.fgo_local_pr_sigma_m,
            max_iterations=self.fgo_local_max_iterations,
            lambda_enabled=self.fgo_local_lambda,
            lambda_ratio_threshold=self.fgo_local_lambda_ratio_threshold,
            lambda_sigma_cycles=self.fgo_local_lambda_sigma_cycles,
            lambda_min_epochs=self.fgo_local_lambda_min_epochs,
            motion_source=self.fgo_local_motion_source,
            tdcp_rms_max_m=self.fgo_local_tdcp_rms_max_m,
            tdcp_spp_max_diff_mps=self.fgo_local_tdcp_spp_max_diff_mps,
            two_step=self.fgo_local_two_step,
            stage1_prior_sigma_m=self.fgo_local_stage1_prior_sigma_m,
            stage1_motion_sigma_m=self.fgo_local_stage1_motion_sigma_m,
            stage1_pr_sigma_m=self.fgo_local_stage1_pr_sigma_m,
        )

    @property
    def diagnostics(self) -> DiagnosticsConfig:
        return DiagnosticsConfig(
            collect_epoch_diagnostics=self.collect_epoch_diagnostics,
        )

    def parts(self) -> PfSmootherConfigParts:
        return PfSmootherConfigParts(
            run_selection=self.run_selection,
            particle_filter=self.particle_filter,
            motion=self.motion,
            doppler=self.doppler,
            tdcp_position_update=self.tdcp_position,
            observations=self.observations,
            smoother=self.smoother,
            local_fgo=self.local_fgo,
            diagnostics=self.diagnostics,
        )

    def to_kwargs(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def with_overrides(self, **overrides: Any) -> "PfSmootherConfig":
        return replace(self, **overrides)


def coerce_pf_smoother_config(
    config: PfSmootherConfig | None,
    overrides: dict[str, Any] | None = None,
) -> PfSmootherConfig:
    override_values = dict(overrides or {})
    if config is None:
        return PfSmootherConfig(**override_values)
    if override_values:
        return config.with_overrides(**override_values)
    return config


def validate_pf_smoother_config(config: PfSmootherConfig) -> None:
    parts = config.parts()
    if parts.observations.dd_pseudorange.enabled and parts.observations.robust.use_gmm:
        raise ValueError("dd_pseudorange cannot be combined with --gmm")
    if parts.doppler.rbpf_velocity_kf and parts.doppler.per_particle:
        raise ValueError("--rbpf-velocity-kf and --doppler-per-particle are mutually exclusive")
    fgo_motion_source = str(parts.local_fgo.motion_source).strip().lower()
    stop_segment_source = (
        str(parts.smoother.stop_segment_source).strip().lower().replace("-", "_")
    )
    tdcp_gate_logic = str(parts.tdcp_position_update.gate_logic).strip().lower()
    tdcp_gate_stop_mode = str(parts.tdcp_position_update.gate_stop_mode).strip().lower()
    if fgo_motion_source not in {"predict", "tdcp", "prefer_tdcp"}:
        raise ValueError(
            "fgo_local_motion_source must be one of: predict, tdcp, prefer_tdcp"
        )
    if tdcp_gate_logic not in {"any", "all"}:
        raise ValueError("tdcp_pu_gate_logic must be one of: any, all")
    if tdcp_gate_stop_mode not in {"any", "stopped", "moving"}:
        raise ValueError("tdcp_pu_gate_stop_mode must be one of: any, stopped, moving")
    if stop_segment_source not in {
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
    }:
        raise ValueError(
            "stop_segment_source must be one of: smoothed, forward, combined, "
            "smoothed_density, forward_density, combined_density, "
            "smoothed_auto, forward_auto, combined_auto, "
            "smoothed_auto_tail, forward_auto_tail, combined_auto_tail"
        )
    if int(parts.smoother.stop_segment_density_neighbors) < 1:
        raise ValueError("stop_segment_density_neighbors must be >= 1")
    if (
        parts.smoother.tail_guard_expand_epochs is not None
        and int(parts.smoother.tail_guard_expand_epochs) < 0
    ):
        raise ValueError("tail_guard_expand_epochs must be >= 0")
    if (
        parts.smoother.tail_guard_expand_min_shift_m is not None
        and float(parts.smoother.tail_guard_expand_min_shift_m) < 0.0
    ):
        raise ValueError("tail_guard_expand_min_shift_m must be >= 0")
    if (
        parts.smoother.tail_guard_expand_dd_pseudorange_max_pairs is not None
        and int(parts.smoother.tail_guard_expand_dd_pseudorange_max_pairs) < 0
    ):
        raise ValueError("tail_guard_expand_dd_pseudorange_max_pairs must be >= 0")
