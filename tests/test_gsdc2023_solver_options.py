from __future__ import annotations

from experiments.gsdc2023_bridge_config import (
    TAROZ_CLOCK_DRIFT_SIGMA_M,
    TAROZ_FGO_WEIGHT_MODE,
    TAROZ_HEIGHT_SIGMA_M,
    TAROZ_STOP_ATTITUDE_SIGMA_RAD,
    TAROZ_STOP_POSITION_SIGMA_M,
    TAROZ_STOP_VELOCITY_SIGMA_MPS,
    BridgeConfig,
    apply_taroz_fgo_preset,
    apply_taroz_full_init_pass_preset,
    apply_taroz_gnss_only_preset,
    apply_taroz_marupaku_preset,
)
from experiments.gsdc2023_solver_options import FgoRunOptions, fgo_run_options_from_config


def test_fgo_run_options_from_config_maps_solver_fields() -> None:
    cfg = BridgeConfig(
        motion_sigma_m=4.5,
        clock_drift_sigma_m=0.25,
        fgo_iters=13,
        fgo_tol=2e-6,
        fgo_line_search=False,
        fgo_lm_damping=0.125,
        chunk_epochs=17,
        use_vd=False,
        stop_velocity_sigma_mps=0.4,
        stop_position_sigma_m=0.9,
        stop_attitude_sigma_rad=0.03,
        stop_velocity_huber_k=0.5,
        stop_position_huber_k=0.6,
        apply_imu_prior=True,
        imu_position_sigma_m=12.0,
        imu_velocity_sigma_mps=1.5,
        imu_attitude_state=True,
        imu_attitude_sigma_rad=0.002,
        imu_preintegration_velocity_noise_mps_sqrt_hz=1.25,
        imu_preintegration_attitude_noise_rad_sqrt_hz=0.003,
        imu_diagonal_covariance=True,
        imu_preintegration_covariance=True,
        imu_factor_use_next_bias=True,
        imu_bias_between_sample_count_scaling=True,
        imu_accel_bias_state=True,
        imu_accel_bias_prior_sigma_mps2=0.03,
        imu_accel_bias_between_sigma_mps2=0.04,
        imu_gyro_bias_state=True,
        imu_gyro_bias_prior_sigma_radps=0.003,
        imu_gyro_bias_between_sigma_radps=0.004,
        graph_relative_height=True,
        relative_height_sigma_m=0.7,
        relative_height_huber_k=0.8,
        apply_absolute_height=True,
        absolute_height_sigma_m=8.0,
        absolute_height_huber_k=0.9,
    )

    options = fgo_run_options_from_config(cfg)

    assert options == FgoRunOptions(
        motion_sigma_m=4.5,
        clock_drift_sigma_m=0.25,
        stop_velocity_sigma_mps=0.4,
        stop_position_sigma_m=0.9,
        stop_attitude_sigma_rad=0.03,
        stop_velocity_huber_k=0.5,
        stop_position_huber_k=0.6,
        apply_imu_prior=True,
        imu_position_sigma_m=12.0,
        imu_velocity_sigma_mps=1.5,
        imu_attitude_state=True,
        imu_attitude_sigma_rad=0.002,
        imu_preintegration_velocity_noise_mps_sqrt_hz=1.25,
        imu_preintegration_attitude_noise_rad_sqrt_hz=0.003,
        imu_diagonal_covariance=True,
        imu_preintegration_covariance=True,
        imu_factor_use_next_bias=True,
        imu_bias_between_sample_count_scaling=True,
        imu_accel_bias_state=True,
        imu_accel_bias_prior_sigma_mps2=0.03,
        imu_accel_bias_between_sigma_mps2=0.04,
        imu_gyro_bias_state=True,
        imu_gyro_bias_prior_sigma_radps=0.003,
        imu_gyro_bias_between_sigma_radps=0.004,
        fgo_iters=13,
        tol=2e-6,
        fgo_line_search=False,
        fgo_lm_damping=0.125,
        chunk_epochs=17,
        use_vd=False,
        graph_relative_height=True,
        relative_height_sigma_m=0.7,
        relative_height_huber_k=0.8,
        apply_absolute_height=True,
        absolute_height_sigma_m=8.0,
        absolute_height_huber_k=0.9,
    )


def test_fgo_run_options_kwargs_match_run_fgo_chunked_solver_arguments() -> None:
    cfg = BridgeConfig(apply_imu_prior=True, graph_relative_height=True, apply_absolute_height=True)

    kwargs = fgo_run_options_from_config(cfg).run_kwargs()

    assert set(kwargs) == {
        "motion_sigma_m",
        "clock_drift_sigma_m",
        "stop_velocity_sigma_mps",
        "stop_position_sigma_m",
        "stop_attitude_sigma_rad",
        "stop_velocity_huber_k",
        "stop_position_huber_k",
        "apply_imu_prior",
        "imu_position_sigma_m",
        "imu_velocity_sigma_mps",
        "imu_attitude_state",
        "imu_attitude_sigma_rad",
        "imu_preintegration_velocity_noise_mps_sqrt_hz",
        "imu_preintegration_attitude_noise_rad_sqrt_hz",
        "imu_diagonal_covariance",
        "imu_preintegration_covariance",
        "imu_factor_use_next_bias",
        "imu_bias_between_sample_count_scaling",
        "imu_accel_bias_state",
        "imu_accel_bias_prior_sigma_mps2",
        "imu_accel_bias_between_sigma_mps2",
        "imu_gyro_bias_state",
        "imu_gyro_bias_prior_sigma_radps",
        "imu_gyro_bias_between_sigma_radps",
        "fgo_iters",
        "tol",
        "fgo_line_search",
        "fgo_lm_damping",
        "chunk_epochs",
        "use_vd",
        "graph_relative_height",
        "relative_height_sigma_m",
        "relative_height_huber_k",
        "apply_absolute_height",
        "absolute_height_sigma_m",
        "absolute_height_huber_k",
        "fgo_robust_kernel",
        "fgo_cauchy_c_m",
        "fgo_cauchy_outer_iters",
        "fgo_huber_k_pr",
        "fgo_huber_k_doppler",
        "fgo_huber_k_tdcp",
        "fgo_fixed_linearization",
    }
    assert kwargs["tol"] == 1e-7
    assert kwargs["fgo_line_search"] is True
    assert kwargs["fgo_lm_damping"] == 0.0
    assert kwargs["apply_imu_prior"] is True
    assert kwargs["imu_factor_use_next_bias"] is False
    assert kwargs["imu_bias_between_sample_count_scaling"] is False
    assert kwargs["graph_relative_height"] is True
    assert kwargs["apply_absolute_height"] is True
    assert kwargs["stop_velocity_huber_k"] == 0.0
    assert kwargs["stop_position_huber_k"] == 0.0
    assert kwargs["relative_height_huber_k"] == 0.0
    assert kwargs["absolute_height_huber_k"] == 0.0
    assert kwargs["fgo_robust_kernel"] == "huber"
    assert kwargs["fgo_huber_k_pr"] == 0.0
    assert kwargs["fgo_huber_k_doppler"] == 0.0
    assert kwargs["fgo_huber_k_tdcp"] == 0.0
    assert kwargs["fgo_fixed_linearization"] is False


def test_fgo_run_options_allows_explicit_tol_override() -> None:
    cfg = BridgeConfig(fgo_tol=2e-6)

    options = fgo_run_options_from_config(cfg, tol=1e-10)

    assert options.tol == 1e-10


def test_apply_taroz_fgo_preset_enables_parameters_m_levers() -> None:
    cfg = apply_taroz_fgo_preset(
        BridgeConfig(
            weight_mode="sin2el",
            fgo_weight_mode=None,
            clock_drift_sigma_m=1.0,
            stop_velocity_sigma_mps=0.0,
            stop_position_sigma_m=0.0,
            graph_relative_height=False,
            apply_absolute_height=False,
            per_type_kernel_enabled=False,
            per_type_kernel_motion_enabled=False,
        ),
    )

    assert cfg.weight_mode == "sin2el"
    assert cfg.dual_frequency is True
    assert cfg.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert cfg.clock_drift_sigma_m == TAROZ_CLOCK_DRIFT_SIGMA_M
    assert cfg.clock_use_average_drift is True
    assert cfg.stop_velocity_sigma_mps == TAROZ_STOP_VELOCITY_SIGMA_MPS
    assert cfg.stop_position_sigma_m == TAROZ_STOP_POSITION_SIGMA_M
    assert cfg.stop_attitude_sigma_rad == TAROZ_STOP_ATTITUDE_SIGMA_RAD
    assert cfg.stop_velocity_huber_k == 0.5
    assert cfg.stop_position_huber_k == 0.5
    assert cfg.per_type_kernel_enabled is True
    assert cfg.per_type_kernel_huber_enabled is True
    assert cfg.per_type_kernel_motion_enabled is True
    assert cfg.fgo_fixed_linearization is True
    assert cfg.graph_relative_height is True
    assert cfg.relative_height_sigma_m == TAROZ_HEIGHT_SIGMA_M
    assert cfg.relative_height_huber_k == 0.5
    assert cfg.apply_absolute_height is True
    assert cfg.absolute_height_sigma_m == TAROZ_HEIGHT_SIGMA_M
    assert cfg.absolute_height_huber_k == 0.5
    assert cfg.apply_observation_mask is True
    assert cfg.observation_min_cn0_dbhz == 20.0
    assert cfg.observation_min_elevation_deg == 5.0
    assert cfg.pseudorange_residual_mask_m == 20.0
    assert cfg.pseudorange_residual_mask_l5_m == 15.0
    assert cfg.doppler_residual_mask_mps == 3.0
    assert cfg.pseudorange_doppler_mask_m == 40.0


def test_apply_taroz_fgo_preset_uses_l5_elevation_mask_for_dual_frequency() -> None:
    cfg = apply_taroz_fgo_preset(BridgeConfig(dual_frequency=True))

    assert cfg.apply_observation_mask is True
    assert cfg.observation_min_elevation_deg == 5.0


def test_apply_taroz_marupaku_preset_forces_fgo_source() -> None:
    cfg = apply_taroz_marupaku_preset(
        BridgeConfig(
            position_source="gated",
            fgo_weight_mode=None,
            clock_drift_sigma_m=1.0,
            graph_relative_height=False,
            apply_absolute_height=False,
        ),
    )

    assert cfg.position_source == "fgo"
    assert cfg.dual_frequency is True
    assert cfg.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert cfg.clock_drift_sigma_m == TAROZ_CLOCK_DRIFT_SIGMA_M
    assert cfg.graph_relative_height is True
    assert cfg.apply_absolute_height is True
    assert cfg.apply_imu_prior is True
    assert cfg.imu_frame == "taroz_body"
    assert cfg.imu_sample_dt_mode == "taroz"
    assert cfg.imu_position_sigma_m == 0.05
    assert cfg.imu_velocity_sigma_mps == 0.025
    assert cfg.imu_attitude_state is True
    assert cfg.imu_attitude_sigma_rad == 0.0005
    assert cfg.imu_preintegration_velocity_noise_mps_sqrt_hz == 0.05
    assert cfg.imu_preintegration_attitude_noise_rad_sqrt_hz == 0.001
    assert cfg.imu_diagonal_covariance is True
    assert cfg.imu_preintegration_covariance is True
    assert cfg.imu_factor_use_next_bias is True
    assert cfg.imu_bias_between_sample_count_scaling is True
    assert cfg.imu_accel_bias_state is True
    assert cfg.imu_accel_bias_prior_sigma_mps2 == 0.0
    assert cfg.imu_accel_bias_between_sigma_mps2 == 0.00025
    assert cfg.imu_gyro_bias_state is True
    assert cfg.imu_gyro_bias_prior_sigma_radps == 0.0
    assert cfg.taroz_imu_noise_enabled is True
    assert cfg.taroz_stop_mask_from_seed_velocity is True
    assert cfg.imu_acc_sigma_mps2_sqrt_hz == 0.05
    assert cfg.imu_gyro_sigma_radps_sqrt_hz == 0.001
    assert cfg.imu_acc_sync_coefficient == 0.5
    assert cfg.imu_gyro_sync_coefficient == 0.5
    assert cfg.imu_gyro_bias_between_sigma_radps == 0.0000005
    assert cfg.stop_attitude_sigma_rad == TAROZ_STOP_ATTITUDE_SIGMA_RAD
    assert cfg.stop_velocity_huber_k == 0.5
    assert cfg.stop_position_huber_k == 0.5
    assert cfg.relative_height_huber_k == 0.5
    assert cfg.absolute_height_huber_k == 0.5


def test_apply_taroz_full_init_pass_preset_matches_initflag_true_factor_set() -> None:
    cfg = apply_taroz_full_init_pass_preset(apply_taroz_marupaku_preset(BridgeConfig()))

    assert cfg.stop_velocity_sigma_mps == 0.0
    assert cfg.stop_velocity_huber_k == 0.0
    assert cfg.stop_position_sigma_m == TAROZ_STOP_POSITION_SIGMA_M
    assert cfg.stop_attitude_sigma_rad == TAROZ_STOP_ATTITUDE_SIGMA_RAD
    assert cfg.graph_relative_height is False
    assert cfg.apply_absolute_height is False
    assert cfg.apply_relative_height is False
    assert cfg.relative_height_huber_k == 0.0
    assert cfg.absolute_height_huber_k == 0.0
    assert cfg.apply_imu_prior is True
    assert cfg.imu_accel_bias_state is True
    assert cfg.imu_gyro_bias_state is True


def test_apply_taroz_gnss_only_preset_excludes_imu_stop_height_priors() -> None:
    cfg = apply_taroz_gnss_only_preset(
        BridgeConfig(
            weight_mode="sin2el",
            fgo_weight_mode=None,
            clock_drift_sigma_m=1.0,
            stop_velocity_sigma_mps=3.0,
            stop_position_sigma_m=4.0,
            apply_imu_prior=True,
            imu_accel_bias_state=True,
            graph_relative_height=True,
            apply_absolute_height=True,
            apply_relative_height=True,
            per_type_kernel_enabled=False,
            per_type_kernel_motion_enabled=False,
        ),
    )

    assert cfg.weight_mode == "sin2el"
    assert cfg.dual_frequency is True
    assert cfg.fgo_weight_mode == TAROZ_FGO_WEIGHT_MODE
    assert cfg.clock_drift_sigma_m == TAROZ_CLOCK_DRIFT_SIGMA_M
    assert cfg.clock_use_average_drift is True
    assert cfg.fgo_fixed_linearization is True
    assert cfg.stop_velocity_sigma_mps == 0.0
    assert cfg.stop_position_sigma_m == 0.0
    assert cfg.stop_attitude_sigma_rad == 0.0
    assert cfg.taroz_stop_mask_from_seed_velocity is False
    assert cfg.stop_velocity_huber_k == 0.0
    assert cfg.stop_position_huber_k == 0.0
    assert cfg.apply_imu_prior is False
    assert cfg.imu_attitude_state is False
    assert cfg.imu_diagonal_covariance is False
    assert cfg.imu_preintegration_covariance is False
    assert cfg.imu_factor_use_next_bias is False
    assert cfg.imu_bias_between_sample_count_scaling is False
    assert cfg.imu_accel_bias_state is False
    assert cfg.imu_gyro_bias_state is False
    assert cfg.per_type_kernel_enabled is True
    assert cfg.per_type_kernel_huber_enabled is True
    assert cfg.per_type_kernel_motion_enabled is True
    assert cfg.graph_relative_height is False
    assert cfg.relative_height_huber_k == 0.0
    assert cfg.apply_absolute_height is False
    assert cfg.absolute_height_huber_k == 0.0
    assert cfg.apply_relative_height is False
    assert cfg.apply_observation_mask is True
    assert cfg.pseudorange_residual_mask_l5_m == 15.0


def test_apply_taroz_gnss_only_preset_uses_l5_elevation_mask_for_dual_frequency() -> None:
    cfg = apply_taroz_gnss_only_preset(BridgeConfig(dual_frequency=True))

    assert cfg.apply_observation_mask is True
    assert cfg.observation_min_elevation_deg == 5.0
