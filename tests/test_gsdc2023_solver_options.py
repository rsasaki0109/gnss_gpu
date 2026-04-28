from __future__ import annotations

from experiments.gsdc2023_bridge_config import BridgeConfig
from experiments.gsdc2023_solver_options import FgoRunOptions, fgo_run_options_from_config


def test_fgo_run_options_from_config_maps_solver_fields() -> None:
    cfg = BridgeConfig(
        motion_sigma_m=4.5,
        clock_drift_sigma_m=0.25,
        fgo_iters=13,
        chunk_epochs=17,
        use_vd=False,
        stop_velocity_sigma_mps=0.4,
        stop_position_sigma_m=0.9,
        apply_imu_prior=True,
        imu_position_sigma_m=12.0,
        imu_velocity_sigma_mps=1.5,
        imu_accel_bias_state=True,
        imu_accel_bias_prior_sigma_mps2=0.03,
        imu_accel_bias_between_sigma_mps2=0.04,
        graph_relative_height=True,
        relative_height_sigma_m=0.7,
        apply_absolute_height=True,
        absolute_height_sigma_m=8.0,
    )

    options = fgo_run_options_from_config(cfg, tol=1e-5)

    assert options == FgoRunOptions(
        motion_sigma_m=4.5,
        clock_drift_sigma_m=0.25,
        stop_velocity_sigma_mps=0.4,
        stop_position_sigma_m=0.9,
        apply_imu_prior=True,
        imu_position_sigma_m=12.0,
        imu_velocity_sigma_mps=1.5,
        imu_accel_bias_state=True,
        imu_accel_bias_prior_sigma_mps2=0.03,
        imu_accel_bias_between_sigma_mps2=0.04,
        fgo_iters=13,
        tol=1e-5,
        chunk_epochs=17,
        use_vd=False,
        graph_relative_height=True,
        relative_height_sigma_m=0.7,
        apply_absolute_height=True,
        absolute_height_sigma_m=8.0,
    )


def test_fgo_run_options_kwargs_match_run_fgo_chunked_solver_arguments() -> None:
    cfg = BridgeConfig(apply_imu_prior=True, graph_relative_height=True, apply_absolute_height=True)

    kwargs = fgo_run_options_from_config(cfg).run_kwargs()

    assert set(kwargs) == {
        "motion_sigma_m",
        "clock_drift_sigma_m",
        "stop_velocity_sigma_mps",
        "stop_position_sigma_m",
        "apply_imu_prior",
        "imu_position_sigma_m",
        "imu_velocity_sigma_mps",
        "imu_accel_bias_state",
        "imu_accel_bias_prior_sigma_mps2",
        "imu_accel_bias_between_sigma_mps2",
        "fgo_iters",
        "tol",
        "chunk_epochs",
        "use_vd",
        "graph_relative_height",
        "relative_height_sigma_m",
        "apply_absolute_height",
        "absolute_height_sigma_m",
    }
    assert kwargs["tol"] == 1e-7
    assert kwargs["apply_imu_prior"] is True
    assert kwargs["graph_relative_height"] is True
    assert kwargs["apply_absolute_height"] is True
