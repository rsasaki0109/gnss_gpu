from __future__ import annotations

from pathlib import Path

import pytest

from experiments.patch_taroz_fgo_imu_export import (
    GRAPH_INIT_ANCHOR,
    IMU_FACTOR_ANCHOR,
    MAX_ITER_ANCHOR,
    OPTIMIZER_ANCHOR,
    OPTSTATUS_ANCHOR,
    main,
    patch_fgo_gnss_imu_text,
)
from experiments.patch_taroz_fgo_gnss_export import (
    DOPPLER_FACTOR_ANCHOR,
    PSEUDORANGE_FACTOR_ANCHOR,
    TDCP_XXCC_ANCHOR,
    TDCP_XXDD_ANCHOR,
    TDCP_XXDD_OFFSET_ANCHOR,
)


def _minimal_fgo_text() -> str:
    return "\n".join(
        [
            "function optstatus = fgo_gnss_imu(datapath, setting, initflag)",
            GRAPH_INIT_ANCHOR.rstrip("\n"),
            "% unrelated graph setup",
            PSEUDORANGE_FACTOR_ANCHOR.rstrip("\n"),
            DOPPLER_FACTOR_ANCHOR.rstrip("\n"),
            TDCP_XXDD_OFFSET_ANCHOR.rstrip("\n"),
            TDCP_XXDD_ANCHOR.rstrip("\n"),
            TDCP_XXCC_ANCHOR.rstrip("\n"),
            IMU_FACTOR_ANCHOR.rstrip("\n"),
            "optparameters = gtsam.LevenbergMarquardtParams;",
            MAX_ITER_ANCHOR.rstrip("\n"),
            OPTIMIZER_ANCHOR.rstrip("\n"),
            "% Optimize!",
            OPTSTATUS_ANCHOR.rstrip("\n"),
            'save(fname,"posest","clkest","velest","dclkest","imubiasest","rpyest");',
            "",
        ]
    )


def test_patch_fgo_gnss_imu_text_inserts_imu_export_hooks() -> None:
    patched = patch_fgo_gnss_imu_text(_minimal_fgo_text())

    assert "imu_export_factors = {};" in patched
    assert "imu_export_preintegrations = {};" in patched
    assert "gnss_export_factors = {};" in patched
    assert "phone_data_gnss_factor_mask.csv" in patched
    assert "gnss_factor = gtsam_gnss.PseudorangeFactor_XC" in patched
    assert "gnss_factor = gtsam_gnss.TDCPFactor_XXCC" in patched
    assert "gnss_initial_pose = initials.atPose3(sym('p', gnss_state_epoch));" in patched
    assert "'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw'" in patched
    assert "imu_factor = gtsam.ImuFactor(keyP1, keyV1, keyP2, keyV2, keyB2, currentSummarizedMeasurement);" in patched
    assert "imu_export_factors{end + 1, 1} = imu_factor;" in patched
    assert "imu_export_preintegrations{end + 1, 1} = currentSummarizedMeasurement;" in patched
    assert "currentSummarizedMeasurement.deltaPij()'" in patched
    assert "gtsam.Rot3.Logmap(currentSummarizedMeasurement.deltaRij())'" in patched
    assert 'phone_data_imu_factor_mask.csv' in patched
    assert 'phone_data_imu_residual_diagnostics.csv' in patched
    assert 'phone_data_imu_state.csv' in patched
    assert 'phone_data_imu_preintegration.csv' in patched
    assert "imu_state_clock_col(imu_state_idx, :) = results.atVector(sym('c', imu_state_epoch))';" in patched
    assert "imu_state_drift_col(imu_state_idx, :) = results.atVector(sym('d', imu_state_epoch))';" in patched
    assert "'clock_bias_m_0', 'clock_bias_m_1', 'clock_bias_m_2', 'clock_bias_m_3'" in patched
    assert "'clock_bias_m_4', 'clock_bias_m_5', 'clock_bias_m_6', 'clock_drift_mps'" in patched
    assert 'imu_block_names = ["IMU_R", "IMU_P", "IMU_V"];' in patched
    assert "results.atConstantBias(sym('b', imu_next_epoch))" in patched
    assert "gtsam.NavState(gtsam.Pose3(), zeros(3, 1))" in patched
    assert "imu_export_preintegrations{imu_export_idx}.predict(imu_zero_nav, imu_bias)" in patched
    assert "corrected_delta_p_x" in patched
    assert "currentSummarizedMeasurement.delPdelBiasAcc()" in patched
    assert "currentSummarizedMeasurement.delVdelBiasOmega()" in patched
    assert "currentSummarizedMeasurement.delRdelBiasOmega()" in patched
    assert "currentSummarizedMeasurement.preintMeasCov()" in patched
    assert "imu_export_fd_bias_plus = gtsam.imuBias.ConstantBias" in patched
    assert "currentSummarizedMeasurement.predict(imu_export_fd_nav0, imu_export_fd_bias_plus)" in patched
    assert '"delta_p_bias_accel_jac", "delta_v_bias_accel_jac"' in patched
    assert "imu_preintegration_jac_prefixes(imu_prefix_idx) + \"_\" +" in patched
    assert "preint_meas_cov_\" + string(imu_cov_row)" in patched
    assert "imu_preintegration_table = [imu_preintegration_table, imu_preintegration_extra_table];" in patched
    assert "'sample_count', 'graph_dt_s', 'preintegrated_dt_s', 'residual'" in patched
    assert 'imu_export_max_iter = string(getenv("GSDC2023_LM_MAX_ITER"));' in patched
    assert "optparameters.setMaxIterations(str2double(imu_export_max_iter));" in patched
    assert 'imu_export_lm_verbosity = string(getenv("GSDC2023_LM_VERBOSITY"));' in patched
    assert "optparameters.setVerbosityLM(char(imu_export_lm_verbosity));" in patched
    assert 'imu_export_lm_log_file = string(getenv("GSDC2023_LM_LOG_FILE"));' in patched
    assert "optparameters.setLogFile(char(imu_export_lm_log_file));" in patched
    assert 'imu_linear_prefix = string(getenv("GSDC2023_LINEAR_EXPORT_PREFIX"));' in patched
    assert "imu_linear_ordering.push_back(sym('x', imu_linear_i));" in patched
    assert "imu_linear_ordering.push_back(sym('b', imu_linear_i));" in patched
    assert "imu_linear_graph = graph.linearize(initials);" in patched
    assert "imu_linear_augmented = imu_linear_graph.augmentedJacobian(imu_linear_ordering);" in patched
    assert "writematrix(imu_linear_hessian, imu_linear_prefix + \"_H.csv\");" in patched
    assert "initials.atPose3(sym('p', imu_state_epoch)).localCoordinates(results.atPose3(sym('p', imu_state_epoch)))" in patched
    assert "writematrix(imu_delta_state, imu_linear_prefix + \"_delta_gtsam_order.csv\");" in patched


def test_patch_fgo_gnss_imu_text_rejects_missing_anchor() -> None:
    with pytest.raises(ValueError, match="IMU factor anchor"):
        patch_fgo_gnss_imu_text(_minimal_fgo_text().replace(IMU_FACTOR_ANCHOR.rstrip("\n"), "% missing"))


def test_patch_cli_writes_patched_copy(tmp_path: Path) -> None:
    source = tmp_path / "fgo_gnss_imu.m"
    output = tmp_path / "patched" / "fgo_gnss_imu.m"
    source.write_text(_minimal_fgo_text(), encoding="utf-8")

    main([str(source), str(output)])

    patched = output.read_text(encoding="utf-8")
    assert "imu_export_preintegrated_dt_s" in patched
    assert "writetable(imu_state_table, imu_export_state_file);" in patched
    assert "writetable(imu_preintegration_table, imu_export_preintegration_file);" in patched
    assert "writetable(imu_residuals, imu_export_residual_file);" in patched
