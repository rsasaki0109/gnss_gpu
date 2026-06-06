#!/usr/bin/env python3
"""Patch Taroz ``fgo_gnss_imu.m`` to export IMU factor residual diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from experiments.patch_taroz_fgo_gnss_export import (
        DOPPLER_FACTOR_ANCHOR as GNSS_DOPPLER_FACTOR_ANCHOR,
        DOPPLER_FACTOR_EXPORT_BLOCK as GNSS_DOPPLER_FACTOR_EXPORT_BLOCK,
        GRAPH_INIT_EXPORT_BLOCK as GNSS_GRAPH_INIT_EXPORT_BLOCK,
        PSEUDORANGE_FACTOR_ANCHOR as GNSS_PSEUDORANGE_FACTOR_ANCHOR,
        PSEUDORANGE_FACTOR_EXPORT_BLOCK as GNSS_PSEUDORANGE_FACTOR_EXPORT_BLOCK,
        RESIDUAL_EXPORT_BLOCK as GNSS_RESIDUAL_EXPORT_BLOCK,
        TDCP_XXCC_ANCHOR as GNSS_TDCP_XXCC_ANCHOR,
        TDCP_XXCC_EXPORT_BLOCK as GNSS_TDCP_XXCC_EXPORT_BLOCK,
        TDCP_XXDD_ANCHOR as GNSS_TDCP_XXDD_ANCHOR,
        TDCP_XXDD_EXPORT_BLOCK as GNSS_TDCP_XXDD_EXPORT_BLOCK,
        TDCP_XXDD_OFFSET_ANCHOR as GNSS_TDCP_XXDD_OFFSET_ANCHOR,
        TDCP_XXDD_OFFSET_EXPORT_BLOCK as GNSS_TDCP_XXDD_OFFSET_EXPORT_BLOCK,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution from experiments/
    from patch_taroz_fgo_gnss_export import (
        DOPPLER_FACTOR_ANCHOR as GNSS_DOPPLER_FACTOR_ANCHOR,
        DOPPLER_FACTOR_EXPORT_BLOCK as GNSS_DOPPLER_FACTOR_EXPORT_BLOCK,
        GRAPH_INIT_EXPORT_BLOCK as GNSS_GRAPH_INIT_EXPORT_BLOCK,
        PSEUDORANGE_FACTOR_ANCHOR as GNSS_PSEUDORANGE_FACTOR_ANCHOR,
        PSEUDORANGE_FACTOR_EXPORT_BLOCK as GNSS_PSEUDORANGE_FACTOR_EXPORT_BLOCK,
        RESIDUAL_EXPORT_BLOCK as GNSS_RESIDUAL_EXPORT_BLOCK,
        TDCP_XXCC_ANCHOR as GNSS_TDCP_XXCC_ANCHOR,
        TDCP_XXCC_EXPORT_BLOCK as GNSS_TDCP_XXCC_EXPORT_BLOCK,
        TDCP_XXDD_ANCHOR as GNSS_TDCP_XXDD_ANCHOR,
        TDCP_XXDD_EXPORT_BLOCK as GNSS_TDCP_XXDD_EXPORT_BLOCK,
        TDCP_XXDD_OFFSET_ANCHOR as GNSS_TDCP_XXDD_OFFSET_ANCHOR,
        TDCP_XXDD_OFFSET_EXPORT_BLOCK as GNSS_TDCP_XXDD_OFFSET_EXPORT_BLOCK,
    )


GRAPH_INIT_ANCHOR = "% Create a factor graph container\ngraph = gtsam.NonlinearFactorGraph;\n"
_GNSS_GRAPH_EXPORT_BODY = GNSS_GRAPH_INIT_EXPORT_BLOCK.split("graph = gtsam.NonlinearFactorGraph;\n\n", 1)[1]

GRAPH_INIT_EXPORT_BLOCK = """% Create a factor graph container
graph = gtsam.NonlinearFactorGraph;

""" + _GNSS_GRAPH_EXPORT_BODY + """
% gnss_gpu parity export hook: keep Taroz/GTSAM IMU factors and interval keys.
imu_export_factors = {};
imu_export_preintegrations = {};
imu_export_epoch = zeros(0, 1);
imu_export_next_epoch = zeros(0, 1);
imu_export_utcms = zeros(0, 1);
imu_export_next_utcms = zeros(0, 1);
imu_export_sample_count = zeros(0, 1);
imu_export_graph_dt_s = zeros(0, 1);
imu_export_preintegrated_dt_s = zeros(0, 1);
imu_export_delta_r = zeros(0, 3);
imu_export_delta_p = zeros(0, 3);
imu_export_delta_v = zeros(0, 3);
imu_export_delta_p_bias_accel_jac = zeros(0, 9);
imu_export_delta_v_bias_accel_jac = zeros(0, 9);
imu_export_delta_p_bias_gyro_jac = zeros(0, 9);
imu_export_delta_v_bias_gyro_jac = zeros(0, 9);
imu_export_delta_r_bias_gyro_jac = zeros(0, 9);
imu_export_preint_meas_cov = zeros(0, 81);
"""

IMU_FACTOR_ANCHOR = (
    "        % IMU factor\n"
    "        graph.add(gtsam.ImuFactor(keyP1, keyV1, keyP2, keyV2, keyB2, currentSummarizedMeasurement));\n"
)

IMU_FACTOR_EXPORT_BLOCK = """        % IMU factor
        imu_factor = gtsam.ImuFactor(keyP1, keyV1, keyP2, keyV2, keyB2, currentSummarizedMeasurement);
        graph.add(imu_factor);
        imu_export_factors{end + 1, 1} = imu_factor;
        imu_export_preintegrations{end + 1, 1} = currentSummarizedMeasurement;
        imu_export_epoch(end + 1, 1) = i;
        imu_export_next_epoch(end + 1, 1) = i + 1;
        imu_export_utcms(end + 1, 1) = obs.utcms(i);
        imu_export_next_utcms(end + 1, 1) = obs.utcms(i + 1);
        imu_export_sample_count(end + 1, 1) = numel(IMUindices);
        imu_export_graph_dt_s(end + 1, 1) = dtgps;
        imu_export_preintegrated_dt_s(end + 1, 1) = currentSummarizedMeasurement.deltaTij();
        imu_export_delta_r(end + 1, :) = gtsam.Rot3.Logmap(currentSummarizedMeasurement.deltaRij())';
        imu_export_delta_p(end + 1, :) = currentSummarizedMeasurement.deltaPij()';
        imu_export_delta_v(end + 1, :) = currentSummarizedMeasurement.deltaVij()';
        imu_export_tmp_3x3 = NaN(3, 3);
        try
            imu_export_tmp_3x3 = currentSummarizedMeasurement.delPdelBiasAcc();
        catch
            try
                imu_export_tmp_3x3 = currentSummarizedMeasurement.delPdelBiasAcc;
            catch
            end
        end
        imu_export_delta_p_bias_accel_jac(end + 1, :) = reshape((-imu_export_tmp_3x3)', 1, []);
        imu_export_tmp_3x3 = NaN(3, 3);
        try
            imu_export_tmp_3x3 = currentSummarizedMeasurement.delVdelBiasAcc();
        catch
            try
                imu_export_tmp_3x3 = currentSummarizedMeasurement.delVdelBiasAcc;
            catch
            end
        end
        imu_export_delta_v_bias_accel_jac(end + 1, :) = reshape((-imu_export_tmp_3x3)', 1, []);
        imu_export_tmp_3x3 = NaN(3, 3);
        try
            imu_export_tmp_3x3 = currentSummarizedMeasurement.delPdelBiasOmega();
        catch
            try
                imu_export_tmp_3x3 = currentSummarizedMeasurement.delPdelBiasOmega;
            catch
            end
        end
        imu_export_delta_p_bias_gyro_jac(end + 1, :) = reshape((-imu_export_tmp_3x3)', 1, []);
        imu_export_tmp_3x3 = NaN(3, 3);
        try
            imu_export_tmp_3x3 = currentSummarizedMeasurement.delVdelBiasOmega();
        catch
            try
                imu_export_tmp_3x3 = currentSummarizedMeasurement.delVdelBiasOmega;
            catch
            end
        end
        imu_export_delta_v_bias_gyro_jac(end + 1, :) = reshape((-imu_export_tmp_3x3)', 1, []);
        imu_export_tmp_3x3 = NaN(3, 3);
        try
            imu_export_tmp_3x3 = currentSummarizedMeasurement.delRdelBiasOmega();
        catch
            try
                imu_export_tmp_3x3 = currentSummarizedMeasurement.delRdelBiasOmega;
            catch
            end
        end
        imu_export_delta_r_bias_gyro_jac(end + 1, :) = reshape((-imu_export_tmp_3x3)', 1, []);
        imu_export_tmp_9x9 = NaN(9, 9);
        try
            imu_export_tmp_9x9 = currentSummarizedMeasurement.preintMeasCov();
        catch
            try
                imu_export_tmp_9x9 = currentSummarizedMeasurement.preintMeasCov;
            catch
            end
        end
        imu_export_preint_meas_cov(end + 1, :) = reshape(imu_export_tmp_9x9', 1, []);
        if any(isnan(imu_export_delta_p_bias_accel_jac(end, :))) || ...
                any(isnan(imu_export_delta_v_bias_accel_jac(end, :))) || ...
                any(isnan(imu_export_delta_p_bias_gyro_jac(end, :))) || ...
                any(isnan(imu_export_delta_v_bias_gyro_jac(end, :))) || ...
                any(isnan(imu_export_delta_r_bias_gyro_jac(end, :)))
            imu_export_fd_eps = 1.0e-6;
            imu_export_fd_zero = zeros(3, 1);
            imu_export_fd_nav0 = gtsam.NavState(gtsam.Pose3(), imu_export_fd_zero);
            imu_export_fd_dt = currentSummarizedMeasurement.deltaTij();
            imu_export_fd_gravity = [0; 0; -prm.g];
            imu_export_fd_p_acc = zeros(3, 3);
            imu_export_fd_v_acc = zeros(3, 3);
            imu_export_fd_p_gyro = zeros(3, 3);
            imu_export_fd_v_gyro = zeros(3, 3);
            imu_export_fd_r_gyro = zeros(3, 3);
            for imu_export_fd_axis = 1:3
                imu_export_fd_step = zeros(3, 1);
                imu_export_fd_step(imu_export_fd_axis) = imu_export_fd_eps;

                imu_export_fd_bias_plus = gtsam.imuBias.ConstantBias(imu_export_fd_step, imu_export_fd_zero);
                imu_export_fd_bias_minus = gtsam.imuBias.ConstantBias(-imu_export_fd_step, imu_export_fd_zero);
                imu_export_fd_nav_plus = currentSummarizedMeasurement.predict(imu_export_fd_nav0, imu_export_fd_bias_plus);
                imu_export_fd_nav_minus = currentSummarizedMeasurement.predict(imu_export_fd_nav0, imu_export_fd_bias_minus);
                imu_export_fd_p_plus = imu_export_fd_nav_plus.position - ...
                    0.5 * imu_export_fd_gravity * imu_export_fd_dt * imu_export_fd_dt;
                imu_export_fd_p_minus = imu_export_fd_nav_minus.position - ...
                    0.5 * imu_export_fd_gravity * imu_export_fd_dt * imu_export_fd_dt;
                imu_export_fd_v_plus = imu_export_fd_nav_plus.velocity - imu_export_fd_gravity * imu_export_fd_dt;
                imu_export_fd_v_minus = imu_export_fd_nav_minus.velocity - imu_export_fd_gravity * imu_export_fd_dt;
                imu_export_fd_p_acc(:, imu_export_fd_axis) = ...
                    -(imu_export_fd_p_plus - imu_export_fd_p_minus) / (2 * imu_export_fd_eps);
                imu_export_fd_v_acc(:, imu_export_fd_axis) = ...
                    -(imu_export_fd_v_plus - imu_export_fd_v_minus) / (2 * imu_export_fd_eps);

                imu_export_fd_bias_plus = gtsam.imuBias.ConstantBias(imu_export_fd_zero, imu_export_fd_step);
                imu_export_fd_bias_minus = gtsam.imuBias.ConstantBias(imu_export_fd_zero, -imu_export_fd_step);
                imu_export_fd_nav_plus = currentSummarizedMeasurement.predict(imu_export_fd_nav0, imu_export_fd_bias_plus);
                imu_export_fd_nav_minus = currentSummarizedMeasurement.predict(imu_export_fd_nav0, imu_export_fd_bias_minus);
                imu_export_fd_p_plus = imu_export_fd_nav_plus.position - ...
                    0.5 * imu_export_fd_gravity * imu_export_fd_dt * imu_export_fd_dt;
                imu_export_fd_p_minus = imu_export_fd_nav_minus.position - ...
                    0.5 * imu_export_fd_gravity * imu_export_fd_dt * imu_export_fd_dt;
                imu_export_fd_v_plus = imu_export_fd_nav_plus.velocity - imu_export_fd_gravity * imu_export_fd_dt;
                imu_export_fd_v_minus = imu_export_fd_nav_minus.velocity - imu_export_fd_gravity * imu_export_fd_dt;
                imu_export_fd_r_plus = gtsam.Rot3.Logmap(imu_export_fd_nav_plus.attitude());
                imu_export_fd_r_minus = gtsam.Rot3.Logmap(imu_export_fd_nav_minus.attitude());
                imu_export_fd_p_gyro(:, imu_export_fd_axis) = ...
                    -(imu_export_fd_p_plus - imu_export_fd_p_minus) / (2 * imu_export_fd_eps);
                imu_export_fd_v_gyro(:, imu_export_fd_axis) = ...
                    -(imu_export_fd_v_plus - imu_export_fd_v_minus) / (2 * imu_export_fd_eps);
                imu_export_fd_r_gyro(:, imu_export_fd_axis) = ...
                    -(imu_export_fd_r_plus - imu_export_fd_r_minus) / (2 * imu_export_fd_eps);
            end
            if any(isnan(imu_export_delta_p_bias_accel_jac(end, :)))
                imu_export_delta_p_bias_accel_jac(end, :) = reshape(imu_export_fd_p_acc', 1, []);
            end
            if any(isnan(imu_export_delta_v_bias_accel_jac(end, :)))
                imu_export_delta_v_bias_accel_jac(end, :) = reshape(imu_export_fd_v_acc', 1, []);
            end
            if any(isnan(imu_export_delta_p_bias_gyro_jac(end, :)))
                imu_export_delta_p_bias_gyro_jac(end, :) = reshape(imu_export_fd_p_gyro', 1, []);
            end
            if any(isnan(imu_export_delta_v_bias_gyro_jac(end, :)))
                imu_export_delta_v_bias_gyro_jac(end, :) = reshape(imu_export_fd_v_gyro', 1, []);
            end
            if any(isnan(imu_export_delta_r_bias_gyro_jac(end, :)))
                imu_export_delta_r_bias_gyro_jac(end, :) = reshape(imu_export_fd_r_gyro', 1, []);
            end
        end
"""

OPTSTATUS_ANCHOR = "optstatus.OptError = optimizer.error;\n"
MAX_ITER_ANCHOR = "optparameters.setMaxIterations(1000);\n"
OPTIMIZER_ANCHOR = "optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initials, optparameters);\n"

MAX_ITER_EXPORT_BLOCK = """imu_export_max_iter = string(getenv("GSDC2023_LM_MAX_ITER"));
if strlength(imu_export_max_iter) > 0
    optparameters.setMaxIterations(str2double(imu_export_max_iter));
else
    optparameters.setMaxIterations(1000);
end
imu_export_lm_verbosity = string(getenv("GSDC2023_LM_VERBOSITY"));
if strlength(imu_export_lm_verbosity) > 0
    optparameters.setVerbosityLM(char(imu_export_lm_verbosity));
end
imu_export_lm_log_file = string(getenv("GSDC2023_LM_LOG_FILE"));
if strlength(imu_export_lm_log_file) > 0
    optparameters.setLogFile(char(imu_export_lm_log_file));
end
"""

LINEAR_SYSTEM_EXPORT_BLOCK = """optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initials, optparameters);

% gnss_gpu parity export hook: write GTSAM initial linear system in a stable order.
imu_linear_prefix = string(getenv("GSDC2023_LINEAR_EXPORT_PREFIX"));
if strlength(imu_linear_prefix) > 0
    [imu_linear_dir, ~, ~] = fileparts(imu_linear_prefix);
    if strlength(imu_linear_dir) > 0 && ~isfolder(imu_linear_dir)
        mkdir(imu_linear_dir);
    end
    imu_linear_ordering = gtsam.Ordering;
    for imu_linear_i = is:ie
        imu_linear_ordering.push_back(sym('x', imu_linear_i));
        imu_linear_ordering.push_back(sym('v', imu_linear_i));
        imu_linear_ordering.push_back(sym('c', imu_linear_i));
        imu_linear_ordering.push_back(sym('d', imu_linear_i));
        imu_linear_ordering.push_back(sym('p', imu_linear_i));
        imu_linear_ordering.push_back(sym('b', imu_linear_i));
    end
    imu_linear_graph = graph.linearize(initials);
    imu_linear_augmented = imu_linear_graph.augmentedJacobian(imu_linear_ordering);
    imu_linear_jacobian = imu_linear_augmented(:, 1:end-1);
    imu_linear_rhs = imu_linear_augmented(:, end);
    imu_linear_hessian = imu_linear_jacobian' * imu_linear_jacobian;
    imu_linear_gradient = imu_linear_jacobian' * imu_linear_rhs;
    imu_linear_meta = [optimizer.lambda(); optimizer.error(); optimizer.iterations(); imu_linear_ordering.size()];
    writematrix(imu_linear_hessian, imu_linear_prefix + "_H.csv");
    writematrix(imu_linear_gradient(:), imu_linear_prefix + "_g.csv");
    writematrix(imu_linear_meta(:), imu_linear_prefix + "_meta.csv");
end
"""

IMU_RESIDUAL_EXPORT_BLOCK = """optstatus.OptError = optimizer.error;

% gnss_gpu parity export hook: write Taroz/GTSAM IMU factor keys and residuals.
if exist("imu_export_factors", "var") && ~isempty(imu_export_factors)
    imu_export_trip_dir = datapath + course + "/" + phone;
    imu_export_mask_file = fullfile(imu_export_trip_dir, "phone_data_imu_factor_mask.csv");
    imu_export_residual_file = fullfile(imu_export_trip_dir, "phone_data_imu_residual_diagnostics.csv");
    imu_export_state_file = fullfile(imu_export_trip_dir, "phone_data_imu_state.csv");
    imu_export_preintegration_file = fullfile(imu_export_trip_dir, "phone_data_imu_preintegration.csv");

    imu_state_epoch_col = unique([imu_export_epoch; imu_export_next_epoch]);
    imu_state_utc_col = zeros(numel(imu_state_epoch_col), 1);
    imu_state_position_col = zeros(numel(imu_state_epoch_col), 3);
    imu_state_rpy_col = zeros(numel(imu_state_epoch_col), 3);
    imu_state_velocity_col = zeros(numel(imu_state_epoch_col), 3);
    imu_state_clock_col = zeros(numel(imu_state_epoch_col), 7);
    imu_state_drift_col = zeros(numel(imu_state_epoch_col), 1);
    imu_state_bias_col = zeros(numel(imu_state_epoch_col), 6);
    for imu_state_idx = 1:numel(imu_state_epoch_col)
        imu_state_epoch = imu_state_epoch_col(imu_state_idx);
        imu_state_pose = results.atPose3(sym('p', imu_state_epoch));
        imu_state_utc_col(imu_state_idx) = obs.utcms(imu_state_epoch);
        imu_state_position_col(imu_state_idx, :) = imu_state_pose.translation';
        imu_state_rpy_col(imu_state_idx, :) = imu_state_pose.rotation.rpy';
        imu_state_velocity_col(imu_state_idx, :) = results.atVector(sym('v', imu_state_epoch))';
        imu_state_clock_col(imu_state_idx, :) = results.atVector(sym('c', imu_state_epoch))';
        imu_state_drift_col(imu_state_idx, :) = results.atVector(sym('d', imu_state_epoch))';
        imu_state_bias_col(imu_state_idx, :) = results.atConstantBias(sym('b', imu_state_epoch)).vector';
    end
    imu_state_table = table(imu_state_epoch_col, imu_state_utc_col, ...
        imu_state_position_col(:, 1), imu_state_position_col(:, 2), imu_state_position_col(:, 3), ...
        imu_state_rpy_col(:, 1), imu_state_rpy_col(:, 2), imu_state_rpy_col(:, 3), ...
        imu_state_velocity_col(:, 1), imu_state_velocity_col(:, 2), imu_state_velocity_col(:, 3), ...
        imu_state_clock_col(:, 1), imu_state_clock_col(:, 2), imu_state_clock_col(:, 3), ...
        imu_state_clock_col(:, 4), imu_state_clock_col(:, 5), imu_state_clock_col(:, 6), ...
        imu_state_clock_col(:, 7), imu_state_drift_col(:, 1), ...
        imu_state_bias_col(:, 1), imu_state_bias_col(:, 2), imu_state_bias_col(:, 3), ...
        imu_state_bias_col(:, 4), imu_state_bias_col(:, 5), imu_state_bias_col(:, 6), ...
        'VariableNames', {'epoch_index', 'utcTimeMillis', ...
        'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw', ...
        'velocity_x', 'velocity_y', 'velocity_z', ...
        'clock_bias_m_0', 'clock_bias_m_1', 'clock_bias_m_2', 'clock_bias_m_3', ...
        'clock_bias_m_4', 'clock_bias_m_5', 'clock_bias_m_6', 'clock_drift_mps', ...
        'bias_acc_x', 'bias_acc_y', 'bias_acc_z', 'bias_gyro_x', 'bias_gyro_y', 'bias_gyro_z'});
    writetable(imu_state_table, imu_export_state_file);

    imu_gravity_row = [0, 0, -prm.g];
    imu_corrected_delta_r = zeros(numel(imu_export_factors), 3);
    imu_corrected_delta_p = zeros(numel(imu_export_factors), 3);
    imu_corrected_delta_v = zeros(numel(imu_export_factors), 3);
    imu_zero_nav = gtsam.NavState(gtsam.Pose3(), zeros(3, 1));
    for imu_export_idx = 1:numel(imu_export_factors)
        imu_next_epoch = imu_export_next_epoch(imu_export_idx);
        imu_bias = results.atConstantBias(sym('b', imu_next_epoch));
        imu_corrected_nav = imu_export_preintegrations{imu_export_idx}.predict(imu_zero_nav, imu_bias);
        imu_dt = imu_export_preintegrated_dt_s(imu_export_idx);
        imu_corrected_delta_r(imu_export_idx, :) = gtsam.Rot3.Logmap(imu_corrected_nav.attitude())';
        imu_corrected_delta_p(imu_export_idx, :) = imu_corrected_nav.position' - 0.5 * imu_gravity_row * imu_dt * imu_dt;
        imu_corrected_delta_v(imu_export_idx, :) = imu_corrected_nav.velocity' - imu_gravity_row * imu_dt;
    end
    imu_preintegration_table = table(imu_export_epoch, imu_export_utcms, ...
        imu_export_next_epoch, imu_export_next_utcms, imu_export_sample_count, ...
        imu_export_graph_dt_s, imu_export_preintegrated_dt_s, ...
        imu_export_delta_r(:, 1), imu_export_delta_r(:, 2), imu_export_delta_r(:, 3), ...
        imu_export_delta_p(:, 1), imu_export_delta_p(:, 2), imu_export_delta_p(:, 3), ...
        imu_export_delta_v(:, 1), imu_export_delta_v(:, 2), imu_export_delta_v(:, 3), ...
        imu_corrected_delta_r(:, 1), imu_corrected_delta_r(:, 2), imu_corrected_delta_r(:, 3), ...
        imu_corrected_delta_p(:, 1), imu_corrected_delta_p(:, 2), imu_corrected_delta_p(:, 3), ...
        imu_corrected_delta_v(:, 1), imu_corrected_delta_v(:, 2), imu_corrected_delta_v(:, 3), ...
        repmat(imu_gravity_row(1), numel(imu_export_factors), 1), ...
        repmat(imu_gravity_row(2), numel(imu_export_factors), 1), ...
        repmat(imu_gravity_row(3), numel(imu_export_factors), 1), ...
        'VariableNames', {'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sample_count', ...
        'graph_dt_s', 'preintegrated_dt_s', ...
        'delta_r_x', 'delta_r_y', 'delta_r_z', ...
        'delta_p_x', 'delta_p_y', 'delta_p_z', ...
        'delta_v_x', 'delta_v_y', 'delta_v_z', ...
        'corrected_delta_r_x', 'corrected_delta_r_y', 'corrected_delta_r_z', ...
        'corrected_delta_p_x', 'corrected_delta_p_y', 'corrected_delta_p_z', ...
        'corrected_delta_v_x', 'corrected_delta_v_y', 'corrected_delta_v_z', ...
        'gravity_x', 'gravity_y', 'gravity_z'});
    imu_preintegration_extra_values = [imu_export_delta_p_bias_accel_jac, ...
        imu_export_delta_v_bias_accel_jac, imu_export_delta_p_bias_gyro_jac, ...
        imu_export_delta_v_bias_gyro_jac, imu_export_delta_r_bias_gyro_jac, ...
        imu_export_preint_meas_cov];
    imu_preintegration_extra_names = strings(1, 0);
    imu_preintegration_jac_prefixes = ["delta_p_bias_accel_jac", "delta_v_bias_accel_jac", ...
        "delta_p_bias_gyro_jac", "delta_v_bias_gyro_jac", "delta_r_bias_gyro_jac"];
    for imu_prefix_idx = 1:numel(imu_preintegration_jac_prefixes)
        for imu_jac_row = 0:2
            for imu_jac_col = 0:2
                imu_preintegration_extra_names(end + 1) = ...
                    imu_preintegration_jac_prefixes(imu_prefix_idx) + "_" + ...
                    string(imu_jac_row) + "_" + string(imu_jac_col);
            end
        end
    end
    for imu_cov_row = 0:8
        for imu_cov_col = 0:8
            imu_preintegration_extra_names(end + 1) = ...
                "preint_meas_cov_" + string(imu_cov_row) + "_" + string(imu_cov_col);
        end
    end
    imu_preintegration_extra_table = array2table(imu_preintegration_extra_values, ...
        'VariableNames', cellstr(imu_preintegration_extra_names));
    imu_preintegration_table = [imu_preintegration_table, imu_preintegration_extra_table];
    writetable(imu_preintegration_table, imu_export_preintegration_file);

    imu_field_col = strings(0, 1);
    imu_freq_col = strings(0, 1);
    imu_epoch_col = zeros(0, 1);
    imu_utc_col = zeros(0, 1);
    imu_next_epoch_col = zeros(0, 1);
    imu_next_utc_col = zeros(0, 1);
    imu_sys_col = zeros(0, 1);
    imu_svid_col = zeros(0, 1);
    imu_axis_col = zeros(0, 1);
    imu_sample_count_col = zeros(0, 1);
    imu_graph_dt_col = zeros(0, 1);
    imu_preint_dt_col = zeros(0, 1);
    imu_residual_col = zeros(0, 1);

    imu_block_names = ["IMU_R", "IMU_P", "IMU_V"];
    for imu_export_idx = 1:numel(imu_export_factors)
        imu_epoch = imu_export_epoch(imu_export_idx);
        imu_next_epoch = imu_export_next_epoch(imu_export_idx);
        imu_error = imu_export_factors{imu_export_idx}.evaluateError( ...
            results.atPose3(sym('p', imu_epoch)), ...
            results.atVector(sym('v', imu_epoch)), ...
            results.atPose3(sym('p', imu_next_epoch)), ...
            results.atVector(sym('v', imu_next_epoch)), ...
            results.atConstantBias(sym('b', imu_next_epoch)));
        for imu_block = 1:3
            for imu_axis = 1:3
                imu_component = (imu_block - 1) * 3 + imu_axis;
                imu_field_col(end + 1, 1) = imu_block_names(imu_block);
                imu_freq_col(end + 1, 1) = "IMU";
                imu_epoch_col(end + 1, 1) = imu_epoch;
                imu_utc_col(end + 1, 1) = imu_export_utcms(imu_export_idx);
                imu_next_epoch_col(end + 1, 1) = imu_next_epoch;
                imu_next_utc_col(end + 1, 1) = imu_export_next_utcms(imu_export_idx);
                imu_sys_col(end + 1, 1) = 0;
                imu_svid_col(end + 1, 1) = 0;
                imu_axis_col(end + 1, 1) = imu_axis - 1;
                imu_sample_count_col(end + 1, 1) = imu_export_sample_count(imu_export_idx);
                imu_graph_dt_col(end + 1, 1) = imu_export_graph_dt_s(imu_export_idx);
                imu_preint_dt_col(end + 1, 1) = imu_export_preintegrated_dt_s(imu_export_idx);
                imu_residual_col(end + 1, 1) = imu_error(imu_component);
            end
        end
    end

    imu_factor_mask = table(imu_field_col, imu_freq_col, imu_epoch_col, imu_utc_col, ...
        imu_next_epoch_col, imu_next_utc_col, imu_sys_col, imu_svid_col, imu_axis_col, ...
        imu_sample_count_col, imu_graph_dt_col, imu_preint_dt_col, ...
        'VariableNames', {'field', 'freq', 'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sys', 'svid', 'axis', ...
        'sample_count', 'graph_dt_s', 'preintegrated_dt_s'});
    writetable(imu_factor_mask, imu_export_mask_file);

    imu_residuals = table(imu_field_col, imu_freq_col, imu_epoch_col, imu_utc_col, ...
        imu_next_epoch_col, imu_next_utc_col, imu_sys_col, imu_svid_col, imu_axis_col, ...
        imu_sample_count_col, imu_graph_dt_col, imu_preint_dt_col, imu_residual_col, ...
        'VariableNames', {'field', 'freq', 'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sys', 'svid', 'axis', ...
        'sample_count', 'graph_dt_s', 'preintegrated_dt_s', 'residual'});
    writetable(imu_residuals, imu_export_residual_file);

    if exist("imu_linear_prefix", "var") && strlength(imu_linear_prefix) > 0
        imu_delta_state = zeros(numel(imu_state_epoch_col), 26);
        for imu_state_idx = 1:numel(imu_state_epoch_col)
            imu_state_epoch = imu_state_epoch_col(imu_state_idx);
            imu_delta_state(imu_state_idx, 1:3) = ...
                (results.atVector(sym('x', imu_state_epoch)) - initials.atVector(sym('x', imu_state_epoch)))';
            imu_delta_state(imu_state_idx, 4:6) = ...
                (results.atVector(sym('v', imu_state_epoch)) - initials.atVector(sym('v', imu_state_epoch)))';
            imu_delta_state(imu_state_idx, 7:13) = ...
                (results.atVector(sym('c', imu_state_epoch)) - initials.atVector(sym('c', imu_state_epoch)))';
            imu_delta_state(imu_state_idx, 14) = ...
                results.atVector(sym('d', imu_state_epoch)) - initials.atVector(sym('d', imu_state_epoch));
            imu_delta_state(imu_state_idx, 15:20) = ...
                initials.atPose3(sym('p', imu_state_epoch)).localCoordinates(results.atPose3(sym('p', imu_state_epoch)))';
            imu_delta_state(imu_state_idx, 21:26) = ...
                (results.atConstantBias(sym('b', imu_state_epoch)).vector - ...
                 initials.atConstantBias(sym('b', imu_state_epoch)).vector)';
        end
        imu_linear_post_meta = [optimizer.lambda(); optimizer.error(); optimizer.iterations()];
        writematrix(imu_delta_state, imu_linear_prefix + "_delta_gtsam_order.csv");
        writematrix(imu_linear_post_meta(:), imu_linear_prefix + "_post_meta.csv");
    end
end
"""

RESIDUAL_EXPORT_BLOCK = (
    GNSS_RESIDUAL_EXPORT_BLOCK
    + "\n"
    + IMU_RESIDUAL_EXPORT_BLOCK.removeprefix("optstatus.OptError = optimizer.error;\n")
)


def _replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"expected exactly one {label} anchor, found {count}")
    return text.replace(anchor, replacement, 1)


def patch_fgo_gnss_imu_text(text: str) -> str:
    """Return ``fgo_gnss_imu.m`` text with IMU CSV export hooks inserted."""
    patched = _replace_once(text, GRAPH_INIT_ANCHOR, GRAPH_INIT_EXPORT_BLOCK, "graph init")
    patched = _replace_once(patched, GNSS_PSEUDORANGE_FACTOR_ANCHOR, GNSS_PSEUDORANGE_FACTOR_EXPORT_BLOCK, "GNSS pseudorange factor")
    patched = _replace_once(patched, GNSS_DOPPLER_FACTOR_ANCHOR, GNSS_DOPPLER_FACTOR_EXPORT_BLOCK, "GNSS Doppler factor")
    patched = _replace_once(patched, GNSS_TDCP_XXDD_OFFSET_ANCHOR, GNSS_TDCP_XXDD_OFFSET_EXPORT_BLOCK, "GNSS TDCP XXDD offset factor")
    patched = _replace_once(patched, GNSS_TDCP_XXDD_ANCHOR, GNSS_TDCP_XXDD_EXPORT_BLOCK, "GNSS TDCP XXDD factor")
    patched = _replace_once(patched, GNSS_TDCP_XXCC_ANCHOR, GNSS_TDCP_XXCC_EXPORT_BLOCK, "GNSS TDCP XXCC factor")
    patched = _replace_once(patched, IMU_FACTOR_ANCHOR, IMU_FACTOR_EXPORT_BLOCK, "IMU factor")
    patched = _replace_once(patched, MAX_ITER_ANCHOR, MAX_ITER_EXPORT_BLOCK, "LM max iteration")
    patched = _replace_once(patched, OPTIMIZER_ANCHOR, LINEAR_SYSTEM_EXPORT_BLOCK, "optimizer init")
    patched = _replace_once(patched, OPTSTATUS_ANCHOR, RESIDUAL_EXPORT_BLOCK, "optimizer status")
    return patched


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source fgo_gnss_imu.m")
    parser.add_argument("output", type=Path, help="patched output fgo_gnss_imu.m")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    source = Path(args.source)
    output = Path(args.output)
    patched = patch_fgo_gnss_imu_text(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
