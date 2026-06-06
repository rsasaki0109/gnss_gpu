#!/usr/bin/env python3
"""Patch Taroz ``fgo_gnss.m`` to export GNSS factor residual diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


GRAPH_INIT_ANCHOR = "% Create a factor graph container\ngraph = gtsam.NonlinearFactorGraph;\n"

GRAPH_INIT_EXPORT_BLOCK = """% Create a factor graph container
graph = gtsam.NonlinearFactorGraph;

% gnss_gpu parity export hook: keep Taroz/GTSAM GNSS factors and metadata.
gnss_export_factors = {};
gnss_export_field = strings(0, 1);
gnss_export_factor_model = strings(0, 1);
gnss_export_freq = strings(0, 1);
gnss_export_epoch = zeros(0, 1);
gnss_export_utc = zeros(0, 1);
gnss_export_next_epoch = zeros(0, 1);
gnss_export_next_utc = zeros(0, 1);
gnss_export_sys = zeros(0, 1);
gnss_export_svid = zeros(0, 1);
gnss_export_sat_col = zeros(0, 1);
gnss_export_sigtype = zeros(0, 1);
gnss_export_sigma = zeros(0, 1);
gnss_export_measurement = zeros(0, 1);
gnss_export_dt = zeros(0, 1);
gnss_export_los = zeros(0, 3);
gnss_export_org1 = zeros(0, 3);
gnss_export_org2 = zeros(0, 3);
"""

PSEUDORANGE_FACTOR_ANCHOR = (
    "                    graph.add(gtsam_gnss.PseudorangeFactor_XC(keyX, keyC, losvec, "
    "obsr.(f).resPc(i,j), sigtype(j), orgx, noise_rubust));\n"
)

PSEUDORANGE_FACTOR_EXPORT_BLOCK = """                    gnss_factor = gtsam_gnss.PseudorangeFactor_XC(keyX, keyC, losvec, obsr.(f).resPc(i,j), sigtype(j), orgx, noise_rubust);
                    graph.add(gnss_factor);
                    gnss_export_factors{end + 1, 1} = gnss_factor;
                    gnss_export_field(end + 1, 1) = "P";
                    gnss_export_factor_model(end + 1, 1) = "XC";
                    gnss_export_freq(end + 1, 1) = string(f);
                    gnss_export_epoch(end + 1, 1) = i;
                    gnss_export_utc(end + 1, 1) = obs.utcms(i);
                    gnss_export_next_epoch(end + 1, 1) = 0;
                    gnss_export_next_utc(end + 1, 1) = 0;
                    gnss_export_sys(end + 1, 1) = obs.sys(j);
                    gnss_export_svid(end + 1, 1) = obs.prn(j);
                    gnss_export_sat_col(end + 1, 1) = j;
                    gnss_export_sigtype(end + 1, 1) = sigtype(j);
                    gnss_export_sigma(end + 1, 1) = obserr.(f).P(i,j);
                    gnss_export_measurement(end + 1, 1) = obsr.(f).resPc(i,j);
                    gnss_export_dt(end + 1, 1) = 0;
                    gnss_export_los(end + 1, :) = losvec';
                    gnss_export_org1(end + 1, :) = orgx';
                    gnss_export_org2(end + 1, :) = [NaN NaN NaN];
"""

DOPPLER_FACTOR_ANCHOR = (
    "                    graph.add(gtsam_gnss.DopplerFactor_VD(keyV, keyD, losvec, "
    "obsr.(f).resD(i,j), orgv, noise_rubust));\n"
)

DOPPLER_FACTOR_EXPORT_BLOCK = """                    gnss_factor = gtsam_gnss.DopplerFactor_VD(keyV, keyD, losvec, obsr.(f).resD(i,j), orgv, noise_rubust);
                    graph.add(gnss_factor);
                    gnss_export_factors{end + 1, 1} = gnss_factor;
                    gnss_export_field(end + 1, 1) = "D";
                    gnss_export_factor_model(end + 1, 1) = "VD";
                    gnss_export_freq(end + 1, 1) = string(f);
                    gnss_export_epoch(end + 1, 1) = i;
                    gnss_export_utc(end + 1, 1) = obs.utcms(i);
                    gnss_export_next_epoch(end + 1, 1) = 0;
                    gnss_export_next_utc(end + 1, 1) = 0;
                    gnss_export_sys(end + 1, 1) = obs.sys(j);
                    gnss_export_svid(end + 1, 1) = obs.prn(j);
                    gnss_export_sat_col(end + 1, 1) = j;
                    gnss_export_sigtype(end + 1, 1) = 0;
                    gnss_export_sigma(end + 1, 1) = obserr.(f).D(i,j);
                    gnss_export_measurement(end + 1, 1) = obsr.(f).resD(i,j);
                    gnss_export_dt(end + 1, 1) = 0;
                    gnss_export_los(end + 1, :) = losvec';
                    gnss_export_org1(end + 1, :) = orgv';
                    gnss_export_org2(end + 1, :) = [NaN NaN NaN];
"""

TDCP_XXDD_OFFSET_ANCHOR = (
    "                            graph.add(gtsam_gnss.TDCPFactor_XXDD(keyX1, keyX2, keyD1, "
    "keyD2, losvec, tdcp+prm.Loffset, dtgps, orgx1, orgx2, noise_rubust));\n"
)

TDCP_XXDD_OFFSET_EXPORT_BLOCK = """                            tdcp_measurement = tdcp + prm.Loffset;
                            gnss_factor = gtsam_gnss.TDCPFactor_XXDD(keyX1, keyX2, keyD1, keyD2, losvec, tdcp_measurement, dtgps, orgx1, orgx2, noise_rubust);
                            graph.add(gnss_factor);
                            gnss_export_factors{end + 1, 1} = gnss_factor;
                            gnss_export_field(end + 1, 1) = "L";
                            gnss_export_factor_model(end + 1, 1) = "XXDD";
                            gnss_export_freq(end + 1, 1) = string(f);
                            gnss_export_epoch(end + 1, 1) = i;
                            gnss_export_utc(end + 1, 1) = obs.utcms(i);
                            gnss_export_next_epoch(end + 1, 1) = i + 1;
                            gnss_export_next_utc(end + 1, 1) = obs.utcms(i + 1);
                            gnss_export_sys(end + 1, 1) = obs.sys(j);
                            gnss_export_svid(end + 1, 1) = obs.prn(j);
                            gnss_export_sat_col(end + 1, 1) = j;
                            gnss_export_sigtype(end + 1, 1) = 0;
                            gnss_export_sigma(end + 1, 1) = obserr.(f).L(i,j);
                            gnss_export_measurement(end + 1, 1) = tdcp_measurement;
                            gnss_export_dt(end + 1, 1) = dtgps;
                            gnss_export_los(end + 1, :) = losvec';
                            gnss_export_org1(end + 1, :) = orgx1';
                            gnss_export_org2(end + 1, :) = orgx2';
"""

TDCP_XXDD_ANCHOR = (
    "                            graph.add(gtsam_gnss.TDCPFactor_XXDD(keyX1, keyX2, keyD1, "
    "keyD2, losvec, tdcp, dtgps, orgx1, orgx2, noise_rubust));\n"
)

TDCP_XXDD_EXPORT_BLOCK = """                            tdcp_measurement = tdcp;
                            gnss_factor = gtsam_gnss.TDCPFactor_XXDD(keyX1, keyX2, keyD1, keyD2, losvec, tdcp_measurement, dtgps, orgx1, orgx2, noise_rubust);
                            graph.add(gnss_factor);
                            gnss_export_factors{end + 1, 1} = gnss_factor;
                            gnss_export_field(end + 1, 1) = "L";
                            gnss_export_factor_model(end + 1, 1) = "XXDD";
                            gnss_export_freq(end + 1, 1) = string(f);
                            gnss_export_epoch(end + 1, 1) = i;
                            gnss_export_utc(end + 1, 1) = obs.utcms(i);
                            gnss_export_next_epoch(end + 1, 1) = i + 1;
                            gnss_export_next_utc(end + 1, 1) = obs.utcms(i + 1);
                            gnss_export_sys(end + 1, 1) = obs.sys(j);
                            gnss_export_svid(end + 1, 1) = obs.prn(j);
                            gnss_export_sat_col(end + 1, 1) = j;
                            gnss_export_sigtype(end + 1, 1) = 0;
                            gnss_export_sigma(end + 1, 1) = obserr.(f).L(i,j);
                            gnss_export_measurement(end + 1, 1) = tdcp_measurement;
                            gnss_export_dt(end + 1, 1) = dtgps;
                            gnss_export_los(end + 1, :) = losvec';
                            gnss_export_org1(end + 1, :) = orgx1';
                            gnss_export_org2(end + 1, :) = orgx2';
"""

TDCP_XXCC_ANCHOR = (
    "                            graph.add(gtsam_gnss.TDCPFactor_XXCC(keyX1, keyX2, keyC1, "
    "keyC2, losvec, tdcp, orgx1, orgx2, noise_rubust));\n"
)

TDCP_XXCC_EXPORT_BLOCK = """                            tdcp_measurement = tdcp;
                            gnss_factor = gtsam_gnss.TDCPFactor_XXCC(keyX1, keyX2, keyC1, keyC2, losvec, tdcp_measurement, orgx1, orgx2, noise_rubust);
                            graph.add(gnss_factor);
                            gnss_export_factors{end + 1, 1} = gnss_factor;
                            gnss_export_field(end + 1, 1) = "L";
                            gnss_export_factor_model(end + 1, 1) = "XXCC";
                            gnss_export_freq(end + 1, 1) = string(f);
                            gnss_export_epoch(end + 1, 1) = i;
                            gnss_export_utc(end + 1, 1) = obs.utcms(i);
                            gnss_export_next_epoch(end + 1, 1) = i + 1;
                            gnss_export_next_utc(end + 1, 1) = obs.utcms(i + 1);
                            gnss_export_sys(end + 1, 1) = obs.sys(j);
                            gnss_export_svid(end + 1, 1) = obs.prn(j);
                            gnss_export_sat_col(end + 1, 1) = j;
                            gnss_export_sigtype(end + 1, 1) = 0;
                            gnss_export_sigma(end + 1, 1) = obserr.(f).L(i,j);
                            gnss_export_measurement(end + 1, 1) = tdcp_measurement;
                            gnss_export_dt(end + 1, 1) = dtgps;
                            gnss_export_los(end + 1, :) = losvec';
                            gnss_export_org1(end + 1, :) = orgx1';
                            gnss_export_org2(end + 1, :) = orgx2';
"""

OPTSTATUS_ANCHOR = "optstatus.OptError = optimizer.error;\n"

RESIDUAL_EXPORT_BLOCK = """optstatus.OptError = optimizer.error;

% gnss_gpu parity export hook: write Taroz/GTSAM GNSS factor residuals.
if exist("gnss_export_factors", "var") && ~isempty(gnss_export_factors)
    gnss_export_trip_dir = datapath + course + "/" + phone;
    gnss_export_mask_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_factor_mask.csv");
    gnss_export_residual_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_factor_residuals.csv");
    gnss_export_summary_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_factor_summary.csv");
    gnss_export_graph_state_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_graph_state.csv");
    gnss_export_initial_state_file = fullfile(gnss_export_trip_dir, "phone_data_gnss_initial_state.csv");

    gnss_state_epoch_col = (is:ie)';
    gnss_state_utc_col = zeros(numel(gnss_state_epoch_col), 1);
    gnss_state_position_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_state_rpy_col = NaN(numel(gnss_state_epoch_col), 3);
    gnss_state_velocity_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_state_clock_col = zeros(numel(gnss_state_epoch_col), 7);
    gnss_state_drift_col = zeros(numel(gnss_state_epoch_col), 1);
    for gnss_state_idx = 1:numel(gnss_state_epoch_col)
        gnss_state_epoch = gnss_state_epoch_col(gnss_state_idx);
        gnss_state_utc_col(gnss_state_idx) = obs.utcms(gnss_state_epoch);
        gnss_state_position_col(gnss_state_idx, :) = results.atVector(sym('x', gnss_state_epoch))';
        try
            gnss_state_pose = results.atPose3(sym('p', gnss_state_epoch));
            gnss_state_rpy_col(gnss_state_idx, :) = gnss_state_pose.rotation.rpy';
        catch
        end
        gnss_state_velocity_col(gnss_state_idx, :) = results.atVector(sym('v', gnss_state_epoch))';
        gnss_state_clock = results.atVector(sym('c', gnss_state_epoch))';
        gnss_state_clock_col(gnss_state_idx, 1:numel(gnss_state_clock)) = gnss_state_clock;
        gnss_state_drift_col(gnss_state_idx) = results.atVector(sym('d', gnss_state_epoch));
    end
    gnss_graph_state_table = table(gnss_state_epoch_col, gnss_state_utc_col, ...
        gnss_state_position_col(:, 1), gnss_state_position_col(:, 2), gnss_state_position_col(:, 3), ...
        gnss_state_rpy_col(:, 1), gnss_state_rpy_col(:, 2), gnss_state_rpy_col(:, 3), ...
        gnss_state_velocity_col(:, 1), gnss_state_velocity_col(:, 2), gnss_state_velocity_col(:, 3), ...
        gnss_state_clock_col(:, 1), gnss_state_clock_col(:, 2), gnss_state_clock_col(:, 3), ...
        gnss_state_clock_col(:, 4), gnss_state_clock_col(:, 5), gnss_state_clock_col(:, 6), ...
        gnss_state_clock_col(:, 7), gnss_state_drift_col, ...
        'VariableNames', {'epoch_index', 'utcTimeMillis', ...
        'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw', ...
        'velocity_x', 'velocity_y', 'velocity_z', ...
        'clock_bias_m_0', 'clock_bias_m_1', 'clock_bias_m_2', 'clock_bias_m_3', ...
        'clock_bias_m_4', 'clock_bias_m_5', 'clock_bias_m_6', 'clock_drift_mps'});
    writetable(gnss_graph_state_table, gnss_export_graph_state_file);

    gnss_initial_utc_col = zeros(numel(gnss_state_epoch_col), 1);
    gnss_initial_position_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_initial_rpy_col = NaN(numel(gnss_state_epoch_col), 3);
    gnss_initial_velocity_col = zeros(numel(gnss_state_epoch_col), 3);
    gnss_initial_clock_col = zeros(numel(gnss_state_epoch_col), 7);
    gnss_initial_drift_col = zeros(numel(gnss_state_epoch_col), 1);
    for gnss_state_idx = 1:numel(gnss_state_epoch_col)
        gnss_state_epoch = gnss_state_epoch_col(gnss_state_idx);
        gnss_initial_utc_col(gnss_state_idx) = obs.utcms(gnss_state_epoch);
        gnss_initial_position_col(gnss_state_idx, :) = initials.atVector(sym('x', gnss_state_epoch))';
        try
            gnss_initial_pose = initials.atPose3(sym('p', gnss_state_epoch));
            gnss_initial_rpy_col(gnss_state_idx, :) = gnss_initial_pose.rotation.rpy';
        catch
        end
        gnss_initial_velocity_col(gnss_state_idx, :) = initials.atVector(sym('v', gnss_state_epoch))';
        gnss_initial_clock = initials.atVector(sym('c', gnss_state_epoch))';
        gnss_initial_clock_col(gnss_state_idx, 1:numel(gnss_initial_clock)) = gnss_initial_clock;
        gnss_initial_drift_col(gnss_state_idx) = initials.atVector(sym('d', gnss_state_epoch));
    end
    gnss_initial_state_table = table(gnss_state_epoch_col, gnss_initial_utc_col, ...
        gnss_initial_position_col(:, 1), gnss_initial_position_col(:, 2), gnss_initial_position_col(:, 3), ...
        gnss_initial_rpy_col(:, 1), gnss_initial_rpy_col(:, 2), gnss_initial_rpy_col(:, 3), ...
        gnss_initial_velocity_col(:, 1), gnss_initial_velocity_col(:, 2), gnss_initial_velocity_col(:, 3), ...
        gnss_initial_clock_col(:, 1), gnss_initial_clock_col(:, 2), gnss_initial_clock_col(:, 3), ...
        gnss_initial_clock_col(:, 4), gnss_initial_clock_col(:, 5), gnss_initial_clock_col(:, 6), ...
        gnss_initial_clock_col(:, 7), gnss_initial_drift_col, ...
        'VariableNames', {'epoch_index', 'utcTimeMillis', ...
        'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw', ...
        'velocity_x', 'velocity_y', 'velocity_z', ...
        'clock_bias_m_0', 'clock_bias_m_1', 'clock_bias_m_2', 'clock_bias_m_3', ...
        'clock_bias_m_4', 'clock_bias_m_5', 'clock_bias_m_6', 'clock_drift_mps'});
    writetable(gnss_initial_state_table, gnss_export_initial_state_file);

    gnss_initial_residual_col = NaN(numel(gnss_export_factors), 1);
    gnss_residual_col = NaN(numel(gnss_export_factors), 1);
    gnss_initial_factor_error_col = NaN(numel(gnss_export_factors), 1);
    gnss_factor_error_col = NaN(numel(gnss_export_factors), 1);
    for gnss_export_idx = 1:numel(gnss_export_factors)
        gnss_epoch = gnss_export_epoch(gnss_export_idx);
        gnss_next_epoch = gnss_export_next_epoch(gnss_export_idx);
        if gnss_export_field(gnss_export_idx) == "P"
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('x', gnss_epoch)), ...
                initials.atVector(sym('c', gnss_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('x', gnss_epoch)), ...
                results.atVector(sym('c', gnss_epoch)));
        elseif gnss_export_field(gnss_export_idx) == "D"
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('v', gnss_epoch)), ...
                initials.atVector(sym('d', gnss_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('v', gnss_epoch)), ...
                results.atVector(sym('d', gnss_epoch)));
        elseif gnss_export_factor_model(gnss_export_idx) == "XXDD"
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('x', gnss_epoch)), ...
                initials.atVector(sym('x', gnss_next_epoch)), ...
                initials.atVector(sym('d', gnss_epoch)), ...
                initials.atVector(sym('d', gnss_next_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('x', gnss_epoch)), ...
                results.atVector(sym('x', gnss_next_epoch)), ...
                results.atVector(sym('d', gnss_epoch)), ...
                results.atVector(sym('d', gnss_next_epoch)));
        else
            gnss_initial_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                initials.atVector(sym('x', gnss_epoch)), ...
                initials.atVector(sym('x', gnss_next_epoch)), ...
                initials.atVector(sym('c', gnss_epoch)), ...
                initials.atVector(sym('c', gnss_next_epoch)));
            gnss_error = gnss_export_factors{gnss_export_idx}.evaluateError( ...
                results.atVector(sym('x', gnss_epoch)), ...
                results.atVector(sym('x', gnss_next_epoch)), ...
                results.atVector(sym('c', gnss_epoch)), ...
                results.atVector(sym('c', gnss_next_epoch)));
        end
        gnss_initial_residual_col(gnss_export_idx) = gnss_initial_error(1);
        gnss_residual_col(gnss_export_idx) = gnss_error(1);
        try
            gnss_initial_factor_error_col(gnss_export_idx) = gnss_export_factors{gnss_export_idx}.error(initials);
            gnss_factor_error_col(gnss_export_idx) = gnss_export_factors{gnss_export_idx}.error(results);
        catch
            gnss_initial_factor_error_col(gnss_export_idx) = NaN;
            gnss_factor_error_col(gnss_export_idx) = NaN;
        end
    end

    gnss_factor_mask = table(gnss_export_field, gnss_export_freq, gnss_export_epoch, ...
        gnss_export_utc, gnss_export_next_epoch, gnss_export_next_utc, ...
        gnss_export_sys, gnss_export_svid, gnss_export_sat_col, gnss_export_factor_model, ...
        gnss_export_sigtype, gnss_export_sigma, gnss_export_measurement, gnss_export_dt, ...
        gnss_export_los(:, 1), gnss_export_los(:, 2), gnss_export_los(:, 3), ...
        gnss_export_org1(:, 1), gnss_export_org1(:, 2), gnss_export_org1(:, 3), ...
        gnss_export_org2(:, 1), gnss_export_org2(:, 2), gnss_export_org2(:, 3), ...
        'VariableNames', {'field', 'freq', 'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sys', 'svid', 'sat_col', ...
        'factor_model', 'sigtype', 'sigma', 'measurement', 'dt_s', ...
        'los_e', 'los_n', 'los_u', 'origin1_e', 'origin1_n', 'origin1_u', ...
        'origin2_e', 'origin2_n', 'origin2_u'});
    writetable(gnss_factor_mask, gnss_export_mask_file);

    gnss_factor_residuals = table(gnss_export_field, gnss_export_freq, gnss_export_epoch, ...
        gnss_export_utc, gnss_export_next_epoch, gnss_export_next_utc, ...
        gnss_export_sys, gnss_export_svid, gnss_export_sat_col, gnss_export_factor_model, ...
        gnss_export_sigtype, gnss_export_sigma, gnss_export_measurement, gnss_export_dt, ...
        gnss_initial_residual_col, gnss_residual_col, ...
        gnss_residual_col - gnss_initial_residual_col, ...
        gnss_initial_factor_error_col, gnss_factor_error_col, ...
        'VariableNames', {'field', 'freq', 'epoch_index', 'utcTimeMillis', ...
        'next_epoch_index', 'nextUtcTimeMillis', 'sys', 'svid', 'sat_col', ...
        'factor_model', 'sigtype', 'sigma', 'measurement', 'dt_s', ...
        'initial_residual', 'residual', 'residual_delta', ...
        'initial_factor_error', 'factor_error'});
    writetable(gnss_factor_residuals, gnss_export_residual_file);

    try
        gnss_export_initial_graph_error = graph.error(initials);
        gnss_export_final_graph_error = graph.error(results);
    catch
        gnss_export_initial_graph_error = NaN;
        gnss_export_final_graph_error = optimizer.error;
    end
    gnss_factor_count = numel(gnss_export_factors);
    gnss_p_count = sum(gnss_export_field == "P");
    gnss_d_count = sum(gnss_export_field == "D");
    gnss_l_count = sum(gnss_export_field == "L");
    gnss_iteration_count = optimizer.iterations;
    gnss_summary = table(gnss_factor_count, gnss_p_count, gnss_d_count, gnss_l_count, ...
        gnss_iteration_count, gnss_export_initial_graph_error, gnss_export_final_graph_error, ...
        'VariableNames', {'factor_count', 'p_count', 'd_count', 'l_count', ...
        'iterations', 'initial_graph_error', 'final_graph_error'});
    writetable(gnss_summary, gnss_export_summary_file);
end
"""


def _replace_once(text: str, anchor: str, replacement: str, label: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise ValueError(f"expected exactly one {label} anchor, found {count}")
    return text.replace(anchor, replacement, 1)


def patch_fgo_gnss_text(text: str) -> str:
    """Return ``fgo_gnss.m`` text with GNSS CSV export hooks inserted."""
    patched = _replace_once(text, GRAPH_INIT_ANCHOR, GRAPH_INIT_EXPORT_BLOCK, "graph init")
    patched = _replace_once(patched, PSEUDORANGE_FACTOR_ANCHOR, PSEUDORANGE_FACTOR_EXPORT_BLOCK, "P factor")
    patched = _replace_once(patched, DOPPLER_FACTOR_ANCHOR, DOPPLER_FACTOR_EXPORT_BLOCK, "D factor")
    patched = _replace_once(patched, TDCP_XXDD_OFFSET_ANCHOR, TDCP_XXDD_OFFSET_EXPORT_BLOCK, "TDCP XXDD offset factor")
    patched = _replace_once(patched, TDCP_XXDD_ANCHOR, TDCP_XXDD_EXPORT_BLOCK, "TDCP XXDD factor")
    patched = _replace_once(patched, TDCP_XXCC_ANCHOR, TDCP_XXCC_EXPORT_BLOCK, "TDCP XXCC factor")
    patched = _replace_once(patched, OPTSTATUS_ANCHOR, RESIDUAL_EXPORT_BLOCK, "optimizer status")
    return patched


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source fgo_gnss.m")
    parser.add_argument("output", type=Path, help="patched output fgo_gnss.m")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    source = Path(args.source)
    output = Path(args.output)
    patched = patch_fgo_gnss_text(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(patched, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
