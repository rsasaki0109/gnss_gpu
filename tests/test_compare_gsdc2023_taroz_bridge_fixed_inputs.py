from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from experiments.compare_gsdc2023_taroz_bridge_fixed_inputs import (
    apply_taroz_residual_diagnostics_pseudorange,
    apply_taroz_residual_diagnostics_products,
    build_arg_parser,
    compare_factor_frames,
    compare_base_correction_frames,
    infer_taroz_base_correction_frame,
    load_taroz_initial_state_ecef,
    raw_bridge_fixed_factor_frame,
    restrict_bridge_to_taroz_factor_keys,
    summarize_base_correction_comparison,
    summarize_factor_comparison,
)
from experiments.gsdc2023_imu import ecef_to_enu_relative, enu_to_ecef_relative
from experiments.gsdc2023_observation_matrix import TripArrays
from experiments.gsdc2023_raw_bridge import _geometric_range_rate_with_sagnac, _geometric_range_with_sagnac


def test_arg_parser_accepts_matlab_residual_diagnostics_mask() -> None:
    args = build_arg_parser().parse_args(
        [
            "export",
            "--trip",
            "course/phone",
            "--matlab-residual-diagnostics-mask",
            "phone_data_residual_diagnostics.csv",
        ]
    )

    assert str(args.matlab_residual_diagnostics_mask) == "phone_data_residual_diagnostics.csv"


def test_arg_parser_accepts_taroz_residual_diagnostics_products_flag() -> None:
    args = build_arg_parser().parse_args(
        [
            "export",
            "--trip",
            "course/phone",
            "--use-taroz-residual-diagnostics-products",
        ]
    )

    assert args.use_taroz_residual_diagnostics_products is True


def test_arg_parser_accepts_taroz_residual_diagnostics_pseudorange_flag() -> None:
    args = build_arg_parser().parse_args(
        [
            "export",
            "--trip",
            "course/phone",
            "--use-taroz-residual-diagnostics-pseudorange",
        ]
    )

    assert args.use_taroz_residual_diagnostics_pseudorange is True


def test_arg_parser_accepts_bridge_factor_output() -> None:
    args = build_arg_parser().parse_args(
        [
            "export",
            "--trip",
            "course/phone",
            "--bridge-factor-output",
            "bridge_factor.csv",
        ]
    )

    assert str(args.bridge_factor_output) == "bridge_factor.csv"


def _synthetic_batch_and_state() -> tuple[TripArrays, np.ndarray, np.ndarray]:
    origin_ecef = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    pos_enu = np.array([[0.0, 0.0, 0.0], [1.5, 2.0, -0.25]], dtype=np.float64)
    vel_enu = np.array([[1.5, 2.0, -0.25], [1.5, 2.0, -0.25]], dtype=np.float64)
    pos_ecef = enu_to_ecef_relative(pos_enu, origin_ecef)
    vel_ecef = enu_to_ecef_relative(vel_enu, origin_ecef) - origin_ecef
    state = np.zeros((2, 8), dtype=np.float64)
    state[:, :3] = pos_ecef
    state[:, 3:6] = vel_ecef
    state[:, 6] = np.array([4.0, 5.0], dtype=np.float64)
    state[:, 7] = np.array([0.3, 0.4], dtype=np.float64)

    sat_ecef = np.array(
        [
            [[2.2e7, 1.1e7, 1.9e7]],
            [[2.2001e7, 1.1002e7, 1.9003e7]],
        ],
        dtype=np.float64,
    )
    sat_vel = np.array([[[10.0, -3.0, 2.0]], [[11.0, -2.0, 2.5]]], dtype=np.float64)
    pseudorange = np.zeros((2, 1), dtype=np.float64)
    doppler = np.zeros((2, 1), dtype=np.float64)
    for epoch_idx in range(2):
        pseudorange[epoch_idx, 0] = _geometric_range_with_sagnac(sat_ecef[epoch_idx, 0], pos_ecef[epoch_idx]) + 12.0
        doppler[epoch_idx, 0] = (
            _geometric_range_rate_with_sagnac(
                sat_ecef[epoch_idx, 0],
                pos_ecef[epoch_idx],
                sat_vel[epoch_idx, 0],
                vel_ecef[epoch_idx],
            )
            + 0.75
        )
    rho0 = _geometric_range_with_sagnac(sat_ecef[0, 0], pos_ecef[0])
    rho1 = _geometric_range_with_sagnac(sat_ecef[1, 0], pos_ecef[1])
    tdcp_raw = np.array([[rho1 - rho0 + 0.5]], dtype=np.float64)
    batch = TripArrays(
        times_ms=np.array([1000.0, 2000.0], dtype=np.float64),
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=np.array([[4.0], [9.0]], dtype=np.float64),
        kaggle_wls=pos_ecef,
        truth=np.full((2, 3), np.nan, dtype=np.float64),
        max_sats=1,
        has_truth=False,
        sys_kind=np.zeros((2, 1), dtype=np.int32),
        n_clock=1,
        sat_vel=sat_vel,
        doppler=doppler,
        doppler_weights=np.array([[16.0], [25.0]], dtype=np.float64),
        dt=np.array([1.0, 0.0], dtype=np.float64),
        tdcp_meas=tdcp_raw.copy(),
        tdcp_raw_meas=tdcp_raw,
        tdcp_weights=np.array([[36.0]], dtype=np.float64),
        slot_keys=((1, 3, "GPS_L1_CA"),),
    )
    return batch, state, origin_ecef


def test_apply_taroz_residual_diagnostics_products_overwrites_matching_rows(tmp_path) -> None:
    batch, _state, _origin_ecef = _synthetic_batch_and_state()
    residual_csv = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "sat_x_m": 101.0,
                "sat_y_m": 102.0,
                "sat_z_m": 103.0,
                "sat_vx_mps": 1.5,
                "sat_vy_mps": 2.5,
                "sat_vz_mps": 3.5,
                "sat_clock_bias_m": -4.0,
                "sat_clock_drift_mps": 0.25,
            }
        ]
    ).to_csv(residual_csv, index=False)

    patched = apply_taroz_residual_diagnostics_products(batch, residual_csv)

    np.testing.assert_allclose(patched.sat_ecef[0, 0], [101.0, 102.0, 103.0])
    np.testing.assert_allclose(patched.sat_vel[0, 0], [1.5, 2.5, 3.5])
    assert patched.sat_clock_bias_matrix[0, 0] == -4.0
    assert patched.sat_clock_drift_mps[0, 0] == 0.25
    np.testing.assert_allclose(patched.sat_ecef[1, 0], batch.sat_ecef[1, 0])
    np.testing.assert_allclose(batch.sat_ecef[0, 0], [2.2e7, 1.1e7, 1.9e7])


def test_apply_taroz_residual_diagnostics_pseudorange_overwrites_matching_rows(tmp_path) -> None:
    batch, _state, _origin_ecef = _synthetic_batch_and_state()
    residual_csv = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_corrected_m": 120.0,
            }
        ]
    ).to_csv(residual_csv, index=False)

    patched = apply_taroz_residual_diagnostics_pseudorange(batch, residual_csv)

    assert patched.pseudorange[0, 0] == 120.0
    assert patched.pseudorange[1, 0] == batch.pseudorange[1, 0]
    assert batch.pseudorange[0, 0] != 120.0


def test_apply_taroz_residual_diagnostics_pseudorange_applies_valid_base_correction(tmp_path) -> None:
    batch, _state, _origin_ecef = _synthetic_batch_and_state()
    batch = replace(batch, weights=np.array([[4.0], [0.0]], dtype=np.float64))
    residual_csv = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_corrected_m": 120.0,
            },
            {
                "freq": "L1",
                "epoch_index": 2,
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 3,
                "p_corrected_m": 220.0,
            },
        ]
    ).to_csv(residual_csv, index=False)

    patched = apply_taroz_residual_diagnostics_pseudorange(
        batch,
        residual_csv,
        base_correction=np.array([[2.5], [9.0]], dtype=np.float64),
    )

    assert patched.pseudorange[0, 0] == 117.5
    assert patched.pseudorange[1, 0] == 220.0


def test_raw_bridge_fixed_factor_frame_round_trips_against_itself() -> None:
    batch, state, origin_ecef = _synthetic_batch_and_state()

    frame = raw_bridge_fixed_factor_frame(batch, state, origin_ecef)
    joined = compare_factor_frames(frame, frame)
    summary = summarize_factor_comparison(joined).set_index(["field", "freq"])

    assert frame["field"].tolist() == ["D", "D", "L", "P", "P"]
    assert int((joined["_merge"] == "both").sum()) == 5
    assert summary.loc[("P", "L1"), "matched_count"] == 2
    assert summary.loc[("D", "L1"), "matched_count"] == 2
    assert summary.loc[("L", "L1"), "matched_count"] == 1
    assert summary.loc[("P", "L1"), "measurement_delta_max_abs"] == 0.0
    assert summary.loc[("L", "L1"), "los_delta_norm_max_abs"] == 0.0


def test_factor_comparison_reports_missing_and_value_deltas() -> None:
    batch, state, origin_ecef = _synthetic_batch_and_state()
    taroz = raw_bridge_fixed_factor_frame(batch, state, origin_ecef)
    bridge = taroz.copy()
    bridge.loc[bridge["field"].eq("P"), "measurement"] += 2.0
    bridge = bridge[~(bridge["field"].eq("L"))].reset_index(drop=True)

    joined = compare_factor_frames(taroz, bridge)
    summary = summarize_factor_comparison(joined).set_index(["field", "freq"])

    assert summary.loc[("P", "L1"), "matched_count"] == 2
    assert summary.loc[("P", "L1"), "measurement_delta_mean_abs"] == 2.0
    assert summary.loc[("L", "L1"), "taroz_only_count"] == 1
    assert summary.loc[("L", "L1"), "bridge_count"] == 0


def test_restrict_bridge_to_taroz_factor_keys_drops_bridge_only_rows() -> None:
    batch, state, origin_ecef = _synthetic_batch_and_state()
    bridge = raw_bridge_fixed_factor_frame(batch, state, origin_ecef)
    taroz = bridge[~bridge["field"].eq("L")].reset_index(drop=True)

    restricted = restrict_bridge_to_taroz_factor_keys(bridge, taroz)
    joined = compare_factor_frames(taroz, restricted)

    assert restricted["field"].tolist() == taroz["field"].tolist()
    assert int((joined["_merge"] == "both").sum()) == len(taroz)
    assert int((joined["_merge"] == "right_only").sum()) == 0


def test_load_taroz_initial_state_ecef_converts_local_enu_state(tmp_path) -> None:
    origin_ecef = np.array([6378137.0, 0.0, 0.0], dtype=np.float64)
    state_csv = tmp_path / "phone_data_gnss_initial_state.csv"
    pd.DataFrame(
        [
            {
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "position_x": 1.0,
                "position_y": 2.0,
                "position_z": 3.0,
                "velocity_x": 0.5,
                "velocity_y": 0.25,
                "velocity_z": -0.75,
                "clock_bias_m_0": 4.0,
                "clock_drift_mps": -1.0,
            }
        ]
    ).to_csv(state_csv, index=False)

    state = load_taroz_initial_state_ecef(
        state_csv,
        batch_times_ms=np.array([1000.0], dtype=np.float64),
        origin_ecef=origin_ecef,
        n_clock=1,
    )

    np.testing.assert_allclose(ecef_to_enu_relative(state[:, :3], origin_ecef), np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(
        ecef_to_enu_relative(origin_ecef + state[:, 3:6], origin_ecef),
        np.array([[0.5, 0.25, -0.75]]),
    )
    assert state[0, 6] == 4.0
    assert state[0, 7] == -1.0


def test_base_correction_comparison_infers_taroz_correction_from_factor_measurement(tmp_path) -> None:
    residual_csv = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_pre_respc_m": 12.5,
            },
            {
                "freq": "L5",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_pre_respc_m": -2355.0,
            },
        ]
    ).to_csv(residual_csv, index=False)
    taroz_factors = pd.DataFrame(
        [
            {
                "field": "P",
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "nextUtcTimeMillis": 0,
                "sys": 1,
                "svid": 3,
                "measurement": 2.0,
            },
            {
                "field": "D",
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "nextUtcTimeMillis": 0,
                "sys": 1,
                "svid": 3,
                "measurement": 0.25,
            },
            {
                "field": "P",
                "freq": "L5",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "nextUtcTimeMillis": 0,
                "sys": 1,
                "svid": 3,
                "measurement": -2360.0,
            },
        ]
    )

    taroz = infer_taroz_base_correction_frame(taroz_factors, residual_csv)
    bridge = pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "bridge_correction_m": 11.0,
            },
            {
                "freq": "L5",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "bridge_correction_m": 5.0,
            },
        ]
    )

    joined = compare_base_correction_frames(taroz, bridge)
    summary = summarize_base_correction_comparison(joined).set_index("freq")

    assert taroz.set_index("freq").loc["L1", "taroz_correction_m"] == 10.5
    assert taroz.set_index("freq").loc["L5", "taroz_correction_m"] == 5.0
    assert summary.loc["L1", "matched_count"] == 1
    assert summary.loc["L1", "correction_delta_mean_abs"] == 0.5
    assert summary.loc["L5", "correction_delta_max_abs"] == 0.0
