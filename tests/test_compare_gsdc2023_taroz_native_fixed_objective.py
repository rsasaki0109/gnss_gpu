from __future__ import annotations

import numpy as np
import pytest

from experiments.compare_gsdc2023_taroz_native_fixed_objective import (
    align_state_position_origin_to_reference,
    build_arg_parser,
    load_taroz_export_as_native_fixed_arrays,
    state_to_taroz_graph_state_frame,
    taroz_graph_cost_for_native_state,
)


def test_arg_parser_accepts_factor_mask_csv() -> None:
    args = build_arg_parser().parse_args(
        [
            "export",
            "--factor-mask-csv",
            "bridge_factor.csv",
        ]
    )

    assert str(args.factor_mask_csv) == "bridge_factor.csv"


def test_align_state_position_origin_to_reference_translates_positions_only() -> None:
    state = np.array(
        [
            [0.0, 1.0, 2.0, 0.5, 0.0, 0.0, 4.0, 0.2],
            [3.0, 5.0, 7.0, 0.6, 0.0, 0.0, 5.0, 0.4],
        ],
        dtype=np.float64,
    )
    reference = np.array(
        [
            [100.0, 101.0, 102.0, 9.0, 9.0, 9.0, 0.0, 0.0],
            [110.0, 111.0, 112.0, 9.0, 9.0, 9.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    aligned = align_state_position_origin_to_reference(state, reference)

    np.testing.assert_allclose(aligned[:, :3], np.array([[100.0, 101.0, 102.0], [103.0, 105.0, 107.0]]))
    np.testing.assert_allclose(aligned[:, 3:], state[:, 3:])
    np.testing.assert_allclose(state[0, :3], np.array([0.0, 1.0, 2.0]))


def test_load_taroz_export_as_native_fixed_arrays_builds_dense_inputs(tmp_path) -> None:
    export_dir = tmp_path
    (export_dir / "phone_data_gnss_graph_state.csv").write_text(
        "\n".join(
            [
                "epoch_index,utcTimeMillis,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,clock_bias_m_0,clock_drift_mps",
                "1,1000,0,0,0,1,0,0,1,0.2",
                "2,2000,2,0,0,1,0,0,2,0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (export_dir / "phone_data_gnss_initial_state.csv").write_text(
        "\n".join(
            [
                "epoch_index,utcTimeMillis,position_x,position_y,position_z,velocity_x,velocity_y,velocity_z,clock_bias_m_0,clock_drift_mps",
                "1,1000,10,0,0,3,0,0,4,0.6",
                "2,2000,20,0,0,3,0,0,5,0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    header = (
        "field,freq,epoch_index,utcTimeMillis,next_epoch_index,nextUtcTimeMillis,sys,svid,sat_col,"
        "factor_model,sigtype,sigma,measurement,dt_s,los_e,los_n,los_u,"
        "origin1_e,origin1_n,origin1_u,origin2_e,origin2_n,origin2_u"
    )
    rows = [
        "P,L1,1,1000,0,0,1,3,0,XC,0,1,0.5,0,1,0,0,0,0,0,NaN,NaN,NaN",
        "D,L1,1,1000,0,0,1,3,0,VD,0,2,1.0,0,0,1,0,0,0,0,NaN,NaN,NaN",
        "L,L1,1,1000,2,2000,1,3,0,XXCC,0,0.5,2.5,1,1,0,0,0,0,0,0,0,0",
    ]
    (export_dir / "phone_data_gnss_factor_mask.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    bridge_rows = rows.copy()
    bridge_rows[0] = "P,L1,1,1000,0,0,1,3,0,XC,0,1,0.0,0,1,0,0,0,0,0,NaN,NaN,NaN"
    (export_dir / "bridge_factor.csv").write_text(
        header + "\n" + "\n".join(bridge_rows) + "\n",
        encoding="utf-8",
    )

    arrays = load_taroz_export_as_native_fixed_arrays(export_dir, n_clock=1)
    bridge_arrays = load_taroz_export_as_native_fixed_arrays(
        export_dir,
        n_clock=1,
        factor_csv="bridge_factor.csv",
    )
    initial_arrays = load_taroz_export_as_native_fixed_arrays(
        export_dir,
        n_clock=1,
        state_csv="phone_data_gnss_initial_state.csv",
    )

    assert arrays.state.shape == (2, 8)
    assert initial_arrays.state.shape == (2, 8)
    assert initial_arrays.state[0, 0] == 10.0
    assert initial_arrays.state[1, 6] == 5.0
    assert arrays.pseudorange.shape == (2, 4)
    assert arrays.tdcp_meas.shape == (1, 4)
    assert arrays.dt.tolist() == [1.0, 0.0]
    assert arrays.pseudorange[0, 0] == 0.5
    assert bridge_arrays.pseudorange[0, 0] == 0.0
    assert arrays.pseudorange_weights[0, 0] == 1.0
    assert arrays.doppler_weights[0, 0] == 0.25
    assert arrays.tdcp_weights[0, 0] == 4.0
    np.testing.assert_allclose(arrays.pr_linearization_los_ecef[0, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(arrays.doppler_linearization_los_ecef[0, 0], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(arrays.tdcp_linearization_ref_ecef, np.zeros((2, 3)))

    updated = arrays.state.copy()
    updated[1, 0] = 3.0
    frame = state_to_taroz_graph_state_frame(arrays.state_frame, updated, n_clock=1)
    assert frame.loc[1, "position_x"] == 3.0

    cost = taroz_graph_cost_for_native_state(
        export_dir,
        arrays.state_frame,
        arrays.state,
        n_clock=1,
        pr_huber_k=0.0,
        doppler_huber_k=0.0,
        carrier_huber_k=0.0,
        motion_sigma_m=2.0,
        clock_sigma_m=0.1,
    )
    assert cost == pytest.approx(25.33)
    bridge_cost = taroz_graph_cost_for_native_state(
        export_dir,
        arrays.state_frame,
        arrays.state,
        n_clock=1,
        factor_csv="bridge_factor.csv",
        pr_huber_k=0.0,
        doppler_huber_k=0.0,
        carrier_huber_k=0.0,
        motion_sigma_m=2.0,
        clock_sigma_m=0.1,
    )
    assert bridge_cost == pytest.approx(25.705)
