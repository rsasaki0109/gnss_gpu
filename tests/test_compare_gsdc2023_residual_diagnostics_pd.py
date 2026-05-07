from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.compare_gsdc2023_residual_diagnostics_pd import (
    PD_WIDE_FRAME_COLUMNS,
    bridge_residual_diagnostics_export_frame,
    bridge_residual_diagnostics_pd_export_frame,
    bridge_residual_diagnostics_pd_wide_export_frame,
    compare_residual_diagnostics_pd_values,
    matlab_residual_diagnostics_pd_values,
)


def _diagnostics_row() -> dict[str, object]:
    return {
        "freq": "L1",
        "epoch_index": 1,
        "utcTimeMillis": 1000,
        "sys": 1,
        "svid": 3,
        "p_residual_m": 10.0,
        "d_residual_mps": -2.0,
        "p_pre_respc_m": 110.0,
        "d_pre_resd_m": 8.0,
        "p_clock_bias_m": 100.0,
        "d_clock_bias_mps": 10.0,
        "p_corrected_m": 210.0,
        "p_range_m": 100.0,
        "d_obs_mps": 28.0,
        "d_model_mps": 20.0,
        "p_pre_finite": 1,
        "d_pre_finite": 1,
    }


def test_matlab_residual_diagnostics_pd_values_uses_sidecar_column_names(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame([_diagnostics_row()]).to_csv(diagnostics_path, index=False)

    values = matlab_residual_diagnostics_pd_values(diagnostics_path)
    by_column = values.set_index(["field", "diagnostics_column"])

    assert len(values) == 10
    assert by_column.loc[("P", "p_residual_m"), "matlab_value"] == 10.0
    assert by_column.loc[("P", "p_clock_bias_m"), "matlab_value"] == 100.0
    assert by_column.loc[("D", "d_obs_mps"), "matlab_value"] == 28.0
    assert by_column.loc[("D", "d_model_mps"), "matlab_value"] == 20.0


def test_compare_residual_diagnostics_pd_values_summarizes_bridge_deltas(tmp_path: Path) -> None:
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame([_diagnostics_row()]).to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    def fake_bridge_frame(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": 10.25,
                    "bridge_pre_residual": 110.0,
                    "bridge_common_bias": 99.75,
                    "bridge_observation": 210.0,
                    "bridge_model": 100.0,
                },
                {
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": -2.0,
                    "bridge_pre_residual": 8.0,
                    "bridge_common_bias": 10.0,
                    "bridge_observation": 28.0,
                    "bridge_model": 20.0,
                },
            ],
        )

    merged, summary, bridge, payload = compare_residual_diagnostics_pd_values(
        trip_dir,
        max_epochs=0,
        multi_gnss=False,
        bridge_frame_fn=fake_bridge_frame,
    )

    assert payload["total_matched_count"] == 10
    assert payload["total_matlab_only"] == 0
    assert payload["total_bridge_only"] == 0
    assert payload["max_abs_delta"] == 0.25
    assert payload["passed"] is False
    residual = merged[
        (merged["field"] == "P")
        & (merged["diagnostics_column"] == "p_residual_m")
        & (merged["side"] == "both")
    ].iloc[0]
    assert residual["delta"] == 0.25
    by_column = summary.set_index(["field", "diagnostics_column", "freq"])
    assert by_column.loc[("P", "p_clock_bias_m", "L1"), "max_abs_delta"] == 0.25
    wide = bridge_residual_diagnostics_pd_export_frame(bridge)
    assert wide.loc[0, "p_residual_m"] == 10.25
    assert wide.loc[0, "d_model_mps"] == 20.0


def test_compare_residual_diagnostics_pd_values_does_not_use_sidecar_as_inactive_filter(tmp_path: Path) -> None:
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    pd.DataFrame([_diagnostics_row()]).to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    def fake_bridge_frame(*_args, **kwargs) -> pd.DataFrame:
        assert kwargs["include_inactive_observations"] is True
        assert kwargs.get("inactive_key_filter") is None
        return pd.DataFrame(
            [
                {
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": 10.0,
                    "bridge_pre_residual": 110.0,
                    "bridge_common_bias": 100.0,
                    "bridge_observation": 210.0,
                    "bridge_model": 100.0,
                },
                {
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_residual": -2.0,
                    "bridge_pre_residual": 8.0,
                    "bridge_common_bias": 10.0,
                    "bridge_observation": 28.0,
                    "bridge_model": 20.0,
                },
            ],
        )

    _merged, _summary, _bridge, payload = compare_residual_diagnostics_pd_values(
        trip_dir,
        include_inactive_observations=True,
        bridge_frame_fn=fake_bridge_frame,
    )

    assert payload["passed"] is True


def test_bridge_residual_diagnostics_pd_wide_export_adds_sat_col_and_components() -> None:
    bridge_residuals = pd.DataFrame(
        [
            {
                "field": "P",
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "bridge_sat_col": 2,
                "bridge_residual": 1.25,
                "bridge_pre_residual": 11.25,
                "bridge_common_bias": 10.0,
                "bridge_observation": 210.0,
                "bridge_model": 200.0,
                "bridge_obs_clk": 90.0,
                "bridge_isb": 10.0,
                "bridge_pre_finite": 1,
                "bridge_factor_finite": 1,
                "bridge_l_pre_finite": 1,
                "bridge_l_factor_finite": 0,
                "bridge_sat_x": 1.0,
                "bridge_sat_y": 2.0,
                "bridge_sat_z": 3.0,
                "bridge_sat_vx": 0.1,
                "bridge_sat_vy": 0.2,
                "bridge_sat_vz": 0.3,
                "bridge_rcv_x": 4.0,
                "bridge_rcv_y": 5.0,
                "bridge_rcv_z": 6.0,
            },
            {
                "field": "D",
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "bridge_sat_col": 2,
                "bridge_residual": -2.0,
                "bridge_pre_residual": 8.0,
                "bridge_common_bias": 10.0,
                "bridge_observation": 28.0,
                "bridge_model": 20.0,
                "bridge_obs_dclk": 10.0,
                "bridge_pre_finite": 1,
                "bridge_factor_finite": 0,
                "bridge_l_pre_finite": 1,
                "bridge_l_factor_finite": 0,
                "bridge_sat_x": 1.0,
                "bridge_sat_y": 2.0,
                "bridge_sat_z": 3.0,
                "bridge_sat_vx": 0.1,
                "bridge_sat_vy": 0.2,
                "bridge_sat_vz": 0.3,
                "bridge_rcv_x": 4.0,
                "bridge_rcv_y": 5.0,
                "bridge_rcv_z": 6.0,
            },
        ],
    )

    wide = bridge_residual_diagnostics_pd_wide_export_frame(bridge_residuals)

    assert len(wide) == 1
    row = wide.iloc[0]
    assert row["sat_col"] == 2
    assert row["p_residual_m"] == 1.25
    assert row["d_residual_mps"] == -2.0
    assert row["sat_range_m"] == 200.0
    assert row["sat_rate_mps"] == 20.0
    assert row["sat_x_m"] == 1.0
    assert row["rcv_z_m"] == 6.0
    assert row["obs_clk_m"] == 90.0
    assert row["obs_dclk_m"] == 10.0
    assert row["p_isb_m"] == 10.0
    assert row["p_pre_finite"] == 1
    assert row["d_pre_finite"] == 1
    assert row["p_factor_finite"] == 1
    assert row["d_factor_finite"] == 0
    assert row["l_pre_finite"] == 1
    assert row["l_factor_finite"] == 0


def test_bridge_residual_diagnostics_export_frame_returns_sidecar_columns(tmp_path: Path) -> None:
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)

    def fake_bridge_frame(*_args, **_kwargs) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "field": "P",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_sat_col": 2,
                    "bridge_residual": 1.0,
                    "bridge_pre_residual": 2.0,
                    "bridge_common_bias": 3.0,
                    "bridge_observation": 4.0,
                    "bridge_model": 5.0,
                    "bridge_pre_finite": 1,
                    "bridge_factor_finite": 1,
                    "bridge_l_pre_finite": 1,
                    "bridge_l_factor_finite": 0,
                },
                {
                    "field": "D",
                    "freq": "L1",
                    "epoch_index": 1,
                    "utcTimeMillis": 1000,
                    "sys": 1,
                    "svid": 3,
                    "bridge_sat_col": 2,
                    "bridge_residual": 6.0,
                    "bridge_pre_residual": 7.0,
                    "bridge_common_bias": 8.0,
                    "bridge_observation": 9.0,
                    "bridge_model": 10.0,
                    "bridge_pre_finite": 1,
                    "bridge_factor_finite": 0,
                    "bridge_l_pre_finite": 1,
                    "bridge_l_factor_finite": 0,
                },
            ],
        )

    export = bridge_residual_diagnostics_export_frame(
        trip_dir,
        bridge_frame_fn=fake_bridge_frame,
        factor_finite_key_filter=set(),
    )

    assert tuple(export.columns) == PD_WIDE_FRAME_COLUMNS
    assert export.loc[0, "p_residual_m"] == 1.0
    assert export.loc[0, "d_residual_mps"] == 6.0
