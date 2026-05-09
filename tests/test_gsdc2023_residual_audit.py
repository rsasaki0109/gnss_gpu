from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.compare_gsdc2023_residual_values import compare_residual_values
from experiments.gsdc2023_residual_audit import (
    append_bridge_residual_row,
    matlab_residual_frame,
    merge_residual_value_frames,
    residual_value_summary_frame,
)


def test_matlab_residual_frame_normalizes_p_and_d_rows(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": "1",
                "utcTimeMillis": "1000",
                "sys": "1",
                "svid": "3",
                "p_residual_m": "10.5",
                "d_residual_mps": "-2.25",
                "p_pre_respc_m": "110.5",
                "d_pre_resd_m": "8.0",
                "p_clock_bias_m": "100.0",
                "d_clock_bias_mps": "10.25",
                "p_corrected_m": "210.0",
                "p_range_m": "99.5",
                "d_obs_mps": "28.0",
                "d_model_mps": "20.0",
                "sat_x_m": "1.0",
                "sat_y_m": "2.0",
                "sat_z_m": "3.0",
                "p_pre_finite": 1,
                "d_pre_finite": 1,
            },
        ],
    ).to_csv(diagnostics_path, index=False)

    frame = matlab_residual_frame(diagnostics_path).set_index("field")

    assert set(frame.index) == {"P", "D"}
    assert frame.loc["P", "epoch_index"] == 1
    assert frame.loc["P", "utcTimeMillis"] == 1000
    assert frame.loc["P", "sys"] == 1
    assert frame.loc["P", "svid"] == 3
    assert frame.loc["P", "matlab_residual"] == 10.5
    assert frame.loc["P", "matlab_pre_residual"] == 110.5
    assert frame.loc["P", "matlab_common_bias"] == 100.0
    assert frame.loc["P", "matlab_observation"] == 210.0
    assert frame.loc["P", "matlab_model"] == 99.5
    assert frame.loc["D", "matlab_residual"] == -2.25
    assert frame.loc["D", "matlab_pre_residual"] == 8.0
    assert frame.loc["D", "matlab_common_bias"] == 10.25
    assert frame.loc["D", "matlab_observation"] == 28.0
    assert frame.loc["D", "matlab_model"] == 20.0
    np.testing.assert_allclose(
        frame.loc["P", ["matlab_sat_x", "matlab_sat_y", "matlab_sat_z"]].to_numpy(dtype=np.float64),
        [1.0, 2.0, 3.0],
    )


def test_merge_residual_value_frames_adds_deltas_and_summary() -> None:
    rows: list[dict[str, object]] = []
    append_bridge_residual_row(
        rows,
        field="P",
        freq="L1",
        times_ms=np.array([1000.0], dtype=np.float64),
        slot_keys=((1, 3, "GPS_L1_CA"),),
        epoch_idx=0,
        slot_idx=0,
        residual=12.0,
        pre_residual=112.0,
        common_bias=100.0,
        observation=212.0,
        model=100.0,
    )
    bridge = pd.DataFrame(rows)
    bridge["bridge_sat_x"] = 2.0
    bridge["bridge_sat_y"] = 2.0
    bridge["bridge_sat_z"] = 3.0
    matlab = pd.DataFrame(
        [
            {
                "field": "P",
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "matlab_residual": 10.0,
                "matlab_pre_residual": 110.0,
                "matlab_common_bias": 100.0,
                "matlab_observation": 210.0,
                "matlab_model": 200.0,
                "matlab_sat_x": 1.0,
                "matlab_sat_y": 2.0,
                "matlab_sat_z": 3.0,
                "matlab_obs_clk": 90.0,
                "matlab_isb": 10.0,
            },
            {
                "field": "D",
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 4,
                "matlab_residual": 5.0,
            },
        ],
    )

    merged = merge_residual_value_frames(matlab, bridge)
    summary, payload = residual_value_summary_frame(merged)

    side_counts = merged["side"].value_counts().to_dict()
    assert side_counts["both"] == 1
    assert side_counts["matlab_only"] == 1
    assert side_counts.get("bridge_only", 0) == 0
    matched = merged[merged["side"] == "both"].iloc[0]
    assert matched["delta"] == 2.0
    assert matched["pre_residual_delta"] == 2.0
    assert matched["common_bias_delta"] == 0.0
    assert matched["bridge_isb"] == 10.0
    assert matched["isb_delta"] == 0.0
    assert matched["observation_delta"] == 2.0
    assert matched["model_delta"] == -100.0
    assert matched["sat_position_delta_norm"] == 1.0
    assert payload["total_matlab_count"] == 2
    assert payload["total_bridge_count"] == 1
    assert payload["total_matched_count"] == 1
    assert payload["total_matlab_only"] == 1
    assert payload["total_bridge_only"] == 0
    assert payload["median_abs_delta"] == 2.0
    assert payload["median_abs_isb_delta"] == 0.0
    assert payload["median_abs_model_delta"] == 100.0
    assert payload["median_abs_sat_position_delta_norm"] == 1.0
    by_field = summary.set_index(["field", "freq"])
    assert by_field.loc[("P", "L1"), "matched_count"] == 1
    assert by_field.loc[("D", "L1"), "matlab_only"] == 1
    assert pd.isna(by_field.loc[("D", "L1"), "median_abs_delta"])


def test_matlab_residual_frame_requires_columns(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame([{"freq": "L1"}]).to_csv(diagnostics_path, index=False)

    with pytest.raises(ValueError, match="diagnostics CSV missing required columns"):
        matlab_residual_frame(diagnostics_path)


def test_real_matlab_export_residual_values_sm_a205u_snapshot() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    trip_dir = (
        repo_root.parent
        / "ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/train/"
        / "2022-10-06-21-51-us-ca-mtv-n/sm-a205u"
    )
    diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    if not diagnostics_path.is_file():
        pytest.skip(f"MATLAB residual diagnostics fixture is not available: {diagnostics_path}")
    if not (trip_dir / "device_gnss.csv").is_file():
        pytest.skip(f"raw bridge fixture is not available: {trip_dir}")

    merged, summary, payload = compare_residual_values(
        trip_dir,
        max_epochs=50,
        multi_gnss=False,
    )

    assert payload["total_matlab_count"] == 680
    assert payload["total_bridge_count"] == 580
    assert payload["total_matched_count"] == 580
    assert payload["total_matlab_only"] == 100
    assert payload["total_bridge_only"] == 0
    assert payload["median_abs_delta"] == pytest.approx(4.470348319296491e-08)
    assert payload["p95_abs_delta"] == pytest.approx(2.3697519201973366e-05)
    assert payload["max_abs_delta"] == pytest.approx(4.323213641971302e-05)
    assert payload["median_abs_sat_position_delta_norm"] == pytest.approx(1.0908206617897573e-07)
    assert payload["median_abs_sat_clock_bias_delta"] == pytest.approx(4.3291947804391384e-10)
    assert payload["median_abs_sat_clock_drift_delta"] == pytest.approx(7.209944447028604e-17)
    assert payload["median_abs_sat_trop_delta"] == pytest.approx(2.842170943040401e-14)
    assert payload["median_abs_common_bias_delta"] == pytest.approx(4.083205844551685e-07)
    assert payload["p95_abs_common_bias_delta"] == pytest.approx(1.7323779010780518e-05)
    assert payload["median_abs_isb_delta"] == pytest.approx(1.862647813766216e-08)
    assert payload["p95_abs_isb_delta"] == pytest.approx(4.0978193283081055e-08)
    assert payload["median_abs_pre_residual_delta"] == pytest.approx(8.940696405446147e-08)
    assert payload["median_abs_observation_delta"] == pytest.approx(4.831690603168681e-13)
    assert payload["max_abs_observation_delta"] == pytest.approx(1.0058283805847168e-07)
    assert payload["median_abs_model_delta"] == pytest.approx(7.82310962677002e-08)
    assert payload["max_abs_common_bias_delta"] == pytest.approx(3.240809505200559e-05)
    assert payload["max_abs_isb_delta"] == pytest.approx(6.705524668859653e-08)
    assert int(np.count_nonzero(merged["side"] == "both")) == 580
    assert int(np.count_nonzero(merged["side"] == "matlab_only")) == 100
    assert int(np.count_nonzero(merged["side"] == "bridge_only")) == 0
    diagnostics = pd.read_csv(diagnostics_path)
    matlab_only = merged.loc[
        merged["side"] == "matlab_only",
        ["field", "freq", "epoch_index", "utcTimeMillis", "sys", "svid"],
    ].merge(
        diagnostics[
            [
                "freq",
                "epoch_index",
                "utcTimeMillis",
                "sys",
                "svid",
                "p_factor_finite",
                "d_factor_finite",
            ]
        ],
        on=["freq", "epoch_index", "utcTimeMillis", "sys", "svid"],
        how="left",
    )
    p_only = matlab_only["field"] == "P"
    d_only = matlab_only["field"] == "D"
    assert int(matlab_only.loc[p_only, "p_factor_finite"].sum()) == 0
    assert int(matlab_only.loc[d_only, "d_factor_finite"].sum()) == 0

    by_field = summary.set_index(["field", "freq"])
    assert by_field.loc[("P", "L1"), "matched_count"] == 290
    assert by_field.loc[("D", "L1"), "matched_count"] == 290
    assert by_field.loc[("P", "L1"), "matlab_only"] == 50
    assert by_field.loc[("D", "L1"), "matlab_only"] == 50
    assert by_field.loc[("P", "L1"), "median_abs_delta"] == pytest.approx(1.4901161193847656e-08)
    assert by_field.loc[("D", "L1"), "median_abs_delta"] == pytest.approx(8.283568e-06, abs=1e-12)
    assert by_field.loc[("D", "L1"), "median_abs_common_bias_delta"] == pytest.approx(8.729387e-06, abs=1e-12)
    assert by_field.loc[("D", "L1"), "max_abs_common_bias_delta"] == pytest.approx(3.240809505200559e-05)
    assert by_field.loc[("P", "L1"), "median_abs_common_bias_delta"] == pytest.approx(1.862647813766216e-08)
    assert by_field.loc[("P", "L1"), "max_abs_common_bias_delta"] == pytest.approx(6.705524668859653e-08)
    assert by_field.loc[("P", "L1"), "median_abs_isb_delta"] == pytest.approx(1.862647813766216e-08)
    assert by_field.loc[("P", "L1"), "max_abs_isb_delta"] == pytest.approx(6.705524668859653e-08)
    assert pd.isna(by_field.loc[("D", "L1"), "median_abs_isb_delta"])


def test_real_matlab_export_residual_values_pixel4_l5_isb_snapshot() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    trip_dir = (
        repo_root.parent
        / "ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/train/"
        / "2020-07-17-23-13-us-ca-sf-mtv-280/pixel4"
    )
    diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    if not diagnostics_path.is_file():
        pytest.skip(f"MATLAB residual diagnostics fixture is not available: {diagnostics_path}")
    if not (trip_dir / "device_gnss.csv").is_file():
        pytest.skip(f"raw bridge fixture is not available: {trip_dir}")

    _merged, summary, payload = compare_residual_values(
        trip_dir,
        max_epochs=50,
        multi_gnss=False,
    )

    assert payload["total_matched_count"] == 1182
    assert payload["max_abs_delta"] == pytest.approx(5.060694127195786e-05)
    assert payload["max_abs_isb_delta"] == pytest.approx(5.301088094711304e-06)
    by_field = summary.set_index(["field", "freq"])
    assert by_field.loc[("P", "L5"), "matched_count"] == 136
    assert by_field.loc[("P", "L5"), "median_abs_isb_delta"] == pytest.approx(5.301088094711304e-06)
    assert by_field.loc[("P", "L5"), "max_abs_model_delta"] == pytest.approx(9.890645742416382e-06)
    assert by_field.loc[("P", "L1"), "median_abs_isb_delta"] == pytest.approx(2.6077276782388026e-08)
    assert pd.isna(by_field.loc[("D", "L5"), "median_abs_isb_delta"])
