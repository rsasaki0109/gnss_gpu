from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert matched["observation_delta"] == 2.0
    assert matched["model_delta"] == -100.0
    assert matched["sat_position_delta_norm"] == 1.0
    assert payload["total_matlab_count"] == 2
    assert payload["total_bridge_count"] == 1
    assert payload["total_matched_count"] == 1
    assert payload["total_matlab_only"] == 1
    assert payload["total_bridge_only"] == 0
    assert payload["median_abs_delta"] == 2.0
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
