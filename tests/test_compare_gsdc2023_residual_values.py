from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import compare_gsdc2023_residual_values as residual_values


def test_build_bridge_residual_frame_keeps_observation_mask_for_context_batch(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_trip_arrays(_trip_dir: Path, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        if len(calls) == 2:
            assert kwargs["apply_observation_mask"] is True
            assert kwargs["raw_frame_epoch_window"] is True
            raise RuntimeError("context checked")
        return SimpleNamespace(
            times_ms=np.array([1000.0, 2000.0]),
            kaggle_wls=np.zeros((2, 3), dtype=np.float64),
            clock_drift_mps=None,
        )

    monkeypatch.setattr(residual_values, "_settings_epoch_window_for_trip", lambda _trip_dir, _max_epochs: (0, 2))
    monkeypatch.setattr(residual_values, "_build_trip_arrays", fake_build_trip_arrays)
    monkeypatch.setattr(
        residual_values,
        "_receiver_velocity_from_reference",
        lambda times_ms, xyz: np.zeros((len(times_ms), 3), dtype=np.float64),
    )

    with pytest.raises(RuntimeError, match="context checked"):
        residual_values.build_bridge_residual_frame(Path("trip"), max_epochs=2, multi_gnss=True)

    assert len(calls) == 2


def test_build_bridge_residual_frame_allows_disabling_observation_mask(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_trip_arrays(_trip_dir: Path, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        raise RuntimeError("mask flag checked")

    monkeypatch.setattr(residual_values, "_settings_epoch_window_for_trip", lambda _trip_dir, _max_epochs: (0, 2))
    monkeypatch.setattr(residual_values, "_build_trip_arrays", fake_build_trip_arrays)

    with pytest.raises(RuntimeError, match="mask flag checked"):
        residual_values.build_bridge_residual_frame(Path("trip"), max_epochs=2, apply_observation_mask=False)

    assert calls[0]["apply_observation_mask"] is False


def test_bridge_batch_component_frame_fills_inactive_row_components() -> None:
    frame = residual_values._bridge_batch_component_frame(
        np.array([1000.0]),
        receiver_xyz=np.array([[6378137.0, 0.0, 0.0]], dtype=np.float64),
        receiver_vel=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        slot_keys=((1, 3, "GPS_L1_CA"),),
        sat_ecef=np.array([[[6878137.0, 0.0, 0.0]]], dtype=np.float64),
        sat_vel=np.array([[[4.0, 5.0, 6.0]]], dtype=np.float64),
        sat_clock_bias_m=np.array([[7.0]], dtype=np.float64),
        sat_clock_drift_mps=np.array([[8.0]], dtype=np.float64),
        rtklib_iono_m=np.array([[9.0]], dtype=np.float64),
        rtklib_tropo_m=np.array([[10.0]], dtype=np.float64),
        sat_col_lookup={(1, 3): 2},
    )

    assert frame["field"].tolist() == ["P", "D"]
    row = frame.iloc[0]
    assert row["bridge_sat_col"] == 2
    assert row["bridge_sat_x"] == 6878137.0
    assert row["bridge_sat_vz"] == 6.0
    assert row["bridge_sat_clock_bias"] == 7.0
    assert row["bridge_sat_clock_drift"] == 8.0
    assert row["bridge_sat_iono"] == 9.0
    assert row["bridge_sat_trop"] == 10.0
    assert row["bridge_rcv_x"] == 6378137.0
    assert row["bridge_rcv_vy"] == 2.0
    assert np.isclose(row["bridge_sat_elevation"], 90.0)
