from __future__ import annotations

import csv
import io
import json
import zipfile

import numpy as np
import pandas as pd

from experiments.gsdc2023_raw_bridge import (
    BridgeResult,
    _build_trip_arrays,
    _export_bridge_outputs,
    _fit_state_with_clock_bias,
)


def _write_zipped_csv(path, rows, fieldnames):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name, buf.getvalue())


def test_build_trip_arrays_from_raw_zip(tmp_path):
    trip = tmp_path / "dataset_2023" / "train" / "course" / "phone"
    trip.mkdir(parents=True)

    gt = pd.DataFrame(
        [
            {
                "MessageType": "Fix",
                "Provider": "GT",
                "LatitudeDegrees": 35.0,
                "LongitudeDegrees": 139.0,
                "AltitudeMeters": 10.0,
                "UnixTimeMillis": 1000,
            },
            {
                "MessageType": "Fix",
                "Provider": "GT",
                "LatitudeDegrees": 35.00001,
                "LongitudeDegrees": 139.00001,
                "AltitudeMeters": 10.0,
                "UnixTimeMillis": 2000,
            },
        ],
    )
    gt.to_csv(trip / "ground_truth.csv", index=False)

    rows = []
    for utc_ms in (1000, 2000):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                },
            )
            # lower-C/N0 duplicate should be dropped
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid + 500.0,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 10.0,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert batch.sat_ecef.shape == (2, 4, 3)
    assert batch.pseudorange.shape == (2, 4)
    assert batch.weights.shape == (2, 4)
    assert batch.kaggle_wls.shape == (2, 3)
    assert batch.truth.shape == (2, 3)
    expected_pr = 2.1e7 + 1000 * 1 + 10.0 - 2.0 - 3.0
    assert np.isclose(batch.pseudorange[0, 0], expected_pr)
    assert batch.weights[0, 0] > 0.0
    assert batch.has_truth is True


def test_build_trip_arrays_without_ground_truth(tmp_path):
    trip = tmp_path / "dataset_2023" / "test" / "course" / "phone"
    trip.mkdir(parents=True)

    rows = []
    for utc_ms in (1000, 2000, 3000):
        for svid in range(1, 5):
            rows.append(
                {
                    "utcTimeMillis": utc_ms,
                    "Svid": svid,
                    "ConstellationType": 1,
                    "SignalType": "GPS_L1_CA",
                    "RawPseudorangeMeters": 2.1e7 + 1000 * svid,
                    "IonosphericDelayMeters": 2.0,
                    "TroposphericDelayMeters": 3.0,
                    "SvClockBiasMeters": 10.0,
                    "SvPositionXEcefMeters": 2.6e7 - 1e5 * svid,
                    "SvPositionYEcefMeters": 1.3e7 + 2e5 * svid,
                    "SvPositionZEcefMeters": 2.1e7 - 3e5 * svid,
                    "SvElevationDegrees": 30.0 + svid,
                    "Cn0DbHz": 35.0 + svid,
                    "WlsPositionXEcefMeters": -3947460.0 + utc_ms * 0.001,
                    "WlsPositionYEcefMeters": 3431490.0 + utc_ms * 0.001,
                    "WlsPositionZEcefMeters": 3637870.0 + utc_ms * 0.001,
                },
            )

    _write_zipped_csv(trip / "device_gnss.csv", rows, list(rows[0].keys()))

    batch = _build_trip_arrays(
        trip,
        max_epochs=10,
        start_epoch=0,
        constellation_type=1,
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
    )

    assert batch.has_truth is False
    assert np.isnan(batch.truth).all()


def test_export_bridge_outputs(tmp_path):
    export_dir = tmp_path / "bridge"
    times_ms = np.array([1000, 2000], dtype=np.float64)
    ecef = np.array(
        [
            [-3947460.0, 3431490.0, 3637870.0],
            [-3947459.5, 3431490.5, 3637870.5],
        ],
        dtype=np.float64,
    )
    metrics = {
        "rms_2d": 1.0,
        "rms_3d": 1.2,
        "mean_2d": 0.9,
        "mean_3d": 1.0,
        "std_2d": 0.2,
        "p50": 0.8,
        "p67": 0.9,
        "p95": 1.4,
        "max_2d": 1.5,
        "n_epochs": 2,
    }
    result = BridgeResult(
        trip="train/course/phone",
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        selected_source_mode="auto",
        times_ms=times_ms,
        kaggle_wls=ecef,
        raw_wls=np.column_stack([ecef, np.zeros(2)]),
        fgo_state=np.column_stack([ecef + 1.0, np.zeros(2)]),
        selected_state=np.column_stack([ecef, np.zeros(2)]),
        selected_sources=np.array(["baseline", "fgo"], dtype=object),
        truth=ecef - 1.0,
        max_sats=7,
        fgo_iters=3,
        failed_chunks=0,
        selected_mse_pr=12.5,
        baseline_mse_pr=10.0,
        raw_wls_mse_pr=11.0,
        fgo_mse_pr=12.5,
        selected_source_counts={"baseline": 1, "raw_wls": 0, "fgo": 1},
        metrics_selected=metrics,
        metrics_kaggle=metrics,
        metrics_raw_wls=metrics,
        metrics_fgo=metrics,
    )

    _export_bridge_outputs(export_dir, result)

    pos = pd.read_csv(export_dir / "bridge_positions.csv")
    meta = json.loads((export_dir / "bridge_metrics.json").read_text(encoding="utf-8"))

    assert list(pos["UnixTimeMillis"]) == [1000, 2000]
    assert list(pos["SelectedSource"]) == ["baseline", "fgo"]
    assert "FgoLatitudeDegrees" in pos.columns
    assert "LatitudeDegrees" in pos.columns
    assert "GroundTruthLongitudeDegrees" in pos.columns
    assert meta["trip"] == "train/course/phone"
    assert meta["fgo_iters"] == 3
    assert meta["fgo_score_m"] == 1.1
    assert meta["selected_score_m"] == 1.1
    assert meta["baseline_mse_pr"] == 10.0


def test_export_bridge_outputs_without_ground_truth(tmp_path):
    export_dir = tmp_path / "bridge"
    times_ms = np.array([1000], dtype=np.float64)
    ecef = np.array([[-3947460.0, 3431490.0, 3637870.0]], dtype=np.float64)

    result = BridgeResult(
        trip="test/course/phone",
        signal_type="GPS_L1_CA",
        weight_mode="sin2el",
        selected_source_mode="baseline",
        times_ms=times_ms,
        kaggle_wls=ecef,
        raw_wls=np.column_stack([ecef, np.zeros(1)]),
        fgo_state=np.column_stack([ecef + 1.0, np.zeros(1)]),
        selected_state=np.column_stack([ecef, np.zeros(1)]),
        selected_sources=np.array(["raw_wls"], dtype=object),
        truth=None,
        max_sats=7,
        fgo_iters=2,
        failed_chunks=1,
        selected_mse_pr=5.0,
        baseline_mse_pr=4.0,
        raw_wls_mse_pr=5.0,
        fgo_mse_pr=6.0,
        selected_source_counts={"baseline": 0, "raw_wls": 1, "fgo": 0},
        metrics_selected=None,
        metrics_kaggle=None,
        metrics_raw_wls=None,
        metrics_fgo=None,
    )

    _export_bridge_outputs(export_dir, result)

    pos = pd.read_csv(export_dir / "bridge_positions.csv")
    meta = json.loads((export_dir / "bridge_metrics.json").read_text(encoding="utf-8"))

    assert np.isnan(pos.loc[0, "GroundTruthLatitudeDegrees"])
    assert meta["fgo_score_m"] is None
    assert meta["selected_source_counts"]["raw_wls"] == 1


def test_fit_state_with_clock_bias_estimates_bias_and_residual():
    sat_ecef = np.array(
        [
            [
                [15600000.0, 0.0, 20100000.0],
                [0.0, 17600000.0, 21300000.0],
                [-16600000.0, 0.0, 20800000.0],
                [0.0, -18600000.0, 21700000.0],
            ],
        ],
        dtype=np.float64,
    )
    xyz = np.array([[1113194.9, -4841695.5, 3985355.2]], dtype=np.float64)
    rho = np.linalg.norm(sat_ecef[0] - xyz[0], axis=1)
    bias = 73.0
    pseudorange = (rho + bias).reshape(1, -1)
    weights = np.ones_like(pseudorange)

    state, weighted_sse, weight_sum, per_epoch_wmse = _fit_state_with_clock_bias(
        sat_ecef,
        pseudorange,
        weights,
        xyz,
    )

    assert np.isclose(state[0, 3], bias)
    assert np.isclose(weighted_sse, 0.0, atol=1e-9)
    assert np.isclose(weight_sum, 4.0)
    assert np.isclose(per_epoch_wmse[0], 0.0, atol=1e-9)
