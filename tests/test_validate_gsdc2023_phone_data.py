from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
from scipy.io import savemat

from experiments.validate_gsdc2023_phone_data import discover_trip_dirs, validate_trip


def _position_struct(xyz: np.ndarray, llh: np.ndarray) -> dict:
    return {
        "xyz": np.asarray(xyz, dtype=np.float64),
        "llh": np.asarray(llh, dtype=np.float64),
    }


def test_validate_trip_reads_phone_data_and_scores_results(tmp_path):
    trip_dir = tmp_path / "dataset_2023" / "train" / "courseA" / "phoneX"
    trip_dir.mkdir(parents=True)

    baseline_xyz = np.array([
        [1000.0, 2000.0, 3000.0],
        [1001.0, 2000.5, 3000.0],
        [1002.0, 2001.0, 3000.0],
    ])
    gt_xyz = baseline_xyz + np.array([
        [0.5, -0.2, 0.1],
        [0.4, -0.1, 0.1],
        [0.6, -0.3, 0.1],
    ])
    gnss_xyz = baseline_xyz + np.array([
        [0.1, -0.05, 0.02],
        [0.1, -0.02, 0.01],
        [0.1, -0.04, 0.01],
    ])

    baseline_llh = np.array([
        [35.00000, 139.00000, 10.0],
        [35.00001, 139.00001, 10.0],
        [35.00002, 139.00002, 10.0],
    ])
    gt_llh = baseline_llh + np.array([
        [4e-6, -2e-6, 0.0],
        [3e-6, -1e-6, 0.0],
        [5e-6, -3e-6, 0.0],
    ])
    gnss_llh = baseline_llh + np.array([
        [1e-6, -5e-7, 0.0],
        [1e-6, -3e-7, 0.0],
        [1e-6, -4e-7, 0.0],
    ])

    phone_data = {
        "obs": {
            "n": np.array([[3]]),
            "nsat": np.array([[7]]),
            "dt": np.array([[1.0]]),
            "L1": {
                "P": np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]]),
                "D": np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),
                "L": np.array([[10.0, 11.0], [12.0, 13.0], [14.0, np.nan]]),
            },
            "L5": {
                "P": np.array([[7.0], [8.0], [np.nan]]),
            },
        },
        "posbl": _position_struct(baseline_xyz, baseline_llh),
        "timebl": {"t": np.array([[10.0], [11.0], [12.0]])},
    }
    gt_data = {
        "posgt": _position_struct(gt_xyz, gt_llh),
        "timegt": {"t": np.array([[10.0], [11.0], [12.0]])},
    }
    result_gnss = {
        "posest": _position_struct(gnss_xyz, gnss_llh),
    }

    savemat(trip_dir / "phone_data.mat", phone_data)
    savemat(trip_dir / "gt.mat", gt_data)
    savemat(trip_dir / "result_gnss.mat", result_gnss)

    trips = discover_trip_dirs(tmp_path / "dataset_2023", dataset="train")
    assert trips == [trip_dir]

    result = validate_trip(trip_dir)
    assert result.trip_name == "train/courseA/phoneX"
    assert result.obs_epochs == 3
    assert result.nsat == 7
    assert result.baseline_epochs == 3
    assert result.gt_epochs == 3
    assert result.counts_by_freq["L1"]["P"] == 5
    assert result.counts_by_freq["L1"]["D"] == 5
    assert result.counts_by_freq["L1"]["L"] == 5
    assert result.counts_by_freq["L5"]["P"] == 2
    assert result.baseline_metrics is not None
    assert result.result_gnss_metrics is not None
    assert result.baseline_metrics.score_m > result.result_gnss_metrics.score_m


def test_validate_trip_falls_back_to_raw_device_gnss_zip(tmp_path):
    trip_dir = tmp_path / "dataset_2023" / "train" / "courseB" / "phoneY"
    trip_dir.mkdir(parents=True)

    gt_rows = [
        {
            "MessageType": "Fix",
            "Provider": "GT",
            "LatitudeDegrees": "35.0",
            "LongitudeDegrees": "139.0",
            "AltitudeMeters": "10.0",
            "UnixTimeMillis": "1000",
        },
        {
            "MessageType": "Fix",
            "Provider": "GT",
            "LatitudeDegrees": "35.00001",
            "LongitudeDegrees": "139.00001",
            "AltitudeMeters": "10.0",
            "UnixTimeMillis": "2000",
        },
    ]
    with (trip_dir / "ground_truth.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "MessageType", "Provider", "LatitudeDegrees", "LongitudeDegrees",
                "AltitudeMeters", "UnixTimeMillis",
            ],
        )
        writer.writeheader()
        writer.writerows(gt_rows)

    device_rows = [
        {
            "utcTimeMillis": "1000",
            "WlsPositionXEcefMeters": "-3947460.0",
            "WlsPositionYEcefMeters": "3431490.0",
            "WlsPositionZEcefMeters": "3637870.0",
        },
        {
            "utcTimeMillis": "1000",
            "WlsPositionXEcefMeters": "-3947460.0",
            "WlsPositionYEcefMeters": "3431490.0",
            "WlsPositionZEcefMeters": "3637870.0",
        },
        {
            "utcTimeMillis": "2000",
            "WlsPositionXEcefMeters": "-3947459.5",
            "WlsPositionYEcefMeters": "3431491.0",
            "WlsPositionZEcefMeters": "3637871.0",
        },
    ]
    csv_buf = io.StringIO()
    writer = csv.DictWriter(
        csv_buf,
        fieldnames=[
            "utcTimeMillis",
            "WlsPositionXEcefMeters",
            "WlsPositionYEcefMeters",
            "WlsPositionZEcefMeters",
        ],
    )
    writer.writeheader()
    writer.writerows(device_rows)
    with zipfile.ZipFile(trip_dir / "device_gnss.csv", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("device_gnss.csv", csv_buf.getvalue())

    result = validate_trip(trip_dir)
    assert result.obs_epochs is None
    assert result.nsat is None
    assert result.baseline_epochs == 2
    assert result.gt_epochs == 2
    assert result.baseline_metrics is not None
    assert result.baseline_metrics.matched_epochs == 2
