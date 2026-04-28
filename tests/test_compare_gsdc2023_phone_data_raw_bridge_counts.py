from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.io import savemat

from experiments.compare_gsdc2023_factor_masks import build_bridge_factor_mask, compare_factor_masks
from experiments.compare_gsdc2023_phone_data_raw_bridge_counts import build_comparison_frames
from experiments.gsdc2023_raw_bridge import _geometric_range_with_sagnac


def _write_zipped_csv(path, rows, fieldnames):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.name, buf.getvalue())


def _phone_data_obs() -> dict:
    finite_2x4 = np.arange(1.0, 9.0, dtype=np.float64).reshape(2, 4)
    finite_1x4 = np.arange(1.0, 5.0, dtype=np.float64).reshape(1, 4)
    return {
        "obs": {
            "n": np.array([[2]]),
            "nsat": np.array([[8]]),
            "dt": np.array([[1.0]]),
            "L1": {
                "P": finite_2x4,
                "D": finite_2x4,
                "L": finite_1x4,
                "resPc": finite_2x4,
                "resD": finite_2x4,
                "resL": finite_1x4,
            },
            "L5": {
                "P": finite_2x4 + 10.0,
                "D": finite_2x4 + 10.0,
                "L": finite_1x4 + 10.0,
                "resPc": finite_2x4 + 10.0,
                "resD": finite_2x4 + 10.0,
                "resL": finite_1x4 + 10.0,
            },
        },
    }


def _raw_rows():
    rx = np.array([-3947460.0, 3431490.0, 3637870.0], dtype=np.float64)
    offsets = np.array(
        [
            [2.1e7, 0.0, 0.0],
            [0.0, 2.1e7, 0.0],
            [0.0, 0.0, 2.1e7],
            [-1.5e7, 1.5e7, 0.0],
        ],
        dtype=np.float64,
    )
    rows = []
    for utc_ms in (1000, 2000):
        for svid, offset in enumerate(offsets, start=1):
            sat = rx + offset
            rng = float(_geometric_range_with_sagnac(sat, rx))
            for constellation, signal_type in ((1, "GPS_L1_CA"), (1, "GPS_L5_Q")):
                rows.append(
                    {
                        "utcTimeMillis": utc_ms,
                        "Svid": svid,
                        "ConstellationType": constellation,
                        "SignalType": signal_type,
                        "RawPseudorangeMeters": rng + 95.0,
                        "IonosphericDelayMeters": 2.0,
                        "TroposphericDelayMeters": 3.0,
                        "SvClockBiasMeters": 10.0,
                        "SvPositionXEcefMeters": sat[0],
                        "SvPositionYEcefMeters": sat[1],
                        "SvPositionZEcefMeters": sat[2],
                        "SvElevationDegrees": 35.0,
                        "Cn0DbHz": 40.0,
                        "WlsPositionXEcefMeters": rx[0],
                        "WlsPositionYEcefMeters": rx[1],
                        "WlsPositionZEcefMeters": rx[2],
                        "PseudorangeRateMetersPerSecond": 0.0,
                        "PseudorangeRateUncertaintyMetersPerSecond": 0.2,
                        "AccumulatedDeltaRangeState": 1,
                        "AccumulatedDeltaRangeMeters": 10.0,
                        "AccumulatedDeltaRangeUncertaintyMeters": 0.02,
                    },
                )
    return rows


def _raw_rows_with_multignss_extra():
    rows = _raw_rows()
    extras = []
    for row in rows:
        if row["Svid"] != 1:
            continue
        for constellation, svid, signal_type in (
            (6, 11, "GAL_E1_C_P"),
            (6, 11, "GAL_E5A_Q"),
        ):
            extra = dict(row)
            extra["ConstellationType"] = constellation
            extra["Svid"] = svid
            extra["SignalType"] = signal_type
            extras.append(extra)
    return rows + extras


def test_compare_phone_data_counts_against_raw_bridge(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    savemat(trip_dir / "phone_data.mat", _phone_data_obs())
    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))

    comparison_df, summary_df, summary = build_comparison_frames(
        data_root,
        ["train"],
        limit=1,
        max_epochs=10,
    )

    assert summary["trip_count"] == 1
    assert summary["trips_with_phone_data"] == 1
    assert summary["trips_with_device_gnss"] == 1
    assert summary["matched_rows"] == 12
    assert comparison_df.shape[0] == 12
    assert summary_df.shape[0] == 1

    assert comparison_df["count_delta"].fillna(0).eq(0).all()
    l1 = comparison_df[(comparison_df["freq"] == "L1") & (comparison_df["field"] == "P")].iloc[0]
    l5 = comparison_df[(comparison_df["freq"] == "L5") & (comparison_df["field"] == "L")].iloc[0]
    assert int(l1["phone_count"]) == 8
    assert int(l1["bridge_count"]) == 8
    assert int(l5["phone_count"]) == 4
    assert int(l5["bridge_count"]) == 4


def test_compare_phone_data_counts_defaults_to_gps_only_scope(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    savemat(trip_dir / "phone_data.mat", _phone_data_obs())
    rows = _raw_rows_with_multignss_extra()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))

    default_df, _default_summary_df, default_summary = build_comparison_frames(
        data_root,
        ["train"],
        limit=1,
        max_epochs=10,
    )
    multi_df, _multi_summary_df, multi_summary = build_comparison_frames(
        data_root,
        ["train"],
        limit=1,
        max_epochs=10,
        multi_gnss=True,
    )

    assert default_summary["multi_gnss"] is False
    assert default_summary["bridge_count_scope"] == "gps_l1_l5"
    assert default_summary["matched_abs_delta_total"] == 0
    assert default_df["count_delta"].fillna(0).eq(0).all()

    assert multi_summary["multi_gnss"] is True
    assert multi_summary["bridge_count_scope"] == "multi_gnss_l1_l5"
    assert multi_summary["matched_bridge_count_total"] > multi_summary["matched_phone_count_total"]
    assert multi_df["count_delta"].fillna(0).gt(0).any()


def test_compare_phone_data_counts_gracefully_handles_missing_phone_data(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "test" / "courseB" / "phoneB"
    trip_dir.mkdir(parents=True)

    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))

    comparison_df, summary_df, summary = build_comparison_frames(
        data_root,
        ["test"],
        limit=1,
        max_epochs=10,
    )

    assert summary["trip_count"] == 1
    assert summary["trips_with_phone_data"] == 0
    assert summary["trips_with_device_gnss"] == 1
    assert comparison_df["phone_count"].isna().all()
    assert comparison_df["bridge_count"].notna().all()
    assert bool(summary_df.iloc[0]["phone_data_present"]) is False
    assert bool(summary_df.iloc[0]["bridge_available"]) is True


def test_compare_phone_data_counts_can_filter_single_trip(tmp_path):
    data_root = tmp_path / "dataset_2023"
    rows = _raw_rows()
    fieldnames = list(rows[0].keys())

    trip_a = data_root / "train" / "courseA" / "phoneA"
    trip_a.mkdir(parents=True)
    savemat(trip_a / "phone_data.mat", _phone_data_obs())
    _write_zipped_csv(trip_a / "device_gnss.csv", rows, fieldnames)

    trip_b = data_root / "train" / "courseB" / "phoneB"
    trip_b.mkdir(parents=True)
    savemat(trip_b / "phone_data.mat", _phone_data_obs())
    _write_zipped_csv(trip_b / "device_gnss.csv", rows, fieldnames)

    comparison_df, summary_df, summary = build_comparison_frames(
        data_root,
        ["train"],
        trips=["courseB/phoneB"],
        max_epochs=10,
    )

    assert summary["trip_count"] == 1
    assert summary["trip_filters"] == ["courseB/phoneB"]
    assert set(comparison_df["trip"]) == {"train/courseB/phoneB"}
    assert summary_df["trip"].tolist() == ["train/courseB/phoneB"]


def test_compare_phone_data_counts_can_offset_trip_scan(tmp_path):
    data_root = tmp_path / "dataset_2023"
    rows = _raw_rows()
    fieldnames = list(rows[0].keys())

    for course in ("courseA", "courseB", "courseC"):
        trip_dir = data_root / "train" / course / "phoneA"
        trip_dir.mkdir(parents=True)
        savemat(trip_dir / "phone_data.mat", _phone_data_obs())
        _write_zipped_csv(trip_dir / "device_gnss.csv", rows, fieldnames)

    _comparison_df, summary_df, summary = build_comparison_frames(
        data_root,
        ["train"],
        offset=1,
        limit=1,
        max_epochs=10,
    )

    assert summary["offset"] == 1
    assert summary["trip_count"] == 1
    assert summary_df["trip"].tolist() == ["train/courseB/phoneA"]


def test_compare_phone_data_counts_uses_matlab_count_export(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    (trip_dir / "phone_data.mat").write_bytes(b"matlab class bundle placeholder")
    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))
    pd.DataFrame(
        [
            {"freq": "L1", "field": "P", "count": 8},
            {"freq": "L1", "field": "D", "count": 8},
            {"freq": "L1", "field": "L", "count": 4},
            {"freq": "L5", "field": "P", "count": 8},
            {"freq": "L5", "field": "D", "count": 8},
            {"freq": "L5", "field": "L", "count": 4},
        ],
    ).to_csv(trip_dir / "phone_data_observation_counts.csv", index=False)

    comparison_df, summary_df, summary = build_comparison_frames(
        data_root,
        ["train"],
        limit=1,
        max_epochs=10,
    )

    matched = comparison_df[comparison_df["phone_count"].notna()]
    assert summary["trip_count"] == 1
    assert summary["trips_with_phone_data"] == 1
    assert summary["phone_errors"] == 0
    assert summary_df.iloc[0]["phone_total_p_count"] == 16
    assert matched["count_delta"].eq(0).all()


def test_compare_phone_data_counts_respects_settings_epoch_window(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"Course": "courseA", "Phone": "phoneA", "IdxStart": 1, "IdxEnd": 1}],
    ).to_csv(data_root / "settings_train.csv", index=False)

    (trip_dir / "phone_data.mat").write_bytes(b"matlab class bundle placeholder")
    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))
    pd.DataFrame(
        [
            {"freq": freq, "field": field, "count": count}
            for freq in ("L1", "L5")
            for field, count in (
                ("P", 4),
                ("D", 4),
                ("L", 0),
                ("resPc", 4),
                ("resD", 4),
                ("resL", 0),
            )
        ],
    ).to_csv(trip_dir / "phone_data_factor_counts.csv", index=False)

    comparison_df, _summary_df, summary = build_comparison_frames(data_root, ["train"])

    assert summary["trip_count"] == 1
    assert comparison_df["count_delta"].fillna(0).eq(0).all()
    l1_p = comparison_df[(comparison_df["freq"] == "L1") & (comparison_df["field"] == "P")].iloc[0]
    l1_l = comparison_df[(comparison_df["freq"] == "L1") & (comparison_df["field"] == "L")].iloc[0]
    assert int(l1_p["bridge_count"]) == 4
    assert int(l1_l["bridge_count"]) == 0


def test_compare_factor_masks_matches_exported_bridge_mask(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))
    bridge_mask = build_bridge_factor_mask(trip_dir, max_epochs=10, multi_gnss=False)
    bridge_mask.to_csv(trip_dir / "phone_data_factor_mask.csv", index=False)

    merged, summary_df, summary = compare_factor_masks(
        trip_dir,
        max_epochs=10,
        multi_gnss=False,
    )

    assert summary["total_matched_count"] == summary["total_matlab_count"]
    assert summary["total_matlab_only"] == 0
    assert summary["total_bridge_only"] == 0
    assert merged["side"].eq("both").all()
    assert summary_df["symmetric_parity"].eq(1.0).all()


def test_compare_factor_masks_respects_settings_epoch_window(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))
    expected_mask = build_bridge_factor_mask(trip_dir, max_epochs=1, multi_gnss=False)
    expected_mask.to_csv(trip_dir / "phone_data_factor_mask.csv", index=False)
    pd.DataFrame(
        [{"Course": "courseA", "Phone": "phoneA", "IdxStart": 1, "IdxEnd": 1}],
    ).to_csv(data_root / "settings_train.csv", index=False)

    merged, summary_df, summary = compare_factor_masks(
        trip_dir,
        max_epochs=0,
        multi_gnss=False,
    )

    assert summary["total_matched_count"] == summary["total_matlab_count"]
    assert summary["total_matlab_only"] == 0
    assert summary["total_bridge_only"] == 0
    assert merged["side"].eq("both").all()
    assert summary_df["symmetric_parity"].eq(1.0).all()


def test_compare_factor_masks_trims_matlab_mask_to_requested_max_epochs(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))
    full_mask = build_bridge_factor_mask(trip_dir, max_epochs=10, multi_gnss=False)
    full_mask.to_csv(trip_dir / "phone_data_factor_mask.csv", index=False)

    merged, summary_df, summary = compare_factor_masks(
        trip_dir,
        max_epochs=1,
        multi_gnss=False,
    )

    assert summary["start_epoch"] == 0
    assert summary["max_epochs"] == 1
    assert summary["total_matlab_only"] == 0
    assert summary["total_bridge_only"] == 0
    assert merged["epoch_index"].max() == 1
    assert not merged["field"].isin(("L", "resL")).any()
    assert summary_df["symmetric_parity"].eq(1.0).all()


def test_bridge_factor_mask_can_follow_residual_diagnostics_mask(tmp_path):
    data_root = tmp_path / "dataset_2023"
    trip_dir = data_root / "train" / "courseA" / "phoneA"
    trip_dir.mkdir(parents=True)

    rows = _raw_rows()
    _write_zipped_csv(trip_dir / "device_gnss.csv", rows, list(rows[0].keys()))
    diagnostics_path = trip_dir / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 1,
                "p_factor_finite": "1",
                "d_factor_finite": "1",
                "l_factor_finite": "1",
            },
            {
                "freq": "L1",
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 1,
                "p_factor_finite": "0",
                "d_factor_finite": "0",
                "l_factor_finite": "1",
            },
            {
                "freq": "L5",
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 2,
                "p_factor_finite": "1",
                "d_factor_finite": "1",
                "l_factor_finite": "0",
            },
        ],
    ).to_csv(diagnostics_path, index=False)

    bridge_mask = build_bridge_factor_mask(
        trip_dir,
        max_epochs=10,
        multi_gnss=False,
        matlab_residual_diagnostics_mask_path=diagnostics_path,
    )

    counts = bridge_mask.groupby(["field", "freq"]).size().to_dict()
    assert counts == {
        ("D", "L1"): 1,
        ("D", "L5"): 1,
        ("L", "L1"): 1,
        ("P", "L1"): 1,
        ("P", "L5"): 1,
        ("resD", "L1"): 1,
        ("resD", "L5"): 1,
        ("resL", "L1"): 1,
        ("resPc", "L1"): 1,
        ("resPc", "L5"): 1,
    }
