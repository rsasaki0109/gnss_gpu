from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.compare_gsdc2023_residual_diagnostics_factor_mask import compare_residual_diagnostics_factor_mask
from experiments.gsdc2023_factor_mask import (
    FACTOR_MASK_KEY_COLUMNS,
    append_factor_rows,
    build_factor_mask_from_residual_diagnostics,
    factor_mask_side_summary,
    merge_factor_mask_keys,
    normalize_factor_mask_frame,
)


REAL_MATLAB_EXPORT_PARITY_CASES = (
    ("2020-06-25-00-34-us-ca-mtv-sb-101/pixel4", 83640),
    ("2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl", 86768),
    ("2020-07-08-22-28-us-ca/pixel4", 113338),
    ("2020-07-08-22-28-us-ca/pixel4xl", 115168),
    ("2020-07-17-22-27-us-ca-mtv-sf-280/pixel4", 114570),
    ("2020-07-17-23-13-us-ca-sf-mtv-280/pixel4", 102120),
    ("2020-08-04-00-19-us-ca-sb-mtv-101/pixel4", 114728),
    ("2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl", 111740),
    ("2020-08-04-00-20-us-ca-sb-mtv-101/pixel5", 98868),
    ("2021-12-08-20-28-us-ca-lax-c/pixel5", 57846),
    ("2022-10-06-21-51-us-ca-mtv-n/sm-a205u", 41920),
)
REAL_MATLAB_EXPORT_FACTOR_COUNT_CASES = (
    *REAL_MATLAB_EXPORT_PARITY_CASES[:6],
    ("2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl", 87350),
    *REAL_MATLAB_EXPORT_PARITY_CASES[6:],
)


def _real_matlab_export_trip_dir(relative_trip: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root.parent
        / "ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/train"
        / relative_trip
    )


def _skip_without_real_matlab_export(trip_dir: Path) -> None:
    if not (trip_dir / "phone_data_residual_diagnostics.csv").is_file():
        pytest.skip(f"MATLAB export fixture is not available: {trip_dir}")
    if not (trip_dir / "phone_data_factor_mask.csv").is_file():
        pytest.skip(f"MATLAB factor-mask fixture is not available: {trip_dir}")


def _skip_without_real_matlab_factor_counts(trip_dir: Path) -> None:
    if not (trip_dir / "phone_data_factor_counts.csv").is_file():
        pytest.skip(f"MATLAB factor-count fixture is not available: {trip_dir}")
    if not (trip_dir / "phone_data_factor_mask.csv").is_file():
        pytest.skip(f"MATLAB factor-mask fixture is not available: {trip_dir}")


def test_append_factor_rows_builds_matlab_style_keys() -> None:
    rows: list[dict[str, object]] = []

    append_factor_rows(
        rows,
        field_names=("P", "resPc"),
        freq="L1",
        epoch_indices=np.array([0, 1], dtype=np.int64),
        slot_indices=np.array([0, 1], dtype=np.int64),
        times_ms=np.array([1000.0, 2000.0, 3000.0], dtype=np.float64),
        slot_keys=((1, 3, "GPS_L1_CA"), (1, 4, "GPS_L1_CA")),
        next_epoch_indices=np.array([1, 0], dtype=np.int64),
        epoch_offset=10,
    )

    frame = normalize_factor_mask_frame(pd.DataFrame(rows))

    assert frame["field"].tolist() == ["P", "P", "resPc", "resPc"]
    assert frame["epoch_index"].tolist() == [11, 12, 11, 12]
    assert frame["utcTimeMillis"].tolist() == [1000, 2000, 1000, 2000]
    assert frame["next_epoch_index"].tolist() == [12, 0, 12, 0]
    assert frame["nextUtcTimeMillis"].tolist() == [2000, 0, 2000, 0]
    assert frame["sys"].tolist() == [1, 1, 1, 1]
    assert frame["svid"].tolist() == [3, 4, 3, 4]
    assert frame["signal_type"].tolist() == ["GPS_L1_CA", "GPS_L1_CA", "GPS_L1_CA", "GPS_L1_CA"]


def test_factor_mask_merge_and_side_summary_counts_symmetric_parity() -> None:
    left = normalize_factor_mask_frame(
        pd.DataFrame(
            [
                _mask_row("P", 1, 1000, 3),
                _mask_row("D", 2, 2000, 4),
            ],
        ),
        keep_extra_columns=False,
    )
    right = normalize_factor_mask_frame(
        pd.DataFrame(
            [
                _mask_row("P", 1, 1000, 3),
                _mask_row("L", 1, 1000, 5, next_epoch_index=2, next_utc=2000),
            ],
        ),
        keep_extra_columns=False,
    )

    merged = merge_factor_mask_keys(
        left,
        right,
        left_only_side="matlab_only",
        right_only_side="bridge_only",
    )
    summary, payload = factor_mask_side_summary(
        merged,
        left_name="matlab",
        right_name="bridge",
        left_only_side="matlab_only",
        right_only_side="bridge_only",
        include_jaccard=True,
    )

    assert merged["side"].value_counts().to_dict() == {"both": 1, "matlab_only": 1, "bridge_only": 1}
    assert payload["total_matlab_count"] == 2
    assert payload["total_bridge_count"] == 2
    assert payload["total_matched_count"] == 1
    assert payload["total_matlab_only"] == 1
    assert payload["total_bridge_only"] == 1
    assert payload["jaccard"] == 1.0 / 3.0
    assert payload["symmetric_parity"] == 0.5
    assert set(summary["field"]) == {"D", "L", "P"}


def test_build_factor_mask_from_residual_diagnostics_rebuilds_l_pairs(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            {
                "freq": "L1",
                "epoch_index": 1,
                "utcTimeMillis": 1000,
                "sys": 1,
                "svid": 3,
                "p_factor_finite": 1,
                "d_factor_finite": 0,
                "l_factor_finite": 1,
            },
            {
                "freq": "L1",
                "epoch_index": 2,
                "utcTimeMillis": 2000,
                "sys": 1,
                "svid": 3,
                "p_factor_finite": 0,
                "d_factor_finite": 1,
                "l_factor_finite": 1,
            },
        ],
    ).to_csv(diagnostics_path, index=False)

    mask = build_factor_mask_from_residual_diagnostics(diagnostics_path)

    assert mask["field"].tolist() == ["D", "L", "P", "resD", "resL", "resPc"]
    assert mask.set_index("field").loc["L", "next_epoch_index"] == 2
    assert mask.set_index("field").loc["L", "nextUtcTimeMillis"] == 2000
    assert mask.set_index("field").loc["P", "epoch_index"] == 1
    assert mask.set_index("field").loc["D", "epoch_index"] == 2


def test_factor_mask_from_matlab_residual_diagnostics_snapshot(tmp_path: Path) -> None:
    diagnostics_path = tmp_path / "phone_data_residual_diagnostics.csv"
    pd.DataFrame(
        [
            _diagnostics_row("L1", 1, 1000, 1, 7, p=1, d=1, l=1),
            _diagnostics_row("L1", 2, 2000, 1, 7, p=1, d=0, l=1),
            _diagnostics_row("L1", 4, 4000, 1, 7, p=0, d=1, l=1),
            _diagnostics_row("L5", 2, 2000, 1, 7, p=1, d=1, l=1),
            _diagnostics_row("L5", 3, 3000, 1, 7, p=0, d=0, l=1),
            _diagnostics_row("L1", 2, 2000, 3, 5, p=1, d=0, l=0),
        ],
    ).to_csv(diagnostics_path, index=False)

    mask = build_factor_mask_from_residual_diagnostics(diagnostics_path)

    assert list(mask[FACTOR_MASK_KEY_COLUMNS].itertuples(index=False, name=None)) == [
        ("D", "L1", 1, 1000, 0, 0, 1, 7),
        ("D", "L1", 4, 4000, 0, 0, 1, 7),
        ("D", "L5", 2, 2000, 0, 0, 1, 7),
        ("L", "L1", 1, 1000, 2, 2000, 1, 7),
        ("L", "L5", 2, 2000, 3, 3000, 1, 7),
        ("P", "L1", 1, 1000, 0, 0, 1, 7),
        ("P", "L1", 2, 2000, 0, 0, 1, 7),
        ("P", "L1", 2, 2000, 0, 0, 3, 5),
        ("P", "L5", 2, 2000, 0, 0, 1, 7),
        ("resD", "L1", 1, 1000, 0, 0, 1, 7),
        ("resD", "L1", 4, 4000, 0, 0, 1, 7),
        ("resD", "L5", 2, 2000, 0, 0, 1, 7),
        ("resL", "L1", 1, 1000, 2, 2000, 1, 7),
        ("resL", "L5", 2, 2000, 3, 3000, 1, 7),
        ("resPc", "L1", 1, 1000, 0, 0, 1, 7),
        ("resPc", "L1", 2, 2000, 0, 0, 1, 7),
        ("resPc", "L1", 2, 2000, 0, 0, 3, 5),
        ("resPc", "L5", 2, 2000, 0, 0, 1, 7),
    ]

    summary, payload = factor_mask_side_summary(
        merge_factor_mask_keys(
            mask,
            mask,
            left_only_side="matlab_only",
            right_only_side="bridge_only",
        ),
        left_name="matlab",
        right_name="bridge",
        left_only_side="matlab_only",
        right_only_side="bridge_only",
        include_jaccard=True,
    )

    assert payload == {
        "total_matlab_count": 18,
        "total_bridge_count": 18,
        "total_matched_count": 18,
        "total_matlab_only": 0,
        "total_bridge_only": 0,
        "symmetric_parity": 1.0,
        "jaccard": 1.0,
    }
    assert summary.set_index(["field", "freq"])["matched_count"].to_dict() == {
        ("D", "L1"): 2,
        ("D", "L5"): 1,
        ("L", "L1"): 1,
        ("L", "L5"): 1,
        ("P", "L1"): 3,
        ("P", "L5"): 1,
        ("resD", "L1"): 2,
        ("resD", "L5"): 1,
        ("resL", "L1"): 1,
        ("resL", "L5"): 1,
        ("resPc", "L1"): 3,
        ("resPc", "L5"): 1,
    }


def test_real_matlab_export_residual_diagnostics_matches_factor_mask_snapshot() -> None:
    trip_dir = _real_matlab_export_trip_dir("2022-10-06-21-51-us-ca-mtv-n/sm-a205u")
    _skip_without_real_matlab_export(trip_dir)

    merged, summary, payload = compare_residual_diagnostics_factor_mask(trip_dir)

    assert payload["total_factor_mask_count"] == 41920
    assert payload["total_diagnostics_count"] == 41920
    assert payload["total_matched_count"] == 41920
    assert payload["total_factor_mask_only"] == 0
    assert payload["total_diagnostics_only"] == 0
    assert payload["symmetric_parity"] == 1.0
    assert int(np.count_nonzero(merged["side"] == "both")) == 41920
    assert int(np.count_nonzero(merged["side"] == "factor_mask_only")) == 0
    assert int(np.count_nonzero(merged["side"] == "diagnostics_only")) == 0
    assert summary.set_index(["field", "freq"])["matched_count"].to_dict() == {
        ("D", "L1"): 7798,
        ("L", "L1"): 5225,
        ("P", "L1"): 7937,
        ("resD", "L1"): 7798,
        ("resL", "L1"): 5225,
        ("resPc", "L1"): 7937,
    }


@pytest.mark.parametrize(
    ("relative_trip", "expected_count"),
    REAL_MATLAB_EXPORT_PARITY_CASES,
    ids=[case[0].replace("/", "__") for case in REAL_MATLAB_EXPORT_PARITY_CASES],
)
def test_real_matlab_export_factor_mask_parity_available_trips(
    relative_trip: str,
    expected_count: int,
) -> None:
    trip_dir = _real_matlab_export_trip_dir(relative_trip)
    _skip_without_real_matlab_export(trip_dir)

    _merged, _summary, payload = compare_residual_diagnostics_factor_mask(trip_dir)

    assert payload["total_factor_mask_count"] == expected_count
    assert payload["total_diagnostics_count"] == expected_count
    assert payload["total_matched_count"] == expected_count
    assert payload["total_factor_mask_only"] == 0
    assert payload["total_diagnostics_only"] == 0
    assert payload["symmetric_parity"] == 1.0


@pytest.mark.parametrize(
    ("relative_trip", "expected_count"),
    REAL_MATLAB_EXPORT_FACTOR_COUNT_CASES,
    ids=[case[0].replace("/", "__") for case in REAL_MATLAB_EXPORT_FACTOR_COUNT_CASES],
)
def test_real_matlab_export_factor_counts_match_factor_mask_rows(
    relative_trip: str,
    expected_count: int,
) -> None:
    trip_dir = _real_matlab_export_trip_dir(relative_trip)
    _skip_without_real_matlab_factor_counts(trip_dir)

    counts = pd.read_csv(trip_dir / "phone_data_factor_counts.csv")
    mask = normalize_factor_mask_frame(pd.read_csv(trip_dir / "phone_data_factor_mask.csv"))
    mask_counts = (
        mask.groupby(["freq", "field"], sort=True)
        .size()
        .rename("mask_count")
        .reset_index()
    )
    merged = counts.merge(mask_counts, on=["freq", "field"], how="outer").fillna(0)

    assert int(counts["count"].sum()) == expected_count
    assert int(mask_counts["mask_count"].sum()) == expected_count
    assert int((merged["count"] - merged["mask_count"]).abs().sum()) == 0
    by_freq_field = counts.set_index(["freq", "field"])["count"].to_dict()
    for freq in set(counts["freq"]):
        assert by_freq_field.get((freq, "P"), 0) == by_freq_field.get((freq, "resPc"), 0)
        assert by_freq_field.get((freq, "D"), 0) == by_freq_field.get((freq, "resD"), 0)
        assert by_freq_field.get((freq, "L"), 0) == by_freq_field.get((freq, "resL"), 0)


def _diagnostics_row(
    freq: str,
    epoch_index: int,
    utc_time_ms: int,
    sys: int,
    svid: int,
    *,
    p: int,
    d: int,
    l: int,
) -> dict[str, object]:
    return {
        "freq": freq,
        "epoch_index": epoch_index,
        "utcTimeMillis": utc_time_ms,
        "sys": sys,
        "svid": svid,
        "p_factor_finite": p,
        "d_factor_finite": d,
        "l_factor_finite": l,
    }


def _mask_row(
    field: str,
    epoch_index: int,
    utc_time_ms: int,
    svid: int,
    *,
    next_epoch_index: int = 0,
    next_utc: int = 0,
) -> dict[str, object]:
    return {
        "field": field,
        "freq": "L1",
        "epoch_index": epoch_index,
        "utcTimeMillis": utc_time_ms,
        "next_epoch_index": next_epoch_index,
        "nextUtcTimeMillis": next_utc,
        "sys": 1,
        "svid": svid,
    }
