from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.audit_gsdc2023_residual_diagnostics_sidecar import (
    DIAGNOSTICS_EXPECTED_COLUMNS,
    diagnostics_column_roles,
    diagnostics_trip_records,
    summary_from_records,
)


def test_diagnostics_column_roles_cover_expected_schema() -> None:
    roles = diagnostics_column_roles()
    by_column = {role.column: role for role in roles}

    assert tuple(by_column) == DIAGNOSTICS_EXPECTED_COLUMNS
    assert by_column["freq"].role == "key"
    assert by_column["p_factor_finite"].used_by == "diagnostics mask overlay and factor-mask rebuild"
    assert by_column["p_residual_m"].bridge_status == "implemented in bridge residual audit"
    assert by_column["sat_x_m"].role == "internal state component"


def test_diagnostics_trip_records_summarize_schema_and_boolean_counts(tmp_path: Path) -> None:
    trip_dir = tmp_path / "train" / "course" / "phone"
    trip_dir.mkdir(parents=True)
    rows = [
        {column: 0 for column in DIAGNOSTICS_EXPECTED_COLUMNS},
        {column: 0 for column in DIAGNOSTICS_EXPECTED_COLUMNS},
    ]
    rows[0].update(
        {
            "freq": "L1",
            "epoch_index": 1,
            "utcTimeMillis": 1000,
            "sys": 1,
            "svid": 3,
            "sat_col": 1,
            "p_pre_finite": 1,
            "d_pre_finite": 1,
            "p_factor_finite": 1,
        },
    )
    rows[1].update(
        {
            "freq": "L5",
            "epoch_index": 2,
            "utcTimeMillis": 2000,
            "sys": 1,
            "svid": 5,
            "sat_col": 2,
            "l_pre_finite": 1,
            "l_factor_finite": 1,
        },
    )
    pd.DataFrame(rows).to_csv(trip_dir / "phone_data_residual_diagnostics.csv", index=False)

    records = diagnostics_trip_records(tmp_path, ["train/course/phone", "train/missing/phone"])
    records_by_trip = records.set_index("trip")

    present = records_by_trip.loc["train/course/phone"]
    assert bool(present["diagnostics_present"]) is True
    assert present["row_count"] == 2
    assert present["column_count"] == len(DIAGNOSTICS_EXPECTED_COLUMNS)
    assert present["missing_expected_columns"] == ""
    assert present["p_pre_finite_count"] == 1
    assert present["l_factor_finite_count"] == 1

    missing = records_by_trip.loc["train/missing/phone"]
    assert bool(missing["diagnostics_present"]) is False

    summary = summary_from_records(records)
    assert summary["trip_count"] == 2
    assert summary["diagnostics_present_count"] == 1
    assert summary["diagnostics_complete_schema_count"] == 1
    assert summary["total_rows"] == 2
    assert summary["total_p_pre_finite_count"] == 1
    assert summary["total_l_factor_finite_count"] == 1
