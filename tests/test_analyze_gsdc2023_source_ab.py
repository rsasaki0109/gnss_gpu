from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_source_ab import (
    chunk_delta_summary,
    compare_submissions,
    comparison_summary,
    metrics_chunk_summary,
    phone_delta_summary,
    phone_from_trip_id,
    write_outputs,
)


def _submission(lat_shift_phone_b: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [
                "2023-01-01-00-00-us-ca-a/pixel5",
                "2023-01-01-00-00-us-ca-a/pixel5",
                "2023-01-02-00-00-us-ca-b/sm-a505u",
                "2023-01-02-00-00-us-ca-b/sm-a505u",
            ],
            "UnixTimeMillis": [1000, 2000, 1000, 2000],
            "LatitudeDegrees": [37.0, 37.00001, 38.0 + lat_shift_phone_b, 38.00001 + lat_shift_phone_b],
            "LongitudeDegrees": [-122.0, -122.00001, -123.0, -123.00001],
        },
    )


def test_compare_submissions_adds_phone_and_summary_tables() -> None:
    row_delta, trip_summary = compare_submissions(_submission(), _submission(lat_shift_phone_b=0.00001), "candidate")
    summary = comparison_summary(row_delta)
    phones = phone_delta_summary(row_delta)

    assert phone_from_trip_id("trip/pixel5") == "pixel5"
    assert set(row_delta["phone"]) == {"pixel5", "sm-a505u"}
    assert set(trip_summary["phone"]) == {"pixel5", "sm-a505u"}
    assert int(summary.iloc[0]["rows"]) == 4
    assert int(summary.iloc[0]["changed_rows_gt_1e_9m"]) == 2
    by_phone = {row.phone: row for row in phones.itertuples(index=False)}
    assert by_phone["pixel5"].changed_rows_gt_1e_9m == 0
    assert by_phone["sm-a505u"].changed_rows_gt_1e_9m == 2


def test_write_outputs_includes_comparison_phone_and_worst_rows(tmp_path) -> None:
    row_delta, trip_summary = compare_submissions(_submission(), _submission(lat_shift_phone_b=0.00001), "candidate")
    summary = comparison_summary(row_delta)
    phones = phone_delta_summary(row_delta)
    chunks = chunk_delta_summary(row_delta, target_trips=set(), chunk_epochs=2)
    metrics = metrics_chunk_summary([])

    write_outputs(tmp_path, row_delta, summary, trip_summary, phones, chunks, metrics)

    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["comparison_summary_csv"].endswith("comparison_summary.csv")
    assert payload["phone_delta_summary_csv"].endswith("phone_delta_summary.csv")
    assert payload["comparison_summary"][0]["comparison"] == "candidate"
    assert payload["top_phones_by_p95_delta"][0]["phone"] == "sm-a505u"
    assert payload["worst_rows_by_delta"][0]["phone"] == "sm-a505u"
    assert (tmp_path / "comparison_summary.csv").is_file()
    assert (tmp_path / "phone_delta_summary.csv").is_file()
