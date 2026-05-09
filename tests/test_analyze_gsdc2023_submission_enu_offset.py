from __future__ import annotations

import json

import pandas as pd

from experiments.analyze_gsdc2023_submission_enu_offset import (
    analyze_submission_enu_offset,
    main,
)
from experiments.smooth_gsdc2023_submission import latlon_to_local_m, local_m_to_latlon


TRIP_A = "trip-a/sm-a325f"
TRIP_B = "trip-b/mi8"


def _shift_latlon(lat: list[float], lon: list[float], east_m: float, north_m: float) -> tuple[list[float], list[float]]:
    east, north, origin_lat, origin_lon = latlon_to_local_m(pd.Series(lat).to_numpy(), pd.Series(lon).to_numpy())
    out_lat, out_lon = local_m_to_latlon(east + east_m, north + north_m, origin_lat, origin_lon)
    return out_lat.tolist(), out_lon.tolist()


def _write_submission(path, trip_ids: list[str], lats: list[float], lons: list[float]) -> None:
    frame = pd.DataFrame(
        {
            "tripId": trip_ids,
            "UnixTimeMillis": [1000 + 1000 * index for index in range(len(trip_ids))],
            "LatitudeDegrees": lats,
            "LongitudeDegrees": lons,
        },
    )
    frame.to_csv(path, index=False)


def test_analyze_submission_enu_offset_detects_constant_trip_offset(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    trip_ids = [TRIP_A] * 5
    ref_lats = [37.0, 37.00001, 37.00002, 37.00003, 37.00004]
    ref_lons = [-122.0, -122.00001, -122.00002, -122.00003, -122.00004]
    cand_lats, cand_lons = _shift_latlon(ref_lats, ref_lons, east_m=0.4, north_m=-0.2)
    _write_submission(reference, trip_ids, ref_lats, ref_lons)
    _write_submission(candidate, trip_ids, cand_lats, cand_lons)

    rows, summary, payload = analyze_submission_enu_offset(
        reference_submission=reference,
        candidate_submission=candidate,
        target_phones={"sm-a325f"},
    )

    trip = summary[summary["group_type"] == "trip"].iloc[0]
    assert len(rows) == 5
    assert payload["rows"] == 5
    assert abs(trip["median_candidate_minus_reference_east_m"] - 0.4) < 1e-6
    assert abs(trip["median_candidate_minus_reference_north_m"] + 0.2) < 1e-6
    assert trip["residual_after_median_p95_m"] < 1e-6


def test_analyze_submission_enu_offset_keeps_nonconstant_residual_tail(tmp_path) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    trip_ids = [TRIP_A] * 5 + [TRIP_B] * 5
    ref_lats = [37.0 + index * 0.00001 for index in range(10)]
    ref_lons = [-122.0 - index * 0.00001 for index in range(10)]
    cand_lats, cand_lons = _shift_latlon(ref_lats, ref_lons, east_m=0.4, north_m=-0.2)
    spike_lat, spike_lon = _shift_latlon([ref_lats[4]], [ref_lons[4]], east_m=8.0, north_m=0.0)
    cand_lats[4] = spike_lat[0]
    cand_lons[4] = spike_lon[0]
    _write_submission(reference, trip_ids, ref_lats, ref_lons)
    _write_submission(candidate, trip_ids, cand_lats, cand_lons)

    _, summary, _ = analyze_submission_enu_offset(
        reference_submission=reference,
        candidate_submission=candidate,
        target_phones={"sm-a325f"},
    )

    trip = summary[(summary["group_type"] == "trip") & (summary["group"] == TRIP_A)].iloc[0]
    assert trip["original_rows_gt_5m"] == 1
    assert trip["residual_after_median_rows_gt_5m"] == 1
    assert trip["residual_after_median_max_m"] > 7.0


def test_analyze_submission_enu_offset_cli_writes_outputs(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    output = tmp_path / "out"
    trip_ids = [TRIP_A] * 3
    ref_lats = [37.0, 37.00001, 37.00002]
    ref_lons = [-122.0, -122.00001, -122.00002]
    cand_lats, cand_lons = _shift_latlon(ref_lats, ref_lons, east_m=0.2, north_m=0.1)
    _write_submission(reference, trip_ids, ref_lats, ref_lons)
    _write_submission(candidate, trip_ids, cand_lats, cand_lons)

    assert (
        main(
            [
                "--reference-submission",
                str(reference),
                "--candidate-submission",
                str(candidate),
                "--target-trip",
                TRIP_A,
                "--output-dir",
                str(output),
            ],
        )
        == 0
    )

    assert "analyzed: 3 row(s)" in capsys.readouterr().out
    assert (output / "enu_row_offsets.csv").is_file()
    assert (output / "enu_offset_summary.csv").is_file()
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["target_trips"] == [TRIP_A]
