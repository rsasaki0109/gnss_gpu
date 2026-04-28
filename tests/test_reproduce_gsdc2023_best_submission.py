import pandas as pd
import pytest

from experiments.reproduce_gsdc2023_best_submission import apply_row_coordinate_overrides
from experiments.reproduce_gsdc2023_best_submission import apply_patch_source_preset
from experiments.reproduce_gsdc2023_best_submission import CURRENT_1450_SUBMISSION
from experiments.reproduce_gsdc2023_best_submission import promote_final_row
from experiments.reproduce_gsdc2023_best_submission import replace_trip_coordinates


def _frame(lat: float, lon: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": [
                "2020-12-11-19-30-us-ca-mtv-e/pixel4xl",
                "other/phone",
            ],
            "UnixTimeMillis": [1607716162442, 1],
            "LatitudeDegrees": [lat, 1.0],
            "LongitudeDegrees": [lon, 2.0],
        }
    )


def test_promote_final_row_replaces_only_scored_pixel4xl_row() -> None:
    cap100 = _frame(37.0, -122.0)
    cap1000 = _frame(38.0, -123.0)

    final, summary = promote_final_row(cap100, cap1000)

    assert summary["row_index"] == 0
    assert final.loc[0, "LatitudeDegrees"] == 38.0
    assert final.loc[0, "LongitudeDegrees"] == -123.0
    assert final.loc[1, "LatitudeDegrees"] == 1.0
    assert final.loc[1, "LongitudeDegrees"] == 2.0


def test_promote_final_row_rejects_key_mismatch() -> None:
    cap100 = _frame(37.0, -122.0)
    cap1000 = _frame(38.0, -123.0)
    cap1000.loc[0, "UnixTimeMillis"] = 2

    with pytest.raises(ValueError, match="different key columns"):
        promote_final_row(cap100, cap1000)


def test_replace_trip_coordinates_replaces_requested_trips_only() -> None:
    base = pd.DataFrame(
        {
            "tripId": ["trip/a", "trip/b", "trip/a"],
            "UnixTimeMillis": [1, 2, 3],
            "LatitudeDegrees": [10.0, 20.0, 30.0],
            "LongitudeDegrees": [40.0, 50.0, 60.0],
        }
    )
    patch = base.copy()
    patch["LatitudeDegrees"] = [11.0, 21.0, 31.0]
    patch["LongitudeDegrees"] = [41.0, 51.0, 61.0]

    out, summary = replace_trip_coordinates(base, patch, ("trip/a",))

    assert summary["rows_replaced"] == 2
    assert summary["rows_by_trip"] == {"trip/a": 2}
    assert summary["source_by_trip"] == {"trip/a": "patch_submission"}
    assert out["LatitudeDegrees"].tolist() == [11.0, 20.0, 31.0]
    assert out["LongitudeDegrees"].tolist() == [41.0, 50.0, 61.0]


def test_replace_trip_coordinates_can_use_bridge_position_override(tmp_path) -> None:
    base = pd.DataFrame(
        {
            "tripId": ["trip/a", "trip/b", "trip/a"],
            "UnixTimeMillis": [1, 2, 3],
            "LatitudeDegrees": [10.0, 20.0, 30.0],
            "LongitudeDegrees": [40.0, 50.0, 60.0],
        }
    )
    patch = base.copy()
    patch["LatitudeDegrees"] = [11.0, 21.0, 31.0]
    patch["LongitudeDegrees"] = [41.0, 51.0, 61.0]
    bridge_positions = tmp_path / "bridge_positions.csv"
    pd.DataFrame(
        {
            "UnixTimeMillis": [1, 3],
            "LatitudeDegrees": [12.0, 32.0],
            "LongitudeDegrees": [42.0, 62.0],
        }
    ).to_csv(bridge_positions, index=False)

    out, summary = replace_trip_coordinates(
        base,
        patch,
        ("trip/a",),
        trip_position_overrides={"trip/a": bridge_positions},
    )

    assert summary["rows_replaced"] == 2
    assert summary["source_by_trip"] == {"trip/a": "bridge_positions"}
    assert summary["trip_position_overrides"]["trip/a"]["rows"] == 2
    assert out["LatitudeDegrees"].tolist() == [12.0, 20.0, 32.0]
    assert out["LongitudeDegrees"].tolist() == [42.0, 50.0, 62.0]


def test_apply_row_coordinate_overrides_replaces_keyed_subset(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "tripId": ["trip/a", "trip/a", "trip/b"],
            "UnixTimeMillis": [1, 3, 1],
            "LatitudeDegrees": [10.0, 30.0, 50.0],
            "LongitudeDegrees": [40.0, 60.0, 80.0],
        }
    )
    row_positions = tmp_path / "row_positions.csv"
    pd.DataFrame(
        {
            "tripId": ["trip/a"],
            "UnixTimeMillis": [3],
            "LatitudeDegrees": [31.0],
            "LongitudeDegrees": [61.0],
        }
    ).to_csv(row_positions, index=False)

    out, summary = apply_row_coordinate_overrides(frame, {"trip/a": row_positions})

    assert summary["rows_replaced"] == 1
    assert summary["trips"]["trip/a"]["rows"] == 1
    assert out["LatitudeDegrees"].tolist() == [10.0, 31.0, 50.0]
    assert out["LongitudeDegrees"].tolist() == [40.0, 61.0, 80.0]


def test_bridge_exception_patch_source_uses_base_as_key_submission() -> None:
    base = CURRENT_1450_SUBMISSION.with_name("submission_20260421_0555.csv")

    patch_submission, trip_positions, row_positions = apply_patch_source_preset(
        "bridge-exception",
        base_submission=base,
        patch_submission=CURRENT_1450_SUBMISSION,
        patch_trip_positions=[],
        patch_row_positions=[],
    )

    assert patch_submission == base
    assert len(trip_positions) == 2
    assert len(row_positions) == 1
    assert any("pixel4xl=" in value for value in trip_positions)
    assert any("sm-a505u=" in value for value in trip_positions)
    assert "sm-a505u=" in row_positions[0]
