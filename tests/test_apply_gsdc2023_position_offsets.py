import argparse

import numpy as np
import pandas as pd
import pytest

from experiments.apply_gsdc2023_position_offsets import (
    apply_offsets,
    parse_phone_scale_override,
    parse_trip_scale_override,
    scale_map_for_policy,
)
from experiments.smooth_gsdc2023_submission import haversine_m


def test_scale_map_for_policy():
    supported = scale_map_for_policy("supported", 2.0)
    assert supported["mi8"] == 2.0
    assert supported["sm-g988b"] == 2.0
    assert "pixel5" not in supported

    tuned = scale_map_for_policy("phone-tuned", 1.0)
    assert tuned["mi8"] == 2.5
    assert tuned["pixel6pro"] == 3.0
    assert tuned["pixel7pro"] == 4.0
    assert tuned["sm-s908b"] == 2.0

    with pytest.raises(ValueError):
        scale_map_for_policy("bad-policy", 1.0)


def test_parse_phone_scale_override():
    assert parse_phone_scale_override("Pixel6Pro=3.25") == ("pixel6pro", 3.25)
    assert parse_phone_scale_override("sm-s908b=0") == ("sm-s908b", 0.0)

    for value in ["pixel6pro", "=3.0", "pixel6pro=bad", "pixel6pro=-1"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_phone_scale_override(value)


def test_parse_trip_scale_override():
    assert parse_trip_scale_override("2023-05-25/test/pixel6pro=3.25") == (
        "2023-05-25/test/pixel6pro",
        3.25,
    )
    assert parse_trip_scale_override("trip/phone=0") == ("trip/phone", 0.0)

    for value in ["trip/phone", "=3.0", "trip/phone=bad", "trip/phone=-1"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_trip_scale_override(value)


def test_apply_offsets_changes_only_configured_phones():
    source = pd.DataFrame(
        {
            "tripId": [
                "2021-01-01-00-00-us-ca-test/mi8",
                "2021-01-01-00-00-us-ca-test/mi8",
                "2021-01-01-00-00-us-ca-test/pixel5",
                "2021-01-01-00-00-us-ca-test/pixel5",
            ],
            "UnixTimeMillis": [1000, 2000, 1000, 2000],
            "LatitudeDegrees": [37.0, 37.00001, 37.1, 37.10001],
            "LongitudeDegrees": [-122.0, -121.99999, -122.1, -122.09999],
        },
    )

    output, trip_summary = apply_offsets(source, {"mi8": 2.0})

    mi8_mask = source["tripId"].str.endswith("/mi8").to_numpy()
    pixel5_mask = source["tripId"].str.endswith("/pixel5").to_numpy()
    mi8_delta = haversine_m(
        source.loc[mi8_mask, "LatitudeDegrees"].to_numpy(),
        source.loc[mi8_mask, "LongitudeDegrees"].to_numpy(),
        output.loc[mi8_mask, "LatitudeDegrees"].to_numpy(),
        output.loc[mi8_mask, "LongitudeDegrees"].to_numpy(),
    )
    pixel5_delta = haversine_m(
        source.loc[pixel5_mask, "LatitudeDegrees"].to_numpy(),
        source.loc[pixel5_mask, "LongitudeDegrees"].to_numpy(),
        output.loc[pixel5_mask, "LatitudeDegrees"].to_numpy(),
        output.loc[pixel5_mask, "LongitudeDegrees"].to_numpy(),
    )

    assert np.all(mi8_delta > 0.1)
    assert np.allclose(pixel5_delta, 0.0)
    assert dict(zip(trip_summary["phone"], trip_summary["scale"])) == {"mi8": 2.0, "pixel5": 0.0}


def test_apply_offsets_trip_scale_overrides_phone_scale():
    source = pd.DataFrame(
        {
            "tripId": [
                "2021-01-01-00-00-us-ca-test-a/mi8",
                "2021-01-01-00-00-us-ca-test-a/mi8",
                "2021-01-01-00-00-us-ca-test-b/mi8",
                "2021-01-01-00-00-us-ca-test-b/mi8",
            ],
            "UnixTimeMillis": [1000, 2000, 1000, 2000],
            "LatitudeDegrees": [37.0, 37.00001, 37.1, 37.10001],
            "LongitudeDegrees": [-122.0, -121.99999, -122.1, -122.09999],
        },
    )

    output, trip_summary = apply_offsets(
        source,
        {"mi8": 0.0},
        {"2021-01-01-00-00-us-ca-test-a/mi8": 2.0},
    )

    trip_a = source["tripId"].str.contains("test-a").to_numpy()
    trip_b = source["tripId"].str.contains("test-b").to_numpy()
    trip_a_delta = haversine_m(
        source.loc[trip_a, "LatitudeDegrees"].to_numpy(),
        source.loc[trip_a, "LongitudeDegrees"].to_numpy(),
        output.loc[trip_a, "LatitudeDegrees"].to_numpy(),
        output.loc[trip_a, "LongitudeDegrees"].to_numpy(),
    )
    trip_b_delta = haversine_m(
        source.loc[trip_b, "LatitudeDegrees"].to_numpy(),
        source.loc[trip_b, "LongitudeDegrees"].to_numpy(),
        output.loc[trip_b, "LatitudeDegrees"].to_numpy(),
        output.loc[trip_b, "LongitudeDegrees"].to_numpy(),
    )

    assert np.all(trip_a_delta > 0.1)
    assert np.allclose(trip_b_delta, 0.0)
    assert dict(zip(trip_summary["tripId"], trip_summary["scale"])) == {
        "2021-01-01-00-00-us-ca-test-a/mi8": 2.0,
        "2021-01-01-00-00-us-ca-test-b/mi8": 0.0,
    }
