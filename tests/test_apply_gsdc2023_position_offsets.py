import numpy as np
import pandas as pd
import pytest

from experiments.apply_gsdc2023_position_offsets import apply_offsets, scale_map_for_policy
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
