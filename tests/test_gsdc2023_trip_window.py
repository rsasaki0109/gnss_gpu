import pandas as pd

from experiments.gsdc2023_trip_window import (
    FULL_EPOCH_COUNT,
    settings_epoch_window_for_trip,
    trim_epoch_window,
)


def test_settings_epoch_window_uses_trip_settings_and_requested_limit(tmp_path):
    trip_dir = tmp_path / "train" / "course-a" / "pixel4"
    trip_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Course": "course-a", "Phone": "pixel4", "IdxStart": 3, "IdxEnd": 12},
            {"Course": "course-a", "Phone": "pixel5", "IdxStart": 1, "IdxEnd": 99},
        ],
    ).to_csv(tmp_path / "settings_train.csv", index=False)

    assert settings_epoch_window_for_trip(trip_dir, requested_max_epochs=0) == (2, 10)
    assert settings_epoch_window_for_trip(trip_dir, requested_max_epochs=4) == (2, 4)


def test_settings_epoch_window_defaults_to_full_span_without_settings(tmp_path):
    trip_dir = tmp_path / "train" / "course-a" / "pixel4"
    trip_dir.mkdir(parents=True)

    assert settings_epoch_window_for_trip(trip_dir, requested_max_epochs=0) == (0, FULL_EPOCH_COUNT)
    assert settings_epoch_window_for_trip(trip_dir, requested_max_epochs=5) == (0, 5)


def test_trim_epoch_window_can_exclude_tdcp_edges():
    frame = pd.DataFrame(
        [
            {"epoch_index": 2, "next_epoch_index": 3, "value": "inside"},
            {"epoch_index": 3, "next_epoch_index": 4, "value": "inside_next"},
            {"epoch_index": 4, "next_epoch_index": 5, "value": "edge_out"},
            {"epoch_index": 4, "next_epoch_index": 0, "value": "single_epoch_ok"},
            {"epoch_index": 5, "next_epoch_index": 0, "value": "epoch_out"},
        ],
    )

    trimmed = trim_epoch_window(frame, start_epoch=1, max_epochs=3, next_epoch_column="next_epoch_index")

    assert list(trimmed["value"]) == ["inside", "inside_next", "single_epoch_ok"]
