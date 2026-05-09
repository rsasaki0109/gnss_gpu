from __future__ import annotations

import json

import pandas as pd

from experiments.apply_gsdc2023_bridge_position_offsets import (
    apply_bridge_position_offsets,
    main,
)


def _bridge_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tripId": ["course/mi8", "course/mi8", "course/mi8"],
            "UnixTimeMillis": [1000, 2000, 3000],
            "SelectedSource": ["baseline", "raw_wls", "baseline"],
            "BaselineLatitudeDegrees": [37.0, 37.00001, 37.00002],
            "BaselineLongitudeDegrees": [-122.0, -121.99999, -121.99998],
            "BaselineAltitudeMeters": [5.0, 6.0, 7.0],
            "RawWlsLatitudeDegrees": [37.0001, 37.00011, 37.00012],
            "RawWlsLongitudeDegrees": [-122.0001, -122.00009, -122.00008],
            "RawWlsAltitudeMeters": [8.0, 9.0, 10.0],
            "LatitudeDegrees": [37.0, 37.00011, 37.00002],
            "LongitudeDegrees": [-122.0, -122.00009, -121.99998],
            "AltitudeMeters": [5.0, 9.0, 7.0],
        },
    )


def test_apply_bridge_position_offsets_scale_zero_is_identity() -> None:
    frame = _bridge_frame()

    output, summary = apply_bridge_position_offsets(frame, phone=None, scale=0.0)

    pd.testing.assert_frame_equal(output, frame)
    assert summary["phone"] == "mi8"
    assert summary["source_count"] == 3
    assert summary["source_summary"]["baseline"]["changed_rows_gt_0p01m"] == 0


def test_apply_bridge_position_offsets_updates_each_source_pair() -> None:
    frame = _bridge_frame()

    output, summary = apply_bridge_position_offsets(frame, phone="mi8", scale=1.0)

    for column in [
        "BaselineLatitudeDegrees",
        "RawWlsLatitudeDegrees",
        "LatitudeDegrees",
        "BaselineAltitudeMeters",
    ]:
        assert not output[column].equals(frame[column])
    assert summary["source_summary"]["baseline"]["changed_rows_gt_0p01m"] == 3
    assert summary["source_summary"]["raw_wls"]["changed_rows_gt_0p01m"] == 3
    assert summary["source_summary"]["selected"]["changed_rows_gt_0p01m"] == 3


def test_apply_bridge_position_offsets_cli_writes_outputs(tmp_path, capsys) -> None:
    bridge = tmp_path / "bridge_positions.csv"
    output_dir = tmp_path / "out"
    _bridge_frame().to_csv(bridge, index=False)

    assert (
        main(
            [
                "--bridge-positions",
                str(bridge),
                "--scale",
                "1.0",
                "--output-dir",
                str(output_dir),
            ],
        )
        == 0
    )

    assert "wrote bridge position offsets: rows=3 sources=3" in capsys.readouterr().out
    output = pd.read_csv(output_dir / "bridge_positions.csv")
    assert len(output) == 3
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["phone"] == "mi8"
    assert payload["source_count"] == 3
    assert payload["sha256"]
