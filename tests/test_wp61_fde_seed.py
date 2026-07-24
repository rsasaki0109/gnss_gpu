from __future__ import annotations

import csv

from experiments.build_wp61_fde_seed import build


def test_build_uses_only_accepted_anchor_coordinates(tmp_path) -> None:
    trajectory = tmp_path / "trajectory.csv"
    with trajectory.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["epoch", "tow", "ecef_x", "ecef_y", "ecef_z"]
        )
        writer.writeheader()
        for epoch in range(3):
            writer.writerow(
                {
                    "epoch": epoch,
                    "tow": epoch,
                    "ecef_x": epoch,
                    "ecef_y": 0,
                    "ecef_z": 0,
                }
            )
    anchors = tmp_path / "anchors.csv"
    with anchors.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "tow",
                "status",
                "anchor_x_m",
                "anchor_y_m",
                "anchor_z_m",
                "anchor_error_m",
            ],
        )
        writer.writeheader()
        for epoch in range(3):
            writer.writerow(
                {
                    "tow": epoch,
                    "status": "accepted",
                    "anchor_x_m": epoch + 2,
                    "anchor_y_m": 3,
                    "anchor_z_m": 4,
                    "anchor_error_m": 999,
                }
            )

    result = build(anchors, trajectory, segment=(0, 3))

    assert result["seeds"] == [{"offset_ecef_m": [2.0, 3.0, 4.0]}]
    assert result["truth_usage"] == "none"
    assert "anchor_error" not in str(result)
