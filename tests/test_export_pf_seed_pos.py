from __future__ import annotations

import csv
from pathlib import Path

from experiments.export_pf_seed_pos import export_seed_pos


def test_export_seed_pos_writes_gnssplusplus_contract(tmp_path: Path) -> None:
    source = tmp_path / "trajectory.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["tow", "ecef_x", "ecef_y", "ecef_z"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "tow": 123.2,
                "ecef_x": -3960000.0,
                "ecef_y": 3340000.0,
                "ecef_z": 3690000.0,
            }
        )
    output = tmp_path / "seed.pos"
    assert export_seed_pos(source, output, gps_week=2324) == 1
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("% gps_week")
    assert lines[1] == (
        "2324 123.200000000 -3960000.000000000 "
        "3340000.000000000 3690000.000000000"
    )
