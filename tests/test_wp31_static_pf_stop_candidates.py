from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from extract_wp31_static_pf_stop_candidates import (
    extract_stop_candidates,
    parse_segment,
)


def test_parse_segment_rejects_invalid_interval() -> None:
    with pytest.raises(Exception):
        parse_segment("8:8")


def test_extracts_multiple_segments_without_cross_contamination(tmp_path: Path) -> None:
    path = tmp_path / "basins.csv"
    rows = []
    for epoch, center in ((10, 0.0), (11, 0.0), (30, 100.0), (31, 100.0)):
        for delta in (0.0, 0.02):
            rows.append(
                {
                    "epoch": epoch,
                    "ecef_x": center + delta,
                    "ecef_y": 0.0,
                    "ecef_z": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    nodes = extract_stop_candidates(
        path,
        [(10, 12), (30, 32)],
        sample_stride_epochs=1,
        radius_m=0.2,
        dedup_radius_m=0.2,
        max_candidates=4,
        chunksize=3,
    )
    assert [node["segment"] for node in nodes] == [[10, 12], [30, 32]]
    assert all(node["candidate_count"] == 1 for node in nodes)
    assert nodes[0]["candidates"][0]["position_ecef"][0] < 1.0
    assert nodes[1]["candidates"][0]["position_ecef"][0] > 99.0
