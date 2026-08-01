from __future__ import annotations

from pathlib import Path

import pytest

from experiments.compose_ppc_safe_basin_union import compose_safe_union


def _write_pos(path: Path, statuses: list[int]) -> None:
    path.write_text(
        "% test\n"
        + "".join(
            f"0 {index} {10.0 * index} 0 0 0 0 0 {status}\n"
            for index, status in enumerate(statuses, start=1)
        ),
        encoding="utf-8",
    )


def test_safe_union_prioritizes_monitor_then_guarded_promotion_then_pf(
    tmp_path: Path,
) -> None:
    monitor = tmp_path / "monitor.pos"
    active = tmp_path / "active.pos"
    integrity = tmp_path / "integrity.csv"
    tracker = tmp_path / "tracker.csv"
    _write_pos(monitor, [4, 3, 3, 3, 3])
    _write_pos(active, [4, 4, 4, 3, 4])
    integrity.write_text(
        "tow,satellite_par_surplus_passed,causal_arc_resets\n"
        "1,0,0\n2,0,0\n3,0,0\n4,0,0\n5,0,0\n",
        encoding="utf-8",
    )
    tracker.write_text(
        "tow,shadow_fixed,x,y,z\n"
        "1,1,999,0,0\n2,1,20,0,0\n3,0,30,0,0\n"
        "4,0,40,0,0\n5,0,50,0,0\n",
        encoding="utf-8",
    )

    rows = compose_safe_union(monitor, active, integrity, tracker)

    assert [row["source"] for row in rows] == [
        "library_monitor",
        "pf_fgo_rescue",
        "library_guarded_promotion",
        "abstain",
        "library_guarded_promotion",
    ]
    assert rows[0]["x"] == 10.0
    assert rows[4]["motion_pass"] == 1


def test_safe_union_strict_surplus_accepts_first_promotion(tmp_path: Path) -> None:
    monitor = tmp_path / "monitor.pos"
    active = tmp_path / "active.pos"
    integrity = tmp_path / "integrity.csv"
    tracker = tmp_path / "tracker.csv"
    _write_pos(monitor, [3])
    _write_pos(active, [4])
    integrity.write_text(
        "tow,satellite_par_surplus_passed,causal_arc_resets\n1,1,9\n",
        encoding="utf-8",
    )
    tracker.write_text("tow,shadow_fixed,x,y,z\n1,0,10,0,0\n", encoding="utf-8")
    rows = compose_safe_union(monitor, active, integrity, tracker)
    assert rows[0]["source"] == "library_guarded_promotion"


def test_safe_union_rejects_duplicate_integrity_tow(tmp_path: Path) -> None:
    monitor = tmp_path / "monitor.pos"
    active = tmp_path / "active.pos"
    integrity = tmp_path / "integrity.csv"
    tracker = tmp_path / "tracker.csv"
    _write_pos(monitor, [3])
    _write_pos(active, [4])
    integrity.write_text(
        "tow,satellite_par_surplus_passed,causal_arc_resets\n"
        "1,1,0\n1,1,0\n",
        encoding="utf-8",
    )
    tracker.write_text("tow,shadow_fixed,x,y,z\n1,0,10,0,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate TOW"):
        compose_safe_union(monitor, active, integrity, tracker)
