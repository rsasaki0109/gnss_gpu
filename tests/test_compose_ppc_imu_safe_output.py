from __future__ import annotations

from pathlib import Path

from experiments.compose_ppc_imu_safe_output import compose_safe_output


def test_safe_output_never_inherits_baseline_fixed_status(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.pos"
    baseline.write_text(
        "% test\n"
        "1 10.0 100 0 0 0 0 0 4\n"
        "1 11.0 110 0 0 0 0 0 3\n",
        encoding="utf-8",
    )
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "epoch_index,tow,shadow_fixed,x,y,z\n"
        "0,10.0,0,101,0,0\n"
        "1,11.0,1,111,0,0\n"
        "2,12.0,0,120,0,0\n",
        encoding="utf-8",
    )

    rows = compose_safe_output(baseline, tracker)
    assert [row["status"] for row in rows] == [3, 4, 3]
    assert [row["source"] for row in rows] == [
        "baseline_float",
        "imu_pf_fgo_fixed",
        "imu_pf_fgo_float",
    ]
    assert rows[0]["baseline_status"] == 4
    assert rows[0]["shadow_fixed"] == 0
    assert rows[1]["x"] == 111.0


def test_safe_output_rejects_nonfinite_fixed_position(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.pos"
    baseline.write_text("% test\n1 10.0 100 0 0 0 0 0 3\n", encoding="utf-8")
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "epoch_index,tow,shadow_fixed,x,y,z\n0,10.0,1,nan,0,0\n",
        encoding="utf-8",
    )

    try:
        compose_safe_output(baseline, tracker)
    except ValueError as exc:
        assert "non-finite fixed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-finite fixed position rejection")
