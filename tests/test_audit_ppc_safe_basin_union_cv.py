from pathlib import Path

from experiments.audit_ppc_safe_basin_union_cv import CitySpec, blocked_cv


def _city(tmp_path: Path, name: str, offset: float) -> CitySpec:
    monitor = tmp_path / f"{name}.monitor.pos"
    active = tmp_path / f"{name}.active.pos"
    integrity = tmp_path / f"{name}.integrity.csv"
    tracker = tmp_path / f"{name}.tracker.csv"
    reference = tmp_path / f"{name}.reference.csv"
    monitor.write_text(
        "% test\n"
        + "".join(
            f"0 {index} {offset + index} 0 0 0 0 0 4\n" for index in range(1, 5)
        ),
        encoding="utf-8",
    )
    active.write_text(monitor.read_text(encoding="utf-8"), encoding="utf-8")
    integrity.write_text(
        "tow,satellite_par_surplus_passed,causal_arc_resets\n"
        "1,0,0\n2,0,0\n3,0,0\n4,0,0\n",
        encoding="utf-8",
    )
    tracker.write_text(
        "tow,shadow_fixed,x,y,z\n"
        + "".join(f"{index},0,0,0,0\n" for index in range(1, 5)),
        encoding="utf-8",
    )
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
        + "".join(
            f"{index},{offset + index},0,0\n" for index in range(1, 5)
        ),
        encoding="utf-8",
    )
    return CitySpec(name, monitor, active, integrity, tracker, reference)


def test_blocked_cv_keeps_holdout_truth_out_of_policy_decisions(tmp_path: Path) -> None:
    result = blocked_cv(
        [_city(tmp_path, "tokyo", 10.0), _city(tmp_path, "nagoya", 20.0)],
        block_count=2,
    )
    assert result["passed"] is True
    assert len(result["folds"]) == 4
    assert result["holdout_fixed"] == 8
    assert result["holdout_false"] == 0
