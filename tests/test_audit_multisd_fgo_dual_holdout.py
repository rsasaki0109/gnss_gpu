import csv
from pathlib import Path

from experiments.audit_multisd_fgo_dual_holdout import audit_dual_holdout


SHADOW_FIELDS = ["epoch_index", "tow", "shadow_fixed", "x", "y", "z", "runtime_ms"]


def _write_shadow(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(SHADOW_FIELDS)
        writer.writerows(rows)


def test_dual_holdout_is_baseline_priority_and_fail_closed(tmp_path: Path) -> None:
    primary = tmp_path / "primary.csv"
    secondary = tmp_path / "secondary.csv"
    reference = tmp_path / "reference.csv"
    baseline = tmp_path / "baseline.pos"

    _write_shadow(
        primary,
        [
            [0, 1.0, 1, 10.0, 0.0, 0.0, 2.0],
            [1, 2.0, 1, 20.0, 0.0, 0.0, 3.0],
            [2, 3.0, 1, 30.0, 0.0, 0.0, 4.0],
            [3, 4.0, 0, 0.0, 0.0, 0.0, 5.0],
        ],
    )
    _write_shadow(
        secondary,
        [
            [0, 1.0, 1, 99.0, 0.0, 0.0, 7.0],
            [1, 2.0, 1, 20.05, 0.0, 0.0, 8.0],
            [2, 3.0, 1, 31.0, 0.0, 0.0, 9.0],
            [3, 4.0, 1, 41.2, 0.0, 0.0, 10.0],
        ],
    )
    reference.write_text(
        "GPS TOW (s),ECEF X (m),ECEF Y (m),ECEF Z (m)\n"
        "1.0,10,0,0\n2.0,20,0,0\n3.0,30,0,0\n4.0,40,0,0\n",
        encoding="utf-8",
    )
    baseline.write_text(
        "% test\n"
        "0 1.0 10 0 0 0 0 0 4\n"
        "0 2.0 0 0 0 0 0 0 3\n"
        "0 3.0 0 0 0 0 0 0 3\n"
        "0 4.0 0 0 0 0 0 0 3\n",
        encoding="utf-8",
    )

    audit = audit_dual_holdout(
        primary,
        secondary,
        reference,
        baseline_pos_path=baseline,
        maximum_conflict_separation_m=0.1,
    )

    assert audit["result"]["fixed"] == 3
    assert audit["result"]["correct"] == 2
    assert audit["result"]["false"] == 1
    assert audit["result"]["above_1m"] == 1
    assert audit["consensus"] == {
        "both_fixed": 2,
        "primary_only": 0,
        "secondary_only": 1,
        "conflicts_rejected": 1,
        "maximum_accepted_separation_m": 0.05000000000000071,
    }
    assert audit["runtime"]["sequential_max_ms"] == 15.0
    assert audit["runtime"]["parallel_lower_bound_max_ms"] == 10.0
