import csv
from pathlib import Path

import pytest

from experiments.run_multisd_fgo_ppc_cv import (
    Policy,
    artifacts_complete,
    nested_leave_one_run_out,
    score_artifact,
)


def _write_inputs(tmp_path: Path, errors: list[float]) -> tuple[Path, Path, Path]:
    pos = tmp_path / "solution.pos"
    pos.write_text(
        "% header\n"
        + "".join(
            f"2324 {100.0 + index:.1f} 0 0 0 0 0 0 3\n"
            for index in range(len(errors))
        ),
        encoding="utf-8",
    )
    reference = tmp_path / "reference.csv"
    with reference.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["GPS TOW (s)", "ECEF X (m)", "ECEF Y (m)", "ECEF Z (m)"]
        )
        for index in range(len(errors)):
            writer.writerow((100.0 + index, 0.0, 0.0, 0.0))
    shadow = tmp_path / "shadow.csv"
    with shadow.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch_index",
                "tow",
                "shadow_fixed",
                "x",
                "y",
                "z",
                "runtime_ms",
            ),
        )
        writer.writeheader()
        for index, error in enumerate(errors):
            writer.writerow(
                {
                    "epoch_index": index,
                    "tow": 100.0 + index,
                    "shadow_fixed": "1",
                    "x": error,
                    "y": 0.0,
                    "z": 0.0,
                    "runtime_ms": 10.0 + index,
                }
            )
    return pos, shadow, reference


def test_score_artifact_counts_warmup_correct_false_and_blocks(tmp_path: Path) -> None:
    pos, shadow, reference = _write_inputs(tmp_path, [0.1, 0.49, 0.5, 1.1])
    score = score_artifact(
        "tokyo",
        "run1",
        Policy("test", 2, 2, 0, 4, 0.5, 1, 1.0, 4, 4),
        pos,
        shadow,
        reference,
        block_count=2,
    )

    assert score["route"]["epochs"] == 4
    assert score["route"]["correct_fixed_epochs"] == 2
    assert score["route"]["false_fixed_epochs"] == 2
    assert score["route"]["false_fixed_above_1m_epochs"] == 1
    assert score["route"]["false_per_fixed"] == pytest.approx(0.5)
    assert score["baseline"]["fixed_epochs"] == 0
    assert score["baseline_priority_union"]["correct_fixed_epochs"] == 2
    assert score["baseline_priority_union"]["false_fixed_epochs"] == 2
    assert score["baseline_priority_union"]["shadow_rescue_epochs"] == 4
    assert len(score["contiguous_time_blocks"]) == 2


def test_artifacts_complete_rejects_trailing_corrupt_shadow_row(tmp_path: Path) -> None:
    pos, shadow, _ = _write_inputs(tmp_path, [0.1, 0.2])
    assert artifacts_complete(pos, shadow, 0)

    with shadow.open("a", encoding="utf-8") as stream:
        stream.write("75574\n")

    assert not artifacts_complete(pos, shadow, 0)


def _synthetic_score(city: str, run: str, policy: str, correct: int, false: int):
    route = {
        "epochs": 10,
        "fixed_epochs": correct + false,
        "correct_fixed_epochs": correct,
        "false_fixed_epochs": false,
        "false_fixed_above_1m_epochs": false,
        "correct_fix_rate": correct / 10,
        "false_per_fixed": false / (correct + false) if correct + false else 0.0,
    }
    return {
        "city": city,
        "run": run,
        "policy": {"name": policy},
        "route": route,
        "baseline_priority_union": route,
        "contiguous_time_blocks": [
            {"false_fixed_epochs": false, "baseline_priority_union": route}
        ],
    }


def test_nested_cv_prefers_zero_false_policy_before_higher_availability() -> None:
    scores = []
    for city in ("tokyo", "nagoya"):
        for run in ("run1", "run2", "run3"):
            scores.append(_synthetic_score(city, run, "safe", 7, 0))
            scores.append(_synthetic_score(city, run, "unsafe", 9, 1))

    audit = nested_leave_one_run_out(scores)

    assert audit["complete"] is True
    assert len(audit["folds"]) == 6
    assert all(fold["selected_policy"] == "safe" for fold in audit["folds"])
    assert audit["aggregate"]["false_fixed_epochs"] == 0
