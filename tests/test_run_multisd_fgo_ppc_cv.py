import csv
import argparse
from pathlib import Path

import pytest

from experiments.run_multisd_fgo_ppc_cv import (
    Policy,
    _parse_policy,
    artifacts_complete,
    nested_leave_one_run_out,
    read_shadow,
    score_artifact,
)


def test_parse_policy_keeps_constellation_par_explicit() -> None:
    policy = _parse_policy(
        "p:10:10:2:4:0.5:3:0.75:6:4:1:1.1:8:2:0.1:0.25"
    )
    assert policy.constellation_ranked_par
    assert policy.candidate_ratio == pytest.approx(1.1)
    assert policy.candidate_groups == 8
    assert policy.fallback_consensus_groups == 2
    assert policy.fallback_consensus_separation_m == pytest.approx(0.1)
    assert policy.fallback_max_seed_separation_m == pytest.approx(0.25)
    assert not policy.quality_ranked_par

    quality_policy = _parse_policy(
        "q:10:10:2:4:0.5:3:0.75:6:4:1:1.1:8:2:0.1:0.25:1"
    )
    assert quality_policy.quality_ranked_par
    interleaved_policy = _parse_policy(
        "qi:10:10:2:4:0.5:3:0.75:6:4:1:1.1:8:2:0.1:0.25:1:1"
    )
    assert interleaved_policy.quality_ranked_par
    assert interleaved_policy.interleave_constellation_par
    success_policy = _parse_policy(
        "qs:10:10:2:4:0.5:3:0.75:6:4:1:1.1:8:2:0.1:0.25:1:0:"
        "0.9999:0.1"
    )
    assert success_policy.minimum_bootstrapped_success_rate == pytest.approx(
        0.9999
    )
    assert success_policy.maximum_adop_cycles == pytest.approx(0.1)
    fallback_success_policy = _parse_policy(
        "qf:10:10:2:4:0.5:3:0.75:6:4:1:1.1:8:2:0.1:0.25:1:0:"
        "0:0:0.9999"
    )
    assert (
        fallback_success_policy.fallback_minimum_bootstrapped_success_rate
        == pytest.approx(0.9999)
    )

    with pytest.raises(argparse.ArgumentTypeError, match="must be 0 or 1"):
        _parse_policy("p:10:10:2:4:0.5:3:0.75:6:4:2:1.5:1:1:0:0")
    with pytest.raises(argparse.ArgumentTypeError, match="QUALITY_RANKED"):
        _parse_policy("p:10:10:2:4:0.5:3:0.75:6:4:1:1.5:1:1:0:0:2")
    with pytest.raises(argparse.ArgumentTypeError, match="INTERLEAVE"):
        _parse_policy("p:10:10:2:4:0.5:3:0.75:6:4:1:1.5:1:1:0:0:1:2")
    with pytest.raises(argparse.ArgumentTypeError, match="invalid policy"):
        _parse_policy(
            "p:10:10:2:4:0.5:3:0.75:6:4:1:1.5:1:1:0:0:1:0:1.1:0.1"
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
        Policy(
            "test", 2, 2, 0, 4, 0.5, 1, 1.0, 4, 4, False,
            1.5, 1, 1, 0.0, 0.0,
        ),
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


def test_read_shadow_accepts_only_scientifically_identical_duplicate(
    tmp_path: Path,
) -> None:
    _, shadow, _ = _write_inputs(tmp_path, [0.1])
    lines = shadow.read_text(encoding="utf-8").splitlines()
    duplicate = lines[1].rsplit(",", 1)[0] + ",99.0"
    shadow.write_text("\n".join((*lines, duplicate)) + "\n", encoding="utf-8")

    assert read_shadow(shadow)[100.0]["runtime_ms"] == "99.0"

    conflicting = duplicate.replace(",1,0.1,", ",0,0.1,")
    shadow.write_text(
        "\n".join((*lines, conflicting)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        read_shadow(shadow)


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


def test_city_stratified_nested_cv_learns_city_specific_policy() -> None:
    scores = []
    for run in ("run1", "run2", "run3"):
        scores.append(_synthetic_score("tokyo", run, "tokyo_policy", 9, 0))
        scores.append(_synthetic_score("tokyo", run, "nagoya_policy", 2, 0))
        scores.append(_synthetic_score("nagoya", run, "tokyo_policy", 1, 0))
        scores.append(_synthetic_score("nagoya", run, "nagoya_policy", 8, 0))

    audit = nested_leave_one_run_out(scores, stratify_city=True)

    assert audit["complete"] is True
    assert all(
        fold["selected_policy"] == f"{fold['holdout_city']}_policy"
        for fold in audit["folds"]
    )
    assert audit["aggregate"]["correct_fixed_epochs"] == 51


def test_nested_cv_handles_intentionally_partial_route_matrix() -> None:
    scores = [
        _synthetic_score("tokyo", "run1", "p", 7, 0),
        _synthetic_score("tokyo", "run2", "p", 8, 0),
    ]

    audit = nested_leave_one_run_out(scores, stratify_city=True)

    assert audit["complete"] is False
    assert audit["folds"] == []
