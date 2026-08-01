import json
from pathlib import Path

import pytest

from experiments.aggregate_ppc_basin_fgo_six_route import aggregate_summaries


def _write(path: Path, routes: list[str], *, top_k: int = 8) -> Path:
    items = []
    for index, route in enumerate(routes, start=1):
        items.append(
            {
                "route": route,
                "audit": {
                    "total_epochs": 100,
                    "fixed": index,
                    "correct_fix": index,
                    "false_fix": 0,
                    "false_fix_above_1m": 0,
                    "integrity": {"passed": True},
                },
                "candidate_supply_audit": {
                    "evaluated_epochs": 90,
                    "oracle_correct_epochs": 80,
                    "passed_correct_epochs": index,
                    "unique_pass_epochs": index,
                    "unique_pass_correct_epochs": index,
                    "integrity": {"passed": True},
                },
            }
        )
    payload = {
        "schema": "gnss_gpu_ppc_basin_fgo_six_route_v1",
        "production_input_truth": False,
        "binary_sha256": "abc",
        "max_epochs": 0,
        "skip_epochs": 0,
        "top_k": top_k,
        "fix_min_streak": 2,
        "validation_gap_tolerance_epochs": 1,
        "cuda_mode": "off",
        "imu_enabled": False,
        "native_imu_enabled": True,
        "native_imu_aperture_m": 0.0,
        "native_imu_fix_min_streak": 0,
        "routes": items,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_aggregates_complete_disjoint_six_route_set(tmp_path: Path) -> None:
    names = [f"{city}/run{run}" for city in ("tokyo", "nagoya") for run in range(1, 4)]
    paths = [_write(tmp_path / f"{i}.json", [name]) for i, name in enumerate(names)]
    result = aggregate_summaries(paths)
    assert result["totals"]["total_epochs"] == 600
    assert result["totals"]["correct_fix"] == 6
    assert result["integrity"]["passed"] is True


def test_rejects_config_mismatch(tmp_path: Path) -> None:
    names = [f"{city}/run{run}" for city in ("tokyo", "nagoya") for run in range(1, 4)]
    paths = [
        _write(tmp_path / f"{i}.json", [name], top_k=4 if i == 5 else 8)
        for i, name in enumerate(names)
    ]
    with pytest.raises(ValueError, match="config mismatch"):
        aggregate_summaries(paths)
