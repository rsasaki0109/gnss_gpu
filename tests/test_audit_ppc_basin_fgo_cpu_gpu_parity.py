import csv
import json
from pathlib import Path

from experiments.audit_ppc_basin_fgo_cpu_gpu_parity import audit_cpu_gpu_parity


def _shadow(path: Path, x: float, fixed: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "tow", "shadow_fixed", "validation_pass", "selected_rank",
                "x", "y", "z", "runtime_ms", "cuda_selected",
                "cuda_hypothesis_batch_successes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {"tow": 1, "shadow_fixed": fixed, "validation_pass": fixed,
             "selected_rank": 0, "x": x, "y": 0, "z": 0,
             "runtime_ms": 2, "cuda_selected": 1,
             "cuda_hypothesis_batch_successes": 1}
        )


def _basin(path: Path, x: float, passed: bool) -> None:
    path.write_text(
        json.dumps(
            {"schema": "gnsspp_multisd_basin_v1", "epoch_index": 0,
             "group_index": 0, "rank": 0, "pass": passed,
             "position_ecef": [x, 0, 0], "incremental_log_likelihood": 1,
             "fixed_integers": [{"fixed_cycles": 1}]}
        ) + "\n",
        encoding="utf-8",
    )


def test_parity_passes_with_identical_acceptance_and_ten_micrometre_position() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        _shadow(root / "cpu.csv", 1.0, 1)
        _shadow(root / "gpu.csv", 1.0 + 1.0e-6, 1)
        _basin(root / "cpu.jsonl", 1.0, True)
        _basin(root / "gpu.jsonl", 1.0 + 1.0e-6, True)
        result = audit_cpu_gpu_parity(
            root / "cpu.csv", root / "gpu.csv",
            root / "cpu.jsonl", root / "gpu.jsonl",
        )
    assert result["passed"] is True
    assert result["acceptance_identity"] is True
    assert result["maximum_ecef_difference_m"] < 1.0e-5


def test_parity_fails_on_acceptance_difference(tmp_path: Path) -> None:
    _shadow(tmp_path / "cpu.csv", 1.0, 1)
    _shadow(tmp_path / "gpu.csv", 1.0, 0)
    _basin(tmp_path / "cpu.jsonl", 1.0, True)
    _basin(tmp_path / "gpu.jsonl", 1.0, False)
    result = audit_cpu_gpu_parity(
        tmp_path / "cpu.csv", tmp_path / "gpu.csv",
        tmp_path / "cpu.jsonl", tmp_path / "gpu.jsonl",
    )
    assert result["passed"] is False
