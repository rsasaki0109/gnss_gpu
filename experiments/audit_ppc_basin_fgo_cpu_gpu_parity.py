#!/usr/bin/env python3
"""Audit scientific parity and runtime of CPU/GPU native basin artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Iterable[float], probability: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(value))
    if not ordered:
        return None
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _shadow(path: Path) -> dict[float, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {round(float(row["tow"]), 3): row for row in csv.DictReader(stream)}


def _basins(path: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    output = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if row.get("schema") != "gnsspp_multisd_basin_v1":
                raise ValueError(f"invalid basin schema on line {line_number}")
            key = (
                int(row["epoch_index"]),
                int(row.get("group_index", -1)),
                int(row.get("rank", -1)),
            )
            if key in output:
                raise ValueError(f"duplicate basin key {key}")
            output[key] = row
    return output


def audit_cpu_gpu_parity(
    cpu_shadow_path: Path,
    gpu_shadow_path: Path,
    cpu_basin_path: Path,
    gpu_basin_path: Path,
    *,
    maximum_ecef_difference_m: float = 1.0e-5,
) -> dict[str, Any]:
    cpu_shadow = _shadow(cpu_shadow_path)
    gpu_shadow = _shadow(gpu_shadow_path)
    cpu_basins = _basins(cpu_basin_path)
    gpu_basins = _basins(gpu_basin_path)
    shadow_keys_identical = set(cpu_shadow) == set(gpu_shadow)
    basin_keys_identical = set(cpu_basins) == set(gpu_basins)
    shadow_position_differences: list[float] = []
    acceptance_identity = shadow_keys_identical and basin_keys_identical
    for tow in set(cpu_shadow) & set(gpu_shadow):
        cpu = cpu_shadow[tow]
        gpu = gpu_shadow[tow]
        acceptance_identity &= all(
            cpu.get(field) == gpu.get(field)
            for field in ("shadow_fixed", "validation_pass", "selected_rank")
        )
        shadow_position_differences.append(
            math.dist(
                tuple(float(cpu[axis]) for axis in "xyz"),
                tuple(float(gpu[axis]) for axis in "xyz"),
            )
        )
    candidate_position_differences: list[float] = []
    evidence_differences: list[float] = []
    integer_identity = basin_keys_identical
    for key in set(cpu_basins) & set(gpu_basins):
        cpu = cpu_basins[key]
        gpu = gpu_basins[key]
        acceptance_identity &= cpu.get("pass") == gpu.get("pass")
        integer_identity &= cpu.get("fixed_integers") == gpu.get("fixed_integers")
        if cpu.get("position_ecef") is not None and gpu.get("position_ecef") is not None:
            candidate_position_differences.append(
                math.dist(cpu["position_ecef"], gpu["position_ecef"])
            )
        if (
            cpu.get("incremental_log_likelihood") is not None
            and gpu.get("incremental_log_likelihood") is not None
        ):
            evidence_differences.append(
                abs(
                    float(cpu["incremental_log_likelihood"])
                    - float(gpu["incremental_log_likelihood"])
                )
            )
    maximum_position_delta = max(
        shadow_position_differences + candidate_position_differences,
        default=math.inf,
    )
    cpu_runtime = [float(row["runtime_ms"]) for row in cpu_shadow.values()]
    gpu_runtime = [float(row["runtime_ms"]) for row in gpu_shadow.values()]
    cpu_p95 = _quantile(cpu_runtime, 0.95)
    gpu_p95 = _quantile(gpu_runtime, 0.95)
    gpu_selected = sum(int(row.get("cuda_selected", "0")) for row in gpu_shadow.values())
    gpu_batch_successes = sum(
        int(row.get("cuda_hypothesis_batch_successes", "0"))
        for row in gpu_shadow.values()
    )
    parity_passed = (
        acceptance_identity
        and integer_identity
        and maximum_position_delta <= maximum_ecef_difference_m
    )
    return {
        "schema": "gnss_gpu_ppc_basin_fgo_cpu_gpu_parity_v1",
        "acceptance_identity": acceptance_identity,
        "integer_identity": integer_identity,
        "maximum_ecef_difference_m": maximum_position_delta,
        "maximum_incremental_log_likelihood_difference": max(
            evidence_differences, default=None
        ),
        "threshold_m": maximum_ecef_difference_m,
        "passed": parity_passed,
        "runtime": {
            "cpu_p95_ms": cpu_p95,
            "gpu_p95_ms": gpu_p95,
            "gpu_to_cpu_p95_ratio": (
                gpu_p95 / cpu_p95 if cpu_p95 and gpu_p95 is not None else None
            ),
            "gpu_selected_epochs": gpu_selected,
            "gpu_batch_successes": gpu_batch_successes,
            "gpu_faster": (
                gpu_p95 < cpu_p95 if cpu_p95 is not None and gpu_p95 is not None else False
            ),
        },
        "artifacts": {
            "cpu_shadow_sha256": _sha256(cpu_shadow_path),
            "gpu_shadow_sha256": _sha256(gpu_shadow_path),
            "cpu_basin_sha256": _sha256(cpu_basin_path),
            "gpu_basin_sha256": _sha256(gpu_basin_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-shadow", type=Path, required=True)
    parser.add_argument("--gpu-shadow", type=Path, required=True)
    parser.add_argument("--cpu-basins", type=Path, required=True)
    parser.add_argument("--gpu-basins", type=Path, required=True)
    parser.add_argument("--maximum-ecef-difference-m", type=float, default=1.0e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_cpu_gpu_parity(
        args.cpu_shadow,
        args.gpu_shadow,
        args.cpu_basins,
        args.gpu_basins,
        maximum_ecef_difference_m=args.maximum_ecef_difference_m,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
