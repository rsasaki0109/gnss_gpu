#!/usr/bin/env python3
"""Fail-closed WP29 CPU/GPU/audit/fast parity and runtime gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trajectory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trajectory_parity(
    reference: list[dict[str, str]], candidate: list[dict[str, str]]
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise RuntimeError("trajectory row counts differ")
    fix_mismatches = 0
    max_position_delta_m = 0.0
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if float(left["tow"]) != float(right["tow"]):
            raise RuntimeError(f"trajectory TOW mismatch at row {index}")
        fix_mismatches += int(left["fix"] != right["fix"])
        left_position = np.asarray(
            [float(left[key]) for key in ("ecef_x", "ecef_y", "ecef_z")]
        )
        right_position = np.asarray(
            [float(right[key]) for key in ("ecef_x", "ecef_y", "ecef_z")]
        )
        max_position_delta_m = max(
            max_position_delta_m,
            float(np.linalg.norm(left_position - right_position)),
        )
    return {
        "rows": len(reference),
        "fix_state_mismatches": fix_mismatches,
        "max_position_delta_m": max_position_delta_m,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--cpu-trajectory", type=Path, required=True)
    parser.add_argument("--gpu-audit-summary", type=Path, required=True)
    parser.add_argument("--gpu-audit-trajectory", type=Path, required=True)
    parser.add_argument("--gpu-fast-summary", type=Path, required=True)
    parser.add_argument("--gpu-fast-trajectory", type=Path, required=True)
    parser.add_argument("--position-tolerance-m", type=float, default=1e-7)
    parser.add_argument("--min-total-epochs-per-second", type=float, default=5.0)
    parser.add_argument("--max-steady-memory-growth-mib", type=float, default=4.0)
    parser.add_argument("--overnight-hours", type=float, default=12.0)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    paths = (
        args.cpu_summary,
        args.cpu_trajectory,
        args.gpu_audit_summary,
        args.gpu_audit_trajectory,
        args.gpu_fast_summary,
        args.gpu_fast_trajectory,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    cpu_summary = _load_json(args.cpu_summary)
    audit_summary = _load_json(args.gpu_audit_summary)
    fast_summary = _load_json(args.gpu_fast_summary)
    cpu_trajectory = _load_trajectory(args.cpu_trajectory)
    audit_trajectory = _load_trajectory(args.gpu_audit_trajectory)
    fast_trajectory = _load_trajectory(args.gpu_fast_trajectory)

    cpu_gpu = _trajectory_parity(cpu_trajectory, audit_trajectory)
    audit_fast = _trajectory_parity(audit_trajectory, fast_trajectory)
    tolerance = float(args.position_tolerance_m)
    parity_pass = all(
        result["fix_state_mismatches"] == 0
        and result["max_position_delta_m"] <= tolerance
        for result in (cpu_gpu, audit_fast)
    )
    core_fields = (
        "n_epochs",
        "basin_oracle_sub50cm_epochs",
        "sub50cm_all_epochs",
        "declared_fix_epochs",
        "false_fix_epochs",
        "evidence_records",
        "commit_replay_mismatches",
    )
    summary_mismatches = {
        key: [cpu_summary.get(key), audit_summary.get(key), fast_summary.get(key)]
        for key in core_fields
        if not (
            cpu_summary.get(key) == audit_summary.get(key)
            and audit_summary.get(key) == fast_summary.get(key)
        )
    }

    max_growth_bytes = int(float(args.max_steady_memory_growth_mib) * 1024**2)
    peak_over_start = int(fast_summary["rss_peak_bytes"]) - int(
        fast_summary["rss_start_bytes"]
    )
    memory_pass = (
        int(fast_summary["rss_growth_bytes"]) <= max_growth_bytes
        and int(fast_summary["rss_last_quarter_growth_bytes"]) <= max_growth_bytes
        and peak_over_start <= max_growth_bytes
    )
    throughput = float(fast_summary["runtime_total_epochs_per_second"])
    runtime_pass = throughput >= float(args.min_total_epochs_per_second)
    p99 = float(fast_summary["epoch_compute_p99_seconds"])
    latency_measured = math.isfinite(p99) and p99 >= 0.0
    projected_six_run_hours = 6.0 * float(fast_summary["runtime_total_seconds"]) / 3600.0
    overnight_pass = projected_six_run_hours <= float(args.overnight_hours)
    engine_pass = (
        audit_summary.get("lambda_engine") == "gpu-batch"
        and fast_summary.get("lambda_engine") == "gpu-batch"
        and fast_summary.get("runtime_mode") == "fast"
        and int(fast_summary.get("lambda_batch_problems", 0)) > 1
    )
    status = "pass" if all(
        (
            parity_pass,
            not summary_mismatches,
            memory_pass,
            runtime_pass,
            latency_measured,
            overnight_pass,
            engine_pass,
        )
    ) else "fail"
    report = {
        "gate": "WP29_GPU_SCALE",
        "status": status,
        "position_tolerance_m": tolerance,
        "cpu_vs_gpu_audit": cpu_gpu,
        "gpu_audit_vs_fast": audit_fast,
        "summary_mismatches": summary_mismatches,
        "gpu_batch_problems": int(fast_summary["lambda_batch_problems"]),
        "gpu_batch_calls": int(fast_summary["lambda_batch_calls"]),
        "runtime_epoch_loop_seconds": float(fast_summary["runtime_seconds"]),
        "runtime_total_seconds": float(fast_summary["runtime_total_seconds"]),
        "runtime_total_epochs_per_second": throughput,
        "min_total_epochs_per_second": float(args.min_total_epochs_per_second),
        "epoch_compute_p99_seconds": p99,
        "rss_start_bytes": int(fast_summary["rss_start_bytes"]),
        "rss_peak_bytes": int(fast_summary["rss_peak_bytes"]),
        "rss_end_bytes": int(fast_summary["rss_end_bytes"]),
        "rss_peak_over_start_bytes": peak_over_start,
        "rss_growth_bytes": int(fast_summary["rss_growth_bytes"]),
        "rss_last_quarter_growth_bytes": int(
            fast_summary["rss_last_quarter_growth_bytes"]
        ),
        "max_steady_memory_growth_bytes": max_growth_bytes,
        "projected_six_run_hours": projected_six_run_hours,
        "overnight_hours": float(args.overnight_hours),
        "checks": {
            "engine": engine_pass,
            "parity": parity_pass and not summary_mismatches,
            "runtime": runtime_pass,
            "p99_latency_measured": latency_measured,
            "steady_memory": memory_pass,
            "six_run_overnight_projection": overnight_pass,
        },
        "artifact_sha256": {str(path): _sha256(path) for path in paths},
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if status != "pass":
        raise RuntimeError("WP29 GPU scale gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
