#!/usr/bin/env python3
"""Benchmark persistent-stream PF epochs against Phase 4 realtime budgets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from gnss_gpu import ParticleFilterDevice
from gnss_gpu.evaluation_contract import M4_PRESERVED_SHA256, sha256_file, write_json
from gnss_gpu.realtime_runtime import (
    RealtimeBudget,
    RealtimeMonitor,
    RuntimeMode,
    estimate_pf_device_memory_mb,
)
from gnss_gpu.realtime_batch import BatchWorkspaceCapacity, CudaBatchWorkspace


TRUE_POS = np.array([-3_947_484.0, 3_366_824.0, 3_699_140.0])


def _satellites() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = np.asarray(
        [
            [0.8, 0.1, 0.6],
            [-0.4, 0.7, 0.6],
            [0.1, -0.9, 0.5],
            [-0.7, -0.3, 0.6],
            [0.5, 0.7, 0.5],
            [-0.6, 0.5, 0.7],
            [0.3, -0.4, 0.85],
            [-0.2, -0.6, 0.77],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sat = TRUE_POS + directions * 20_200_000.0
    pseudorange = np.linalg.norm(sat - TRUE_POS, axis=1) + 100.0
    return sat, pseudorange, np.ones(len(sat))


def _gpu_process_memory_mb() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    current = str(os.getpid())
    values = []
    for line in completed.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 2 and fields[0] == current:
            try:
                values.append(float(fields[1]))
            except ValueError:
                continue
    return max(values) if values else None


def _gpu_name() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.splitlines()[0].strip() if completed.stdout else None


def _run_mode(
    monitor: RealtimeMonitor,
    *,
    mode: RuntimeMode,
    particles: int,
    iterations: int,
    epoch_offset: int,
    batch_workspace: CudaBatchWorkspace | None = None,
) -> str:
    sat, pseudorange, weights = _satellites()
    pf = ParticleFilterDevice(
        n_particles=particles,
        sigma_pos=1.0,
        sigma_cb=300.0,
        sigma_pr=5.0,
        resampling="megopolis",
        seed=42,
    )
    pf.initialize(TRUE_POS, clock_bias=100.0, spread_pos=20.0, spread_cb=100.0)
    for _ in range(3):
        pf.predict(dt=1.0)
        pf.update(sat, pseudorange, weights)
    pf.sync()
    measured_memory = _gpu_process_memory_mb()
    memory_source = (
        "nvidia-smi_process_used_memory"
        if measured_memory is not None
        else "capacity_estimate_320_bytes_per_particle"
    )
    memory = (
        measured_memory
        if measured_memory is not None
        else estimate_pf_device_memory_mb(particles)
    )
    for index in range(iterations):
        def synchronize_epoch() -> None:
            pf.sync()
            if batch_workspace is not None:
                batch_workspace.synchronize()

        with monitor.epoch(
            epoch=epoch_offset + index,
            mode=mode,
            particle_count=particles,
            gpu_memory_mb=memory,
            synchronize=synchronize_epoch,
        ) as timer:
            with timer.measure("predict"):
                pf.predict(dt=1.0)
            with timer.measure("weight_resample"):
                pf.update(sat, pseudorange, weights)
            if batch_workspace is not None:
                with timer.measure("arc_screen_batch"):
                    batch_workspace.arc_outlier_fraction(
                        batch_workspace.benchmark_screen_residuals,
                        batch_workspace.benchmark_screen_valid,
                        edge_m=5.0,
                    )
                with timer.measure("candidate_score_batch"):
                    batch_workspace.candidate_rms(
                        batch_workspace.benchmark_candidate_residuals,
                        batch_workspace.benchmark_candidate_weights,
                    )
                with timer.measure("affine_refit_batch"):
                    batch_workspace.affine_refit(
                        batch_workspace.benchmark_epochs,
                        batch_workspace.benchmark_offsets,
                        batch_workspace.benchmark_affine_weights,
                    )
    return memory_source


def _benchmark_workspace() -> CudaBatchWorkspace:
    workspace = CudaBatchWorkspace(
        BatchWorkspaceCapacity(
            candidates=256,
            observations=64,
            epochs=16,
            satellites=32,
        )
    )
    rng = np.random.default_rng(42)
    workspace.benchmark_candidate_residuals = rng.normal(size=(256, 64))
    workspace.benchmark_candidate_weights = rng.uniform(0.2, 1.0, size=(256, 64))
    workspace.benchmark_epochs = np.arange(16, dtype=np.float64)
    slopes = rng.normal(scale=0.1, size=(256, 1, 3))
    intercepts = rng.normal(size=(256, 1, 3))
    workspace.benchmark_offsets = (
        intercepts
        + slopes * workspace.benchmark_epochs[np.newaxis, :, np.newaxis]
        + rng.normal(scale=0.05, size=(256, 16, 3))
    )
    workspace.benchmark_affine_weights = np.ones((256, 16))
    workspace.benchmark_screen_residuals = rng.normal(size=(16, 32))
    workspace.benchmark_screen_residuals[:, -2:] += 20.0
    workspace.benchmark_screen_valid = np.ones((16, 32), dtype=bool)
    # Compile kernels and populate persistent allocations outside the timed run.
    workspace.arc_outlier_fraction(
        workspace.benchmark_screen_residuals,
        workspace.benchmark_screen_valid,
        edge_m=5.0,
    )
    workspace.candidate_rms(
        workspace.benchmark_candidate_residuals,
        workspace.benchmark_candidate_weights,
    )
    workspace.affine_refit(
        workspace.benchmark_epochs,
        workspace.benchmark_offsets,
        workspace.benchmark_affine_weights,
    )
    workspace.synchronize()
    return workspace


def benchmark(
    repo_root: Path,
    *,
    normal_particles: int,
    search_particles: int,
    iterations: int,
) -> dict:
    monitor = RealtimeMonitor(RealtimeBudget())
    started = time.perf_counter()
    normal_memory_source = _run_mode(
        monitor,
        mode=RuntimeMode.NORMAL,
        particles=normal_particles,
        iterations=iterations,
        epoch_offset=0,
    )
    workspace = _benchmark_workspace()
    search_memory_source = _run_mode(
        monitor,
        mode=RuntimeMode.SEARCH,
        particles=search_particles,
        iterations=max(3, iterations // 2),
        epoch_offset=iterations,
        batch_workspace=workspace,
    )
    assessment = monitor.assess()
    m4 = {
        path: {
            "expected_sha256": expected,
            "actual_sha256": sha256_file(repo_root / path),
        }
        for path, expected in M4_PRESERVED_SHA256.items()
    }
    passed = assessment.passed and all(
        value["expected_sha256"] == value["actual_sha256"] for value in m4.values()
    )
    return {
        "schema": "gnss_gpu_phase4_realtime_benchmark_v1",
        "backend": "ParticleFilterDevice persistent CUDA stream",
        "gpu": _gpu_name(),
        "normal_particles": normal_particles,
        "search_particles": search_particles,
        "search_batches": {
            "arc_screen": [16, 32],
            "candidate_score": [256, 64],
            "affine_refit": [256, 16, 3],
            "backend": "Numba CUDA persistent workspace",
        },
        "memory_accounting": {
            "normal": normal_memory_source,
            "search": search_memory_source,
        },
        "iterations": iterations,
        "wall_time_s": time.perf_counter() - started,
        "assessment": {
            "passed": assessment.passed,
            "sample_count": assessment.sample_count,
            "latency_p50_ms": assessment.latency_p50_ms,
            "latency_p95_ms": assessment.latency_p95_ms,
            "normal_latency_max_ms": assessment.normal_latency_max_ms,
            "search_latency_max_ms": assessment.search_latency_max_ms,
            "peak_gpu_memory_mb": assessment.peak_gpu_memory_mb,
            "deadline_misses": assessment.deadline_misses,
            "reasons": list(assessment.reasons),
        },
        "m4": m4,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--normal-particles", type=int, default=100_000)
    parser.add_argument("--search-particles", type=int, default=500_000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = benchmark(
        args.repo_root.resolve(),
        normal_particles=args.normal_particles,
        search_particles=args.search_particles,
        iterations=args.iterations,
    )
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
