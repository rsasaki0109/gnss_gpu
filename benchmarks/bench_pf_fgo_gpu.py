#!/usr/bin/env python3
"""Focused PF/FGO GPU latency benchmark with machine-readable output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from gnss_gpu import ParticleFilterDevice
from gnss_gpu.fgo import fgo_gnss_lm


def _stats(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
    }


def _measure(fn, *, iterations: int, synchronize=None) -> dict[str, float]:
    for _ in range(3):
        fn()
        if synchronize is not None:
            synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return _stats(samples)


def _gpu_name() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return "unknown"


def benchmark_pf(n_particles: int, iterations: int) -> dict[str, object]:
    center = np.array([-3957199.0, 3310205.0, 3737911.0])
    pf = ParticleFilterDevice(n_particles=n_particles, seed=42)
    pf.initialize(center, clock_bias=3000.0)
    return {
        "particles": n_particles,
        "ess": _measure(
            lambda: pf._pf_device_ess(pf._state),
            iterations=iterations,
        ),
        "estimate": _measure(pf.estimate, iterations=iterations),
        "position_spread": _measure(
            lambda: pf.get_position_spread(center=center),
            iterations=iterations,
        ),
        "systematic_resample": _measure(
            lambda: pf._pf_device_resample_systematic(pf._state, 42),
            iterations=max(5, iterations // 2),
            synchronize=pf.sync,
        ),
    }


def _fgo_problem(n_epoch: int, n_sat: int = 12):
    rng = np.random.default_rng(20260731 + n_epoch)
    truth = np.array([-3957199.0, 3310205.0, 3737911.0])
    clock_bias = 3000.0
    satellites = rng.normal(size=(n_sat, 3))
    satellites /= np.linalg.norm(satellites, axis=1)[:, None]
    satellites *= 26_560_000.0
    sat = np.broadcast_to(satellites, (n_epoch, n_sat, 3)).copy()
    ranges = np.linalg.norm(sat - truth[None, None, :], axis=2)
    pseudorange = ranges + clock_bias + rng.normal(0.0, 1.0, ranges.shape)
    weights = np.ones_like(pseudorange)
    state = np.empty((n_epoch, 4), dtype=np.float64)
    state[:, :3] = truth + rng.normal(0.0, 20.0, (n_epoch, 3))
    state[:, 3] = clock_bias + rng.normal(0.0, 5.0, n_epoch)
    return sat, pseudorange, weights, state


def benchmark_fgo(n_epoch: int, iterations: int) -> dict[str, object]:
    sat, pseudorange, weights, initial = _fgo_problem(n_epoch)
    states: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float]] = {}
    previous = os.environ.get("GNSS_GPU_FGO_GPU_SOLVER")
    try:
        for label, enabled in (("cpu", "0"), ("gpu", "1")):
            os.environ["GNSS_GPU_FGO_GPU_SOLVER"] = enabled

            def solve() -> None:
                state = initial.copy()
                fgo_gnss_lm(
                    sat,
                    pseudorange,
                    weights,
                    state,
                    motion_sigma_m=2.0,
                    max_iter=1,
                    tol=0.0,
                    line_search=False,
                )
                states[label] = state

            timings[label] = _measure(solve, iterations=iterations)
    finally:
        if previous is None:
            os.environ.pop("GNSS_GPU_FGO_GPU_SOLVER", None)
        else:
            os.environ["GNSS_GPU_FGO_GPU_SOLVER"] = previous

    cpu_mean = timings["cpu"]["mean_ms"]
    gpu_mean = timings["gpu"]["mean_ms"]
    return {
        "epochs": n_epoch,
        "state_size": 4 * n_epoch,
        **timings,
        "gpu_speedup": cpu_mean / gpu_mean,
        "max_state_delta": float(np.max(np.abs(states["cpu"] - states["gpu"]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--pf-iterations", type=int, default=30)
    parser.add_argument("--fgo-epochs", type=int, nargs="+", default=[50, 200, 500])
    parser.add_argument("--fgo-iterations", type=int, default=3)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = {
        "schema": "gnss_gpu.pf_fgo_gpu_benchmark.v1",
        "gpu": _gpu_name(),
        "pf": benchmark_pf(args.particles, args.pf_iterations),
        "fgo": [
            benchmark_fgo(n_epoch, args.fgo_iterations)
            for n_epoch in args.fgo_epochs
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")


if __name__ == "__main__":
    main()
