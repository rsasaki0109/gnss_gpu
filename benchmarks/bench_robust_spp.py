"""Compare CPU robust SPP with a repository revision on identical observations.

Run from any directory: python benchmarks/bench_robust_spp.py --reference-ref REF.
Timings cover one complete SPP call, not the GNSS/IMU estimator or GPU pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "python/gnss_gpu/robust_spp.py"


def _solver(source: bytes, label: str):
    namespace = {"__name__": label}
    exec(compile(source, label, "exec"), namespace)
    return namespace["robust_spp"]


def _scenario(n_sat: int, seed: int, case: str):
    rng = np.random.default_rng(seed)
    position = np.array([-3959340.0, 3352854.0, 3697471.0])
    azimuth = np.linspace(0, 2 * np.pi, n_sat, endpoint=False)
    elevation = rng.uniform(np.radians(15), np.radians(60), n_sat)
    satellites = position + 26_600_000.0 * np.column_stack([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    observations = np.linalg.norm(satellites - position, axis=1) + 1e6
    observations += rng.normal(0, 1, n_sat)
    weights = rng.uniform(0.3, 2, n_sat)
    if case != "clean":
        observations[1] += 200.0
        weights[0] = 0.0
    return dict(sat_ecef=satellites, pseudoranges=observations, weights=weights,
                init_pos=position + np.array([5.0, -10.0, 20.0]),
                weight_func="huber" if case == "huber_outlier" else "cauchy")


def _measure(solver, inputs, calls: int):
    start = time.perf_counter_ns()
    for _ in range(calls):
        solver(**inputs)
    return (time.perf_counter_ns() - start) / calls / 1000.0


def benchmark(reference_ref: str, satellites: list[int], rounds: int, calls: int):
    if rounds < 2 or calls < 1 or not satellites or any(n < 6 for n in satellites):
        raise ValueError("At least two rounds, positive calls and satellite counts >=6 required")
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", reference_ref + "^{commit}"], cwd=ROOT,
    ).decode().strip()
    reference_source = subprocess.check_output(["git", "show", baseline_commit + ":" + SOURCE], cwd=ROOT)
    current_source = (ROOT / SOURCE).read_bytes()
    if reference_source == current_source:
        raise ValueError("Reference and current solver sources are identical; choose a prior revision")
    solvers = {"reference": _solver(reference_source, "reference_spp"),
               "current": _solver(current_source, "current_spp")}
    result = dict(scope="CPU robust_spp calls on synthetic observations; no whole estimator or GPU claim",
                  python_version=sys.version.split()[0], numpy_version=np.__version__,
                  baseline_commit=baseline_commit,
                  reference_source_sha256=hashlib.sha256(reference_source).hexdigest(),
                  current_source_sha256=hashlib.sha256(current_source).hexdigest(),
                  rounds=rounds, calls_per_round=calls,
                  host_isolation_established=False, position_tolerance_m=1e-8, cases=[])
    for n_sat in satellites:
        for case in ("clean", "cauchy_outlier", "huber_outlier"):
            inputs = _scenario(n_sat, n_sat, case)
            originals = {k: v.copy() for k, v in inputs.items() if isinstance(v, np.ndarray)}
            expected = solvers["reference"](**inputs)
            actual = solvers["current"](**inputs)
            assert expected is not None and actual is not None, (n_sat, case, "solution failed")
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-8)
            error = float(np.max(np.abs(actual - expected)))
            samples = {name: [] for name in solvers}
            for _ in range(10):
                for solver in solvers.values():
                    solver(**inputs)
            for repetition in range(rounds):
                order = ("reference", "current") if repetition % 2 == 0 else ("current", "reference")
                for name in order:
                    samples[name].append(_measure(solvers[name], inputs, calls))
            for key, value in originals.items():
                np.testing.assert_array_equal(inputs[key], value)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            record = dict(satellites=n_sat, case=case, per_call_us=samples, median_us=medians,
                          speedup=medians["reference"] / medians["current"],
                          max_position_difference_m=error, position_bit_identical=bool(np.array_equal(actual, expected)),
                          inputs_unchanged=True, removed_dense_weight_matrix_bytes=n_sat * n_sat * 8)
            result["cases"].append(record)
            print(f"{n_sat:3d} {case:15s} reference={medians['reference']:.2f} us "
                  f"current={medians['current']:.2f} us {record['speedup']:.2f}x delta={error:.3g} m", flush=True)
    assert (ROOT / SOURCE).read_bytes() == current_source
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", default="HEAD")
    parser.add_argument("--satellites", nargs="+", type=int, default=[8, 16, 32, 64, 96])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; use a new file to preserve the previous measurement")
    result = benchmark(args.reference_ref, args.satellites, args.rounds, args.calls)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8", newline="\n") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")


if __name__ == "__main__":
    main()
