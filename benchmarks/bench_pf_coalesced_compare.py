"""Compare two CUDA PF builds in separate processes, with identical inputs.

Build/copy the baseline extension before editing pf_device.cu, rebuild the
candidate with the same configuration, then pass both extension paths. Every
timed operation ends in a stream synchronization. Restoring weights/state is
outside the measured interval; validation covers all 16 state fields.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np


def _load_extension(path):
    handles = []
    if sys.platform == "win32":
        directories = [path.parent]
        for variable in ("CUDA_PATH", "CUDA_HOME"):
            if root := os.environ.get(variable):
                directories.append(Path(root) / "bin")
        handles = [os.add_dll_directory(str(directory)) for directory in directories
                   if directory.is_dir()]
    spec = importlib.util.spec_from_file_location("_gnss_gpu_pf_device", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._dll_directory_handles = handles
    return module


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    return digest.hexdigest()


def _scenario():
    position = np.array([-3957199., 3310205., 3737911.])
    rng = np.random.default_rng(7)
    direction = rng.normal(size=(12, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    satellites = 26_560_000. * direction
    ranges = np.linalg.norm(satellites - position, axis=1) + 3000.
    return position, satellites, ranges, np.ones(12)


def _initialize(native, state):
    position, _, _, _ = _scenario()
    native.pf_device_initialize(state, *position, 3000., 10., 30., 42,
                                1., 2., 3., 0.5, 2.)
    native.pf_device_sync(state)


def _operations(native, state):
    _, satellites, ranges, weights = _scenario()

    def predict():
        native.pf_device_predict(state, 1., 2., 3., 0.1, 1., 3., 42, 1)

    def weight():
        native.pf_device_weight(state, satellites, ranges, weights, 12, 5.)

    def resample(iterations=60):
        native.pf_device_resample_megopolis_coalesced(state, iterations, 42)

    def pipeline():
        predict()
        weight()
        native.pf_device_ess(state)
        resample()
        native.pf_device_estimate(state)

    return predict, weight, resample, pipeline


def _measure(native, state, prepare, operation, repeats):
    samples = []
    for repetition in range(repeats + 3):
        prepare()
        native.pf_device_sync(state)
        start = time.perf_counter_ns()
        operation()
        native.pf_device_sync(state)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if repetition >= 3:
            samples.append(elapsed)
    return {"median_ms": float(np.median(samples)), "samples_ms": samples}


def _profile(native, sizes, repeats):
    measurements = {}
    for count in sizes:
        state = native.pf_device_create(count)
        _initialize(native, state)
        predict, weight, resample, pipeline = _operations(native, state)
        log_weights = np.random.default_rng(0).normal(0., 1.5, count)

        def restore_weights():
            native.pf_device_set_log_weights(state, log_weights)

        def restore_state():
            _initialize(native, state)

        stages = {
            "predict": (restore_state, predict),
            "weight_12_sat": (restore_state, weight),
            "ess": (restore_weights, lambda: native.pf_device_ess(state)),
            "estimate": (restore_weights, lambda: native.pf_device_estimate(state)),
            "resample_B60": (restore_weights, resample),
            "resample_B240": (restore_weights, lambda: resample(240)),
            "native_predict_weight_ess_resample_estimate": (restore_state, pipeline),
        }
        for name, (prepare, operation) in stages.items():
            measurements[f"N{count}/{name}"] = _measure(
                native, state, prepare, operation, repeats
            )
        native.pf_device_destroy(state)
    return measurements


def _validate(native):
    cases = {}
    # Partial warps/blocks and small/large clouds with different chain lengths.
    small_sizes = (1, 2, 31, 32, 33, 63, 65, 255, 256, 257, 1000, 2048,
                   8191, 8192, 8193, 16384, 16385)
    for count in (*small_sizes, 100_000, 1_000_000):
        state = native.pf_device_create(count)
        rng = np.random.default_rng(count)
        original = rng.normal(size=(count, 16))
        original[:, 0] = np.arange(count)
        original[::17, 4] = -0.0
        distributions = {
            "uniform": np.zeros(count),
            "normal": rng.normal(0., 1.5, count),
            "dominant": np.full(count, -1000.),
        }
        distributions["dominant"][count // 2] = 0.
        iterations = (0, 1, 15, 60, 240, 257) if count in small_sizes else (60,)
        seeds = (0, 1, 42, 2**64 - 1) if count in small_sizes else (1, 42)
        for steps in iterations:
            for name, log_weights in distributions.items():
                for seed in seeds:
                    native.pf_device_set_particle_states(state, original)
                    native.pf_device_set_log_weights(state, log_weights)
                    native.pf_device_resample_megopolis_coalesced(state, steps, seed)
                    actual = native.pf_device_get_particle_states(state)
                    result_weights = np.empty(count)
                    native.pf_device_get_log_weights(state, result_weights)
                    indices = actual[:, 0].astype(np.int64)
                    np.testing.assert_array_equal(actual, original[indices])
                    np.testing.assert_array_equal(result_weights, np.zeros(count))
                    key = f"N{count}/B{steps}/{name}/seed{seed}"
                    cases[key] = _digest(actual, result_weights)
        # Include seeded prediction/likelihood and repeated gather pointer swaps.
        _initialize(native, state)
        _, _, _, pipeline = _operations(native, state)
        estimates = []
        for _ in range(3):
            pipeline()
            estimates.append(native.pf_device_estimate(state))
        cases[f"N{count}/pipeline_3_epochs"] = _digest(
            native.pf_device_get_particle_states(state), estimates
        )
        native.pf_device_set_particle_states(state, original)
        native.pf_device_set_log_weights(state, distributions["normal"])
        native.pf_device_resample_megopolis_coalesced(state, 60, 42)
        native.pf_device_set_log_weights(state, distributions["normal"])
        native.pf_device_resample_megopolis(state, 15, 1)
        native.pf_device_set_log_weights(state, distributions["normal"])
        native.pf_device_resample_systematic(state, 1)
        native.pf_device_set_log_weights(state, distributions["normal"])
        native.pf_device_resample_megopolis_coalesced(state, 60, 42)
        cases[f"N{count}/mixed_resamplers"] = _digest(native.pf_device_get_particle_states(state))
        native.pf_device_destroy(state)
    return cases


def _worker(args):
    native = _load_extension(args.extension.resolve())
    return {
        "validation": _validate(native) if args.verify else {},
        "measurements": _profile(native, args.sizes, args.repeats),
    }


def _compare(args):
    runs = {"baseline": [], "candidate": []}
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    for round_index in range(args.rounds):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for label in order:
            command = [sys.executable, str(Path(__file__).resolve()), "--extension",
                       str(paths[label].resolve()), "--repeats", str(args.repeats),
                       "--sizes", *map(str, args.sizes)]
            if round_index == 0:
                command.append("--verify")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{label} worker failed:\n{result.stderr}")
            runs[label].append(json.loads(result.stdout))
            print(f"round {round_index + 1}/{args.rounds}: {label} complete", flush=True)
        if round_index == 0:
            before = runs["baseline"][0]["validation"]
            after = runs["candidate"][0]["validation"]
            if before != after:
                mismatches = [key for key in before if before[key] != after.get(key)]
                raise RuntimeError(f"Seeded output mismatch: {mismatches[:10]}")
    summary = {}
    for key in runs["baseline"][0]["measurements"]:
        medians = {label: [run["measurements"][key]["median_ms"] for run in data]
                   for label, data in runs.items()}
        baseline = float(np.median(medians["baseline"]))
        candidate = float(np.median(medians["candidate"]))
        summary[key] = {"baseline_ms": baseline, "candidate_ms": candidate,
                        "speedup": baseline / candidate, "round_medians_ms": medians}
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": sys.platform,
        "build_configuration": args.build_configuration,
        "method": "separate processes, alternating build order, synchronized wall time",
        "scope": "native calls; pipeline unconditionally resamples with B=60; state restoration excluded",
        "rounds": args.rounds, "repeats": args.repeats,
        "warmup_calls_per_stage": 3,
        "source_sha256": {
            "baseline": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest()
                        if args.baseline_source else None,
            "candidate": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "src/particle_filter/pf_device.cu").read_bytes()
            ).hexdigest(),
        },
        "extension_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                             for label, path in paths.items()},
        "validation": {"cases": len(before), "bit_identical": True,
                       "baseline_sha256_by_case": before, "candidate_sha256_by_case": after},
        "summary": summary,
        "runs": {label: [run["measurements"] for run in data] for label, data in runs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--extension", type=Path, help="Run one isolated worker")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 10_000, 100_000, 1_000_000])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or min(args.sizes) < 1:
        parser.error("rounds, repeats, and sizes must be positive")
    if not args.extension and not (args.baseline and args.candidate and args.output):
        parser.error("comparison requires --baseline, --candidate, and --output")
    report = _worker(args) if args.extension else _compare(args)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
