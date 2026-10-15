"""Compare actual PF builds, including public default steps and covariance fallback.

Archive the installed preceding binary and sources before rebuilding. Separate
workers avoid pybind type collisions; alternating build order limits drift.
Initialization/restoration and output checks are excluded from synchronized
timing. No test implementation replaces the production kernels.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from bench_pf_coalesced_compare import _initialize, _load_extension, _scenario, _validate


def _digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        result.update(np.ascontiguousarray(array).tobytes())
    return result.hexdigest()


def _snapshot(native, state):
    particles = native.pf_device_get_particle_states(state)
    weights = np.empty(len(particles))
    native.pf_device_get_log_weights(state, weights)
    return particles, weights


def _extra_validation(native):
    records = {}
    for count in (1, 2, 3, 31, 32, 33, 255, 256, 257, 8193, 100_000):
        state = native.pf_device_create(count)
        try:
            rng = np.random.default_rng(count)
            original = rng.normal(size=(count, 16))
            original[:, 0] = np.arange(count)
            original[::17, 8] = -0.0
            distributions = {
                "tiny": rng.normal(0., 1e-7, count),
                "shifted": rng.normal(size=count) - 1e6,
                "nonfinite": np.resize([0., np.inf, -np.inf, np.nan], count),
                "extreme": np.resize([0., 1e300, -1e300, np.nextafter(0., 1.)], count),
            }
            for iterations in (2, 3, 4, 5, 6, 7, 8, 9):
                for name, weights in distributions.items():
                    native.pf_device_set_particle_states(state, original)
                    native.pf_device_set_log_weights(state, weights)
                    native.pf_device_resample_megopolis_coalesced(state, iterations, 2**64 - 1)
                    particles, reset_weights = _snapshot(native, state)
                    indices = particles[:, 0].astype(int)
                    np.testing.assert_array_equal(particles.view(np.uint64), original[indices].view(np.uint64))
                    np.testing.assert_array_equal(reset_weights, np.zeros(count))
                    records[f"N{count}/B{iterations}/{name}"] = _digest(particles, reset_weights)
            # Compare a proven-common covariance against forcing the full-copy
            # path via an import of the exact same state. No float comparison.
            for velocity_kf in (False, True):
                histories = []
                for imported in (False, True):
                    _initialize(native, state)
                    if imported:
                        native.pf_device_set_particle_states(state, native.pf_device_get_particle_states(state))
                    history = []
                    for step in range(1, 4):
                        native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42,
                                                step, 0., .3, velocity_kf, .7)
                        native.pf_device_set_log_weights(state, np.linspace(-4., 0., count))
                        native.pf_device_resample_megopolis_coalesced(state, 5 + step, 42)
                        history.extend(_snapshot(native, state))
                        history.append(native.pf_device_estimate(state))
                    histories.append(_digest(*history))
                assert histories[0] == histories[1], "Common covariance differs from full-copy path"
                records[f"N{count}/common_vs_import/KF{velocity_kf}"] = histories[0]
            _initialize(native, state)
            _, satellites, _, weights_sat = _scenario()
            native.pf_device_resample_megopolis_coalesced(state, 5, 42)
            native.pf_device_doppler_kf_update(state, satellites, np.zeros_like(satellites),
                                              np.zeros(12), weights_sat, 12, .19, .2)
            native.pf_device_set_log_weights(state, np.linspace(-4., 0., count))
            native.pf_device_resample_megopolis_coalesced(state, 7, 42)
            records[f"N{count}/Doppler_KF_invalidation"] = _digest(*_snapshot(native, state))
        finally:
            native.pf_device_destroy(state)
    return records


def _measure(native, state, prepare, operation, repeats):
    samples = []
    output = None
    for repetition in range(repeats + 3):
        prepare()
        native.pf_device_sync(state)
        start = time.perf_counter_ns()
        output = operation()
        native.pf_device_sync(state)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if repetition >= 3:
            samples.append(elapsed)
    arrays = list(_snapshot(native, state))
    if output is not None:
        arrays.extend(output if isinstance(output, tuple) else (output,))
    return {"median_ms": float(np.median(samples)), "samples_ms": samples,
            "output_sha256": _digest(*arrays)}


def _profile(native, sizes, repeats):
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    measurements = {}
    position, satellites, ranges, weights_sat = _scenario()
    satellite_velocities = np.zeros_like(satellites)
    doppler = np.zeros(len(satellites))
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            _initialize(native, state)
            imported = native.pf_device_get_particle_states(state)
            covariance = np.arange(count, dtype=float) + 1.
            imported[:, 7] = covariance
            imported[:, 11] = covariance * 2.
            imported[:, 15] = covariance * 3.
            log_weights = np.random.default_rng(0).normal(0., 1.5, count)

            def prepare_resample(common):
                if common:
                    _initialize(native, state)
                else:
                    native.pf_device_set_particle_states(state, imported)
                native.pf_device_set_log_weights(state, log_weights)

            def predict():
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1)

            def weight():
                native.pf_device_weight(state, satellites, ranges, weights_sat, 12, 5.)

            def prepare_weighted():
                _initialize(native, state)
                predict()
                weight()

            def resample(iterations=60):
                native.pf_device_resample_megopolis_coalesced(state, iterations, 42)

            def prepare_resampled():
                prepare_weighted()
                resample()

            def step(mode="default"):
                predict()
                if mode == "doppler_kf":
                    native.pf_device_doppler_kf_update(state, satellites, satellite_velocities,
                                                      doppler, weights_sat, 12, .19, .2)
                weight()
                ess = native.pf_device_ess(state)
                did_resample = ess < .5 * count
                if did_resample:
                    if mode == "systematic":
                        native.pf_device_resample_systematic(state, 42)
                    else:
                        resample()
                return native.pf_device_estimate(state), np.array([ess, did_resample])

            for common in (True, False):
                for iterations in (0, 60, 240):
                    name = f"resample_normal_B{iterations}/{'common' if common else 'imported'}_cov"
                    measurements[f"N{count}/{name}"] = _measure(native, state,
                        lambda common=common: prepare_resample(common),
                        lambda iterations=iterations: resample(iterations), repeats)
            stages = {
                "predict_control": (lambda: _initialize(native, state), predict),
                "weight_control": (lambda: _initialize(native, state), weight),
                "weighted_ess_control": (prepare_weighted, lambda: native.pf_device_ess(state)),
                "weighted_resample_B60": (prepare_weighted, resample),
                "resampled_estimate_control": (prepare_resampled, lambda: native.pf_device_estimate(state)),
                "native_default_step": (lambda: _initialize(native, state), step),
                "native_doppler_kf_step": (lambda: _initialize(native, state), lambda: step("doppler_kf")),
                "native_systematic_control_step": (lambda: _initialize(native, state), lambda: step("systematic")),
            }
            for name, (prepare, operation) in stages.items():
                measurements[f"N{count}/{name}"] = _measure(native, state, prepare, operation, repeats)
        finally:
            native.pf_device_destroy(state)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare_wrapper():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.)

            def wrapper_step():
                pf.predict([1., 2., 3.], dt=.1)
                pf.update(satellites, ranges, weights_sat)
                return pf.estimate()

            measurements[f"N{count}/public_default_predict_update_estimate"] = _measure(
                native, pf._state, prepare_wrapper, wrapper_step, repeats)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    validation = {}
    if args.verify:
        validation = _validate(native)
        validation.update(_extra_validation(native))
    return {"validation": validation, "measurements": _profile(native, args.sizes, args.repeats)}


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "benchmarks/pf_systematic_gpu_20261011.json").read_text())
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    extension_hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()}
    source_paths = ["src/particle_filter/pf_device.cu", "include/gnss_gpu/pf_device.h",
                    "python/gnss_gpu/_pf_device_bindings.cpp", "python/gnss_gpu/pf_device_runtime.py",
                    "python/gnss_gpu/pf_device_config.py"]
    baseline_hashes = {path: hashlib.sha256((args.baseline_dir / Path(path).name).read_bytes()).hexdigest()
                       for path in source_paths}
    assert baseline_hashes == reference["source_sha256"]["candidate"], "Wrong preceding sources"
    assert extension_hashes["baseline"] == reference["extension_sha256"]["candidate"], "Wrong preceding binary"
    runs = {"baseline": [], "candidate": []}
    for round_index in range(args.rounds):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for label in order:
            command = [sys.executable, str(Path(__file__).resolve()), "--extension", str(paths[label].resolve()),
                       "--sizes", *map(str, args.sizes), "--repeats", str(args.repeats)]
            if round_index == 0:
                command.append("--verify")
            worker = subprocess.run(command, capture_output=True, text=True)
            if worker.returncode:
                raise RuntimeError(f"{label} worker failed:\n{worker.stderr}")
            result = json.loads(worker.stdout)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / f"{args.output.stem}_{label}_round{round_index + 1}.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")
            runs[label].append(result)
            print(f"round {round_index + 1}/{args.rounds}: {label} complete", flush=True)
        if round_index == 0:
            before, after = [runs[label][0]["validation"] for label in ("baseline", "candidate")]
            mismatches = [key for key in before if before[key] != after.get(key)]
            assert before.keys() == after.keys() and not mismatches, f"Seeded output mismatch: {mismatches[:10]}"
    summary = {}
    for key in runs["baseline"][0]["measurements"]:
        outputs = {run["measurements"][key]["output_sha256"] for data in runs.values() for run in data}
        assert len(outputs) == 1, f"Timed outputs differ: {key}"
        medians = {label: [run["measurements"][key]["median_ms"] for run in data] for label, data in runs.items()}
        baseline, candidate = [float(np.median(medians[label])) for label in ("baseline", "candidate")]
        summary[key] = {"baseline_ms": baseline, "candidate_ms": candidate,
                        "speedup": baseline / candidate, "round_medians_ms": medians,
                        "output_sha256": outputs.pop()}
    return {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "method": "separate actual-build workers; alternating order; synchronized wall time; non-exclusive host",
        "scope": "adaptive native and actual public default step; state restoration and output checks excluded",
        "rounds": args.rounds, "repeats": args.repeats, "warmup_calls_per_stage": 3,
        "source_sha256": {"baseline": baseline_hashes,
                          "candidate": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths}},
        "extension_sha256": extension_hashes,
        "benchmark_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in
                             ("benchmarks/bench_pf_covariance_compare.py", "benchmarks/bench_pf_coalesced_compare.py")},
        "validation": {"cases": len(before), "all_state_outputs_bit_identical": True,
                       "baseline_sha256_by_case": before, "candidate_sha256_by_case": after},
        "covariance_copy_saving": {"bytes_per_particle_per_common_covariance_gather": 144,
                                    "additional_device_allocations": 0},
        "summary": summary,
        "runs": {label: [run["measurements"] for run in data] for label, data in runs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 10_000, 100_000, 1_000_000])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.rounds < 1 or args.repeats < 1:
        parser.error("sizes, rounds, repeats must be positive")
    if not args.extension and not (args.baseline and args.candidate and args.baseline_dir and args.output):
        parser.error("comparison requires baseline, candidate, baseline-dir and output")
    report = _worker(args) if args.extension else _compare(args)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
