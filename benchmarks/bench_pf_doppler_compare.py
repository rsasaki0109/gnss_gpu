"""Compare actual CUDA PF builds for Doppler KF and public combined updates.

Archive the preceding installed extension and five PF sources before editing.
Workers use actual production handles; restoration and output checks are outside
the synchronized timed interval. Build order alternates between rounds.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from bench_pf_coalesced_compare import _initialize, _load_extension, _scenario
from bench_pf_covariance_compare import _digest, _measure, _snapshot


def _observations(count):
    position, _, _, _ = _scenario()
    rng = np.random.default_rng(200 + count)
    direction = rng.normal(size=(count, 3))
    if count:
        direction /= np.linalg.norm(direction, axis=1)[:, None]
    satellites = direction * 26_560_000.
    velocities = rng.normal(0., 2000., (count, 3))
    los = satellites - position
    if count:
        los /= np.linalg.norm(los, axis=1)[:, None]
    range_rate = np.sum((velocities - np.array([1., 2., 3.])) * los, axis=1) + 5.
    doppler = -(range_rate + .2 * np.sin(np.arange(count))) / .19
    weights = rng.uniform(.1, 1., count)
    return satellites, velocities, doppler, weights


def _kf(native, state, observations, sigma=.5):
    sat, velocity, doppler, weights = observations
    native.pf_device_doppler_kf_update(state, sat, velocity, doppler, weights, len(weights), .19, sigma)


def _validate(native):
    records = {}
    sizes = (1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257, 2048, 8193)
    satellites = (0, 1, 2, 3, 4, 8, 12, 15, 16, 17, 24, 31, 32, 33, 64, 65, 128)
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            _initialize(native, state)
            original = native.pf_device_get_particle_states(state)
            rng = np.random.default_rng(count)
            factors = rng.normal(size=(count, 3, 3))
            covariance = factors @ np.swapaxes(factors, 1, 2) + np.eye(3)[None] * .1
            original[:, 7:] = covariance.reshape(count, 9)
            prior_weights = rng.normal(size=count)
            for n_sat in satellites:
                observations = _observations(n_sat)
                for variant in ("normal", "masked", "near", "invalid_S", "empty", "tiny_sigma", "nonfinite"):
                    initial = original.copy()
                    sat, velocity, doppler, weights = [array.copy() for array in observations]
                    sigma = .5
                    if variant == "masked":
                        weights[::3] = 0.
                        weights[1::3] = -1.
                    elif variant == "near" and n_sat:
                        sat[0] = initial[0, :3] + np.array([.5, 0., 0.])
                        if n_sat > 1:
                            sat[1] = initial[0, :3] + np.array([1., 0., 0.])
                    elif variant == "invalid_S":
                        initial[:, 7:] = -np.eye(3).ravel() * 1e10
                    elif variant == "empty":
                        weights[:] = 0.
                    elif variant == "tiny_sigma":
                        sigma = 1e-200
                    elif variant == "nonfinite":
                        weights[::3] = np.nan
                        weights[1::3] = np.inf
                    native.pf_device_set_particle_states(state, initial)
                    native.pf_device_set_log_weights(state, prior_weights)
                    for _ in range(2):
                        _kf(native, state, (sat, velocity, doppler, weights), sigma)
                    actual, log_weights = _snapshot(native, state)
                    np.testing.assert_array_equal(actual[:, :4].view(np.uint64), initial[:, :4].view(np.uint64))
                    if variant == "empty" or n_sat == 0:
                        np.testing.assert_array_equal(actual.view(np.uint64), initial.view(np.uint64))
                        np.testing.assert_array_equal(log_weights.view(np.uint64), prior_weights.view(np.uint64))
                    records[f"N{count}/sat{n_sat}/{variant}/two_updates"] = _digest(actual, log_weights)
            # Geometry cache boundaries, observation scratch growth, covariance
            # invalidation, and particle pointer swaps in the same owner.
            _initialize(native, state)
            for n_sat in (12, 32, 33, 128, 16, 65, 1):
                _kf(native, state, _observations(n_sat))
                native.pf_device_resample_megopolis_coalesced(state, 7, 42)
            records[f"N{count}/mixed_counts_and_resample"] = _digest(*_snapshot(native, state))
        finally:
            native.pf_device_destroy(state)
    for count in (100_000, 1_000_000):
        state = native.pf_device_create(count)
        try:
            for n_sat in (12, 32, 65):
                _initialize(native, state)
                _kf(native, state, _observations(n_sat))
                records[f"N{count}/sat{n_sat}/large_cloud"] = _digest(*_snapshot(native, state))
        finally:
            native.pf_device_destroy(state)
    return records


def _profile(native, sizes, repeats):
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    position, pr_satellites, ranges, weights_pr = _scenario()
    measurements = {}
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            for n_sat in (4, 12, 16, 17, 32, 64, 65):
                observations = _observations(n_sat)
                measurements[f"N{count}/KF_sat{n_sat}"] = _measure(native, state,
                    lambda: _initialize(native, state), lambda: _kf(native, state, observations), repeats)
                measurements[f"N{count}/KF_sat{n_sat}"]["observation_sha256"] = _digest(*observations)
            observations = _observations(12)

            def predict():
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1, 0., .3, True, .1)

            def update_step(doppler):
                predict()
                native.pf_device_weight(state, pr_satellites, ranges, weights_pr, 12, 5.)
                if doppler:
                    _kf(native, state, observations)
                ess = native.pf_device_ess(state)
                did_resample = ess < .5 * count
                if did_resample:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return native.pf_device_estimate(state), np.array([ess, did_resample])

            for doppler in (False, True):
                name = "native_PR_KF_step" if doppler else "native_PR_control_step"
                measurements[f"N{count}/{name}"] = _measure(native, state,
                    lambda: _initialize(native, state), lambda: update_step(doppler), repeats)
        finally:
            native.pf_device_destroy(state)
        observations = _observations(12)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare_wrapper():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)

            def wrapper_step():
                pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3,
                           rbpf_velocity_kf=True, velocity_process_noise=.1)
                pf.update(pr_satellites, ranges, weights_pr, resample=False)
                pf.update_doppler_kf(*observations[:3], weights=observations[3], wavelength=.19, sigma_mps=.5)
                return pf.estimate()

            measurements[f"N{count}/public_PR_KF_step"] = _measure(
                native, pf._state, prepare_wrapper, wrapper_step, repeats)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    return {"validation": _validate(native) if args.verify else {},
            "measurements": {} if args.validation_only else _profile(native, args.sizes, args.repeats)}


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "benchmarks/pf_covariance_gpu_20261011.json").read_text())
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    source_paths = list(reference["source_sha256"]["candidate"])
    baseline_hashes = {name: hashlib.sha256((args.baseline_dir / Path(name).name).read_bytes()).hexdigest()
                       for name in source_paths}
    extension_hashes = {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in paths.items()}
    assert baseline_hashes == reference["source_sha256"]["candidate"], "Wrong preceding sources"
    assert extension_hashes["baseline"] == reference["extension_sha256"]["candidate"], "Wrong preceding binary"
    runs = {label: [] for label in paths}
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
            before, after = [runs[label][0]["validation"] for label in paths]
            mismatches = [key for key in before if before[key] != after.get(key)]
            assert before.keys() == after.keys() and not mismatches, f"Seeded output mismatch: {mismatches[:10]}"
    summary = {}
    for key in runs["baseline"][0]["measurements"]:
        hashes = {run["measurements"][key]["output_sha256"] for data in runs.values() for run in data}
        assert len(hashes) == 1, f"Timed output differs: {key}"
        medians = {label: [run["measurements"][key]["median_ms"] for run in data] for label, data in runs.items()}
        baseline, candidate = [float(np.median(medians[label])) for label in paths]
        summary[key] = {"baseline_ms": baseline, "candidate_ms": candidate, "speedup": baseline / candidate,
                        "round_medians_ms": medians, "output_sha256": hashes.pop()}
    benchmark_files = ("benchmarks/bench_pf_doppler_compare.py", "benchmarks/bench_pf_coalesced_compare.py",
                       "benchmarks/bench_pf_covariance_compare.py")
    return {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "method": "actual production workers; alternating build order; synchronized wall time; non-exclusive host",
        "scope": "KF calls and predict/PR/KF/ESS/conditional B60 resample/estimate; restoration and dumps excluded",
        "rounds": args.rounds, "repeats": args.repeats, "sizes": args.sizes, "warmup_calls_per_stage": 3,
        "source_sha256": {"baseline": baseline_hashes,
                          "candidate": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_paths}},
        "extension_sha256": extension_hashes,
        "benchmark_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in benchmark_files},
        "validation": {"cases": len(before), "all_state_outputs_bit_identical": True,
                       "baseline_sha256_by_case": before, "candidate_sha256_by_case": after},
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
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 10_000, 100_000, 1_000_000])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.rounds < 1 or args.repeats < 1:
        parser.error("sizes, rounds and repeats must be positive")
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
