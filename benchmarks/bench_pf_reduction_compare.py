"""Compare ordered FP64 PF query reductions in actual production binaries.

Archive the preceding Doppler-KF build and five PF sources before editing.
Separate sequential workers avoid pybind class collisions; build order alternates
per round. Time synchronized queries and actual native/public update steps.
Restoration and output checks are excluded. Check every timed returned value and
the final particle/log-weight snapshot for each stage, including unchanged controls.
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

from bench_pf_coalesced_compare import _initialize, _load_extension, _scenario
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations


def _queries(native, state, center):
    return np.array([native.pf_device_ess(state), *native.pf_device_estimate(state),
                     native.pf_device_position_spread(state, *center)])


def _weights(count, rng):
    normal = rng.normal(0., 3., count)
    dominant = np.full(count, -1000.)
    dominant[count // 2] = 0.
    single = np.full(count, -np.inf)
    single[count // 2] = 0.
    return dict(
        uniform=np.zeros(count), normal=normal, shifted=normal - 1e6,
        dominant=dominant, block_contrast=normal + ((np.arange(count) // 256) % 3 - 1) * 1e4,
        signed_zero=np.resize([0., -0.], count), tiny=normal * 1e-310,
        nearly_uniform=normal * 1e-7, extreme=np.resize([0., 1e300, -1e300, 1e-310], count),
        nonfinite=np.resize([0., np.nan, np.inf, -np.inf], count),
        all_nan=np.full(count, np.nan), all_inf=np.full(count, np.inf),
        all_neginf=np.full(count, -np.inf), one_finite=single)


def _cloud(count, rng, variant):
    states = rng.normal(size=(count, 16))
    center = np.array([-3957201., 3310204., 3737913.])
    states[:, :3] += [-3957199., 3310205., 3737911.]
    states[:, 3] += 3000.
    if variant != "ecef":
        center = np.zeros(3)
        values = {
            "signed_zero": [0., -0.], "cancellation": [1e15, 1., -1e15, 3.],
            "tiny": [1e-300, -1e-300, 0.], "nonfinite": [np.nan, np.inf, -np.inf, -0.],
            "huge": [1e300, -1e300, 1.],
        }[variant]
        states[:, :4] = np.resize(values, count)[:, None] * np.array([1., -1., 1., -1.])
    return states, center


def _validate(native):
    records = {}
    sizes = (1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
             511, 512, 513, 1023, 1024, 1025, 8191, 8192, 8193, 65535, 65536, 65537,
             100000, 1000000)
    for count in sizes:
        state = native.pf_device_create(count)
        rng = np.random.default_rng(count)
        distributions = _weights(count, rng)
        variants = ("ecef", "signed_zero", "cancellation", "tiny", "nonfinite", "huge")
        if count >= 100000:
            variants = ("ecef", "cancellation")
            distributions = {key: distributions[key] for key in (
                "uniform", "normal", "shifted", "block_contrast")}
        try:
            for variant in variants:
                original, center = _cloud(count, rng, variant)
                native.pf_device_set_particle_states(state, original)
                for name, weights in distributions.items():
                    native.pf_device_set_log_weights(state, weights)
                    result = _queries(native, state, center)
                    # Query scratch is shared; repeating in a different order
                    # must retain every bit and must not overwrite earlier arrays.
                    spread = native.pf_device_position_spread(state, *center)
                    mean = native.pf_device_estimate(state)
                    ess = native.pf_device_ess(state)
                    repeated = np.array([ess, *mean, spread])
                    np.testing.assert_array_equal(result.view(np.uint64), repeated.view(np.uint64))
                    records[f"N{count}/{variant}/{name}"] = dict(
                        input_sha256=_digest(original, weights, center), output_sha256=_digest(result),
                        output_bits=[f"{bits:016x}" for bits in result.view(np.uint64)])
                actual, actual_weights = _snapshot(native, state)
                np.testing.assert_array_equal(actual.view(np.uint64), original.view(np.uint64))
                np.testing.assert_array_equal(actual_weights.view(np.uint64), weights.view(np.uint64))
                records[f"N{count}/{variant}/unchanged_state"] = dict(
                    input_sha256=_digest(original, weights), output_sha256=_digest(actual, actual_weights))
        finally:
            native.pf_device_destroy(state)
    position, sat, ranges, obs_weights = _scenario()
    observations = _observations(12)
    for count in (2048, 10000, 100000, 1000000):
        state = native.pf_device_create(count)
        try:
            for mode in ("PR", "PR_KF", "PR_no_resample"):
                _initialize(native, state)
                history = []
                for step in range(1, 6):
                    native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42,
                                             step, 0., .3, mode == "PR_KF", .1)
                    native.pf_device_weight(state, sat, ranges, obs_weights, 12,
                                            1000. if mode == "PR_no_resample" else 5.)
                    if mode == "PR_KF":
                        _kf(native, state, observations)
                    ess = native.pf_device_ess(state)
                    resampled = ess < .5 * count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state, 60, 42 + step)
                    mean = native.pf_device_estimate(state)
                    spread = native.pf_device_position_spread(state, *mean[:3])
                    history.append(np.array([ess, resampled, *mean, spread]))
                records[f"N{count}/five_epoch/{mode}"] = dict(
                    input_sha256=_digest(position, sat, ranges, obs_weights, *observations),
                    output_sha256=_digest(*history, *_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    return records


def _measure(native, state, prepare, operation, repeats):
    samples, return_hashes = [], set()
    for repetition in range(repeats + 3):
        prepare()
        native.pf_device_sync(state)
        start = time.perf_counter_ns()
        result = operation()
        native.pf_device_sync(state)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        output = np.empty(0) if result is None else np.asarray(result, dtype=np.float64)
        return_hashes.add(_digest(output))
        if repetition >= 3:
            samples.append(elapsed)
    assert len(return_hashes) == 1, "Repeated returned outputs changed"
    particles, weights = _snapshot(native, state)
    return dict(median_ms=float(np.median(samples)), samples_ms=samples,
                output_sha256=_digest(particles, weights, output),
                returned_output_sha256=return_hashes.pop(), all_returned_outputs_identical=True)


def _profile(native, sizes, repeats):
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    position, sat, ranges, obs_weights = _scenario()
    observations = _observations(12)
    measurements = {}
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            _initialize(native, state)
            weights = np.random.default_rng(count).normal(0., 1.5, count)
            center = position + [1., 2., 3.]
            for distribution in ("normal", "uniform"):
                log_weights = weights if distribution == "normal" else np.zeros(count)

                def prepare_weights():
                    native.pf_device_set_log_weights(state, log_weights)

                operations = dict(ESS=lambda: native.pf_device_ess(state),
                                  estimate=lambda: native.pf_device_estimate(state),
                                  spread=lambda: native.pf_device_position_spread(state, *center))
                for name, call in operations.items():
                    key = f"N{count}/{name}_{distribution}"
                    measurements[key] = _measure(native, state, prepare_weights, call, repeats)
                    measurements[key]["input_sha256"] = _digest(log_weights, center)
            measurements[f"N{count}/native_diagnostics"] = _measure(native, state,
                lambda: native.pf_device_set_log_weights(state, weights),
                lambda: _queries(native, state, center), repeats)
            measurements[f"N{count}/weight_control"] = _measure(native, state,
                lambda: _initialize(native, state),
                lambda: native.pf_device_weight(state, sat, ranges, obs_weights, 12, 5.), repeats)

            def step(kf=False, sigma=5., spread=False):
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1,
                                         0., .3 if kf else 1., kf, .1 if kf else 0.)
                native.pf_device_weight(state, sat, ranges, obs_weights, 12, sigma)
                if kf:
                    _kf(native, state, observations)
                ess = native.pf_device_ess(state)
                resampled = ess < .5 * count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                mean = native.pf_device_estimate(state)
                rms = native.pf_device_position_spread(state, *mean[:3]) if spread else 0.
                return np.array([ess, resampled, *mean, rms])

            for name, call in dict(native_PR_step=lambda: step(),
                                   native_PR_no_resample_step=lambda: step(sigma=1000.),
                                   native_PR_KF_step=lambda: step(kf=True),
                                   native_PR_spread_step=lambda: step(spread=True)).items():
                measurements[f"N{count}/{name}"] = _measure(native, state,
                    lambda: _initialize(native, state), call, repeats)
        finally:
            native.pf_device_destroy(state)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare_public():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)

            def public_step(kf=False):
                pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3 if kf else 1.,
                           rbpf_velocity_kf=kf, velocity_process_noise=.1 if kf else 0.)
                pf.update(sat, ranges, obs_weights, resample=not kf)
                if kf:
                    pf.update_doppler_kf(*observations[:3], weights=observations[3],
                                        wavelength=.19, sigma_mps=.5)
                return pf.estimate()

            for name, call in dict(public_PR_step=lambda: public_step(),
                                   public_PR_KF_step=lambda: public_step(kf=True),
                                   public_diagnostics=lambda: np.array([
                                       pf.get_ess(), *pf.estimate(), pf.get_position_spread()])).items():
                measurements[f"N{count}/{name}"] = _measure(native, pf._state,
                    prepare_public, call, repeats)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    return dict(validation=_validate(native) if args.verify else {},
                measurements={} if args.validation_only else _profile(native, args.sizes, args.repeats))


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "benchmarks/pf_doppler_gpu_20261012.json").read_text(encoding="utf-8"))
    paths = dict(baseline=args.baseline, candidate=args.candidate)
    sources = list(reference["source_sha256"]["candidate"])
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    baseline_hashes = {name: sha(args.baseline_dir / Path(name).name) for name in sources}
    extensions = {label: sha(path) for label, path in paths.items()}
    assert baseline_hashes == reference["source_sha256"]["candidate"], "Wrong preceding sources"
    assert extensions["baseline"] == reference["extension_sha256"]["candidate"], "Wrong preceding binary"
    runs = {label: [] for label in paths}
    for round_index in range(args.rounds):
        for label in (("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")):
            command = [sys.executable, str(Path(__file__).resolve()), "--extension", str(paths[label].resolve()),
                       "--sizes", *map(str, args.sizes), "--repeats", str(args.repeats)]
            if round_index == 0:
                command.append("--verify")
            if args.validation_only:
                command.append("--validation-only")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{label} worker failed:\n{result.stderr}")
            run = json.loads(result.stdout)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / f"{args.output.stem}_{label}_round{round_index + 1}.json").write_text(
                json.dumps(run, indent=2) + "\n", encoding="utf-8")
            runs[label].append(run)
            print(f"round {round_index + 1}/{args.rounds}: {label} complete", flush=True)
        if round_index == 0:
            before, after = [runs[label][0]["validation"] for label in paths]
            assert before.keys() == after.keys(), "Validation cases changed"
            nan_bit_differences = []
            for key, old_case in before.items():
                new_case = after[key]
                assert old_case["input_sha256"] == new_case["input_sha256"], f"Input changed: {key}"
                if old_case == new_case:
                    continue
                # CUDA can choose a different NaN sign/payload after register
                # reduction. Preserve the raw evidence and require all differing
                # values to remain NaN; every other value must match every bit.
                assert "output_bits" in old_case and "output_bits" in new_case, f"State changed: {key}"
                old_bits = np.array([int(bits, 16) for bits in old_case["output_bits"]], dtype=np.uint64)
                new_bits = np.array([int(bits, 16) for bits in new_case["output_bits"]], dtype=np.uint64)
                different = old_bits != new_bits
                assert np.all(np.isnan(old_bits.view(np.float64)[different]) &
                              np.isnan(new_bits.view(np.float64)[different])), f"Numerical output changed: {key}"
                nan_bit_differences.append(key)
    summary = {}
    for key in runs["baseline"][0]["measurements"]:
        hashes = {run["measurements"][key]["output_sha256"] for data in runs.values() for run in data}
        return_hashes = {run["measurements"][key]["returned_output_sha256"]
                         for data in runs.values() for run in data}
        assert len(hashes) == len(return_hashes) == 1, f"Timed output changed: {key}"
        medians = {label: [run["measurements"][key]["median_ms"] for run in data]
                   for label, data in runs.items()}
        old, new = [float(np.median(medians[label])) for label in paths]
        summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old / new,
                            round_medians_ms=medians, output_sha256=hashes.pop(),
                            returned_output_sha256=return_hashes.pop())
        print(f"{key}: {old:.4f} -> {new:.4f} ms ({old / new:.3f}x)", flush=True)
    benchmarks = ("benchmarks/bench_pf_reduction_compare.py", "benchmarks/bench_pf_coalesced_compare.py",
                  "benchmarks/bench_pf_covariance_compare.py", "benchmarks/bench_pf_doppler_compare.py")
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
                baseline="preceding Doppler-KF geometry reuse implementation",
                method="actual sequential production workers; alternating build order by round; nonexclusive host",
                scope="synchronized diagnostic queries and native/public PR + optional KF / ESS / "
                      "conditional B60 resample / estimate / optional spread; restoration and dumps excluded",
                rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, warmup_calls_per_stage=3,
                source_sha256=dict(baseline=baseline_hashes, candidate={name: sha(root / name) for name in sources}),
                extension_sha256=extensions, benchmark_sha256={name: sha(root / name) for name in benchmarks},
                validation=dict(cases=len(before), bit_identical_cases=len(before) - len(nan_bit_differences),
                                nan_only_bit_difference_cases=nan_bit_differences,
                                all_non_nan_outputs_bit_identical=True,
                                all_state_snapshots_and_trajectories_bit_identical=True,
                                baseline_by_case=before, candidate_by_case=after),
                summary=summary, runs={label: [run["measurements"] for run in data] for label, data in runs.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 10000, 100000, 1000000])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
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
