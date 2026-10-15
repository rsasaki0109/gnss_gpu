"""Compare production PF builds with four/eight proposal weights fetched ahead.

Archive the preceding query-reduction build and five PF sources before editing.
Separate sequential workers avoid pybind registration collisions. Alternate
build order by round; time synchronized resampling and actual update steps.
Initialization/restoration, snapshots and output checks are outside timing.
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
from bench_pf_covariance_compare import _digest, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from bench_pf_reduction_compare import _measure


def _validate(native):
    records = {}
    sizes = (1, 31, 33, 255, 256, 257, 8191, 8192, 8193, 16384, 65535, 65536, 65537,
             100000, 1000000)
    iterations = (0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 19, 20, 59, 60, 61, 63, 64,
                  65, 239, 240, 241, 255, 256, 257)
    for count in sizes:
        rng = np.random.default_rng(count)
        original = rng.normal(size=(count, 16))
        original[:, 0] = np.arange(count)
        original[::17, 8] = -0.
        original[::31, 12] = np.nan
        normal = rng.normal(0., 1.5, count)
        dominant = np.full(count, -1000.)
        dominant[count // 2] = 0.
        distributions = dict(uniform=np.zeros(count), normal=normal, dominant=dominant,
                             shifted=normal - 1e6, tiny=normal * 1e-7,
                             extreme=np.resize([0., 1e300, -1e300, np.nextafter(0., 1.)], count),
                             nonfinite=np.resize([0., np.inf, -np.inf, np.nan], count))
        chains = iterations
        if count >= 100000:
            chains = (8, 15, 16, 17, 60, 61, 63, 64, 240, 241, 257)
            distributions = {key: distributions[key] for key in ("uniform", "normal", "dominant", "nonfinite")}
        state = native.pf_device_create(count)
        try:
            for chain in chains:
                for distribution, weights in distributions.items():
                    for seed in (0, 2**64 - 1):
                        native.pf_device_set_particle_states(state, original)
                        native.pf_device_set_log_weights(state, weights)
                        native.pf_device_resample_megopolis_coalesced(state, chain, seed)
                        particles, reset_weights = _snapshot(native, state)
                        indices = particles[:, 0].astype(np.int64)
                        np.testing.assert_array_equal(particles.view(np.uint64), original[indices].view(np.uint64))
                        np.testing.assert_array_equal(reset_weights.view(np.uint64), np.zeros(count).view(np.uint64))
                        records[f"N{count}/B{chain}/{distribution}/seed{seed}"] = dict(
                            input_sha256=_digest(original, weights), output_sha256=_digest(particles, reset_weights))
            if count in (8192, 8193):
                _initialize(native, state)
                for chain in (60, 7, 240, 8, 61):
                    native.pf_device_set_log_weights(state, normal)
                    native.pf_device_resample_megopolis_coalesced(state, chain, 42)
                    native.pf_device_set_log_weights(state, normal)
                    native.pf_device_resample_megopolis(state, chain % 5, 1)
                    native.pf_device_set_log_weights(state, normal)
                    native.pf_device_resample_systematic(state, 1)
                records[f"N{count}/mixed_legacy_systematic"] = dict(
                    input_sha256=_digest(normal), output_sha256=_digest(*_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    _, satellites, ranges, obs_weights = _scenario()
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
                    native.pf_device_weight(state, satellites, ranges, obs_weights, 12,
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
                    input_sha256=_digest(satellites, ranges, obs_weights, *observations),
                    output_sha256=_digest(*history, *_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    return records


def _profile(native, sizes, repeats):
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    position, satellites, ranges, obs_weights = _scenario()
    observations = _observations(12)
    measurements = {}
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            _initialize(native, state)
            imported = native.pf_device_get_particle_states(state)
            imported[:, 7] = np.arange(count) + 1.
            imported[:, 11] = imported[:, 7] * 2.
            imported[:, 15] = imported[:, 7] * 3.
            normal = np.random.default_rng(count).normal(0., 1.5, count)
            uniform = np.zeros(count)
            for mode in ("common", "imported"):
                def prepare(mode=mode):
                    if mode == "common":
                        _initialize(native, state)
                    else:
                        native.pf_device_set_particle_states(state, imported)
                    native.pf_device_set_log_weights(state, normal)

                for chain in (0, 16, 60, 240):
                    key = f"N{count}/resample_normal_B{chain}/{mode}_cov"
                    measurements[key] = _measure(native, state, prepare,
                        lambda chain=chain: native.pf_device_resample_megopolis_coalesced(state, chain, 42), repeats)
                    measurements[key]["input_sha256"] = _digest(normal, imported if mode == "imported" else position)

            def prepare_uniform():
                _initialize(native, state)
                native.pf_device_set_log_weights(state, uniform)

            measurements[f"N{count}/resample_uniform_B60"] = _measure(native, state, prepare_uniform,
                lambda: native.pf_device_resample_megopolis_coalesced(state, 60, 42), repeats)

            def predict(kf=False):
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1,
                                         0., .3 if kf else 1., kf, .1 if kf else 0.)

            def weight(sigma=5.):
                native.pf_device_weight(state, satellites, ranges, obs_weights, 12, sigma)

            def prepare_weighted():
                _initialize(native, state)
                predict()
                weight()

            measurements[f"N{count}/weighted_resample_B60"] = _measure(native, state, prepare_weighted,
                lambda: native.pf_device_resample_megopolis_coalesced(state, 60, 42), repeats)

            def step(kf=False, sigma=5.):
                predict(kf)
                weight(sigma)
                if kf:
                    _kf(native, state, observations)
                ess = native.pf_device_ess(state)
                resampled = ess < .5 * count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return np.array([ess, resampled, *native.pf_device_estimate(state)])

            for name, call in dict(native_PR_step=lambda: step(), native_PR_KF_step=lambda: step(kf=True),
                                   native_PR_no_resample_control=lambda: step(sigma=1000.),
                                   predict_control=predict, weight_control=weight,
                                   KF_control=lambda: _kf(native, state, observations)).items():
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
                pf.update(satellites, ranges, obs_weights, resample=not kf)
                if kf:
                    pf.update_doppler_kf(*observations[:3], weights=observations[3], wavelength=.19, sigma_mps=.5)
                return pf.estimate()

            for name, call in dict(public_PR_step=public_step, public_PR_KF_step=lambda: public_step(kf=True)).items():
                measurements[f"N{count}/{name}"] = _measure(native, pf._state, prepare_public, call, repeats)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    return dict(validation=_validate(native) if args.verify else {},
                measurements={} if args.validation_only else _profile(native, args.sizes, args.repeats))


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "benchmarks/pf_reduction_gpu_20261012.json").read_text(encoding="utf-8"))
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
            worker = subprocess.run(command, capture_output=True, text=True)
            if worker.returncode:
                raise RuntimeError(f"{label} worker failed:\n{worker.stderr}")
            run = json.loads(worker.stdout)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            (args.output.parent / f"{args.output.stem}_{label}_round{round_index + 1}.json").write_text(
                json.dumps(run, indent=2) + "\n", encoding="utf-8")
            runs[label].append(run)
            print(f"round {round_index + 1}/{args.rounds}: {label} complete", flush=True)
        if round_index == 0:
            before, after = [runs[label][0]["validation"] for label in paths]
            assert before == after, "Seeded particle / log-weight / trajectory / input hash mismatch"
    summary = {}
    for key in runs["baseline"][0]["measurements"]:
        hashes = {run["measurements"][key]["output_sha256"] for data in runs.values() for run in data}
        returns = {run["measurements"][key]["returned_output_sha256"] for data in runs.values() for run in data}
        assert len(hashes) == len(returns) == 1, f"Timed output changed: {key}"
        medians = {label: [run["measurements"][key]["median_ms"] for run in data] for label, data in runs.items()}
        old, new = [float(np.median(medians[label])) for label in paths]
        summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old / new,
                            round_medians_ms=medians, output_sha256=hashes.pop(),
                            returned_output_sha256=returns.pop())
        print(f"{key}: {old:.4f} -> {new:.4f} ms ({old / new:.3f}x)", flush=True)
    benchmark_files = ("benchmarks/bench_pf_prefetch_compare.py", "benchmarks/bench_pf_reduction_compare.py",
                       "benchmarks/bench_pf_coalesced_compare.py", "benchmarks/bench_pf_covariance_compare.py",
                       "benchmarks/bench_pf_doppler_compare.py")
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
                baseline="preceding ordered FP64 query-reduction implementation, four proposals prefetched",
                method="actual sequential production workers; alternating build order by round; nonexclusive host",
                scope="synchronized resampling and native/public predict/PR/optional KF/ESS/conditional B60 "
                      "resample/estimate; initialization/restoration and output checks excluded",
                rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, warmup_calls_per_stage=3,
                source_sha256=dict(baseline=baseline_hashes, candidate={name: sha(root / name) for name in sources}),
                extension_sha256=extensions, benchmark_sha256={name: sha(root / name) for name in benchmark_files},
                validation=dict(cases=len(before), all_outputs_bit_identical=True,
                                baseline_by_case=before, candidate_by_case=after), summary=summary,
                runs={label: [run["measurements"] for run in data] for label, data in runs.items()})


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
