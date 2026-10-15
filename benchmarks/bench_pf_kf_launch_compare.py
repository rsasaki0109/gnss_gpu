"""Compare actual PF Doppler-KF launch layouts and complete update steps.

Archive the preceding specialized-prediction build and five PF sources. Separate
sequential production workers alternate build order by round. Restoration and
output checks are excluded from synchronized wall time.
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
from bench_pf_doppler_compare import _kf, _observations, _validate as _preceding_validation
from bench_pf_reduction_compare import _measure


def _cloud(count, variant="normal"):
    rng = np.random.default_rng(count)
    original = rng.normal(size=(count, 16))
    original[:, :3] += _scenario()[0]
    original[:, 3] += 3000.
    factors = rng.normal(size=(count, 3, 3))
    original[:, 7:] = (factors @ factors.swapaxes(1, 2) + np.eye(3) * .1).reshape(count, 9)
    if variant == "nonsymmetric":
        original[:, 7:] = rng.normal(size=(count, 9))
    elif variant == "zero":
        original[:, 7:] = 0.
    elif variant == "nonfinite":
        original[:, 7:] = np.resize([np.nan, np.inf, -np.inf, -0.], count)[:, None]
    elif variant == "signed_zero":
        original[:, 7:] = [4., 0., -0., -0., 3., -0., 0., 0., 2.]
    elif variant == "large":
        original[:, 7:] *= 1e150
    elif variant == "tiny":
        original[:, 7:] *= 1e-300
    return original, rng.normal(size=count)


def _validate(native):
    records = _preceding_validation(native)
    for count in (1, 33, 63, 64, 65, 127, 128, 129, 257):
        state = native.pf_device_create(count)
        try:
            for variant in ("nonsymmetric", "zero", "nonfinite", "signed_zero", "large", "tiny"):
                original, prior = _cloud(count, variant)
                for n_sat in (1, 2, 4, 12, 16, 17, 32, 33, 65):
                    observations = _observations(n_sat)
                    native.pf_device_set_particle_states(state, original)
                    native.pf_device_set_log_weights(state, prior)
                    for _ in range(2):
                        _kf(native, state, observations)
                    actual, weights = _snapshot(native, state)
                    np.testing.assert_array_equal(actual[:, :4].view(np.uint64), original[:, :4].view(np.uint64))
                    records[f"N{count}/sat{n_sat}/extra_{variant}"] = dict(
                        input_sha256=_digest(original, prior, *observations), output_sha256=_digest(actual, weights))
        finally:
            native.pf_device_destroy(state)
    for count in (100000, 1000000):
        state = native.pf_device_create(count)
        try:
            original, prior = _cloud(count)
            for n_sat in (12, 32):
                native.pf_device_set_particle_states(state, original)
                native.pf_device_set_log_weights(state, prior)
                observations = _observations(n_sat)
                _kf(native, state, observations)
                records[f"N{count}/sat{n_sat}/imported_large"] = dict(
                    input_sha256=_digest(original, prior, *observations), output_sha256=_digest(*_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    _, sat, pr, weights = _scenario()
    observations = _observations(12)
    for count in (33, 65, 257, 2048, 10000, 100000, 1000000):
        state = native.pf_device_create(count)
        try:
            for doppler in (False, True):
                _initialize(native, state)
                history = []
                for step in range(1, 6):
                    native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, step, 0., .3, True, .1)
                    native.pf_device_weight(state, sat, pr, weights, 12, 5.)
                    if doppler:
                        _kf(native, state, observations)
                    ess = native.pf_device_ess(state)
                    resampled = ess < .5 * count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state, 60, 42 + step)
                    history.append(np.array([ess, resampled, *native.pf_device_estimate(state)]))
                records[f"N{count}/five_epoch/doppler{doppler}"] = dict(
                    input_sha256=_digest(sat, pr, weights, *observations),
                    output_sha256=_digest(*history, *_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    return records


def _profile(native, sizes, repeats):
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    measurements = {}
    position, sat, pr, weights = _scenario()
    observations = _observations(12)
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            for n_sat in (4, 12, 16, 17, 32, 33, 65):
                inputs = _observations(n_sat)
                measurements[f"N{count}/KF_sat{n_sat}"] = _measure(native, state,
                    lambda: _initialize(native, state), lambda: _kf(native, state, inputs), repeats)
                measurements[f"N{count}/KF_sat{n_sat}"]["input_sha256"] = _digest(*inputs)
            original, prior = _cloud(count)

            def restore():
                native.pf_device_set_particle_states(state, original)
                native.pf_device_set_log_weights(state, prior)

            for n_sat in (12, 32):
                inputs = _observations(n_sat)
                measurements[f"N{count}/KF_imported_sat{n_sat}"] = _measure(native, state, restore,
                    lambda: _kf(native, state, inputs), repeats)
                measurements[f"N{count}/KF_imported_sat{n_sat}"]["input_sha256"] = _digest(original, prior, *inputs)

            def predict():
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1, 0., .3, True, .1)

            def chain():
                for _ in range(5):
                    _kf(native, state, observations)

            def step(doppler=True, resample=True):
                predict()
                native.pf_device_weight(state, sat, pr, weights, 12, 5.)
                if doppler:
                    _kf(native, state, observations)
                ess = native.pf_device_ess(state)
                did_resample = resample and ess < .5 * count
                if did_resample:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return np.array([ess, did_resample, *native.pf_device_estimate(state)])

            calls = dict(KF_five_updates=chain, native_PR_KF_step=step,
                         native_PR_control_step=lambda: step(doppler=False),
                         native_PR_KF_no_resample=lambda: step(resample=False),
                         Gaussian_control=lambda: native.pf_device_weight(state, sat, pr, weights, 12, 5.),
                         resample_control=lambda: native.pf_device_resample_megopolis_coalesced(state, 60, 42),
                         predict_control=predict)
            for name, call in calls.items():
                measurements[f"N{count}/{name}"] = _measure(native, state,
                    lambda: _initialize(native, state), call, repeats)
        finally:
            native.pf_device_destroy(state)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)

            def public_step(doppler=True):
                pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3,
                           rbpf_velocity_kf=True, velocity_process_noise=.1)
                pf.update(sat, pr, weights, resample=not doppler)
                if doppler:
                    pf.update_doppler_kf(*observations[:3], weights=observations[3], wavelength=.19, sigma_mps=.5)
                return pf.estimate()

            for name, call in dict(public_PR_KF_step=public_step,
                                   public_PR_control_step=lambda: public_step(doppler=False)).items():
                measurements[f"N{count}/{name}"] = _measure(native, pf._state, prepare, call, repeats)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    return dict(validation=_validate(native) if args.verify else {},
                measurements={} if args.validation_only else _profile(native, args.sizes, args.repeats))


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "benchmarks/pf_predict_gpu_20261012.json").read_text(encoding="utf-8"))
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
    benchmark_files = ("benchmarks/bench_pf_kf_launch_compare.py", "benchmarks/bench_pf_reduction_compare.py",
                       "benchmarks/bench_pf_coalesced_compare.py", "benchmarks/bench_pf_covariance_compare.py",
                       "benchmarks/bench_pf_doppler_compare.py")
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
                baseline="preceding specialized prediction and Gaussian PR with 256-thread Doppler KF blocks",
                method="actual sequential production workers; alternating build order by round; nonexclusive host",
                scope="synchronized Doppler KF/common and imported covariance; native/public predict/PR/optional KF/ESS/conditional B60 "
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
