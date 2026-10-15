"""Compare actual general/specialized Student PF kernels and complete update steps.

Archive the preceding KF-launch build and five PF sources. Separate
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
from bench_pf_doppler_compare import _kf, _observations
from bench_pf_reduction_compare import _measure


def _pr(count):
    position = _scenario()[0]
    satellite = _observations(count)[0]
    ranges = np.linalg.norm(satellite - position, axis=1) + 3000.
    weights = np.random.default_rng(count).uniform(.1, 1., count)
    return satellite, ranges, weights


def _weight(native, state, observations, sigma=5., nu=0., gate=0., huber=False, k=1.5):
    sat, pr, weights = observations
    native.pf_device_weight(state, sat, pr, weights, len(weights), sigma, nu, gate, huber, k)


def _validate(native):
    records = {}
    modes = dict(gaussian=(5., 0., 0., False, 1.5), negative_nu=(5., -3., 0., False, 1.5),
                 disabled_huber=(5., 0., -1., True, 0.), zero_weight=(5., 0., 0., False, 1.5),
                 tiny_sigma=(1e-200, 0., 0., False, 1.5), huge_sigma=(1e200, 0., 0., False, 1.5),
                 nonfinite_state=(5., 0., 0., False, 1.5), near=(5., 0., 0., False, 1.5),
                 student_nu1=(5., 1., 0., False, 1.5), student_nu2=(5., 2., 0., False, 1.5),
                 student_half=(5., .5, 0., False, 1.5), student_nu30=(5., 30., 0., False, 1.5),
                 student_tiny_nu=(5., 1e-200, 0., False, 1.5), student_large_nu=(5., 1e200, 0., False, 1.5),
                 student_tiny_sigma=(1e-200, 3., 0., False, 1.5), student_huge_sigma=(1e200, 3., 0., False, 1.5),
                 student_nonfinite_state=(5., 3., 0., False, 1.5), ignored_huber=(5., 3., 0., True, 1.5),
                 student=(5., 3., 0., False, 1.5), huber=(5., 0., 0., True, 1.5),
                 gate=(5., 0., 20., False, 1.5), student_huber_gate=(5., 3., 20., True, 1.5))
    for count in (1, 31, 33, 255, 256, 257, 8193):
        rng = np.random.default_rng(count)
        original = rng.normal(size=(count, 16))
        original[:, :3] += _scenario()[0]
        original[:, 3] += 3000.
        prior = rng.normal(size=count)
        state = native.pf_device_create(count)
        try:
            for n_sat in (1, 2, 3, 4, 8, 11, 12, 13, 16, 17, 32, 64, 65, 128):
                for mode, arguments in modes.items():
                    initial = original.copy()
                    observations = [array.copy() for array in _pr(n_sat)]
                    if mode == "zero_weight":
                        observations[2][:] = 0.
                    elif mode in ("nonfinite_state", "student_nonfinite_state"):
                        initial[:, :4] = np.resize([np.nan, np.inf, -np.inf, -0.], count)[:, None]
                    elif mode == "near":
                        observations[0][0] = initial[0, :3]
                        observations[1][0] = initial[0, 3]
                    native.pf_device_set_particle_states(state, initial)
                    native.pf_device_set_log_weights(state, prior)
                    _weight(native, state, observations, *arguments)
                    actual, weights = _snapshot(native, state)
                    np.testing.assert_array_equal(actual.view(np.uint64), initial.view(np.uint64))
                    first = weights.copy()
                    _weight(native, state, observations, *arguments)
                    _, repeated = _snapshot(native, state)
                    np.testing.assert_array_equal(weights.view(np.uint64), first.view(np.uint64))
                    np.testing.assert_array_equal(repeated.view(np.uint64), first.view(np.uint64))
                    records[f"N{count}/sat{n_sat}/{mode}"] = dict(
                        input_sha256=_digest(initial, prior, *observations, np.asarray(arguments, dtype=float)),
                        output_sha256=_digest(actual, weights))
            # Alternating dispatch modes and observation-buffer growth within
            # one owner must not change data, overwrite returned arrays or add.
            native.pf_device_set_particle_states(state, original)
            history = []
            for n_sat, mode in ((12, "gaussian"), (65, "student"), (13, "gaussian"),
                                (128, "huber"), (1, "near"), (16, "disabled_huber")):
                _weight(native, state, _pr(n_sat), *modes[mode])
                history.extend(_snapshot(native, state))
            records[f"N{count}/mixed_modes_growth"] = dict(
                input_sha256=_digest(original), output_sha256=_digest(*history))
        finally:
            native.pf_device_destroy(state)
    for count in (100000, 1000000):
        state = native.pf_device_create(count)
        try:
            for n_sat in (4, 12, 16, 32, 65):
                _initialize(native, state)
                observations = _pr(n_sat)
                _weight(native, state, observations)
                records[f"N{count}/sat{n_sat}/large"] = dict(
                    input_sha256=_digest(*observations), output_sha256=_digest(*_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    _, sat, pr, weights = _scenario()
    doppler = _observations(12)
    for count in (2048, 10000, 100000, 1000000):
        state = native.pf_device_create(count)
        try:
            for mode in ("PR", "PR_KF", "PR_high_ESS"):
                _initialize(native, state)
                history = []
                for step in range(1, 6):
                    native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, step, 0., .3, mode == "PR_KF", .1)
                    _weight(native, state, (sat, pr, weights), 1000. if mode == "PR_high_ESS" else 5., nu=3.)
                    if mode == "PR_KF":
                        _kf(native, state, doppler)
                    ess = native.pf_device_ess(state)
                    resampled = ess < .5 * count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state, 60, 42 + step)
                    mean = native.pf_device_estimate(state)
                    history.append(np.array([ess, resampled, *mean]))
                records[f"N{count}/five_epoch/{mode}"] = dict(
                    input_sha256=_digest(sat, pr, weights, *doppler),
                    output_sha256=_digest(*history, *_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    return records


def _profile(native, sizes, repeats):
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    measurements = {}
    position, sat, pr, weights = _scenario()
    doppler = _observations(12)
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            for n_sat in (4, 12, 16, 32, 64, 65):
                observations = _pr(n_sat)
                measurements[f"N{count}/Student_sat{n_sat}"] = _measure(native, state,
                    lambda: _initialize(native, state), lambda: _weight(native, state, observations, nu=3.), repeats)
                measurements[f"N{count}/Student_sat{n_sat}"]["input_sha256"] = _digest(*observations)
            observations = _pr(12)
            for mode, arguments in dict(Gaussian_control=(5., 0.), Student_nu1=(5., 1.), Student_nu10=(5., 10.),
                                        Student_gate_control=(5., 3., 20.), Huber_control=(5., 0., 0., True),
                                        Gate_control=(5., 0., 20.)).items():
                measurements[f"N{count}/{mode}"] = _measure(native, state,
                    lambda: _initialize(native, state), lambda: _weight(native, state, observations, *arguments), repeats)

            def predict(kf=False):
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1,
                                         0., .3 if kf else 1., kf, .1 if kf else 0.)

            def step(kf=False, sigma=5.):
                predict(kf)
                _weight(native, state, (sat, pr, weights), sigma, nu=3.)
                if kf:
                    _kf(native, state, doppler)
                ess = native.pf_device_ess(state)
                resampled = ess < .5 * count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return np.array([ess, resampled, *native.pf_device_estimate(state)])

            for name, call in dict(native_PR_step=step, native_PR_KF_step=lambda: step(kf=True),
                                   native_PR_high_ESS_step=lambda: step(sigma=1000.), predict_control=predict,
                                   resample_control=lambda: native.pf_device_resample_megopolis_coalesced(state, 60, 42),
                                   KF_control=lambda: _kf(native, state, doppler)).items():
                measurements[f"N{count}/{name}"] = _measure(native, state,
                    lambda: _initialize(native, state), call, repeats)
        finally:
            native.pf_device_destroy(state)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., nu=3., seed=42)
        try:
            def prepare_public():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)

            def public_step(kf=False):
                pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3 if kf else 1.,
                           rbpf_velocity_kf=kf, velocity_process_noise=.1 if kf else 0.)
                pf.update(sat, pr, weights, resample=not kf)
                if kf:
                    pf.update_doppler_kf(*doppler[:3], weights=doppler[3], wavelength=.19, sigma_mps=.5)
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
    reference = json.loads((root / "benchmarks/pf_kf_launch_gpu_20261012.json").read_text(encoding="utf-8"))
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
    benchmark_files = ("benchmarks/bench_pf_student_compare.py", "benchmarks/bench_pf_reduction_compare.py",
                       "benchmarks/bench_pf_coalesced_compare.py", "benchmarks/bench_pf_covariance_compare.py",
                       "benchmarks/bench_pf_doppler_compare.py")
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
                baseline="preceding 128-thread cached KF and specialized Gaussian PR with general Student kernel",
                method="actual sequential production workers; alternating build order by round; nonexclusive host",
                scope="synchronized Student PR weights and native/public predict/Student PR/optional KF/ESS/conditional B60 "
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
