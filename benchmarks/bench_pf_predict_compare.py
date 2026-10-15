"""Compare actual specialized PF predictions and complete update steps.

Archive the preceding Gaussian-PR build and five PF sources. Separate
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


def _cloud(count, variant):
    rng = np.random.default_rng(9 + count)
    original = rng.normal(size=(count, 16))
    factors = rng.normal(size=(count, 3, 3))
    original[:, 7:] = (factors @ factors.swapaxes(1, 2) + np.eye(3) * .5).reshape(count, 9)
    if variant == "ecef":
        original[:, :3] += _scenario()[0]
        original[:, 3] += 3000.
    elif variant == "zero_cov":
        original[:, 7:] = 0.
    elif variant == "indefinite_cov":
        original[:, 7:] = [-3., 8., -6., 8., 2., 4., -6., 4., -1.]
    elif variant == "nonsymmetric_cov":
        original[:, 7:] = [4., 1., .5, 9., 3., .25, -7., 8., 2.]
    elif variant == "nonfinite_velocity":
        original[:, 4:7] = np.resize([np.nan, np.inf, -np.inf, -0.], count)[:, None]
    elif variant == "nonfinite_cov":
        original[:, 7:] = np.resize([np.nan, np.inf, -np.inf, -0.], count)[:, None]
    elif variant == "tiny":
        original *= 1e-300
    elif variant == "huge":
        original[:, :7] *= 1e150
        original[:, 7:] *= 1e150
    return original, rng.normal(size=count)


def _parameters():
    # alpha, sampled-velocity sigma, KF, dt, position sigma, clock sigma, Qv, guide
    cases = {}
    for index, alpha in enumerate((-1., -0., 0., .3, np.nextafter(1., 0.), 1., 2.)):
        cases[f"CV_alpha{index}"] = (alpha, 0., False, .17, 1.3, 2.7, .13, (.71, -2.3, 3.14159))
    for index, alpha in enumerate((0., .3, 1., 2.)):
        cases[f"noisy_alpha{index}"] = (alpha, .2, False, .17, 1.3, 2.7, .13, (.71, -2.3, 3.14159))
        for sigma in (0., .2):
            cases[f"KF_alpha{index}_sv{sigma}"] = (alpha, sigma, True, .17, 1.3, 2.7, .13,
                                                  (.71, -2.3, 3.14159))
    for name, case in dict(
        negative_sigma=(.3, -1., False, 0., 1.3, 2.7, 0., (.71, -2.3, 3.14159)),
        negative_Q=(.3, 0., True, .17, 1.3, 2.7, -1., (.71, -2.3, 3.14159)),
        zero_dt=(.3, .2, True, 0., 1.3, 2.7, .13, (.71, -2.3, 3.14159)),
        tiny_noise=(.3, 1e-300, False, 1e-300, 1e-300, 1e-300, 0., (.71, -2.3, 3.14159)),
        huge_noise=(.3, 0., True, .17, 1e150, 1e150, 1e150, (.71, -2.3, 3.14159)),
        signed_zero=(0., 0., False, -0., 1.3, 2.7, -0., (-0., 0., -0.)),
    ).items():
        cases[name] = case
    return cases


def _predict(native, state, case, seed=42, step=1):
    alpha, sigma_vel, kf, dt, sigma_pos, sigma_cb, q, guide = case
    native.pf_device_predict(state, *guide, dt, sigma_pos, sigma_cb, seed, step, sigma_vel, alpha, kf, q)


def _validate(native):
    records = {}
    variants = ("small", "ecef", "zero_cov", "indefinite_cov", "nonsymmetric_cov",
                "nonfinite_velocity", "nonfinite_cov", "tiny", "huge")
    for count in (1, 31, 32, 33, 255, 256, 257, 8193):
        state = native.pf_device_create(count)
        try:
            for variant in variants:
                original, prior = _cloud(count, variant)
                for index, (mode, case) in enumerate(_parameters().items()):
                    seed, step = (2**64 - 1, 13) if index % 2 else (0, 0)
                    native.pf_device_set_particle_states(state, original)
                    native.pf_device_set_log_weights(state, prior)
                    _predict(native, state, case, seed, step)
                    actual, weights = _snapshot(native, state)
                    np.testing.assert_array_equal(weights.view(np.uint64), prior.view(np.uint64))
                    if not case[2]:
                        np.testing.assert_array_equal(actual[:, 7:].view(np.uint64), original[:, 7:].view(np.uint64))
                    records[f"N{count}/{variant}/{mode}"] = dict(
                        input_sha256=_digest(original, prior, np.asarray([*case[:7], *case[7]], dtype=float)),
                        seed=seed, step=step, output_sha256=_digest(actual, weights))
            original, prior = _cloud(count, "small")
            native.pf_device_set_particle_states(state, original)
            native.pf_device_set_log_weights(state, prior)
            history = []
            for index, mode in enumerate(("CV_alpha3", "noisy_alpha1", "KF_alpha1_sv0.2", "CV_alpha5")):
                case = list(_parameters()[mode])
                case[7] = (.71 + index, -2.3 - index, 3.14159 / (index + 1))
                _predict(native, state, case, 42, index + 1)
                # Synchronize before changing the guide in the old pinned-buffer
                # reference; queued changing-guide correctness has its own test.
                history.extend(_snapshot(native, state))
            records[f"N{count}/mixed_modes"] = dict(input_sha256=_digest(original, prior),
                                                    output_sha256=_digest(*history))
        finally:
            native.pf_device_destroy(state)
    for count in (100000, 1000000):
        state = native.pf_device_create(count)
        try:
            for mode in ("CV_alpha3", "noisy_alpha1", "KF_alpha1_sv0.2"):
                for variant in ("small", "ecef"):
                    original, prior = _cloud(count, variant)
                    native.pf_device_set_particle_states(state, original)
                    native.pf_device_set_log_weights(state, prior)
                    _predict(native, state, _parameters()[mode], 2**64 - 1, 13)
                    records[f"N{count}/{variant}/{mode}/large"] = dict(input_sha256=_digest(original, prior),
                        output_sha256=_digest(*_snapshot(native, state)))
        finally:
            native.pf_device_destroy(state)
    _, sat, pr, weights = _scenario()
    doppler = _observations(12)
    for count in (2048, 10000, 100000, 1000000):
        state = native.pf_device_create(count)
        try:
            for mode in ("PR", "PR_noisy", "PR_KF", "PR_high_ESS"):
                _initialize(native, state)
                history = []
                for step in range(1, 6):
                    _predict(native, state, (.3, .2 if mode == "PR_noisy" else 0., mode == "PR_KF",
                                            .1, 1., 3., .1, (1., 2., 3.)), 42, step)
                    _weight(native, state, (sat, pr, weights), 1000. if mode == "PR_high_ESS" else 5.)
                    if mode == "PR_KF":
                        _kf(native, state, doppler)
                    ess = native.pf_device_ess(state)
                    resampled = ess < .5 * count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state, 60, 42 + step)
                    history.append(np.array([ess, resampled, *native.pf_device_estimate(state)]))
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
    modes = dict(default=(1., 0., False), blend=(.3, 0., False), noisy=(.3, .2, False), KF=(.3, 0., True))
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            def predict(mode="default"):
                alpha, sigma_vel, kf = modes[mode]
                _predict(native, state, (alpha, sigma_vel, kf, .1, 1., 3., .1, (1., 2., 3.)))

            for mode in modes:
                measurements[f"N{count}/native_predict_{mode}"] = _measure(native, state,
                    lambda: _initialize(native, state), lambda: predict(mode), repeats)
            imported, _ = _cloud(count, "ecef")
            measurements[f"N{count}/native_predict_KF_imported"] = _measure(native, state,
                lambda: native.pf_device_set_particle_states(state, imported), lambda: predict("KF"), repeats)

            def step(mode="default", sigma=5.):
                predict(mode)
                _weight(native, state, (sat, pr, weights), sigma)
                if mode == "KF":
                    _kf(native, state, doppler)
                ess = native.pf_device_ess(state)
                resampled = ess < .5 * count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return np.array([ess, resampled, *native.pf_device_estimate(state)])

            calls = dict(native_PR_step=step, native_PR_noisy_step=lambda: step("noisy"),
                         native_PR_KF_step=lambda: step("KF"),
                         native_PR_high_ESS_step=lambda: step(sigma=1000.),
                         Gaussian_control=lambda: _weight(native, state, (sat, pr, weights)),
                         resample_control=lambda: native.pf_device_resample_megopolis_coalesced(state, 60, 42),
                         KF_control=lambda: _kf(native, state, doppler))
            for name, call in calls.items():
                measurements[f"N{count}/{name}"] = _measure(native, state,
                    lambda: _initialize(native, state), call, repeats)
        finally:
            native.pf_device_destroy(state)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare_public():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)

            def public_predict(mode="default"):
                alpha, sigma_vel, kf = modes[mode]
                pf.predict([1., 2., 3.], dt=.1, sigma_vel=sigma_vel, velocity_guide_alpha=alpha,
                           rbpf_velocity_kf=kf, velocity_process_noise=.1)

            def public_step(mode="default"):
                public_predict(mode)
                pf.update(sat, pr, weights, resample=mode != "KF")
                if mode == "KF":
                    pf.update_doppler_kf(*doppler[:3], weights=doppler[3], wavelength=.19, sigma_mps=.5)
                return pf.estimate()

            def queued_predicts():
                # The old implementation borrows one pinned guide buffer. Keep
                # the guide identical so this timing comparison is well-defined.
                for _ in range(8):
                    public_predict()

            for mode in ("default", "noisy", "KF"):
                for stage, operation in (("predict", public_predict), ("PR", public_step)):
                    measurements[f"N{count}/public_{stage}_{mode}"] = _measure(native, pf._state,
                        prepare_public, lambda: operation(mode), repeats)
            measurements[f"N{count}/public_queue8_predict"] = _measure(native, pf._state,
                prepare_public, queued_predicts, repeats)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    return dict(validation=_validate(native) if args.verify else {},
                measurements={} if args.validation_only else _profile(native, args.sizes, args.repeats))


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "benchmarks/pf_gaussian_gpu_20261012.json").read_text(encoding="utf-8"))
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
    benchmark_files = ("benchmarks/bench_pf_predict_compare.py", "benchmarks/bench_pf_reduction_compare.py",
                       "benchmarks/bench_pf_coalesced_compare.py", "benchmarks/bench_pf_covariance_compare.py",
                       "benchmarks/bench_pf_doppler_compare.py")
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
                baseline="preceding specialized Gaussian PR with mixed prediction and pinned guide transfer",
                method="actual sequential production workers; alternating build order by round; nonexclusive host",
                scope="synchronized native/public default/noisy/KF prediction and predict/PR/optional KF/ESS/conditional B60 "
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
