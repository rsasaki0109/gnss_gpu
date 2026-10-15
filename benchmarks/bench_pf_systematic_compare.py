"""Compare systematic weight preparation and complete PF steps in two builds.

Each build runs in its own process because PF handles use a global pybind type.
Build order alternates by round. Every timed operation includes stream sync;
state/weight restoration and diagnostics are outside the measured interval.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from bench_pf_coalesced_compare import _initialize, _load_extension, _scenario


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _distributions(count):
    rng = np.random.default_rng(count)
    normal = rng.normal(0., 1.5, count)
    dominant = np.full(count, -1000.)
    dominant[count // 2] = 0.
    sparse = np.full(count, -np.inf)
    sparse[::17] = 0.
    underflow = np.resize(np.array([0., -1., -708., -744., -745., -746., -1000.]), count)
    result = {"uniform": np.zeros(count), "normal": normal, "dominant": dominant,
              "sparse": sparse, "underflow": underflow}
    if count <= 65537:
        result.update({"shift_positive": normal + 1e6, "shift_negative": normal - 1e6,
                       "tied": np.resize(np.array([0., -1., -2., -1.]), count),
                       "all_negative_infinity": np.full(count, -np.inf)})
    return result


def _snapshot(native, state, ancestors=False):
    particles = native.pf_device_get_particle_states(state)
    weights = np.empty(state.n_particles)
    native.pf_device_get_log_weights(state, weights)
    arrays = [particles, weights]
    if ancestors:
        indices = np.empty(state.n_particles, dtype=np.int32)
        native.pf_device_get_resample_ancestors(state, indices)
        arrays.append(indices)
    return arrays


def _native_step(native, state, systematic=True, step=1, observations=None):
    satellites, ranges, weights = _scenario()[1:] if observations is None else observations
    native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, step)
    native.pf_device_weight(state, satellites, ranges, weights, 12, 5.)
    ess = native.pf_device_ess(state)
    resampled = ess < .5 * state.n_particles
    if resampled:
        if systematic:
            native.pf_device_resample_systematic(state, 42)
        else:
            native.pf_device_resample_megopolis_coalesced(state, 60, 42)
    estimate = native.pf_device_estimate(state)
    return estimate, np.array([ess, float(resampled)])


def _load_probe(path):
    spec = importlib.util.spec_from_file_location("_pf_systematic_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scan_repeatability(native, probe):
    records = {}
    for count in (65535, 1_000_000):
        state = native.pf_device_create(count)
        try:
            _initialize(native, state)
            for distribution in ("uniform", "sparse"):
                lw = _distributions(count)[distribution]
                previous = None
                samples = []
                for _ in range(8):
                    native.pf_device_set_log_weights(state, lw)
                    native.pf_device_resample_systematic(state, 0)
                    weights, cdf, max_sum = probe.read(state)
                    ancestors = np.empty(count, dtype=np.int32)
                    native.pf_device_get_resample_ancestors(state, ancestors)
                    if previous is None:
                        previous = (cdf.copy(), ancestors.copy(), _digest(weights, max_sum))
                    assert _digest(weights, max_sum) == previous[2], "Weight preparation varied within one build"
                    samples.append({"cdf_sha256": _digest(cdf), "ancestors_sha256": _digest(ancestors),
                        "different_ancestors_from_first": int(np.count_nonzero(ancestors != previous[1])),
                        "max_index_change_from_first": int(np.max(np.abs(ancestors - previous[1]))),
                        "max_abs_cdf_change_from_first": float(np.max(np.abs(cdf - previous[0])))})
                records[f"N{count}/{distribution}/seed0"] = samples
        finally:
            native.pf_device_destroy(state)
    return records


def _validate(native, probe):
    records = {}
    sizes = (1, 2, 31, 32, 33, 255, 256, 257, 511, 512, 513,
             8191, 8192, 8193, 65535, 65536, 65537, 100_000, 1_000_000)
    for count in sizes:
        state = native.pf_device_create(count)
        rng = np.random.default_rng(count)
        original = rng.normal(size=(count, 16))
        original[:, 0] = np.arange(count)
        original[::17, 4] = -0.0
        seeds = (0, 1, 42, 999999, 1000000, 2**64 - 1) if count <= 65537 else (0, 42, 2**64 - 1)
        try:
            for name, log_weights in _distributions(count).items():
                for seed in seeds:
                    native.pf_device_set_particle_states(state, original)
                    native.pf_device_set_log_weights(state, log_weights)
                    native.pf_device_resample_systematic(state, seed)
                    outputs = _snapshot(native, state, ancestors=True)
                    normalized, cdf, max_sum = probe.read(state)
                    np.testing.assert_array_equal(outputs[0].view(np.uint64),
                                                  original[outputs[2]].view(np.uint64))
                    np.testing.assert_array_equal(outputs[1], np.zeros(count))
                    cdf_diagnostic = None
                    if seed % 1000000 == 0 and np.all(np.isfinite(cdf)):
                        # The existing parallel scan can also decrease by an
                        # ulp on sparse plateaus. Record that separately; a
                        # monotone-array CPU search is not an oracle there.
                        expected = np.searchsorted(cdf, np.arange(count) / count, side="left")
                        expected = np.minimum(expected, count - 1)
                        cdf_steps = np.diff(cdf)
                        cdf_diagnostic = {
                            "cpu_monotone_search_different_ancestors": int(np.count_nonzero(outputs[2] != expected)),
                            "decreasing_steps": int(np.count_nonzero(cdf_steps < 0.)),
                            "minimum_step": float(np.min(cdf_steps)) if count > 1 else 0.,
                        }
                    key = f"N{count}/{name}/seed{seed}"
                    records[key] = {"input_sha256": _digest(original, log_weights),
                                    "output_sha256": _digest(*outputs),
                                    "normalized_weights_sha256": _digest(normalized),
                                    "max_and_sum_sha256": _digest(max_sum), "cdf_sha256": _digest(cdf),
                                    "zero_offset_cdf_diagnostic": cdf_diagnostic}
            _initialize(native, state)
            history = []
            for step in range(1, 4):
                history.extend(_native_step(native, state, step=step))
            records[f"N{count}/three_systematic_epochs"] = {
                "output_sha256": _digest(*history, *_snapshot(native, state, ancestors=True))}
            native.pf_device_set_particle_states(state, original)
            native.pf_device_set_log_weights(state, rng.normal(size=count))
            native.pf_device_resample_systematic(state, 42)
            # Reuse the workspace and swap pointers before requesting host data.
            native.pf_device_resample_systematic(state, 999999)
            native.pf_device_resample_megopolis_coalesced(state, 15, 42)
            native.pf_device_set_log_weights(state, rng.normal(size=count))
            native.pf_device_resample_megopolis(state, 3, 1)
            native.pf_device_resample_systematic(state, 1)
            records[f"N{count}/mixed_and_chained_resamplers"] = {"output_sha256": _digest(*_snapshot(native, state))}
        finally:
            native.pf_device_destroy(state)
    return records


def _measure(native, state, prepare, operation, repeats, ancestors):
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
    arrays = _snapshot(native, state, ancestors=ancestors)
    if output is not None:
        arrays.extend(output if isinstance(output, tuple) else (output,))
    return {"median_ms": float(np.median(samples)), "samples_ms": samples,
            "output_sha256": _digest(*arrays)}


def _profile(native, sizes, repeats):
    # Route the unchanged public wrapper to this worker's actual build.
    sys.modules["gnss_gpu._gnss_gpu_pf_device"] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    measurements = {}
    position, satellites, ranges, weights = _scenario()
    for count in sizes:
        state = native.pf_device_create(count)
        try:
            distributions = _distributions(count)
            for distribution in ("uniform", "normal", "dominant"):
                log_weights = distributions[distribution]

                def prepare():
                    _initialize(native, state)
                    native.pf_device_set_log_weights(state, log_weights)

                key = f"N{count}/resample_{distribution}"
                measurements[key] = _measure(native, state, prepare,
                    lambda: native.pf_device_resample_systematic(state, 42), repeats, True)
                measurements[key]["input_sha256"] = _digest(log_weights)
            for systematic in (True, False):
                name = "systematic_step" if systematic else "megopolis_control_step"
                measurements[f"N{count}/{name}"] = _measure(native, state,
                    lambda: _initialize(native, state),
                    lambda: _native_step(native, state, systematic, observations=(satellites, ranges, weights)),
                    repeats, systematic)
        finally:
            native.pf_device_destroy(state)
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3.,
                                  sigma_pr=5., resampling="systematic", seed=42)
        try:
            def prepare_wrapper():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.)

            def wrapper_step():
                pf.predict([1., 2., 3.], dt=.1)
                pf.update(satellites, ranges, weights)
                return pf.estimate()

            measurements[f"N{count}/public_predict_update_estimate"] = _measure(
                native, pf._state, prepare_wrapper, wrapper_step, repeats, True)
        finally:
            native.pf_device_destroy(pf._state)
    return measurements


def _worker(args):
    native = _load_extension(args.extension.resolve())
    probe = _load_probe(args.probe.resolve()) if args.verify else None
    return {"validation": _validate(native, probe) if args.verify else {},
            "scan_repeatability": _scan_repeatability(native, probe) if args.verify else {},
            "measurements": {} if args.validation_only else _profile(native, args.sizes, args.repeats)}


def _hash_files(root, paths):
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    extensions = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    archive_hashes = _hash_files(args.baseline_dir,
        ["pf_device.cu", "pf_device.h", "_pf_device_bindings.cpp", "pf_device_runtime.py", "pf_device_config.py"])
    reference = json.loads((root / "benchmarks/pf_gather_gpu_20261011.json").read_text())
    assert archive_hashes["pf_device.cu"] == reference["source_sha256"]["candidate"], "Wrong preceding source"
    assert extensions["baseline"] == reference["extension_sha256"]["candidate"], "Wrong preceding binary"
    runs = {"baseline": [], "candidate": []}
    for round_index in range(args.rounds):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for label in order:
            command = [sys.executable, str(Path(__file__).resolve()), "--extension", str(paths[label].resolve()),
                       "--probe", str(args.probe.resolve()),
                       "--repeats", str(args.repeats), "--sizes", *map(str, args.sizes)]
            if round_index == 0:
                command.append("--verify")
            if args.validation_only:
                command.append("--validation-only")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"{label} worker failed:\n{result.stderr}")
            worker_path = args.output.with_name(f"{args.output.stem}_{label}_round{round_index + 1}.json")
            worker_path.write_text(result.stdout, encoding="utf-8")
            runs[label].append(json.loads(result.stdout))
            print(f"round {round_index + 1}/{args.rounds}: {label} complete", flush=True)
        if round_index == 0:
            before, after = [runs[label][0]["validation"] for label in runs]
            assert before.keys() == after.keys()
            boundary_cases = []
            for key, previous in before.items():
                current = after[key]
                for field in ("input_sha256", "normalized_weights_sha256", "max_and_sum_sha256"):
                    if field in previous:
                        assert previous[field] == current[field], f"Weight/input mismatch: {key}/{field}"
                if previous["output_sha256"] != current["output_sha256"]:
                    # The unchanged parallel scan is not bit reproducible on
                    # exact boundaries. Report those exceptions explicitly.
                    assert "cdf_sha256" in previous and previous["cdf_sha256"] != current["cdf_sha256"], key
                    assert any(f"/{family}/" in key for family in ("uniform", "sparse")), key
                    assert key.endswith(("/seed0", "/seed1000000")), key
                    boundary_cases.append(key)
            print(f"{len(before) - len(boundary_cases)}/{len(before)} state outputs match; "
                  f"{len(boundary_cases)} parallel-scan boundary exceptions", flush=True)
    summary = {}
    for key in runs["baseline"][0]["measurements"]:
        outputs = {run["measurements"][key]["output_sha256"] for data in runs.values() for run in data}
        assert len(outputs) == 1, f"Timed output mismatch across builds/rounds: {key}"
        medians = {label: [run["measurements"][key]["median_ms"] for run in data] for label, data in runs.items()}
        old, new = [float(np.median(values)) for values in medians.values()]
        summary[key] = {"baseline_ms": old, "candidate_ms": new, "speedup": old / new,
                        "round_medians_ms": medians, "output_sha256": next(iter(outputs))}
        print(f"{key}: {old:.4f} -> {new:.4f} ms ({old / new:.2f}x)", flush=True)
    return {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "baseline": "preceding coalesced covariance gather implementation",
        "method": "separate processes; alternating build order per round; synchronized wall time; nonexclusive host",
        "scope": "resample or complete native/public PF step; state/weight restoration and dumps excluded; "
                 "native/public steps use ESS-based adaptive resampling",
        "rounds": args.rounds, "repeats": args.repeats, "warmup_calls_per_stage": 3,
        "source_sha256": {"baseline": archive_hashes, "candidate": _hash_files(root,
            ["src/particle_filter/pf_device.cu", "include/gnss_gpu/pf_device.h", "python/gnss_gpu/_pf_device_bindings.cpp",
             "python/gnss_gpu/pf_device_runtime.py", "python/gnss_gpu/pf_device_config.py"])},
        "extension_sha256": extensions,
        "benchmark_sha256": _hash_files(root, ["benchmarks/bench_pf_systematic_compare.py",
            "benchmarks/bench_pf_coalesced_compare.py", "benchmarks/probes/pf_systematic_probe.cpp",
            "benchmarks/probes/CMakeLists.txt"]),
        "probe_sha256": hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        "validation": {"cases": len(before), "all_state_outputs_bit_identical": not boundary_cases,
            "state_output_bit_identical_cases": len(before) - len(boundary_cases),
            "normalized_weights_and_sums_bit_identical": True,
            "normalized_weight_cases": sum("normalized_weights_sha256" in record for record in before.values()),
            "parallel_scan_boundary_exceptions": boundary_cases, "baseline": before, "candidate": after,
            "scan_repeatability": {label: runs[label][0]["scan_repeatability"] for label in runs}},
        "summary": summary,
        "runs": {label: [run["measurements"] for run in data] for label, data in runs.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--extension", type=Path, help="Run one isolated worker")
    parser.add_argument("--probe", type=Path, required=True, help="Read-only benchmark probe extension")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2048, 10_000, 100_000, 1_000_000])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or min(args.sizes) < 1:
        parser.error("rounds, repeats and sizes must be positive")
    if not args.extension and not (args.baseline and args.candidate and args.baseline_dir and args.output):
        parser.error("comparison requires --baseline, --candidate, --baseline-dir and --output")
    report = _worker(args) if args.extension else _compare(args)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
