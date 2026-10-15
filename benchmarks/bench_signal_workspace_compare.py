"""Compare the preceding fused IQ generator with reusable per-simulator buffers.

Actual production binaries run sequentially in alternating order. Native,
public-wrapper, and urban timings include synchronized host IQ retrieval.
Cold controls retain the standalone per-call API. Code-table setup and CPU BVH
construction are excluded. Hash every timed output outside the timed interval.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from bench_bvh_combined_compare import _assert_tree, _tree_hash
from bench_raytrace import _make_building_mesh, _make_satellites
from bench_signal_sim_compare import (
    _channels, _digest, _input_digest, _load_extension, _timing_cases, _validation_cases,
)

SOURCE_PATHS = (
    "src/signal_sim/signal_sim.cu", "include/gnss_gpu/signal_sim.h",
    "python/gnss_gpu/_signal_sim_bindings.cpp", "python/gnss_gpu/signal_sim.py",
    "python/gnss_gpu/urban_signal_sim.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _activate(native):
    import gnss_gpu

    sys.modules["gnss_gpu._gnss_gpu_signal_sim"] = native
    gnss_gpu._gnss_gpu_signal_sim = native


def _validate(before, after):
    records = {}
    workspace = after.DeviceSignalWorkspace()
    retained = []
    try:
        for name, arguments in _validation_cases():
            input_hash = _input_digest(arguments)
            old = before.generate_signal(*arguments)
            new = workspace.generate_signal(*arguments)
            cold = after.generate_signal(*arguments)
            for actual in (new, cold):
                np.testing.assert_array_equal(actual.view(np.uint32), old.view(np.uint32),
                                              err_msg=name)
            assert _input_digest(arguments) == input_hash, f"Input mutated: {name}"
            records[name] = dict(input_sha256=input_hash, baseline_sha256=_digest(old),
                                 candidate_sha256=_digest(new), cold_sha256=_digest(cold),
                                 iq_floats=int(old.size))
        for index, (samples, count, nav) in enumerate((
                (1, 1, False), (8193, 65, True), (2, 32, False), (260000, 33, True),
                (257, 0, False), (2600, 12, True), (1, 33, True), (26000, 65, False))):
            arguments = (4.092e6, -1e6, _channels(count, nav), samples, -20., 37)
            input_hash = _input_digest(arguments)
            old, new = before.generate_signal(*arguments), workspace.generate_signal(*arguments)
            np.testing.assert_array_equal(new.view(np.uint32), old.view(np.uint32))
            records[f"reuse/{index}"] = dict(input_sha256=input_hash,
                                              baseline_sha256=_digest(old),
                                              candidate_sha256=_digest(new),
                                              iq_floats=int(old.size))
            retained.append((new, _digest(new)))
            for channel in arguments[2]:
                if "nav_bits" in channel:
                    channel["nav_bits"].fill(0)
            for output, digest in retained:
                assert _digest(output) == digest, "Returned array overwritten"
        allocated = workspace.allocated_bytes
    finally:
        workspace.close()
    assert workspace.allocated_bytes == 0
    for output, digest in retained:
        assert _digest(output) == digest
    return dict(cases=len(records), bit_identical=True, records=records,
                retained_device_bytes_before_close=allocated, device_bytes_after_close=0)


def _measure(old_call, new_call, rounds, repeats):
    calls = {"baseline": old_call, "candidate": new_call}
    expected, actual = old_call(), new_call()
    _assert_tree(expected, actual)
    digest = _tree_hash(expected)
    for call in calls.values():
        for _ in range(3):
            assert _tree_hash(call()) == digest
    samples = {label: [] for label in calls}
    for round_index in range(rounds):
        timings = {label: [] for label in calls}
        for repetition in range(repeats):
            order = ("baseline", "candidate") if (round_index + repetition) % 2 == 0 else (
                "candidate", "baseline")
            for label in order:
                start = time.perf_counter_ns()
                output = calls[label]()
                timings[label].append((time.perf_counter_ns() - start) / 1e6)
                assert _tree_hash(output) == digest, f"Timed output changed: {label}"
        for label in samples:
            samples[label].append(timings[label])
    medians = {label: [float(np.median(sample)) for sample in values]
               for label, values in samples.items()}
    old, new = [float(np.median(medians[label])) for label in calls]
    return dict(baseline_ms=old, candidate_ms=new, speedup=old / new,
                round_medians_ms=medians, samples_ms=samples, output_sha256=digest,
                all_timed_outputs_identical=True)


def _native(before, after, rounds, repeats):
    measurements = {}
    cases = list(_timing_cases())
    cases += [(f"samples{samples}/channels33", (2.6e6, 0., _channels(33), samples, -20., 42))
              for samples in (2600, 260000)]
    for name, arguments in cases:
        workspace = after.DeviceSignalWorkspace()
        try:
            record = _measure(lambda: before.generate_signal(*arguments),
                              lambda: workspace.generate_signal(*arguments), rounds, repeats)
            record["input_sha256"] = _input_digest(arguments)
            record["retained_device_bytes"] = workspace.allocated_bytes
            measurements[f"native/warm/{name}"] = record
            _print(f"native/warm/{name}", record)
        finally:
            workspace.close()
        assert workspace.allocated_bytes == 0
    for samples in (2600, 260000):
        for count, nav in ((12, False), (33, True)):
            args = (2.6e6, 0., _channels(count, nav), samples, -20., 42)
            record = _measure(lambda: before.generate_signal(*args),
                              lambda: after.generate_signal(*args), rounds, repeats)
            record["input_sha256"] = _input_digest(args)
            name = f"native/cold/samples{samples}/channels{count}/nav{nav}"
            measurements[name] = record
            _print(name, record)
    return measurements


def _wrapper_call(native, simulator, channels, samples):
    _activate(native)
    return simulator.generate_epoch(channels, samples)


def _public(before, after, old_module, new_module, rounds, repeats):
    measurements = {}
    cases = [(samples, count, False) for samples in (2600, 26000, 260000, 2600000)
             for count in (0, 12, 33)]
    cases += [(samples, 65, True) for samples in (2600, 260000)]
    for samples, count, nav in cases:
        old = old_module.SignalSimulator(noise_seed=42)
        new = new_module.SignalSimulator(noise_seed=42)
        channels = _channels(count, nav)
        try:
            record = _measure(lambda: _wrapper_call(before, old, channels, samples),
                              lambda: _wrapper_call(after, new, channels, samples), rounds, repeats)
            record["input_sha256"] = _input_digest((2.6e6, 0., channels, samples, -20., 42))
            record["retained_device_bytes"] = new._workspace.allocated_bytes
            name = f"public/warm/samples{samples}/channels{count}/nav{nav}"
            measurements[name] = record
            _print(name, record)
        finally:
            new.release_gpu()
    return measurements


def _urban(before, after, old_module, new_module, baseline_dir, root, rounds, repeats):
    from gnss_gpu.bvh import BVHAccelerator

    old_urban = _load_python(baseline_dir / "urban_signal_sim.py", "baseline_urban")
    new_urban = _load_python(root / "python/gnss_gpu/urban_signal_sim.py", "candidate_urban")
    old_urban.SignalSimulator, new_urban.SignalSimulator = (
        old_module.SignalSimulator, new_module.SignalSimulator)
    rx = np.array([6378137., 0., 0.])
    sat = _make_satellites(12)[:, [2, 0, 1]] + rx
    measurements = {}
    for triangles_count in (12, 1200, 12000):
        triangles = _make_building_mesh(triangles_count // 12)[:, :, [2, 0, 1]] + rx
        old_model, new_model = BVHAccelerator(triangles), BVHAccelerator(triangles)
        old, new = old_urban.UrbanSignalSimulator(old_model), new_urban.UrbanSignalSimulator(new_model)
        old.sim.noise_seed = new.sim.noise_seed = 37
        try:
            for samples in (2600, 26000, 260000):
                def old_call():
                    _activate(before)
                    return old.compute_epoch(rx, sat, n_samples=samples)

                def new_call():
                    _activate(after)
                    return new.compute_epoch(rx, sat, n_samples=samples)

                expected, actual = old_call(), new_call()
                _assert_tree(expected, actual)
                assert np.all(expected["visible"]), "Need 12 visible satellites"
                record = _measure(old_call, new_call, rounds, repeats)
                record.update(input_sha256=_input_digest((rx, sat, triangles)),
                              iq_sha256=_digest(expected["iq"]), channels=len(expected["channels"]),
                              blocked_rays=int(np.count_nonzero(~expected["is_los"])),
                              nonzero_delays=int(np.count_nonzero(expected["excess_delays"])),
                              retained_device_bytes=new.sim._workspace.allocated_bytes)
                name = f"urban/warm/triangles{triangles_count}/rays12/samples{samples}"
                measurements[name] = record
                _print(name, record)
        finally:
            new.sim.release_gpu()
            old_model.release_gpu()
            new_model.release_gpu()
    return measurements


def _print(name, record):
    print(f"{name}: {record['baseline_ms']:.4f} -> {record['candidate_ms']:.4f} ms "
          f"({record['speedup']:.3f}x)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    baseline_hashes = {str(path): _sha(args.baseline_dir / Path(path).name) for path in SOURCE_PATHS}
    candidate_hashes = {str(path): _sha(root / path) for path in SOURCE_PATHS}
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    reference = json.loads((root / "benchmarks/signal_fused_gpu_20261011.json").read_text())
    assert baseline_hashes[SOURCE_PATHS[0]] == reference["source_sha256"]["candidate"]
    assert extensions["baseline"] == reference["extension_sha256"]["candidate"]
    assert baseline_hashes[SOURCE_PATHS[-1]] == candidate_hashes[SOURCE_PATHS[-1]]
    before = _load_extension(args.baseline.resolve(), "baseline._gnss_gpu_signal_sim")
    after = _load_extension(args.candidate.resolve(), "candidate._gnss_gpu_signal_sim")
    # Reuse the loaded binary when the package is imported; don't register the
    # same pybind class a second time under a third module name.
    sys.modules["gnss_gpu._gnss_gpu_signal_sim"] = after
    _activate(after)
    validation = _validate(before, after)
    print(f"{validation['cases']} old/reusable cases bit-identical; 263 cold cases also match", flush=True)
    measurements = {}
    if not args.validation_only:
        old_module = _load_python(args.baseline_dir / "signal_sim.py", "baseline_signal_wrapper")
        new_module = _load_python(root / "python/gnss_gpu/signal_sim.py", "candidate_signal_wrapper")
        measurements.update(_native(before, after, args.rounds, args.repeats))
        measurements.update(_public(before, after, old_module, new_module, args.rounds, args.repeats))
        measurements.update(_urban(before, after, old_module, new_module, args.baseline_dir,
                                   root, args.rounds, args.repeats))
    import gnss_gpu._bvh as bvh_native

    report = dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
        method="same process; sequential alternating actual builds per call; nonexclusive host",
        scope="native conversion/uploads/kernel/device synchronization/host output included; "
              "public adds Python validation; urban adds geometry and channel construction; "
              "code-table initialization and CPU BVH construction excluded; hash checks outside timing; "
              "wrapper backend selection for comparison included symmetrically",
        baseline="preceding fused signal implementation; one-shot allocation/free per epoch",
        rounds=args.rounds, repeats=args.repeats, warmup_calls_per_build=3,
        source_sha256=dict(baseline=baseline_hashes, candidate=candidate_hashes),
        extension_sha256=extensions,
        bvh_extension_sha256=_sha(Path(bvh_native.__file__)),
        benchmark_sha256={path: _sha(root / path) for path in (
            "benchmarks/bench_signal_workspace_compare.py", "benchmarks/bench_signal_sim_compare.py",
            "benchmarks/bench_bvh_combined_compare.py", "benchmarks/bench_bvh_resident_compare.py",
            "benchmarks/bench_bvh_multipath_compare.py", "benchmarks/bench_raytrace.py")},
        validation=validation, measurements=measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
