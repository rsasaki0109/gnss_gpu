"""Compare separate resident queries with combined LOS/reflection retrieval.

Load two actual builds in one process and alternate synchronized whole calls.
Urban timing also includes Python channel construction and real CUDA IQ output.
CPU BVH construction is excluded; cold native timing includes scene lifetime.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _validation_cases
from bench_bvh_resident_compare import _assert_outputs, _hash_files, _measure, _timing_cases
from bench_raytrace import _make_building_mesh, _make_satellites


def _separate(scene, rx, sat):
    return scene.check_los_batch(rx, sat), *scene.multipath_batch(rx, sat)


def _check_case(before, after, name, arguments):
    rx, sat, nodes, triangles = arguments
    input_hash = _digest(*arguments)
    expected = _separate(before, rx, sat)
    actual = after.trace_batch(rx, sat)
    np.testing.assert_array_equal(actual[0], expected[0], err_msg=name)
    _assert_outputs(expected[1:], actual[1:], name)
    # The refactored separate methods must also retain the preceding output.
    separate = _separate(after, rx, sat)
    np.testing.assert_array_equal(separate[0], expected[0], err_msg=name)
    _assert_outputs(expected[1:], separate[1:], name)
    single_hash = None
    if np.all(np.isfinite(rx[0])) and np.all(np.isfinite(sat[0])):
        old_points, old_delays = before.multipath_batch(rx[:1], sat[:1])
        _assert_outputs((expected[1][0], expected[2][0]), (old_points[0], old_delays[0]), name)
        single = after.trace(rx[0], sat[0])
        np.testing.assert_array_equal(single[0], before.check_los(rx[0], sat[0]), err_msg=name)
        _assert_outputs((old_points[0], old_delays[0]), single[1:], name)
        single_hash = _digest(*single)
    assert _digest(*arguments) == input_hash, f"Input mutated: {name}"
    return {"input_sha256": input_hash, "baseline_sha256": _digest(*expected),
            "candidate_sha256": _digest(*actual), "single_sha256": single_hash,
            "nonzero_delays": int(np.count_nonzero(expected[2])),
            "blocked_rays": int(np.count_nonzero(~expected[0]))}


def _validate(before, after):
    records = {}
    for name, arguments in _validation_cases(before):
        old, new = before.DeviceBVH(*arguments[2:]), after.DeviceBVH(*arguments[2:])
        try:
            records[name] = _check_case(old, new, name, arguments)
        finally:
            old.close()
            new.close()
        assert old.allocated_bytes == new.allocated_bytes == 0
    nodes, triangles = _build(before, _make_building_mesh(100))
    old, new = before.DeviceBVH(nodes, triangles), after.DeviceBVH(nodes, triangles)
    retained = []
    try:
        for index, (epochs, satellites) in enumerate(
                ((1, 4097), (1, 12), (3, 7), (1, 257), (16, 256), (1, 1), (2, 5), (1, 12), (1, 1))):
            rx, sat = _rays(epochs, satellites, seed=100 + index, near=True)
            name = f"reuse/{index}/epochs{epochs}/satellites{satellites}"
            records[name] = _check_case(old, new, name, (rx, sat, nodes, triangles))
            outputs = new.trace_batch(rx, sat)
            retained.append((outputs, _digest(*outputs)))
            for outputs, digest in retained:
                assert _digest(*outputs) == digest, "Later calls overwrote returned outputs"
        rx, sat = _rays(1, 12, near=True)
        nodes.fill(0.)
        triangles.fill(0.)
        records["owned_snapshot"] = _check_case(old, new, "owned_snapshot", (rx, sat, nodes, triangles))
    finally:
        old.close()
        new.close()
    assert old.allocated_bytes == new.allocated_bytes == 0
    return {"cases": len(records), "bit_identical": True, "records": records}


def _measure_case(before, after, arguments, rounds, repeats):
    rx, sat, nodes, triangles = arguments
    old, new = before.DeviceBVH(nodes, triangles), after.DeviceBVH(nodes, triangles)

    def cold(native, combined):
        scene = native.DeviceBVH(nodes, triangles)
        try:
            return scene.trace_batch(rx, sat) if combined else _separate(scene, rx, sat)
        finally:
            scene.close()

    try:
        validation = _check_case(old, new, "timing", arguments)
        measurements = {
            "warm_batch": _measure(lambda: _separate(old, rx, sat),
                                   lambda: new.trace_batch(rx, sat), rounds, repeats),
            "cold_batch": _measure(lambda: cold(before, False),
                                   lambda: cold(after, True), rounds, repeats),
        }
        if len(rx) == 1:
            measurements["warm_single"] = _measure(
                lambda: (old.check_los(rx[0], sat[0]), old.multipath_batch(rx, sat)),
                lambda: new.trace(rx[0], sat[0]), rounds, repeats)
        assert old.allocated_bytes == new.allocated_bytes, "Different workspace sizes for the same query sequence"
        measurements["allocated_bytes"] = {"baseline": old.allocated_bytes, "candidate": new.allocated_bytes}
        measurements["validation"] = validation
        return measurements
    finally:
        old.close()
        new.close()


def _load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_tree(old, new):
    if isinstance(old, np.ndarray):
        assert old.dtype == new.dtype and old.shape == new.shape
        np.testing.assert_array_equal(new.view(np.uint8), old.view(np.uint8))
    elif isinstance(old, dict):
        assert old.keys() == new.keys()
        for key in old:
            _assert_tree(old[key], new[key])
    elif isinstance(old, list):
        assert len(old) == len(new)
        for previous, current in zip(old, new, strict=True):
            _assert_tree(previous, current)
    else:
        assert old == new


def _tree_hash(tree):
    def encode(value):
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _digest(value)}
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return hashlib.sha256(json.dumps(encode(tree), sort_keys=True).encode()).hexdigest()


def _measure_urban(before, after, baseline_dir, root, rounds, repeats):
    old_bvh = _load_python(baseline_dir / "bvh.py", "baseline_bvh_wrapper")
    new_bvh = _load_python(root / "python/gnss_gpu/bvh.py", "candidate_bvh_wrapper")
    old_urban = _load_python(baseline_dir / "urban_signal_sim.py", "baseline_urban")
    new_urban = _load_python(root / "python/gnss_gpu/urban_signal_sim.py", "candidate_urban")
    records = {}
    rx = np.array([6378137., 0., 0.])
    sat = _make_satellites(12)[:, [2, 0, 1]] + rx
    for count in (12, 1200, 12000):
        triangles = _make_building_mesh(count // 12)[:, :, [2, 0, 1]] + rx
        old_model, new_model = old_bvh.BVHAccelerator(triangles), new_bvh.BVHAccelerator(triangles)
        old_model._device_scene = before.DeviceBVH(old_model._nodes_flat, old_model._sorted_tris)
        new_model._device_scene = after.DeviceBVH(new_model._nodes_flat, new_model._sorted_tris)
        old_sim, new_sim = old_urban.UrbanSignalSimulator(old_model), new_urban.UrbanSignalSimulator(new_model)
        old_sim.sim.noise_seed = new_sim.sim.noise_seed = 37
        try:
            for samples in (2600, 26000):
                def old_call():
                    return old_sim.compute_epoch(rx, sat, n_samples=samples)

                def new_call():
                    return new_sim.compute_epoch(rx, sat, n_samples=samples)

                expected, actual = old_call(), new_call()
                _assert_tree(expected, actual)
                assert np.all(expected["visible"]), "Timed geometry must have 12 visible satellites"
                name = f"triangles{count}/rays12/samples{samples}"
                records[name] = _measure(old_call, new_call, rounds, repeats)
                records[name]["validation"] = {
                    "input_sha256": _digest(rx, sat, triangles),
                    "baseline_sha256": _tree_hash(expected), "candidate_sha256": _tree_hash(actual),
                    "iq_sha256": _digest(expected["iq"]), "channels": len(expected["channels"]),
                    "nonzero_delays": int(np.count_nonzero(expected["excess_delays"])),
                    "blocked_rays": int(np.count_nonzero(~expected["is_los"])),
                }
        finally:
            old_model.release_gpu()
            new_model.release_gpu()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    root = Path(__file__).resolve().parents[1]
    source_paths = ["src/raytrace/bvh.cu", "include/gnss_gpu/bvh.h", "python/gnss_gpu/_bvh_bindings.cpp",
                    "python/gnss_gpu/bvh.py", "python/gnss_gpu/urban_signal_sim.py"]
    archive_hashes = _hash_files(args.baseline_dir, [Path(path).name for path in source_paths])
    extension_hashes = {"baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                        "candidate": hashlib.sha256(args.candidate.read_bytes()).hexdigest()}
    reference = json.loads((root / "benchmarks/bvh_resident_gpu_20261011.json").read_text())
    assert extension_hashes["baseline"] == reference["extension_sha256"]["candidate"], "Wrong baseline binary"
    for path, digest in reference["source_sha256"]["candidate"].items():
        assert archive_hashes[Path(path).name] == digest, f"Wrong baseline source: {path}"
    before = _load_extension(args.baseline.resolve(), "baseline._bvh")
    after = _load_extension(args.candidate.resolve(), "candidate._bvh")
    # Wrappers import the current CPU builder. Reuse this loaded candidate
    # instead of loading the same shared library again under a third name.
    sys.modules["gnss_gpu._bvh"] = after
    validation = _validate(before, after)
    print(f"{validation['cases']} combined/separate cases are bit-identical", flush=True)
    measurements, urban = {}, {}
    if not args.validation_only:
        for name, arguments in _timing_cases(before):
            measurements[name] = _measure_case(before, after, arguments, args.rounds, args.repeats)
            for stage, record in measurements[name].items():
                if "speedup" in record:
                    print(f"{name}/{stage}: {record['baseline_ms']:.4f} -> "
                          f"{record['candidate_ms']:.4f} ms ({record['speedup']:.2f}x)", flush=True)
        urban = _measure_urban(before, after, args.baseline_dir, root, args.rounds, args.repeats)
        for name, record in urban.items():
            print(f"urban/{name}: {record['baseline_ms']:.4f} -> "
                  f"{record['candidate_ms']:.4f} ms ({record['speedup']:.2f}x)", flush=True)
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "baseline": "preceding resident BVH implementation, separate LOS and multipath calls",
        "method": "same process, alternating call order, synchronized whole-call wall time; nonexclusive host",
        "scope": "native validation/transfers/kernels/output included; cold also construction/close; "
                 "urban includes Python wrappers/channels and real CUDA IQ, same signal build; CPU BVH build excluded",
        "rounds": args.rounds, "repeats": args.repeats, "warmup_calls_per_build": 3,
        "source_sha256": {"baseline": archive_hashes, "candidate": _hash_files(root, source_paths)},
        "extension_sha256": extension_hashes,
        "signal_extension_sha256": _hash_files(root, [path.relative_to(root) for path in
            (root / "python/gnss_gpu").glob("_gnss_gpu_signal_sim*.pyd")]),
        "benchmark_sha256": _hash_files(root, ["benchmarks/bench_bvh_combined_compare.py",
            "benchmarks/bench_bvh_resident_compare.py", "benchmarks/bench_bvh_multipath_compare.py",
            "benchmarks/bench_raytrace.py"]),
        "validation": validation, "measurements": measurements, "urban_measurements": urban,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
