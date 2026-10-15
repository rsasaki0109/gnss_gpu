"""Compare two BVH extensions: exact outputs and alternating native-call timing.

Build/copy the baseline before editing bvh.cu, then rebuild the candidate using
the same configuration. Native calls include allocation, uploads, synchronization,
output copies, and frees. CPU BVH construction is outside the measured interval.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from bench_raytrace import _make_building_mesh, _make_satellites


def _load_extension(path, name):
    handles = []
    if os.name == "nt":
        directories = [path.parent]
        for variable in ("CUDA_PATH", "CUDA_HOME"):
            if root := os.environ.get(variable):
                directories.append(Path(root) / "bin")
        handles = [os.add_dll_directory(str(directory)) for directory in directories
                   if directory.is_dir()]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._dll_directory_handles = handles
    return module


def _digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _build(native, triangles):
    nodes, indices = native.bvh_build(triangles)
    return nodes, triangles[indices].copy()


def _random_mesh(count):
    rng = np.random.default_rng(count)
    triangles = rng.uniform(-100., 100., (count, 3, 3))
    triangles[::17, 2] = triangles[::17, 0]
    return triangles


def _rays(epochs, satellites, seed=7, near=False, shift=None):
    rng = np.random.default_rng(seed)
    rx = rng.uniform(-30., 30., (epochs, 3))
    sat = np.broadcast_to(_make_satellites(satellites), (epochs, satellites, 3)).copy()
    if near:
        sat = rng.uniform(-300., 300., (epochs, satellites, 3))
    if shift is not None:
        rx += shift
        sat += shift
    return rx, sat


def _tie_scene():
    def plane(x):
        return np.array([[x, -20., -10.], [x, 20., -10.], [x, 0., 20.]])

    triangles = np.repeat(np.stack([plane(-1.), plane(1.)]), 256, axis=0)
    nodes = np.array([
        [-1., -20., -10., 1., 20., 20., 1., 2., 0., 0.],
        [-1., -20., -10., -1., 20., 20., -1., -1., 0., 256.],
        [1., -20., -10., 1., 20., 20., -1., -1., 256., 256.],
    ])
    rx = np.zeros((1, 3))
    sat = np.array([[[0., 0., 10.]]])
    return rx, sat, nodes, triangles


def _validation_cases(native):
    for count in (1, 4, 127, 511, 512, 513, 1023, 1024, 1025, 4097):
        nodes, triangles = _build(native, _random_mesh(count))
        for satellites in (1, 12, 255, 256, 257, 4096, 4097):
            rx, sat = _rays(1, satellites, near=True)
            yield f"random/N{count}/rays{satellites}", (rx, sat, nodes, triangles)
    for boxes in (100, 1000):
        for shifted in (False, True):
            shift = np.array([-3957199., 3310205., 3737911.]) if shifted else np.zeros(3)
            nodes, triangles = _build(native, _make_building_mesh(boxes) + shift)
            for epochs in (1, 4):
                rx, sat = _rays(epochs, 12, shift=shift)
                yield f"boxes{boxes}/ECEF{shifted}/epochs{epochs}", (rx, sat, nodes, triangles)
    nodes, triangles = _build(native, _random_mesh(513))
    rx, sat = _rays(3, 5, near=True)
    sat[0, 0] = np.nan
    sat[1, 1] = np.inf
    rx[2] = np.nan
    yield "nonfinite_padding", (rx, sat, nodes, triangles)
    yield "all_nonfinite", (np.full_like(rx, np.nan), sat, nodes, triangles)
    yield "coincident_rx_sat", (np.zeros((1, 3)), np.zeros((1, 3, 3)), nodes, triangles)
    for scale in (1e-8, 1., 1e8):
        nodes, triangles = _build(native, _random_mesh(512) * scale)
        rx, sat = _rays(2, 7, near=True)
        yield f"scale{scale}", (rx * scale, sat * scale, nodes, triangles)
    nodes, triangles = _build(native, np.zeros((512, 3, 3)))
    rx, sat = _rays(2, 3)
    yield "all_degenerate", (rx, sat, nodes, triangles)
    rx, sat, nodes, triangles = _tie_scene()
    yield "equal_delay_right_first", (rx, sat, nodes, triangles)
    reversed_nodes = nodes.copy()
    reversed_nodes[0, [6, 7]] = nodes[0, [7, 6]]
    yield "equal_delay_reversed_root", (rx, sat, reversed_nodes, triangles)
    overlapping_nodes = nodes.copy()
    overlapping_nodes[1:, 8:10] = [0, 512]
    yield "overlapping_leaves_fallback", (rx, sat, overlapping_nodes, triangles)
    # A right-deep tree reaches the same 64-entry stack guard as the serial GPU
    # walk. The optimized path must not include descendants that guard skips.
    nodes = np.zeros((129, 10))
    nodes[:, :3] = -100.
    nodes[:, 3:6] = 100.
    nodes[:, 6:8] = -1
    for depth in range(64):
        node, leaf = 2 * depth, 2 * depth + 1
        nodes[node, 6:8] = [leaf, node + 2]
        nodes[leaf, 8:10] = [depth * 8, 8]
    nodes[128, 8:10] = [504, 8]
    triangles = _random_mesh(512)
    rx, sat = _rays(1, 12, near=True)
    yield "deep_tree_stack_guard", (rx, sat, nodes, triangles)
    nodes = nodes[:2].copy()
    nodes[0, 6:8] = [1, 1]
    nodes[1, 8:10] = [0, 512]
    yield "shared_leaf_fallback", (rx, sat, nodes, triangles)


def _validate(before, after):
    records = {}
    for name, arguments in _validation_cases(before):
        input_hash = _digest(*arguments)
        old = before.raytrace_multipath_bvh_batch(*arguments)
        new = after.raytrace_multipath_bvh_batch(*arguments)
        for expected, actual in zip(old, new, strict=True):
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64),
                                          err_msg=name)
        old_los = before.raytrace_los_check_bvh_batch(*arguments)
        new_los = after.raytrace_los_check_bvh_batch(*arguments)
        np.testing.assert_array_equal(new_los, old_los, err_msg=f"LOS/{name}")
        if _digest(*arguments) != input_hash:
            raise RuntimeError(f"Input mutated: {name}")
        records[name] = {
            "input_sha256": input_hash,
            "baseline_sha256": _digest(*old), "candidate_sha256": _digest(*new),
            "los_sha256": _digest(old_los),
            "nonzero_delays": int(np.count_nonzero(old[1])),
        }
    return {"cases": len(records), "bit_identical": True, "records": records}


def _timing_cases(native):
    for count in (12, 512, 1200, 12000):
        triangles = _make_building_mesh((count + 11) // 12)[:count]
        nodes, triangles = _build(native, triangles)
        for rays in (1, 12):
            rx = np.zeros((1, 3))
            sat = _make_satellites(rays)[None].copy()
            yield f"N{count}/rays{rays}", (rx, sat, nodes, triangles)
    nodes, triangles = _build(native, _make_building_mesh(100))
    for rays in (256, 257, 4096, 4097):
        rx, sat = _rays(1, rays)
        yield f"N1200/rays{rays}", (rx, sat, nodes, triangles)


def _measure(before, after, arguments, rounds, repeats):
    modules = {"baseline": before, "candidate": after}
    for module in modules.values():
        for _ in range(3):
            module.raytrace_multipath_bvh_batch(*arguments)
    samples = {label: [] for label in modules}
    for round_index in range(rounds):
        timings = {label: [] for label in modules}
        for repetition in range(repeats):
            order = ("baseline", "candidate") if (round_index + repetition) % 2 == 0 else (
                "candidate", "baseline")
            for label in order:
                start = time.perf_counter_ns()
                modules[label].raytrace_multipath_bvh_batch(*arguments)
                timings[label].append((time.perf_counter_ns() - start) / 1e6)
        for label in samples:
            samples[label].append(timings[label])
    medians = {label: [float(np.median(sample)) for sample in data]
               for label, data in samples.items()}
    old, new = [float(np.median(medians[label])) for label in modules]
    return {"baseline_ms": old, "candidate_ms": new, "speedup": old / new,
            "round_medians_ms": medians, "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--build-configuration", default="unspecified")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    before = _load_extension(args.baseline.resolve(), "baseline._bvh")
    after = _load_extension(args.candidate.resolve(), "candidate._bvh")
    validation = _validate(before, after)
    print(f"{validation['cases']} validation cases are bit-identical", flush=True)
    measurements = {}
    for name, arguments in _timing_cases(before):
        measurements[name] = _measure(before, after, arguments, args.rounds, args.repeats)
        record = measurements[name]
        print(f"{name}: {record['baseline_ms']:.3f} -> {record['candidate_ms']:.3f} ms "
              f"({record['speedup']:.2f}x)", flush=True)
    source = Path(__file__).resolve().parents[1] / "src/raytrace/bvh.cu"
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "method": "same process, alternating build order per call, synchronized native wall time",
        "scope": "native allocation/upload/kernels/output/free included; BVH construction excluded",
        "rounds": args.rounds, "repeats": args.repeats, "warmup_calls_per_build": 3,
        "source_sha256": {"baseline": hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                          "candidate": hashlib.sha256(source.read_bytes()).hexdigest()},
        "extension_sha256": {
            "baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
            "candidate": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        },
        "validation": validation, "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
