"""Compare exact BVH trees and scene construction using two actual builds.

CPU build, public construction, initial trace and initial urban IQ calls include
the BVH builder. Warm controls retain the scene. GPU calls run sequentially.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from bench_bvh_combined_compare import _assert_tree, _tree_hash
from bench_bvh_multipath_compare import _digest, _load_extension, _rays
from bench_raytrace import _make_building_mesh, _make_satellites

SOURCE_PATHS = (
    "src/raytrace/bvh.cu", "include/gnss_gpu/bvh.h",
    "python/gnss_gpu/_bvh_bindings.cpp", "python/gnss_gpu/bvh.py",
    "python/gnss_gpu/urban_signal_sim.py",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mesh_cases():
    for count in (1, 2, 3, 4, 5, 7, 12, 31, 32, 33, 127, 256, 511, 512, 513, 1024, 4097):
        for seed in (7, 37, 103):
            mesh = np.random.default_rng(seed).uniform(-100., 100., (count, 3, 3))
            mesh[::17, 2] = mesh[::17, 0]
            yield f"random/triangles{count}/seed{seed}", mesh
    # Larger fully degenerate inputs exhaust the preceding recursive builder's
    # Windows stack. Keep both builds within its supported comparison range.
    for count in (4, 5, 33, 257):
        mesh = np.random.default_rng(count).uniform(-10., 10., (count, 3, 3))
        yield f"coplanar/triangles{count}", mesh * np.array([1., 1., 0.])
        yield f"collinear/triangles{count}", mesh * np.array([1., 0., 0.])
        yield f"points/triangles{count}", np.repeat(mesh[:, :1], 3, axis=1)
        yield f"duplicate/triangles{count}", np.repeat(mesh[:1], count, axis=0)
        equal_centroids = np.stack((mesh[:, 0], -mesh[:, 0], np.zeros((count, 3))), axis=1)
        yield f"equal_centroids/triangles{count}", equal_centroids
        zeros = np.zeros_like(mesh)
        zeros[::2, ::2, ::2] = -0.
        yield f"signed_zero/triangles{count}", zeros
    yield "coplanar/triangles1025", np.random.default_rng(1025).uniform(
        -10., 10., (1025, 3, 3)) * np.array([1., 1., 0.])
    mesh = np.random.default_rng(13).uniform(-1., 1., (257, 3, 3))
    for scale in (1e-100, 1e-16, 1e-14, 1., 1e50, 1e100):
        yield f"scale/{scale}", mesh * scale
    yield "ecef/small_offsets", mesh * 1e-6 + [6378137., -4500000., 3500000.]
    yield "strided", mesh[::2, ::-1, ::-1]
    yield "fortran", np.asfortranarray(mesh)
    yield "float32", mesh.astype(np.float32)
    yield "integer", (mesh * 100).astype(np.int64)
    for count in (12, 1200, 12000):
        yield f"buildings/triangles{count}", _make_building_mesh(count // 12)
    for seed in (19, 23, 29):
        mesh = np.random.default_rng(seed).normal(size=(1025, 3, 3))
        mesh[:1000] *= 1e-5
        yield f"clusters/seed{seed}", mesh


def _validate(before, after):
    records = {}
    retained = []
    for index, (name, mesh) in enumerate(_mesh_cases()):
        input_hash = _digest(mesh)
        old, new = list(before.bvh_build(mesh)), list(after.bvh_build(mesh))
        _assert_tree(old, new)
        assert _digest(mesh) == input_hash
        record = dict(input_sha256=input_hash, triangles=len(mesh), nodes=len(old[0]),
                      baseline_sha256=_tree_hash(old), candidate_sha256=_tree_hash(new))
        # Very small/large scales remain CPU-only: traversal has preexisting
        # epsilon limits. Ordinary, degenerate and tied trees use both builds.
        if not name.startswith("scale/"):
            rx, sat = _rays(2, 17, seed=13 + index, near=True)
            scenes = [native.DeviceBVH(nodes, np.asarray(mesh, dtype=np.float64)[indices])
                      for native, (nodes, indices) in zip((before, after), (old, new), strict=True)]
            try:
                a, b = [list(scene.trace_batch(rx, sat)) for scene in scenes]
                _assert_tree(a, b)
                assert scenes[0].allocated_bytes == scenes[1].allocated_bytes
                record.update(trace_sha256=_tree_hash(a),
                              blocked_rays=int(np.count_nonzero(~a[0])),
                              nonzero_delays=int(np.count_nonzero(a[2])))
            finally:
                for scene in scenes:
                    scene.close()
            assert all(scene.allocated_bytes == 0 for scene in scenes)
        records[name] = record
        if index % 7 == 0:
            retained.append((new, _tree_hash(new)))
        for output, digest in retained:
            assert _tree_hash(output) == digest, "Later build overwrote returned arrays"
    # Same storage, modified vertices: cached geometry must be rebuilt per call.
    mesh = _make_building_mesh(10)
    for index, shift in enumerate((0., 1000., -1000., 0.)):
        mesh += shift
        old, new = list(before.bvh_build(mesh)), list(after.bvh_build(mesh))
        _assert_tree(old, new)
        records[f"mutated_storage/{index}"] = dict(
            input_sha256=_digest(mesh), triangles=len(mesh), nodes=len(old[0]),
            baseline_sha256=_tree_hash(old), candidate_sha256=_tree_hash(new))
    return dict(cases=len(records), bit_identical=True, records=records)


def _measure(old_call, new_call, rounds, repeats):
    calls = dict(baseline=old_call, candidate=new_call)
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


def _activate(native):
    import gnss_gpu

    sys.modules["gnss_gpu._bvh"] = native
    gnss_gpu._bvh = native


def _public_build(native, mesh):
    from gnss_gpu.bvh import BVHAccelerator

    _activate(native)
    model = BVHAccelerator(mesh)
    return [model._nodes_flat, model._sorted_indices, model._sorted_tris]


def _cold_trace(native, mesh, rx, sat):
    from gnss_gpu.bvh import BVHAccelerator

    _activate(native)
    model = BVHAccelerator(mesh)
    try:
        return [model._nodes_flat, model._sorted_indices, *model.trace_batch(rx, sat)]
    finally:
        model.release_gpu()


def _cold_urban(native, mesh, rx, sat, samples):
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

    _activate(native)
    model = BVHAccelerator(mesh)
    sim = UrbanSignalSimulator(model)
    sim.sim.noise_seed = 37
    try:
        return dict(nodes=model._nodes_flat, indices=model._sorted_indices,
                    epoch=sim.compute_epoch(rx, sat, n_samples=samples))
    finally:
        sim.sim.release_gpu()
        model.release_gpu()


def _timing(before, after, rounds, repeats, counts):
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

    measurements = {}

    def measure(name, old, new, mesh):
        record = _measure(old, new, rounds, repeats)
        record["input_mesh_sha256"] = _digest(mesh)
        measurements[name] = record
        print(f"{name}: {record['baseline_ms']:.4f} -> {record['candidate_ms']:.4f} ms "
              f"({record['speedup']:.3f}x)", flush=True)

    for count in counts:
        mesh = _make_building_mesh(count // 12)
        measure(f"native_build/triangles{count}", lambda: list(before.bvh_build(mesh)),
                lambda: list(after.bvh_build(mesh)), mesh)
        measure(f"public_build/triangles{count}", lambda: _public_build(before, mesh),
                lambda: _public_build(after, mesh), mesh)
        rx, sat = np.zeros((1, 3)), _make_satellites(12)[None]
        measure(f"public_build_trace_close/triangles{count}/rays12",
                lambda: _cold_trace(before, mesh, rx, sat),
                lambda: _cold_trace(after, mesh, rx, sat), mesh)
        ecef_rx = np.array([6378137., 0., 0.])
        ecef_sat = _make_satellites(12)[:, [2, 0, 1]] + ecef_rx
        ecef_mesh = mesh[:, :, [2, 0, 1]] + ecef_rx
        for samples in (2600, 260000):
            measure(f"urban_build_epoch_close/triangles{count}/rays12/samples{samples}",
                    lambda: _cold_urban(before, ecef_mesh, ecef_rx, ecef_sat, samples),
                    lambda: _cold_urban(after, ecef_mesh, ecef_rx, ecef_sat, samples), ecef_mesh)
        sims = []
        for native in (before, after):
            _activate(native)
            model = BVHAccelerator(ecef_mesh)
            sim = UrbanSignalSimulator(model)
            sim.sim.noise_seed = 37
            # First call binds each model to the correct extension and allocates
            # the signal workspace. Warm controls then contain no BVH rebuild.
            sim.compute_epoch(ecef_rx, ecef_sat, n_samples=260000)
            sims.append(sim)
        try:
            def epoch(index, rebuild=False):
                _activate((before, after)[index])
                sim = sims[index]
                if rebuild:
                    sim.building_model.update_triangles(ecef_mesh)
                return sim.compute_epoch(ecef_rx, ecef_sat, n_samples=260000)

            measure(f"urban_rebuild_epoch/triangles{count}/rays12/samples260000",
                    lambda: epoch(0, True), lambda: epoch(1, True), ecef_mesh)
            measure(f"urban_warm_control/triangles{count}/rays12/samples260000",
                    lambda: epoch(0), lambda: epoch(1), ecef_mesh)
        finally:
            for sim in sims:
                sim.sim.release_gpu()
                sim.building_model.release_gpu()
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--counts", nargs="+", type=int, default=[12, 1200, 12000, 120000])
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or any(n < 12 or n % 12 for n in args.counts):
        parser.error("positive rounds/repeats and triangle counts divisible by 12 required")
    root = Path(__file__).resolve().parents[1]
    # Measure this checkout's wrappers, even when an older wheel is installed.
    sys.path.insert(0, str(root / "python"))
    reference = json.loads((root / "benchmarks/bvh_combined_gpu_20261011.json").read_text(encoding="utf-8"))
    hashes = dict(baseline={path: _sha(args.baseline_dir / Path(path).name) for path in SOURCE_PATHS},
                  candidate={path: _sha(root / path) for path in SOURCE_PATHS})
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert hashes["baseline"] == reference["source_sha256"]["candidate"], "Wrong baseline sources"
    assert extensions["baseline"] == reference["extension_sha256"]["candidate"], "Wrong baseline binary"
    assert all(hashes["baseline"][path] == hashes["candidate"][path] for path in SOURCE_PATHS[1:])
    before = _load_extension(args.baseline.resolve(), "baseline._bvh")
    after = _load_extension(args.candidate.resolve(), "candidate._bvh")
    _activate(after)
    validation = _validate(before, after)
    print(f"{validation['cases']} build cases have identical node/index bits", flush=True)
    measurements = {} if args.validation_only else _timing(
        before, after, args.rounds, args.repeats, args.counts)
    report = dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration="CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math",
        baseline="preceding combined resident BVH; repeated bounds/centroid calculations and per-node scratch",
        method="same process; sequential alternating actual builds per call; nonexclusive host",
        scope="CPU build includes binding validation and returned arrays; public adds snapshots/reordering; "
              "initial trace/urban includes CPU BVH build, allocation, transfers, GPU sync, output and close; "
              "urban includes actual CUDA IQ and channel metadata; warm controls exclude BVH rebuild; "
              "mesh generation, CUDA/code-table initialization and output hashing excluded; "
              "backend selection included symmetrically in public calls",
        rounds=args.rounds, repeats=args.repeats, warmup_calls_per_build=3,
        source_sha256=hashes, extension_sha256=extensions,
        signal_extension_sha256={str(path.relative_to(root)): _sha(path) for path in
                                 (root / "python/gnss_gpu").glob("_gnss_gpu_signal_sim*.pyd")},
        benchmark_sha256={path: _sha(root / path) for path in (
            "benchmarks/bench_bvh_build_compare.py", "benchmarks/bench_bvh_combined_compare.py",
            "benchmarks/bench_bvh_resident_compare.py", "benchmarks/bench_bvh_multipath_compare.py",
            "benchmarks/bench_raytrace.py")},
        validation=validation, measurements=measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
