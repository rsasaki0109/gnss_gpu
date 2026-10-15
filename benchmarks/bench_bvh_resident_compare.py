"""Compare per-call BVH uploads with a resident mesh and reusable work buffers.

The baseline must include the preceding parallel multipath optimization. Warm
timing includes ray validation, transfers, kernels, synchronization, and output
copies. Cold timing additionally includes resident construction and close.
CPU BVH construction and Python class-wrapper validation are outside both.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _validation_cases
from bench_raytrace import _make_building_mesh, _make_satellites


def _assert_outputs(expected, actual, name):
    for old, new in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(new.view(np.uint64), old.view(np.uint64), err_msg=name)


def _check_case(before, scene, name, arguments):
    rx, sat, nodes, triangles = arguments
    input_hash = _digest(*arguments)
    old = before.raytrace_multipath_bvh_batch(*arguments)
    new = scene.multipath_batch(rx, sat)
    _assert_outputs(old, new, name)
    old_los = before.raytrace_los_check_bvh_batch(*arguments)
    new_los = scene.check_los_batch(rx, sat)
    np.testing.assert_array_equal(new_los, old_los, err_msg=f"LOS/{name}")
    single_hash = None
    if np.all(np.isfinite(rx[0])) and np.all(np.isfinite(sat[0])):
        old_single = before.raytrace_los_check_bvh(rx[0], sat[0], nodes, triangles)
        new_single = scene.check_los(rx[0], sat[0])
        np.testing.assert_array_equal(new_single, old_single, err_msg=f"single LOS/{name}")
        single_hash = _digest(old_single)
    if _digest(*arguments) != input_hash:
        raise RuntimeError(f"Input mutated: {name}")
    return {"input_sha256": input_hash, "baseline_sha256": _digest(*old),
            "candidate_sha256": _digest(*new), "los_sha256": _digest(old_los),
            "single_los_sha256": single_hash, "nonzero_delays": int(np.count_nonzero(old[1])),
            "allocated_bytes": scene.allocated_bytes}


def _validate(before, after):
    records = {}
    for name, arguments in _validation_cases(before):
        scene = after.DeviceBVH(*arguments[2:])
        try:
            records[name] = _check_case(before, scene, name, arguments)
        finally:
            scene.close()
        assert scene.allocated_bytes == 0
    triangles = _make_building_mesh(100)
    nodes, triangles = _build(before, triangles)
    scene = after.DeviceBVH(nodes, triangles)
    retained = []
    try:
        for index, (epochs, satellites) in enumerate(
                ((1, 1), (1, 12), (3, 7), (1, 257), (16, 256), (1, 4097), (2, 5), (1, 12), (1, 1))):
            rx, sat = _rays(epochs, satellites, seed=100 + index, near=True)
            name = f"reuse/{index}/epochs{epochs}/satellites{satellites}"
            records[name] = _check_case(before, scene, name, (rx, sat, nodes, triangles))
            outputs = scene.multipath_batch(rx, sat)
            retained.append((outputs, _digest(*outputs)))
            for outputs, digest in retained:
                assert _digest(*outputs) == digest, "Later calls overwrote earlier output arrays"
        # Ownership: no CPU input references are needed after construction.
        rx, sat = _rays(1, 12, near=True)
        expected = before.raytrace_multipath_bvh_batch(rx, sat, nodes, triangles)
        nodes.fill(0.)
        triangles.fill(0.)
        actual = scene.multipath_batch(rx, sat)
        _assert_outputs(expected, actual, "owned_snapshot")
        records["owned_snapshot"] = {"baseline_sha256": _digest(*expected),
                                     "candidate_sha256": _digest(*actual)}
    finally:
        scene.close()
    scene.close()
    assert scene.allocated_bytes == 0
    return {"cases": len(records), "bit_identical": True, "records": records}


def _measure(old_call, new_call, rounds, repeats):
    calls = {"baseline": old_call, "candidate": new_call}
    for call in calls.values():
        for _ in range(3):
            call()
    samples = {label: [] for label in calls}
    for round_index in range(rounds):
        timings = {label: [] for label in calls}
        for repetition in range(repeats):
            order = ("baseline", "candidate") if (round_index + repetition) % 2 == 0 else (
                "candidate", "baseline")
            for label in order:
                start = time.perf_counter_ns()
                calls[label]()
                timings[label].append((time.perf_counter_ns() - start) / 1e6)
        for label in samples:
            samples[label].append(timings[label])
    medians = {label: [float(np.median(sample)) for sample in data]
               for label, data in samples.items()}
    old, new = [float(np.median(medians[label])) for label in calls]
    return {"baseline_ms": old, "candidate_ms": new, "speedup": old / new,
            "round_medians_ms": medians, "samples_ms": samples}


def _timing_cases(before):
    for count in (12, 1200, 12000):
        nodes, triangles = _build(before, _make_building_mesh(count // 12))
        for epochs, satellites in ((1, 12), (16, 12)):
            rx = np.zeros((epochs, 3))
            sat = np.broadcast_to(_make_satellites(satellites), (epochs, satellites, 3)).copy()
            yield f"triangles{count}/rays{epochs * satellites}", (rx, sat, nodes, triangles)
    nodes, triangles = _build(before, _make_building_mesh(100))
    rx, sat = _rays(1, 4097, near=True)
    yield "triangles1200/rays4097", (rx, sat, nodes, triangles)


def _measure_case(before, after, arguments, rounds, repeats):
    rx, sat, nodes, triangles = arguments

    def old_los():
        return before.raytrace_los_check_bvh_batch(*arguments)

    def old_multipath():
        return before.raytrace_multipath_bvh_batch(*arguments)

    def old_pipeline():
        return old_los(), old_multipath()

    def cold_pipeline():
        scene = after.DeviceBVH(nodes, triangles)
        try:
            return scene.check_los_batch(rx, sat), scene.multipath_batch(rx, sat)
        finally:
            scene.close()

    def cold_los():
        scene = after.DeviceBVH(nodes, triangles)
        try:
            return scene.check_los_batch(rx, sat)
        finally:
            scene.close()

    scene = after.DeviceBVH(nodes, triangles)
    try:
        def new_pipeline():
            return scene.check_los_batch(rx, sat), scene.multipath_batch(rx, sat)

        # Validate the precise timed input, including large/fallback batches.
        validation = _check_case(before, scene, "timing", arguments)
        stages = {"warm_los": (old_los, lambda: scene.check_los_batch(rx, sat)),
                  "warm_multipath": (old_multipath, lambda: scene.multipath_batch(rx, sat)),
                  "warm_los_multipath": (old_pipeline, new_pipeline),
                  "cold_los": (old_los, cold_los),
                  "cold_los_multipath": (old_pipeline, cold_pipeline)}
        measurements = {label: _measure(*calls, rounds, repeats) for label, calls in stages.items()}
        measurements["resident_allocated_bytes"] = scene.allocated_bytes
        measurements["validation"] = validation
        return measurements
    finally:
        scene.close()


def _hash_files(root, relative_paths):
    return {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in relative_paths}


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
    before = _load_extension(args.baseline.resolve(), "baseline._bvh")
    after = _load_extension(args.candidate.resolve(), "candidate._bvh")
    validation = _validate(before, after)
    print(f"{validation['cases']} validation cases are bit-identical", flush=True)
    measurements = {}
    if not args.validation_only:
        for name, arguments in _timing_cases(before):
            measurements[name] = _measure_case(before, after, arguments, args.rounds, args.repeats)
            for stage, record in measurements[name].items():
                if isinstance(record, dict) and "speedup" in record:
                    print(f"{name}/{stage}: {record['baseline_ms']:.3f} -> "
                          f"{record['candidate_ms']:.3f} ms ({record['speedup']:.2f}x)", flush=True)
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "build_configuration": args.build_configuration,
        "baseline": "preceding parallel multipath implementation; per-call geometry upload",
        "method": "same process, alternating call order, synchronized native wall time",
        "scope": "native validation/transfers/kernels/output included; cold also construction/close; "
                 "CPU BVH construction and Python class wrapper excluded",
        "rounds": args.rounds, "repeats": args.repeats, "warmup_calls_per_build": 3,
        "source_sha256": {
            "baseline": _hash_files(args.baseline_dir, ["bvh.cu", "bvh.h", "_bvh_bindings.cpp", "bvh.py"]),
            "candidate": _hash_files(root, ["src/raytrace/bvh.cu", "include/gnss_gpu/bvh.h",
                                            "python/gnss_gpu/_bvh_bindings.cpp", "python/gnss_gpu/bvh.py"]),
        },
        "extension_sha256": {"baseline": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                             "candidate": hashlib.sha256(args.candidate.read_bytes()).hexdigest()},
        "benchmark_sha256": _hash_files(root, ["benchmarks/bench_bvh_resident_compare.py",
                                               "benchmarks/bench_bvh_multipath_compare.py",
                                               "benchmarks/bench_raytrace.py"]),
        "validation": validation, "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
