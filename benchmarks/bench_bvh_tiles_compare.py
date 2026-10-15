"""Measure low-ray reflection tiling against the preceding mirror sharing build.

Use two actual builds, sequential alternating synchronous whole calls and
bit hashes of every timed output. Keep every GPU kernel unchanged; compare whole calls.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _sha
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _tie_scene
from bench_bvh_mirror_compare import _validate as _base_validate
from bench_bvh_combined_compare import _assert_tree, _tree_hash
from bench_raytrace import _make_building_mesh, _make_satellites

import numpy as np



def _validate(before, after):
    result = _base_validate(before, after)
    records = result['records']
    for count in (4096, 4097, 8192, 8193, 32768, 32769):
        mesh = np.random.default_rng(count).uniform(-100., 100., (count, 3, 3))
        mesh[::17, 2] = mesh[::17, 0]
        nodes, mesh = _build(before, mesh)
        for rays in (1, 2, 7, 8):
            rx, sat = _rays(1, rays, seed=count + rays, near=True)
            scenes = [native.DeviceBVH(nodes, mesh) for native in (before, after)]
            try:
                expected, actual = [list(scene.trace_batch(rx, sat)) for scene in scenes]
                _assert_tree(expected, actual)
                _assert_tree(expected, list(scenes[1].trace_batch(rx, sat)))
                _assert_tree(expected[1:], list(after.raytrace_multipath_bvh_batch(rx, sat, nodes, mesh)))
                records[f'tile_boundary/triangles{count}/rays{rays}'] = dict(
                    input_sha256=_digest(rx, sat, nodes, mesh), baseline_sha256=_tree_hash(expected),
                    candidate_sha256=_tree_hash(actual), nonzero_delays=int(np.count_nonzero(actual[2])),
                    allocated_bytes=dict(baseline=scenes[0].allocated_bytes, candidate=scenes[1].allocated_bytes))
            finally:
                for scene in scenes:
                    scene.close()
    for reverse in (False, True):
        rx, sat, nodes, mesh = _tie_scene()
        mesh = np.repeat(mesh, 65, axis=0)
        nodes[1, 8:10] = [0., 16640.]
        nodes[2, 8:10] = [16640., 16640.]
        if reverse:
            nodes[0, [6, 7]] = nodes[0, [7, 6]]
        scenes = [native.DeviceBVH(nodes, mesh) for native in (before, after)]
        try:
            expected, actual = [list(scene.trace_batch(rx, sat)) for scene in scenes]
            _assert_tree(expected, actual)
            records[f'tile_equal_delay/reverse{reverse}'] = dict(
                input_sha256=_digest(rx, sat, nodes, mesh), baseline_sha256=_tree_hash(expected),
                candidate_sha256=_tree_hash(actual))
        finally:
            for scene in scenes:
                scene.close()
    result['cases'] = len(records)
    return result


def _timing(before, after, rounds, repeats, counts, ray_counts, urban, checks):
    from bench_bvh_build_compare import _measure
    from gnss_gpu.bvh import BVHAccelerator

    records = {}
    def measure(name, old, new, args):
        record = _measure(old, new, rounds, repeats)
        record['input_sha256'] = _digest(*args)
        records[name] = record
        print(f"{name}: {record['baseline_ms']:.4f} -> {record['candidate_ms']:.4f} ms "
              f"({record['speedup']:.3f}x)", flush=True)

    for count in counts:
        mesh = _make_building_mesh((count + 11) // 12)[:count]
        nodes, sorted_mesh = _build(before, mesh)
        for rays in ray_counts:
            rx, sat = np.zeros((1, 3)), _make_satellites(rays)[None]
            args = rx, sat, nodes, sorted_mesh
            models = []
            for native in (before, after):
                _activate(native)
                model = BVHAccelerator(mesh)
                model._device_scene = native.DeviceBVH(nodes, sorted_mesh)
                models.append(model)
            old, new = [model._device_scene for model in models]
            def cold(native):
                scene = native.DeviceBVH(nodes, sorted_mesh)
                try:
                    return list(scene.trace_batch(rx, sat))
                finally:
                    scene.close()
            try:
                prefix = f'triangles{count}/rays{rays}'
                measure(prefix + '/warm_multipath', lambda: list(old.multipath_batch(rx, sat)),
                        lambda: list(new.multipath_batch(rx, sat)), args)
                measure(prefix + '/public_warm_trace', lambda: list(models[0].trace_batch(rx, sat)),
                        lambda: list(models[1].trace_batch(rx, sat)), args)
                measure(prefix + '/native_cold_trace_close', lambda: cold(before), lambda: cold(after), args)
                measure(prefix + '/unchanged_warm_los', lambda: old.check_los_batch(rx, sat),
                        lambda: new.check_los_batch(rx, sat), args)
                records[prefix + '/memory'] = dict(
                    baseline_allocated_bytes=old.allocated_bytes, candidate_allocated_bytes=new.allocated_bytes,
                    extra_bytes=new.allocated_bytes - old.allocated_bytes)
            finally:
                for model in models:
                    model.release_gpu()
    if checks:
        # Reuse the preceding public pipeline as unchanged >=8-ray controls.
        from bench_bvh_normal_compare import _timing as normal_timing
        controls = normal_timing(before, after, rounds, repeats, [12000], False, True)
        records.update({'control/' + name: value for name, value in controls.items()})
    if urban:
        from bench_bvh_mirror_compare import _urban32
        records.update(_urban32(before, after, rounds, repeats))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--counts', nargs='+', type=int, default=[512, 1200, 12000, 120000])
    parser.add_argument('--rays', nargs='+', type=int, default=[1, 2, 4, 7, 8])
    parser.add_argument('--skip-urban', action='store_true')
    parser.add_argument('--skip-controls', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or any(n < 1 for n in (*args.counts, *args.rays)):
        parser.error('positive rounds, repeats and triangle counts required')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'python'))
    reference_path = 'benchmarks/bvh_mirror_gpu_20261012.json'
    reference = json.loads((root / reference_path).read_text(encoding='utf-8'))
    hashes = dict(baseline={path: _sha(args.baseline_dir / Path(path).name) for path in SOURCE_PATHS},
                  candidate={path: _sha(root / path) for path in SOURCE_PATHS})
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert hashes['baseline'] == reference['source_sha256']['candidate'], 'Wrong baseline sources'
    assert extensions['baseline'] == reference['extension_sha256']['candidate'], 'Wrong baseline binary'
    assert all(hashes['baseline'][path] == hashes['candidate'][path] for path in SOURCE_PATHS[1:])
    before = _load_extension(args.baseline.resolve(), 'baseline._bvh')
    after = _load_extension(args.candidate.resolve(), 'candidate._bvh')
    _activate(after)
    validation = _validate(before, after)
    print(f"{validation['cases']} LOS/reflection cases are bit-identical", flush=True)
    measurements = {} if args.validation_only else _timing(
        before, after, args.rounds, args.repeats, args.counts, args.rays,
        not args.skip_urban, not args.skip_controls)
    helper_paths = (
        'benchmarks/bench_bvh_tiles_compare.py', 'benchmarks/bench_bvh_mirror_compare.py',
        'benchmarks/bench_bvh_normal_compare.py',
        'benchmarks/bench_bvh_build_compare.py', 'benchmarks/bench_bvh_combined_compare.py',
        'benchmarks/bench_bvh_resident_compare.py', 'benchmarks/bench_bvh_multipath_compare.py',
        'benchmarks/bench_raytrace.py')
    report = dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
        baseline='preceding mirror sharing build; candidate tiles capped at 32 despite 256-block target',
        baseline_reference=dict(path=reference_path, sha256=_sha(root / reference_path)),
        method='same process; sequential alternating actual builds per call; nonexclusive host',
        scope='warm multipath/LOS includes native validation, transfers, kernels, sync and returned arrays; '
              'public trace adds Python ray validation; native cold trace includes owner construction, '
              'first reflection normal/mirror preparation and close; urban includes actual CUDA IQ and '
              'channel metadata; CPU BVH construction, CUDA/code-table initialization and output hashing excluded',
        rounds=args.rounds, repeats=args.repeats, warmup_calls_per_build=3,
        source_sha256=hashes, extension_sha256=extensions,
        signal_extension_sha256={str(p.relative_to(root)): _sha(p) for p in
                                 (root / 'python/gnss_gpu').glob('_gnss_gpu_signal_sim*.pyd')},
        benchmark_sha256={path: _sha(root / path) for path in helper_paths},
        validation=validation, measurements=measurements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
