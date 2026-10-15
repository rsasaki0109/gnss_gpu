"""Compare resident GPU normal reuse with the preceding actual BVH build.

All GPU calls run sequentially. Alternating measurements include synchronous
whole calls, transfers and returned arrays; every timed output is bit-checked.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _measure, _sha
from bench_bvh_combined_compare import _assert_tree, _tree_hash, _validate as _legacy_validate
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays
from bench_raytrace import _make_building_mesh, _make_satellites


def _normal_cases():
    rng = np.random.default_rng(713)
    base = rng.uniform(-1., 1., (513, 3, 3))
    base[::17, 2] = base[::17, 0]
    for scale in (1e-100, 1e-8, 3e-8, 1e-7, 1., 1e50, 1e100):
        mesh = base * scale
        bounds = np.concatenate((mesh.min(axis=(0, 1)), mesh.max(axis=(0, 1))))
        nodes = np.array([[*bounds, -1., -1., 0., len(mesh)]])
        rx, sat = _rays(2, 17, near=True)
        yield f"normal_scale/{scale}", (rx * scale, sat * scale, nodes, mesh)
    # The strict normal cutoff and signed-zero behavior stay on the GPU.
    for area in (np.nextafter(1e-15, 0.), 1e-15, np.nextafter(1e-15, np.inf)):
        mesh = np.repeat(np.array([[[0., 0., 0.], [1., 0., 0.], [0., area, 0.]]]), 513, axis=0)
        mesh[::2, :, 2] = -0.
        nodes = np.array([[0., 0., -0., 1., area, 0., -1., -1., 0., len(mesh)]])
        rx = np.array([[.25, area / 4., 1.]])
        sat = np.array([[[.25, area / 4., 10.], [.5, area / 4., -10.]]])
        yield f"normal_cutoff/{area:.17g}", (rx, sat, nodes, mesh)


def _validate(before, after):
    result = _legacy_validate(before, after)
    records = result['records']
    for name, args in _normal_cases():
        rx, sat, nodes, mesh = args
        inputs = _digest(*args)
        expected = [before.raytrace_los_check_bvh_batch(*args),
                    *before.raytrace_multipath_bvh_batch(*args)]
        scene = after.DeviceBVH(nodes, mesh)
        try:
            for _ in range(2):
                actual = list(scene.trace_batch(rx, sat))
                _assert_tree(expected, actual)
            records[name] = dict(input_sha256=inputs, baseline_sha256=_tree_hash(expected),
                                 candidate_sha256=_tree_hash(actual),
                                 nonzero_delays=int(np.count_nonzero(actual[2])),
                                 allocated_bytes=scene.allocated_bytes)
        finally:
            scene.close()
        assert scene.allocated_bytes == 0 and _digest(*args) == inputs
    result['cases'] = len(records)
    return result


def _timing(before, after, rounds, repeats, counts, urban, checks):
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator

    records = {}

    def measure(name, old, new, args, **metadata):
        record = _measure(old, new, rounds, repeats)
        record.update(input_sha256=_digest(*args), **metadata)
        records[name] = record
        print(f"{name}: {record['baseline_ms']:.4f} -> {record['candidate_ms']:.4f} ms "
              f"({record['speedup']:.3f}x)", flush=True)

    for count in counts:
        mesh = _make_building_mesh((count + 11) // 12)[:count]
        nodes, sorted_mesh = _build(before, mesh)
        for rays in (12, 192):
            rx = np.zeros((1, 3))
            sat = _make_satellites(rays)[None]
            args = (rx, sat, nodes, sorted_mesh)
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
                prefix = f"triangles{count}/rays{rays}"
                measure(prefix + '/warm_multipath', lambda: list(old.multipath_batch(rx, sat)),
                        lambda: list(new.multipath_batch(rx, sat)), args)
                measure(prefix + '/public_warm_trace', lambda: list(models[0].trace_batch(rx, sat)),
                        lambda: list(models[1].trace_batch(rx, sat)), args)
                measure(prefix + '/native_cold_trace_close', lambda: cold(before),
                        lambda: cold(after), args)
                measure(prefix + '/unchanged_warm_los', lambda: old.check_los_batch(rx, sat),
                        lambda: new.check_los_batch(rx, sat), args)
                records[prefix + '/memory'] = dict(
                    baseline_allocated_bytes=old.allocated_bytes,
                    candidate_allocated_bytes=new.allocated_bytes,
                    extra_bytes=new.allocated_bytes - old.allocated_bytes)
            finally:
                for model in models:
                    model.release_gpu()
    if checks:
        nodes, mesh = _build(before, _make_building_mesh(100))
        rx, sat = _rays(1, 4097)
        args = rx, sat, nodes, mesh
        scenes = [native.DeviceBVH(nodes, mesh) for native in (before, after)]
        try:
            measure('triangles1200/rays4097/unchanged_serial_multipath',
                    lambda: list(scenes[0].multipath_batch(rx, sat)),
                    lambda: list(scenes[1].multipath_batch(rx, sat)), args)
            measure('triangles1200/rays4097/unchanged_serial_trace',
                    lambda: list(scenes[0].trace_batch(rx, sat)),
                    lambda: list(scenes[1].trace_batch(rx, sat)), args)
            assert scenes[0].allocated_bytes == scenes[1].allocated_bytes
        finally:
            for scene in scenes:
                scene.close()
    if urban:
        rx = np.array([6378137., 0., 0.])
        sat = _make_satellites(12)[:, [2, 0, 1]] + rx
        for count in (1200, 12000, 120000):
            mesh = _make_building_mesh(count // 12)[:, :, [2, 0, 1]] + rx
            sims = []
            for native in (before, after):
                _activate(native)
                model = BVHAccelerator(mesh)
                model._device_scene = native.DeviceBVH(model._nodes_flat, model._sorted_tris)
                sim = UrbanSignalSimulator(model)
                sim.sim.noise_seed = 37
                sims.append(sim)
            try:
                for samples in (2600, 260000):
                    def epoch(index):
                        return sims[index].compute_epoch(rx, sat, n_samples=samples)

                    output = epoch(0)
                    measure(f'urban_warm/triangles{count}/rays12/samples{samples}',
                            lambda: epoch(0), lambda: epoch(1), (mesh, rx, sat),
                            channels=len(output['channels']))
            finally:
                for sim in sims:
                    sim.sim.release_gpu()
                    sim.building_model.release_gpu()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--counts', nargs='+', type=int, default=[12, 512, 1200, 12000, 120000])
    parser.add_argument('--skip-urban', action='store_true')
    parser.add_argument('--skip-controls', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or any(n < 1 for n in args.counts):
        parser.error('positive rounds, repeats and triangle counts required')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'python'))
    reference_path = 'benchmarks/bvh_build_gpu_20261012.json'
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
        before, after, args.rounds, args.repeats, args.counts,
        not args.skip_urban, not args.skip_controls)
    helper_paths = ('benchmarks/bench_bvh_normal_compare.py', 'benchmarks/bench_bvh_build_compare.py',
                    'benchmarks/bench_bvh_combined_compare.py', 'benchmarks/bench_bvh_resident_compare.py',
                    'benchmarks/bench_bvh_multipath_compare.py', 'benchmarks/bench_raytrace.py')
    report = dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
        baseline='preceding CPU geometry/scratch build cache; resident reflection normals recomputed per ray',
        baseline_reference=dict(path=reference_path, sha256=_sha(root / reference_path)),
        method='same process; sequential alternating actual builds per call; nonexclusive host',
        scope='warm multipath/LOS includes native validation, transfers, kernels, sync and returned arrays; '
              'public trace adds Python ray validation; native cold trace includes owner construction, '
              'first reflection order/normal preparation and close; urban includes actual CUDA IQ and '
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
