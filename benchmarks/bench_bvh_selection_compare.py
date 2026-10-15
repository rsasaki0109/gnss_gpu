"""Measure reflection winner selection against the preceding mirror sharing build.

Use two actual builds, sequential alternating synchronous whole calls and
bit hashes of every timed output. Include whole synchronous calls and scene lifetime costs.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _sha
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays
from bench_bvh_normal_compare import _timing
from bench_bvh_mirror_compare import _urban32, _validate
from bench_raytrace import _make_building_mesh



def _dense(before, after, rounds, repeats):
    from bench_bvh_build_compare import _measure
    from gnss_gpu.bvh import BVHAccelerator

    records = {}
    for count in (512, 1200):
        mesh = _make_building_mesh((count + 11) // 12)[:count]
        nodes, sorted_mesh = _build(before, mesh)
        rx, sat = _rays(1, 4096)
        models = []
        for native in (before, after):
            _activate(native)
            model = BVHAccelerator(mesh)
            model._device_scene = native.DeviceBVH(nodes, sorted_mesh)
            models.append(model)
        try:
            for public in (False, True):
                def call(index):
                    model = models[index]
                    return list(model.trace_batch(rx, sat) if public else
                                model._device_scene.multipath_batch(rx, sat))

                record = _measure(lambda: call(0), lambda: call(1), rounds, repeats)
                record['input_sha256'] = _digest(rx, sat, nodes, sorted_mesh)
                stage = 'public_warm_trace' if public else 'warm_multipath'
                name = f'triangles{count}/rays4096/{stage}'
                records[name] = record
                print(f"{name}: {record['baseline_ms']:.4f} -> {record['candidate_ms']:.4f} ms "
                      f"({record['speedup']:.3f}x)", flush=True)
        finally:
            for model in models:
                model.release_gpu()
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
    parser.add_argument('--skip-dense', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1 or any(n < 1 for n in args.counts):
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
        before, after, args.rounds, args.repeats, args.counts,
        not args.skip_urban, not args.skip_controls)
    if not args.validation_only and not args.skip_urban:
        measurements.update(_urban32(before, after, args.rounds, args.repeats))
    if not args.validation_only and not args.skip_dense:
        measurements.update(_dense(before, after, args.rounds, args.repeats))
    helper_paths = (
        'benchmarks/bench_bvh_selection_compare.py', 'benchmarks/bench_bvh_mirror_compare.py',
        'benchmarks/bench_bvh_normal_compare.py',
        'benchmarks/bench_bvh_build_compare.py', 'benchmarks/bench_bvh_combined_compare.py',
        'benchmarks/bench_bvh_resident_compare.py', 'benchmarks/bench_bvh_multipath_compare.py',
        'benchmarks/bench_raytrace.py')
    report = dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
        baseline='preceding receiver mirror sharing; full candidate shared tree with eight barriers',
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
