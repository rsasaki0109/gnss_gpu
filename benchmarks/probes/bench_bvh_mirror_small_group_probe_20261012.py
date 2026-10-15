"""Measure receiver mirror sharing against the preceding resident normal cache.

Use two actual builds, sequential alternating synchronous whole calls and
bit hashes of every timed output. Every call includes fresh receiver-dependent mirror preparation when selected.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _sha
from bench_bvh_multipath_compare import _build, _digest, _load_extension, _rays, _tie_scene
from bench_bvh_combined_compare import _assert_tree, _tree_hash
from bench_raytrace import _make_building_mesh

import numpy as np
from bench_bvh_normal_compare import _timing, _validate as _base_validate



def _extra_cases(before):
    for count in (8191, 8192, 8193):
        mesh = np.random.default_rng(count).uniform(-100., 100., (count, 3, 3))
        mesh[::17, 2] = mesh[::17, 0]
        nodes, mesh = _build(before, mesh)
        for sats in (7, 8, 9):
            rx, sat = _rays(2, sats, seed=count + sats, near=True)
            yield f'mirror_boundary/triangles{count}/satellites{sats}', (rx, sat, nodes, mesh)
    # Test the 32 MiB scratch cap on either side without building a deep tree.
    mesh = np.random.default_rng(137).uniform(-100., 100., (8192, 3, 3))
    bounds = np.concatenate((mesh.min(axis=(0, 1)), mesh.max(axis=(0, 1))))
    nodes = np.array([[*bounds, -1., -1., 0., len(mesh)]])
    for epochs in (170, 171):
        rx, sat = _rays(epochs, 8, seed=epochs, near=True)
        yield f'mirror_memory_cap/epochs{epochs}', (rx, sat, nodes, mesh)
    for reverse in (False, True):
        rx, sat, nodes, mesh = _tie_scene()
        mesh = np.repeat(mesh, 16, axis=0)
        nodes[1, 8:10] = [0., 4096.]
        nodes[2, 8:10] = [4096., 4096.]
        if reverse:
            nodes[0, [6, 7]] = nodes[0, [7, 6]]
        yield f'mirror_equal_delay/reverse{reverse}', (rx, np.repeat(sat, 8, axis=1), nodes, mesh)
    nodes, mesh = _build(before, _make_building_mesh(1000))
    rx, sat = _rays(3, 17, near=True)
    sat[0, 0] = np.nan
    sat[1, 1] = np.inf
    sat[1, 2] = rx[1]
    rx[2] = np.nan
    yield 'mirror_nonfinite', (rx, sat, nodes, mesh)
    yield 'mirror_all_nonfinite', (np.full_like(rx, np.nan), sat, nodes, mesh)


def _validate(before, after):
    result = _base_validate(before, after)
    records = result['records']
    for name, args in _extra_cases(before):
        rx, sat, nodes, mesh = args
        old, new = [native.DeviceBVH(nodes, mesh) for native in (before, after)]
        try:
            expected = list(old.trace_batch(rx, sat))
            actual = list(new.trace_batch(rx, sat))
            _assert_tree(expected, actual)
            _assert_tree(expected, list(new.trace_batch(rx, sat)))
            records[name] = dict(input_sha256=_digest(*args), baseline_sha256=_tree_hash(expected),
                                 candidate_sha256=_tree_hash(actual),
                                 nonzero_delays=int(np.count_nonzero(actual[2])),
                                 allocated_bytes=dict(baseline=old.allocated_bytes,
                                                      candidate=new.allocated_bytes))
        finally:
            old.close()
            new.close()
        assert old.allocated_bytes == new.allocated_bytes == 0
    nodes, mesh = _build(before, _make_building_mesh(1000))
    old, new = [native.DeviceBVH(nodes, mesh) for native in (before, after)]
    retained = []
    try:
        # Changing receivers, epochs, satellites and fallback paths must always
        # refresh the mirror values; later calls must not overwrite outputs.
        for index, (epochs, sats) in enumerate(((1, 12), (2, 8), (1, 7), (1, 4097),
                                              (4, 9), (1, 12), (1, 1))):
            rx, sat = _rays(epochs, sats, seed=71 + index, near=True)
            expected = list(old.trace_batch(rx, sat))
            actual = list(new.trace_batch(rx, sat))
            _assert_tree(expected, actual)
            retained.append((actual, _tree_hash(actual)))
            for arrays, digest in retained:
                assert _tree_hash(arrays) == digest
            records[f'mirror_refresh/{index}/epochs{epochs}/satellites{sats}'] = dict(
                input_sha256=_digest(rx, sat, nodes, mesh), baseline_sha256=_tree_hash(expected),
                candidate_sha256=_tree_hash(actual))
    finally:
        old.close()
        new.close()
    result['cases'] = len(records)
    return result


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
    reference_path = 'benchmarks/bvh_normal_gpu_20261012.json'
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
    helper_paths = (
        'benchmarks/bench_bvh_mirror_compare.py', 'benchmarks/bench_bvh_normal_compare.py',
        'benchmarks/bench_bvh_build_compare.py', 'benchmarks/bench_bvh_combined_compare.py',
        'benchmarks/bench_bvh_resident_compare.py', 'benchmarks/bench_bvh_multipath_compare.py',
        'benchmarks/bench_raytrace.py')
    report = dict(
        schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
        baseline='preceding resident GPU normal cache; receiver mirror recomputed for each satellite',
        baseline_reference=dict(path=reference_path, sha256=_sha(root / reference_path)),
        method='same process; sequential alternating actual builds per call; nonexclusive host',
        scope='warm multipath/LOS includes native validation, transfers, kernels, sync and returned arrays; '
              'public trace adds Python ray validation; native cold trace includes owner construction, '
              'first reflection normal and per-call mirror preparation and close; urban includes actual CUDA IQ and '
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
