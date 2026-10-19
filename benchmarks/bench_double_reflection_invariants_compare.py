"""Compare complete physical single/double-reflection CUDA IQ epochs exactly."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_raytrace import _make_building_mesh
from bench_reflection_geometry_compare import compare, load, sha, tree_hash

ROOT = Path(__file__).resolve().parents[1]


def scene(count, satellites, kind='corner'):
    from gnss_gpu.raytrace import BuildingModel

    corner = np.array([
        [[5., -10., -5.], [5., 10., -5.], [5., 0., 5.]],
        [[-10., 5., -5.], [10., 5., -5.], [0., 5., 5.]],
    ])
    background = _make_building_mesh((count-2)//12)+np.array([2000., 2000., 2000.])
    mesh = np.concatenate((corner, background))
    rx = np.zeros(3)
    direction = np.stack((-np.linspace(1.35, 1.75, satellites), -np.ones(satellites),
        np.linspace(.12, .3, satellites)), axis=1)
    sat = 20200e3*direction/np.linalg.norm(direction, axis=1)[:, None]
    if kind == 'blocked':
        mesh[-12:] = BuildingModel.create_box([2., 2., 0.], 1., 30., 30.).triangles
    origin = np.array([6378137., 0., 0.])
    return BuildingModel(mesh[:, :, [2, 0, 1]]+origin), rx[[2, 0, 1]]+origin, sat[:, [2, 0, 1]]+origin


def make_call(module, count, satellites, samples, kind='corner', stage='urban', double_paths=4, single_paths=4, max_pairs=None):
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    import gnss_gpu.urban_signal_sim as urban

    model, rx, sat = scene(count, satellites, kind)
    if stage == 'paths':
        def call():
            return module.compute_double_reflection_paths(model.triangles, rx, sat,
                max_paths=double_paths, max_pairs=max_pairs)
        return call, None, tree_hash((model.triangles, rx, sat))
    assert max_pairs is None
    owner = UrbanSignalSimulator(model, max_reflection_paths=single_paths,
        max_double_reflection_paths=double_paths, elevation_mask_deg=0.)
    owner.sim.noise_seed = 37
    velocity = np.random.default_rng(219).normal(size=(satellites, 3))*1000
    clock = np.linspace(-1e-7, 1e-7, satellites)

    def call():
        original = urban.compute_double_reflection_paths
        urban.compute_double_reflection_paths = module.compute_double_reflection_paths
        try:
            return owner.compute_epoch(rx, sat, sat_vel=velocity, sat_clk=clock,
                rx_vel=[3., 2., 1.], n_samples=samples)
        finally:
            urban.compute_double_reflection_paths = original

    return call, owner, tree_hash((model.triangles, rx, sat))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[14, 50])
    parser.add_argument('--satellites', nargs='+', type=int, default=[12, 32])
    parser.add_argument('--samples', nargs='+', type=int, default=[2600, 260000])
    parser.add_argument('--scenes', nargs='+', default=['corner'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    record = json.loads(args.source_record.read_text(encoding='utf8'))
    for label, sources in record['source_sha256'].items():
        for name, digest in sources.items():
            path = args.baseline_dir/Path(name).name if label == 'baseline' else ROOT/name
            assert sha(path) == digest
    for name, digest in record['extension_sha256'].items():
        assert sha(ROOT/'python/gnss_gpu'/name) == digest
    _pin()
    sys.path.insert(0, str(ROOT/'python'))
    before = load(args.baseline_dir/'double_reflection.py', 'double_invariants_before')
    after = load(ROOT/'python/gnss_gpu/double_reflection.py', 'double_invariants_after')
    results = {}
    for count in args.counts:
        for n_sat in args.satellites:
            for kind in args.scenes:
                for samples in args.samples:
                    calls, owners, inputs = [], [], []
                    try:
                        for module in (before, after):
                            call, owner, digest = make_call(module, count, n_sat, samples, kind)
                            calls.append(call)
                            owners.append(owner)
                            inputs.append(digest)
                        assert inputs[0] == inputs[1]
                        row = compare(*calls, args.rounds, args.repeats)
                        output = calls[0]()
                        row['double_reflection_paths'] = output['n_double_reflection_paths']
                        row['single_reflection_paths'] = output['n_reflection_paths']
                        row['input_sha256'] = inputs[0]
                        row['ordered_pairs_per_satellite'] = count*(count-1)
                        if kind == 'corner':
                            assert row['double_reflection_paths'] > 0
                        key = f'{kind}/triangles{count}/satellites{n_sat}/samples{samples}/urban'
                        results[key] = row
                        print(key, row['baseline_ms'], '->', row['candidate_ms'],
                            'double paths', row['double_reflection_paths'], flush=True)
                    finally:
                        for owner in owners:
                            if owner is not None:
                                owner.sim.release_gpu()
    helpers = {str(Path(m.__file__).resolve().relative_to(ROOT)).replace('\\', '/'):sha(Path(m.__file__))
        for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).resolve().is_relative_to(ROOT)}
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_record_sha256=sha(args.source_record), source_sha256=record['source_sha256'], extension_sha256=record['extension_sha256'],
        helper_sha256=helpers, rounds=args.rounds, repeats=args.repeats, measurements=results,
        method='Serial alternating complete real CUDA LOS/IQ epochs; same native context, separate Python models. Own last allowed CPU, normal priority, nonexclusive host/GPU. Every returned field hashed outside timing. Warm first use and scene setup excluded.',
        scope='Physical single4/double4 paths, every original ordered pair and occlusion predicate, seed37. Function selection/restoration wrapper included symmetrically in timing. Real returned double-bounce replicas; no pair pruning.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
