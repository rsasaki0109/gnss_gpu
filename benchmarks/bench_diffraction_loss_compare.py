"""Compare full CPU knife-edge/native LOS/CUDA IQ epochs with exact outputs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_reflection_geometry_compare import compare, load, sha, tree_hash

ROOT = Path(__file__).resolve().parents[1]


def scene(count, satellites, kind='normal'):
    from gnss_gpu.raytrace import BuildingModel

    rng = np.random.default_rng(222)
    start = np.column_stack((np.linspace(80., 200., count), rng.uniform(-8., 8., count), np.full(count, -5.)))
    end = start+np.array([0., 0., 70.])
    if kind == 'sparse':
        start[2:, 1] += 1000.
        end[2:, 1] += 1000.
    first = np.stack((start, end, start+[0., 1., 0.]), axis=1)
    second = np.stack((end, end+[0., 1., 0.], start+[0., 1., 0.]), axis=1)
    mesh = np.concatenate((first, second))
    direction = np.stack((np.ones(satellites), np.linspace(-.04, .04, satellites),
        np.linspace(.12, .3, satellites)), axis=1)
    satellites_xyz = 20200e3*direction/np.linalg.norm(direction, axis=1)[:, None]
    origin = np.array([6378137., 0., 0.])
    start, end = start[:, [2, 0, 1]]+origin, end[:, [2, 0, 1]]+origin
    edges = SimpleNamespace(start=start, end=end, midpoint=.5*(start+end), size=count,
        face_dir_a=np.repeat(np.array([[0., 0., 1.]]), count, axis=0))
    return BuildingModel(mesh[:, :, [2, 0, 1]]+origin), edges, origin.copy(), satellites_xyz[:, [2, 0, 1]]+origin


def make_call(module, count, satellites, samples, kind='normal', stage='urban', paths=4, gpu=False, shadow=False):
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    import gnss_gpu.urban_signal_sim as urban

    model, edges, rx, sat = scene(count, satellites, kind)
    options = dict(lit_shadow=True) if shadow else {}
    digest = tree_hash((model.triangles, edges.start, edges.end, rx, sat, options))
    if stage == 'paths':
        def call():
            return module.compute_diffraction_paths(rx, sat, edges, max_paths=paths, **options)
        return call, None, digest
    owner = UrbanSignalSimulator(model, max_diffraction_paths=paths, diffraction_edges=edges,
        diffraction_use_gpu=gpu, diffraction_path_kwargs=options, elevation_mask_deg=0.)
    owner.sim.noise_seed = 37
    velocity = np.random.default_rng(222).normal(size=(satellites, 3))*1000.
    clock = np.linspace(-1e-7, 1e-7, satellites)

    def call():
        original = urban.compute_diffraction_paths
        urban.compute_diffraction_paths = module.compute_diffraction_paths
        try:
            return owner.compute_epoch(rx, sat, sat_vel=velocity, sat_clk=clock,
                rx_vel=[3., 2., 1.], n_samples=samples)
        finally:
            urban.compute_diffraction_paths = original
    return call, owner, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-dir', 'source-record', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[128, 512])
    parser.add_argument('--satellites', nargs='+', type=int, default=[12, 32])
    parser.add_argument('--samples', nargs='+', type=int, default=[2600, 260000])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    record = json.loads(args.source_record.read_text(encoding='utf8'))
    for label,sources in record['source_sha256'].items():
        for name,digest in sources.items():
            assert sha(args.baseline_dir/Path(name).name if label == 'baseline' else ROOT/name) == digest
    for name,digest in record['extension_sha256'].items():
        assert sha(ROOT/'python/gnss_gpu'/name) == digest
    _pin()
    sys.path.insert(0, str(ROOT/'python'))
    before = load(args.baseline_dir/'diffraction.py', 'diffraction_loss_before')
    after = load(ROOT/'python/gnss_gpu/diffraction.py', 'diffraction_loss_after')
    measurements = {}
    for count in args.counts:
        for satellites in args.satellites:
            for samples in args.samples:
                calls, owners, inputs = [], [], []
                try:
                    for module in (before,after):
                        call, owner, digest = make_call(module, count, satellites, samples)
                        calls.append(call)
                        owners.append(owner)
                        inputs.append(digest)
                    assert inputs[0] == inputs[1]
                    row = compare(*calls, args.rounds, args.repeats)
                    value = calls[0]()
                    row['diffraction_paths'] = value['n_diffraction_paths']
                    row['input_sha256'] = inputs[0]
                    assert row['diffraction_paths'] == satellites*4
                    key = f'edges{count}/satellites{satellites}/samples{samples}/urban'
                    measurements[key] = row
                    print(key, row['baseline_ms'], '->', row['candidate_ms'], row['reduction_percent'], flush=True)
                finally:
                    for owner in owners:
                        if owner is not None:
                            owner.sim.release_gpu()
    helpers = {str(Path(module.__file__).resolve().relative_to(ROOT)).replace('\\', '/'):sha(Path(module.__file__))
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(ROOT)}
    output = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_record_sha256=sha(args.source_record), source_sha256=record['source_sha256'], extension_sha256=record['extension_sha256'],
        helper_sha256=helpers, rounds=args.rounds, repeats=args.repeats, measurements=measurements,
        method='Serial alternating complete native LOS/legacy multipath/CPU knife-edge paths/clock/Doppler/channels/real CUDA IQ. Every output field hashed outside timing; same native context, independent models, seed37. Own last allowed CPU, normal priority, nonexclusive host/GPU. Setup/warm/hash excluded.',
        scope='All original edge candidates, ordering, geometry and loss formulas; four actual returned diffraction replicas per satellite. No edge pruning, changed precision or model. Function selection/restoration wrapper included symmetrically.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(output, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
