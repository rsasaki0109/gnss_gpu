"""Compare exact detailed reflection geometry and complete real urban IQ calls."""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

from bench_pf_private_weights_compare import _pin
from bench_raytrace import _make_building_mesh, _make_satellites

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(value):
    result = hashlib.sha256()

    def add(item):
        if dataclasses.is_dataclass(item):
            add({field.name: getattr(item, field.name) for field in dataclasses.fields(item)})
        elif isinstance(item, np.ndarray):
            result.update(str((item.dtype.str, item.shape)).encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(b'dict')
            for key, val in item.items():
                add(key)
                add(val)
        elif isinstance(item, (list, tuple)):
            result.update(str(len(item)).encode())
            for val in item:
                add(val)
        elif isinstance(item, (float, np.floating)):
            result.update(struct.pack('d', item))
        else:
            result.update(repr(item).encode())
    add(value)
    return result.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scene(module, count, satellites, kind):
    origin = np.array([6378137., 0., 0.])
    if kind == 'grid':
        mesh = _make_building_mesh(count//12)
        rx = np.zeros(3)
        sat = _make_satellites(satellites)
    else:
        wall = module.BuildingModel.create_box([.05, 0., 0.], .1, 20., 20.).triangles
        background = _make_building_mesh((count-12)//12)+np.array([2000., 2000., 2000.]) if count > 12 else np.empty((0, 3, 3))
        mesh = np.concatenate((wall, background))
        if kind == 'blocked':
            mesh[-12:] = module.BuildingModel.create_box([-15., 0., 0.], 1., 100., 100.).triangles
        rx = np.array([-10., -3., 0.])
        direction = np.stack((-np.ones(satellites), np.linspace(-.3, .3, satellites), np.linspace(.2, .8, satellites)), axis=1)
        sat = rx+20200e3*direction/np.linalg.norm(direction, axis=1)[:, None]
    return module.BuildingModel(mesh[:, :, [2, 0, 1]]+origin), rx[[2, 0, 1]]+origin, sat[:, [2, 0, 1]]+origin


def compare(before, after, rounds, repeats):
    rows, medians = [], {'baseline': [], 'candidate': []}
    calls = dict(baseline=before, candidate=after)
    expected = tree_hash(before())
    assert tree_hash(after()) == expected
    for round_index in range(rounds):
        samples = {key: [] for key in calls}
        for repetition in range(repeats):
            order = ['baseline', 'candidate'] if (round_index+repetition) % 2 == 0 else ['candidate', 'baseline']
            pair = {}
            for label in order:
                start = time.perf_counter_ns()
                value = calls[label]()
                elapsed = (time.perf_counter_ns()-start)/1e6
                output = tree_hash(value)
                assert output == expected
                samples[label].append(elapsed)
                pair[label] = dict(ms=elapsed, output_sha256=output)
            rows.append(dict(round=round_index, repetition=repetition, order=order, runs=pair))
        for label in calls:
            medians[label].append(float(np.median(samples[label])))
    a, b = (float(np.median(medians[key])) for key in calls)
    return dict(baseline_ms=a, candidate_ms=b, reduction_percent=100*(1-b/a), round_medians_ms=medians,
        improving_rounds=sum(y < x for x, y in zip(*medians.values(), strict=True)),
        all_timed_outputs_bit_identical=True, pairs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[120, 360])
    parser.add_argument('--satellites', nargs='+', type=int, default=[12, 32])
    parser.add_argument('--samples', nargs='+', type=int, default=[2600])
    parser.add_argument('--scenes', nargs='+', default=['wall', 'grid'])
    parser.add_argument('--stages', nargs='+', default=['paths', 'urban'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    record = json.loads(args.source_record.read_text(encoding='utf8'))
    for label, sources in record['source_sha256'].items():
        for name, digest in sources.items():
            path = args.baseline_dir/Path(name).name if label == 'baseline' else ROOT/name
            assert sha(path) == digest, (label, name)
    for name, digest in record['extension_sha256'].items():
        assert sha(ROOT/'python/gnss_gpu'/name) == digest
    _pin()
    sys.path.insert(0, str(ROOT/'python'))
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    before = load(args.baseline_dir/'raytrace.py', 'reflection_geometry_before')
    after = load(ROOT/'python/gnss_gpu/raytrace.py', 'reflection_geometry_after')
    results = {}
    for count in args.counts:
        for n_sat in args.satellites:
            for kind in args.scenes:
                for samples in args.samples:
                    for stage in args.stages:
                        sims, calls = [], []
                        try:
                            for module in (before, after):
                                model, rx, sat = scene(module, count, n_sat, kind)
                                if stage == 'paths':
                                    def call(m=model, r=rx, s=sat):
                                        return m.compute_reflection_paths(r, s, max_paths=4)
                                else:
                                    simulator = UrbanSignalSimulator(model, max_reflection_paths=4, elevation_mask_deg=0.)
                                    simulator.sim.noise_seed = 37
                                    sims.append(simulator)
                                    velocity = np.random.default_rng(215).normal(size=(n_sat, 3))*1000
                                    clock = np.linspace(-1e-7, 1e-7, n_sat)
                                    def call(owner=simulator, r=rx, s=sat, v=velocity, c=clock):
                                        return owner.compute_epoch(r, s, sat_vel=v,
                                            sat_clk=c, rx_vel=[3., 2., 1.], n_samples=samples)
                                calls.append(call)
                            key = f'{kind}/triangles{count}/satellites{n_sat}/samples{samples}/{stage}'
                            row = compare(*calls, args.rounds, args.repeats)
                            value = calls[0]()
                            row['reflection_paths'] = sum(map(len, value)) if stage == 'paths' else value['n_reflection_paths']
                            row['input_sha256'] = tree_hash((model.triangles, rx, sat))
                            if kind == 'wall':
                                assert row['reflection_paths'] > 0
                            results[key] = row
                            print(key, row['baseline_ms'], '->', row['candidate_ms'], 'paths', row['reflection_paths'], flush=True)
                        finally:
                            for sim in sims:
                                sim.sim.release_gpu()
    helpers = {str(Path(m.__file__).resolve().relative_to(ROOT)).replace('\\', '/'): sha(Path(m.__file__))
        for m in list(sys.modules.values()) if getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
        and Path(m.__file__).resolve().is_relative_to(ROOT)}
    helpers['benchmarks/bench_reflection_geometry_compare.py'] = sha(Path(__file__))
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(), source_record_sha256=sha(args.source_record),
        source_sha256=record['source_sha256'], extension_sha256=record['extension_sha256'], helper_sha256=helpers,
        rounds=args.rounds, repeats=args.repeats, measurements=results,
        method='Own process, last allowed CPU, normal priority. One native context and independent Python BuildingModels. Serial alternating complete calls, every returned path/IQ/channel/metadata field hashed outside timing. Nonexclusive host/GPU.',
        scope='Detailed CPU image-method paths with all original triangles/occlusion checks, or complete real CUDA LOS+IQ urban physical4-path epochs. Scene construction excluded; no candidate pruning or cross-call geometry cache; seed37.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
