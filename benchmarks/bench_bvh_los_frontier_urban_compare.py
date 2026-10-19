"""Measure actual single-epoch ray APIs and real urban CUDA IQ generation."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from bench_bvh_build_compare import SOURCE_PATHS, _activate, _measure, _sha
from bench_bvh_multipath_compare import _digest, _load_extension
from bench_raytrace import _make_building_mesh, _make_satellites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'baseline-dir', 'source-record', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--counts', type=int, nargs='+', default=[12000, 120000])
    parser.add_argument('--samples', type=int, nargs='+', default=[2600, 26000, 260000])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'python'))
    assert args.rounds > 0 and args.repeats > 0 and min(args.counts) > 0 and min(args.samples) > 0
    edit = json.loads(args.source_record.read_text(encoding='utf8'))
    sources = {}
    for label in ('baseline', 'candidate'):
        sources[label] = {}
        for name in SOURCE_PATHS:
            path = args.baseline_dir/Path(name).name if label == 'baseline' else root/name
            assert hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == edit['source_normalized_lf_sha256'][label][name]
            sources[label][name] = _sha(path)
    extensions = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert extensions == edit['extension_sha256']
    signal = json.loads((root/'benchmarks/signal_wide_static_gpu_20261014.json').read_text())
    for name, digest in signal['source_sha256']['candidate'].items():
        assert _sha(root/name) == digest, name
    signal_path = root/'python/gnss_gpu/_gnss_gpu_signal_sim.cp312-win_amd64.pyd'
    assert _sha(signal_path) == signal['extension_sha256']['candidate']
    if sys.platform == 'win32':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        handle = kernel.GetCurrentProcess()
        assert kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system))
        assert kernel.SetProcessAffinityMask(handle, 1 << (allowed.value.bit_length()-1))
    before = _load_extension(args.baseline.resolve(), 'baseline._bvh')
    after = _load_extension(args.candidate.resolve(), 'candidate._bvh')
    from gnss_gpu.bvh import BVHAccelerator
    from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
    records = {}
    rx = np.array([6378137., 0., 0.])
    for count in args.counts:
        # Apply the same per-vertex coordinate permutation used by the urban
        # baseline fixture, preserving triangle orientation and ECEF units.
        mesh = _make_building_mesh((count+11)//12)[:count][:, :, [2, 0, 1]]+rx
        models = []
        sims = []
        for native in (before, after):
            _activate(native)
            model = BVHAccelerator(mesh)
            model._device_scene = native.DeviceBVH(model._nodes_flat, model._sorted_tris)
            sim = UrbanSignalSimulator(model)
            sim.sim.noise_seed = 37
            models.append(model)
            sims.append(sim)
        try:
            for satellites in (12, 32):
                sat = _make_satellites(satellites)[:, [2, 0, 1]]+rx
                calls = [lambda model=model: list(model.trace(rx, sat)) for model in models]
                key = f'public_single_trace/triangles{count}/satellites{satellites}'
                records[key] = _measure(*calls, args.rounds, 2*args.repeats)
                records[key]['input_sha256'] = _digest(rx, sat, mesh)
                print(key, records[key]['baseline_ms'], '->', records[key]['candidate_ms'], flush=True)
            sat = _make_satellites(12)[:, [2, 0, 1]]+rx
            for samples in args.samples:
                calls = [lambda sim=sim: sim.compute_epoch(rx, sat, n_samples=samples) for sim in sims]
                key = f'urban_real_iq/triangles{count}/satellites12/samples{samples}'
                records[key] = _measure(*calls, args.rounds, args.repeats)
                records[key]['input_sha256'] = _digest(rx, sat, mesh)
                records[key]['channels'] = len(calls[0]()['channels'])
                print(key, records[key]['baseline_ms'], '->', records[key]['candidate_ms'], flush=True)
        finally:
            for sim in sims:
                sim.sim.release_gpu()
                sim.building_model.release_gpu()
    helpers = tuple(sorted({str(Path(module.__file__).resolve().relative_to(root)).replace('\\', '/')
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(root)}))
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_record_sha256=_sha(args.source_record), source_sha256=sources, extension_sha256=extensions,
        signal_extension_sha256=_sha(signal_path), rounds=args.rounds, repeats=args.repeats,
        single_trace_repeats=2*args.repeats, samples=args.samples, counts=args.counts,
        method='Same owned normal-priority process pinned to one allowed CPU; sequential alternating complete synchronized calls; nonexclusive host; every returned field/IQ byte hashed outside timing.',
        scope='Public single trace and real UrbanSignalSimulator.compute_epoch including ray tracing, channels, atmosphere, actual CUDA signal generation, transfers/synchronization and returned arrays. CPU BVH build, initial allocation/code-table setup, and hashing excluded. Same adopted signal build, noise seed37 and full sample/channel counts.',
        benchmark_sha256={name: _sha(root/name) for name in helpers}, measurements=records)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
