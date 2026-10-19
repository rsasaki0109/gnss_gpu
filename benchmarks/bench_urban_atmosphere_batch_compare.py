"""Compare exact Python urban pipelines using the same native GPU builds."""

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

from bench_bvh_build_compare import _activate, _measure
from bench_bvh_multipath_compare import _digest, _load_extension
from bench_raytrace import _make_building_mesh, _make_satellites


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timing(before, after, counts, satellites, samples, rounds, repeats, controls):
    from gnss_gpu import atmosphere
    from gnss_gpu.bvh import BVHAccelerator

    class Subclass(atmosphere.AtmosphereCorrection):
        pass

    class Ionosphere(atmosphere.AtmosphereCorrection):
        def iono(self, rx, az, el, gps_time, **params):
            return float(el) * .01

    def wrapped(self, *args):
        return atmosphere.AtmosphereCorrection.tropo(self, *args)

    records = {}
    shapes = [(n, s, z, 'builtin_gpu') for n in counts for s in satellites for z in samples]
    if controls:
        shapes = [(n, 12, 2600, mode) for n in (0, 12000, 120000) for mode in ('none', 'cold_builtin')]
        shapes += [(0, 12, 2600, mode) for mode in ('cpu', 'subclass', 'instance_override', 'iono')]
        shapes += [(0, n, 2600, 'builtin_gpu') for n in (1, 2)]
    rx = np.array([6378137., 0., 0.])
    original_gpu = atmosphere._HAS_GPU
    for count, n_sat, n_samples, mode in shapes:
        models, sims, calls = [], [], []
        sat = _make_satellites(n_sat)[:, [2, 0, 1]] + rx
        clock = np.linspace(-1e-7, 1e-7, n_sat)
        velocity = np.random.default_rng(213).normal(size=(n_sat, 3))*1000
        try:
            atmosphere._HAS_GPU = mode != 'cpu' and original_gpu
            for module in (before, after):
                model = None if not count else BVHAccelerator(_make_building_mesh((count+11)//12)[:count][:, :, [2, 0, 1]]+rx)
                sim = module.UrbanSignalSimulator(model)
                sim.sim.noise_seed = 37
                atmo = None if mode == 'none' else Subclass() if mode == 'subclass' else Ionosphere() if mode == 'iono' else atmosphere.AtmosphereCorrection()
                if mode == 'instance_override':
                    atmo.tropo = types.MethodType(wrapped, atmo)
                kwargs = dict(sat_clk=clock, sat_vel=velocity, rx_vel=np.array([3., -2., .5]),
                    gps_time=12345., n_samples=n_samples, atmo_correction=atmo)
                if mode == 'iono':
                    kwargs['iono_params'] = dict(alpha=[0.]*4, beta=[72000.]*4)

                def call(owner=sim, arguments=kwargs, scene=model, cold=mode == 'cold_builtin'):
                    if not cold:
                        return owner.compute_epoch(rx, sat, **arguments)
                    fresh = type(owner)(scene)
                    fresh.sim.noise_seed = 37
                    try:
                        return fresh.compute_epoch(rx, sat, **arguments)
                    finally:
                        fresh.sim.release_gpu()
                        if scene is not None:
                            scene.release_gpu()

                models.append(model)
                sims.append(sim)
                calls.append(call)
            key = f'{mode}/triangles{count}/satellites{n_sat}/samples{n_samples}'
            records[key] = _measure(*calls, rounds, repeats)
            records[key]['input_sha256'] = _digest(rx, sat, clock, velocity)
            records[key]['channels'] = len(calls[0]()['channels'])
            print(key, records[key]['baseline_ms'], '->', records[key]['candidate_ms'], flush=True)
        finally:
            atmosphere._HAS_GPU = original_gpu
            for sim in sims:
                sim.sim.release_gpu()
            for model in models:
                if model is not None:
                    model.release_gpu()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--counts', type=int, nargs='+', default=[0, 12000, 120000])
    parser.add_argument('--satellites', type=int, nargs='+', default=[12, 32])
    parser.add_argument('--samples', type=int, nargs='+', default=[2600, 26000, 260000])
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    assert args.rounds > 0 and args.repeats > 0 and min(args.counts) >= 0
    assert min(args.satellites) > 0 and max(args.satellites) <= 32 and min(args.samples) > 0
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'python'))
    record = json.loads(args.source_record.read_text(encoding='utf8'))
    for label, sources in record['source_sha256'].items():
        for name, digest in sources.items():
            path = args.baseline_dir/Path(name).name if label == 'baseline' else root/name
            assert _sha(path) == digest, (label, name)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
    kernel.SetProcessAffinityMask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    handle = kernel.GetCurrentProcess()
    assert kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system))
    assert kernel.SetProcessAffinityMask(handle, 1 << (allowed.value.bit_length()-1))
    native = _load_extension(root/'build/diag/Release/_bvh.cp312-win_amd64.pyd', 'same._bvh')
    _activate(native)
    from gnss_gpu import atmosphere
    import gnss_gpu._gnss_gpu_atmosphere as native_atmo
    import gnss_gpu._gnss_gpu_signal_sim as native_signal
    for module in (native, native_atmo, native_signal):
        path = Path(module.__file__)
        assert _sha(path) == record['extension_sha256'][path.name], path
    assert atmosphere._HAS_GPU
    before = _python(args.baseline_dir/'urban_signal_sim.py', 'baseline_urban')
    after = _python(root/'python/gnss_gpu/urban_signal_sim.py', 'candidate_urban')
    measurements = _timing(before, after, args.counts, args.satellites, args.samples, args.rounds, args.repeats, args.controls_only)
    helpers = {str(Path(module.__file__).resolve().relative_to(root)).replace('\\', '/'): _sha(Path(module.__file__))
        for module in list(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(root)}
    helpers['benchmarks/bench_urban_atmosphere_batch_compare.py'] = _sha(Path(__file__))
    result = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_record_sha256=_sha(args.source_record), source_sha256=record['source_sha256'], extension_sha256=record['extension_sha256'],
        benchmark_sha256=helpers, rounds=args.rounds, repeats=args.repeats, measurements=measurements,
        method='Own normal-priority process pinned to last allowed CPU; independent before/after Python modules, separate owners and identical native PEs. Serial alternating complete calls; every returned field and IQ byte checked outside timing; nonexclusive host.',
        scope='Real compute_epoch including atmospheric transfers/allocations/synchronization, full channels, clock/Doppler, ray tracing, CUDA IQ generation and returned arrays. Mesh CPU construction, first-use warm setup and hashing excluded. Cold controls include new urban/signal owner and GPU BVH creation/trace/close but retain the CPU BVH. Seed37.',
        controls_only=args.controls_only)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
