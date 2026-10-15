"""Compare signal input packing without unused arrays or successful-validation diagnostic strings.

Old and new production binaries live in separate persistent workers. GPU work
is sequential; measured batches alternate by round. Every output is hashed.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from bench_signal_kernel_compare import ROOT, SOURCE_PATHS, _mixed_channels, _sha, _tree_hash, _Worker as BaseWorker
from bench_signal_global_frequency_compare import _validate as validate_single
from bench_signal_sim_compare import _channels, _input_digest, _load_extension
from bench_raytrace import _make_building_mesh, _make_satellites


def _load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Profile:
    def __init__(self, extension, baseline_dir=None):
        sys.path.insert(0, str(ROOT / 'python'))
        self.native = _load_extension(extension.resolve(), 'gnss_gpu._gnss_gpu_signal_sim')
        sys.modules['gnss_gpu._gnss_gpu_signal_sim'] = self.native
        import gnss_gpu
        gnss_gpu._gnss_gpu_signal_sim = self.native
        source = baseline_dir or ROOT / 'python/gnss_gpu'
        self.signal = _load_python(source / 'signal_sim.py', 'gnss_gpu.signal_sim')
        self.urban_module = _load_python(source / 'urban_signal_sim.py', 'gnss_gpu.urban_signal_sim')
        self.batch = True
        self.workspace = self.native.DeviceSignalWorkspace()
        self.sim = self.signal.SignalSimulator(noise_seed=42)
        self.urban = {}

    def native_epochs(self, rows, samples, seeds=None, noise=-20.):
        if self.batch:
            return self.workspace.generate_epochs(2.6e6, -1.1e6, rows, samples, noise, 42, seeds)
        if not rows:
            return np.empty((0, samples * 2), np.float32)
        return np.stack([self.workspace.generate_signal(2.6e6, -1.1e6, row, samples, noise,
            42 if seeds is None else seeds[e]) for e, row in enumerate(rows)])

    def public_epochs(self, rows, samples):
        if self.batch:
            return self.sim.generate_epochs(rows, samples)
        return np.stack([self.sim.generate_epoch(row, samples) for row in rows])

    def stage(self, name):
        kind, epochs, samples, count = name.split('/')
        epochs, samples, count = int(epochs[6:]), int(samples[7:]), int(count[8:])
        if kind != 'urban':
            rows = [_mixed_channels(count, not kind.endswith('_plain')) for _ in range(epochs)]
            for e, row in enumerate(rows):
                for ch in row:
                    ch['carrier_phase'] += e * .125
            if kind == 'single_control':
                args = (2.6e6, -1.1e6, rows[0], samples, -20., 42)
                return lambda: self.workspace.generate_signal(*args), _input_digest(args)
            call = self.native_epochs if kind.startswith('native') else self.public_epochs
            return lambda: call(rows, samples), _input_digest((rows, samples, 42))
        from gnss_gpu.bvh import BVHAccelerator
        if count not in self.urban:
            rx = np.array([6378137., 0., 0.])
            sats = _make_satellites(12)[:, [2, 0, 1]] + rx
            mesh = _make_building_mesh(count // 12)[:, :, [2, 0, 1]] + rx
            sim = self.urban_module.UrbanSignalSimulator(BVHAccelerator(mesh))
            sim.sim.noise_seed = 37
            self.urban[count] = sim, rx, sats, mesh
        sim, rx, sats, mesh = self.urban[count]
        positions = np.repeat(rx[None], epochs, axis=0)
        positions[:, 1] += np.arange(epochs) * .01
        def call():
            kwargs = dict(n_samples=samples)
            if self.batch:
                kwargs['batch_size'] = 32
            return [[i, result] for i, result in sim.simulate_trajectory(positions,
                lambda i: sats, **kwargs)]
        return call, _input_digest((positions, sats, mesh, samples, 37))

    def validate(self):
        records = validate_single(self.native, self.workspace)
        for samples in (1, 31, 257, 2600, 4095, 4096, 4097):
            for counts in ((0, 1, 12), (12, 32, 33), (0, 65, 256), (1, 256, 257), (0, 2, 32), (12, 31, 32)):
                for noise in (-20., -10000.):
                    for seeds in (None, [1, 37, 2**64 - 1]):
                        rows = [_mixed_channels(n, True) for n in counts]
                        out = self.native_epochs(rows, samples, seeds, noise)
                        name = f'batch/Ns{samples}/channels{counts}/noise{noise}/seeds{seeds}'
                        records[name] = dict(input_sha256=_input_digest((rows, samples, noise, seeds)),
                            output_sha256=_tree_hash(out), shape=list(out.shape), dtype=str(out.dtype))
        for intermediate in (-1e16, 1e16, -1e308, 1e308, 1e-200):
            rows = [_mixed_channels(n, True) for n in (2, 12, 32)]
            for row in rows:
                for index, channel in enumerate(row):
                    channel['doppler_hz'] = -intermediate + index * .125
                    channel['code_phase'] = -2147483648. if index % 2 else 2147483647.
            arguments = (4.092e6, intermediate, rows, 4097, -10000., 37, [1, 37, 2**64-1])
            out = self.workspace.generate_epochs(*arguments)
            records[f'batch_extreme/IF{intermediate}'] = dict(input_sha256=_input_digest(arguments),
                output_sha256=_tree_hash(out), shape=list(out.shape), dtype=str(out.dtype))
        for name in ('native/epochs8/samples257/channels33', 'public/epochs8/samples257/channels12',
                     'urban/epochs8/samples257/channels1200'):
            call, digest = self.stage(name)
            records[name] = dict(input_sha256=digest, output_sha256=_tree_hash(call()))
        class TemporaryNav:
            def __init__(self, values):
                self.values = values
            def __array__(self, dtype=None, copy=None):
                return np.array(self.values, dtype=dtype, copy=True)
        for count in (1,12,33,65):
            for representation in ('list','float64','temporary'):
                base = _mixed_channels(count, False)
                for channel in base:
                    channel.update(nav_bits=np.array([-3,0,2,1,-1],np.int32),nav_bit_rate=2.6e6/3)
                arguments = (2.6e6,-1.1e6,base,257,-20.,37)
                for batch in (False,True):
                    rows = [[dict(channel) for channel in base] for _ in range(3 if batch else 1)]
                    for row in rows:
                        for channel in row:
                            values = channel['nav_bits']
                            channel['nav_bits'] = values.tolist() if representation == 'list' else (
                                values.astype(np.float64) if representation == 'float64' else TemporaryNav(values.tolist()))
                    expected = self.workspace.generate_epochs(*arguments[:2],[base]*3,*arguments[3:]) if batch else self.workspace.generate_signal(*arguments)
                    actual = self.workspace.generate_epochs(*arguments[:2],rows,*arguments[3:]) if batch else self.workspace.generate_signal(*arguments[:2],rows[0],*arguments[3:])
                    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
                    name = f'nav_lifetime/channels{count}/{representation}/batch{batch}'
                    records[name] = dict(input_sha256=_input_digest((arguments,representation,batch)),
                        output_sha256=_tree_hash(actual),shape=list(actual.shape),dtype=str(actual.dtype))
        invalid = [(field,value) for field in ('code_phase','carrier_phase','doppler_hz','amplitude')
                   for value in (np.nan,np.inf,-np.inf)]
        invalid += [('nav_bit_rate',value) for value in (0.,-1.,np.nan,np.inf,-np.inf)]
        invalid += [('prn',0),('system',-1),('precedence',np.nan)]
        for index in (0,2):
            for field,value in invalid:
                row = _mixed_channels(3,False)
                if field == 'precedence':
                    row[index].update(code_phase=value,amplitude=value)
                else:
                    row[index][field] = value
                for batch in (False,True):
                    try:
                        if batch:
                            self.workspace.generate_epochs(2.6e6,0.,[[],row],257,-20.,37)
                        else:
                            self.workspace.generate_signal(2.6e6,0.,row,257,-20.,37)
                    except RuntimeError as error:
                        records[f'error_contract/index{index}/{field}/{value}/batch{batch}'] = dict(
                            input_sha256=_input_digest((row,batch)),exception_type=type(error).__name__,message=str(error))
                    else:
                        raise AssertionError((field,value,index,batch))
        return records

    def measure(self, name, repeats):
        call, digest = self.stage(name)
        expected = _tree_hash(call())
        for _ in range(3):
            assert _tree_hash(call()) == expected
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            out = call()
            samples.append((time.perf_counter_ns() - start) / 1e6)
            assert _tree_hash(out) == expected
        return dict(median_ms=float(np.median(samples)), samples_ms=samples,
            input_sha256=digest, output_sha256=expected, all_timed_outputs_identical=True)

    def close(self):
        self.workspace.close()
        self.sim.release_gpu()
        for sim, *_ in self.urban.values():
            sim.sim.release_gpu()
            sim.building_model.release_gpu()


def serve(args):
    profile = Profile(args.extension, args.baseline_dir)
    print(json.dumps(dict(ready=True)), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request['command'] == 'close':
                break
            try:
                result = profile.validate() if request['command'] == 'validate' else profile.measure(
                    request['stage'], request['repeats'])
                print(json.dumps(dict(result=result)), flush=True)
            except Exception:
                print(json.dumps(dict(error=traceback.format_exc())), flush=True)
    finally:
        profile.close()


class Worker(BaseWorker):
    def __init__(self, extension, baseline_dir=None):
        argv = [sys.executable, str(Path(__file__).resolve()), '--extension', str(extension.resolve())]
        if baseline_dir is not None:
            argv += ['--baseline-dir', str(baseline_dir.resolve())]
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf8', bufsize=1, cwd=ROOT)
        assert self._read()['ready']


def compare(args):
    refpath = 'benchmarks/signal_batch_gpu_20261012.json'
    reference = json.loads((ROOT / refpath).read_text(encoding='utf8'))
    sources = dict(baseline={p: _sha(args.baseline_dir / Path(p).name) for p in SOURCE_PATHS},
        candidate={p: _sha(ROOT / p) for p in SOURCE_PATHS})
    binaries = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert sources['baseline'] == reference['source_sha256']['candidate']
    assert binaries['baseline'] == reference['extension_sha256']['candidate']
    workers, measurements = {}, {}
    try:
        workers['baseline'] = Worker(args.baseline, args.baseline_dir)
        workers['candidate'] = Worker(args.candidate)
        validation = {k: w.request(command='validate') for k, w in workers.items()}
        assert validation['baseline'] == validation['candidate'], 'IQ/metadata bit mismatch'
        print(f"{len(validation['baseline'])} old/new IQ and metadata cases identical", flush=True)
        for name in args.stages:
            runs = {k: [] for k in workers}
            for r in range(args.rounds):
                for k in (('baseline', 'candidate') if r % 2 == 0 else ('candidate', 'baseline')):
                    runs[k].append(workers[k].request(command='measure', stage=name, repeats=args.repeats))
            for key in ('input_sha256', 'output_sha256'):
                assert len({run[key] for values in runs.values() for run in values}) == 1, (name, key)
            medians = {k: [r['median_ms'] for r in values] for k, values in runs.items()}
            old, new = [float(np.median(medians[k])) for k in workers]
            measurements[name] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                round_medians_ms=medians, runs=runs)
            print(f'{name}: {old:.4f} -> {new:.4f} ms ({old/new:.3f}x)', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = ['benchmarks/bench_signal_input_packing_compare.py','benchmarks/bench_signal_batch_descriptor_compare.py', 'benchmarks/bench_signal_batch_compare.py', 'benchmarks/bench_signal_global_frequency_compare.py',
        'benchmarks/bench_signal_kernel_compare.py', 'benchmarks/bench_signal_sim_compare.py',
        'benchmarks/bench_bvh_combined_compare.py', 'benchmarks/bench_raytrace.py']
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        baseline='preceding epoch-batch signal build; channel parsing default-constructs one empty NumPy array per channel',
        baseline_reference=dict(path=refpath, sha256=_sha(ROOT/refpath)), source_sha256=sources,
        extension_sha256=binaries, benchmark_sha256={p: _sha(ROOT/p) for p in helpers},
        bvh_extension_sha256={str(p.relative_to(ROOT)): _sha(p) for p in (ROOT/'python/gnss_gpu').glob('_bvh*.pyd')},
        build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
        method='persistent separate production workers; sequential GPU work; alternating stage batches; nonexclusive host',
        scope='complete epoch group: Python conversion/validation, transfers, kernels, synchronization, returned IQ; '
            'urban includes scene metadata/LOS/reflection and trajectory yields; setup, code-table/BVH construction, IPC/hashes excluded',
        rounds=args.rounds, repeats=args.repeats, warmup_calls=3, reference_calls=1,
        validation=dict(cases=len(validation['baseline']), all_iq_and_metadata_bits_identical=True,
            baseline_by_case=validation['baseline'], candidate_by_case=validation['candidate']), measurements=measurements)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('extension', 'baseline', 'candidate', 'baseline-dir', 'output'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--stages', nargs='+', default=[
        f'{kind}/epochs{epochs}/samples{samples}/channels{count}'
        for kind, epochs, samples, count in (
            ('native', 8, 2600, 12), ('native', 32, 2600, 12), ('native', 32, 2600, 65),
            ('native', 8, 260000, 12), ('native', 8, 260000, 257),
            ('public', 8, 2600, 12), ('public', 32, 2600, 12), ('public', 32, 2600, 65),
            ('public', 8, 260000, 12), ('urban', 8, 2600, 1200), ('urban', 32, 2600, 1200),
            ('urban', 8, 260000, 1200), ('single_control', 1, 2600, 12))])
    args = p.parse_args()
    if args.extension:
        serve(args)
    else:
        if not all((args.baseline, args.candidate, args.baseline_dir, args.output)):
            p.error('comparison needs baseline, candidate, baseline-dir and output')
        if min(args.rounds, args.repeats) < 1:
            p.error('rounds and repeats must be positive')
        report = compare(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
