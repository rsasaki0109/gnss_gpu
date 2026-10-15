"""Compare exact Doppler geometry reuse with64-bit validity for33..64 satellites using actual production builds."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from bench_pf_dd_reference_compare import _digest, _inputs, _load_extension, _snapshot


def _stages():
    return [f'doppler_sat{sat}_gain{gain}' for sat in (33,48,49,64) for gain in (0,.25)] + [
        f'public_doppler_sat{sat}_gain{gain}' for sat in (33,48,49,64) for gain in (0,.25)] + [
        'doppler_sat32_gain0.25','doppler_sat65_gain0.25',
        'public_doppler_sat32_gain0.25','public_doppler_sat65_gain0.25','public_PR_control']


def _update(native,state,observations,gain=.25,sigma=.5,wavelength=.19,maximum=10.):
    native.pf_device_weight_doppler(state,*observations,len(observations[3]),wavelength,sigma,gain,maximum)


def _validate(native):
    from bench_pf_doppler_large_cache_compare import _validate as preceding_validate
    from bench_pf_dd_reference_compare import _initialize
    from bench_pf_doppler_compare import _observations
    records = preceding_validate(native)
    for count in (1,33,127,128,129,257,8193):
        owner = native.pf_device_create(count)
        try:
            _initialize(native,owner)
            states,prior = _snapshot(native,owner)
            states[:,4:7] = np.random.default_rng(738+count).normal(size=(count,3))
            for satellites in (33,47,48,49,63,64,65):
                for variant in ('normal','masked','near','nonfinite','last_only'):
                    arrays = [a.copy() for a in _observations(satellites)]
                    if variant=='masked':
                        arrays[3][::3] = 0.
                        arrays[3][1::3] = -1.
                    elif variant=='near':
                        arrays[0][0] = states[0,:3] + [.5,0.,0.]
                        arrays[0][1] = states[0,:3] + [1.,0.,0.]
                    elif variant=='nonfinite':
                        arrays[0][::5,0] = np.nan
                        arrays[1][::7,1] = np.inf
                        arrays[3][::3] = np.nan
                        arrays[3][1::3] = np.inf
                    elif variant=='last_only':
                        arrays[0][:-1] = states[0,:3]
                        arrays[3][:-1] = 0.
                    for gain in (0.,.25,1.5,np.nan):
                        native.pf_device_set_particle_states(owner,states)
                        native.pf_device_set_log_weights(owner,prior)
                        for _ in range(2):
                            _update(native,owner,arrays,gain)
                        actual,weights = _snapshot(native,owner)
                        np.testing.assert_array_equal(actual[:,:4].view(np.uint64),states[:,:4].view(np.uint64))
                        np.testing.assert_array_equal(actual[:,7:].view(np.uint64),states[:,7:].view(np.uint64))
                        records[f'doppler_wide_cache/N{count}/sat{satellites}/{variant}/gain{gain}'] = dict(
                            input_sha256=_digest(states,prior,*arrays,np.array([gain])),output_sha256=_digest(actual,weights))
            _initialize(native,owner)
            inputs = []
            for epoch,satellites in enumerate((32,33,48,49,64,65,96,8),1):
                native.pf_device_predict(owner,1.,2.,3.,.1,1.,3.,42,epoch)
                arrays = [a.copy() for a in _observations(satellites)]
                arrays[2] += epoch*.125
                inputs.append(_digest(*arrays))
                _update(native,owner,arrays,.25 if epoch%2 else 0.)
                for array in arrays:
                    array.fill(-999.)
                if epoch%3==0:
                    native.pf_device_resample_megopolis_coalesced(owner,7,42)
            records[f'doppler_wide_cache/N{count}/queued_growth_mutation_resample'] = dict(
                input_sha256=inputs,output_sha256=_digest(*_snapshot(native,owner),native.pf_device_estimate(owner)))
        finally:
            native.pf_device_destroy(owner)
    return records


def _profile(native,count,stage,repeats):
    from bench_pf_dd_reference_compare import _initialize,_measure
    from bench_pf_doppler_compare import _observations
    from bench_pf_coalesced_compare import _scenario
    import re
    match = re.fullmatch(r'(public_)?doppler_sat(\d+)_gain(.*)',stage)
    if match:
        n_sat,gain = int(match[2]),float(match[3])
        observations = _observations(n_sat)
    else:
        assert stage=='public_PR_control'
    if match and not match[1]:
        state = native.pf_device_create(count)
        try:
            return _measure(native,state,lambda:_initialize(native,state),
                            lambda:_update(native,state,observations,gain),repeats,count)
        finally:
            native.pf_device_destroy(state)
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    position,sat,pr,weights = _scenario()
    pf = ParticleFilterDevice(n_particles=count,sigma_pos=1.,sigma_cb=3.,sigma_pr=5.,seed=42)
    try:
        def prepare():
            pf.initialize(position,clock_bias=3000.,spread_pos=10.,spread_cb=30.,
                          velocity=[1.,2.,3.],spread_vel=.5,velocity_init_sigma=2.)
        def operation():
            pf.predict([1.,2.,3.],dt=.1)
            pf.update(sat,pr,weights,resample=False)
            if match:
                pf.update_doppler(*observations[:3],weights=observations[3],wavelength=.19,
                                  sigma_mps=.5,velocity_update_gain=gain,resample=False)
            ess = pf.get_ess()
            resampled = pf.resample_if_needed()
            return np.r_[pf.estimate(),ess,float(resampled)]
        return _measure(native,pf._state,prepare,operation,repeats,count)
    finally:
        native.pf_device_destroy(pf._state)


def _worker(extension):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'python'))
    native = _load_extension(extension.resolve())
    print(json.dumps(dict(ready=True)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request['mode'] == 'close':
            return
        try:
            result = _validate(native) if request['mode'] == 'validate' else _profile(
                native, request['size'], request['stage'], request['repeats'])
            print(json.dumps(result), flush=True)
        except Exception as error:
            print(json.dumps(dict(error=repr(error))), flush=True)


class _Worker:
    def __init__(self, extension):
        self.process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()), '--extension', str(extension.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        assert json.loads(self.process.stdout.readline()) == dict(ready=True)
    def request(self, **args):
        self.process.stdin.write(json.dumps(args) + '\n')
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps(dict(mode='close')) + '\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=30)


def _compare(args):
    root = Path(__file__).resolve().parents[1]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    reference_path = 'benchmarks/pf_doppler_large_cache_gpu_20261013.json'
    reference = json.loads((root / reference_path).read_text(encoding='utf-8'))
    files = list(reference['source_sha256']['candidate'])
    sources = dict(baseline={name: sha(args.baseline_dir / Path(name).name) for name in files},
                   candidate={name: sha(root / name) for name in files})
    extensions = dict(baseline=sha(args.baseline), candidate=sha(args.candidate))
    assert sources['baseline'] == reference['source_sha256']['candidate']
    assert extensions['baseline'] == reference['extension_sha256']['candidate']
    assert all(sources['baseline'][name] == sources['candidate'][name] for name in files[1:])
    workers = {}
    runs, summary = {}, {}
    try:
        for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
            workers[label] = _Worker(path)
        validations = {label: worker.request(mode='validate') for label, worker in workers.items()}
        if validations['baseline'] != validations['candidate']:
            mismatches = {key: {label: values.get(key) for label, values in validations.items()}
                          for key in validations['baseline'].keys() | validations['candidate'].keys()
                          if validations['baseline'].get(key) != validations['candidate'].get(key)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(dict(source_sha256=sources, extension_sha256=extensions,
                                                   mismatches=mismatches), indent=2) + '\n', encoding='utf-8')
        assert validations['baseline'] == validations['candidate'], 'Input/state/log-weight/trajectory bit mismatch'
        print(f"{len(validations['baseline'])} validation cases are bit-identical", flush=True)
        if not args.validation_only:
            for count in args.sizes:
                for stage in args.stages or _stages():
                    key = f'N{count}/{stage}'
                    data = {label: [] for label in workers}
                    for round_index in range(args.rounds):
                        order = ('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')
                        for label in order:
                            data[label].append(workers[label].request(
                                mode='measure', size=count, stage=stage, repeats=args.repeats))
                    for field in ('output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                        assert len({r[field] for values in data.values() for r in values}) == 1, (key, field)
                    medians = {label: [r['median_ms'] for r in values] for label, values in data.items()}
                    old, new = [float(np.median(medians[label])) for label in workers]
                    summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                                        round_medians_ms=medians, output_sha256=data['baseline'][0]['output_sha256'])
                    runs[key] = data
                    print(f'{key}: {old:.4f} -> {new:.4f} ms ({old/new:.3f}x)', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = ('benchmarks/bench_pf_doppler_wide_cache_compare.py','benchmarks/bench_pf_doppler_large_cache_compare.py','benchmarks/bench_pf_doppler_weight_cache_compare.py','benchmarks/bench_pf_packed_satellite_compare.py','benchmarks/bench_pf_doppler_compare.py', 'benchmarks/bench_pf_pr_gate_compare.py', 'benchmarks/bench_pf_gmm_compare.py', 'benchmarks/bench_pf_dd_mixed_joint_gate_compare.py', 'benchmarks/bench_pf_dd_mixed_gate_compare.py', 'benchmarks/bench_pf_dd_single_huber_gate_compare.py',
               'benchmarks/bench_pf_dd_single_gate_compare.py',
               'benchmarks/bench_pf_dd_huber_gate_compare.py',
               'benchmarks/bench_pf_dd_gate_compare.py',
               'benchmarks/bench_pf_dd_sigma_compare.py',
               'benchmarks/bench_pf_dd_joint_reference_compare.py',
               'benchmarks/bench_pf_dd_joint_options_check.py',
               'benchmarks/bench_pf_dd_reference_compare.py', 'benchmarks/bench_pf_coalesced_compare.py',
               'benchmarks/bench_pf_covariance_compare.py', 'benchmarks/bench_pf_gaussian_compare.py',
               'python/gnss_gpu/pf_device_runtime.py', 'python/gnss_gpu/pf_device_config.py', 'python/gnss_gpu/pf_device_smoother.py')
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
                baseline='preceding cache8/cache16/cache24/cache32 build;33..64 satellites repeat original geometry in up to three loops',
                baseline_reference=dict(path=reference_path, sha256=sha(root / reference_path)),
                method='persistent actual production workers; sequential stage batches; alternating order per round; nonexclusive host',
                scope='native Doppler/WLS updates and public predict/PR/Doppler/ESS/conditional B60 resample/estimate; all synchronization included; initialization/restoration/IPC/output checks excluded; unchanged public PR control',
                rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, warmup_calls_per_stage=3,
                source_sha256=sources, extension_sha256=extensions,
                benchmark_sha256={name: sha(root / name) for name in helpers},
                validation=dict(cases=len(validations['baseline']), all_outputs_bit_identical=True,
                                baseline_by_case=validations['baseline'], candidate_by_case=validations['candidate']),
                summary=summary, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--validation-only', action='store_true')
    parser.add_argument('--stages', nargs='+')
    parser.add_argument('--sizes', type=int, nargs='+', default=[2048, 10000, 100000, 1000000])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.rounds < 1 or args.repeats < 1:
        parser.error('sizes, rounds and repeats must be positive')
    if args.extension:
        _worker(args.extension)
        return
    if not (args.baseline and args.candidate and args.baseline_dir and args.output):
        parser.error('comparison requires baseline, candidate, baseline-dir and output')
    report = _compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
