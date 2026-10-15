"""Compare exact bounded geometry reuse in velocity KF updates for33..64 satellites."""

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
    return [f'KF_sat{n}' for n in (33,48,49,64)] + [
        f'public_KF_sat{n}' for n in (33,48,49,64)] + [
        'KF_sat32','KF_sat65','public_KF_sat32','public_KF_sat65','public_PR_step']


def _validate(native):
    from bench_pf_doppler_wide_cache_compare import _validate as wide_validate
    from bench_pf_predict_compare import _validate as predict_validate
    from bench_pf_doppler_compare import _validate as kf_validate
    records = wide_validate(native)
    for prefix, validator in (('covariance_planes_predict', predict_validate),
                              ('covariance_planes_KF', kf_validate)):
        records.update({prefix+'/'+name: value for name, value in validator(native).items()})
    from bench_pf_coalesced_compare import _initialize
    from bench_pf_doppler_compare import _kf, _observations
    for count in (1,33,127,128,129,257,8193):
        owner = native.pf_device_create(count)
        try:
            _initialize(native,owner)
            states,prior = _snapshot(native,owner)
            rng = np.random.default_rng(438+count)
            factors = rng.normal(size=(count,3,3))
            states[:,7:] = (factors @ factors.swapaxes(1,2)+np.eye(3)*.1).reshape(count,9)
            for satellites in (33,47,48,49,63,64,65):
                for variant in ('normal','masked','near','nonfinite','last_only','nonsymmetric','invalid_S'):
                    arrays = [a.copy() for a in _observations(satellites)]
                    initial = states.copy()
                    if variant == 'masked':
                        arrays[3][::3] = 0.
                        arrays[3][1::3] = -1.
                    elif variant == 'near':
                        arrays[0][0] = initial[0,:3]+[.5,0.,0.]
                        arrays[0][1] = initial[0,:3]+[1.,0.,0.]
                    elif variant == 'nonfinite':
                        arrays[0][::5,0] = np.nan
                        arrays[1][::7,1] = np.inf
                        arrays[3][::3] = np.nan
                    elif variant == 'last_only':
                        arrays[3][:-1] = 0.
                    elif variant == 'nonsymmetric':
                        initial[:,7:] = [4.,1.,.5,9.,3.,.25,-7.,8.,2.]
                    elif variant == 'invalid_S':
                        initial[:,7:] = -np.eye(3).ravel()*1e10
                    native.pf_device_set_particle_states(owner,initial)
                    native.pf_device_set_log_weights(owner,prior)
                    for _ in range(2):
                        _kf(native,owner,arrays)
                    records[f'KF_wide/N{count}/sat{satellites}/{variant}'] = dict(
                        input_sha256=_digest(initial,prior,*arrays),output_sha256=_digest(*_snapshot(native,owner)))
            _initialize(native,owner)
            for epoch,satellites in enumerate((32,33,48,49,64,65,96,8),1):
                native.pf_device_predict(owner,1.,2.,3.,.1,1.,3.,42,epoch,0.,.3,True,.1)
                arrays = [a.copy() for a in _observations(satellites)]
                _kf(native,owner,arrays)
                for array in arrays:
                    array.fill(-999.)
                if epoch % 3 == 0:
                    native.pf_device_resample_megopolis_coalesced(owner,7,42)
            records[f'KF_wide/N{count}/queued_predict_growth_mutation_resample'] = dict(
                output_sha256=_digest(*_snapshot(native,owner),native.pf_device_estimate(owner)))
        finally:
            native.pf_device_destroy(owner)
    return records


def _profile(native, count, stage, repeats):
    from bench_pf_dd_reference_compare import _measure
    from bench_pf_coalesced_compare import _initialize, _scenario
    from bench_pf_doppler_compare import _kf, _observations
    import re
    position, sat, pr, weights = _scenario()
    match = re.fullmatch(r'(?:public_)?KF_sat(\d+)', stage)
    observations = _observations(int(match[1]) if match else 12)
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    if stage.startswith('public_'):
        from gnss_gpu.particle_filter_device import ParticleFilterDevice
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare():
                pf.initialize(position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1.,2.,3.], spread_vel=.5, velocity_init_sigma=2.)
            def operation():
                kf = stage != 'public_PR_step'
                pf.predict([1.,2.,3.], dt=.1, velocity_guide_alpha=.3 if kf else 1.,
                           rbpf_velocity_kf=kf, velocity_process_noise=.1 if kf else 0.)
                pf.update(sat, pr, weights, resample=False)
                if kf:
                    pf.update_doppler_kf(*observations[:3], weights=observations[3],
                                         wavelength=.19, sigma_mps=.5, resample=False)
                ess = pf.get_ess()
                resampled = pf.resample_if_needed()
                return np.r_[pf.estimate(), ess, float(resampled)]
            return _measure(native, pf._state, prepare, operation, repeats, count)
        finally:
            native.pf_device_destroy(pf._state)
    state = native.pf_device_create(count)
    try:
        _initialize(native, state)
        imported = native.pf_device_get_particle_states(state)
        rng = np.random.default_rng(count)
        factors = rng.normal(size=(count,3,3))
        imported[:,7:] = (factors @ factors.swapaxes(1,2) + np.eye(3)*.1).reshape(count,9)
        prior = rng.normal(0.,3.,count)
        def prepare():
            _initialize(native, state)
            if 'imported' in stage:
                native.pf_device_set_particle_states(state, imported)
            if 'resample' in stage or stage.startswith(('systematic','legacy')):
                native.pf_device_set_log_weights(state, prior)
        def predict(kf=True):
            native.pf_device_predict(state,1.,2.,3.,.1,1.,3.,42,1,0.,.3 if kf else 1.,kf,.1 if kf else 0.)
        def operation():
            if stage.startswith('predict'):
                predict()
            elif match:
                _kf(native,state,observations)
            elif stage.startswith('resample'):
                native.pf_device_resample_megopolis_coalesced(state,60,42)
            elif stage.startswith('systematic'):
                native.pf_device_resample_systematic(state,42)
            elif stage.startswith('legacy'):
                native.pf_device_resample_megopolis(state,7,42)
            else:
                kf = stage == 'native_PR_KF_step'
                predict(kf)
                native.pf_device_weight(state,sat,pr,weights,len(weights),5.)
                if kf:
                    _kf(native,state,observations)
                ess = native.pf_device_ess(state)
                resampled = ess < .5*count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state,60,42)
                return np.r_[native.pf_device_estimate(state),ess,float(resampled)]
        return _measure(native,state,prepare,operation,repeats,count)
    finally:
        native.pf_device_destroy(state)


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
    reference_path = 'benchmarks/pf_doppler_wide_cache_gpu_20261013.json'
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
    helpers = ('benchmarks/bench_pf_kf_wide_cache_compare.py','benchmarks/bench_pf_predict_compare.py','benchmarks/bench_pf_doppler_wide_cache_compare.py','benchmarks/bench_pf_doppler_large_cache_compare.py','benchmarks/bench_pf_doppler_weight_cache_compare.py','benchmarks/bench_pf_packed_satellite_compare.py','benchmarks/bench_pf_doppler_compare.py', 'benchmarks/bench_pf_pr_gate_compare.py', 'benchmarks/bench_pf_gmm_compare.py', 'benchmarks/bench_pf_dd_mixed_joint_gate_compare.py', 'benchmarks/bench_pf_dd_mixed_gate_compare.py', 'benchmarks/bench_pf_dd_single_huber_gate_compare.py',
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
                baseline='preceding wide Doppler likelihood cache build; velocity KF above32 satellites repeats geometry in both passes',
                baseline_reference=dict(path=reference_path, sha256=sha(root / reference_path)),
                method='persistent actual production workers; sequential stage batches; alternating order per round; nonexclusive host',
                scope='native covariance prediction/KF/resampling and complete native/public predict/PR/KF/ESS/conditional B60 resample/estimate; synchronization included; initialization/restoration/IPC/output hashes excluded; PR control',
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
