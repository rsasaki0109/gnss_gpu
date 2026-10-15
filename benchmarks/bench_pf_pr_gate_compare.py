"""Compare single-pass undifferenced PR gates against the preceding exact-GMM build.

Actual persistent production workers run sequentially in alternating order.
The preceding comparator supplies unchanged measurement operations and its
validation cases. Additional cases cover odd tails, sigma and joint extremes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from bench_pf_dd_reference_compare import _digest, _inputs, _load_extension, _scenario, _snapshot


def _stages():
    return [f'PR_{mode}_sat{sat}' for mode in ('gaussian', 'huber', 'student') for sat in (4, 12, 32)] + [
        f'PR_{mode}_{scope}_step' for mode in ('gaussian', 'huber', 'student') for scope in ('public', 'native')] + [
        'GMM_public_step', 'mixed_joint_public_gate_step', 'Student_control', 'resample_control']


def _params(mode):
    return (5., 3. if mode == 'student' else 0., 30., mode == 'huber', 1.5)


def _validate(native):
    from bench_pf_gmm_compare import _validate as preceding_validate
    from bench_pf_gaussian_compare import _pr, _weight
    records = preceding_validate(native)
    parameters = [(5., 0., 30., False, 1.5), (5., 0., 30., True, 1.5),
        (5., 3., 30., False, 1.5), (5., 2.5, 30., True, 1.5),
        (.75, .1, 1e-100, True, .75), (1e-200, 0., 30., False, 1.5),
        (1e200, 3., 30., True, 1.5), (5., -3., 30., True, 0.),
        (5., 0., -1., True, 1.5), (5., 1e200, 30., True, 1.5),
        (5., 0., 30., True, -1.), (np.nextafter(0., 1.), 0., 30., True, 1.5)]
    for count in (1, 31, 257):
        rng = np.random.default_rng(count + 641)
        original = rng.normal(size=(count, 16))
        original[:, :3] += _scenario()[0]
        original[:, 3] += 3000.
        prior = rng.normal(size=count)
        state = native.pf_device_create(count)
        try:
            for satellites in (1, 3, 4, 5, 33, 128):
                for variant in ('partial', 'too_few', 'zero_weights', 'negative_weights', 'nonfinite_state', 'geometry_overflow', 'nonfinite_obs'):
                    states = original.copy()
                    obs = [v.copy() for v in _pr(satellites)]
                    obs[1][::5] += 50.
                    if variant == 'too_few':
                        obs[1][:] += 1000.
                    elif variant == 'zero_weights':
                        obs[2][:] = 0.
                    elif variant == 'negative_weights':
                        obs[2] *= -1.
                    elif variant == 'nonfinite_state':
                        states[:, :4] = np.resize([np.nan, np.inf, -np.inf, -0.], (count, 4))
                    elif variant == 'geometry_overflow':
                        obs[0][0, 0] = 1e308
                    elif variant == 'nonfinite_obs':
                        obs[1][0] = np.nan
                    for index, arguments in enumerate(parameters):
                        native.pf_device_set_particle_states(state, states)
                        native.pf_device_set_log_weights(state, prior)
                        rejected = None
                        try:
                            _weight(native, state, obs, *arguments)
                        except RuntimeError as error:
                            if variant not in ('negative_weights', 'nonfinite_obs'):
                                raise
                            rejected = str(error)
                        actual, weights = _snapshot(native, state)
                        records[f'PRgate/N{count}/sat{satellites}/{variant}/parameters{index}'] = dict(
                            input_sha256=_digest(states, prior, *obs, np.asarray(arguments)), output_sha256=_digest(actual, weights), rejected=rejected)
            native.pf_device_set_particle_states(state, original)
            history = []
            for satellites, mode in ((3, 'gaussian'), (128, 'huber'), (5, 'student'), (1228, 'gaussian'), (4, 'student'), (12, 'huber')):
                _weight(native, state, _pr(satellites), *_params(mode))
                history.extend(_snapshot(native, state))
            records[f'PRgate/N{count}/capacity_mode_growth'] = dict(input_sha256=_digest(original), output_sha256=_digest(*history))
        finally:
            native.pf_device_destroy(state)
    for count in (2048, 10000, 100000, 1000000):
        for mode in ('gaussian', 'huber', 'student'):
            state = native.pf_device_create(count)
            try:
                from bench_pf_dd_reference_compare import _initialize
                _initialize(native, state)
                obs = [v.copy() for v in _pr(12)]
                obs[1][::5] += 50.
                history = []
                for step in range(1, 6):
                    native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, step)
                    _weight(native, state, obs, *_params(mode))
                    ess = native.pf_device_ess(state)
                    resampled = ess < .5 * count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state, 60, 42+step)
                    history.append(np.array([ess, resampled, *native.pf_device_estimate(state)]))
                records[f'PRgate/N{count}/{mode}/five_epoch'] = dict(input_sha256=_digest(*obs), output_sha256=_digest(*history, *_snapshot(native, state)))
            finally:
                native.pf_device_destroy(state)
    return records


def _profile(native, count, stage, repeats):
    if not stage.startswith('PR_'):
        from bench_pf_gmm_compare import _profile as preceding_profile
        return preceding_profile(native, count, stage, repeats)
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    from bench_pf_dd_reference_compare import _initialize, _measure
    from bench_pf_gaussian_compare import _pr, _weight
    mode = stage.split('_')[1]
    obs = [v.copy() for v in _pr(int(stage.split('sat')[1]) if '_sat' in stage else 12)]
    obs[1][::5] += 50.
    parameters = _params(mode)
    if '_public_' in stage:
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42,
            nu=parameters[1], per_particle_nlos_gate=True, per_particle_nlos_undiff_pr_threshold_m=30.,
            per_particle_huber=parameters[3], per_particle_huber_undiff_pr_k=1.5)
        try:
            def prepare():
                pf.initialize(_scenario()[0], clock_bias=3000., spread_pos=10., spread_cb=30.,
                    velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)
            def operation():
                pf.predict([1., 2., 3.], dt=.1)
                pf.update(*obs[:2], weights=obs[2])
                return pf.estimate()
            return _measure(native, pf._state, prepare, operation, repeats, count)
        finally:
            native.pf_device_destroy(pf._state)
            pf._state = None
    state = native.pf_device_create(count)
    try:
        def operation():
            if '_sat' in stage:
                return _weight(native, state, obs, *parameters)
            native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1)
            _weight(native, state, obs, *parameters)
            ess = native.pf_device_ess(state)
            resampled = ess < .5 * count
            if resampled:
                native.pf_device_resample_megopolis_coalesced(state, 60, 42)
            return np.array([ess, resampled, *native.pf_device_estimate(state)])
        return _measure(native, state, lambda: _initialize(native, state), operation, repeats, count)
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
    reference_path = 'benchmarks/pf_gmm_gpu_20261013.json'
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
    helpers = ('benchmarks/bench_pf_pr_gate_compare.py', 'benchmarks/bench_pf_gmm_compare.py', 'benchmarks/bench_pf_dd_mixed_joint_gate_compare.py', 'benchmarks/bench_pf_dd_mixed_gate_compare.py', 'benchmarks/bench_pf_dd_single_huber_gate_compare.py',
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
                baseline='preceding exact GMM; positive undifferenced PR gates compute geometry in two loops',
                baseline_reference=dict(path=reference_path, sha256=sha(root / reference_path)),
                method='persistent actual production workers; sequential stage batches; alternating order per round; nonexclusive host',
                scope='synchronized native Gaussian/Huber/Student positive-PR-gate weight updates and native/public predict/PR/ESS/conditional B60 resample/estimate plus unchanged GMM/DD/control paths; '
                      'initialization/restoration, output checks and IPC excluded; all timed log weights and returned values hashed',
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
