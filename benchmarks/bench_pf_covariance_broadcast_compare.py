"""Compare synchronous covariance broadcast with complete covariance-guided PF steps."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from bench_pf_common_predict_compare import Session as BaseSession
from bench_pf_common_predict_compare import _validate as validate_previous
from bench_pf_dd_reference_compare import _digest, _load_extension, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from pf_covariance_broadcast_cases import COUNTS, PREDICTIONS, covariance, initial_arrays, invalid_covariances, representation

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_python(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _setup(native, python_dir=None):
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    import gnss_gpu
    gnss_gpu._gnss_gpu_pf_device = native
    folder = python_dir or ROOT/'python/gnss_gpu'
    for stem in ('pf_device_config', 'pf_device_runtime', 'particle_filter_device'):
        module = _load_python(folder/(stem+'.py'), 'gnss_gpu.'+stem)
        setattr(gnss_gpu, stem, module)
    return sys.modules['gnss_gpu.particle_filter_device'].ParticleFilterDevice


def _new_pf(klass, count):
    pf = klass(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
    pf.initialize([6378137., 0., 0.], clock_bias=3000., spread_pos=10., spread_cb=30.,
                  velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)
    return pf


def _restore(pf, states, weights):
    pf.set_particle_states(states)
    pf.set_log_weights(weights)


def _bits_pair(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


def _broadcast_validate(native, klass):
    records = {}
    for count in COUNTS:
        actual, original = _new_pf(klass, count), _new_pf(klass, count)
        original._pf_device_set_velocity_covariance = None
        try:
            states, weights = initial_arrays(actual.get_particle_states())
            for case in range(8):
                matrix = covariance(case)
                for form in range(3):
                    arg = representation(matrix, form)
                    for pf in (actual, original):
                        _restore(pf, states, weights)
                        pf.set_velocity_covariance(arg)
                    arg.fill(-999.)
                    a, b = _snapshot(native, actual._state), _snapshot(native, original._state)
                    _bits_pair(a, b)
                    np.testing.assert_array_equal(a[0][:, :7].view(np.uint64), states[:, :7].view(np.uint64))
                    np.testing.assert_array_equal(a[0][:, 7:].view(np.uint64), np.broadcast_to(matrix.ravel(), (count, 9)).view(np.uint64))
                    np.testing.assert_array_equal(a[1].view(np.uint64), weights.view(np.uint64))
                    name = f'broadcast/N{count}/case{case}/form{form}'
                    records[name] = dict(input_sha256=_digest(states, weights, matrix), output_sha256=_digest(*a),
                        retained_roundtrip_oracle_identical=True)
                    for index, params in enumerate(PREDICTIONS):
                        for pf in (actual, original):
                            _restore(pf, states, weights)
                            pf.set_velocity_covariance(representation(matrix, form))
                            alpha, dt, sp, sc, q, sv = params
                            pf.sigma_cb = sc
                            pf.predict([1., 2., 3.], dt=dt, sigma_pos=sp, sigma_vel=sv,
                                velocity_guide_alpha=alpha, rbpf_velocity_kf=True, velocity_process_noise=q)
                        a, b = _snapshot(native, actual._state), _snapshot(native, original._state)
                        _bits_pair(a, b)
                        records[f'{name}/predict{index}'] = dict(input_sha256=_digest(states, weights, matrix, np.asarray(params)),
                            output_sha256=_digest(*a), retained_individual_prediction_oracle_identical=True)
            for pf in (actual, original):
                pf.initialize([6378137., 0., 0.], velocity=[1., 2., 3.], spread_vel=.5,
                    velocity_init_sigma=2., spread_pos=10., spread_cb=30.)
            history = []
            for step, sat in enumerate((8, 12, 33, 64, 65, 16), 1):
                matrix = covariance(step % 4)
                for pf in (actual, original):
                    pf.set_velocity_covariance(matrix)
                    pf.predict([1.+step, 2.-step, 3./step], dt=.17, velocity_guide_alpha=.3,
                        rbpf_velocity_kf=True, velocity_process_noise=.13)
                    arrays = [a.copy() for a in _observations(sat)]
                    _kf(native, pf._state, arrays)
                    for arr in arrays:
                        arr.fill(-999.)
                    if step % 2 == 0:
                        native.pf_device_resample_megopolis_coalesced(pf._state, 7, 42+step)
                a, b = _snapshot(native, actual._state), _snapshot(native, original._state)
                _bits_pair(a, b)
                history.extend(a)
            records[f'broadcast/N{count}/queued_KF_growth_mutation_resample'] = dict(output_sha256=_digest(*history),
                retained_roundtrip_oracle_identical=True)
            for index, arg in enumerate(invalid_covariances()):
                before = _snapshot(native, actual._state)
                try:
                    actual.set_velocity_covariance(arg)
                except (ValueError, TypeError) as error:
                    records[f'broadcast/N{count}/invalid{index}'] = dict(exception_type=type(error).__name__, message=str(error))
                else:
                    raise AssertionError('Invalid covariance accepted')
                _bits_pair(before, _snapshot(native, actual._state))
            fresh = klass(n_particles=count)
            try:
                try:
                    fresh.set_velocity_covariance(np.eye(3))
                except RuntimeError as error:
                    records[f'broadcast/N{count}/uninitialized'] = dict(exception_type=type(error).__name__, message=str(error))
                else:
                    raise AssertionError('Uninitialized covariance setter accepted')
            finally:
                native.pf_device_destroy(fresh._state)
                fresh._state = None
        finally:
            for pf in (actual, original):
                native.pf_device_destroy(pf._state)
                pf._state = None
    return records


class Session(BaseSession):
    def __init__(self, native, count, stage):
        super().__init__(native, count, stage)
        self.covariance = covariance(0)
        self.input_sha256 = _digest(np.array([count]), self.position, self.sat, self.pr, self.weights,
                                   *self.obs, self.covariance)

    def operation(self):
        if not self.stage.startswith('public_covariance_'):
            return super().operation()
        if self.stage == 'public_covariance_set':
            self.pf.set_velocity_covariance(self.covariance)
            return np.empty(0)
        history = []
        for _ in range(5 if self.stage.endswith('_five') else 1):
            self.pf.set_velocity_covariance(self.covariance)
            self.pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3,
                rbpf_velocity_kf=True, velocity_process_noise=.1)
            if self.stage == 'public_covariance_predict':
                return np.empty(0)
            self.pf.update(self.sat, self.pr, self.weights, resample=False)
            if '_KF_' in self.stage:
                self.pf.update_doppler_kf(*self.obs[:3], weights=self.obs[3], wavelength=.19, sigma_mps=.5, resample=False)
            ess = self.pf.get_ess()
            resampled = self.pf.resample_if_needed()
            history.append(np.r_[self.pf.estimate(), ess, float(resampled)])
        return np.concatenate(history)


def worker(extension, python_dir):
    sys.path.insert(0, str(ROOT/'python'))
    native = _load_extension(extension.resolve())
    klass = _setup(native, python_dir)
    session, key = None, None
    print(json.dumps(dict(ready=True)), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request['mode'] == 'close':
                return
            try:
                if request['mode'] in ('validate', 'validate_broadcast'):
                    result = _broadcast_validate(native, klass)
                    if request['mode'] == 'validate':
                        result.update(validate_previous(native))
                else:
                    current = (request['size'], request['stage'])
                    if key != current:
                        if session is not None:
                            session.close()
                        session = Session(native, *current)
                        key = current
                    result = session.snapshot() if request['mode'] == 'snapshot' else session.measure(
                        request['repeats'], request.get('snapshot', True))
                print(json.dumps(result), flush=True)
            except Exception as error:
                import traceback
                print(json.dumps(dict(error=repr(error), traceback=traceback.format_exc())), flush=True)
    finally:
        if session is not None:
            session.close()


class Worker:
    def __init__(self, extension, python_dir=None):
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--extension', str(extension.resolve())]
        if python_dir:
            argv += ['--python-dir', str(python_dir.resolve())]
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding='utf8', cwd=ROOT)
        assert json.loads(self.process.stdout.readline()) == dict(ready=True)

    def request(self, **args):
        self.process.stdin.write(json.dumps(args)+'\n')
        self.process.stdin.flush()
        value = json.loads(self.process.stdout.readline())
        if 'error' in value:
            raise RuntimeError(value)
        return value

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps(dict(mode='close'))+'\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=30)


def compare(args):
    refpath = 'benchmarks/pf_common_predict_gpu_20261014.json'
    ref = json.loads((ROOT/refpath).read_text(encoding='utf8'))
    sources = dict(baseline={p: _sha(args.baseline_dir/Path(p).name) for p in ref['source_sha256']['candidate']},
                   candidate={p: _sha(ROOT/p) for p in ref['source_sha256']['candidate']})
    binaries = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert sources['baseline'] == ref['source_sha256']['candidate']
    assert binaries['baseline'] == ref['extension_sha256']['candidate']
    for p, digest in ref['benchmark_sha256'].items():
        if p not in sources['candidate']:
            assert _sha(ROOT/p) == digest, p
    workers, summary, runs = {}, {}, {}
    try:
        workers['baseline'] = Worker(args.baseline, args.baseline_dir)
        workers['candidate'] = Worker(args.candidate)
        baseline = workers['baseline'].request(mode='validate_broadcast')
        candidate = workers['candidate'].request(mode='validate' if args.validate else 'validate_broadcast')
        if args.validate:
            baseline.update(ref['validation']['candidate_by_case'])
        if baseline != candidate:
            mismatch = {key: dict(baseline=baseline.get(key), candidate=candidate.get(key))
                for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
            args.output.write_text(json.dumps(dict(mismatches=mismatch), indent=2)+'\n', encoding='utf8')
        assert baseline == candidate, 'Full state/weights/trajectory/exception mismatch'
        validation = dict(cases=len(baseline), all_outputs_bit_identical=True,
            baseline_by_case=baseline, candidate_by_case=candidate,
            prior_fixture_reuse_verified_source_binary_and_unchanged_validators=bool(args.validate))
        print(f'{len(baseline)} original/candidate records identical', flush=True)
        for count in args.sizes:
            for stage in args.stages:
                key = f'N{count}/{stage}'
                data = {label: [] for label in workers}
                for r in range(args.rounds):
                    for label in (('baseline', 'candidate') if r % 2 == 0 else ('candidate', 'baseline')):
                        data[label].append(workers[label].request(mode='measure', size=count, stage=stage, repeats=args.repeats))
                for field in ('input_sha256', 'output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                    assert len({v[field] for rows in data.values() for v in rows}) == 1, (key, field)
                medians = {label: [v['median_ms'] for v in rows] for label, rows in data.items()}
                old, new = [float(np.median(medians[label])) for label in workers]
                summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                    improving_paired_rounds=sum(b<a for a, b in zip(medians['baseline'], medians['candidate'])),
                    round_medians_ms=medians)
                runs[key] = data
                print(f'{key}: {old:.4f}->{new:.4f}ms', flush=True)
    finally:
        for value in workers.values():
            value.close()
    helpers = ['benchmarks/bench_pf_covariance_broadcast_compare.py', 'benchmarks/pf_covariance_broadcast_cases.py',
               'python/gnss_gpu/particle_filter_device.py', *ref['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources, extension_sha256=binaries, benchmark_sha256={p: _sha(ROOT/p) for p in helpers},
        baseline_reference=dict(path=refpath, sha256=_sha(ROOT/refpath)), build_configuration=ref['build_configuration'],
        method='Separate persistent workers; frozen baseline Python modules/PE; sequential alternating stage batches; nonexclusive host.',
        scope='Covariance setter/sync; setter+prediction; complete covariance/predict/PR/optional DopplerKF/ESS/conditional B60 resample/estimate or continuous five epochs; initialization,3warmups,IPC/hash/input setup excluded; every timed weight/return and batch-end16-state hashed.',
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, warmup_calls_per_batch=3,
        validation=validation, summary=summary, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('extension', 'python-dir', 'baseline', 'candidate', 'baseline-dir', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--sizes', type=int, nargs='+', default=[2048, 100000, 1000000])
    parser.add_argument('--stages', nargs='+', default=['public_covariance_set', 'public_covariance_predict',
        'public_covariance_PR_step', 'public_covariance_PR_five', 'public_covariance_KF_step', 'public_covariance_KF_five', 'PR_control'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.extension:
        worker(args.extension, args.python_dir)
        return
    if not all((args.baseline, args.candidate, args.baseline_dir, args.output)):
        parser.error('Comparison requires baseline,candidate,baseline-dir,output')
    if min(*args.sizes, args.rounds, args.repeats) < 1:
        parser.error('Positive sizes,rounds,repeats required')
    args.output.write_text(json.dumps(compare(args), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
