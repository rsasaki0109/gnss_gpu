"""Compare common DD reference-range reuse with the preceding actual PF build.

Separate persistent production workers avoid pybind type collisions. GPU calls
run sequentially; order alternates by stage and round. Hash every timed returned
value and log-weight array outside timing, plus final complete particle state.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from bench_pf_coalesced_compare import _initialize, _load_extension, _scenario
from bench_pf_covariance_compare import _digest, _snapshot


def _observations(count, variant='mixed'):
    position = _scenario()[0]
    rng = np.random.default_rng(count + 713)
    directions = rng.normal(size=(count + 1, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    satellites = position + directions * 2.5e7
    sat_k = satellites[1:].copy()
    ref = np.repeat(satellites[:1], count, axis=0)
    base = position + [5., -3., 2.]
    br_k = np.linalg.norm(sat_k - base, axis=1)
    br_ref = np.linalg.norm(ref - base, axis=1)
    dd_range = np.linalg.norm(sat_k - position, axis=1) - np.linalg.norm(ref - position, axis=1) - br_k + br_ref
    wavelengths = np.resize([.19, .24, .25, np.nextafter(.19, np.inf)], count)
    if variant == 'constant':
        wavelengths[:] = .19
    elif variant == 'tiny':
        wavelengths[:] = 1e-200
    elif variant == 'huge':
        wavelengths[:] = 1e200
    elif variant == 'nonfinite':
        wavelengths[:] = np.resize([0., -0., np.nan, np.inf, -np.inf, -.19], count)
    weights = rng.uniform(.1, 1., count)
    if variant == 'zero_weight':
        weights[:] = 0.
    return SimpleNamespace(
        sat_ecef_k=sat_k, sat_ecef_ref=ref, base_range_k=br_k, base_range_ref=br_ref,
        dd_pseudorange_m=dd_range + rng.normal(0., .05, count),
        dd_carrier_cycles=dd_range / .19 + np.arange(count) * 7 + rng.normal(0., .01, count),
        dd_weights=weights, wavelengths_m=wavelengths, n_dd=count)


def _inputs(obs):
    return [obs.sat_ecef_k, obs.sat_ecef_ref, obs.dd_pseudorange_m, obs.dd_carrier_cycles,
            obs.base_range_k, obs.base_range_ref, obs.dd_weights, obs.wavelengths_m]


def _cp(native, state, obs, gate=0., huber=False, k=1.5):
    native.pf_device_weight_dd_carrier_afv(
        state, obs.sat_ecef_k, obs.sat_ecef_ref, obs.dd_carrier_cycles,
        obs.base_range_k, obs.base_range_ref, obs.dd_weights, obs.wavelengths_m,
        obs.n_dd, .05, gate, huber, k)


def _joint(native, state, obs, gate=0., huber=False, k=1.5):
    native.pf_device_weight_dd_joint(
        state, obs.sat_ecef_k, obs.sat_ecef_ref, obs.dd_pseudorange_m, obs.dd_carrier_cycles,
        obs.base_range_k, obs.base_range_ref, obs.dd_weights, obs.wavelengths_m,
        obs.n_dd, .75, .05, 10. if gate > 0. else 0., gate, huber, k, k)


def _validate_wavelength_cases(native):
    records = {}
    for count in (1, 31, 255, 256, 257, 8193):
        rng = np.random.default_rng(count)
        states = rng.normal(size=(count, 16))
        states[:, :3] += _scenario()[0]
        prior = rng.normal(size=count)
        state = native.pf_device_create(count)
        try:
            for n_dd in (1, 2, 4, 12, 32, 33, 257, 512):
                for variant in ('mixed', 'constant', 'zero_weight', 'tiny', 'huge', 'nonfinite'):
                    obs = _observations(n_dd, variant)
                    for name, gate, huber, k in (('gaussian', 0., False, 1.5), ('gate', .25, False, 1.5),
                                                ('huber', 0., True, 1.5), ('huber_gate', .25, True, 1.5)):
                        for label, operation in (('CP', _cp), ('joint', _joint)):
                            native.pf_device_set_particle_states(state, states)
                            native.pf_device_set_log_weights(state, prior)
                            operation(native, state, obs, gate, huber, k)
                            actual, weights = _snapshot(native, state)
                            np.testing.assert_array_equal(actual.view(np.uint64), states.view(np.uint64))
                            records[f'N{count}/dd{n_dd}/{variant}/{name}/{label}'] = dict(
                                input_sha256=_digest(states, prior, *_inputs(obs)),
                                output_sha256=_digest(actual, weights))
            retained = []
            for index, n_dd in enumerate((1, 512, 12, 257, 33, 2)):
                obs = _observations(n_dd)
                obs.wavelengths_m *= 1. + index / 10.
                _joint(native, state, obs)
                retained.extend(_snapshot(native, state))
            records[f'N{count}/changed_observations'] = dict(output_sha256=_digest(*retained))
        finally:
            native.pf_device_destroy(state)
    for count in (2048, 100000, 1000000):
        for label, operation in (('CP', _cp), ('joint', _joint)):
            state = native.pf_device_create(count)
            try:
                _initialize(native, state)
                history = []
                obs = _observations(12)
                for step in range(1, 4):
                    native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, step)
                    operation(native, state, obs)
                    ess = native.pf_device_ess(state)
                    resampled = ess < .5 * count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state, 60, 42 + step)
                    history.append(np.array([ess, resampled, *native.pf_device_estimate(state)]))
                records[f'N{count}/three_epoch/{label}'] = dict(
                    input_sha256=_digest(*_inputs(obs)), output_sha256=_digest(*history, *_snapshot(native, state)))
            finally:
                native.pf_device_destroy(state)
    return records



def _pr(native, state, obs, gate=0., huber=False, k=1.5):
    native.pf_device_weight_dd_pseudorange(
        state, obs.sat_ecef_k, obs.sat_ecef_ref, obs.dd_pseudorange_m,
        obs.base_range_k, obs.base_range_ref, obs.dd_weights,
        obs.n_dd, .75, 10. if gate > 0. else 0., huber, k)


def _reference_observations(count, pattern):
    obs = _observations(count)
    ref = obs.sat_ecef_ref
    if pattern == 'alternating':
        ref[1::2] += [12., -4., 8.]
    elif pattern == 'last_ulp':
        ref[-1, 0] = np.nextafter(ref[-1, 0], np.inf)
    elif pattern == 'signed_zero':
        ref[:] = 0.
        ref[-1, 0] = -0.
    elif pattern == 'origin':
        ref[:] = 0.
    elif pattern == 'huge':
        ref[:] = [1e150, -1e150, 1e150]
    elif pattern == 'nonfinite':
        ref[:] = [np.inf, np.nan, -np.inf]
    elif pattern != 'common':
        raise ValueError(pattern)
    with np.errstate(invalid='ignore', over='ignore'):
        obs.base_range_ref = np.linalg.norm(ref - (_scenario()[0] + [5., -3., 2.]), axis=1)
        dd_range = (np.linalg.norm(obs.sat_ecef_k - _scenario()[0], axis=1)
                    - np.linalg.norm(ref - _scenario()[0], axis=1)
                    - obs.base_range_k + obs.base_range_ref)
        obs.dd_pseudorange_m = dd_range + np.linspace(-.05, .05, count)
        obs.dd_carrier_cycles = dd_range / .19 + np.arange(count) * 7 + .01
    return obs


def _validate(native):
    records = _validate_wavelength_cases(native)
    for count in (1, 31, 255, 256, 257, 8193):
        rng = np.random.default_rng(count + 37)
        states = rng.normal(size=(count, 16))
        states[:, :3] += _scenario()[0]
        prior = rng.normal(size=count)
        state = native.pf_device_create(count)
        try:
            for pairs in (1, 2, 12, 512):
                for pattern in ('common', 'alternating', 'last_ulp', 'signed_zero', 'origin', 'huge', 'nonfinite'):
                    obs = _reference_observations(pairs, pattern)
                    for mode, gate, huber in (('plain', 0., False), ('gate', .25, False),
                                              ('huber', 0., True), ('huber_gate', .25, True)):
                        for label, operation in (('PR', _pr), ('CP', _cp), ('joint', _joint)):
                            native.pf_device_set_particle_states(state, states)
                            native.pf_device_set_log_weights(state, prior)
                            operation(native, state, obs, gate, huber)
                            actual, weights = _snapshot(native, state)
                            np.testing.assert_array_equal(actual.view(np.uint64), states.view(np.uint64))
                            records[f'N{count}/dd{pairs}/ref_{pattern}/{mode}/{label}'] = dict(
                                input_sha256=_digest(states, prior, *_inputs(obs)), output_sha256=_digest(actual, weights))
            history = []
            for pairs, pattern in ((2, 'common'), (512, 'alternating'), (12, 'common'),
                                   (33, 'last_ulp'), (257, 'origin'), (1, 'common')):
                obs = _reference_observations(pairs, pattern)
                _joint(native, state, obs, .25, True)
                history.extend(_snapshot(native, state))
            records[f'N{count}/reference_changes_and_buffer_growth'] = dict(output_sha256=_digest(*history))
        finally:
            native.pf_device_destroy(state)
    return records


def _measure(native, state, prepare, operation, repeats, count):
    samples, hashes, returns = [], set(), set()
    weights = np.empty(count)
    for repetition in range(repeats + 3):
        prepare()
        native.pf_device_sync(state)
        start = time.perf_counter_ns()
        value = operation()
        native.pf_device_sync(state)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        native.pf_device_get_log_weights(state, weights)
        output = np.empty(0) if value is None else np.asarray(value, dtype=np.float64)
        hashes.add(_digest(weights))
        returns.add(_digest(output))
        if repetition >= 3:
            samples.append(elapsed)
    assert len(hashes) == len(returns) == 1
    particles, weights = _snapshot(native, state)
    assert len(particles) == count
    return dict(median_ms=float(np.median(samples)), samples_ms=samples,
                output_sha256=_digest(particles, weights, output), log_weights_sha256=hashes.pop(),
                returned_output_sha256=returns.pop(), all_timed_log_weights_identical=True,
                all_returned_outputs_identical=True)


def _stages():
    return [*[f'{kind}_dd{count}' for kind in ('PR', 'CP', 'joint') for count in (2, 12, 32)],
            'PR_gate', 'CP_gate', 'joint_gate', 'PR_huber', 'CP_huber', 'joint_huber',
            'PR_mixed', 'CP_mixed', 'joint_mixed', 'native_CP_step', 'native_joint_step',
            'public_CP_step', 'public_joint_step', 'Student_control', 'resample_control']


def _profile(native, count, stage, repeats):
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    obs = _reference_observations(int(stage.split('_dd')[1]) if '_dd' in stage else 12,
                                  'alternating' if stage.endswith('mixed') else 'common')
    if stage.startswith('public'):
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
        try:
            def prepare():
                pf.initialize(_scenario()[0], clock_bias=3000., spread_pos=10., spread_cb=30.,
                              velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)
            def operation():
                pf.predict([1., 2., 3.], dt=.1)
                if 'joint' in stage:
                    pf.update_dd_joint(obs)
                else:
                    pf.update_dd_pseudorange(obs, resample=False)
                    pf.update_dd_carrier_afv(obs)
                return pf.estimate()
            return _measure(native, pf._state, prepare, operation, repeats, count)
        finally:
            native.pf_device_destroy(pf._state)
            pf._state = None
    state = native.pf_device_create(count)
    try:
        def operation():
            if stage == 'Student_control':
                _, sat, pr, weights = _scenario()
                native.pf_device_weight(state, sat, pr, weights, 12, 5., 3.)
                return None
            if stage == 'resample_control':
                native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return None
            if stage.startswith('native'):
                native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1)
                if 'joint' not in stage:
                    native.pf_device_weight_dd_pseudorange(
                        state, obs.sat_ecef_k, obs.sat_ecef_ref, obs.dd_pseudorange_m,
                        obs.base_range_k, obs.base_range_ref, obs.dd_weights, obs.n_dd, .75)
                (_joint if 'joint' in stage else _cp)(native, state, obs)
                ess = native.pf_device_ess(state)
                resampled = ess < .5 * count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state, 60, 42)
                return np.array([ess, resampled, *native.pf_device_estimate(state)])
            return (_joint if stage.startswith('joint') else _pr if stage.startswith('PR') else _cp)(
                native, state, obs, .25 if stage.endswith('gate') else 0., stage.endswith('huber'))
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
    reference_path = 'benchmarks/pf_dd_wavelength_gpu_20261012.json'
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
    helpers = ('benchmarks/bench_pf_dd_reference_compare.py', 'benchmarks/bench_pf_coalesced_compare.py',
               'benchmarks/bench_pf_covariance_compare.py')
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
                baseline='preceding DD inverse-wavelength-cache PF; repeated per-pair reference range',
                baseline_reference=dict(path=reference_path, sha256=sha(root / reference_path)),
                method='persistent actual production workers; sequential stage batches; alternating order per round; nonexclusive host',
                scope='synchronized native DD PR/carrier/joint weights and native/public predict/DD PR+CP/ESS/conditional B60 resample/estimate; '
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
