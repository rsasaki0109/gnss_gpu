"""Compare one-pass mixed joint DD gates against the preceding individual mixed-gate build.

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
    return ['mixed_joint_gate', 'mixed_joint_huber_gate',
            'mixed_joint_public_gate_step', 'mixed_joint_native_gate_step',
            'mixed_joint_public_huber_gate_step', 'mixed_joint_native_huber_gate_step',
            'public_joint_gate_step', 'public_joint_huber_gate_step',
            'mixed_public_gate_step', 'mixed_public_huber_gate_step',
            'public_CP_step', 'Student_control', 'resample_control']


def _validate(native):
    from bench_pf_dd_mixed_gate_compare import _validate as preceding_validate
    from bench_pf_dd_reference_compare import _reference_observations
    records = preceding_validate(native)
    for count in (31, 257):
        rng = np.random.default_rng(count + 319)
        states = rng.normal(size=(count, 16))
        states[:, :3] += _scenario()[0]
        prior = rng.normal(size=count)
        state = native.pf_device_create(count)
        try:
            for pairs in (2, 13, 33, 512):
                obs = _reference_observations(pairs, 'alternating')
                for sigmas in ((.75, .05), (1e-200, 1e200), (1e200, 1e-200)):
                    for huber in (False, True):
                        for gates in ((10., 0.), (0., .25), (10., .25), (1e-100, 1e-100)):
                            for ks in ((1.5, 1.5), (.75, 3.)):
                                native.pf_device_set_particle_states(state, states)
                                native.pf_device_set_log_weights(state, prior)
                                native.pf_device_weight_dd_joint(state, obs.sat_ecef_k, obs.sat_ecef_ref,
                                    obs.dd_pseudorange_m, obs.dd_carrier_cycles, obs.base_range_k, obs.base_range_ref,
                                    obs.dd_weights, obs.wavelengths_m, pairs, *sigmas, *gates, huber, *ks)
                                actual, weights = _snapshot(native, state)
                                records[f'mixed_joint/N{count}/DD{pairs}/sigma{sigmas}/huber{huber}/gate{gates}/k{ks}'] = dict(
                                    input_sha256=_digest(states, prior, *_inputs(obs)), output_sha256=_digest(actual, weights))
        finally:
            native.pf_device_destroy(state)
    for count in (1, 31, 257):
        states = np.zeros((count, 16))
        states[:, 0] = 1.
        prior = np.linspace(-3., 3., count)
        state = native.pf_device_create(count)
        try:
            for pairs in (3, 4):
                obs = _reference_observations(pairs, 'signed_zero')
                obs.sat_ecef_k[:] = 0.
                obs.sat_ecef_k[0, 0] = 1e308
                obs.base_range_k[:] = obs.base_range_ref[:] = 0.
                obs.dd_pseudorange_m[:] = obs.dd_carrier_cycles[:] = 0.
                obs.dd_weights[:] = 1.
                for huber in (False, True):
                    for ks in ((1.5, 1.5), (0., 1.5), (1.5, 0.)):
                        native.pf_device_set_particle_states(state, states)
                        native.pf_device_set_log_weights(state, prior)
                        native.pf_device_weight_dd_joint(state, obs.sat_ecef_k, obs.sat_ecef_ref,
                            obs.dd_pseudorange_m, obs.dd_carrier_cycles, obs.base_range_k, obs.base_range_ref,
                            obs.dd_weights, obs.wavelengths_m, pairs, .75, .05, 10., .25, huber, *ks)
                        actual, weights = _snapshot(native, state)
                        records[f'mixed_joint_partial_nan/N{count}/DD{pairs}/huber{huber}/k{ks}'] = dict(
                            input_sha256=_digest(states, prior, *_inputs(obs)), output_sha256=_digest(actual, weights))
        finally:
            native.pf_device_destroy(state)
    return records


def _profile(native, count, stage, repeats):
    if not stage.startswith('mixed_joint_'):
        from bench_pf_dd_mixed_gate_compare import _profile as preceding_profile
        return preceding_profile(native, count, stage, repeats)
    sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    from bench_pf_dd_reference_compare import _initialize, _joint, _measure, _reference_observations
    obs = _reference_observations(12, 'alternating')
    huber = 'huber' in stage
    if stage.startswith('mixed_joint_public'):
        pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42,
            per_particle_nlos_gate=True, per_particle_nlos_dd_pr_threshold_m=10.,
            per_particle_nlos_dd_carrier_threshold_cycles=.25, per_particle_huber=huber)
        try:
            def prepare():
                pf.initialize(_scenario()[0], clock_bias=3000., spread_pos=10., spread_cb=30.,
                    velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)
            def operation():
                pf.predict([1., 2., 3.], dt=.1)
                pf.update_dd_joint(obs)
                return pf.estimate()
            return _measure(native, pf._state, prepare, operation, repeats, count)
        finally:
            native.pf_device_destroy(pf._state)
            pf._state = None
    state = native.pf_device_create(count)
    try:
        def operation():
            if not stage.startswith('mixed_joint_native'):
                return _joint(native, state, obs, .25, huber)
            native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1)
            _joint(native, state, obs, .25, huber)
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
    reference_path = 'benchmarks/pf_dd_mixed_gate_gpu_20261012.json'
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
    helpers = ('benchmarks/bench_pf_dd_mixed_joint_gate_compare.py', 'benchmarks/bench_pf_dd_mixed_gate_compare.py', 'benchmarks/bench_pf_dd_single_huber_gate_compare.py',
               'benchmarks/bench_pf_dd_single_gate_compare.py',
               'benchmarks/bench_pf_dd_huber_gate_compare.py',
               'benchmarks/bench_pf_dd_gate_compare.py',
               'benchmarks/bench_pf_dd_sigma_compare.py',
               'benchmarks/bench_pf_dd_joint_reference_compare.py',
               'benchmarks/bench_pf_dd_joint_options_check.py',
               'benchmarks/bench_pf_dd_reference_compare.py', 'benchmarks/bench_pf_coalesced_compare.py',
               'benchmarks/bench_pf_covariance_compare.py')
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                build_configuration='CUDA 12.8 / MSVC Release / compute 7.5 / C++20 / no fast math',
                baseline='preceding individual mixed gates; mixed joint PR+carrier gates use two geometry loops',
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
