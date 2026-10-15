"""Compare individual-covariance gathers with persistent inputs and owners.

Separate workers execute GPU calls sequentially. Inputs are constructed once
per stage, outside timing. Every measured log-weight and returned result is
hashed; full states are compared at batch/paired-round boundaries. Candidate
fixtures can be checked against exact frozen original-build records.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from bench_pf_dd_reference_compare import _digest, _initialize, _load_extension, _scenario, _snapshot
from bench_pf_doppler_compare import _kf, _observations
from bench_pf_shared_proposals_compare import _validate

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stages():
    return ['resample_B4_imported', 'resample_B60_imported', 'resample_B240_imported',
            'native_PR_KF_step', 'public_PR_KF_step', 'resample_B60_common',
            'public_PR_step', 'PR_control', 'resample_B0_imported']


class Session:
    def __init__(self, native, count, stage):
        self.native, self.count, self.stage = native, count, stage
        self.position, self.sat, self.pr, self.weights = _scenario()
        self.obs = _observations(12)
        self.resample = re.fullmatch(r'resample_B(\d+)_(imported|common)', stage)
        self.public = stage.startswith('public_')
        self.pf = None
        if self.public:
            sys.modules['gnss_gpu._gnss_gpu_pf_device'] = native
            from gnss_gpu.particle_filter_device import ParticleFilterDevice
            self.pf = ParticleFilterDevice(n_particles=count, sigma_pos=1., sigma_cb=3., sigma_pr=5., seed=42)
            self.state = self.pf._state
        else:
            self.state = native.pf_device_create(count)
        self.imported = None
        self.prior = None
        if self.resample:
            _initialize(native, self.state)
            self.imported = native.pf_device_get_particle_states(self.state)
            rng = np.random.default_rng(count)
            factors = rng.normal(size=(count, 3, 3))
            self.imported[:, 7:] = (factors @ factors.swapaxes(1, 2)+np.eye(3)*.1).reshape(count, 9)
            self.prior = rng.normal(0., 3., count)
        inputs = [self.position, self.sat, self.pr, self.weights, *self.obs]
        if self.resample:
            inputs += [self.imported, self.prior]
        self.input_sha256 = _digest(*inputs)
        self.last_returned = np.empty(0)

    def prepare(self):
        if self.pf is not None:
            self.pf.initialize(self.position, clock_bias=3000., spread_pos=10., spread_cb=30.,
                velocity=[1., 2., 3.], spread_vel=.5, velocity_init_sigma=2.)
        elif self.resample and self.resample[2] == 'imported':
            # Import replaces every state component; the following prior
            # replaces every log-weight. Initialization is redundant here.
            self.native.pf_device_set_particle_states(self.state, self.imported)
        else:
            _initialize(self.native, self.state)
        if self.resample:
            self.native.pf_device_set_log_weights(self.state, self.prior)

    def operation(self):
        native, state = self.native, self.state
        if self.resample:
            native.pf_device_resample_megopolis_coalesced(state, int(self.resample[1]), 42)
            return np.empty(0)
        if self.stage == 'PR_control':
            native.pf_device_weight(state, self.sat, self.pr, self.weights, 12, 5.)
            return np.empty(0)
        kf = '_KF_' in self.stage
        if self.pf is not None:
            self.pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3 if kf else 1.,
                rbpf_velocity_kf=kf, velocity_process_noise=.1 if kf else 0.)
            self.pf.update(self.sat, self.pr, self.weights, resample=False)
            if kf:
                self.pf.update_doppler_kf(*self.obs[:3], weights=self.obs[3], wavelength=.19, sigma_mps=.5, resample=False)
            ess = self.pf.get_ess()
            resampled = self.pf.resample_if_needed()
            return np.r_[self.pf.estimate(), ess, float(resampled)]
        native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1,
            0., .3 if kf else 1., kf, .1 if kf else 0.)
        native.pf_device_weight(state, self.sat, self.pr, self.weights, 12, 5.)
        if kf:
            _kf(native, state, self.obs)
        ess = native.pf_device_ess(state)
        resampled = ess < .5*self.count
        if resampled:
            native.pf_device_resample_megopolis_coalesced(state, 60, 42)
        return np.r_[native.pf_device_estimate(state), ess, float(resampled)]

    def measure(self, repeats, snapshot=True):
        samples, hashes, returns = [], set(), set()
        weights = np.empty(self.count)
        for i in range(repeats+3):
            self.prepare()
            self.native.pf_device_sync(self.state)
            start = time.perf_counter_ns()
            value = self.operation()
            self.native.pf_device_sync(self.state)
            elapsed = (time.perf_counter_ns()-start)/1e6
            self.native.pf_device_get_log_weights(self.state, weights)
            hashes.add(_digest(weights))
            returns.add(_digest(value))
            self.last_returned = value
            if i >= 3:
                samples.append(elapsed)
        assert len(hashes) == len(returns) == 1
        result = dict(input_sha256=self.input_sha256, median_ms=float(np.median(samples)), samples_ms=samples,
            log_weights_sha256=hashes.pop(), returned_output_sha256=returns.pop(),
            all_timed_log_weights_identical=True, all_returned_outputs_identical=True)
        if snapshot:
            result.update(self.snapshot())
        return result

    def snapshot(self):
        return dict(input_sha256=self.input_sha256,
            output_sha256=_digest(*_snapshot(self.native, self.state), self.last_returned))

    def close(self):
        self.native.pf_device_destroy(self.state)
        if self.pf is not None:
            self.pf._state = None


def worker(extension):
    sys.path.insert(0, str(ROOT/'python'))
    native = _load_extension(extension.resolve())
    session, key = None, None
    print(json.dumps(dict(ready=True)), flush=True)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request['mode'] == 'close':
                return
            try:
                if request['mode'] == 'validate':
                    result = _validate(native)
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
                print(json.dumps(dict(error=repr(error))), flush=True)
    finally:
        if session is not None:
            session.close()


class Worker:
    def __init__(self, extension):
        self.process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--extension', str(extension.resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf8')
        assert json.loads(self.process.stdout.readline()) == dict(ready=True)

    def request(self, **args):
        self.process.stdin.write(json.dumps(args)+'\n')
        self.process.stdin.flush()
        result = json.loads(self.process.stdout.readline())
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(json.dumps(dict(mode='close'))+'\n')
            self.process.stdin.flush()
        self.process.communicate(timeout=30)


def compare(args):
    reference_path = 'benchmarks/pf_packed_gather_gpu_20261013.json'
    ref = json.loads((ROOT/reference_path).read_text(encoding='utf8'))
    sources = dict(baseline={p: _sha(args.baseline_dir/Path(p).name) for p in ref['source_sha256']['candidate']},
                   candidate={p: _sha(ROOT/p) for p in ref['source_sha256']['candidate']})
    binaries = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert sources['baseline'] == ref['source_sha256']['candidate']
    assert binaries['baseline'] == ref['extension_sha256']['candidate']
    for p, digest in ref['benchmark_sha256'].items():
        assert _sha(ROOT/p) == digest, p
    workers, summary, runs = {}, {}, {}
    validation = None
    try:
        for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
            workers[label] = Worker(path)
        if args.validate:
            baseline = ref['validation']['candidate_by_case']
            candidate = workers['candidate'].request(mode='validate')
            if candidate != baseline:
                mismatches = {key: dict(baseline=baseline.get(key), candidate=candidate.get(key))
                    for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
                args.output.write_text(json.dumps(dict(source_sha256=sources, extension_sha256=binaries,
                    mismatches=mismatches), indent=2)+'\n', encoding='utf8')
            assert baseline == candidate, 'Input/state/log-weight/trajectory bit mismatch'
            validation = dict(cases=len(baseline), all_outputs_bit_identical=True,
                baseline_by_case=baseline, candidate_by_case=candidate,
                baseline_reference=dict(path=reference_path, sha256=_sha(ROOT/reference_path)),
                unchanged_original_source_binary_and_validator_dependencies=True)
            print(f'{len(baseline)} fixture records match the retained baseline', flush=True)
        for count in args.sizes:
            for stage in args.stages or _stages():
                key = f'N{count}/{stage}'
                data = {label: [] for label in workers}
                for i in range(args.rounds):
                    order = ('baseline', 'candidate') if i % 2 == 0 else ('candidate', 'baseline')
                    for label in order:
                        data[label].append(workers[label].request(mode='measure', size=count, stage=stage, repeats=args.repeats))
                for field in ('input_sha256', 'output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                    assert len({v[field] for values in data.values() for v in values}) == 1, (key, field)
                medians = {label: [v['median_ms'] for v in values] for label, values in data.items()}
                old, new = [float(np.median(medians[label])) for label in workers]
                summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                    improving_paired_rounds=sum(b < a for a, b in zip(medians['baseline'], medians['candidate'])),
                    round_medians_ms=medians, output_sha256=data['baseline'][0]['output_sha256'])
                runs[key] = data
                print(f'{key}: {old:.4f}->{new:.4f}ms ({old/new:.3f}x)', flush=True)
    finally:
        for value in workers.values():
            value.close()
    helpers = ['benchmarks/bench_pf_individual_gather_compare.py', *ref['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources, extension_sha256=binaries, benchmark_sha256={p: _sha(ROOT/p) for p in helpers},
        baseline_reference=dict(path=reference_path, sha256=_sha(ROOT/reference_path)),
        build_configuration=ref['build_configuration'],
        method='Persistent separate workers and stage inputs/owners; sequential alternating stage batches; nonexclusive host.',
        scope='Synchronized native individual/common resampling and complete native/public predict/PR/KF/ESS/conditional B60 resample/estimate; initialization/restoration/IPC/hash/input setup excluded; every timed weight and returned result hashed; full state at each batch end.',
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, warmup_calls_per_batch=3,
        validation=validation, summary=summary, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path)
    for name in ('baseline', 'candidate', 'baseline-dir', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--sizes', type=int, nargs='+', default=[2048, 100000, 1000000])
    parser.add_argument('--stages', nargs='+')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.extension:
        worker(args.extension)
        return
    if not all((args.baseline, args.candidate, args.baseline_dir, args.output)):
        parser.error('comparison requires baseline, candidate, baseline-dir and output')
    if min(*args.sizes, args.rounds, args.repeats) < 1:
        parser.error('positive sizes, rounds and repeats required')
    result = compare(args)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
