"""Compare original Doppler likelihood prefixes with complete PF step and trajectory timing.

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
from bench_pf_shared_proposals_compare import _validate as validate_original

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _common_cases():
    # alpha, dt, position sigma, clock sigma, Qv, sampled-velocity sigma
    return ((.3,.17,1.3,2.7,.13,0.), (2.,0.,1.3,2.7,0.,.2),
            (-1.,1e-300,1e-300,1e-300,-1.,.2), (.3,.17,1e-300,2.7,.13,0.),
            (.3,.17,1e150,1e150,1e150,0.), (.3,1e154,1.3,2.7,.13,0.),
            (.3,1.17,1.3,2.7,-1.,0.))


def _common_validate(native):
    records = {}
    for count in (1,31,32,33,255,256,257,8193):
        actual = native.pf_device_create(count)
        original = native.pf_device_create(count)
        try:
            for sigma in (0.,-0.,1e-300,1e-150,2.,1e150,1e154,1e308):
                for index, case in enumerate(_common_cases()):
                    native.pf_device_initialize(actual,*_scenario()[0],3000.,10.,30.,42,1.,2.,3.,.5,sigma)
                    initial = native.pf_device_get_particle_states(actual)
                    native.pf_device_set_particle_states(original,initial)
                    alpha,dt,sp,sc,q,sv = case
                    seed,step = (2**64-1,13) if index % 2 else (0,0)
                    args = (1.,2.,3.,dt,sp,sc,seed,step,sv,alpha,True,q)
                    for state in (actual,original):
                        native.pf_device_predict(state,*args)
                    a,b = _snapshot(native,actual),_snapshot(native,original)
                    for left,right in zip(a,b,strict=True):
                        np.testing.assert_array_equal(left.view(np.uint64),right.view(np.uint64))
                    name = f'common_predict/N{count}/sigma{sigma}/case{index}'
                    records[name] = dict(input_sha256=_digest(initial,np.asarray(case)),seed=seed,step=step,
                        output_sha256=_digest(*a),original_imported_producer_identical=True)
            for seed in (0,2**64-1):
                for state in (actual,original):
                    _initialize(native,state)
                initial = native.pf_device_get_particle_states(actual)
                native.pf_device_set_particle_states(original,initial)
                history=[]
                for step in range(1,9):
                    args = (1.+step,2.-step,3./step,.01*step,1.,3.,seed,step,.2,.3,True,.13)
                    for state in (actual,original):
                        native.pf_device_predict(state,*args)
                for state in (actual,original):
                    native.pf_device_resample_megopolis_coalesced(state,4,seed)
                a,b = _snapshot(native,actual),_snapshot(native,original)
                for left,right in zip(a,b,strict=True):
                    np.testing.assert_array_equal(left.view(np.uint64),right.view(np.uint64))
                history.extend(a)
                for state in (actual,original):
                    _kf(native,state,_observations(12))
                    native.pf_device_predict(state,1.,2.,3.,.17,1.,3.,seed,9,0.,.3,True,.13)
                a,b = _snapshot(native,actual),_snapshot(native,original)
                for left,right in zip(a,b,strict=True):
                    np.testing.assert_array_equal(left.view(np.uint64),right.view(np.uint64))
                history.extend(a)
                records[f'common_predict/N{count}/queue_resample_KF/seed{seed}']=dict(
                    input_sha256=_digest(initial),output_sha256=_digest(*history),original_imported_producer_identical=True)
        finally:
            native.pf_device_destroy(actual)
            native.pf_device_destroy(original)
    return records


def _validate(native):
    records=validate_original(native)
    records.update(_common_validate(native))
    records.update(_likelihood_validate(native))
    return records


def _stages():
    return ['resample_B4_imported', 'resample_B60_imported', 'resample_B240_imported',
            'native_PR_KF_step', 'public_PR_KF_step', 'resample_B60_common',
            'public_PR_step', 'PR_control', 'resample_B0_imported']


class OriginalSession:
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
        if self.resample or stage == 'native_predict_KF_imported':
            _initialize(native, self.state)
            self.imported = native.pf_device_get_particle_states(self.state)
            rng = np.random.default_rng(count)
            if self.resample:
                factors = rng.normal(size=(count, 3, 3))
                self.imported[:, 7:] = (factors @ factors.swapaxes(1, 2)+np.eye(3)*.1).reshape(count, 9)
            self.prior = rng.normal(0., 3., count)
        inputs = [self.position, self.sat, self.pr, self.weights, *self.obs]
        if self.imported is not None:
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
        if self.imported is not None and not self.resample:
            self.native.pf_device_set_particle_states(self.state, self.imported)
        if self.resample:
            self.native.pf_device_set_log_weights(self.state, self.prior)

    def operation(self):
        native, state = self.native, self.state
        if self.resample:
            native.pf_device_resample_megopolis_coalesced(state, int(self.resample[1]), 42)
            return np.empty(0)
        if 'predict_KF_' in self.stage:
            if self.pf is not None:
                self.pf.predict([1.,2.,3.],dt=.1,velocity_guide_alpha=.3,rbpf_velocity_kf=True,velocity_process_noise=.1)
            else:
                native.pf_device_predict(state,1.,2.,3.,.1,1.,3.,42,1,0.,.3,True,.1)
            return np.empty(0)
        if 'RBPF_five' in self.stage:
            history=[]
            for step in range(1,6):
                if self.pf is not None:
                    self.pf.predict([1.,2.,3.],dt=.1,velocity_guide_alpha=.3,rbpf_velocity_kf=True,velocity_process_noise=.1)
                    self.pf.update(self.sat,self.pr,self.weights,resample=False)
                    ess=self.pf.get_ess()
                    resampled=self.pf.resample_if_needed()
                    history.append(np.r_[self.pf.estimate(),ess,float(resampled)])
                else:
                    native.pf_device_predict(state,1.,2.,3.,.1,1.,3.,42,step,0.,.3,True,.1)
                    native.pf_device_weight(state,self.sat,self.pr,self.weights,12,5.)
                    ess=native.pf_device_ess(state)
                    resampled=ess<.5*self.count
                    if resampled:
                        native.pf_device_resample_megopolis_coalesced(state,60,42+step)
                    history.append(np.r_[native.pf_device_estimate(state),ess,float(resampled)])
            return np.concatenate(history)
        if self.stage == 'PR_control':
            native.pf_device_weight(state, self.sat, self.pr, self.weights, 12, 5.)
            return np.empty(0)
        kf = '_KF_' in self.stage
        rbpf = kf or 'RBPF' in self.stage
        if self.pf is not None:
            self.pf.predict([1., 2., 3.], dt=.1, velocity_guide_alpha=.3 if rbpf else 1.,
                rbpf_velocity_kf=rbpf, velocity_process_noise=.1 if rbpf else 0.)
            self.pf.update(self.sat, self.pr, self.weights, resample=False)
            if kf:
                self.pf.update_doppler_kf(*self.obs[:3], weights=self.obs[3], wavelength=.19, sigma_mps=.5, resample=False)
            ess = self.pf.get_ess()
            resampled = self.pf.resample_if_needed()
            return np.r_[self.pf.estimate(), ess, float(resampled)]
        native.pf_device_predict(state, 1., 2., 3., .1, 1., 3., 42, 1,
            0., .3 if rbpf else 1., rbpf, .1 if rbpf else 0.)
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



def _doppler_update(native,state,arrays,gain):
    native.pf_device_weight_doppler(state,*arrays,len(arrays[3]),.19,.5,gain,10.)


def _likelihood_validate(native):
    records={}
    satellites=(0,1,3,4,8,9,16,17,24,25,32,33,48,49,64,65)
    gains=(0.,-0.,-1.,-np.inf,np.nan,1e-300,.25,np.inf)
    for count in (1,127,128,129,8193):
        actual=native.pf_device_create(count)
        original=native.pf_device_create(count)
        try:
            _initialize(native,actual)
            initial,prior=_snapshot(native,actual)
            for sat in satellites:
                for gain_index,gain in enumerate(gains):
                    for variant in ('normal','masked','nonfinite'):
                        arrays=[a.copy() for a in _observations(sat)]
                        if variant=='masked':
                            arrays[3][::3]=0.
                            arrays[3][1::3]=-1.
                        elif variant=='nonfinite':
                            arrays[0][::5,0]=np.nan
                            arrays[1][::7,1]=np.inf
                            arrays[3][::3]=np.nan
                        for state in (actual,original):
                            native.pf_device_set_particle_states(state,initial)
                            native.pf_device_set_log_weights(state,prior)
                        for _ in range(2):
                            _doppler_update(native,actual,arrays,gain)
                        output=_snapshot(native,actual)
                        if gain<=0. or np.isnan(gain):
                            # NaN retains the original producer. Its original
                            # fmax(0,NaN) clamp disables only the WLS tail.
                            # This oracle is also executed on the baseline.
                            for _ in range(2):
                                _doppler_update(native,original,arrays,np.nan)
                            oracle=_snapshot(native,original)
                            for a,b in zip(output,oracle,strict=True):
                                np.testing.assert_array_equal(a.view(np.uint64),b.view(np.uint64))
                            np.testing.assert_array_equal(output[0].view(np.uint64),initial.view(np.uint64))
                        key=f'likelihood/N{count}/sat{sat}/gain{gain_index}/{variant}'
                        records[key]=dict(input_sha256=_digest(initial,prior,*arrays,np.array([gain])),
                            output_sha256=_digest(*output),retained_original_oracle_identical=bool(gain<=0. or np.isnan(gain)))
            _initialize(native,actual)
            history=[]
            for step,sat in enumerate((8,9,16,17,24,25,32,33,48,49,64,65),1):
                native.pf_device_predict(actual,1.,2.,3.,.17,1.,3.,42,step)
                arrays=[a.copy() for a in _observations(sat)]
                _doppler_update(native,actual,arrays,0. if step%2 else .25)
                for a in arrays:
                    a.fill(-999.)
                if step%3==0:
                    native.pf_device_resample_megopolis_coalesced(actual,7,42+step)
                history.extend(_snapshot(native,actual))
            records[f'likelihood/N{count}/queued_growth_mutation_resample']=dict(output_sha256=_digest(*history))
        finally:
            native.pf_device_destroy(actual)
            native.pf_device_destroy(original)
    return records


class Session(OriginalSession):
    def __init__(self,native,count,stage):
        self.doppler=re.fullmatch(r'(native|public)_Doppler(\d+)_gain([^_]+)_(update|step|five)',stage)
        if not self.doppler:
            super().__init__(native,count,stage)
            return
        super().__init__(native,count,'public_PR_step' if self.doppler[1]=='public' else 'native_PR_step')
        self.stage=stage
        self.obs=_observations(int(self.doppler[2]))
        self.gain=float(self.doppler[3])
        self.input_sha256=_digest(self.position,self.sat,self.pr,self.weights,*self.obs,np.array([self.gain]))

    def operation(self):
        if not self.doppler:
            return super().operation()
        native,state=self.native,self.state
        if self.doppler[4]=='update':
            _doppler_update(native,state,self.obs,self.gain)
            return np.empty(0)
        history=[]
        for step in range(1,6 if self.doppler[4]=='five' else 2):
            if self.pf is not None:
                self.pf.predict([1.,2.,3.],dt=.1)
                self.pf.update(self.sat,self.pr,self.weights,resample=False)
                self.pf.update_doppler(*self.obs[:3],weights=self.obs[3],wavelength=.19,sigma_mps=.5,
                    velocity_update_gain=self.gain,resample=False)
                ess=self.pf.get_ess()
                resampled=self.pf.resample_if_needed()
                history.append(np.r_[self.pf.estimate(),ess,float(resampled)])
            else:
                native.pf_device_predict(state,1.,2.,3.,.1,1.,3.,42,step)
                native.pf_device_weight(state,self.sat,self.pr,self.weights,12,5.)
                _doppler_update(native,state,self.obs,self.gain)
                ess=native.pf_device_ess(state)
                resampled=ess<.5*self.count
                if resampled:
                    native.pf_device_resample_megopolis_coalesced(state,60,42+step)
                history.append(np.r_[native.pf_device_estimate(state),ess,float(resampled)])
        return np.concatenate(history)


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
                if request['mode'] in ('validate','validate_common'):
                    result = _likelihood_validate(native) if request['mode']=='validate_common' else _validate(native)
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
    reference_path = 'benchmarks/pf_common_predict_gpu_20261014.json'
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
        if args.validate or args.common_only:
            if args.common_only:
                baseline = workers['baseline'].request(mode='validate_common')
                candidate = workers['candidate'].request(mode='validate_common')
            else:
                baseline = dict(ref['validation']['candidate_by_case'])
                baseline.update(workers['baseline'].request(mode='validate_common'))
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
    helpers = ['benchmarks/bench_pf_doppler_likelihood_compare.py', *ref['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources, extension_sha256=binaries, benchmark_sha256={p: _sha(ROOT/p) for p in helpers},
        baseline_reference=dict(path=reference_path, sha256=_sha(ROOT/reference_path)),
        build_configuration=ref['build_configuration'],
        method='Persistent separate workers and stage inputs/owners; sequential alternating stage batches; nonexclusive host.',
        scope='Synchronized Doppler update and complete native/public predict/PR/Doppler/ESS/conditional B60 resample/estimate or five-epoch history; initialization/restoration/IPC/hash/input setup excluded; every timed weight and returned result hashed; full state at each batch end.',
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, warmup_calls_per_batch=3,
        validation=validation, summary=summary, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path)
    for name in ('baseline', 'candidate', 'baseline-dir', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--common-only', action='store_true')
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
