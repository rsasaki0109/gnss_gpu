"""Compare exact scalar log-weight fill during persistent particle-pool migration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from bench_pf_common_predict_compare import Session as BaseSession
from bench_pf_covariance_broadcast_compare import ROOT, _load_extension, _setup as _setup_pf, _load_python, _sha
from bench_pf_covariance_compare import _digest
from pf_uniform_weights_cases import validate
from bench_pf_particle_readout_compare import Session as ReadoutSession
from bench_pf_mixture_geometry_compare import Session as MixtureSession


def _setup(native,python_dir=None):
    klass=_setup_pf(native,python_dir)
    folder=python_dir or ROOT/'python/gnss_gpu'
    _load_python(folder/'realtime_runtime.py','gnss_gpu.realtime_runtime')
    return klass


STAGES=('public_uniform_weights','public_uniform_weights_five','public_pool_roundtrip','public_pool_posterior','public_pool_PR_five','public_states','public_restore','PR_control')


class Session(ReadoutSession):
    def __init__(self,native,count,stage):
        super().__init__(native,count,stage)
        self.peer=None
        self.saved=None
        if stage in ('public_restore','public_state_roundtrip'):
            self.prepare()
            self.saved=self.pf.get_particle_states()
        if stage.startswith('public_pool_'):
            from gnss_gpu.particle_filter_device import ParticleFilterDevice
            from gnss_gpu.realtime_runtime import PersistentParticleRuntimePool
            self.peer=ParticleFilterDevice(n_particles=count//2,sigma_pos=1.,sigma_cb=3.,sigma_pr=5.,seed=42)
            self.pool=PersistentParticleRuntimePool(lambda n: self.peer if n==count//2 else self.pf,(count//2,count))
            self.pool.runtimes={count:self.pf,count//2:self.peer}

    def prepare(self):
        super().prepare()
        if self.peer is not None:
            self.peer.initialize(self.position,clock_bias=3000.,spread_pos=10.,spread_cb=30.,
                velocity=[1.,2.,3.],spread_vel=.5,velocity_init_sigma=2.)
            self.pool.active=self.pf
            if self.stage=='public_pool_posterior':
                self.pf.update(self.sat,self.pr,self.weights,resample=False)

    def operation(self):
        if self.stage in ('public_uniform_weights','public_uniform_weights_five'):
            import math
            value=-math.log(self.count)
            for _ in range(5 if self.stage.endswith('five') else 1):
                method=getattr(self.pf,'set_uniform_log_weights',None)
                if method is None:
                    self.pf.set_log_weights(np.full(self.count,value,dtype=np.float64))
                else:
                    method(value)
            return (np.empty(0),)
        if self.stage=='public_states':
            return (self.pf.get_particle_states(),)
        if self.stage=='public_restore':
            self.pf.set_particle_states(self.saved)
            return (np.empty(0),)
        if self.stage=='public_state_roundtrip':
            value=self.pf.get_particle_states()
            self.pf.set_particle_states(value)
            return (value,)
        if self.stage in ('public_pool_roundtrip','public_pool_posterior'):
            self.pool.activate(self.count//2)
            self.pool.activate(self.count)
            return (self.pf.estimate(),self.peer.estimate())
        if self.stage=='public_pool_PR_five':
            history=[]
            for _ in range(5):
                history.append(BaseSession.operation(self))
                self.pool.activate(self.count//2)
                self.pool.activate(self.count)
                history.append(self.pf.estimate())
            return tuple(history)
        if 'state_' in self.stage:
            outputs=[]
            for _ in range(5 if self.stage.endswith('five') else 1):
                outputs.append(BaseSession.operation(self))
                outputs.append(self.pf.get_particle_states())
            return tuple(outputs)
        return (MixtureSession.operation(self),)

    def snapshot(self):
        from bench_pf_covariance_compare import _snapshot
        extra=_snapshot(self.native,self.peer._state) if self.peer is not None else ()
        return dict(input_sha256=self.input_sha256,
                    output_sha256=_digest(*_snapshot(self.native,self.state),*extra,*self.last_returned))

    def close(self):
        if self.peer is not None:
            self.native.pf_device_destroy(self.peer._state)
            self.peer._state=None
        super().close()

    def measure(self, repeats, snapshot=True, warmups=1):
        samples, hashes, returns = [], set(), set()
        weights = np.empty(self.count)
        for index in range(repeats+warmups):
            self.prepare()
            self.native.pf_device_sync(self.state)
            start = time.perf_counter_ns()
            value = self.operation()
            self.native.pf_device_sync(self.state)
            elapsed = (time.perf_counter_ns()-start)/1e6
            self.native.pf_device_get_log_weights(self.state, weights)
            hashes.add(_digest(weights))
            returns.add(_digest(*value))
            self.last_returned = value
            if index >= warmups:
                samples.append(elapsed)
        assert len(hashes) == len(returns) == 1
        result = dict(input_sha256=self.input_sha256, samples_ms=samples, median_ms=float(np.median(samples)),
            log_weights_sha256=hashes.pop(), returned_output_sha256=returns.pop(),
            all_timed_log_weights_identical=True, all_returned_outputs_identical=True)
        if snapshot:
            result.update(self.snapshot())
        return result



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
                if request['mode'] == 'validate':
                    result = validate(native, klass)
                else:
                    current = request['size'], request['stage']
                    if current != key:
                        if session:
                            session.close()
                        session, key = Session(native, *current), current
                    if request['mode'] == 'snapshot':
                        result = session.snapshot()
                    elif request['mode'] == 'memory':
                        result = session.memory()
                    else:
                        result = session.measure(request['repeats'], request.get('snapshot', True), request.get('warmups', 1))
                print(json.dumps(result), flush=True)
            except Exception as error:
                import traceback
                print(json.dumps(dict(error=repr(error), traceback=traceback.format_exc())), flush=True)
    finally:
        if session:
            session.close()


class Worker:
    def __init__(self, extension, python_dir=None):
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--extension', str(extension.resolve())]
        if python_dir:
            argv += ['--python-dir', str(python_dir.resolve())]
        label = 'baseline' if python_dir else 'candidate'
        self.error_log = (ROOT/f'dist/pf_uniform_weights_{label}_worker_20261014.log').open('wb')
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.error_log, text=True, encoding='utf8', cwd=ROOT)
        (ROOT/f'dist/pf_uniform_weights_{label}_worker_pid_20261014.json').write_text(
            json.dumps(dict(pid=self.process.pid)), encoding='utf8')
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
        self.error_log.close()


def compare(args):
    edit = json.loads((ROOT/'dist/pf_uniform_weights_source_edit_20261014.json').read_text(encoding='utf8'))
    refpath = 'benchmarks/pf_pool_normalize_gpu_20261014.json'
    ref = json.loads((ROOT/refpath).read_text(encoding='utf8'))
    sources = edit['source_sha256']
    for name, digest in sources['baseline'].items():
        assert _sha(args.baseline_dir/Path(name).name) == digest, name
    for name, digest in sources['candidate'].items():
        assert _sha(ROOT/name) == digest, name
    binaries = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert all(sources['baseline'][p]==d for p,d in ref['source_sha256']['candidate'].items())
    assert binaries['baseline'] == ref['extension_sha256']['candidate']
    helpers = ['benchmarks/bench_pf_uniform_weights_compare.py', 'benchmarks/pf_uniform_weights_cases.py',
        *[p for p in ref['benchmark_sha256'] if p not in sources['candidate']]]
    for name, digest in ref['benchmark_sha256'].items():
        if name not in sources['candidate']:
            assert _sha(ROOT/name) == digest, name
    frozen_helpers = {p: _sha(ROOT/p) for p in helpers}
    reused = None
    if args.reference:
        reused = json.loads(args.reference.read_text(encoding='utf8'))
        assert reused['source_sha256'] == sources and reused['extension_sha256'] == binaries
        assert reused['benchmark_sha256'] == frozen_helpers
        assert reused['validation']['all_outputs_bit_identical'] and reused['validation']['cases'] >= 1567
    workers, summary, runs, memory = {}, {}, {}, {}
    try:
        workers['baseline'] = Worker(args.baseline, args.baseline_dir)
        workers['candidate'] = Worker(args.candidate)
        if reused:
            validation = reused['validation']
        else:
            values = {label: worker.request(mode='validate') for label, worker in workers.items()}
            assert values['baseline'] == values['candidate'], 'State, weight, query or rejection mismatch'
            validation = dict(cases=len(values['baseline']), all_outputs_bit_identical=True,
                baseline_by_case=values['baseline'], candidate_by_case=values['candidate'])
            print(f'{validation["cases"]} original/candidate records identical', flush=True)
        for count in args.sizes:
            for stage in args.stages:
                key = f'N{count}/{stage}'
                data, medians, pairs, snapshots = {label: [] for label in workers}, {label: [] for label in workers}, [], []
                for round_index in range(args.rounds):
                    order = ('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')
                    if args.paired:
                        # Warm each process once per round, then alternate one
                        # restored synchronized call without redundant warmups.
                        for label in order:
                            workers[label].request(mode='measure', size=count, stage=stage, repeats=1, snapshot=False, warmups=0)
                        times = {label: [] for label in workers}
                        for sample in range(args.repeats):
                            order = ('baseline', 'candidate') if (round_index+sample) % 2 == 0 else ('candidate', 'baseline')
                            pair = {label: workers[label].request(mode='measure', size=count, stage=stage,
                                repeats=1, snapshot=False, warmups=0) for label in order}
                            for field in ('input_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                                assert pair['baseline'][field] == pair['candidate'][field], (key, field)
                            pairs.append(dict(round=round_index, sample=sample, order=order, runs=pair))
                            for label in workers:
                                times[label].append(pair[label]['samples_ms'][0])
                        snapshots.append({label: worker.request(mode='snapshot', size=count, stage=stage) for label, worker in workers.items()})
                        assert snapshots[-1]['baseline'] == snapshots[-1]['candidate'], key
                        for label in workers:
                            medians[label].append(float(np.median(times[label])))
                    else:
                        for label in order:
                            data[label].append(workers[label].request(mode='measure', size=count, stage=stage, repeats=args.repeats))
                        for label in workers:
                            medians[label].append(data[label][-1]['median_ms'])
                    print(f'{key} round {round_index+1}: {medians["baseline"][-1]:.4f}->{medians["candidate"][-1]:.4f}ms', flush=True)
                if not args.paired:
                    for field in ('input_sha256', 'output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                        assert len({v[field] for rows in data.values() for v in rows}) == 1, (key, field)
                old, new = [float(np.median(medians[label])) for label in workers]
                summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                    improving_paired_rounds=sum(b<a for a,b in zip(medians['baseline'],medians['candidate'])),
                    round_medians_ms=medians)
                runs[key] = dict(stage_batches=data, pairs=pairs, round_snapshots=snapshots)
                if args.memory and '_K' in stage:
                    values = {label: worker.request(mode='memory', size=count, stage=stage) for label,worker in workers.items()}
                    assert values['baseline']['output_sha256'] == values['candidate']['output_sha256'], key
                    memory[key] = values
    finally:
        for worker in workers.values():
            worker.close()
    assert frozen_helpers == {p: _sha(ROOT/p) for p in helpers}, 'Loaded helpers changed during measurement'
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources, extension_sha256=binaries, benchmark_sha256=frozen_helpers,
        baseline_reference=dict(path=refpath,sha256=_sha(ROOT/refpath)),build_configuration=ref['build_configuration'],
        validation=validation, validation_reference=None if not args.reference else dict(path=str(args.reference.relative_to(ROOT)).replace('\\','/'),sha256=_sha(args.reference)),
        method='Separate persistent workers, frozen baseline Python, original/candidate native PE; sequential GPU/CPU calls; nonexclusive host.',
        scope='Persistent pool N->N/2->N for uniform/prior PR posterior, and continuous five predict/PR/ESS/conditional resample/estimate/migration epochs; resident runtimes. Full-state get/set and PR controls unchanged. Initialization/restoration/IPC/hash/input setup excluded.',
        rounds=args.rounds,repeats=args.repeats,sizes=args.sizes,paired=args.paired,
        warmups='One call per batch, or one call per process per paired round; each timed call is restored first.',
        summary=summary,runs=runs,tracemalloc_peak_measurements=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('extension', 'python-dir', 'baseline', 'candidate', 'baseline-dir', 'output', 'reference'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--sizes',nargs='+',type=int,default=[2048,1000000])
    parser.add_argument('--stages',nargs='+',default=STAGES)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--paired',action='store_true')
    parser.add_argument('--memory',action='store_true')
    args=parser.parse_args()
    if args.extension:
        worker(args.extension,args.python_dir)
        return
    if not all((args.baseline,args.candidate,args.baseline_dir,args.output)):
        parser.error('baseline,candidate,baseline-dir,output required')
    if min(*args.sizes,args.rounds,args.repeats)<1:
        parser.error('positive sizes,rounds,repeats required')
    if args.reference:
        args.reference=args.reference.resolve()
    args.output.write_text(json.dumps(compare(args),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
