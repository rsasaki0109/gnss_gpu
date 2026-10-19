"""Complete frozen PF-step screen of the call-local identity parameter-row reuse."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from bench_pf_private_weights_compare import _pin
from bench_rbpf_feedback_mass import _bind_fixture_callbacks, _rtk_digest
from bench_rbpf_irls_replay import digest
from bench_rbpf_move_field import encode
from rbpf_cluster_parameter_reuse import ClusterBatchScorer
from rbpf_cluster_fast_guard import ClusterBatchScorer as PreviousScorer
from rbpf_feedback_cases import make_feedback_case
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_move_cases import comparison, state
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ('original_cpu', 'off_cpu', 'on_cpu', 'original_gpu', 'off_gpu', 'on_gpu', 'cpu_AA', 'gpu_AA', 'baseline_cpu', 'baseline_gpu')
CONTRASTS = (('cpu_prior', 'original_cpu', 'on_cpu'), ('cpu_control', 'off_cpu', 'on_cpu'),
    ('gpu_prior', 'original_gpu', 'on_gpu'), ('gpu_control', 'off_gpu', 'on_gpu'),
    ('gpu_vs_cpu', 'on_cpu', 'on_gpu'), ('cpu_AA', 'original_cpu', 'cpu_AA'), ('gpu_AA', 'original_gpu', 'gpu_AA'),
    ('cpu_increment', 'baseline_cpu', 'on_cpu'), ('gpu_increment', 'baseline_gpu', 'on_gpu'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(records):
    contrasts = {}
    for key, a, b in CONTRASTS:
        rounds = [statistics.median(100*(1-next(r['ns'] for r in records if (r['round'],r['group'],r['mode']) == (rd,g,b)) /
            next(r['ns'] for r in records if (r['round'],r['group'],r['mode']) == (rd,g,a))) for g in range(3)) for rd in range(5)]
        median = statistics.median(rounds)
        positive = sum(x > 0 for x in rounds)
        contrasts[key] = dict(reference=a,candidate=b,round_paired_percent=rounds,
            median_paired_percent=median,improving_rounds=positive,numeric_screen=median >= (3 if key.endswith('_increment') else 10) and positive >= 4)
    caa = abs(contrasts['cpu_AA']['median_paired_percent']) < 5
    gaa = abs(contrasts['gpu_AA']['median_paired_percent']) < 5
    return dict(contrasts=contrasts,cpu_AA_valid=caa,gpu_AA_valid=gaa,
        CPU_pass=caa and all(contrasts[k]['numeric_screen'] for k in ('cpu_prior','cpu_control','cpu_increment')),
        GPU_same_backend_pass=gaa and all(contrasts[k]['numeric_screen'] for k in ('gpu_prior','gpu_control','gpu_increment')),
        GPU_joint_pass=caa and gaa and all(contrasts[k]['numeric_screen'] for k in ('gpu_prior','gpu_control','gpu_increment','gpu_vs_cpu')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dll','cluster-extension','canonical-extension','parent-extension','mass-extension','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'))
    _pin()
    parent = json.loads((ROOT/'dist/rbpf_cluster_parameter_reuse_tests_20261017.plan.json').read_text(encoding='utf8'))
    paths = set(parent['source_sha256']) | {'benchmarks/bench_rbpf_cluster_parameter_reuse_step.py',
        'benchmarks/bench_rbpf_feedback_mass.py','benchmarks/bench_rbpf_irls_replay.py'}
    paths.update(p.relative_to(ROOT).as_posix() for p in (args.dll,args.cluster_extension,args.canonical_extension,args.parent_extension,args.mass_extension))
    frozen = {n:sha(ROOT/n) for n in paths}
    conditions = [dict(n=n,u=u,rows=m,groups=g,cluster=int(m == 128),feedback=True,preload=True)
        for n,u,m,g in ((64,6,64,4),(64,32,64,4),(64,64,64,4),(128,32,64,4),(64,6,128,8),(128,128,128,8))]
    conditions += [dict(n=64,u=6,rows=64,groups=4,cluster=0,feedback=False,preload=False),
        dict(n=64,u=6,rows=64,groups=4,cluster=1,feedback=True,preload=True)]
    planpath = args.output.with_suffix('.plan.json')
    assert not args.output.exists() and not planpath.exists()
    plan = dict(source_and_binary_sha256=frozen,modes=MODES,contrasts=CONTRASTS,rounds=5,groups=3,conditions=conditions,
        acceptance='Paired median3 groups per round then median5,>=10percent original/off and >=3percent previous wrapper,>=4positive rounds each,abs matching AAmedian<5percent. CPU prior/off/increment;GPU prior/off/increment;GPUjoint also onCPU and both AA. All failures/negative/noisy rounds retained,no completed timing retry.',
        scope='Complete literal D236 PF step,all particles/candidates/priors/choice/RNG/conditionals/FIX/feedback/graph callbacks. Extract/search/time/satellite/graph-solve inputs are fixture boundaries. Includes all guards/native selector/counter updates/packing/numerics/transfers/readback. Restore/setup/bind/hash/validation/serialization outside. One allowed CPU/BLAS1/normal priority/nonexclusive. Not whole GTSAM/actual-route performance.')
    planpath.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    all_results, warm_count, timed_count = {}, 0, 0
    with ExitStack() as stack:
        original = stack.enter_context(RegisterLUScorer(args.dll))
        off = stack.enter_context(ClusterBatchScorer(args.dll,args.cluster_extension,enabled=False))
        on = stack.enter_context(ClusterBatchScorer(args.dll,args.cluster_extension))
        previous = stack.enter_context(PreviousScorer(args.dll,args.cluster_extension))
        scorers = {m:on if m.startswith('on_') else off if m.startswith('off_') else original for m in MODES}
        scorers['baseline_cpu'] = scorers['baseline_gpu'] = previous
        raw = stack.enter_context(gzip.open(args.output.with_suffix('.calls.jsonl.gz'),'wt',encoding='utf8',newline='\n'))
        snapshots = stack.enter_context(gzip.open(args.output.with_suffix('.states.jsonl.gz'),'wt',encoding='utf8',newline='\n'))
        for ci, condition in enumerate(conditions):
            fixtures = {}
            for mode in MODES:
                runtime,rtk,ctx = make_feedback_case(condition['n'],condition['u'],rows=condition['rows'],groups=condition['groups'],
                    cluster=bool(condition['cluster']),feedback=condition['feedback'],preload=condition['preload'])
                initial = deepcopy(vars(runtime))
                backend = 'gpu' if mode.endswith('gpu') or mode == 'gpu_AA' else 'native_cpu'
                call = NativeFeedbackMassStep(runtime,scorers[mode],backend,extension=args.canonical_extension,
                    parent_extension=args.parent_extension,mass_extension=args.mass_extension)
                fixtures[mode] = runtime,rtk,ctx,initial,call

            def run(mode):
                runtime,rtk,ctx,initial,call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.step = call
                template = _rtk_digest(rtk)
                rr,cc = deepcopy(rtk),deepcopy(ctx)
                if condition['feedback']:
                    _bind_fixture_callbacks(rr)
                engine = getattr(scorers[mode],'cluster_engine',None)
                before = engine.counts() if engine is not None else None
                start = time.perf_counter_ns()
                output = call(rr,cc)
                elapsed = time.perf_counter_ns()-start
                assert template == _rtk_digest(rtk)
                stats = {k:v-before[k] for k,v in engine.counts().items()} if engine is not None else None
                return elapsed,state(runtime,rr,output),deepcopy(call.last),stats

            references, meta, warm, records = {}, {}, {}, []
            for mode in MODES:
                ns,value,details,stats = run(mode)
                backend = 'gpu' if mode.endswith('gpu') or mode == 'gpu_AA' else 'native_cpu'
                if mode.startswith('original_'):
                    references[backend],meta[backend] = value,details
                check = comparison(references[backend],value)
                assert check['discrete_equal'] and check['numeric_bits_equal'] and check['max_abs_error'] == 0.
                check = comparison(meta[backend],details)
                assert check['discrete_equal'] and check['numeric_bits_equal'] and check['max_abs_error'] == 0.
                row = dict(phase='warm',condition=ci,mode=mode,ns=ns,state_sha256=digest(value),metadata_sha256=digest(details),cluster_batch=stats)
                warm[mode] = row
                raw.write(json.dumps(row,separators=(',',':'))+'\n')
                snapshots.write(json.dumps(dict(condition=ci,mode=mode,state=encode(value),metadata=encode(details)),separators=(',',':'))+'\n')
                warm_count += 1
            for rd in range(5):
                for g in range(3):
                    shift = (ci+rd*3+g) % len(MODES)
                    order = MODES[shift:]+MODES[:shift]
                    if (ci+rd+g) % 2:
                        order = order[::-1]
                    for mode in order:
                        ns,value,details,stats = run(mode)
                        assert digest(value) == warm[mode]['state_sha256'] and digest(details) == warm[mode]['metadata_sha256']
                        assert stats == warm[mode]['cluster_batch']
                        row = dict(phase='timed',condition=ci,round=rd,group=g,mode=mode,order=order,ns=ns,
                            state_sha256=digest(value),metadata_sha256=digest(details),cluster_batch=stats)
                        raw.write(json.dumps(row,separators=(',',':'))+'\n')
                        records.append(row)
                        timed_count += 1
            all_results[str(ci)] = dict(condition=condition,warm=warm,summary=summaries(records))
            raw.flush()
            snapshots.flush()
            print(json.dumps(dict(condition_complete=ci,summary=all_results[str(ci)]['summary'])),flush=True)
    assert warm_count == 80 and timed_count == 1200
    assert all(sha(ROOT/n) == d for n,d in frozen.items())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath),calls_sha256=sha(args.output.with_suffix('.calls.jsonl.gz')),
        states_sha256=sha(args.output.with_suffix('.states.jsonl.gz')),warm_calls=warm_count,timed_calls=timed_count,
        results=all_results,all_same_backend_state_metadata_RNG_graph_bits_equal=True,source_and_binaries_unchanged=True,
        actual_whole_graph_or_route_speed_established=False,default_changed=False),indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
