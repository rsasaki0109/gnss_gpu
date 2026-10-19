"""Paired complete-step screen: immutable score-index aliases against D236."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from bench_rbpf_move_field import encode
from rbpf_likelihood_batch import BatchScorer
from rbpf_move_batch import MoveBatchStep
from rbpf_move_cases import comparison, make_step_case, state
from rbpf_move_reuse import MoveReuseStep
from rbpf_native_canonical import NativeCanonicalStep
from rbpf_parent_score import NativeParentScoreStep
from rbpf_component_mass import NativeComponentMassStep
from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_score_index_memo import ScoreIndexMemoStep
from rbpf_feedback_cases import make_feedback_case
from rbpf_move_key_memo import MoveKeyMemoStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_fixture_callbacks(rtk):
    # Same bodies as make_step_case, but bound to this trial's copied RTK.
    # deepcopy does not rebind Python function closures.
    def inject(values,keys):
        rtk.events.append(('inject',tuple(keys),values.copy()))
        rtk.nav.x = values.copy()

    def release(sat,freq,reason):
        rtk.events.append(('release',sat,freq,reason))
        rtk._committed.pop((sat,freq),None)
        rtk._rbpf_released.add((sat,freq))

    rtk._inject_hold,rtk._release_ambiguity = inject,release


def _rtk_digest(rtk):
    value = {k:v for k,v in vars(rtk).items() if not callable(v)}
    return hashlib.sha256(json.dumps(encode(value),separators=(',',':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--extension', type=Path, required=True)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--canonical-extension', type=Path, required=True)
    parser.add_argument('--parent-extension', type=Path, required=True)
    parser.add_argument('--index-extension',type=Path,required=True)
    parser.add_argument('--index-build-record',type=Path,required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    modes = ['prior_cpu', 'index_cpu', 'index_gpu', 'prior_gpu', 'AA']
    warm_modes = modes
    names = ('benchmarks/rbpf_score_index_memo.py','benchmarks/rbpf_move_phase_profile.py','benchmarks/rbpf_phase_profile.py','benchmarks/bench_rbpf_score_index_memo.py', 'benchmarks/rbpf_native_canonical.py', 'benchmarks/rbpf_parent_score.py', 'benchmarks/rbpf_component_mass.py', 'benchmarks/rbpf_feedback_mass.py', 'benchmarks/rbpf_feedback_cases.py', 'benchmarks/rbpf_move_key_memo.py', 'benchmarks/rbpf_epoch_memo.py', 'benchmarks/rbpf_move_reuse.py',
        'benchmarks/bench_pf_private_weights_compare.py',
        'benchmarks/bench_rbpf_move_field.py', 'benchmarks/rbpf_move_batch.py', 'benchmarks/rbpf_move_cases.py',
        'benchmarks/rbpf_likelihood_batch.py', 'benchmarks/rbpf_likelihood_cases.py',
        'benchmarks/rbpf_likelihood_runtime_adapter.py', 'experiments/rbpf_fgo/rbpf_fgo.py')
    names += tuple(p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_canonical').iterdir() if p.is_file())
    names += tuple(p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_parent_score').iterdir() if p.is_file())
    names += tuple(p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_component_mass').iterdir() if p.is_file())
    names += tuple(p.relative_to(ROOT).as_posix() for p in (ROOT/'benchmarks/probes/rbpf_score_index').iterdir() if p.is_file())
    indexbuild=json.loads(args.index_build_record.read_text(encoding='utf8'))
    assert sha(args.index_extension)==indexbuild['extension_sha256']
    assert all(sha(ROOT/n)==v for n,v in indexbuild['source_sha256'].items())
    build = json.loads(args.build_record.read_text(encoding='utf8'))
    assert build['extension_sha256'] == sha(args.extension)
    assert all(sha(ROOT/name) == value for name,value in build['source_sha256'].items())
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(ROOT/name) for name in names}, DLL_sha256=sha(args.dll),extension_sha256=sha(args.extension),build_record_sha256=sha(args.build_record),canonical_extension_sha256=sha(args.canonical_extension),parent_extension_sha256=sha(args.parent_extension),
        modes=modes, rounds=5, repeats=3,
        conditions=[dict(particles=n, initial_unique=u, rows=m, groups=g, feedback=True, preload=True) for n,u,m,g in
            ((64,6,64,4), (64,32,64,4), (64,64,64,4), (128,32,64,4), (64,6,128,8), (128,128,128,8))]
            +[dict(particles=64,initial_unique=6,rows=64,groups=4,feedback=False,preload=False)],
        scope='Entire frozen RBPF step; extraction/time/satellite/integer-search inputs are fixtures, graph solve excluded. Includes canonical keys, feedback-mass guards/preflight/hold rounding/DD expansion/ordered sums, move generation, score batch/transfers, conditionals, mass/FIX and resampling. Six feedback-enabled conditions preload source-consistent holds;one disabled control. Original release/streak/commit/graph callback code unchanged. Per-run copied RTK callbacks are rebound outside timing;template RTK is verified unchanged.',
        method='One allowed CPU/one BLAS thread, normal priority, nonexclusive host/GPU. Five-mode serial rotation,5rounds x3groups; paired reduction median within each round then median5. Restore/setup/warm/validation/serialization outside step timer.',
        gate='Per condition: native-feedback CPU vs D235 CPU >=10% paired reduction and>=4/5 positive rounds, abs D235 CPU A/A median<5%. Native-feedback GPU must additionally beat native-feedback CPU AND D235 GPU by>=10% with>=4/5 positive rounds. All seven conditions/failures retained; no retries or automatic default adoption.',
        exact_gate='All state/discrete/RNG/position/covariance bits identical against D235 with the same backend. Only held-DD feedback statistics change, reusing unchanged D235 native extension. Original hold rounding/group order,integer DD conversion and particle-order FP64 sums;release/commit code and prelude remain literal. No stale cross-phase DD cache,tree reduction,new tolerance or threshold/RNG/candidate change. Actual graph/full-route accuracy remains a separate gate.')
    plan.update(index_extension_sha256=sha(args.index_extension),index_build_record_sha256=sha(args.index_build_record),
        scope='Entire frozen D236 PF-step versus index aliases,without any profiler/spans. Includes new guards/helper scope refresh,IndexMemo construction,key validation/native alias allocations,canonical requests and all prior staging/score packing/transfers/probabilities/RNG/conditional/FIX/feedback/resampling. Seven same workloads;index aliases active in all seven. Extract/search/time/satellite/graph callbacks are fixtures,not graph solve speed.',
        gate='CPU vs D236CPU >=10percent paired reduction and>=4of5positive rounds,abs CPU A/A median<5percent. GPU joint also requires this versus D236GPU AND newCPU. Every failure/negative retained,no retries/default adoption.',
        exact_gate='All saved state/position/P/RNG bits and every prior caller metadata field must match D236 same backend. Canonical called for every move,original key/index tail for every miss,original particle probability expressions and RNG choice once per particle. Only builtin immutable retained key identity aliases;custom keys disable aliases. No new tolerance or model/particle/candidate/prior/threshold change.')
    planpath = args.output.with_suffix('.plan.json')
    checkpoint = args.output.with_suffix('.checkpoint.json')
    snapshots = args.output.with_suffix('.states.jsonl.gz')
    assert not any(p.exists() for p in (args.output, planpath, checkpoint, snapshots))
    planpath.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    _pin()
    results = {}
    with threadpool_limits(limits=1, user_api='blas'), BatchScorer(args.dll) as scorer, BatchScorer(args.dll) as native, gzip.open(snapshots, 'wt', encoding='utf8', newline='\n') as stream:
        for condition in plan['conditions']:
            key = '/'.join(f'{k}{v}' for k,v in condition.items())
            fixtures = {}
            for mode in warm_modes:
                runtime, rtk, ctx = make_feedback_case(condition['particles'], condition['initial_unique'],
                    rows=condition['rows'], groups=condition['groups'], cluster=condition['rows'] == 128,
                    feedback=condition['feedback'],preload=condition['preload'])
                initial = deepcopy(vars(runtime))
                backend = 'gpu' if mode.endswith('_gpu') else 'native_cpu'
                if mode.startswith('prior_') or mode == 'AA':
                    call = NativeFeedbackMassStep(runtime, scorer, backend, extension=args.canonical_extension, parent_extension=args.parent_extension, mass_extension=args.extension)
                else:
                    call = ScoreIndexMemoStep(runtime, native, backend, index_extension=args.index_extension, extension=args.canonical_extension, parent_extension=args.parent_extension, mass_extension=args.extension)
                fixtures[mode] = (runtime, initial, rtk, ctx, call)

            def run(mode):
                runtime, initial, rtk, ctx, call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.step = call
                template = _rtk_digest(rtk)
                rr, cc = deepcopy(rtk), deepcopy(ctx)
                if condition['feedback']:
                    _bind_fixture_callbacks(rr)
                start = time.perf_counter_ns()
                output = call(rr, cc)
                elapsed = (time.perf_counter_ns() - start)/1e6
                assert _rtk_digest(rtk) == template, 'Fixture template mutated through copied callbacks'
                return elapsed, state(runtime, rr, output), call.last

            references, warm = {}, {}
            for mode in warm_modes:
                _ms, value, details = run(mode)
                references[mode] = value
                warm[mode] = dict(caller_metadata=encode(details))
                stream.write(json.dumps(dict(condition=key, mode=mode, state=encode(value)), separators=(',', ':'))+'\n')
            targets = dict(prior_cpu='prior_cpu', prior_gpu='prior_gpu', index_cpu='prior_cpu', index_gpu='prior_gpu', AA='prior_cpu')
            for mode in modes:
                check = comparison(references[targets[mode]], references[mode])
                assert check['discrete_equal'] and check['numeric_bits_equal'] and check['max_abs_error'] == 0., (key, mode, check)
                warm[mode]['validation'] = check
            pairs = []
            for round_index in range(5):
                for repeat in range(3):
                    offset = (round_index * 3 + repeat) % len(modes)
                    order = modes[offset:] + modes[:offset]
                    sample = dict(round=round_index, repeat=repeat, order=order, runs={})
                    for mode in order:
                        ms, value, _details = run(mode)
                        check = comparison(references[targets[mode]], value)
                        assert check['discrete_equal'] and check['numeric_bits_equal'] and check['max_abs_error'] == 0., (key, mode, check)
                        digest = hashlib.sha256(json.dumps(encode(value), separators=(',', ':')).encode()).hexdigest()
                        sample['runs'][mode] = dict(ms=ms, state_sha256=digest, validation=check)
                    pairs.append(sample)
                    checkpoint.write_text(json.dumps(dict(completed_conditions=list(results), condition=key, samples=pairs), indent=2)+'\n', encoding='utf8')
            summary = {}
            for mode in modes:
                item = dict(p50_ms=statistics.median(p['runs'][mode]['ms'] for p in pairs))
                for baseline in ('prior_cpu', 'prior_gpu', 'index_cpu'):
                    reductions = [statistics.median(100*(1-p['runs'][mode]['ms']/p['runs'][baseline]['ms'])
                        for p in pairs if p['round'] == r) for r in range(5)]
                    item['vs_'+baseline] = dict(paired_round_reduction_percent=reductions,
                        paired_reduction_percent=statistics.median(reductions), improving_rounds=sum(v>0 for v in reductions))
                summary[mode] = item
            results[key] = dict(condition=condition, warm=warm, pairs=pairs, summary=summary)
            print(key, json.dumps({mode: dict(ms=v['p50_ms'], vs_prior_cpu=v['vs_prior_cpu']) for mode,v in summary.items()}), flush=True)
    assert all(sha(ROOT/name) == digest for name,digest in plan['source_sha256'].items())
    assert sha(args.dll) == plan['DLL_sha256'] and sha(args.extension) == plan['extension_sha256']
    assert sha(args.canonical_extension) == plan['canonical_extension_sha256'] and sha(args.parent_extension) == plan['parent_extension_sha256']
    assert sha(args.index_extension)==plan['index_extension_sha256']
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath), measurements=results,
        states_sha256=sha(snapshots), source_and_DLL_unchanged=True), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
