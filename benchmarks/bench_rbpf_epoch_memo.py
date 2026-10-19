"""Paired complete-step screen: epoch canonical/DD memo against D228 move reuse."""

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
from rbpf_epoch_memo import EpochMemoStep

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    modes = ['prior_cpu', 'key_cpu', 'epoch_cpu', 'epoch_gpu', 'AA']
    warm_modes = modes + ['prior_gpu']
    names = ('benchmarks/bench_rbpf_epoch_memo.py', 'benchmarks/rbpf_epoch_memo.py', 'benchmarks/rbpf_move_reuse.py',
        'benchmarks/bench_pf_private_weights_compare.py',
        'benchmarks/bench_rbpf_move_field.py', 'benchmarks/rbpf_move_batch.py', 'benchmarks/rbpf_move_cases.py',
        'benchmarks/rbpf_likelihood_batch.py', 'benchmarks/rbpf_likelihood_cases.py',
        'benchmarks/rbpf_likelihood_runtime_adapter.py', 'experiments/rbpf_fgo/rbpf_fgo.py')
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: sha(ROOT/name) for name in names}, DLL_sha256=sha(args.dll),
        modes=modes, rounds=5, repeats=3,
        conditions=[dict(particles=n, initial_unique=u, rows=m, groups=g) for n,u,m,g in
            ((64,6,64,4), (64,32,64,4), (64,64,64,4), (128,32,64,4), (64,6,128,8), (128,128,128,8))],
        scope='Entire frozen RBPF step; extraction/time/satellite/integer-search inputs are fixtures, graph solve excluded. Includes keying, move generation, score batch/transfers, conditionals, mass/FIX and resampling.',
        method='One allowed CPU/one BLAS thread, normal priority, nonexclusive host/GPU. Five-mode serial rotation,5rounds x3groups; paired reduction median within each round then median5. Restore/setup/warm/validation/serialization outside step timer.',
        gate='Full epoch CPU vs D228 CPU: >=10% paired reduction and>=4/5 positive rounds with abs A/A paired median<5%. Incremental DD vs key-only also >=10% and4/5; GPU vs full epoch CPU uses same gate. High-diversity/noisy failures retained; no retry.',
        exact_gate='All saved state/discrete/RNG/position/covariance bits identical against D228 with the same backend. Key-only isolates canonical reuse; full epoch includes DD. Untimed prior GPU is the GPU exact reference. No numerical tolerance.')
    planpath = args.output.with_suffix('.plan.json')
    checkpoint = args.output.with_suffix('.checkpoint.json')
    snapshots = args.output.with_suffix('.states.jsonl.gz')
    assert not any(p.exists() for p in (args.output, planpath, checkpoint, snapshots))
    planpath.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    _pin()
    results = {}
    with threadpool_limits(limits=1, user_api='blas'), BatchScorer(args.dll) as scorer, gzip.open(snapshots, 'wt', encoding='utf8', newline='\n') as stream:
        for condition in plan['conditions']:
            key = '/'.join(f'{k}{v}' for k,v in condition.items())
            fixtures = {}
            for mode in warm_modes:
                runtime, rtk, ctx = make_step_case(condition['particles'], condition['initial_unique'],
                    rows=condition['rows'], groups=condition['groups'], cluster=condition['rows'] == 128)
                initial = deepcopy(vars(runtime))
                backend = 'gpu' if mode.endswith('_gpu') else 'native_cpu'
                if mode.startswith('prior_') or mode == 'AA':
                    call = MoveReuseStep(runtime, scorer, backend)
                else:
                    call = EpochMemoStep(runtime, scorer, backend, cache_dd=mode != 'key_cpu')
                fixtures[mode] = (runtime, initial, rtk, ctx, call)

            def run(mode):
                runtime, initial, rtk, ctx, call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.step = call
                rr, cc = deepcopy(rtk), deepcopy(ctx)
                start = time.perf_counter_ns()
                output = call(rr, cc)
                elapsed = (time.perf_counter_ns() - start)/1e6
                return elapsed, state(runtime, rr, output), call.last

            references, warm = {}, {}
            for mode in warm_modes:
                _ms, value, details = run(mode)
                references[mode] = value
                warm[mode] = dict(caller_metadata=encode(details))
                stream.write(json.dumps(dict(condition=key, mode=mode, state=encode(value)), separators=(',', ':'))+'\n')
            targets = dict(prior_cpu='prior_cpu', prior_gpu='prior_gpu', key_cpu='prior_cpu',
                epoch_cpu='prior_cpu', epoch_gpu='prior_gpu', AA='prior_cpu')
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
                for baseline in ('prior_cpu', 'key_cpu', 'epoch_cpu'):
                    reductions = [statistics.median(100*(1-p['runs'][mode]['ms']/p['runs'][baseline]['ms'])
                        for p in pairs if p['round'] == r) for r in range(5)]
                    item['vs_'+baseline] = dict(paired_round_reduction_percent=reductions,
                        paired_reduction_percent=statistics.median(reductions), improving_rounds=sum(v>0 for v in reductions))
                summary[mode] = item
            results[key] = dict(condition=condition, warm=warm, pairs=pairs, summary=summary)
            print(key, json.dumps({mode: dict(ms=v['p50_ms'], vs_prior_cpu=v['vs_prior_cpu']) for mode,v in summary.items()}), flush=True)
    assert all(sha(ROOT/name) == digest for name,digest in plan['source_sha256'].items())
    assert sha(args.dll) == plan['DLL_sha256']
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath), measurements=results,
        states_sha256=sha(snapshots), source_and_DLL_unchanged=True), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
