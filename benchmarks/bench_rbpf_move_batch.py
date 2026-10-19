"""Paired complete frozen-step synthetic screen of explicit move batching."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from bench_rbpf_move_field import encode
from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_cases import SCALAR_SOURCE
from rbpf_move_batch import MoveBatchStep
from rbpf_move_cases import comparison, make_step_case, state

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    modes = ['original', 'scalar', 'native_cpu', 'gpu', 'AA']
    names = ['benchmarks/bench_rbpf_move_batch.py', 'benchmarks/bench_rbpf_move_field.py',
        'benchmarks/rbpf_move_batch.py', 'benchmarks/rbpf_move_cases.py',
        'benchmarks/rbpf_likelihood_batch.py', 'benchmarks/rbpf_likelihood_cases.py',
        'benchmarks/rbpf_likelihood_runtime_adapter.py', str(SCALAR_SOURCE.relative_to(ROOT))]
    sources = {name: sha(ROOT/name) for name in names}
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        DLL_sha256=sha(args.dll), rounds=5, repeats=3, modes=modes,
        conditions=[dict(particles=n, initial_unique=u, rows=m, groups=g) for n, u, m, g in
            ((64,6,64,4), (64,16,64,4), (64,32,64,4), (128,32,64,4), (64,6,128,8), (128,32,128,8))],
        scope='Entire frozen RBPF step over seeded synthetic epoch inputs, including move generation exactly once per held particle, conditionals, mass/FIX decision and resampling. Extraction and integer-search inputs are fixtures. Graph solving/real data are excluded.',
        method='One allowed CPU, one BLAS thread, normal priority, nonexclusive host/GPU. Serial rotating five-mode groups; paired within-group reductions median per round then median five rounds. Setup/restore/validation/serialization outside step timer. One workspace, one score batch per epoch, no numerical cache across epochs.',
        gate='For each candidate vs original: >=10% paired reduction and >=4/5 positive rounds, A/A absolute paired median <5%. GPU-specific screen additionally needs same vs native_cpu. Diagnostic only; no default adoption without full-route multi-seed accuracy and whole graph performance.',
        validation='Exact discrete decisions/RNG and conditional position/P bits; scalar staging all numeric bits exact; native continuous state difference <1e-10 as a diagnostic, not a changed accuracy threshold. No retry to obtain a pass.')
    planpath = args.output.with_suffix('.plan.json')
    checkpoint = args.output.with_suffix('.checkpoint.json')
    assert not planpath.exists() and not checkpoint.exists()
    planpath.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    pin = _pin()
    results = {}
    snapshots = args.output.with_suffix('.states.jsonl.gz')
    with threadpool_limits(limits=1, user_api='blas'), BatchScorer(args.dll) as scorer, gzip.open(snapshots, 'wt', encoding='utf8', newline='\n') as stream:
        for condition in plan['conditions']:
            key = '/'.join(f'{k}{v}' for k, v in condition.items())
            fixtures = {}
            for mode in modes:
                runtime, rtk, ctx = make_step_case(condition['particles'], condition['initial_unique'],
                    rows=condition['rows'], groups=condition['groups'], cluster=condition['rows'] == 128)
                initial = deepcopy(vars(runtime))
                call = runtime.step if mode in ('original', 'AA') else MoveBatchStep(runtime, scorer, mode)
                fixtures[mode] = (runtime, initial, rtk, ctx, call)

            def run(mode):
                runtime, initial, rtk, ctx, call = fixtures[mode]
                runtime.__dict__.clear()
                runtime.__dict__.update(deepcopy(initial))
                runtime.step = call
                rr, cc = deepcopy(rtk), deepcopy(ctx)
                start = time.perf_counter_ns()
                output = call(rr, cc)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                return elapsed, state(runtime, rr, output), (None if mode in ('original', 'AA') else call.last)

            _elapsed, reference, _meta = run('original')
            warm = {}
            for mode in modes:
                _elapsed, value, details = run(mode)
                check = comparison(reference, value)
                assert check['discrete_equal'] and check['position_bits_equal'], (key, mode, check)
                assert check['max_abs_error'] < 1e-10, (key, mode, check)
                if mode in ('original', 'scalar', 'AA'):
                    assert check['numeric_bits_equal'], (key, mode, check)
                warm[mode] = dict(validation=check, caller_metadata=encode(details))
                stream.write(json.dumps(dict(condition=key, mode=mode, state=encode(value)), separators=(',', ':'))+'\n')
            pairs = []
            for round_index in range(5):
                for repeat in range(3):
                    offset = (round_index * 3 + repeat) % len(modes)
                    order = modes[offset:] + modes[:offset]
                    sample = dict(round=round_index, repeat=repeat, order=order, runs={})
                    for mode in order:
                        elapsed, value, details = run(mode)
                        check = comparison(reference, value)
                        assert check['discrete_equal'] and check['position_bits_equal'] and check['max_abs_error'] < 1e-10, (key, mode, check)
                        if mode in ('original', 'scalar', 'AA'):
                            assert check['numeric_bits_equal']
                        digest = hashlib.sha256(json.dumps(encode(value), separators=(',', ':')).encode()).hexdigest()
                        sample['runs'][mode] = dict(ms=elapsed, state_sha256=digest, validation=check)
                    pairs.append(sample)
                    checkpoint.write_text(json.dumps(dict(completed_conditions=list(results), condition=key,
                        samples=pairs), indent=2)+'\n', encoding='utf8')
            summary = {}
            for mode in modes:
                item = dict(p50_ms=statistics.median(pair['runs'][mode]['ms'] for pair in pairs))
                for baseline in ('original', 'native_cpu'):
                    reductions = [statistics.median(100. * (1. - pair['runs'][mode]['ms'] / pair['runs'][baseline]['ms'])
                        for pair in pairs if pair['round'] == r) for r in range(5)]
                    item['vs_'+baseline] = dict(paired_round_reduction_percent=reductions,
                        paired_reduction_percent=statistics.median(reductions), improving_rounds=sum(v > 0. for v in reductions))
                summary[mode] = item
            results[key] = dict(condition=condition, pairs=pairs, warm=warm, summary=summary)
            print(key, json.dumps(summary), flush=True)
        stream.flush()
    assert all(sha(ROOT/name) == digest for name, digest in sources.items())
    assert sha(args.dll) == plan['DLL_sha256']
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath), pin=pin,
        measurements=results, states_sha256=sha(snapshots), source_and_DLL_unchanged=True), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
