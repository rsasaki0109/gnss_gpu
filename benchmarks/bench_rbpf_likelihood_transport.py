"""Compare three frozen likelihood transports and audit ambiguity near ties."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_cases import make_case, SCALAR_SOURCE

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def probability(values):
    weights = np.exp(values - np.max(values))
    return weights / np.sum(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-records', nargs=3, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    planpath = args.output.with_suffix('.plan.json')
    checkpoint = args.output.with_suffix('.checkpoint.json')
    assert not planpath.exists() and not checkpoint.exists()
    labels = ('separate', 'pinned', 'pageable')
    builds = {label: json.loads(path.read_text(encoding='utf8')) for label, path in zip(labels, args.build_records)}
    dlls = {label: path.with_suffix('') / 'Release/rbpf_likelihood_probe.dll' for label, path in zip(labels, args.build_records)}
    sources = {name: value for build in builds.values() for name, value in build['source_sha256'].items()}
    for path in (Path(__file__), ROOT/'benchmarks/rbpf_likelihood_batch.py', ROOT/'benchmarks/rbpf_likelihood_cases.py', SCALAR_SOURCE):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    assert all(sha(ROOT/name) == value for name, value in sources.items())
    assert all(sha(dlls[label]) == build['dll_sha256'] for label, build in builds.items())
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        DLL_sha256={label: sha(path) for label, path in dlls.items()},
        build_record_sha256={label: sha(path) for label, path in zip(labels, args.build_records)},
        rounds=5, repeats=3, counts=[16,21,32,64,128,512], modes=['scalar','native_cpu','separate','pinned','pageable','AA'],
        gate='Pageable candidate: >=10% paired reduction and>=4/5 positive rounds against both scalar and native CPU, condition A/A abs<5%. Current N64 relevance needs rows64/N21 andN64. Preserve both earlier failed/frozen experiments; controls here do not override their decisions. No retry.',
        near_tie='Separate untimed symmetric integer pair at half-cycle plus0, +-2^-48, +-2^-40 and+-.1 cycles: record full scores/posteriors/stable order. No new tolerance or positioning claim.',
        scope='Full likelihood batch with all gauge keys, DD/cluster lookup, packing, epoch uploads, native/kernel calculation, readback and scatter. Workspace setup/warm/hash/near-tie diagnostics outside timing. Not a complete RBPF epoch.',
        method='One allowed CPU, one BLAS thread, normal priority, nonexclusive host/GPU. Serial six-mode rotation; median pair percent within round, then median5 rounds. Earlier variants retained as controls; numeric code unchanged.')
    planpath.write_text(json.dumps(plan, indent=2)+'\n', encoding='utf8')
    _pin()
    rows = {}
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(limits=1, user_api='blas'))
        scorers = {label: stack.enter_context(BatchScorer(path, capacity=512)) for label, path in dlls.items()}
        assert len({s.dll._handle for s in scorers.values()}) == 3
        setup = {label: value.setup_ms for label, value in scorers.items()}
        ambiguous = {}
        for delta in (0., -2.**-48, 2.**-48, -2.**-40, 2.**-40, -.1, .1):
            oracle, problem, initial = make_case(2, duplicates=False)
            h0 = initial[1]
            h1 = deepcopy(h0)
            group, _ref, sat = problem.pairs[0]
            h1[group][sat] -= 1
            hypotheses = [h0,h1]
            problem.rows_v0[:] = 0.
            cp = problem.rows_iscp
            integers = oracle._pair_dd(h0, problem)
            problem.rows_bmeas[cp] = integers
            problem.rows_bmeas[len(problem.pairs)] += .5 + delta
            reference = BatchScorer.scalar(hypotheses, problem, oracle)
            record = dict(original_LL=reference.tolist(), original_probability=probability(reference).tolist())
            for mode in ('native_cpu', *labels):
                scorer = scorers['separate' if mode == 'native_cpu' else mode]
                values, details = scorer.score(hypotheses, problem, oracle, 'native_cpu' if mode == 'native_cpu' else 'gpu')
                assert details['scalar_fallbacks'] == 0
                record[mode] = dict(LL=values.tolist(), probability=probability(values).tolist(),
                    max_LL_error=float(np.max(np.abs(values-reference))),
                    stable_ranking_identical=bool(np.array_equal(np.argsort(-values, kind='stable'), np.argsort(-reference, kind='stable'))))
            ambiguous[str(delta)] = record
        for m, groups, cluster, uniform in ((64,4,False,True),(128,8,True,False)):
            for count in plan['counts']:
                oracle, problem, hypotheses = make_case(count, rows=m, groups=groups, cluster=cluster, marg_uniform=uniform)
                expected, validation, outputs = {}, {}, {}
                for mode in plan['modes']:
                    if mode in ('scalar','AA'):
                        values = BatchScorer.scalar(hypotheses, problem, oracle)
                    else:
                        scorer = scorers['separate' if mode == 'native_cpu' else mode]
                        values, details = scorer.score(hypotheses, problem, oracle, 'native_cpu' if mode == 'native_cpu' else 'gpu')
                        assert details['unique'] == count and details['scalar_fallbacks'] == 0
                    outputs[mode] = values
                    expected[mode] = digest(values)
                    if mode not in ('scalar','AA'):
                        error = float(np.max(np.abs(values-outputs['scalar'])))
                        assert np.allclose(values, outputs['scalar'], rtol=1e-12, atol=1e-8)
                        validation[mode] = dict(max_LL_abs_difference=error,
                            maximum_probability_difference=float(np.max(np.abs(probability(values)-probability(outputs['scalar'])))),
                            original_bits_identical=expected[mode] == expected['scalar'],
                            stable_ranking_identical=bool(np.array_equal(np.argsort(-values, kind='stable'), np.argsort(-outputs['scalar'], kind='stable'))))
                assert expected['separate'] == expected['pinned'] == expected['pageable']

                def call(mode):
                    start = time.perf_counter_ns()
                    if mode in ('scalar','AA'):
                        values = BatchScorer.scalar(hypotheses, problem, oracle)
                    else:
                        scorer = scorers['separate' if mode == 'native_cpu' else mode]
                        values, details = scorer.score(hypotheses, problem, oracle, 'native_cpu' if mode == 'native_cpu' else 'gpu')
                        assert details['scalar_fallbacks'] == 0
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    assert digest(values) == expected[mode]
                    return dict(ms=elapsed, output_sha256=expected[mode])

                for mode in plan['modes']:
                    call(mode)
                pairs = []
                modes = plan['modes']
                for r in range(5):
                    for repeat in range(3):
                        offset = (r+repeat) % len(modes)
                        order = modes[offset:]+modes[:offset]
                        pairs.append(dict(round=r, repetition=repeat, order=order, runs={mode: call(mode) for mode in order}))
                summary = {mode: dict(p50_ms=float(np.median([p['runs'][mode]['ms'] for p in pairs])),
                    p95_ms=float(np.percentile([p['runs'][mode]['ms'] for p in pairs],95))) for mode in modes}
                for mode, baseline in [(mode,'scalar') for mode in modes if mode!='scalar'] + [(mode,'native_cpu') for mode in labels]:
                    values = [float(np.median([100*(1-p['runs'][mode]['ms']/p['runs'][baseline]['ms']) for p in pairs if p['round']==r])) for r in range(5)]
                    summary[mode]['vs_'+baseline] = dict(paired_round_reduction_percent=values,
                        paired_reduction_percent=float(np.median(values)), improving_rounds=sum(v>0 for v in values))
                key = f'rows{m}/groups{groups}/unique{count}'
                rows[key] = dict(unique=count, requested=len(hypotheses), cluster=cluster, uniform=uniform,
                    input_sha256={name: digest(value) for name,value in vars(problem).items() if isinstance(value,np.ndarray)},
                    hypotheses_sha256=hashlib.sha256(repr(hypotheses).encode()).hexdigest(),
                    scores={mode: value.tolist() for mode,value in outputs.items()}, validation=validation, pairs=pairs, summary=summary)
                checkpoint.write_text(json.dumps(dict(measurements=rows,near_tie=ambiguous), indent=2)+'\n',encoding='utf8')
                print(key,json.dumps(summary),flush=True)
    assert all(sha(ROOT/name)==value for name,value in sources.items())
    assert all(sha(path)==plan['DLL_sha256'][label] for label,path in dlls.items())
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath), setup_ms=setup,
        near_tie=ambiguous, measurements=rows, source_and_DLLs_unchanged=True, product_changes=[]),indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
