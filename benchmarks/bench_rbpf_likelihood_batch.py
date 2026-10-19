"""Measure complete packed likelihood batches against literal scalar and C++."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_cases import make_case, SCALAR_SOURCE, SCALAR_SHA256

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--build-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    checkpoint = args.output.with_suffix('.checkpoint.json')
    planpath = args.output.with_suffix('.plan.json')
    assert not checkpoint.exists() and not planpath.exists()
    build = json.loads(args.build_record.read_text(encoding='utf8'))
    assert sha(args.dll) == build['dll_sha256']
    for name, value in build['source_sha256'].items():
        assert sha(ROOT / name) == value
    sources = dict(build['source_sha256'])
    for name in ('benchmarks/bench_rbpf_likelihood_batch.py', 'benchmarks/rbpf_likelihood_batch.py',
                 'benchmarks/rbpf_likelihood_cases.py', 'tests/test_rbpf_likelihood_batch_probe.py'):
        sources[name] = sha(ROOT / name)
    assert sha(SCALAR_SOURCE) == SCALAR_SHA256
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        scalar_source_sha256=SCALAR_SHA256, dll_sha256=sha(args.dll), build_record_sha256=sha(args.build_record),
        unique_counts=[16, 21, 32, 64, 128, 512], rounds=5, repeats=3,
        conditions=['64rows/4groups/uniform/IRLS2/shared_prior', '128rows/8groups/Gaussian/IRLS2/individual_priors'],
        performance_screen='GPU must improve>=10% against both the literal scalar and packed native CPU in>=4/5 paired rounds; same-code scalar A/A median abs<5%. Apply per condition without retries. Existing public N64 relevance additionally requires N21 and N64 first condition, not just larger batches.',
        precision='Record every LL bit, absolute LL/softmax errors and stable ranking vs original. Test diagnostic LL1e-8/conditional1e-9 limits do not authorize any positioning tolerance change. No product adoption without original trajectory/discrete/RNG/accuracy gates; exact original bits recorded separately.',
        timing_scope='Full requested likelihood batch including gauge-invariant dedup, original DD/cluster lookup, NumPy packing, every epoch input upload, fused kernel or native CPU computation, scalar fallback if any and host readback/scatter. Owning native/GPU capacity setup/warm/hash/validation excluded and setup cost reported. Not a complete RBPF epoch.',
        invariants='All actually requested candidates retained, first-request order/scatter preserved, original2 IRLS passes, Gaussian/uniform/noise model and full covariance input. No candidate move generation, RNG or graph changes. NumPy/BLAS/native library reduction rounding can differ.',
        method='Own last allowed CPU, one BLAS thread, normal priority, serial rotating scalar/nativeCPU/GPU/scalar-AA. Nonexclusive host/GPU; median within-pair percentages per round then median of5 rounds.')
    planpath.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf8')
    _pin()
    measurements = {}
    with threadpool_limits(limits=1, user_api='blas'), BatchScorer(args.dll, capacity=512) as scorer:
        setup = scorer.setup_ms
        for rows, groups, cluster, uniform in ((64, 4, False, True), (128, 8, True, False)):
            for count in plan['unique_counts']:
                oracle, problem, hypotheses = make_case(count, rows=rows, groups=groups, cluster=cluster, marg_uniform=uniform)
                reference = BatchScorer.scalar(hypotheses, problem, oracle)
                expected = {'scalar': digest(reference), 'AA': digest(reference)}
                validation = {}
                native_values = {}
                for mode in ('native_cpu', 'gpu'):
                    values, details = scorer.score(hypotheses, problem, oracle, mode)
                    assert details['unique'] == count and details['scalar_fallbacks'] == 0, details
                    error = float(np.max(np.abs(values - reference)))
                    assert np.allclose(values, reference, rtol=1e-12, atol=1e-8), (rows, count, mode, error)
                    expected[mode] = digest(values)
                    native_values[mode] = values.tolist()
                    left = np.exp(reference - np.max(reference))
                    left /= np.sum(left)
                    right = np.exp(values - np.max(values))
                    right /= np.sum(right)
                    validation[mode] = dict(max_LL_abs_difference=error, all_LL_bits_identical=expected[mode] == expected['scalar'],
                        maximum_softmax_difference=float(np.max(np.abs(left-right))),
                        stable_ranking_identical=bool(np.array_equal(np.argsort(-reference, kind='stable'), np.argsort(-values, kind='stable'))))

                def call(mode):
                    start = time.perf_counter_ns()
                    if mode in ('scalar', 'AA'):
                        values = BatchScorer.scalar(hypotheses, problem, oracle)
                    else:
                        values, details = scorer.score(hypotheses, problem, oracle, mode)
                        assert details['scalar_fallbacks'] == 0
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    assert digest(values) == expected[mode]
                    return dict(ms=elapsed, output_sha256=expected[mode])

                for mode in ('scalar', 'native_cpu', 'gpu', 'AA'):
                    call(mode)
                pairs = []
                modes = ['scalar', 'native_cpu', 'gpu', 'AA']
                for r in range(5):
                    for repeat in range(3):
                        offset = (r + repeat) % 4
                        order = modes[offset:] + modes[:offset]
                        runs = {mode: call(mode) for mode in order}
                        pairs.append(dict(round=r, repetition=repeat, order=order, runs=runs))
                summary = {mode: dict(p50_ms=float(np.median([p['runs'][mode]['ms'] for p in pairs])),
                    p95_ms=float(np.percentile([p['runs'][mode]['ms'] for p in pairs], 95))) for mode in modes}
                for mode, baseline in (('gpu', 'scalar'), ('native_cpu', 'scalar'), ('AA', 'scalar'), ('gpu', 'native_cpu')):
                    reductions = [float(np.median([100 * (1-p['runs'][mode]['ms']/p['runs'][baseline]['ms']) for p in pairs if p['round'] == r])) for r in range(5)]
                    summary[mode]['vs_' + baseline] = dict(paired_round_reduction_percent=reductions,
                        paired_reduction_percent=float(np.median(reductions)), improving_rounds=sum(v > 0 for v in reductions))
                key = f'rows{rows}/groups{groups}/unique{count}'
                measurements[key] = dict(requested=len(hypotheses), unique=count, cluster=cluster, uniform=uniform,
                    input_sha256={name: digest(value) for name, value in vars(problem).items() if isinstance(value, np.ndarray)},
                    hypotheses_sha256=hashlib.sha256(repr(hypotheses).encode()).hexdigest(), reference_LL=reference.tolist(),
                    native_LL=native_values, validation=validation, pairs=pairs, summary=summary)
                checkpoint.write_text(json.dumps(measurements, indent=2) + '\n', encoding='utf8')
                print(key, json.dumps(dict(summary=summary, validation=validation)), flush=True)
    assert all(sha(ROOT / name) == value for name, value in sources.items()) and sha(args.dll) == build['dll_sha256']
    args.output.write_text(json.dumps(dict(plan=plan, plan_sha256=sha(planpath),
        setup_ms=setup, measurements=measurements, source_and_DLL_unchanged=True, product_changes=[]), indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
