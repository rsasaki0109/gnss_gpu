"""Complete likelihood batches: exact native DD packing with the same math DLL."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from threadpoolctl import threadpool_limits

from bench_pf_private_weights_compare import _pin
from rbpf_likelihood_batch import BatchScorer
from rbpf_likelihood_cases import make_case
from rbpf_dd_pack import PackedDdScorer

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dll',type=Path,required=True)
    parser.add_argument('--pack-dll',type=Path,required=True)
    parser.add_argument('--build-record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    planpath = args.output.with_suffix('.plan.json')
    checkpoint = args.output.with_suffix('.checkpoint.json')
    assert not any(p.exists() for p in (args.output,planpath,checkpoint))
    build = json.loads(args.build_record.read_text(encoding='utf8'))
    assert build['dll_sha256'] == sha(args.pack_dll)
    sources = dict(build['source_sha256'])
    for name in ('benchmarks/bench_rbpf_dd_pack_likelihood.py','benchmarks/rbpf_dd_pack.py',
            'benchmarks/rbpf_likelihood_batch.py','benchmarks/rbpf_likelihood_cases.py',
            'benchmarks/bench_pf_private_weights_compare.py','experiments/rbpf_fgo/rbpf_fgo.py'):
        sources[name] = sha(ROOT/name)
    assert all(sha(ROOT/name) == value for name,value in sources.items())
    modes = ['prior_cpu','prior_gpu','packed_cpu','packed_gpu','AA']
    plan = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,
        math_DLL_sha256=sha(args.dll),pack_DLL_sha256=sha(args.pack_dll),build_record_sha256=sha(args.build_record),
        modes=modes,rounds=5,repeats=3,
        conditions=[dict(rows=64,groups=4,unique=n,cluster=False,uniform=True) for n in (4,21,32,64,128,512)]+
            [dict(rows=128,groups=8,unique=n,cluster=True,uniform=False) for n in (21,64,128)],
        gate='Per condition: packed vs prior same backend>=10% paired reduction,>=4of5 positive rounds,abs prior GPU A/A<5%. Packed GPU additionally vs packed CPU>=10% and>=4of5. Component only; no retries or automatic adoption.',
        exact='Every LL/status/normal/solution/logdet bit must match the prior same-backend scorer. The same frozen native math DLL is loaded for both; no tolerance. DD integer arithmetic remains arbitrary precision. Whole PF/real-data gates separate.',
        scope='Complete requested likelihood batch: canonical/dedup, original DD/cluster lookup, packing, all pageable uploads, kernel/native execution, synchronous readback and scatter. Owning capacity setup, warm/hash/validation excluded; not whole PF/graph.',
        method='One allowed CPU,one BLAS thread,normal priority,nonexclusive host/GPU. Serial5-mode rotation; group percent reduction median within each round then median5. Retain every condition/round and control.')
    planpath.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    _pin()
    results = {}
    with ExitStack() as stack:
        stack.enter_context(threadpool_limits(limits=1,user_api='blas'))
        old = stack.enter_context(BatchScorer(args.dll))
        packed = stack.enter_context(PackedDdScorer(args.dll,pack_dll=args.pack_dll))
        assert old.dll._handle == packed.dll._handle
        setup = dict(prior=old.setup_ms,packed=packed.setup_ms)
        for condition in plan['conditions']:
            oracle,problem,hypotheses = make_case(condition['unique'],rows=condition['rows'],groups=condition['groups'],
                cluster=condition['cluster'],marg_uniform=condition['uniform'])
            inputs = {name:digest(value) for name,value in vars(problem).items() if isinstance(value,np.ndarray)}
            hypothesis_hash = hashlib.sha256(repr(hypotheses).encode()).hexdigest()
            calls = dict(prior_cpu=(old,'native_cpu'),prior_gpu=(old,'gpu'),packed_cpu=(packed,'native_cpu'),packed_gpu=(packed,'gpu'),AA=(old,'gpu'))
            expected,scores,diagnostics = {},{},{}
            for mode,(scorer,backend) in calls.items():
                values,details = scorer.score(hypotheses,problem,oracle,backend,diagnostics=True)
                assert details['unique'] == condition['unique'] and details['scalar_fallbacks'] == 0
                expected[mode],scores[mode] = digest(values),values.tolist()
                diagnostics[mode] = digest(details['diagnostic'])
            assert len({expected[m] for m in ('prior_gpu','packed_gpu','AA')}) == 1
            assert len({diagnostics[m] for m in ('prior_gpu','packed_gpu','AA')}) == 1

            assert expected['prior_cpu'] == expected['packed_cpu']
            assert diagnostics['prior_cpu'] == diagnostics['packed_cpu']

            def call(mode):
                scorer,backend = calls[mode]
                start = time.perf_counter_ns()
                values,details = scorer.score(hypotheses,problem,oracle,backend)
                ms = (time.perf_counter_ns()-start)/1e6
                assert details['scalar_fallbacks'] == 0 and digest(values) == expected[mode]
                return dict(ms=ms,output_sha256=expected[mode])

            for mode in modes:
                call(mode)
            pairs = []
            for r in range(5):
                for k in range(3):
                    offset = (r*3+k) % 5
                    order = modes[offset:]+modes[:offset]
                    pairs.append(dict(round=r,repeat=k,order=order,runs={mode:call(mode) for mode in order}))
            summary = {}
            for mode in modes:
                item = dict(p50_ms=statistics.median(p['runs'][mode]['ms'] for p in pairs))
                for baseline in ('prior_cpu','prior_gpu','packed_cpu'):
                    v = [statistics.median(100*(1-p['runs'][mode]['ms']/p['runs'][baseline]['ms']) for p in pairs if p['round'] == r) for r in range(5)]
                    item['vs_'+baseline] = dict(paired_round_reduction_percent=v,paired_reduction_percent=statistics.median(v),improving_rounds=sum(x>0 for x in v))
                summary[mode] = item
            assert inputs == {name:digest(value) for name,value in vars(problem).items() if isinstance(value,np.ndarray)}
            assert hypothesis_hash == hashlib.sha256(repr(hypotheses).encode()).hexdigest()
            key = f"rows{condition['rows']}/groups{condition['groups']}/unique{condition['unique']}"
            results[key] = dict(condition=condition,requested=len(hypotheses),input_sha256=inputs,hypotheses_sha256=hypothesis_hash,
                scores=scores,expected_output_sha256=expected,diagnostic_sha256=diagnostics,pairs=pairs,summary=summary)
            checkpoint.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
            print(key,json.dumps(dict(ms={m:summary[m]['p50_ms'] for m in modes},packed_gpu=summary['packed_gpu'],packed_cpu=summary['packed_cpu'],AA=summary['AA']['vs_prior_gpu'])),flush=True)
    assert all(sha(ROOT/name) == value for name,value in sources.items())
    assert sha(args.dll) == plan['math_DLL_sha256'] and sha(args.pack_dll) == plan['pack_DLL_sha256']
    args.output.write_text(json.dumps(dict(plan=plan,plan_sha256=sha(planpath),setup_ms=setup,measurements=results,
        sources_and_DLLs_unchanged=True,product_changes=[]),indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
