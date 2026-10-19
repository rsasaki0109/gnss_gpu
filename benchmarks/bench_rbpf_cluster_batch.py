"""Controlled complete-score timings of batched cluster lookup on saved inputs."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from bench_pf_private_weights_compare import _pin
from bench_rbpf_irls_replay import digest, prepare
from rbpf_cluster_batch import ClusterBatchScorer
from rbpf_irls_replay import load_corpus, oracle_from_snapshot
from rbpf_register_lu import RegisterLUScorer

ROOT = Path(__file__).resolve().parents[1]
MODES = ('original_cpu', 'off_cpu', 'on_cpu', 'original_gpu', 'off_gpu', 'on_gpu', 'cpu_AA', 'gpu_AA')
CONTRASTS = (('cpu_prior', 'original_cpu', 'on_cpu'), ('cpu_control', 'off_cpu', 'on_cpu'),
    ('gpu_prior', 'original_gpu', 'on_gpu'), ('gpu_control', 'off_gpu', 'on_gpu'),
    ('gpu_vs_cpu', 'on_cpu', 'on_gpu'), ('cpu_AA', 'original_cpu', 'cpu_AA'), ('gpu_AA', 'original_gpu', 'gpu_AA'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summaries(records):
    output = {}
    for ci in (0, 1):
        totals = {(rd, g, m): sum(r['ns'] for r in records
            if (r['cluster'], r['round'], r['group'], r['mode']) == (ci, rd, g, m))
            for rd in range(5) for g in range(3) for m in MODES}
        contrasts = {}
        for key, a, b in CONTRASTS:
            rounds = [statistics.median(100*(1-totals[rd,g,b]/totals[rd,g,a]) for g in range(3)) for rd in range(5)]
            median = statistics.median(rounds)
            improving = sum(x > 0 for x in rounds)
            contrasts[key] = dict(reference=a, candidate=b, round_paired_percent=rounds,
                median_paired_percent=median, improving_rounds=improving, numeric_screen=median >= 10 and improving >= 4)
        cpu_aa = abs(contrasts['cpu_AA']['median_paired_percent']) < 5
        gpu_aa = abs(contrasts['gpu_AA']['median_paired_percent']) < 5
        output[str(ci)] = dict(contrasts=contrasts, cpu_AA_valid=cpu_aa, gpu_AA_valid=gpu_aa,
            cpu_component_pass=cpu_aa and all(contrasts[k]['numeric_screen'] for k in ('cpu_prior', 'cpu_control')),
            gpu_same_backend_component_pass=gpu_aa and all(contrasts[k]['numeric_screen'] for k in ('gpu_prior', 'gpu_control')),
            gpu_joint_component_pass=cpu_aa and gpu_aa and all(contrasts[k]['numeric_screen'] for k in ('gpu_prior', 'gpu_control', 'gpu_vs_cpu')),
            median_corpus_ns={m: statistics.median(totals[rd,g,m] for rd in range(5) for g in range(3)) for m in MODES})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dll', 'cluster-extension', 'extension', 'parent-extension', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--corpus', type=Path, action='append', required=True)
    parser.add_argument('--corpus-sha', action='append', required=True)
    args = parser.parse_args()
    inputs = list(zip(args.corpus, args.corpus_sha, strict=True))
    assert len(inputs) == 2 and all(sha(p) == d for p,d in inputs)
    assert all(os.environ.get(k) == '1' for k in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'))
    _pin()
    reference = ROOT / 'dist/rbpf_sparse_normal_measurements_20261017.json'
    assert sha(reference) == '690e28b8a43e79aa29022554803a1d4178dcb334178ccf4d696dda0f2a7c6eac'
    expected = {(r['cluster'], r['index'], 'gpu' if r['mode'] == 'old_gpu' else 'native_cpu'): r
        for r in json.loads(reference.read_text(encoding='utf8'))['warm'] if r['mode'] in ('old_gpu', 'cpu')}
    assert len(expected) == 788
    paths = [Path(__file__), ROOT/'benchmarks/rbpf_cluster_batch.py', ROOT/'benchmarks/rbpf_likelihood_batch.py',
        ROOT/'benchmarks/rbpf_register_lu.py', ROOT/'benchmarks/bench_rbpf_irls_replay.py', ROOT/'benchmarks/rbpf_irls_replay.py',
        ROOT/'benchmarks/rbpf_native_canonical.py', ROOT/'benchmarks/rbpf_parent_score.py', ROOT/'experiments/rbpf_fgo/rbpf_fgo.py',
        args.dll, args.cluster_extension, args.extension, args.parent_extension, reference, *args.corpus]
    frozen = {p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    planpath = args.output.with_suffix('.plan.json')
    assert not args.output.exists() and not planpath.exists()
    plan = dict(source_and_input_sha256=frozen, modes=MODES, contrasts=CONTRASTS, rounds=5, groups=3, cases_per_cluster=197,
        acceptance='Every paired corpus aggregate:median3 groups per round then median5,>=10percent and>=4positive rounds. CPU requires prior/control and absCPUAA<5percent;GPU samebackend prior/control with absGPUAA<5percent;GPUjoint also onCPU comparison and both AA. Retain failures/noisy/negative values,no completed timing retry.',
        scope='Complete score including dedup,all guards,all native CPU selection and logical counter updates,allocations,original parameter arithmetic,native numerics/transfers/waits/readback/fallback. Input restore/setup/warm/hash/count validation/serialization outside. One allowed CPU/BLAS1/normal priority/nonexclusive. Component,not wholePF/FGO speed.')
    planpath.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    corpora = [load_corpus(p) for p,_ in inputs]
    assert all(len(c) == 197 and c[0]['inputs']['cfg']['cluster_relin'] == ci for ci,c in enumerate(corpora))
    warm, records, logical = [], [], {}
    with ExitStack() as stack:
        original = stack.enter_context(RegisterLUScorer(args.dll))
        off = stack.enter_context(ClusterBatchScorer(args.dll, args.cluster_extension, enabled=False))
        on = stack.enter_context(ClusterBatchScorer(args.dll, args.cluster_extension))
        scorers = {m: on if m.startswith('on_') else off if m.startswith('off_') else original for m in MODES}
        oracles = {(ci,m):oracle_from_snapshot(c[0]['inputs'],args.extension,args.parent_extension)
            for ci,c in enumerate(corpora) for m in MODES}
        raw = stack.enter_context(gzip.open(args.output.with_suffix('.calls.jsonl.gz'),'wt',encoding='utf8',newline='\n'))

        def call(ci, entry, mode, diagnostics):
            scorer, oracle = scorers[mode], oracles[ci,mode]
            hs, problem = prepare(oracle,entry['inputs'])
            ce = oracle.basin_id.__self__
            pe = oracle._cluster_state.__self__._functions['_basin_parent_score'].__closure__[0].cell_contents
            old_counts = (ce.counts(), pe.counts())
            engine = getattr(scorer,'cluster_engine',None)
            before_stats = engine.counts() if engine is not None else None
            before = digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states)) if diagnostics else None
            backend = 'gpu' if mode.endswith('gpu') or mode == 'gpu_AA' else 'native_cpu'
            start = time.perf_counter_ns()
            values, details = scorer.score(hs,problem,oracle,backend,diagnostics=diagnostics)
            elapsed = time.perf_counter_ns()-start
            new_counts = (ce.counts(), pe.counts())
            counts = [{k:new[k]-old[k] for k in old} for old,new in zip(old_counts,new_counts,strict=True)]
            if diagnostics and mode == 'original_cpu':
                logical[ci,entry['index']] = counts
            assert counts == logical[ci,entry['index']]
            stats = {k:v-before_stats[k] for k,v in engine.counts().items()} if engine is not None else None
            if mode.startswith('on_'):
                assert stats['batch_calls'] == ci and stats['fallback_calls'] == 1-ci
                assert stats['canonical_requests'] <= counts[0]['native_calls'] and stats['parent_requests'] == counts[1]['native_calls']
            dsha = digest((values,details)) if diagnostics else None
            expected_row = expected[ci,entry['index'],backend]
            if diagnostics:
                assert dsha == expected_row['diagnostic_sha256']
                assert before == digest((hs,problem,oracle.cfg,oracle._cluster_state.__self__.cluster_states))
                details = {k:v for k,v in details.items() if k not in ('diagnostic','unique_hypotheses')}
            result_sha = digest((values,details))
            assert result_sha == expected_row['result_sha256'] and details['scalar_fallbacks'] == 0
            return dict(ns=elapsed, result_sha256=result_sha, diagnostic_sha256=dsha,
                warm_inputs_equal=diagnostics, unique=details['unique'], logical_counts=counts, cluster_batch=stats)

        for ci, corpus in enumerate(corpora):
            for entry in corpus:
                for mode in MODES:
                    row = dict(phase='warm',cluster=ci,index=entry['index'],mode=mode,**call(ci,entry,mode,True))
                    warm.append(row)
                    raw.write(json.dumps(row,separators=(',',':'))+'\n')
            print(json.dumps(dict(warm_complete=ci)),flush=True)
        for rd in range(5):
            for group in range(3):
                for ci, corpus in enumerate(corpora):
                    for entry in corpus:
                        shift = (rd*3+group+ci+entry['index']) % len(MODES)
                        order = MODES[shift:]+MODES[:shift]
                        if (rd+group+ci) % 2:
                            order = order[::-1]
                        for mode in order:
                            row = dict(phase='timed',round=rd,group=group,cluster=ci,index=entry['index'],mode=mode,
                                order=order,**call(ci,entry,mode,False))
                            records.append(row)
                            raw.write(json.dumps(row,separators=(',',':'))+'\n')
                raw.flush()
            print(json.dumps(dict(round_complete=rd,timed_calls=len(records))),flush=True)
    assert len(warm) == 3152 and len(records) == 47280
    assert all(sha(ROOT/n) == d for n,d in frozen.items())
    args.output.write_text(json.dumps(dict(plan_sha256=sha(planpath),calls_sha256=sha(args.output.with_suffix('.calls.jsonl.gz')),
        warm=warm,timed_calls=len(records),conditions=summaries(records),all_same_backend_diagnostic_input_result_bits_exact=True,
        all_logical_counts_exact=True,source_inputs_binaries_frozen=True,whole_PF_FGO_speed_established=False,default_changed=False),indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
