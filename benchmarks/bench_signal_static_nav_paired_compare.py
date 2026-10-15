"""Confirm static-NAV batch specialization with sequential alternating one-call batches.

Use the frozen input-packing worker. It rebuilds stage inputs outside timing,
warms the active context, then times one complete call and hashes its output.
Alternating workers per sample reduces slow host drift between stage batches.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from bench_signal_static_nav_compare import ROOT, SOURCE_PATHS, Worker, _sha


def compare(args):
    reference_path = 'benchmarks/signal_batch_gpu_20261012.json'
    reference = json.loads((ROOT/reference_path).read_text(encoding='utf8'))
    sources = dict(baseline={p: _sha(args.baseline_dir/Path(p).name) for p in SOURCE_PATHS},
                   candidate={p: _sha(ROOT/p) for p in SOURCE_PATHS})
    binaries = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert sources['baseline'] == reference['source_sha256']['candidate']
    assert binaries['baseline'] == reference['extension_sha256']['candidate']
    workers, measurements = {}, {}
    try:
        workers['baseline'] = Worker(args.baseline, args.baseline_dir)
        workers['candidate'] = Worker(args.candidate)
        validated = json.loads(args.validated_report.read_text(encoding='utf8'))
        assert validated['validation']['all_iq_and_metadata_bits_identical']
        assert validated['source_sha256'] == sources and validated['extension_sha256'] == binaries
        for name, digest in validated['benchmark_sha256'].items():
            assert _sha(ROOT/name) == digest, name
        validation = {label: validated['validation'][label+'_by_case'] for label in workers}
        assert validation['baseline'] == validation['candidate']
        print(f"{len(validation['baseline'])} identical IQ/metadata/rejection cases", flush=True)
        for stage in args.stages:
            rounds = {label: [] for label in workers}
            records = []
            for round_index in range(args.rounds):
                samples = {label: [] for label in workers}
                for repeat in range(args.repeats):
                    order = ('baseline','candidate') if (round_index+repeat) % 2 == 0 else ('candidate','baseline')
                    pair = {}
                    for label in order:
                        value = workers[label].request(command='measure',stage=stage,repeats=1)
                        assert len(value['samples_ms']) == 1 and value['all_timed_outputs_identical']
                        samples[label].append(value['samples_ms'][0])
                        pair[label] = value
                    assert pair['baseline']['input_sha256'] == pair['candidate']['input_sha256']
                    assert pair['baseline']['output_sha256'] == pair['candidate']['output_sha256']
                    records.append(dict(round=round_index,repeat=repeat,order=list(order),runs=pair))
                for label in rounds:
                    rounds[label].append(float(np.median(samples[label])))
            assert len({p['runs']['baseline']['output_sha256'] for p in records}) == 1
            old,new = [float(np.median(rounds[label])) for label in workers]
            improved = sum(b<a for a,b in zip(rounds['baseline'],rounds['candidate']))
            measurements[stage] = dict(baseline_ms=old,candidate_ms=new,speedup=old/new,
                round_medians_ms=rounds,improving_paired_rounds=improved,
                output_sha256=records[0]['runs']['baseline']['output_sha256'],pairs=records)
            print(f'{stage}: {old:.4f}->{new:.4f}ms;{improved}/{args.rounds} improving pairs',flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = ['benchmarks/bench_signal_static_nav_paired_compare.py'] + list(
        validated['benchmark_sha256'])
    return dict(schema_version=1,timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,extension_sha256=binaries,baseline_reference=dict(path=reference_path,sha256=_sha(ROOT/reference_path)),
        validation_reference=dict(path=str(args.validated_report.resolve().relative_to(ROOT)).replace('\\','/'),sha256=_sha(args.validated_report)),
        validation_reused_only_for_identical_source_binary_and_helpers=True,
        benchmark_sha256={p:_sha(ROOT/p) for p in helpers},build_configuration='CUDA12.8 / MSVC Release / compute7.5 / C++20 / no fast math',
        method='persistent separate workers;sequential GPU calls;alternate build per timed sample;nonexclusive host',
        scope='complete native/public epoch group including validation/conversion/H2D/kernels/sync/returned IQ;urban includes metadata/LOS/reflection;stage inputs and active-context warmup4calls/IPC/hashes outside timing',
        rounds=args.rounds,repeats=args.repeats,warmup_calls_per_sample=4,
        validation=dict(cases=len(validation['baseline']),matching_records=True,baseline_by_case=validation['baseline'],candidate_by_case=validation['candidate']),
        measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','candidate','baseline-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validated-report',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--stages',nargs='+',required=True)
    args = parser.parse_args()
    if min(args.rounds,args.repeats) < 1:
        parser.error('positive rounds and repeats required')
    report = compare(args)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    main()
