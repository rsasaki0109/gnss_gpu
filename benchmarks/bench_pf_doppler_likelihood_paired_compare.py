"""Alternate original/candidate calls, with cached inputs and round snapshots."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from bench_pf_doppler_likelihood_compare import ROOT, Worker, _sha


def compare(args):
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    assert reference['validation']['all_outputs_bit_identical']
    assert reference['validation']['cases'] == 25888
    for name, digest in reference['source_sha256']['candidate'].items():
        assert _sha(ROOT/name) == digest, name
    for name, digest in reference['source_sha256']['baseline'].items():
        assert _sha(args.baseline_dir/Path(name).name) == digest, name
    assert _sha(args.baseline) == reference['extension_sha256']['baseline']
    assert _sha(args.candidate) == reference['extension_sha256']['candidate']
    for name, digest in reference['benchmark_sha256'].items():
        assert _sha(ROOT/name) == digest, name
    workers, measurements = {}, {}
    try:
        for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
            workers[label] = Worker(path)
        for stage in args.stages:
            key = f'N{args.size}/{stage}'
            pairs, snapshots = [], []
            medians = {label: [] for label in workers}
            for round_index in range(args.rounds):
                times = {label: [] for label in workers}
                for sample in range(args.repeats):
                    order = ('baseline', 'candidate') if (round_index+sample) % 2 == 0 else ('candidate', 'baseline')
                    runs = {}
                    for label in order:
                        runs[label] = workers[label].request(mode='measure', size=args.size,
                            stage=stage, repeats=1, snapshot=False)
                        times[label].append(runs[label]['samples_ms'][0])
                    for field in ('input_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                        assert runs['baseline'][field] == runs['candidate'][field], (key, field)
                    pairs.append(dict(round=round_index, sample=sample, order=order, runs=runs))
                outputs = {label: worker.request(mode='snapshot', size=args.size, stage=stage)
                    for label, worker in workers.items()}
                assert outputs['baseline'] == outputs['candidate'], key
                snapshots.append(dict(round=round_index, runs=outputs))
                for label in workers:
                    medians[label].append(float(np.median(times[label])))
                print(f'{key} round {round_index+1}: '
                    f'{medians["baseline"][-1]:.4f}->{medians["candidate"][-1]:.4f}ms', flush=True)
            old, new = [float(np.median(medians[label])) for label in workers]
            measurements[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                improving_paired_rounds=sum(b < a for a, b in zip(medians['baseline'], medians['candidate'])),
                round_medians_ms=medians, pairs=pairs, round_snapshots=snapshots)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = ['benchmarks/bench_pf_doppler_likelihood_paired_compare.py', *reference['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=reference['source_sha256'], extension_sha256=reference['extension_sha256'],
        benchmark_sha256={name: _sha(ROOT/name) for name in helpers},
        validation_reference=dict(path=str(args.reference.relative_to(ROOT)).replace('\\', '/'), sha256=_sha(args.reference)),
        validation_reused_only_for_identical_source_binary_and_helpers=True,
        method='Separate persistent workers; sequential alternating original/candidate order for each sample; nonexclusive host.',
        scope='Initialization/restoration/IPC/hash/input setup excluded; three warmups per request; every timed log-weight and return hashed; full16-state/weights/return snapshot after each paired round.',
        rounds=args.rounds, repeats=args.repeats, warmup_calls_per_request=3, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'baseline-dir', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--size', type=int, default=1000000)
    parser.add_argument('--stages', nargs='+', default=['resample_B60_imported', 'native_PR_KF_step',
        'public_PR_KF_step', 'PR_control', 'resample_B60_common'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if min(args.size, args.rounds, args.repeats) < 1:
        parser.error('positive size, rounds and repeats required')
    args.reference = args.reference.resolve()
    args.output.write_text(json.dumps(compare(args), indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
