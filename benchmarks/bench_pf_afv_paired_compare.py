"""Confirm whole PF AFV steps by alternating frozen workers per timed sample."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench_pf_afv_unroll_compare import _Worker

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(args):
    validated = json.loads(args.validated_report.read_text(encoding='utf8'))
    for name, digest in validated['source_sha256']['candidate'].items():
        assert _sha(ROOT/name) == digest, name
    for name, digest in validated['source_sha256']['baseline'].items():
        assert _sha(args.baseline_dir/Path(name).name) == digest, name
    assert _sha(args.baseline) == validated['extension_sha256']['baseline']
    assert _sha(args.candidate) == validated['extension_sha256']['candidate']
    for name, digest in validated['benchmark_sha256'].items():
        assert _sha(ROOT/name) == digest, name
    assert validated['validation']['all_outputs_bit_identical']
    workers, measurements = {}, {}
    try:
        workers['baseline'] = _Worker(args.baseline)
        workers['candidate'] = _Worker(args.candidate)
        for count in args.sizes:
            for stage in args.stages:
                rounds = {label: [] for label in workers}
                records = []
                for round_index in range(args.rounds):
                    samples = {label: [] for label in workers}
                    for repeat in range(args.repeats):
                        order = ('baseline', 'candidate') if (round_index+repeat) % 2 == 0 else ('candidate', 'baseline')
                        pair = {}
                        for label in order:
                            value = workers[label].request(mode='measure', size=count, stage=stage, repeats=1)
                            assert len(value['samples_ms']) == 1 and value['all_timed_log_weights_identical']
                            samples[label].append(value['samples_ms'][0])
                            pair[label] = value
                        for field in ('output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                            assert pair['baseline'][field] == pair['candidate'][field], (stage, field)
                        records.append(dict(round=round_index, repeat=repeat, order=list(order), runs=pair))
                    for label in rounds:
                        rounds[label].append(float(np.median(samples[label])))
                assert len({p['runs']['baseline']['output_sha256'] for p in records}) == 1
                old, new = [float(np.median(rounds[label])) for label in workers]
                improved = sum(b < a for a, b in zip(rounds['baseline'], rounds['candidate']))
                key = f'N{count}/{stage}'
                measurements[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                    round_medians_ms=rounds, improving_paired_rounds=improved,
                    output_sha256=records[0]['runs']['baseline']['output_sha256'], pairs=records)
                print(f'{key}: {old:.4f}->{new:.4f}ms; {improved}/{args.rounds} improving rounds', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = ['benchmarks/bench_pf_afv_paired_compare.py', *validated['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=validated['source_sha256'], extension_sha256=validated['extension_sha256'],
        benchmark_sha256={p: _sha(ROOT/p) for p in helpers}, build_configuration=validated['build_configuration'],
        method='Persistent separate workers; sequential GPU calls; alternate build per timed sample; nonexclusive host.',
        scope=validated['scope'], warmup_calls_per_sample=3,
        validation_reference=dict(path=str(args.validated_report), sha256=_sha(args.validated_report),
            cases=validated['validation']['cases'], unchanged_exact_source_binary_and_helper_hashes=True),
        rounds=args.rounds, repeats=args.repeats, measurements=measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'baseline-dir', 'validated-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000000])
    parser.add_argument('--stages', nargs='+', default=['AFV_sat12', 'MUPF_public_step', 'PR_gaussian_public_step'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if min(*args.sizes, args.rounds, args.repeats) < 1:
        parser.error('positive sizes, rounds and repeats required')
    report = compare(args)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
