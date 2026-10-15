"""Repeat full PF resampling timings with a previously validated frozen build."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench_pf_shared_proposals_compare import _stages, _Worker

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(args):
    ref = json.loads(args.validated_report.read_text(encoding='utf8'))
    for name, digest in ref['source_sha256']['candidate'].items():
        assert _sha(ROOT/name) == digest, name
    for name, digest in ref['source_sha256']['baseline'].items():
        assert _sha(args.baseline_dir/Path(name).name) == digest, name
    assert _sha(args.baseline) == ref['extension_sha256']['baseline']
    assert _sha(args.candidate) == ref['extension_sha256']['candidate']
    for name, digest in ref['benchmark_sha256'].items():
        assert _sha(ROOT/name) == digest, name
    assert ref['validation']['all_outputs_bit_identical']
    assert ref['validation']['baseline_by_case'] == ref['validation']['candidate_by_case']
    workers, summary, runs = {}, {}, {}
    try:
        for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
            workers[label] = _Worker(path)
        for count in args.sizes:
            for stage in args.stages or _stages():
                key = f'N{count}/{stage}'
                data = {label: [] for label in workers}
                for round_index in range(args.rounds):
                    order = ('baseline', 'candidate') if round_index % 2 == 0 else ('candidate', 'baseline')
                    for label in order:
                        data[label].append(workers[label].request(mode='measure', size=count, stage=stage, repeats=args.repeats))
                for field in ('output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                    assert len({r[field] for values in data.values() for r in values}) == 1, (key, field)
                medians = {label: [r['median_ms'] for r in values] for label, values in data.items()}
                old, new = [float(np.median(medians[label])) for label in workers]
                summary[key] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                    improving_paired_rounds=sum(b < a for a, b in zip(medians['baseline'], medians['candidate'])),
                    round_medians_ms=medians, output_sha256=data['baseline'][0]['output_sha256'])
                runs[key] = data
                print(f'{key}: {old:.4f}->{new:.4f}ms ({old/new:.3f}x); {summary[key]["improving_paired_rounds"]}/{args.rounds}', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = ['benchmarks/bench_pf_shared_proposals_recheck.py', *ref['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=ref['source_sha256'], extension_sha256=ref['extension_sha256'],
        benchmark_sha256={p: _sha(ROOT/p) for p in helpers}, build_configuration=ref['build_configuration'],
        baseline=ref['baseline'], baseline_reference=ref['baseline_reference'],
        method=ref['method'], scope=ref['scope'], warmup_calls_per_stage=3,
        validation=ref['validation'],
        validation_reference=dict(path=str(args.validated_report), sha256=_sha(args.validated_report),
            unchanged_exact_source_binary_and_helper_hashes=True),
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, summary=summary, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'baseline-dir', 'validated-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[2048, 100000, 1000000])
    parser.add_argument('--stages', nargs='+')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if min(*args.sizes, args.rounds, args.repeats) < 1:
        parser.error('positive sizes, rounds and repeats required')
    report = compare(args)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
