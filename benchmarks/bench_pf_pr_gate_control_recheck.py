"""Repeat unchanged controls with reversed worker creation and round order.

The full comparison supplies exact validation and source/binary/helper hashes.
This diagnostic reuses its frozen production measurement and worker routines.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench_pf_pr_gate_compare import _Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    assert reference['validation']['baseline_by_case'] == reference['validation']['candidate_by_case']
    for name, digest in reference['source_sha256']['candidate'].items():
        assert sha(root/name) == digest, name
    for name, digest in reference['benchmark_sha256'].items():
        assert sha(root/name) == digest, name
    assert sha(args.baseline) == reference['extension_sha256']['baseline']
    assert sha(args.candidate) == reference['extension_sha256']['candidate']
    workers, runs, summary = {}, {}, {}
    try:
        for label, path in (('candidate', args.candidate), ('baseline', args.baseline)):
            workers[label] = _Worker(path)
        for stage in ('Student_control', 'resample_control'):
            data = {label: [] for label in ('baseline', 'candidate')}
            for index in range(5):
                order = ('candidate', 'baseline') if index % 2 == 0 else ('baseline', 'candidate')
                for label in order:
                    data[label].append(workers[label].request(mode='measure', size=1000000, stage=stage, repeats=20))
            for field in ('output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                assert len({r[field] for values in data.values() for r in values}) == 1, (stage, field)
                assert data['baseline'][0][field] == reference['runs'][f'N1000000/{stage}']['baseline'][0][field]
            medians = {label: [r['median_ms'] for r in values] for label, values in data.items()}
            old, new = [float(np.median(medians[label])) for label in ('baseline', 'candidate')]
            summary[stage] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new, round_medians_ms=medians)
            runs[stage] = data
            print(f'{stage}: {old:.4f} -> {new:.4f}ms ({old/new:.3f}x)', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Check large unchanged-control timing shifts; candidate worker created first and starts even rounds; nonexclusive host',
        scope=reference['scope'], rounds=5, repeats=20, size=1000000, warmup_calls_per_stage=3,
        validation_reference=dict(path=args.reference.relative_to(root).as_posix() if args.reference.is_absolute() else args.reference.as_posix(),
                                  sha256=sha(args.reference), comparison_cases=reference['validation']['cases']),
        source_sha256=reference['source_sha256'], extension_sha256=reference['extension_sha256'],
        benchmark_sha256={**reference['benchmark_sha256'], 'benchmarks/bench_pf_pr_gate_control_recheck.py': sha(Path(__file__))},
        summary=summary, runs=runs)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
