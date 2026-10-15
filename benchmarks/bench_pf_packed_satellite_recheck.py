"""Reverse worker creation and alternate each whole PF measurement call."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from bench_pf_packed_satellite_compare import _Worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'candidate', 'baseline-dir', 'reference', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    import hashlib
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    reference = json.loads(args.reference.read_text(encoding='utf8'))
    for path, digest in reference['source_sha256']['baseline'].items():
        assert sha(args.baseline_dir/Path(path).name) == digest, path
    for path, digest in reference['source_sha256']['candidate'].items():
        assert sha(root/path) == digest, path
    assert sha(args.baseline) == reference['extension_sha256']['baseline']
    assert sha(args.candidate) == reference['extension_sha256']['candidate']
    workers = {}
    summary, runs = {}, {}
    try:
        for label, path in (('candidate', args.candidate), ('baseline', args.baseline)):
            workers[label] = _Worker(path)
        for stage in ('PR_gaussian_public_step', 'PR_student_public_step', 'GMM_public_step',
                      'MUPF_public_step', 'mixed_joint_public_gate_step', 'resample_control'):
            data = {label:[] for label in workers}
            for round_index in range(5):
                batch = {label:[] for label in workers}
                for repetition in range(20):
                    order = ('candidate', 'baseline') if (round_index+repetition)%2 == 0 else ('baseline', 'candidate')
                    for label in order:
                        batch[label].append(workers[label].request(mode='measure', size=2048, stage=stage, repeats=1))
                for label in workers:
                    data[label].append(batch[label])
            for field in ('output_sha256', 'log_weights_sha256', 'returned_output_sha256'):
                assert len({r[field] for rounds in data.values() for batch in rounds for r in batch}) == 1, (stage, field)
            medians = {label:[float(np.median([r['median_ms'] for r in batch])) for batch in rounds]
                       for label, rounds in data.items()}
            old, new = [float(np.median(medians[label])) for label in ('baseline', 'candidate')]
            summary[stage] = dict(baseline_ms=old, candidate_ms=new, speedup=old/new,
                                 round_medians_ms=medians, output_sha256=data['baseline'][0][0]['output_sha256'])
            runs[stage] = data
            print(f'{stage}: {old:.4f}->{new:.4f}ms ({old/new:.3f}x)', flush=True)
    finally:
        for worker in workers.values():
            worker.close()
    helpers = dict(reference['benchmark_sha256'])
    for path in helpers:
        assert sha(root/path) == helpers[path], path
    helpers['benchmarks/bench_pf_packed_satellite_recheck.py'] = sha(Path(__file__).resolve())
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        reference=dict(path=str(args.reference.relative_to(root)) if args.reference.is_absolute() else str(args.reference), sha256=sha(args.reference)),
        method='separate persistent production workers created candidate first; each profile has three warmup calls and one timed call; alternate each individual profile and round; initialization/export/destroy/IPC excluded; nonexclusive host',
        rounds=5, repeats=20, size=2048, source_sha256=reference['source_sha256'], extension_sha256=reference['extension_sha256'],
        benchmark_sha256=helpers, summary=summary, runs=runs)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
