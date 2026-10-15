"""Compare packed common-covariance gathers against a frozen original build.

Reuse original-build fixture records only with exact original source, binary,
validator and dependency hashes. Run that same complete validator on the new
candidate. Separate persistent workers execute measured stages sequentially.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bench_pf_shared_proposals_compare import _Worker

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stages():
    return ['resample_B4_common', 'resample_B60_common', 'resample_B240_common',
            'resample_B60_imported', 'native_PR_step', 'public_PR_step', 'public_PR_KF_step',
            'PR_control', 'resample_B0_control', 'resample_B257_imported']


def compare(args):
    original = json.loads(args.baseline_validation_report.read_text(encoding='utf8'))
    sources = dict(baseline={p: _sha(args.baseline_dir/Path(p).name) for p in original['source_sha256']['baseline']},
                   candidate={p: _sha(ROOT/p) for p in original['source_sha256']['baseline']})
    binaries = dict(baseline=_sha(args.baseline), candidate=_sha(args.candidate))
    assert sources['baseline'] == original['source_sha256']['baseline']
    assert binaries['baseline'] == original['extension_sha256']['baseline']
    for name, digest in original['benchmark_sha256'].items():
        assert _sha(ROOT/name) == digest, name
    assert original['validation']['all_outputs_bit_identical']
    workers, summary, runs = {}, {}, {}
    try:
        for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
            workers[label] = _Worker(path)
        baseline = original['validation']['baseline_by_case']
        candidate = workers['candidate'].request(mode='validate')
        if candidate != baseline:
            mismatches = {key: dict(baseline=baseline.get(key), candidate=candidate.get(key))
                for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key)}
            args.output.write_text(json.dumps(dict(source_sha256=sources, extension_sha256=binaries, mismatches=mismatches), indent=2)+'\n', encoding='utf8')
        assert baseline == candidate, 'Input/state/log-weight/trajectory bit mismatch'
        print(f'{len(baseline)} complete fixture records match the frozen original build', flush=True)
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
    helpers = ['benchmarks/bench_pf_packed_gather_compare.py', *original['benchmark_sha256']]
    return dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources, extension_sha256=binaries,
        benchmark_sha256={p: _sha(ROOT/p) for p in helpers}, build_configuration=original['build_configuration'],
        baseline='retained wide KF build; original SoA gather and positive-zero reset',
        baseline_reference=original['baseline_reference'],
        method='Persistent separate workers; sequential stage batches; alternating order per round; nonexclusive host.',
        scope=original['scope'], warmup_calls_per_stage=3,
        validation=dict(cases=len(baseline), all_outputs_bit_identical=True, baseline_by_case=baseline, candidate_by_case=candidate),
        baseline_validation_reference=dict(path=str(args.baseline_validation_report), sha256=_sha(args.baseline_validation_report),
            unchanged_exact_original_source_binary_and_validator_dependency_hashes=True),
        rounds=args.rounds, repeats=args.repeats, sizes=args.sizes, summary=summary, runs=runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'baseline-dir', 'baseline-validation-report', 'output'):
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
