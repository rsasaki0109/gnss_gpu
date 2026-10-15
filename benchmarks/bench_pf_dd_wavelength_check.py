"""Repeat selected DD wavelength and unchanged-control stages on the same builds."""

import argparse
import hashlib
import json
from pathlib import Path

import bench_pf_dd_wavelength_compare as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000000])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 1 or args.rounds < 1 or args.repeats < 1:
        parser.error('positive sizes, rounds and repeats required')
    selected = ['CP_dd4', 'CP_dd12', 'joint_dd12', 'joint_gate', 'public_joint_step',
                'Student_control', 'resample_control']
    # Change the benchmark's case selection only. Worker operations and actual
    # production kernels are exactly those used by the primary measurement.
    original = benchmark._stages
    benchmark._stages = lambda: selected
    args.validation_only = False
    try:
        report = benchmark._compare(args)
    finally:
        benchmark._stages = original
    path = Path(__file__).resolve()
    report['benchmark_sha256']['benchmarks/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['selected_stages'] = selected
    report['purpose'] = 'Repeat principal stages and unchanged controls after nonexclusive timing variation; retain both measurements'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
