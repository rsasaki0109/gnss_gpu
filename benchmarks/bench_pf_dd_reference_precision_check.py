"""Check additional particle-coordinate extremes on the two actual PF builds."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from bench_pf_dd_reference_compare import _cp, _digest, _inputs, _load_extension, _pr, _reference_observations, _snapshot


def _validate(extension):
    native = _load_extension(extension.resolve())
    count = 257
    state = native.pf_device_create(count)
    records = {}
    try:
        for variant in ('finite', 'origin', 'signed_zero', 'nonfinite', 'huge', 'subnormal'):
            states = np.random.default_rng(78).normal(size=(count, 16))
            if variant == 'origin':
                states[:, :3] = 0.
            elif variant == 'signed_zero':
                states[:, :3] = np.resize([0., -0.], (count, 3))
            elif variant == 'nonfinite':
                states[:, :3] = np.resize([np.inf, -np.inf, np.nan, 0., 1., -1.], (count, 3))
            elif variant == 'huge':
                states[:, :3] *= 1e150
            elif variant == 'subnormal':
                states[:, :3] *= np.nextafter(0., 1.)
            else:
                states[:, :3] += [-3957199., 3310205., 3737911.]
            prior = np.random.default_rng(79).normal(size=count)
            for pairs in (12, 33, 512):
                obs = _reference_observations(pairs, 'common')
                for mode, gate, huber in (('plain', 0., False), ('gate', .25, False),
                                          ('huber', 0., True), ('huber_gate', .25, True)):
                    for label, operation in (('PR', _pr), ('CP', _cp)):
                        native.pf_device_set_particle_states(state, states)
                        native.pf_device_set_log_weights(state, prior)
                        operation(native, state, obs, gate, huber)
                        actual, weights = _snapshot(native, state)
                        np.testing.assert_array_equal(actual.view(np.uint64), states.view(np.uint64))
                        records[f'{variant}/dd{pairs}/{mode}/{label}'] = dict(
                            input_sha256=_digest(states, prior, *_inputs(obs)), output_sha256=_digest(actual, weights))
    finally:
        native.pf_device_destroy(state)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.extension:
        print(json.dumps(_validate(args.extension)), flush=True)
        return
    if not (args.baseline and args.candidate and args.output):
        parser.error('baseline, candidate and output required')
    root = Path(__file__).resolve().parents[1]
    records = {}
    for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--extension', str(path.resolve())],
                                check=True, capture_output=True, text=True, encoding='utf-8')
        records[label] = json.loads(result.stdout)
    identical = records['baseline'] == records['candidate']
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    helpers = ['benchmarks/bench_pf_dd_reference_precision_check.py', 'benchmarks/bench_pf_dd_reference_compare.py',
               'benchmarks/bench_pf_coalesced_compare.py', 'benchmarks/bench_pf_covariance_compare.py']
    report = dict(schema_version=1, method='separate actual extension processes; GPU sequential; exact uint64 output digests',
                  cases=len(records['baseline']), all_outputs_bit_identical=identical,
                  extension_sha256=dict(baseline=sha(args.baseline), candidate=sha(args.candidate)),
                  benchmark_sha256={name: sha(root / name) for name in helpers},
                  baseline_by_case=records['baseline'], candidate_by_case=records['candidate'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    assert identical, 'Particle-coordinate precision mismatch; inspect raw records'
    print(f"{report['cases']} additional particle-coordinate cases are bit-identical")


if __name__ == '__main__':
    main()
