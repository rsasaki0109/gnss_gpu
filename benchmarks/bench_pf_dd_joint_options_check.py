"""Check independent joint PR/CP gate and Huber options on both actual PF builds."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from bench_pf_dd_reference_compare import _digest, _inputs, _load_extension, _observations, _scenario, _snapshot


def _validate(extension):
    native = _load_extension(extension.resolve())
    records = {}
    for count in (1, 31, 257):
        rng = np.random.default_rng(count + 79)
        states = rng.normal(size=(count, 16))
        states[:, :3] += _scenario()[0]
        prior = rng.normal(size=count)
        state = native.pf_device_create(count)
        try:
            for pairs in (3, 12, 33, 511):
                obs = _observations(pairs)
                for gate_pr, gate_cp in ((0., 0.), (10., 0.), (0., .25), (10., .25)):
                    for enabled, k_pr, k_cp in ((False, 1.5, 1.5), (True, 1.5, 1.5), (True, 1.5, 0.),
                                                (True, 0., 1.5), (True, 0., 0.), (True, .75, 3.)):
                        native.pf_device_set_particle_states(state, states)
                        native.pf_device_set_log_weights(state, prior)
                        native.pf_device_weight_dd_joint(
                            state, obs.sat_ecef_k, obs.sat_ecef_ref, obs.dd_pseudorange_m, obs.dd_carrier_cycles,
                            obs.base_range_k, obs.base_range_ref, obs.dd_weights, obs.wavelengths_m,
                            pairs, .75, .05, gate_pr, gate_cp, enabled, k_pr, k_cp)
                        actual, weights = _snapshot(native, state)
                        np.testing.assert_array_equal(actual.view(np.uint64), states.view(np.uint64))
                        records[f'N{count}/dd{pairs}/gates{gate_pr}_{gate_cp}/huber{enabled}_{k_pr}_{k_cp}'] = dict(
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
    helpers = ['benchmarks/bench_pf_dd_joint_options_check.py', 'benchmarks/bench_pf_dd_reference_compare.py',
               'benchmarks/bench_pf_coalesced_compare.py', 'benchmarks/bench_pf_covariance_compare.py']
    report = dict(schema_version=1, method='separate actual extension processes; GPU sequential; exact uint64 output digests',
                  cases=len(records['baseline']), all_outputs_bit_identical=identical,
                  extension_sha256=dict(baseline=sha(args.baseline), candidate=sha(args.candidate)),
                  benchmark_sha256={name: sha(root / name) for name in helpers},
                  baseline_by_case=records['baseline'], candidate_by_case=records['candidate'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    assert identical, 'Independent joint-option precision mismatch; inspect raw records'
    print(f"{report['cases']} independent joint-option cases are bit-identical")


if __name__ == '__main__':
    main()
