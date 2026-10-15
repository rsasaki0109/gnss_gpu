"""Bit-check both DD paths on small geometry where joint/split FMA differ."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from bench_pf_coalesced_compare import _load_extension
from bench_pf_covariance_compare import _digest


def _validate(native):
    records = {}
    states = np.random.default_rng(33).normal(size=(33, 16))
    states[:, :4] = 0.
    states[:, :3] = np.random.default_rng(31).normal(0., .1, (33, 3))
    for pairs in (1, 2, 12, 257):
        rng = np.random.default_rng(pairs)
        sat = rng.uniform(1., 100., (pairs, 3))
        ranges = np.linalg.norm(sat, axis=1)
        data = SimpleNamespace(
            sat=sat, ref=np.zeros_like(sat), br=np.zeros(pairs),
            pr=ranges + rng.normal(0., .1, pairs),
            cp=ranges / .19 + np.arange(pairs) * 7 + rng.normal(0., .01, pairs),
            weights=rng.uniform(.1, 1., pairs), wavelengths=np.resize([.19, .24, .25], pairs))
        for mode in ('plain', 'gate', 'huber', 'huber_gate'):
            gate_pr, gate_cp = (10., .25) if 'gate' in mode else (0., 0.)
            huber = 'huber' in mode
            state = native.pf_device_create(33)
            try:
                def carrier():
                    native.pf_device_weight_dd_carrier_afv(
                        state, data.sat, data.ref, data.cp, data.br, data.br,
                        data.weights, data.wavelengths, pairs, .05, gate_cp, huber, 1.5)
                native.pf_device_set_particle_states(state, states)
                carrier()
                cp = np.empty(33)
                native.pf_device_get_log_weights(state, cp)
                native.pf_device_set_particle_states(state, states)
                native.pf_device_weight_dd_pseudorange(
                    state, data.sat, data.ref, data.pr, data.br, data.br,
                    data.weights, pairs, .75, gate_pr, huber, 1.5)
                carrier()
                split = np.empty(33)
                native.pf_device_get_log_weights(state, split)
                native.pf_device_set_particle_states(state, states)
                native.pf_device_weight_dd_joint(
                    state, data.sat, data.ref, data.pr, data.cp, data.br, data.br,
                    data.weights, data.wavelengths, pairs, .75, .05, gate_pr, gate_cp, huber, 1.5, 1.5)
                joint = np.empty(33)
                native.pf_device_get_log_weights(state, joint)
                actual = native.pf_device_get_particle_states(state)
                np.testing.assert_array_equal(actual.view(np.uint64), states.view(np.uint64))
                # This difference is recorded separately from before/after
                # equality: the baseline already has joint/split FMA rounding.
                different = int(np.count_nonzero(joint.view(np.uint64) != split.view(np.uint64)))
                records[f'dd{pairs}/{mode}'] = dict(
                    input_sha256=_digest(states, data.sat, data.ref, data.pr, data.cp,
                                         data.br, data.weights, data.wavelengths,
                                         np.array([gate_pr, gate_cp, huber])),
                    carrier_sha256=_digest(cp), split_sha256=_digest(split), joint_sha256=_digest(joint),
                    states_sha256=_digest(actual), joint_split_different_particles=different)
            finally:
                native.pf_device_destroy(state)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--extension', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.extension:
        print(json.dumps(_validate(_load_extension(args.extension.resolve()))))
        return
    if not (args.baseline and args.candidate and args.output):
        parser.error('baseline, candidate and output required')
    results = {}
    for label, path in (('baseline', args.baseline), ('candidate', args.candidate)):
        worker = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--extension', str(path.resolve())],
                                capture_output=True, text=True, encoding='utf-8')
        if worker.returncode:
            raise RuntimeError(worker.stderr)
        results[label] = json.loads(worker.stdout)
    assert results['baseline'] == results['candidate'], 'DD path before/after bit mismatch'
    root = Path(__file__).resolve().parents[1]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    reference = root / 'benchmarks/pf_student_gpu_20261012.json'
    assert sha(args.baseline) == json.loads(reference.read_text(encoding='utf-8'))['extension_sha256']['candidate']
    helpers = ('benchmarks/bench_pf_dd_wavelength_gate_check.py', 'benchmarks/bench_pf_coalesced_compare.py',
               'benchmarks/bench_pf_covariance_compare.py')
    report = dict(schema_version=1, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  method='actual separate sequential production workers; no timing',
                  cases=len(results['baseline']), all_outputs_bit_identical=True,
                  extension_sha256=dict(baseline=sha(args.baseline), candidate=sha(args.candidate)),
                  benchmark_sha256={name: sha(root / name) for name in helpers},
                  baseline_by_case=results['baseline'], candidate_by_case=results['candidate'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('All 16 cases match before/after for CP, joint, split and complete states; preexisting joint/split differences retained.')


if __name__ == '__main__':
    main()
