"""Compare DD bit behavior when sigma squared or geometry over/underflows."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from bench_pf_dd_reference_compare import _digest, _load_extension, _snapshot


def validate(native):
    records = {}
    state = native.pf_device_create(257)
    try:
        for position in (0., 1e200, np.nan):
            states = np.zeros((257, 16))
            states[:, :3] = position
            satellite = np.zeros((12, 3))
            scalar = np.zeros(12)
            for sigma in (1e-200, 1e-100, .05, 1e100, 1e200):
                for wavelength in (.19, np.finfo(float).tiny, np.nextafter(0., 1.), -0.):
                    for gate in (0., .25):
                        for kind in ('PR', 'CP', 'joint'):
                            native.pf_device_set_particle_states(state, states)
                            native.pf_device_set_log_weights(state, np.ones(257))
                            args = [state, satellite, satellite]
                            waves = np.full(12, wavelength)
                            if kind == 'joint':
                                native.pf_device_weight_dd_joint(*args, scalar, scalar, scalar, scalar,
                                    np.ones(12), waves, 12, sigma, sigma, gate, gate)
                            else:
                                args += [scalar, scalar, scalar, np.ones(12)]
                                if kind == 'CP':
                                    args.append(waves)
                                op = native.pf_device_weight_dd_pseudorange if kind == 'PR' else native.pf_device_weight_dd_carrier_afv
                                op(*args, 12, sigma, gate)
                            actual, weights = _snapshot(native, state)
                            np.testing.assert_array_equal(actual.view(np.uint64), states.view(np.uint64))
                            records[f'position{position}/sigma{sigma}/lambda{wavelength}/gate{gate}/{kind}'] = _digest(actual, weights)
    finally:
        native.pf_device_destroy(state)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('extension', 'baseline', 'candidate', 'output'):
        p.add_argument('--'+name, type=Path)
    args = p.parse_args()
    if args.extension:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'python'))
        print(json.dumps(validate(_load_extension(args.extension.resolve()))))
        return
    records = {}
    for label, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        records[label] = json.loads(subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
            '--extension', str(path.resolve())], text=True, encoding='utf8'))
    mismatches = {k: {label: values[k] for label, values in records.items()}
        for k in records['baseline'] if records['baseline'][k] != records['candidate'][k]}
    report = dict(cases=len(records['baseline']), all_outputs_bit_identical=not mismatches,
        extension_sha256={label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in [('baseline', args.baseline), ('candidate', args.candidate)]},
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_by_case=records['baseline'], candidate_by_case=records['candidate'], mismatches=mismatches)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(len(records['baseline']), 'cases;', len(mismatches), 'mismatches')
    assert not mismatches


if __name__ == '__main__':
    main()
