"""Check saved validation identities/counts and operational API parity."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numba
import numpy as np

from experiments.run_cpu_batch_receiver import render
from experiments.run_cpu_other_prn_nulls import scene
from experiments.run_cpu_receiver_stress import NOISE_DB
from experiments.run_cpu_noncoherent_sensitivity import score_detection
from experiments.validate_cpu_multiseed_acquisition import summarize_signals
from gnss_gpu.bit_edge_acquisition_cpu import calibrate_ca_bit_edge
from gnss_gpu.weak_acquisition_cpu import load_weak_ca_experiment, detect_weak_ca


def audit(directory):
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    frozen = json.loads((directory/'frozen_thresholds.json').read_text())
    gate = load_weak_ca_experiment(directory)
    root = Path(__file__).resolve().parents[1]

    def check(condition, message):
        if not condition:
            raise ValueError(message)

    for name, digest in protocol['source_sha256'].items():
        check(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, 'source changed: '+name)
    streams = {name: [json.loads(line) for line in (directory/(name+'.jsonl')).open()]
               for name in ('calibration', 'evaluation', 'signals')}
    for name in ('calibration', 'evaluation'):
        n = protocol[name+'_records_per_profile']
        check(Counter((r['profile'], r['record']) for r in streams[name]) ==
              Counter((p, i) for p in protocol['profiles'] for i in range(n)), name+' record identities')
    hashes = {name: {r['audit']['raw_iq_sha256'] for r in records} for name, records in streams.items()}
    check(not (hashes['calibration'] & hashes['evaluation'] or hashes['calibration'] & hashes['signals']
               or hashes['evaluation'] & hashes['signals']), 'split IQ overlap')
    for method in frozen:
        scores = {p: [r['candidates'][method]['statistic'] for r in streams['calibration'] if r['profile'] == p]
                  for p in protocol['profiles']}
        recalculated = calibrate_ca_bit_edge(scores, gate.detection.config, methods_in_family=2)
        check(json.loads(json.dumps(asdict(recalculated))) == frozen[method], 'frozen gate differs')
        for p in protocol['profiles']:
            records = [r for r in streams['evaluation'] if r['profile'] == p]
            flags = [r['candidates'][method]['statistic'] > frozen[method]['score'] for r in records]
            check(flags == [r['accepted'][method] for r in records], 'null acceptance differs')
            check(sum(flags) == report['null_results'][p][method]['accepted'], 'null count differs')
    check(summarize_signals(streams['signals']) == json.loads((directory/'aggregate.json').read_text()),
          'signal aggregate differs')
    expected = Counter((p, cn0, doppler, i) for p in protocol['profiles'] for cn0 in (37., 41.)
                       for doppler in (170., 700.) for i in range(protocol['signal_replicas']))
    check(Counter((r['profile'], r['nominal_primary_cn0_db_hz'], r['primary_doppler_hz'], r['replica'])
                  for r in streams['signals']) == expected, 'signal record identities')
    parity = []
    for record in streams['signals']:
        for method, value in record['methods'].items():
            check(value['accepted'] == (value['statistic'] > frozen[method]['score']), 'signal acceptance differs')
            check(value['passes_declared_criterion'] == (value['accepted'] and value['carrier_error_hz'] <= 250.
                  and value['initial_phase_lag_error_samples'] <= 2.), 'coarse criterion differs')
        if record['profile'] != 'near_far_20db' or record['nominal_primary_cn0_db_hz'] != 37.:
            continue
        rows, starts, seeds, nominal = scene(record['profile'], protocol['noise_domains']['signal'], record['replica'],
            target_cn0=37., target_doppler=record['primary_doppler_hz'])
        iq = render('batch', rows, starts, seeds, NOISE_DB)
        check(hashlib.sha256(iq.tobytes()).hexdigest() == record['audit']['raw_iq_sha256'], 'regenerated IQ differs')
        result = detect_weak_ca(iq, gate)
        check(score_detection(result.detection, record['primary_doppler_hz'], nominal) == record['methods']['multiseed_20'],
              'operational detection differs')
        check(hashlib.sha256(result.cancellation.iq.tobytes()).hexdigest() ==
              record['audit']['methods']['multiseed_20']['residual_iq_sha256'], 'operational residual differs')
        parity.append(dict(replica=record['replica'], doppler_hz=record['primary_doppler_hz']))
    result = dict(record_counts={name: len(records) for name, records in streams.items()},
                  disjoint_split_iq_hashes=True, frozen_gates_reproduced=True, acceptance_flags_reproduced=True,
                  primary_api_parity_records=len(parity), primary_records=parity,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                               numba_threads=numba.get_num_threads()))
    result['paired_primary_outcomes'] = []
    for doppler in (170., 700.):
        selected = [r for r in streams['signals'] if r['profile'] == 'near_far_20db'
                    and r['nominal_primary_cn0_db_hz'] == 37. and r['primary_doppler_hz'] == doppler]
        outcomes = Counter((r['methods']['single_seed_20']['passes_declared_criterion'],
                            r['methods']['multiseed_20']['passes_declared_criterion']) for r in selected)
        result['paired_primary_outcomes'].append(dict(doppler_hz=doppler, both_pass=outcomes[True, True],
            only_single_pass=outcomes[True, False], only_multiseed_pass=outcomes[False, True], both_fail=outcomes[False, False]))
    (directory/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('output/cpu-multiseed-acquisition'))
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
