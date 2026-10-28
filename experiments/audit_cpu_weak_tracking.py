"""Verify sequence identities/summaries and replay replica zero of every case."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_batch_receiver import render
from experiments.run_cpu_receiver_stress import NOISE_DB
from experiments.run_cpu_weak_tracking import continuous_scene, receive, summarize
from gnss_gpu.weak_acquisition_cpu import load_weak_ca_experiment


def audit(directory, calibration):
    directory, calibration = Path(directory), Path(calibration)
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((directory/'protocol.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    records = [json.loads(line) for line in (directory/'sequences.jsonl').open()]
    def check(condition, message):
        if not condition:
            raise ValueError(message)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    check(digest(directory/'protocol.json') == report['protocol_sha256'], 'protocol hash changed')
    for name, value in protocol['source_sha256'].items():
        check(digest(root/name) == value, 'source changed: '+name)
    for name, value in protocol['calibration_artifact_sha256'].items():
        check(digest(Path(name)) == value, 'calibration artifact changed')
    identities = Counter((r['target'], r['profile'], r['doppler_hz'], r['outage'], r['replica']) for r in records)
    expected = Counter((g['target'], g['profile'], g['doppler_hz'], g['outage'], i)
                       for g in report['groups'] for i in range(protocol['replicas']))
    check(identities == expected, 'missing/duplicate sequence identities')
    check(len(records) == report['sequence_count'], 'sequence total differs')
    check(sum(len(r['trace']) for r in records) == report['window_count'], 'window total differs')
    gate = load_weak_ca_experiment(calibration)
    replayed = 0
    for record in records:
        check([r['relative_start_ms'] for r in record['trace']] == list(range(0, 240, 20)), 'window gap')
        check([r['start_sample'] for r in record['trace']] ==
              [record['start_sample']+i*gate.detection.config.periods*2046 for i in range(12)], 'sample timestamp gap')
        check(summarize(record['trace'], target=record['target'], outage=record['outage']) == record['summary'],
              'sequence summary differs')
        if record['replica'] != 0:
            continue
        domain = protocol['signal_domain'] if record['target'] else protocol['null_domain']
        rows, starts, seeds = continuous_scene(record['profile'], domain, 0, target=record['target'],
                                               doppler=record['doppler_hz'], outage=record['outage'])
        iq = render('batch', rows, starts, seeds, NOISE_DB)
        check(hashlib.sha256(iq.tobytes()).hexdigest() == record['raw_iq_sha256'], 'replayed IQ differs')
        check(receive(iq, starts, gate, target=record['target'], doppler=record['doppler_hz'],
                      outage=record['outage']) == record['trace'], 'replayed trace differs')
        replayed += 1
    for group in report['groups']:
        selected = [r['summary'] for r in records if all(r[k] == group[k] for k in ('target', 'profile', 'doppler_hz', 'outage'))]
        check(len(selected) == group['replicas'], 'group denominator differs')
        for key in ('present_windows', 'used_present_windows', 'absent_windows', 'used_absent_windows', 'accepted_absent_windows'):
            check(sum(r[key] for r in selected) == group[key], 'group total differs: '+key)
        check(sum(r['any_published_state'] for r in selected) == group['published_sequences'], 'published count differs')
        check(sum(r['reacquisition_available_after_return_ms'] is not None for r in selected) == group['reacquired_sequences'],
              'reacquisition denominator differs')
    result = dict(sequence_identities_verified=len(records), window_outcomes_verified=report['window_count'],
                  exact_replayed_sequences=replayed, all_summaries_reproduced=True, source_and_calibration_hashes_verified=True)
    (directory/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('output/cpu-weak-tracking'))
    parser.add_argument('--calibration', type=Path, default=Path('output/cpu-multiseed-acquisition'))
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.calibration), indent=2))
