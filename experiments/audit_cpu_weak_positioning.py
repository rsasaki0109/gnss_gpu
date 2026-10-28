"""Recompute frozen per-PRN gates and all saved track-to-position outcomes."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, calibrate_ca_bit_edge
from gnss_gpu.weak_positioning_cpu import position_from_weak_tracks


def audit(directory):
    directory = Path(directory)
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((directory/'protocol.json').read_text())
    report = json.loads((directory/'report.json').read_text())
    frozen = json.loads((directory/'frozen_thresholds.json').read_text())
    def check(condition, message):
        if not condition: raise ValueError(message)
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def normalize(value): return json.loads(json.dumps(value, default=lambda item:item.tolist()))
    check(digest(directory/'protocol.json') == report['protocol_sha256'], 'protocol changed')
    check(digest(directory/'frozen_thresholds.json') == report['frozen_thresholds_sha256'], 'frozen gates changed')
    for path, value in protocol['source_sha256'].items():
        check(digest(root/path) == value, 'source changed: '+path)
    streams = {name:[json.loads(line) for line in (directory/(name+'.jsonl')).open()]
               for name in ('calibration','evaluation','positions')}
    check([r['record'] for r in streams['calibration']] == list(range(599)), 'calibration identities')
    check([r['record'] for r in streams['evaluation']] == list(range(199)), 'evaluation identities')
    check(not ({r['raw_iq_sha256'] for r in streams['calibration']} &
               {r['raw_iq_sha256'] for r in streams['evaluation']}), 'calibration/evaluation overlap')
    for prn, gate in frozen.items():
        cfg = dict(gate['detection']['config']); cfg['search'] = CaSearch(**cfg['search'])
        search = CaBitEdgeSearch(**cfg)
        calibrated = calibrate_ca_bit_edge({'white':[r['scores'][prn] for r in streams['calibration']]}, search, methods_in_family=8)
        check(normalize(asdict(calibrated)) == gate['detection'], 'gate mismatch')
        flags = [r['scores'][prn]>gate['detection']['score'] for r in streams['evaluation']]
        check(flags == [r['accepted'][prn] for r in streams['evaluation']], 'null flags mismatch')
        check(sum(flags) == report['null_accepted_per_prn'][prn], 'null total mismatch')
    def keyed(name): return {int(p):value for p,value in protocol[name].items()}
    identities = []
    for row in streams['positions']:
        identities.append((row['scenario'],row['replica'],row['window']))
        states = {int(p):SimpleNamespace(**state) for p,state in row['states'].items()}
        result = position_from_weak_tracks(states, keyed('satellite_positions_m'), keyed('prior_range_m'),
            keyed('prior_bound_m'), protocol['sampling_freq_hz'], initial_position_m=protocol['prior_position_m'])
        check(normalize(asdict(result)) == row['result'], 'position bridge differs')
        expected = float(np.linalg.norm(result.solution.position_m-protocol['simulated_receiver_m'])) if result.solution else None
        check(expected == row['position_error_m'], 'position error differs')
        if row['window'] == 0:
            check(result.solution is None, 'pending confirmation unexpectedly produced a position')
    expected = [(scenario,replica,window) for scenario in protocol['scenarios']
                for replica in range(protocol['replicas']) for window in range(4)]
    check(identities == expected, 'missing or duplicate position epochs')
    for group in report['summaries']:
        rows = [r for r in streams['positions'] if r['scenario']==group['scenario'] and r['window']>0]
        errors = [r['position_error_m'] for r in rows if r['position_error_m'] is not None]
        check(len(rows)==group['epochs'] and len(errors)==group['solved'], 'availability denominator differs')
        check((float(np.median(errors)) if errors else None)==group['position_error_p50_m'], 'median differs')
        check((float(np.quantile(errors,.95)) if errors else None)==group['position_error_p95_m'], 'percentile differs')
    result = dict(calibration_records=599, independent_null_records=199, gates_recomputed=8,
                  exact_recomputed_position_epochs=len(identities), retained_initial_confirmation_epochs=len(identities)//4,
                  all_failed_epochs_retained=True, source_and_artifact_hashes_verified=True)
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n'); return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('output/cpu-weak-positioning'))
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
