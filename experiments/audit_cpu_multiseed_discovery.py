"""Reproduce a development-only nuisance-discovery failure; no held-out claim."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_batch_receiver import FS, render
from experiments.run_cpu_receiver_stress import NOISE_DB
from experiments.run_cpu_other_prn_nulls import scene
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, acquire_ca_bit_edge
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig, cancel_other_ca
from gnss_gpu.ca_local_delay_cpu import _refine_local_delays
from gnss_gpu.ca_multiseed_cpu import cancel_other_ca_multiseed


def audit():
    historical = Path('output/cpu-weak-acquisition-refined')
    record = next(json.loads(line) for line in (historical/'calibration.jsonl').open()
                  if (json.loads(line)['profile'], json.loads(line)['record']) == ('near_far_20db', 281))
    rows, starts, seeds, _ = scene('near_far_20db', 0x57524341, 281)
    iq = render('batch', rows, starts, seeds, NOISE_DB)
    iq_hash = hashlib.sha256(iq.tobytes()).hexdigest()
    if iq_hash != record['audit']['raw_iq_sha256']:
        raise ValueError('development IQ differs from preserved record')
    config = CaCancellationConfig(FS)
    search = CaBitEdgeSearch(CaSearch(FS, tuple(float(x) for x in range(-1000, 1001, 250))))
    result = dict(scope='Development diagnostic selected before fresh validation; not held-out evidence.',
                  source_record=dict(directory=str(historical), split='calibration', profile='near_far_20db', record=281),
                  raw_iq_sha256=iq_hash, methods={})
    for name, processor in (('single_seed_20', cancel_other_ca), ('multiseed_20', cancel_other_ca_multiseed)):
        discovery = processor(iq, config)
        refined = _refine_local_delays(iq, config, discovery)
        candidate = acquire_ca_bit_edge(refined.iq, search)
        result['methods'][name] = dict(components=[asdict(value) for value in discovery.components],
                                      candidate=asdict(candidate), removed_energy_fraction=refined.removed_energy_fraction)
    if result['methods']['single_seed_20']['candidate'] != record['candidates']['bit_edge_20']:
        raise ValueError('historical pipeline no longer reproduces development score')
    output = Path('output/cpu-multiseed-development')
    output.mkdir(parents=True, exist_ok=True)
    (output/'diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
