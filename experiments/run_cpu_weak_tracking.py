"""Fresh 240 ms continuous-signal following and target-absent sequence audit."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_batch_receiver import COUNT, render
from experiments.run_cpu_other_prn_nulls import PROFILES, scene
from experiments.run_cpu_receiver_stress import NOISE_DB, seed_for
from experiments.run_cpu_receiver_sensitivity import wilson_interval
from gnss_gpu.input_validation import positive_int
from gnss_gpu.signal_sim_cpu import L1_HZ
from gnss_gpu.weak_acquisition_cpu import load_weak_ca_experiment
from gnss_gpu.weak_tracking_cpu import CpuWeakCaTracker

SIGNAL_DOMAIN, NULL_DOMAIN = 0x57545349, 0x57544E55
TRACKER = dict(gain=.25, max_frequency_step_hz=75., max_delay_step_samples=2.,
               confirmation_windows=2, coast_windows=2, intermediate_freq=0.)
BLOCKS, WINDOW = 240, 20


def continuous_scene(profile, domain, replica, *, doppler=170., target=False, outage=False):
    rows, starts, _, _ = scene(profile, domain, replica, target_cn0=37. if target else None,
                               target_doppler=doppler)
    template = rows[0]
    rows = []
    for block in range(BLOCKS):
        channels = [dict(channel) for channel in template]
        for channel in channels:
            if channel['prn'] == 1:
                channel['nav_bits'] = [1 if i % 2 == 0 else -1 for i in range(16)]
                if outage and 80 <= block < 120:
                    channel['amplitude'] = 0.
        rows.append(channels)
    return rows, [starts[0]+i*COUNT for i in range(BLOCKS)], [seed_for(domain, replica, i) for i in range(BLOCKS)]


def receive(iq, starts, gate, *, target, doppler, outage):
    tracker = CpuWeakCaTracker(gate, **TRACKER)
    trace = []
    for begin in range(0, BLOCKS, WINDOW):
        result = tracker.update(iq[begin:begin+WINDOW], start_sample=starts[begin])
        present = target and not (outage and 80 <= begin < 120)
        row = dict(asdict(result), relative_start_ms=begin, available_after_ms=begin+WINDOW,
                   target_present=present)
        # Truth is used only here, after the receiver has returned.
        row['carrier_error_hz'] = abs(result.carrier_hz-doppler) if present and result.measurement_used else None
        nominal = 37.-starts[begin]*(1+doppler/L1_HZ)
        row['lag_error_samples'] = (abs((result.delay_samples-nominal+COUNT/2) % COUNT-COUNT/2)
                                    if present and result.measurement_used else None)
        trace.append(row)
    return trace


def summarize(trace, *, target, outage):
    present = [r for r in trace if r['target_present']]
    absent = [r for r in trace if not r['target_present']]
    initial = next((r['available_after_ms'] for r in trace if r['status'] == 'acquired'), None)
    pre_acquired = target and any(r['measurement_used'] and r['relative_start_ms'] < 80 for r in trace)
    lost = outage and pre_acquired and any(r['status'] == 'lost' and 80 <= r['relative_start_ms'] < 120 for r in trace)
    reacquired = next((r['available_after_ms']-120 for r in trace
                      if r['relative_start_ms'] >= 120 and r['status'] == 'acquired'), None) if lost else None
    errors = [r['carrier_error_hz'] for r in present if r['measurement_used']]
    lags = [r['lag_error_samples'] for r in present if r['measurement_used']]
    return dict(present_windows=len(present), used_present_windows=sum(r['measurement_used'] for r in present),
        absent_windows=len(absent), accepted_absent_windows=sum(r['detection_accepted'] for r in absent),
        used_absent_windows=sum(r['measurement_used'] for r in absent),
        acquired_before_outage=pre_acquired,
        first_acquisition_available_ms=initial, reacquisition_available_after_return_ms=reacquired,
        lost_during_outage=lost,
        any_published_state=any(r['carrier_hz'] is not None for r in trace),
        maximum_used_carrier_error_hz=max(errors) if errors else None,
        maximum_used_lag_error_samples=max(lags) if lags else None,
        states=dict(Counter(r['status'] for r in trace)))


def run(output, calibration, replicas=32):
    replicas = positive_int('replicas', replicas)
    gate = load_weak_ca_experiment(calibration)
    if gate.detection.config.periods != WINDOW:
        raise ValueError('experiment requires the frozen 20 ms gate')
    output, calibration = Path(output), Path(calibration)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output directory must be empty')
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root/'python/gnss_gpu/weak_tracking_cpu.py', root/'python/gnss_gpu/weak_acquisition_cpu.py',
             root/'python/gnss_gpu/signal_sim_cpu.py', root/'experiments/run_cpu_other_prn_nulls.py',
             root/'experiments/run_cpu_batch_receiver.py', root/'experiments/run_cpu_receiver_stress.py',
             root/'experiments/run_cpu_receiver_sensitivity.py']
    files += [root/name for name in json.loads((calibration/'protocol.json').read_text())['source_sha256']]
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {str(path.relative_to(root)): digest(path) for path in files}
    artifacts = {str(path): digest(path) for path in (calibration/'protocol.json', calibration/'frozen_thresholds.json', calibration/'report.json')}
    protocol = dict(replicas=replicas, duration_ms=BLOCKS, window_ms=WINDOW, tracker=TRACKER,
        signal_domain=SIGNAL_DOMAIN, null_domain=NULL_DOMAIN, target_cn0_db_hz=37., dopplers_hz=[170., 700.],
        target_profile='near_far_20db', target_cases=['continuous', 'target_outage_80_120ms'], null_profiles=PROFILES,
        null_policy='Descriptive sequence-level evaluation; existing marginal gate supplies no track false-alarm guarantee.',
        latency_policy='Window estimates refer to the first row but become available after its final row.',
        source_sha256=sources, calibration_artifact_sha256=artifacts,
        scoring_policy='Keep all replicas, including failed initial acquisition. No threshold/settings retuning.')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    records = []
    cases = [(True, 'near_far_20db', doppler, outage) for doppler in (170., 700.) for outage in (False, True)]
    cases += [(False, profile, 170., False) for profile in PROFILES]
    with (output/'sequences.jsonl').open('w') as handle:
        for target, profile, doppler, outage in cases:
            for replica in range(replicas):
                rows, starts, seeds = continuous_scene(profile, SIGNAL_DOMAIN if target else NULL_DOMAIN, replica,
                                                       target=target, doppler=doppler, outage=outage)
                iq = render('batch', rows, starts, seeds, NOISE_DB)
                trace = receive(iq, starts, gate, target=target, doppler=doppler, outage=outage)
                record = dict(target=target, profile=profile, doppler_hz=doppler, outage=outage, replica=replica,
                    raw_iq_sha256=digest_bytes(iq.tobytes()), start_sample=starts[0], trace=trace,
                    summary=summarize(trace, target=target, outage=outage))
                records.append(record)
                handle.write(json.dumps(record)+'\n')
            handle.flush()
            print(f'completed target={target} profile={profile} doppler={doppler} outage={outage}', flush=True)
    groups = []
    for target, profile, doppler, outage in cases:
        subset = [r for r in records if (r['target'], r['profile'], r['doppler_hz'], r['outage']) == (target, profile, doppler, outage)]
        summaries = [r['summary'] for r in subset]
        published = sum(r['any_published_state'] for r in summaries)
        groups.append(dict(target=target, profile=profile, doppler_hz=doppler, outage=outage, replicas=len(subset),
            published_sequences=published, published_sequence_wilson_95=wilson_interval(published, len(subset)),
            acquired_before_80ms=sum(r['acquired_before_outage'] for r in summaries),
            lost_during_outage=sum(r['lost_during_outage'] for r in summaries),
            reacquired_sequences=sum(r['reacquisition_available_after_return_ms'] is not None for r in summaries),
            first_acquisition_ms=[r['first_acquisition_available_ms'] for r in summaries],
            reacquisition_ms=[r['reacquisition_available_after_return_ms'] for r in summaries],
            present_windows=sum(r['present_windows'] for r in summaries), used_present_windows=sum(r['used_present_windows'] for r in summaries),
            absent_windows=sum(r['absent_windows'] for r in summaries), used_absent_windows=sum(r['used_absent_windows'] for r in summaries),
            accepted_absent_windows=sum(r['accepted_absent_windows'] for r in summaries),
            maximum_used_carrier_error_hz=max((r['maximum_used_carrier_error_hz'] for r in summaries if r['maximum_used_carrier_error_hz'] is not None), default=None),
            maximum_used_lag_error_samples=max((r['maximum_used_lag_error_samples'] for r in summaries if r['maximum_used_lag_error_samples'] is not None), default=None)))
    for name, value in sources.items():
        if digest(root/name) != value:
            raise ValueError('experiment source changed')
    for name, value in artifacts.items():
        if digest(Path(name)) != value:
            raise ValueError('calibration artifact changed')
    report = dict(protocol_sha256=digest(output/'protocol.json'), sequence_count=len(records),
                  window_count=sum(len(r['trace']) for r in records), groups=groups)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/cpu-weak-tracking'))
    parser.add_argument('--calibration', type=Path, default=Path('output/cpu-multiseed-acquisition'))
    parser.add_argument('--replicas', type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.calibration, args.replicas), indent=2))
