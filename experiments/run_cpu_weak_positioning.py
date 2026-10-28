"""Isolated per-PRN IQ -> weak tracking -> assisted ranges -> position/clock.

Static, zero-Doppler channel streams; no composite cross-PRN interference or
physical forest/rain claim. Calibration is new per PRN, white-null only.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.run_cpu_batch_receiver import FS, COUNT, render
from experiments.run_cpu_receiver_stress import NOISE_DB, seed_for
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch, acquire_ca_bit_edge, calibrate_ca_bit_edge
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig
from gnss_gpu.ca_local_delay_cpu import cancel_other_ca_local_delay
from gnss_gpu.weak_acquisition_cpu import CaWeakAcquisitionThreshold
from gnss_gpu.weak_tracking_cpu import CpuWeakCaTracker
from gnss_gpu.weak_positioning_cpu import C_LIGHT, position_from_weak_tracks

CAL, EVAL, SIGNAL = 0x57504341, 0x57504556, 0x57505349
SCENARIOS = ('strong_55', 'weak_37', 'only_three_37', 'three_excess_paths_300m', 'common_clock_plus_300m')
RECEIVER = np.array([120., -80., 20.])
PRIOR_POSITION = np.zeros(3)
CLOCK_M, PRIOR_BOUND_M = 1500., 6000.
START = 7*COUNT+137


def constellation():
    azimuth = np.deg2rad(np.arange(15., 360., 45.))
    elevation = np.deg2rad([20., 55., 30., 70., 25., 60., 40., 75.])
    unit = np.column_stack([np.cos(elevation)*np.sin(azimuth), np.cos(elevation)*np.cos(azimuth), np.sin(elevation)])
    return {i+1: unit[i]*(21e6+i*200000.) for i in range(8)}


def run(output, replicas=16):
    from gnss_gpu.input_validation import positive_int
    replicas = positive_int('replicas', replicas)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output directory must be empty')
    satellites = constellation()
    # Assistance uses the fixed prior position/clock, never RECEIVER or CLOCK_M.
    priors = {p: float(np.linalg.norm(sat-PRIOR_POSITION)) for p, sat in satellites.items()}
    bounds = {p: PRIOR_BOUND_M for p in satellites}
    configs = {p: CaBitEdgeSearch(CaSearch(FS, (-250., 0., 250.), p)) for p in satellites}
    cancellation = {p: CaCancellationConfig(FS, target_prn=p, nuisance_prns=()) for p in satellites}
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/'experiments/run_cpu_batch_receiver.py', root/'experiments/run_cpu_receiver_stress.py']
    paths += [root/'python/gnss_gpu'/(name+'.py') for name in ('weak_positioning_cpu', 'weak_tracking_cpu',
        'weak_acquisition_cpu', 'bit_edge_acquisition_cpu', 'ca_cancellation_cpu', 'ca_local_delay_cpu',
        'positioning_cpu', 'signal_sim_cpu', 'noncoherent_tolerance_cpu', 'code_delay_cpu', 'acquisition_cpu')]
    def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(root)): digest(p) for p in paths}
    protocol = dict(replicas=replicas, scenarios=SCENARIOS, satellite_positions_m={p:s.tolist() for p,s in satellites.items()},
        prior_position_m=PRIOR_POSITION.tolist(), prior_clock_m=0., prior_range_m=priors, prior_bound_m=bounds,
        simulated_receiver_m=RECEIVER.tolist(), simulated_clock_m=CLOCK_M, sampling_freq_hz=FS,
        code_epoch='Known aligned global transmit/receive code zero; absolute sample phase restored before range resolution.',
        iq_model='Eight isolated PRN streams, static zero Doppler, rectangular point sampling, alternating nav bits.',
        calibration=dict(records_per_prn=599, null='white only', max_false_alarm_probability=.01, joint_failure_budget=.05, family_members=8),
        evaluation_null_records_per_prn=199, domains=dict(calibration=CAL, evaluation=EVAL, signal=SIGNAL),
        target_windows=4, window_ms=20, receiver_search={p:asdict(c) for p,c in configs.items()},
        tracker=dict(confirmation_windows=2, coast_windows=2, gain=.25), source_sha256=hashes,
        limitations=['Assisted ambiguity resolution, no navigation-time decoding.', 'Isolated channels, no composite near-far validation.',
            'Illustrative attenuation/obstruction/excess path, not calibrated forest or rain physics.',
            'Coarse integer-delay ranging, no precision specification or PVT integrity guarantee.'])
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    scores = {p:[] for p in satellites}
    with (output/'calibration.jsonl').open('w') as handle:
        for index in range(599):
            iq = render('batch', [[] for _ in range(20)], [START+i*COUNT for i in range(20)],
                        [seed_for(CAL,index,i) for i in range(20)], NOISE_DB)
            row = dict(record=index, raw_iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest(), scores={})
            for p in satellites:
                residual = cancel_other_ca_local_delay(iq, cancellation[p])
                score = acquire_ca_bit_edge(residual.iq, configs[p]).statistic
                scores[p].append(score); row['scores'][p] = score
            handle.write(json.dumps(row)+'\n')
            if (index+1)%100 == 0:handle.flush(); print(f'calibration {index+1}/599', flush=True)
    gates = {p:CaWeakAcquisitionThreshold(cancellation[p], calibrate_ca_bit_edge({'white':scores[p]}, configs[p], methods_in_family=8)) for p in satellites}
    frozen = output/'frozen_thresholds.json'
    frozen.write_text(json.dumps({p:asdict(g) for p,g in gates.items()},indent=2)+'\n'); frozen_hash = digest(frozen)
    null_counts = {p:0 for p in satellites}
    with (output/'evaluation.jsonl').open('w') as handle:
        for index in range(199):
            iq = render('batch', [[] for _ in range(20)], [START+i*COUNT for i in range(20)],
                        [seed_for(EVAL,index,i) for i in range(20)], NOISE_DB)
            values = {p:acquire_ca_bit_edge(iq, configs[p]).statistic for p in satellites}
            flags = {p:value>gates[p].detection.score for p,value in values.items()}
            for p, flag in flags.items(): null_counts[p] += flag
            handle.write(json.dumps(dict(record=index, raw_iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest(), scores=values, accepted=flags))+'\n')
    records = []
    with (output/'positions.jsonl').open('w') as handle:
        for scenario in SCENARIOS:
            for replica in range(replicas):
                states = {i:{} for i in range(4)}
                iq_hashes = {}
                for p, satellite in satellites.items():
                    rho = float(np.linalg.norm(satellite-RECEIVER)+CLOCK_M)
                    excess = 300. if (scenario=='three_excess_paths_300m' and p<=3) or scenario=='common_clock_plus_300m' else 0.
                    cn0 = 55. if scenario=='strong_55' else 37.
                    amplitude = 0. if scenario=='only_three_37' and p>3 else 10**((cn0-55.)/20)
                    phase = float(np.random.default_rng(seed_for(SIGNAL,replica,p)).uniform(0.,2*np.pi))
                    channel = dict(prn=p, code_phase=-(rho+excess)*1.023e6/C_LIGHT, carrier_phase=phase,
                                   doppler_hz=0., amplitude=amplitude, nav_bits=[1,-1,1,-1,1,-1], nav_bit=1)
                    iq = render('batch', [[channel] for _ in range(80)], [START+i*COUNT for i in range(80)],
                                [seed_for(SIGNAL,replica*8+p,i) for i in range(80)], NOISE_DB)
                    iq_hashes[p] = hashlib.sha256(iq.tobytes()).hexdigest()
                    tracker = CpuWeakCaTracker(gates[p])
                    for window in range(4):
                        states[window][p] = tracker.update(iq[window*20:(window+1)*20], start_sample=START+window*20*COUNT)
                for window in range(4):
                    result = position_from_weak_tracks(states[window], satellites, priors, bounds, FS, initial_position_m=PRIOR_POSITION)
                    solution = result.solution
                    error = solution.position_m-RECEIVER if solution else None
                    row = dict(scenario=scenario, replica=replica, window=window, available_after_ms=(window+1)*20,
                        raw_iq_sha256=iq_hashes, states={p:asdict(s) for p,s in states[window].items()},
                        result=asdict(result), position_error_m=float(np.linalg.norm(error)) if error is not None else None,
                        horizontal_error_m=float(np.linalg.norm(error[:2])) if error is not None else None,
                        clock_error_m=solution.clock_bias_m-(CLOCK_M+(300. if scenario=='common_clock_plus_300m' else 0.)) if solution else None)
                    # PositionSolution contains NumPy arrays; preserve all unavailable rows too.
                    handle.write(json.dumps(row, default=lambda value:value.tolist())+'\n'); records.append(row)
            handle.flush(); print(f'finished {scenario}', flush=True)
    summaries = []
    for scenario in SCENARIOS:
        subset = [r for r in records if r['scenario']==scenario and r['window']>0]
        errors = [r['position_error_m'] for r in subset if r['position_error_m'] is not None]
        summaries.append(dict(scenario=scenario, epochs=len(subset), solved=len(errors),
            position_error_p50_m=float(np.median(errors)) if errors else None,
            position_error_p95_m=float(np.quantile(errors,.95)) if errors else None))
    if digest(frozen)!=frozen_hash or any(digest(root/p)!=h for p,h in hashes.items()):
        raise ValueError('source or gate changed during experiment')
    report = dict(protocol_sha256=digest(output/'protocol.json'), frozen_thresholds_sha256=frozen_hash,
                  null_accepted_per_prn=null_counts, null_records_per_prn=199, summaries=summaries,
                  total_position_epochs=len(records), scoring='Post-confirmation epochs; initial confirmation rows retained in positions.jsonl.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n'); return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/cpu-weak-positioning'))
    parser.add_argument('--replicas',type=int,default=16)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.replicas),indent=2))
