"""Fresh weak-acquisition validation after local-delay nuisance refinement."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.run_cpu_batch_receiver import FS, render
from experiments.run_cpu_receiver_stress import NOISE_DB
from experiments.run_cpu_other_prn_nulls import PROFILES, scene
from experiments.run_cpu_receiver_sensitivity import wilson_interval
from experiments.run_cpu_noncoherent_sensitivity import score_detection
from gnss_gpu.acquisition_cpu import CaSearch
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig,cancel_other_ca
from gnss_gpu.ca_local_delay_cpu import _refine_local_delays
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch,acquire_ca_noncoherent
from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeSearch,acquire_ca_bit_edge,calibrate_ca_bit_edge
from gnss_gpu.stratified_tolerance_cpu import calibrate_ca_stratified_tolerance
from gnss_gpu.noncoherent_tolerance_cpu import tolerance_order
from gnss_gpu.input_validation import positive_int

CALIBRATION_DOMAIN,EVALUATION_DOMAIN,SIGNAL_DOMAIN=0x57524341,0x57524556,0x57525349


def summarize_signals(records):
    grouped=[]
    for profile in PROFILES:
        for cn0 in (37.,41.):
            for doppler in (170.,700.):
                subset=[r for r in records if r['profile']==profile and r['nominal_primary_cn0_db_hz']==cn0 and r['primary_doppler_hz']==doppler]
                if not subset:continue
                for method in ('noncoherent_4','bit_edge_20'):
                    n=len(subset); accepted=sum(r['methods'][method]['accepted'] for r in subset)
                    count=sum(r['methods'][method]['passes_declared_criterion'] for r in subset)
                    grouped.append(dict(profile=profile,nominal_primary_cn0_db_hz=cn0,primary_doppler_hz=doppler,
                        method=method,replicas=n,accepted=accepted,criterion=count,criterion_fraction=count/n,
                        criterion_wilson_95=wilson_interval(count,n)))
    return grouped


def run(output,*,calibration_records=499,evaluation_records=499,replicas=32):
    from gnss_gpu.bit_edge_acquisition_cpu import CaBitEdgeDetection
    calibration_records=positive_int('calibration_records',calibration_records)
    evaluation_records=positive_int('evaluation_records',evaluation_records)
    replicas=positive_int('replicas',replicas)
    tolerance_order(calibration_records,failure_probability=.05/6)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    cancellation=CaCancellationConfig(FS)
    old=CaNoncoherentSearch(CaSearch(FS,(-1000.,-500.,0.,500.,1000.)),4,5)
    new=CaBitEdgeSearch(CaSearch(FS,tuple(float(x) for x in range(-1000,1001,250))))
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__)]
    paths += [root/'experiments'/(name+'.py') for name in ('run_cpu_batch_receiver','run_cpu_receiver_stress','run_cpu_other_prn_nulls','run_cpu_receiver_sensitivity','run_cpu_noncoherent_sensitivity')]
    paths += [root/'python/gnss_gpu'/(name+'.py') for name in ('bit_edge_acquisition_cpu','ca_local_delay_cpu','ca_cancellation_cpu','noncoherent_acquisition_cpu','noncoherent_tolerance_cpu','stratified_tolerance_cpu','signal_sim_cpu','acquisition_cpu','code_delay_cpu','code_detection_cpu')]
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes={str(p.relative_to(root)):digest(p) for p in paths}
    protocol=dict(calibration_records_per_profile=calibration_records,evaluation_records_per_profile=evaluation_records,
        signal_replicas=replicas,profiles=PROFILES,horizon_ms=20,cancellation=asdict(cancellation),local_delay_radius_samples=1,
        detectors=dict(noncoherent_4=asdict(old),bit_edge_20=asdict(new)),
        noise_domains=dict(calibration=CALIBRATION_DOMAIN,evaluation=EVALUATION_DOMAIN,signal=SIGNAL_DOMAIN),
        max_false_alarm_probability=.01,joint_calibration_failure_budget=.05,family_members=6,
        acceptance_criteria=dict(profile='near_far_20db',cn0_db_hz=37.,doppler_hz=[170.,700.],
            required_criterion_fraction=.75,criterion='detection plus coarse carrier/phase criterion',
            null_requirement='Full-pipeline 1% stratum tolerance calibration; report independent null outcomes without retuning.'),
        source_sha256=hashes,evaluation_policy='Freeze before evaluation. Do not retune using evaluation outcomes.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def analyse(iq):
        residual=cancel_other_ca(iq,cancellation)
        refined=_refine_local_delays(iq,cancellation,residual,delay_radius_samples=1)
        candidates=dict(noncoherent_4=acquire_ca_noncoherent(residual.iq,old),bit_edge_20=acquire_ca_bit_edge(refined.iq,new))
        metadata=dict(component_count=len(residual.components),removed_energy_fraction=residual.removed_energy_fraction,
                      refined_removed_energy_fraction=refined.removed_energy_fraction,refined_gram_condition=refined.gram_condition,
                      raw_iq_sha256=hashlib.sha256(iq.tobytes()).hexdigest(),residual_iq_sha256=hashlib.sha256(residual.iq.tobytes()).hexdigest(),
                      refined_residual_iq_sha256=hashlib.sha256(refined.iq.tobytes()).hexdigest())
        return candidates,metadata
    scores={method:{profile:[] for profile in PROFILES} for method in ('noncoherent_4','bit_edge_20')}
    with (output/'calibration.jsonl').open('w') as handle:
        for profile in PROFILES:
            for index in range(calibration_records):
                rows,starts,seeds,_=scene(profile,CALIBRATION_DOMAIN,index)
                candidates,audit=analyse(render('batch',rows,starts,seeds,NOISE_DB))
                for name,value in candidates.items():scores[name][profile].append(value.statistic)
                handle.write(json.dumps(dict(profile=profile,record=index,candidates={k:asdict(v) for k,v in candidates.items()},audit=audit))+'\n')
                if (index+1)%100==0:handle.flush();print(f'calibration {profile}: {index+1}/{calibration_records}',flush=True)
    thresholds=dict(noncoherent_4=calibrate_ca_stratified_tolerance(scores['noncoherent_4'],old),
                    bit_edge_20=calibrate_ca_bit_edge(scores['bit_edge_20'],new))
    frozen=output/'frozen_thresholds.json'
    frozen.write_text(json.dumps({k:asdict(v) for k,v in thresholds.items()},indent=2)+'\n');frozen_hash=digest(frozen)
    counts={profile:{method:0 for method in thresholds} for profile in PROFILES}
    with (output/'evaluation.jsonl').open('w') as handle:
        for profile in PROFILES:
            for index in range(evaluation_records):
                rows,starts,seeds,_=scene(profile,EVALUATION_DOMAIN,index)
                candidates,audit=analyse(render('batch',rows,starts,seeds,NOISE_DB))
                accepted={method:candidate.statistic>thresholds[method].score for method,candidate in candidates.items()}
                for method,flag in accepted.items():counts[profile][method]+=flag
                handle.write(json.dumps(dict(profile=profile,record=index,accepted=accepted,candidates={k:asdict(v) for k,v in candidates.items()},audit=audit))+'\n')
                if (index+1)%100==0:handle.flush();print(f'evaluation {profile}: {index+1}/{evaluation_records}',flush=True)
    signals=[]
    with (output/'signals.jsonl').open('w') as handle:
        for profile in PROFILES:
            for cn0 in (37.,41.):
                for doppler in (170.,700.):
                    for replica in range(replicas):
                        rows,starts,seeds,nominal=scene(profile,SIGNAL_DOMAIN,replica,target_cn0=cn0,target_doppler=doppler)
                        candidates,audit=analyse(render('batch',rows,starts,seeds,NOISE_DB))
                        methods={name:score_detection(CaBitEdgeDetection(candidate.statistic>thresholds[name].score,
                            candidate.statistic,thresholds[name].score,candidate),doppler,nominal) for name,candidate in candidates.items()}
                        result=dict(profile=profile,nominal_primary_cn0_db_hz=cn0,primary_doppler_hz=doppler,replica=replica,methods=methods,audit=audit)
                        signals.append(result);handle.write(json.dumps(result)+'\n')
            handle.flush();print(f'signals {profile} complete',flush=True)
    grouped=summarize_signals(signals)
    (output/'aggregate.json').write_text(json.dumps(grouped,indent=2)+'\n')
    primary=[row for row in grouped if row['method']=='bit_edge_20' and row['profile']=='near_far_20db' and row['nominal_primary_cn0_db_hz']==37.]
    success=len(primary)==2 and all(row['criterion_fraction']>=.75 for row in primary)
    if digest(frozen)!=frozen_hash or any(digest(root/path)!=expected for path,expected in hashes.items()):raise ValueError('source or frozen threshold changed')
    report=dict(protocol_sha256=digest(output/'protocol.json'),frozen_thresholds_sha256=frozen_hash,
        null_results={profile:{method:dict(accepted=count,records=evaluation_records,fraction=count/evaluation_records,
            pointwise_wilson_95=wilson_interval(count,evaluation_records)) for method,count in values.items()} for profile,values in counts.items()},
        signal_records=len(signals),primary_results=primary,declared_capture_criteria_met=success,
        limitations=['Common calibrated upper false-alarm budget, not equality of actual false-alarm rates.',
          'One bit reversal model, short fixed carrier/code delay, and finite nuisance strata only.',
          'Coarse capture criterion, not sustained tracking or precision positioning.',
          'Shortlisting can miss peaks; full pipeline calibration includes the selection.',
          'Revised pipeline changes both delay-subspace cancellation and coherent acquisition.',
          'Local predeclared experimental criterion is not an RF sensitivity specification or novelty claim.'])
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/cpu-weak-acquisition-refined'))
    parser.add_argument('--calibration-records',type=int,default=499)
    parser.add_argument('--evaluation-records',type=int,default=499)
    parser.add_argument('--replicas',type=int,default=32)
    args=parser.parse_args()
    print(json.dumps(run(args.output,calibration_records=args.calibration_records,evaluation_records=args.evaluation_records,replicas=args.replicas),indent=2))
