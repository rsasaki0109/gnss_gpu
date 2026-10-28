"""Paired 2x2 ablation: estimated weighting versus calibration coverage."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_noise_aware_multipath import evaluate,PROFILES
from experiments.validate_cpu_bandlimited_multipath import paired_summary
from gnss_gpu.bandlimited_multipath_v2_cpu import BandlimitedFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate,_select_default_fit
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED

OUTPUT=Path('output/cpu-noise-ablation')
ARMS={'unweighted_white':('old_fit','white'), 'unweighted_multi':('old_fit','multi'),
      'weighted_white':('fit','white'), 'weighted_multi':('fit','multi')}
CONTRASTS=[('unweighted_white','weighted_white','weighting_at_white_calibration'),
           ('weighted_white','weighted_multi','coverage_at_weighted_fit'),
           ('unweighted_white','unweighted_multi','coverage_at_unweighted_fit'),
           ('unweighted_multi','weighted_multi','weighting_at_multi_calibration')]


def calibrate_arms(records):
    strata={}
    for field in ('old_fit','fit'):
        strata[field]={p:asdict(calibrate_path_gate([r[field]['bic_improvement'] for r in records if r['profile']==p],
                              confidence_failure=.05/6)) for p in PROFILES}
    gates={}
    for arm,(field,coverage) in ARMS.items():
        if coverage=='white': gates[arm]=strata[field]['white']
        else:
            gates[arm]=asdict(PathSelectionGate(max(g['score'] for g in strata[field].values()),
                       sum(g['calibration_count'] for g in strata[field].values()),.01,.05))
    return dict(arms=gates,strata=strata)


def apply_arms(row,gates):
    outcomes={}
    for arm,(field,_) in ARMS.items():
        selected=_select_default_fit(BandlimitedFit(**row[field]),PathSelectionGate(**gates['arms'][arm]))
        error=((selected.selected_delay_chips-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE
               if selected.selected_delay_chips is not None else None)
        outcomes[arm]=dict(fit=asdict(selected),error_m=error)
    return dict(row,outcomes=outcomes)


def summarize(rows):
    groups={}
    for row in rows: groups.setdefault((row['profile'],row['echo_voltage'],row['phase_rad']),[]).append(row)
    cells=[];contrasts=[]
    for (profile,ratio,phase),group in groups.items():
        group=sorted(group,key=lambda r:r['replica'])
        successes={}
        for arm in ARMS:
            selected=[r['outcomes'][arm] for r in group]
            success=[r['error_m'] is not None and abs(r['error_m'])<10 for r in selected]
            successes[arm]=success
            two=sum(r['fit']['selected_model']=='two_path' for r in selected)
            ci=binomtest(sum(success),len(group)).proportion_ci(confidence_level=.95,method='exact')
            cells.append(dict(profile=profile,echo_voltage=ratio,phase_rad=phase,arm=arm,
                attempts=len(group),successes=sum(success),success_exact_95=list(ci),two_paths=two,
                two_path_exact_95=list(binomtest(two,len(group)).proportion_ci(confidence_level=.95,method='exact')),
                unavailable=sum(r['error_m'] is None for r in selected)))
        for a,b,name in CONTRASTS:
            paired=paired_summary(successes[a],successes[b])
            primary=profile=='white' and ratio==.8 and phase==0 and name in (
                'weighting_at_white_calibration','coverage_at_weighted_fit')
            contrasts.append(dict(profile=profile,echo_voltage=ratio,phase_rad=phase,contrast=name,
                baseline_arm=a,selected_arm=b,**paired,primary=primary,
                adjusted_primary_p=min(1.,2*paired['paired_exact_p']) if primary else None))
    return dict(cells=cells,contrasts=contrasts)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-noise-aware-multipath/protocol.json').read_text())
    sources=dict(previous['source_sha256'])
    for p,h in sources.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('output/cpu-noise-aware-multipath').iterdir() if p.is_file()}
    cal=[(p,0.,0.,0.,100000+j*1000+r,True) for j,p in enumerate(PROFILES) for r in range(477)]
    null=[(p,0.,0.,0.,110000+r,True) for p in PROFILES for r in range(128)]
    echoes=[(p,.8,.1,phase,120000+r,True) for p in PROFILES for phase in (0.,np.pi/2) for r in range(64)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echoes,source_sha256=sources,prior_sha256=prior,
        arms=ARMS,contrasts=CONTRASTS,
        context='PRN1, 37 reference dB-Hz, 200 ms ideal static coherence, 3.683 MHz one-sided; default estimators unchanged',
        design='2x2: unweighted/estimated weighting crossed with white-only/three-profile calibration; same records for all arms',
        calibration='477 null records per profile; six model/profile maxima; joint confidence via 6*.99**477 <= .05; strict exceedance',
        coverage='White-only gates control specified white stratum only; multi gates use max of three profile maxima for each estimator',
        primary='White same-phase echo: weighted_white minus unweighted_white; weighted_multi minus weighted_white. Two exact paired tests, Bonferroni family .05',
        success='Available absolute direct range error <10 m; all attempted records included',
        pairing='Fresh disjoint calibration/null/echo IDs; evaluation replicas shared across profiles and echo phases',
        limits='One fresh calibration draw; effects include estimator-specific recalibration. No gate tuning on echo results.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%159==0: print(f'Calibration {len(calibration)}/1431 paired fits',flush=True)
        gates=calibrate_arms(calibration)
        gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(gates,indent=2))
        gatehash=hashlib.sha256(gatepath.read_bytes()).hexdigest()
        print('Frozen gates: '+json.dumps({a:g['score'] for a,g in gates['arms'].items()}),flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echoes):
                row=apply_arms(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%128==0: print(f'Evaluation {len(rows)}/768 observations, four selections each',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    for p,h in {**sources,**prior}.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256(gatepath.read_bytes()).hexdigest()==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=1431,evaluation=768,gate_sha256=gatehash),indent=2))


if __name__=='__main__': run()
