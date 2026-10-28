"""Three independent calibration realizations, one fresh fixed evaluation set."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_noise_aware_multipath import evaluate,PROFILES
from experiments.validate_cpu_bandlimited_multipath import paired_summary
from gnss_gpu.noise_aware_multipath_cpu import BandlimitedFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate,_select_default_fit
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED

OUTPUT=Path('output/cpu-calibration-repeats')


def calibrate_repeats(records):
    result={}
    for repeat in range(3):
        subset=[r for r in records if r['calibration_repeat']==repeat]
        strata={p:asdict(calibrate_path_gate([r['fit']['bic_improvement'] for r in subset if r['profile']==p],
                             confidence_failure=.05/6)) for p in PROFILES}
        result[str(repeat)]=dict(strata=strata,white=strata['white'],multi=asdict(PathSelectionGate(
            max(g['score'] for g in strata.values()),sum(g['calibration_count'] for g in strata.values()),.01,.05)))
    return result


def apply_repeats(row,gates):
    outcomes={}
    for repeat,group in gates.items():
        outcomes[repeat]={}
        for coverage in ('white','multi'):
            fit=_select_default_fit(BandlimitedFit(**row['fit']),PathSelectionGate(**group[coverage]))
            error=(fit.selected_delay_chips-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE if fit.selected_delay_chips is not None else None
            outcomes[repeat][coverage]=dict(fit=asdict(fit),error_m=error)
    return dict(row,outcomes=outcomes)


def summarize(rows):
    groups={}
    for row in rows:groups.setdefault((row['profile'],row['echo_voltage']),[]).append(row)
    cells=[];contrasts=[]
    for (profile,ratio),group in groups.items():
        group=sorted(group,key=lambda r:r['replica'])
        for repeat in range(3):
            successes={}
            for coverage in ('white','multi'):
                selected=[r['outcomes'][str(repeat)][coverage] for r in group]
                success=[r['error_m'] is not None and abs(r['error_m'])<10 for r in selected]
                successes[coverage]=success
                two=sum(r['fit']['selected_model']=='two_path' for r in selected)
                cells.append(dict(profile=profile,echo_voltage=ratio,repeat=repeat,coverage=coverage,
                    attempts=len(group),successes=sum(success),two_paths=two,
                    unavailable=sum(r['error_m'] is None for r in selected),
                    two_path_exact_95=list(binomtest(two,len(group)).proportion_ci(confidence_level=.95,method='exact'))))
            paired=paired_summary(successes['white'],successes['multi'])
            primary=ratio==.8
            contrasts.append(dict(profile=profile,echo_voltage=ratio,repeat=repeat,**paired,
                                  primary=primary,adjusted_primary_p=min(1.,3*paired['paired_exact_p']) if primary else None))
    return dict(cells=cells,contrasts=contrasts)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-noise-ablation/protocol.json').read_text())
    hashes=dict(previous['source_sha256'])
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    hashes[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('output/cpu-noise-ablation').iterdir() if p.is_file()}
    tasks=[(p,0.,0.,0.,200000+repeat*10000+j*1000+r,False)
           for repeat in range(3) for j,p in enumerate(PROFILES) for r in range(477)]
    null=[(p,0.,0.,0.,240000+r,False) for p in PROFILES for r in range(128)]
    echo=[('white',.8,.1,0.,250000+r,False) for r in range(128)]
    protocol=dict(calibration_tasks=tasks,evaluation_tasks=null+echo,source_sha256=hashes,prior_sha256=prior,
        repeats=3,calibration_per_profile=477,evaluation_per_cell=128,
        context='Frozen estimated weighting; PRN1,37 reference dB-Hz,200 ms ideal coherence,3.683 MHz cutoff',
        calibration='Same 477/profile maximum-score rule as preceding ablation; white vs maximum of 3 profile maxima',
        confidence='Per-repeat conditional calibration confidence >=1-3*.99**477; no 95% joint claim over all 9 strata/repeats',
        primary='Three paired white-noise same-phase .1-chip echo comparisons, multi vs white coverage; Bonferroni family .05',
        evaluation='One shared untouched evaluation set for all calibration realizations; do not pool as independent successes',
        scope='Three independent calibration draws only; descriptive threshold variability, not a precise population distribution',
        success='Available absolute direct range error <10 m; all attempts retained')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        cal=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for i,row in enumerate(pool.map(evaluate,tasks)):
                row['calibration_repeat']=i//1431;cal.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(cal)%477==0:
                    stream.flush();print(f'Calibration {len(cal)}/4293 fits',flush=True)
        gates=calibrate_repeats(cal)
        gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(gates,indent=2))
        gatehash=hashlib.sha256(gatepath.read_bytes()).hexdigest()
        print('All six gates frozen',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echo):
                row=apply_repeats(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%128==0:print(f'Evaluation {len(rows)}/512 observations',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    for p,h in {**hashes,**prior}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256(gatepath.read_bytes()).hexdigest()==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=4293,evaluation=512,gate_sha256=gatehash),indent=2))


if __name__=='__main__':run()
