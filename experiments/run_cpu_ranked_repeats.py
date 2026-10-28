"""Two fresh calibration draws plus the preserved initial rank-gate draw."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
import numpy as np
from experiments.run_cpu_noise_aware_multipath import evaluate,PROFILES
from experiments.run_cpu_ranked_calibration import calibrate,apply_gates,summarize

OUTPUT=Path('output/cpu-ranked-repeats')
INITIAL=Path('output/cpu-ranked-calibration')


def apply_repeats(row,gates):
    return dict(row,outcomes={repeat:apply_gates(row,gate)['outcomes'] for repeat,gate in gates.items()})


def summarize_repeats(rows,gates):
    cells=[];contrasts=[]
    for repeat in gates:
        report=summarize([dict(row,outcomes=row['outcomes'][repeat]) for row in rows])
        cells.extend(dict(cell,repeat=int(repeat)) for cell in report['cells'])
        contrasts.extend(dict(c,repeat=int(repeat),adjusted_primary_p=min(1.,3*c['paired_exact_p']) if c['primary'] else None)
                         for c in report['contrasts'])
    variability={}
    for method in ('maximum','ranked'):
        thresholds=[gates[str(r)]['combined'][method]['score'] for r in range(3)]
        wins=[next(c['successes'] for c in cells if c['repeat']==r and c['echo_voltage']!=0 and c['method']==method) for r in range(3)]
        variability[method]=dict(thresholds=thresholds,threshold_range=float(np.ptp(thresholds)),
            sample_threshold_std=float(np.std(thresholds,ddof=1)),successes_on_shared_echo_set=wins,
            success_count_range=max(wins)-min(wins))
    return dict(cells=cells,contrasts=contrasts,descriptive_variability=variability)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    previous=json.loads((INITIAL/'protocol.json').read_text())
    sources=dict(previous['source_sha256'])
    for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    sources[str(Path(__file__).resolve())]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prior={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in INITIAL.iterdir() if p.is_file()}
    initial=[json.loads(x) for x in (INITIAL/'calibration.jsonl').read_text().splitlines()]
    gates={'0':calibrate(initial)}
    assert gates['0']==json.loads((INITIAL/'frozen_gates.json').read_text())
    tasks=[(p,0.,0.,0.,400000+repeat*30000+j*10000+r,False)
           for repeat in range(2) for j,p in enumerate(PROFILES) for r in range(1000)]
    null=[(p,0.,0.,0.,470000+r,False) for p in PROFILES for r in range(128)]
    echo=[('white',.8,.1,0.,480000+r,False) for r in range(128)]
    protocol=dict(new_calibration_tasks=tasks,evaluation_tasks=null+echo,
        reused_initial_calibration=str(INITIAL/'calibration.jsonl'),source_sha256=sources,prior_sha256=prior,
        rank=4,n_per_profile=1000,calibration_draws=3,new_calibration_draws=2,
        context='Frozen estimated-noise weighting;37 reference dB-Hz,200 ms ideal coherence,3.683 MHz,PRN1,default assistance',
        guarantee='Same fixed fourth-largest rule; per-draw joint failure bound 3*BinomialCDF(3;1000,.01). No 95% simultaneous guarantee across all three draws',
        primary='Shared white same-phase .1-chip echo set, ranked vs maximum in each draw; three exact paired tests with Bonferroni .05',
        stability='Report observed threshold range/sample SD and success range; three draws cannot establish population variance reduction',
        independence='Initial calibration preserved as development reference; two independent new calibration draws; all use same untouched evaluation set',
        limits='No choice of best calibration or rank based on echo outcomes; no pooling shared evaluation as independent trials')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        cal=[]
        with (OUTPUT/'new_calibration.jsonl').open('w') as stream:
            for i,row in enumerate(pool.map(evaluate,tasks)):
                row['calibration_repeat']=1+i//3000;cal.append(row)
                stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(cal)%500==0:
                    stream.flush();print(f'New calibration {len(cal)}/6000 fits',flush=True)
        for repeat in (1,2):gates[str(repeat)]=calibrate([r for r in cal if r['calibration_repeat']==repeat])
        gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(gates,indent=2))
        gatehash=hashlib.sha256(gatepath.read_bytes()).hexdigest()
        print('All calibration gates frozen',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echo):
                row=apply_repeats(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%128==0:print(f'Evaluation {len(rows)}/512',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize_repeats(rows,gates),indent=2,allow_nan=False))
    for p,h in {**sources,**prior}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256(gatepath.read_bytes()).hexdigest()==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(new_calibration=6000,reused_calibration=3000,evaluation=512,gate_sha256=gatehash),indent=2))


if __name__=='__main__':run()
