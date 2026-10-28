"""Fresh paired maximum vs binomial-tolerance rank gate comparison."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
from scipy.stats import binomtest
from experiments.run_cpu_noise_aware_multipath import evaluate,PROFILES
from experiments.validate_cpu_bandlimited_multipath import paired_summary
from gnss_gpu.noise_aware_multipath_cpu import BandlimitedFit
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate,_select_default_fit
from gnss_gpu.ranked_path_gate_cpu import calibrate_ranked_gate,permissible_rank
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED

OUTPUT=Path('output/cpu-ranked-calibration')


def calibrate(records):
    strata={}
    for p in PROFILES:
        scores=[r['fit']['bic_improvement'] for r in records if r['profile']==p]
        strata[p]=dict(maximum=asdict(calibrate_path_gate(scores,confidence_failure=.05/3)),
                       ranked=asdict(calibrate_ranked_gate(scores)))
    gates={name:asdict(PathSelectionGate(max(s[name]['score'] for s in strata.values()),
                     sum(s[name]['calibration_count'] for s in strata.values()),.01,.05)) for name in ('maximum','ranked')}
    return dict(strata=strata,combined=gates)


def apply_gates(row,gates):
    outcomes={}
    for name,gate in gates['combined'].items():
        fit=_select_default_fit(BandlimitedFit(**row['fit']),PathSelectionGate(**gate))
        error=(fit.selected_delay_chips-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE if fit.selected_delay_chips is not None else None
        outcomes[name]=dict(fit=asdict(fit),error_m=error,score_exceeds=row['fit']['bic_improvement']>gate['score'])
    return dict(row,outcomes=outcomes)


def summarize(rows):
    groups={}
    for row in rows:groups.setdefault((row['profile'],row['echo_voltage']),[]).append(row)
    cells=[];contrasts=[]
    for (profile,ratio),group in groups.items():
        successes={}
        for name in ('maximum','ranked'):
            selected=[r['outcomes'][name] for r in group]
            success=[r['error_m'] is not None and abs(r['error_m'])<10 for r in selected]
            successes[name]=success
            two=sum(r['fit']['selected_model']=='two_path' for r in selected)
            cells.append(dict(profile=profile,echo_voltage=ratio,method=name,attempts=len(group),successes=sum(success),
                two_paths=two,two_path_exact_95=list(binomtest(two,len(group)).proportion_ci(confidence_level=.95,method='exact')),
                score_exceedances=sum(r['score_exceeds'] for r in selected),unavailable=sum(r['error_m'] is None for r in selected)))
        paired=paired_summary(successes['maximum'],successes['ranked'])
        primary=profile=='white' and ratio==.8
        contrasts.append(dict(profile=profile,echo_voltage=ratio,**paired,primary=primary,
            primary_improvement_supported=paired['difference']>0 and paired['paired_exact_p']<.05 if primary else None))
    return dict(cells=cells,contrasts=contrasts)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-calibration-repeats/protocol.json').read_text())
    sources=dict(previous['source_sha256'])
    for p,h in sources.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/ranked_path_gate_cpu.py')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    prior={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('output/cpu-calibration-repeats').iterdir() if p.is_file()}
    cal=[(p,0.,0.,0.,300000+j*10000+r,False) for j,p in enumerate(PROFILES) for r in range(1000)]
    null=[(p,0.,0.,0.,350000+r,False) for p in PROFILES for r in range(299)]
    echoes=[(p,.8,.1,0.,360000+r,False) for p in PROFILES for r in range(128)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echoes,source_sha256=sources,prior_sha256=prior,
        calibration_count_per_stratum=1000,rank_from_largest=permissible_rank(1000),tail_budget=.01,confidence_failure_per_stratum=.05/3,
        guarantee='Sum over 3 strata of BinomialCDF(k-1;1000,.01) <= .05; combined=max per-stratum thresholds; maximum comparator is stricter',
        primary='White-noise same-phase .1-chip echo selected range success, ranked vs maximum; one exact two-sided paired test alpha=.05',
        null='Independent single-path evaluation 299 per noise profile; all false selections and unavailable estimates reported',
        context='Frozen estimated-noise weighting,37 reference dB-Hz,200 ms ideal coherence,3.683 MHz one-sided,PRN1,default assistance',
        pairing='Both thresholds use identical calibration observations; both methods use identical fresh evaluation observations',
        scope='One calibration realization; rank fixed from n/alpha/delta before any scores; not a multi-draw stability proof')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%250==0:
                    stream.flush();print(f'Calibration {len(calibration)}/3000',flush=True)
        gates=calibrate(calibration)
        gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(gates,indent=2))
        gatehash=hashlib.sha256(gatepath.read_bytes()).hexdigest()
        print('Frozen gates: '+json.dumps({k:v['score'] for k,v in gates['combined'].items()}),flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echoes):
                row=apply_gates(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%256==0:print(f'Evaluation {len(rows)}/1281',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    for p,h in {**sources,**prior}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert hashlib.sha256(gatepath.read_bytes()).hexdigest()==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=3000,evaluation=1281,gate_sha256=gatehash),indent=2))


if __name__=='__main__':run()
