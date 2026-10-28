"""Paired known-two-path pilot: dominant-direct assumption valid vs violated."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
from dataclasses import asdict
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.cached_drift_carrier_paths_cpu import score_drift_paths
from gnss_gpu.dominant_drift_paths_cpu import fit_dominant_paths,fit_parameters,geometric_interior
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED
from experiments.stress_cpu_drift_gap_amplitude import synthesize,digest,tasks

OUTPUT=Path('output/cpu-dominant-path-pilot')
PRIOR=Path('output/cpu-drift-ambiguity')


def tasklist():
    return [(k,ratio,.03,np.pi,a,f,i) for ratio in (.8,1.2) for k,a,f,i in tasks('echo',1220000,16)]


def evaluate(task):
    b,signal,truth=synthesize(task)
    score,w=score_drift_paths(b)
    fit=fit_dominant_paths(b,w,score['single'],score['two'])
    free=score['two'];scale=LIGHT_SPEED/CHIP_RATE
    free_ok=bool(free.converged and geometric_interior(fit_parameters(free)))
    constrained_ok=fit['converged'] and fit['geometric_interior']
    e0=(free.first_delay_chips-truth)*scale;e1=(fit['parameters'][0]-truth)*scale
    return dict(task=task,replica=task[-1],echo_voltage=task[1],true_delay_chips=truth,
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
        baseline_fits={k:asdict(score[k]) for k in ('single','two','preliminary')},constrained=fit,
        free_error_m=e0,constrained_error_m=e1,free_eligible=free_ok,constrained_eligible=constrained_ok,
        free_success=free_ok and abs(e0)<10,constrained_success=constrained_ok and abs(e1)<10)


def summarize(rows):
    cells=[]
    for ratio in (.8,1.2):
        g=[r for r in rows if r['echo_voltage']==ratio];n=len(g)
        c=sum(r['constrained_success'] for r in g)
        cells.append(dict(echo_voltage=ratio,attempts=n,free_success=sum(r['free_success'] for r in g),
            constrained_success=c,constrained_success_exact_95=list(binomtest(c,n).proportion_ci(method='exact')),
            free_ineligible=sum(not r['free_eligible'] for r in g),constrained_ineligible=sum(not r['constrained_eligible'] for r in g),
            gains=sum(r['constrained_success'] and not r['free_success'] for r in g),
            losses=sum(r['free_success'] and not r['constrained_success'] for r in g),
            amplitude_constraint_active=sum(r['constrained']['amplitude_constraint_active'] for r in g)))
    return dict(cells=cells,scope='Known-two-path exploratory pilot. Same fixed baseline-estimated weights,paired noise across voltage groups. Success requires optimizer convergence,geometric interior,and |error|<10m for both methods. No local-curvature guard,model selection,alarm,or uncertainty calibration. Amplitude boundary is reported,not automatically rejected.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__,'python/gnss_gpu/dominant_drift_paths_cpu.py','python/gnss_gpu/dominant_path_amplitudes_cpu.py'):
        sources[str(Path(path).resolve())]=digest(path)
    assert all(digest(p)==h for p,h in sources.items())
    ts=tasklist()
    p=dict(source_sha256=sources,tasks=ts,
        design='16 fresh paired replicas x voltage ratio.8/1.2. Fixed gap.03chip,phasepi,relative Doppler0,800ms,white37dBHz,PRN1,3.6828MHz one-sided. Common carrier/drift random as previous studies. No true delay/amplitude supplied to fitter.',
        constraint='|echo complex voltage| <= |first complex voltage|. Analytic global amplitude minimization for each geometry;finite multistart geometry search,not guaranteed global.',
        comparison='Known-two-path unconstrained vs constrained;same residual weights from baseline pipeline. Frozen baseline initializer plus same coarse search. Convergence and geometric boundary gates applied equally;no local std guard or gate reuse. Ratio1.2 deliberately violates dominant-first assumption.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,ts):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            print(f'Pilot {len(rows)}/32',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','evaluation.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
