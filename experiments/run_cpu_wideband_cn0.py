"""Fixed wideband receiver: paired prefilter direct C/N0 sensitivity."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from experiments.replicate_cpu_drift_bandwidth import synthesize as baseline,paired_outcomes
from experiments.run_cpu_drift_selection import tasks,digest
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED

OUTPUT=Path('output/cpu-wideband-cn0')
PRIOR=Path('output/cpu-wideband-boundary')
LEVELS=(37.,43.,49.)


def tasklist():
    return [(kind,cn0,a,f,i) for kind,start,count in [('null',1280000,16),('echo',1290000,32)]
            for cn0 in LEVELS for _,a,f,i in tasks(kind,start,count)]


def synthesize(task):
    kind,cn0,drift,offset,replica=task
    if cn0 not in LEVELS:raise ValueError('unsupported C/N0 level')
    b,s,truth=baseline((kind,7.2,drift,offset,replica))
    if cn0==37.:return b,s,truth
    return s+(b-s)*10**((37-cn0)/20),s,truth


def evaluate(task):
    b,_,truth=synthesize(task);score,w=score_drift_paths(b,7.2*CHIP_RATE)
    fit=score['single' if task[0]=='null' else 'two']
    error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    return dict(task=task,kind=task[0],cn0_dbhz=task[1],replica=task[-1],true_delay_chips=truth,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        raw_error_m=error,available=fit.locally_usable,success=bool(fit.locally_usable and abs(error)<10),
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for kind in ('null','echo'):
        for cn0 in LEVELS:
            g=[r for r in rows if r['kind']==kind and r['cn0_dbhz']==cn0];n=len(g);success=sum(r['success'] for r in g)
            fits=[r['fits']['single' if kind=='null' else 'two'] for r in g]
            cells.append(dict(kind=kind,cn0_dbhz=cn0,attempts=n,successes=success,
                success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
                unavailable=sum(not r['available'] for r in g),available_large_errors=sum(r['available'] and abs(r['raw_error_m'])>=10 for r in g),
                boundary=sum(f['reason']=='search boundary' for f in fits),
                nonconverged=sum(not f['converged'] for f in fits),
                weak_information=sum(f['reason']=='weak local delay information' for f in fits)))
    primary=paired_outcomes([r for r in rows if r['kind']=='echo' and r['cn0_dbhz']==37],
                            [r for r in rows if r['kind']=='echo' and r['cn0_dbhz']==49])
    return dict(cells=cells,primary_37_vs_49=primary,
        scope='One prespecified exact two-sided paired contrast37vs49dBHz,alpha.05.43dBHz and null cells descriptive. Known model order,no gate reuse. Same signals and standardized noise within replica,not independent across levels. No causal weather or real-device claim.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    p=dict(source_sha256=sources,tasks=tasklist(),
        design='Fresh32 echo replicas and16 null replicas x prefilter direct C/N0 37/43/49dBHz. Fixed7.3656MHz one-sided bandwidth,800ms,PRN1,common carrier/drift distributions unchanged. Echo voltage.8,gap.03chip,phasepi,relative Doppler0.',
        intervention='Same signal and standardized noise for each replica. Scale37dBHz noise amplitude by10**((37-cn0)/20). No change to receiver,bounds,weights algorithm,guards or integration time. At37dBHz exact original generator bytes.',
        primary='Only confirmatory comparison:echo37vs49dBHz,paired success(available and abs(error)<10m),two-sided exact McNemar,alpha.05;all32 pairs including unavailable,no optional stopping.43dBHz/nulls descriptive.',
        limits='Known single-path order for null,known two-path order for echo;not automatic model selection or warning. Fixed synthetic paths and white noise,not actual rain/forest/RF/PVT. No calibrated gate reused.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,p['tasks']):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if len(rows)%16==0:print(f'C/N0 evaluation {len(rows)}/144',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','evaluation.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
