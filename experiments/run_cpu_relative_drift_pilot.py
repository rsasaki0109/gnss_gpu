"""Paired known-order relative-drift refinement, frozen old residual weights."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_echo_sign_reversal import synthesize,DOPPLERS,RATES
from experiments.run_cpu_drift_selection import tasks,digest
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.relative_drift_paths_cpu import fit_relative_drift
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED

OUTPUT=Path('output/cpu-relative-drift-pilot')
PRIOR=Path('output/cpu-echo-sign-reversal')


def tasklist():
    return [(fd,rate,a,f,i) for fd in DOPPLERS for rate in RATES
            for _,a,f,i in tasks('echo',1390000,16)]


def evaluate(task):
    b,_,truth=synthesize(task)
    old,w=score_drift_paths(b,7.2*CHIP_RATE)
    new=fit_relative_drift(b,w,old['two'],7.2*CHIP_RATE)
    models={}
    for name,fit in (('common_drift',old['two']),('relative_drift',new)):
        error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
        models[name]=dict(fit=asdict(fit),raw_error_m=error,available=fit.locally_usable,
                         success=bool(fit.locally_usable and abs(error)<10))
    fd,rate,drift,offset,replica=task
    return dict(task=task,initial_doppler_hz=fd,relative_drift_hz_s=rate,replica=replica,
        true_delay_chips=truth,models=models,
        estimated_relative_drift_error_hz_s=new.relative_drift_hz_s-rate,
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for fd in DOPPLERS:
        for rate in RATES:
            group=[r for r in rows if r['initial_doppler_hz']==fd and r['relative_drift_hz_s']==rate]
            models={}
            for name in ('common_drift','relative_drift'):
                g=[r['models'][name] for r in group];k=sum(r['success'] for r in g);n=len(g)
                models[name]=dict(successes=k,attempts=n,
                    success_exact_95=list(binomtest(k,n).proportion_ci(method='exact')),
                    unavailable=sum(not r['available'] for r in g),
                    available_large_errors=sum(r['available'] and abs(r['raw_error_m'])>=10 for r in g),
                    median_absolute_raw_error_m=float(np.median([abs(r['raw_error_m']) for r in g])),
                    max_absolute_raw_error_m=max(abs(r['raw_error_m']) for r in g))
            cells.append(dict(initial_doppler_hz=fd,relative_drift_hz_s=rate,models=models,
                median_paired_absolute_error_change_m=float(np.median([
                    abs(r['models']['relative_drift']['raw_error_m'])-abs(r['models']['common_drift']['raw_error_m']) for r in group])),
                median_absolute_relative_drift_error_hz_s=float(np.median([
                    abs(r['estimated_relative_drift_error_hz_s']) for r in group]))))
    return dict(cells=cells,scope='16 fresh paired replicas across6cells. Known two-path order, fixed old weights. Exploratory, no hypothesis test, pointwise intervals. Not model-selection validation.')


def benchmark(tasks_to_time):
    # Serial, after the parallel evaluation; these two samples are descriptive.
    timings=[]
    warm,_,_=synthesize(tasks_to_time[0]);score,w=score_drift_paths(warm,7.2*CHIP_RATE)
    fit_relative_drift(warm,w,score['two'],7.2*CHIP_RATE)
    for task in tasks_to_time:
        b,_,_=synthesize(task);start=time.perf_counter()
        score,w=score_drift_paths(b,7.2*CHIP_RATE);oldtime=time.perf_counter()-start
        start=time.perf_counter();fit_relative_drift(b,w,score['two'],7.2*CHIP_RATE)
        extra=time.perf_counter()-start
        timings.append(dict(task=task,baseline_seconds=oldtime,refinement_seconds=extra,
                            total_seconds=oldtime+extra))
    return timings


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__,'python/gnss_gpu/relative_drift_paths_cpu.py'):
        sources[str(Path(path).resolve())]=digest(path)
    assert all(digest(path)==h for path,h in sources.items())
    protocol=dict(source_sha256=sources,tasks=tasklist(),
        design='Initial Doppler +/-.7Hz x relative drift0,-2,+2Hz/s x16 new shared replicas1390000..15;49dBHz,800ms,7.3656MHz one-sided,PRN1,voltage.8,initial excess.03chip,phasepi. Existing physically coupled generator unchanged.',
        estimator='Run old full receiver; freeze its weights. Refine two paths with extra echo drift bounded+/-4Hz/s,15 deterministic starts,400iterations. Other bounds and usability criteria unchanged; covariance denominator updated to10 real parameters. No truth passed to fitter.',
        evaluation='Known two-path order, compare paired direct range errors and existing usable+abs(error)<10m endpoint. No reused gate or detection claim. Fresh data, no pooling. Descriptive pointwise95% intervals.',
        timing='After all parallel tasks, warm once then serial timing of first replica from each zero-crossing cell. Measure old full pipeline and incremental refinement separately;2samples,not realtime certification.',
        limits='Refinement retains fixed code delays; generator has code motion. Shared residual weights may retain old-model bias. Synthetic harmonic approximation, no RF bin leakage,forest/rain/PVT or public data validation.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row in pool.map(evaluate,protocol['tasks']):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if len(rows)%16==0:print(f'Relative drift {len(rows)}/96',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    timings=benchmark([protocol['tasks'][16],protocol['tasks'][80]])
    (OUTPUT/'timing.json').write_text(json.dumps(timings,indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in
        ('protocol.json','evaluation.jsonl','report.json','timing.json')},indent=2))


if __name__=='__main__':run()
