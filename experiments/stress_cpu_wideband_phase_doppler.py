"""Paired phase/relative-Doppler stress of the frozen wideband receiver."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from experiments.run_cpu_wideband_cn0 import synthesize as baseline
from experiments.run_cpu_drift_selection import tasks,digest
from experiments.stress_cpu_carrier_drift import chirp_blocks
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED,ca_passband_spectrum

OUTPUT=Path('output/cpu-wideband-phase-doppler')
PRIOR=Path('output/cpu-wideband-cn0')
PHASES=(0.,np.pi/2,np.pi)
FREQUENCIES=(0.,.7)
LEVELS=(37.,49.)


def tasklist():
    return [(cn0,fd,phase,a,f,i) for cn0 in LEVELS for fd in FREQUENCIES for phase in PHASES
            for _,a,f,i in tasks('echo',1310000,16)]


def synthesize(task):
    cn0,fd,phase,drift,offset,replica=task
    b,old,truth=baseline(('echo',cn0,drift,offset,replica))
    if fd==0 and phase==np.pi:return b,old,truth
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x42414E44,replica]))
    assert float(rng.uniform(-.3,.3))==truth
    common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(7.2*CHIP_RATE)
    direct=s*np.exp(-2j*np.pi*f*truth/CHIP_RATE+1j*common)
    signal=direct*(chirp_blocks(offset,drift,40)[:,None]+.8*np.exp(1j*phase)*
        chirp_blocks(offset+fd,drift,40)[:,None]*np.exp(-2j*np.pi*f*.03/CHIP_RATE))
    return signal+(b-old),signal,truth


def evaluate(task):
    b,_,truth=synthesize(task);score,w=score_drift_paths(b,7.2*CHIP_RATE)
    fit=score['two'];error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    return dict(task=task,cn0_dbhz=task[0],relative_doppler_hz=task[1],phase_rad=task[2],replica=task[-1],true_delay_chips=truth,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        raw_error_m=error,available=fit.locally_usable,success=bool(fit.locally_usable and abs(error)<10),
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def summarize(rows):
    cells=[];paired=[]
    for cn0 in LEVELS:
        for fd in FREQUENCIES:
            for phase in PHASES:
                g=[r for r in rows if r['cn0_dbhz']==cn0 and r['relative_doppler_hz']==fd and r['phase_rad']==phase]
                n=len(g);success=sum(r['success'] for r in g)
                cells.append(dict(cn0_dbhz=cn0,relative_doppler_hz=fd,phase_rad=phase,attempts=n,successes=success,
                    success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
                    unavailable=sum(not r['available'] for r in g),available_large_errors=sum(r['available'] and abs(r['raw_error_m'])>=10 for r in g),
                    boundary=sum(r['fits']['two']['reason']=='search boundary' for r in g),
                    nonconverged=sum(not r['fits']['two']['converged'] for r in g)))
    for fd in FREQUENCIES:
        for phase in PHASES:
            matched={level:{r['replica']:r for r in rows if r['cn0_dbhz']==level and r['relative_doppler_hz']==fd and r['phase_rad']==phase} for level in LEVELS}
            assert matched[37].keys()==matched[49].keys()
            paired.append(dict(relative_doppler_hz=fd,phase_rad=phase,pairs=len(matched[37]),
                gains=sum(matched[49][i]['success'] and not matched[37][i]['success'] for i in matched[37]),
                losses=sum(matched[37][i]['success'] and not matched[49][i]['success'] for i in matched[37])))
    return dict(cells=cells,cn0_paired_changes=paired,
        scope='Exploratory12 cells sharing16 replicas;no pooled independence,hypothesis tests,or simultaneous interval guarantee. Known two-path order,not a calibrated detector. Fixed code delay and changing relative carrier phase is an approximation,not a moving scatterer model.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(path)==h for path,h in sources.items())
    p=dict(source_sha256=sources,tasks=tasklist(),
        design='16 fresh replicas shared across C/N0(37,49)dBHz x relativeDoppler(0,.7)Hz x initialphase(0,pi/2,pi).7.3656MHz one-sided,800ms,PRN1,voltage.8,gap.03chip. Common carrier/drift distributions unchanged. Same direct signal and standardized noise across cells.',
        receiver='Frozen wideband pipeline and guards. Known two-path order,no model-selection or alarm gate. No generating phase/Doppler given to optimizer.',
        inference='Exploratory pointwise exact95 intervals and paired gain/loss counts;no hypothesis tests or cell pooling. Do not interpret all16 successes as perfect population accuracy.',
        limits='Synthetic relative carrier phase evolution with fixed code delay;not dynamically consistent moving geometry. No RF/forest/rain/loops/PVT. At.7Hz phase rotates201.6degrees in.8s.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,p['tasks']):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if len(rows)%16==0:print(f'Phase/Doppler {len(rows)}/192',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','evaluation.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
