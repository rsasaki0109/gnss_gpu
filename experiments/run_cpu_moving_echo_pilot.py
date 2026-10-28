"""Paired frozen-code versus consistent moving-code echo observations."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from experiments.stress_cpu_wideband_phase_doppler import synthesize as baseline
from experiments.run_cpu_drift_selection import tasks,digest
from experiments.stress_cpu_carrier_drift import chirp_blocks
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.moving_echo_cpu import moving_echo_blocks,range_rate_from_doppler
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED,ca_passband_spectrum

OUTPUT=Path('output/cpu-moving-echo-pilot')
PRIOR=Path('output/cpu-wideband-phase-doppler')
DOPPLERS=(0.,.7,7.,-7.)


def tasklist():
    return [(mode,fd,a,f,i) for mode in ('fixed_code','moving_code') for fd in DOPPLERS
            for _,a,f,i in tasks('echo',1330000,16)]


def synthesize(task):
    mode,fd,drift,offset,replica=task
    if mode not in ('fixed_code','moving_code'):raise ValueError('unknown mode')
    b,old,truth=baseline((49.,fd,np.pi,drift,offset,replica))
    if mode=='fixed_code' or fd==0:return b,old,truth
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x42414E44,replica]))
    assert float(rng.uniform(-.3,.3))==truth
    common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(7.2*CHIP_RATE);rate=range_rate_from_doppler(fd)
    assert .03+min(0.,rate*.8*CHIP_RATE/LIGHT_SPEED)>0
    direct=s*np.exp(-2j*np.pi*f*truth/CHIP_RATE+1j*common)
    moving=moving_echo_blocks(f,offset,drift,rate,40)
    signal=direct*(chirp_blocks(offset,drift,40)[:,None]-.8*moving*np.exp(-2j*np.pi*f*.03/CHIP_RATE))
    return signal+(b-old),signal,truth


def evaluate(task):
    b,_,truth=synthesize(task);score,w=score_drift_paths(b,7.2*CHIP_RATE)
    fit=score['two'];error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    mode,fd,drift,offset,replica=task
    rate=range_rate_from_doppler(fd) if mode=='moving_code' else 0.
    mid=.03+rate*.4*CHIP_RATE/LIGHT_SPEED;end=.03+rate*.8*CHIP_RATE/LIGHT_SPEED
    return dict(task=task,mode=mode,relative_doppler_hz=fd,replica=replica,true_delay_chips=truth,
        excess_range_rate_m_s=rate,excess_midpoint_chips=mid,excess_end_chips=end,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        raw_error_m=error,gap_midpoint_error_m=(fit.excess_chips-mid)*LIGHT_SPEED/CHIP_RATE,
        available=fit.locally_usable,success=bool(fit.locally_usable and abs(error)<10),
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def summarize(rows):
    cells=[];paired=[]
    for mode in ('fixed_code','moving_code'):
        for fd in DOPPLERS:
            g=[r for r in rows if r['mode']==mode and r['relative_doppler_hz']==fd];n=len(g);k=sum(r['success'] for r in g)
            cells.append(dict(mode=mode,relative_doppler_hz=fd,attempts=n,successes=k,
                success_exact_95=list(binomtest(k,n).proportion_ci(method='exact')),unavailable=sum(not r['available'] for r in g),
                available_large_errors=sum(r['available'] and abs(r['raw_error_m'])>=10 for r in g),
                median_absolute_raw_error_m=float(np.median([abs(r['raw_error_m']) for r in g]))))
    for fd in DOPPLERS:
        a={r['replica']:r for r in rows if r['mode']=='fixed_code' and r['relative_doppler_hz']==fd}
        b={r['replica']:r for r in rows if r['mode']=='moving_code' and r['relative_doppler_hz']==fd}
        assert a.keys()==b.keys()
        paired.append(dict(relative_doppler_hz=fd,gains=sum(b[i]['success'] and not a[i]['success'] for i in a),
            losses=sum(a[i]['success'] and not b[i]['success'] for i in a),
            max_absolute_range_estimate_change_m=max(abs(b[i]['raw_error_m']-a[i]['raw_error_m']) for i in a)))
    return dict(cells=cells,paired_changes=paired,
        scope='Exploratory16 shared replicas,known two-path order,49dBHz,800ms. Existing fixed-code receiver unchanged. Midpoint gap is descriptive,not the truth target for direct delay. No significance test or pooled independent sample claim.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__,'python/gnss_gpu/moving_echo_cpu.py'):sources[str(Path(path).resolve())]=digest(path)
    assert all(digest(path)==h for path,h in sources.items())
    p=dict(source_sha256=sources,tasks=tasklist(),
        design='16 fresh shared replicas x fixed/moving code x relativeDoppler(0,.7,7,-7)Hz.7.3656MHz one-sided,800ms,49dBHz,PRN1,voltage.8,initial excess.03chip,phasepi. Direct path fixed;common frequency/drift treated as oscillator terms.',
        motion='Excess path L(t)=L0+v*t;v=-Doppler*c/L1frequency. Echo code delay also changes byv*t/c.24-point block quadrature using effective harmonic frequency common-(L1+f)*v/c. Noise paired with frozen-code approximation;zero speed bit-exact baseline.',
        evaluation='Known two-path order with unchanged fixed-code receiver and guards. No truth passed to fitter,no calibration gate reused. Pointwise intervals and paired descriptive counts only.',
        limits='Consistent code/carrier phase within existing harmonic observation approximation;no inter-bin leakage or full RF/time-domain channel. Fixed path amplitude,linear excess length,positive excess throughout. Not forest/rain/geometry/PVT.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,p['tasks']):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if len(rows)%16==0:print(f'Moving echo {len(rows)}/128',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','evaluation.jsonl','report.json')},indent=2))


if __name__=='__main__':run()
