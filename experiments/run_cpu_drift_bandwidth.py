"""Paired bandwidth pilot with fixed prefilter C/N0 and noise spectral density."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.bandwidth_drift_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import CHIP_RATE,LIGHT_SPEED,ca_passband_spectrum
from experiments.stress_cpu_carrier_drift import chirp_blocks
from experiments.run_cpu_drift_selection import tasks,digest

OUTPUT=Path('output/cpu-drift-bandwidth')
PRIOR=Path('output/cpu-dominant-path-pilot')
BANDS=(1.8,3.6,7.2)


def tasklist():
    return [(band,a,f,i) for band in BANDS for _,a,f,i in tasks('echo',1240000,16)]


def synthesize(task):
    band,drift,offset,replica=task
    if band not in BANDS:raise ValueError('unsupported pilot bandwidth')
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x42414E44,replica]))
    delay=float(rng.uniform(-.3,.3));phase=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(max(BANDS)*CHIP_RATE)
    signal=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*phase)*chirp_blocks(offset,drift,40)[:,None]*(1-.8*np.exp(-2j*np.pi*f*.03/CHIP_RATE))
    noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2*10**3.7*.020)
    keep=np.abs(f)<=band*CHIP_RATE
    return (signal+noise)[:,keep].copy(),signal[:,keep].copy(),delay


def evaluate(task):
    b,_,truth=synthesize(task);score,w=score_drift_paths(b,task[0]*CHIP_RATE)
    fit=score['two'];error=(fit.first_delay_chips-truth)*LIGHT_SPEED/CHIP_RATE
    return dict(task=task,bandwidth_chips=task[0],replica=task[-1],true_delay_chips=truth,
        fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        raw_error_m=error,success=bool(fit.locally_usable and abs(error)<10),available=fit.locally_usable,
        observation_sha256=hashlib.sha256(b.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(w.tobytes()).hexdigest())


def summarize(rows):
    baseline={r['replica']:r for r in rows if r['bandwidth_chips']==3.6}
    cells=[]
    for band in BANDS:
        g=[r for r in rows if r['bandwidth_chips']==band];n=len(g);success=sum(r['success'] for r in g)
        f,s=ca_passband_spectrum(band*CHIP_RATE)
        cells.append(dict(bandwidth_chips=band,one_sided_bandwidth_hz=band*CHIP_RATE,harmonics=len(f),
            retained_direct_power=float(np.vdot(s,s).real),complex_observations=40*len(f),
            expected_block_noise_energy=len(f)/(10**3.7*.020),attempts=n,successes=success,
            success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
            unavailable=sum(not r['available'] for r in g),
            gains_vs_3_6=sum(r['success'] and not baseline[r['replica']]['success'] for r in g),
            losses_vs_3_6=sum(not r['success'] and baseline[r['replica']]['success'] for r in g)))
    return dict(cells=cells,scope='Known-two-path pilot; no gate reuse. Identical IQ on overlapping harmonics; fixed prefilter direct37dBHz and noise PSD,not fixed total passband noise. No passband signal renormalization.16 paired replicas,not48 independent cases.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for path in (__file__,'python/gnss_gpu/bandwidth_drift_paths_cpu.py'):sources[str(Path(path).resolve())]=digest(path)
    assert all(digest(p)==h for p,h in sources.items())
    ts=tasklist()
    p=dict(source_sha256=sources,tasks=ts,
        design='One-sided bandwidth1.8/3.6/7.2 chip-rate;16 shared fresh replicas.800ms,PRN1,prefilter direct37dBHz,echo voltage.8,gap.03chip,phasepi,relative Doppler0. Random common carrier/drift as previous studies. Wideband IQ generated once per seed; narrower bands exact harmonic subsets.',
        noise='Constant per-harmonic proper complex noise variance1/(C/N0*.020). Wider band admits more signal and more total noise;no signal renormalization.',
        receiver='Separate bandwidth-generalized copy of frozen cached receiver,all searches/guards/smoothing widths unchanged. Known two-path order;no model-selection or warning gate applied.',
        timing='Separate sequential full pipeline calls;warm once per band. Two replicas per band in forward/reverse band order. Input generation and output checks outside timing. Descriptive six timings,not a performance guarantee.')
    (OUTPUT/'protocol.json').write_text(json.dumps(p,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for r in pool.map(evaluate,ts):
            rows.append(r);stream.write(json.dumps(r,allow_nan=False)+'\n');stream.flush()
            if len(rows)%8==0:print(f'Bandwidth evaluation {len(rows)}/48',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    timings=[]
    for band in BANDS:
        t=next(t for t in ts if t[0]==band);b,_,_=synthesize(t);score_drift_paths(b,band*CHIP_RATE)
    for replica_index in (0,1):
        for band in BANDS if replica_index==0 else BANDS[::-1]:
            t=next(t for t in ts if t[0]==band and t[-1]==1240000+replica_index)
            b,_,_=synthesize(t);start=perf_counter();s,w=score_drift_paths(b,band*CHIP_RATE);elapsed=perf_counter()-start
            row=next(r for r in rows if r['task']==t)
            assert {k:asdict(s[k]) for k in ('single','two','preliminary')}==row['fits']
            assert hashlib.sha256(w.tobytes()).hexdigest()==row['weights_sha256']
            timings.append(dict(bandwidth_chips=band,replica=t[-1],elapsed_s=elapsed))
    (OUTPUT/'timings.json').write_text(json.dumps(timings,indent=2))
    assert all(digest(path)==h for path,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps({name:digest(OUTPUT/name) for name in ('protocol.json','evaluation.jsonl','report.json','timings.json')},indent=2))


if __name__=='__main__':run()
