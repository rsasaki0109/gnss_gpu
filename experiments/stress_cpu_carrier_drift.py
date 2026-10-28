"""Common linear frequency drift versus frozen constant-carrier duration fit."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
from gnss_gpu.duration_carrier_paths_cpu import temporal_echo,score_common_carrier_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from experiments.run_cpu_carrier_duration import synthesize as baseline
from experiments.run_cpu_common_carrier_selection import tasks,digest

OUTPUT=Path('output/cpu-carrier-drift-stress')
PRIOR=Path('output/cpu-carrier-duration')
RATES=(0.,.5,2.)


def chirp_blocks(frequency,rate,count):
    """Average exp(2pi i(f*t + .5*rate*t²)) over each20ms block."""
    if rate==0:return temporal_echo(frequency,count)[0]
    nodes,weights=np.polynomial.legendre.leggauss(24)
    t=np.arange(count)[:,None]*.020+(nodes+1)*.010
    return np.exp(2j*np.pi*(frequency*t+.5*rate*t*t))@(weights/2)


def synthesize(task):
    count,rate,offset,replica=task
    blocks,signal,delay=baseline((40,.1,offset,replica))
    if rate==0:return blocks[:count].copy(),signal[:count].copy(),delay
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44555241,replica]))
    assert float(rng.uniform(-.3,.3))==delay
    phase=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    direct=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*phase)
    changed=direct*(chirp_blocks(offset,rate,40)[:,None]-.8*chirp_blocks(offset+.1,rate,40)[:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    return (changed+(blocks-signal))[:count].copy(),changed[:count].copy(),delay


def evaluate(task):
    blocks,_,delay=synthesize(task)
    score,weights=score_common_carrier_paths(blocks)
    fit=score['two'];error=(fit.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if fit.locally_usable else None
    count,rate,offset,replica=task
    return dict(block_count=count,duration_ms=count*20,drift_hz_s=rate,initial_carrier_hz=offset,replica=replica,
        true_delay_chips=delay,fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),first10_sha256=hashlib.sha256(blocks[:10].tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for rate in RATES:
        short={r['replica']:r for r in rows if r['drift_hz_s']==rate and r['block_count']==10}
        for count in (10,40):
            group=[r for r in rows if r['drift_hz_s']==rate and r['block_count']==count]
            stable={r['replica']:r for r in rows if r['drift_hz_s']==0 and r['block_count']==count}
            cells.append(dict(drift_hz_s=rate,duration_ms=count*20,attempts=len(group),successes=sum(r['oracle_success'] for r in group),
                unavailable=sum(r['oracle_error_m'] is None for r in group),boundary=sum(r['fits']['two']['reason']=='search boundary' for r in group),
                nonconverged=sum(not r['fits']['two']['converged'] for r in group),
                gains_vs_200ms=sum(r['oracle_success'] and not short[r['replica']]['oracle_success'] for r in group),
                losses_vs_200ms=sum(short[r['replica']]['oracle_success'] and not r['oracle_success'] for r in group),
                gains_vs_zero_drift=sum(r['oracle_success'] and not stable[r['replica']]['oracle_success'] for r in group),
                losses_vs_zero_drift=sum(stable[r['replica']]['oracle_success'] and not r['oracle_success'] for r in group)))
    return dict(cells=cells,scope='Known-order paired model-mismatch pilot; not a drift-aware receiver or calibrated detector.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    evaluation=[(count,rate,offset,i) for rate in RATES for count in (10,40) for _,offset,i in tasks(.8,920000,16)]
    protocol=dict(evaluation_tasks=evaluation,source_sha256=sources,
        context='PRN1,white37dBHz,3.6828MHz,voltage.8,gap.1chip,initial phasepi,relative Doppler+.1Hz; initial common carrier uniform[-1,1]Hz.',
        intervention='Common frequency f(t)=f0+rate*t, rate0/.5/2Hz/s. Phase2pi(f0*t+.5*rate*t²), integrate using24-point Gauss-Legendre inside EACH20ms block for BOTH paths. Constant-carrier receiver frozen.',
        pairing='Fresh16IDs920000+ shared across3rates x2durations200/800ms. Exact prefixes per rate, same additive noise across rates. Zero drift bit-exact prior generator. Do not pool dependent cells.',
        reporting='Known two-path diagnostic only; no model-order gates or truth in fitting. Record failures; exploratory descriptive comparisons, no superiority test.',
        limits='Common residual oscillator-like drift, fixed code delays; NOT a physically complete accelerated receiver trajectory. No nav bits, stochastic phase noise, time-varying amplitudes, tracking, RF or PVT. Longer-duration search grid unchanged.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row in pool.map(evaluate,evaluation):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%16==0:stream.flush();print(f'Evaluation {len(rows)}/96',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
