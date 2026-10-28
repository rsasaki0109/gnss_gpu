"""Paired fresh-data duration pilot, known-order diagnostic only."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from gnss_gpu.duration_carrier_paths_cpu import temporal_echo,score_common_carrier_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from experiments.run_cpu_common_carrier_selection import digest,tasks

OUTPUT=Path('output/cpu-carrier-duration')
PRIOR=Path('output/cpu-common-carrier-phase-stress')


def synthesize(task):
    count,fd,offset,replica=task
    if count not in (10,20,40):raise ValueError('count must be10/20/40')
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44555241,replica]))
    delay=float(rng.uniform(-.3,.3));phase=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    direct=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*phase)
    signal=direct*(temporal_echo(offset,40)[0][:,None]-.8*temporal_echo(offset+fd,40)[0][:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2*10**3.7*.020)
    return (signal+noise)[:count].copy(),signal[:count].copy(),delay


def evaluate(task):
    blocks,_,delay=synthesize(task)
    score,weights=score_common_carrier_paths(blocks)
    fit=score['two'];error=(fit.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if fit.locally_usable else None
    count,fd,offset,replica=task
    return dict(block_count=count,duration_ms=count*20,relative_doppler_hz=fd,residual_carrier_hz=offset,replica=replica,
        true_delay_chips=delay,fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),first10_sha256=hashlib.sha256(blocks[:10].tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for fd in (0.,.1,.7):
        base={r['replica']:r for r in rows if r['relative_doppler_hz']==fd and r['block_count']==10}
        for count in (10,20,40):
            g=[r for r in rows if r['relative_doppler_hz']==fd and r['block_count']==count]
            cells.append(dict(relative_doppler_hz=fd,duration_ms=count*20,attempts=len(g),successes=sum(r['oracle_success'] for r in g),
                unavailable=sum(r['oracle_error_m'] is None for r in g),boundary=sum(r['fits']['two']['reason']=='search boundary' for r in g),
                nonconverged=sum(not r['fits']['two']['converged'] for r in g),
                gains_vs_200ms=sum(r['oracle_success'] and not base[r['replica']]['oracle_success'] for r in g),
                losses_vs_200ms=sum(base[r['replica']]['oracle_success'] and not r['oracle_success'] for r in g)))
    return dict(cells=cells,scope='Fresh paired pilot, known two-path order, no calibrated detector.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/duration_carrier_paths_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    evaluation=[(count,fd,offset,i) for fd in (0.,.1,.7) for count in (10,20,40) for _,offset,i in tasks(.8,900000,8)]
    protocol=dict(evaluation_tasks=evaluation,source_sha256=sources,
        context='PRN1,white37dBHz,20ms blocks,3.6828MHz,voltage.8,gap.1chip,initial phasepi,common carrier uniform[-1,1]Hz.',
        duration='10/20/40 blocks (200/400/800ms). Generate full40 then take exact prefixes, fresh8IDs900000+ shared across durations and relative Doppler0/.1/.7Hz.',
        estimator='Copied frozen common-carrier estimator; only block count and temporal coefficients generalized. Same coarse seeds and bounds,12starts,max250iterations; estimated residual weights recomputed per duration.',
        reporting='Known two-path order only; compute both fits blindly, use two for diagnostic range and preserve all failures. No old gates. Scores uncalibrated per duration.',
        limits='Exploratory8paired replicas, not independent pooled72; no superiority test. Longer observation adds energy and phase evolution jointly, not isolation of either. Perfect coherence, fixed code delay/amplitudes, no nav transitions/oscillator drift/loop/PVT/weather. Fixed search grid may be inadequate at longer durations.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for row in pool.map(evaluate,evaluation):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%8==0:stream.flush();print(f'Evaluation {len(rows)}/72',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items())
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
