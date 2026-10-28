"""Calibrate800ms zero-drift null; stress both durations with null chirps."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.duration_carrier_paths_cpu import score_common_carrier_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,calibrate_path_gate
from experiments.stress_cpu_carrier_drift import chirp_blocks
from experiments.run_cpu_common_carrier_selection import tasks,digest,apply_gate

OUTPUT=Path('output/cpu-drift-null-selection')
PRIOR=Path('output/cpu-carrier-drift-stress')
OLD=Path('output/cpu-common-carrier-selection/frozen_gate.json')
RATES=(0.,.5,2.)


def synthesize(task):
    count,rate,offset,replica=task
    if count not in (10,40):raise ValueError('10 or40 blocks required')
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x4E554C44,replica]))
    delay=float(rng.uniform(-.3,.3));phase=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    signal=chirp_blocks(offset,rate,40)[:,None]*s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*phase)
    noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2*10**3.7*.020)
    return (signal+noise)[:count].copy(),signal[:count].copy(),delay


def evaluate(task):
    blocks,_,delay=synthesize(task);score,weights=score_common_carrier_paths(blocks)
    single=score['single'];error=(single.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if single.locally_usable else None
    count,rate,offset,replica=task
    return dict(block_count=count,duration_ms=count*20,drift_hz_s=rate,initial_carrier_hz=offset,replica=replica,
        true_delay_chips=delay,fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),first10_sha256=hashlib.sha256(blocks[:10].tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    cells=[]
    for count in (10,40):
        for rate in RATES:
            g=[r for r in rows if r['block_count']==count and r['drift_hz_s']==rate];n=len(g)
            two=sum(r['selected']['model']=='two_path' for r in g)
            cells.append(dict(duration_ms=count*20,drift_hz_s=rate,attempts=n,false_two_paths=two,
                false_two_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
                score_exceedances=sum(r['score_exceeds'] for r in g),successes=sum(r['success'] for r in g),
                unavailable=sum(r['error_m'] is None for r in g),oracle_successes=sum(r['oracle_success'] for r in g),
                oracle_unavailable=sum(r['oracle_error_m'] is None for r in g)))
    return dict(cells=cells,scope='Conditional zero-drift calibration; nonzero drift is mismatch, not covered by the guarantee.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    old=json.loads(OLD.read_text());oldhash=digest(OLD)
    calibration=[(40,0.,offset,i) for _,offset,i in tasks(0.,940000,299)]
    evaluation=[(count,rate,offset,i) for count in (10,40) for rate in RATES for _,offset,i in tasks(0.,950000,32)]
    protocol=dict(source_sha256=sources,old_gate_path=str(OLD),old_gate_sha256=oldhash,old_gate=old,
        calibration_tasks=calibration,evaluation_tasks=evaluation,
        context='NO reflection. PRN1,white37dBHz,3.6828MHz,20ms blocks,initial carrier uniform[-1,1]Hz,random delay and phase. Same generating law as prior200ms calibration, new independent RNG stream.',
        calibration='800ms:299 iid zero-drift nulls, strict maximum gate, marginal1%tail with failure bound .99**299<.05. 200ms:preserve prior299-null gate. Per-duration guarantees, NOT joint95 across both (union bound <.10).',
        selection='Frozen duration estimator and guarded common-carrier selector. No truth passed; invalid requested fit stays ambiguous.',
        evaluation='Fresh32IDs shared across two durations and drift0/.5/2Hz/s. Exact temporal prefixes; same noise across rates. Gate frozen before evaluating any stress task.',
        scope='Zero-drift guarantee does not cover nonzero drift. Exploratory pointwise exact95 intervals; no superiority tests, do not pool paired conditions. No recalibration on stress outcomes.',
        limits='Synthetic oscillator-like common chirp; fixed code delay, no geometric acceleration, nav bits, stochastic phase noise, tracking, realRF, weather or PVT.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        cal=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,calibration):
                cal.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(cal)%50==0:stream.flush();print(f'Calibration {len(cal)}/299',flush=True)
        gate=calibrate_path_gate([r['uncalibrated_score'] for r in cal]);gates={'10':old,'40':asdict(gate)}
        gatepath=OUTPUT/'frozen_gates.json';gatepath.write_text(json.dumps(gates,indent=2));gatehash=digest(gatepath)
        print(f'Frozen800ms gate: {gate.score}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,evaluation):
                row=apply_gate(raw,PathSelectionGate(**gates[str(raw['block_count'])]));rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%32==0:stream.flush();print(f'Evaluation {len(rows)}/192',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(OLD)==oldhash and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=299,evaluation=192,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
