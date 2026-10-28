"""Freeze a dynamic-path null gate, then evaluate null and echo panels."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.dynamic_multipath_cpu import DynamicFit,temporal_echo
from gnss_gpu.calibrated_dynamic_paths_cpu import DynamicScore,score_dynamic_paths,select_dynamic_paths
from gnss_gpu.calibrated_path_selection_cpu import calibrate_path_gate,PathSelectionGate
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from experiments.run_cpu_dynamic_path_fit import digest

OUTPUT=Path('output/cpu-dynamic-selection')
CONDITIONS=[(fd,phase) for fd in (0.,.1,.7,2.3,6.7) for phase in (0.,np.pi/2,np.pi)]


def evaluate(task):
    ratio,fd,phase,replica=task
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x44594741,replica]))
    delay=float(rng.uniform(-.3,.3));common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    signal=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*(1+ratio*np.exp(1j*phase)*temporal_echo(fd)[0][:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    blocks=signal+(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2*10**3.7*.020)
    score=score_dynamic_paths(blocks)
    return dict(echo_voltage=ratio,doppler_hz=fd,phase_rad=phase,replica=replica,true_delay_chips=delay,
        score=asdict(score),observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest())


def select(row,gate):
    score=DynamicScore(**dict(row['score'],two_path=DynamicFit(**row['score']['two_path'])))
    chosen=select_dynamic_paths(score,gate)
    error=None if chosen['delay_chips'] is None else (chosen['delay_chips']-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE
    forced_error=(score.two_path.first_delay_chips-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE if score.two_path.locally_usable else None
    return dict(row,selected=chosen,error_m=error,success=error is not None and abs(error)<10,
        forced_error_m=forced_error,forced_success=forced_error is not None and abs(forced_error)<10,
        score_exceeds=score.bic_improvement>gate.score)


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['echo_voltage'],r['doppler_hz'],r['phase_rad']),[]).append(r)
    cells=[]
    for (ratio,fd,phase),group in groups.items():
        n=len(group);two=sum(r['selected']['model']=='two_path' for r in group)
        cells.append(dict(echo_voltage=ratio,doppler_hz=fd,phase_rad=phase,attempts=n,
            successes=sum(r['success'] for r in group),forced_successes=sum(r['forced_success'] for r in group),
            two_paths=two,two_path_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
            forced_locally_usable=sum(r['score']['two_path']['locally_usable'] for r in group),
            score_exceedances=sum(r['score_exceeds'] for r in group),unavailable=sum(r['error_m'] is None for r in group),
            gains=sum(r['success'] and not r['forced_success'] for r in group),
            losses=sum(r['forced_success'] and not r['success'] for r in group)))
    return dict(cells=cells)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads(Path('output/cpu-dynamic-path-fit/protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/calibrated_dynamic_paths_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    cal=[(0.,0.,0.,540000+r) for r in range(299)]
    null=[(0.,0.,0.,550000+r) for r in range(299)]
    echoes=[(.8,fd,phase,560000+r) for fd,phase in CONDITIONS for r in range(16)]
    protocol=dict(calibration_tasks=cal,evaluation_tasks=null+echoes,source_sha256=sources,
        context='PRN1,37dB-Hz white Gaussian noise,200ms,3.6828MHz one-sided,static wiped direct path; delay uniform[-.3,.3],common phase uniform[-pi,pi].',
        score='n*log(single full-block SSE/two-path SSE)-4*log(n), n=147300; same normalization; nuisance Doppler unidentified under null, so no chi-square/BIC posterior interpretation.',
        calibration='299 iid null scores; strict threshold above maximum; tail budget .01, calibration confidence .95 under this fixed null distribution.',
        scope='Guarantee conditional on generating distribution, not uniform in C/N0/delay or arbitrary noise. Final two-path selection is subset of score exceedances.',
        evaluation='299 independent heldout null; five Dopplers including0/.1Hz xthree phases x16 shared replicas. All gates frozen before evaluation.',
        comparison='Calibrated choice vs forced two-path + unchanged local guards; echo panel exploratory, no superiority tests. Do not pool dependent conditions as independent.',
        limits='No measured data, geometry/time-varying code delay, residual direct carrier, or colored-noise guarantee.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        calibration=[]
        with (OUTPUT/'calibration.jsonl').open('w') as stream:
            for row in pool.map(evaluate,cal):
                calibration.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(calibration)%50==0:stream.flush();print(f'Calibration {len(calibration)}/299',flush=True)
        gate=calibrate_path_gate([r['score']['bic_improvement'] for r in calibration])
        gatepath=OUTPUT/'frozen_gate.json';gatepath.write_text(json.dumps(asdict(gate),indent=2));gatehash=digest(gatepath)
        print(f'Frozen gate {gate.score:.6f}',flush=True)
        rows=[]
        with (OUTPUT/'evaluation.jsonl').open('w') as stream:
            for raw in pool.map(evaluate,null+echoes):
                row=select(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                if len(rows)%50==0:stream.flush();print(f'Evaluation {len(rows)}/539',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(calibration=299,evaluation=539,gate_sha256=gatehash,
        calibration_sha256=digest(OUTPUT/'calibration.jsonl'),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
