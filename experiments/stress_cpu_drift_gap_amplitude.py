"""Frozen drift-aware receiver: near-path spacing and reflection amplitude."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.cached_drift_carrier_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_drift_selection import tasks,apply_gate,digest
from experiments.stress_cpu_drift_phase import synthesize as baseline
from experiments.stress_cpu_carrier_drift import chirp_blocks

OUTPUT=Path('output/cpu-drift-gap-amplitude')
PRIOR=Path('output/cpu-drift-phase-stress')
GATE=Path('output/cpu-drift-selection/frozen_gate.json')
CONDITIONS=[(ratio,gap,phase) for ratio in (.2,.8) for gap in (.03,.1,.3) for phase in (0.,np.pi)]


def synthesize(task):
    kind,ratio,gap,phase,drift,offset,replica=task
    b,old,delay=baseline((kind,0.,phase,drift,offset,replica))
    if kind=='null' or (ratio==.8 and gap==.1):return b,old,delay
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44555241,replica]))
    assert float(rng.uniform(-.3,.3))==delay
    common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    signal=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*chirp_blocks(offset,drift,40)[:,None]*(1+ratio*np.exp(1j*phase)*np.exp(-2j*np.pi*f*gap/CHIP_RATE))
    return signal+(b-old),signal,delay


def evaluate(task):
    blocks,_,delay=synthesize(task);score,weights=score_drift_paths(blocks)
    kind,ratio,gap,phase,drift,offset,replica=task
    fit=score['single' if kind=='null' else 'two']
    error=(fit.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if fit.locally_usable else None
    return dict(kind=kind,echo_voltage=ratio,excess_chips=gap,phase_rad=phase,drift_hz_s=drift,initial_carrier_hz=offset,replica=replica,
        true_delay_chips=delay,fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['kind'],r['echo_voltage'],r['excess_chips'],r['phase_rad']),[]).append(r)
    cells=[]
    for (kind,ratio,gap,phase),g in groups.items():
        n=len(g);two=sum(r['selected']['model']=='two_path' for r in g);success=sum(r['success'] for r in g)
        cells.append(dict(kind=kind,echo_voltage=ratio,excess_chips=gap,phase_rad=phase,attempts=n,
            two_paths=two,two_path_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
            successes=success,success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
            unavailable=sum(r['error_m'] is None for r in g),score_exceedances=sum(r['score_exceeds'] for r in g),
            oracle_successes=sum(r['oracle_success'] for r in g),oracle_unavailable=sum(r['oracle_error_m'] is None for r in g),
            gains_vs_oracle=sum(r['success'] and not r['oracle_success'] for r in g),losses_vs_oracle=sum(r['oracle_success'] and not r['success'] for r in g)))
    return dict(cells=cells,scope='Exploratory paired cells; path detection and range success are distinct endpoints, no pooled independence.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    gatehash=digest(GATE);gate=PathSelectionGate(**json.loads(GATE.read_text()))
    null=[(kind,0.,.1,0.,a,f,i) for kind,a,f,i in tasks('null',1050000,32)]
    echo=[(kind,ratio,gap,phase,a,f,i) for ratio,gap,phase in CONDITIONS for kind,a,f,i in tasks('echo',1060000,16)]
    protocol=dict(source_sha256=sources,gate_path=str(GATE),gate_sha256=gatehash,gate=asdict(gate),evaluation_tasks=null+echo,
        context='Frozen cached drift-aware receiver,800ms,PRN1,white37dBHz,3.6828MHz. Drift uniform[-2,2]Hz/s,initial carrier uniform[-1,1]Hz. Relative Doppler0Hz.',
        intervention='Echo voltage.2/.8 xgap.03/.1/.3chip xinitial phase0/pi. Same16echo replicas shared across12cells; same direct path/drift/carrier/noise.32fresh nulls. No true gap/amplitude given to estimator.',
        inference='Exploratory pointwise exact95 intervals, no hypothesis tests or retuning. Do not pool dependent cells. False-negative reflection detection does not necessarily imply >10m range error.',
        limits='Two fixed paths,white noise,common linear drift,ideal nav wipeoff; no changing geometry/amplitude,RF,loop,PVT,forest/rain.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,null+echo):
            row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%16==0:stream.flush();print(f'Evaluation {len(rows)}/224',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(GATE)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),gate_sha256=gatehash,
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
