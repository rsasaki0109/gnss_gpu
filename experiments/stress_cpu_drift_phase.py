"""Frozen cached drift-aware receiver across reflection phase/relative Doppler."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.cached_drift_carrier_paths_cpu import score_drift_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.run_cpu_drift_selection import tasks,apply_gate,digest,null_signal,echo_signal
from experiments.stress_cpu_carrier_drift import chirp_blocks

OUTPUT=Path('output/cpu-drift-phase-stress')
PRIOR=Path('output/cpu-drift-selection')
CACHE=Path('output/cpu-drift-cache-benchmark')
CONDITIONS=[(fd,phase) for fd in (0.,.1,.7) for phase in (0.,np.pi/2,np.pi)]


def synthesize(task):
    kind,fd,phase,drift,offset,replica=task
    original,old_signal,delay=(null_signal if kind=='null' else echo_signal)((40,drift,offset,replica))
    if kind=='null' or (fd==.1 and phase==np.pi):return original,old_signal,delay
    rng=np.random.default_rng(np.random.SeedSequence([20261027,0x44555241,replica]))
    assert float(rng.uniform(-.3,.3))==delay
    common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    direct=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)
    signal=direct*(chirp_blocks(offset,drift,40)[:,None]+.8*np.exp(1j*phase)*
        chirp_blocks(offset+fd,drift,40)[:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    return signal+(original-old_signal),signal,delay


def evaluate(task):
    blocks,_,delay=synthesize(task);score,weights=score_drift_paths(blocks)
    kind,fd,phase,drift,offset,replica=task
    fit=score['single' if kind=='null' else 'two']
    error=(fit.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if fit.locally_usable else None
    return dict(kind=kind,relative_doppler_hz=fd,phase_rad=phase,drift_hz_s=drift,initial_carrier_hz=offset,replica=replica,
        true_delay_chips=delay,fits={k:asdict(score[k]) for k in ('single','two','preliminary')},uncalibrated_score=score['score'],
        oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['kind'],r['relative_doppler_hz'],r['phase_rad']),[]).append(r)
    cells=[]
    for (kind,fd,phase),g in groups.items():
        n=len(g);two=sum(r['selected']['model']=='two_path' for r in g);success=sum(r['success'] for r in g)
        cells.append(dict(kind=kind,relative_doppler_hz=fd,phase_rad=phase,attempts=n,
            two_paths=two,two_path_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
            successes=success,success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
            unavailable=sum(r['error_m'] is None for r in g),score_exceedances=sum(r['score_exceeds'] for r in g),
            oracle_successes=sum(r['oracle_success'] for r in g),oracle_unavailable=sum(r['oracle_error_m'] is None for r in g),
            gains_vs_oracle=sum(r['success'] and not r['oracle_success'] for r in g),losses_vs_oracle=sum(r['oracle_success'] and not r['success'] for r in g)))
    return dict(cells=cells,scope='Exploratory16paired echo replicas per cell; pointwise intervals, no superiority tests or pooled independence.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((CACHE/'protocol.json').read_text())['source_sha256']);sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    gatepath=PRIOR/'frozen_gate.json';gatehash=digest(gatepath);gate=PathSelectionGate(**json.loads(gatepath.read_text()))
    evidence={str(CACHE/name):digest(CACHE/name) for name in ('report.json','audit.json','completion.json')}
    null=[(kind,.1,np.pi,a,f,i) for kind,a,f,i in tasks('null',1020000,32)]
    echo=[(kind,fd,phase,a,f,i) for fd,phase in CONDITIONS for kind,a,f,i in tasks('echo',1030000,16)]
    protocol=dict(source_sha256=sources,cache_evidence_sha256=evidence,gate_path=str(gatepath),gate_sha256=gatehash,gate=asdict(gate),evaluation_tasks=null+echo,
        context='Frozen cached drift-aware receiver,800ms,white37dBHz,PRN1,3.6828MHz; drift uniform[-2,2]Hz/s and initial carrier uniform[-1,1]Hz. Echo voltage.8,gap.1chip.',
        intervention='Relative Doppler0/.1/.7Hz x initial relative phase0/pi2/pi. Shared16echo IDs across9cells with same drift/carrier/delay/common phase/additive noise. Fresh32null controls, disjoint IDs.',
        cache='Prior full395-observation exact replay supports same gate in this numerical environment. Frozen original source hashes checked. Ten-case fresh baseline replay in audit provides additional check.',
        inference='Exploratory descriptive cellwise exact95 intervals, not simultaneous. Do not pool shared replicas. No threshold or receiver retuning.',
        limits='Fixed amplitudes/code delays, synthetic common linear drift, no nav/loop/realRF/PVT/forest/rain. Initial phase at nonzero relative Doppler evolves across800ms.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,null+echo):
            row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%16==0:stream.flush();print(f'Evaluation {len(rows)}/176',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in {**sources,**evidence}.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),gate_sha256=gatehash,
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
