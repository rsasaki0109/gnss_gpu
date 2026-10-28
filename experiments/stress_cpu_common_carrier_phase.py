"""Frozen receiver sensitivity to relative Doppler and reflection phase."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.dynamic_multipath_cpu import temporal_echo
from gnss_gpu.common_carrier_paths_cpu import score_common_carrier_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate
from experiments.stress_cpu_residual_carrier import synthesize as baseline
from experiments.run_cpu_common_carrier_selection import tasks,apply_gate,digest

OUTPUT=Path('output/cpu-common-carrier-phase-stress')
PRIOR=Path('output/cpu-common-carrier-selection')
CONDITIONS=[(fd,phase) for fd in (0.,.1,.7) for phase in (0.,np.pi/2,np.pi)]


def synthesize(task):
    ratio,offset,fd,phase,replica=task
    blocks,old_signal,delay=baseline((ratio,offset,replica))
    if (fd==.7 and phase==0) or ratio==0:return blocks,old_signal,delay
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x44594741,replica]))
    assert float(rng.uniform(-.3,.3))==delay
    common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    direct=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)
    signal=direct*(temporal_echo(offset)[0][:,None]+ratio*np.exp(1j*phase)*
        temporal_echo(offset+fd)[0][:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    return signal+(blocks-old_signal),signal,delay


def evaluate(task):
    blocks,_,delay=synthesize(task)
    score,weights=score_common_carrier_paths(blocks)
    ratio,offset,fd,phase,replica=task
    oracle=score['two' if ratio else 'single']
    error=(oracle.first_delay_chips-delay)*LIGHT_SPEED/CHIP_RATE if oracle.locally_usable else None
    return dict(echo_voltage=ratio,residual_carrier_hz=offset,relative_doppler_hz=fd,phase_rad=phase,
        replica=replica,true_delay_chips=delay,fits={k:asdict(score[k]) for k in ('single','two','preliminary')},
        uncalibrated_score=score['score'],oracle_error_m=error,oracle_success=error is not None and abs(error)<10,
        observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),weights_sha256=hashlib.sha256(weights.tobytes()).hexdigest())


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['echo_voltage'],r['relative_doppler_hz'],r['phase_rad']),[]).append(r)
    cells=[]
    for (ratio,fd,phase),group in groups.items():
        n=len(group);success=sum(r['success'] for r in group);two=sum(r['selected']['model']=='two_path' for r in group)
        cells.append(dict(echo_voltage=ratio,relative_doppler_hz=fd,phase_rad=phase,attempts=n,
            successes=success,success_exact_95=list(binomtest(success,n).proportion_ci(method='exact')),
            two_paths=two,two_path_exact_95=list(binomtest(two,n).proportion_ci(method='exact')),
            unavailable=sum(r['error_m'] is None for r in group),score_exceedances=sum(r['score_exceeds'] for r in group),
            oracle_successes=sum(r['oracle_success'] for r in group),oracle_unavailable=sum(r['oracle_error_m'] is None for r in group),
            gains_vs_oracle=sum(r['success'] and not r['oracle_success'] for r in group),
            losses_vs_oracle=sum(r['oracle_success'] and not r['success'] for r in group)))
    return dict(cells=cells,scope='Exploratory paired phase/Doppler cells, not independent pooled samples or simultaneous confidence intervals.')


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    gatepath=PRIOR/'frozen_gate.json';gatehash=digest(gatepath)
    gate=PathSelectionGate(**json.loads(gatepath.read_text()))
    null=[(r,o,.7,0.,i) for r,o,i in tasks(0.,870000,32)]
    echo=[(r,o,fd,phase,i) for fd,phase in CONDITIONS for r,o,i in tasks(.8,880000,16)]
    protocol=dict(evaluation_tasks=null+echo,source_sha256=sources,gate_path=str(gatepath),gate_sha256=gatehash,gate=asdict(gate),
        context='Frozen common-carrier fit/selector. PRN1,white37dBHz,10x20ms,3.6828MHz; common carrier uniform[-1,1]Hz. Echo voltage.8, excess.1chip.',
        intervention='Relative Doppler0/.1/.7Hz x initial relative phase0/pi2/pi. Both path block coefficients integrated at absolute carrier frequencies. Same delay/common carrier/common phase/additive noise across nine echo conditions.',
        pairing='Fresh32 null IDs870000+, fresh16 echo IDs880000+ shared across9cells. Do not pool dependent cells.',
        inference='Exploratory descriptive pointwise exact95 intervals; no superiority tests or threshold tuning. Null control is small smoke check, not1% certification.',
        limits='Synthetic narrowband fixed delay, no RF/loop/nav/PVT/forest/rain; initial reverse phase at nonzero relative Doppler does not imply constant cancellation.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,null+echo):
            row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%16==0:stream.flush();print(f'Evaluation {len(rows)}/176',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),gate_sha256=gatehash,
        evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))


if __name__=='__main__':run()
