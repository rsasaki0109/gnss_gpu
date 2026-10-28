"""Frozen dynamic path score/gates under controlled spectral noise mismatch."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.dynamic_multipath_cpu import temporal_echo
from gnss_gpu.calibrated_dynamic_paths_cpu import score_dynamic_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape
from experiments.run_cpu_dynamic_ranked_gate import apply_gates,digest
from experiments.run_cpu_dynamic_selection import summarize as summarize_single

OUTPUT=Path('output/cpu-dynamic-noise-stress')
PRIOR=Path('output/cpu-dynamic-ranked-gate')
PROFILES=('white','center','edge')


def synthesize(task):
    profile,ratio,fd,phase,replica=task
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x44594741,replica]))
    delay=float(rng.uniform(-.3,.3));common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    signal=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)*(1+ratio*np.exp(1j*phase)*temporal_echo(fd)[0][:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2*10**3.7*.020)
    blocks=signal+noise*np.sqrt(spectral_noise_shape(f,profile))
    return blocks,signal,delay


def evaluate(task):
    blocks,_,delay=synthesize(task)
    profile,ratio,fd,phase,replica=task
    score=score_dynamic_paths(blocks)
    return dict(profile=profile,echo_voltage=ratio,doppler_hz=fd,phase_rad=phase,replica=replica,true_delay_chips=delay,
        score=asdict(score),observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest())


def summarize(rows,gates):
    cells=[]
    for profile in PROFILES:
        for method in gates:
            group=[dict(r,**r['outcomes'][method]) for r in rows if r['profile']==profile]
            for c in summarize_single(group)['cells']:
                primary=method=='ranked' and c['echo_voltage']==0
                p=binomtest(c['two_paths'],c['attempts'],p=.01,alternative='greater').pvalue if primary else None
                cells.append(dict(c,profile=profile,method=method,primary_null_budget_test=primary,
                    adjusted_null_p=min(1.,3*p) if primary else None))
    return dict(cells=cells)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads((PRIOR/'protocol.json').read_text())['source_sha256'])
    for p in (Path(__file__).resolve(),Path('python/gnss_gpu/spectral_noise_cpu.py')):sources[str(p)]=digest(p)
    assert all(digest(p)==h for p,h in sources.items())
    gatepath=PRIOR/'frozen_gates.json';gatehash=digest(gatepath)
    allgates=json.loads(gatepath.read_text())['gates'];gates={m:allgates[m] for m in ('maximum','ranked')}
    null=[(p,0.,0.,0.,600000+r) for p in PROFILES for r in range(128)]
    echo=[(p,.8,fd,0.,610000+r) for p in PROFILES for fd in (0.,.1,.7) for r in range(32)]
    protocol=dict(evaluation_tasks=null+echo,source_sha256=sources,gate_path=str(gatepath),gate_sha256=gatehash,gates=gates,
        context='Frozen unweighted dynamic fit and white-null calibrated gates; PRN1,200ms,3.6828MHz,37 reference dB-Hz,static wiped direct path.',
        profiles='white=1; center=1+9 exp(-.5*(12*x)^2); edge=1+9*x^4; normalized mean variance1, x=f/max(abs(f)). Same expected total noise energy.',
        primary='Ranked final false-two-path rate >.01, one-sided exact binomial tests per three noise profiles, Bonferroni3.',
        pairing='128 null replicas shared across profiles; 32 echo replicas shared across profiles and Dopplers. Null and echo IDs disjoint, all new. Do not pool dependent cells.',
        scope='Null distribution mismatch stress only; white calibration guarantee does not extend to colored noise. Reference C/N0 uses mean PSD.',
        limits='Synthetic stationary independent harmonic noise, not measured frontend/forest/rain; echo contrasts descriptive; no retuning or calibration on stress outcomes.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,null+echo):
            row=apply_gates(raw,gates);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%96==0:stream.flush();print(f'Evaluation {len(rows)}/672',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows,gates),indent=2))
    assert all(digest(p)==h for p,h in sources.items()) and digest(gatepath)==gatehash
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),gate_sha256=gatehash,evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
