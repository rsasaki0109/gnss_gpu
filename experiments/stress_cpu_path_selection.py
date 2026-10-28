"""Frozen-gate single-path stress across power and noise spectral shape."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate,_select_default_fit
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum,CHIP_RATE,LIGHT_SPEED
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape

OUTPUT=Path('output/cpu-path-selection-stress')
GATE=Path('output/cpu-calibrated-path-selection/frozen_gate.json')


def synthesize(cn0,profile,replica):
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x53545253,replica]))
    delay=float(rng.uniform(-.3,.3));phase=float(rng.uniform(-np.pi,np.pi))
    frequency,spectrum=ca_passband_spectrum(3.6*CHIP_RATE)
    shape=spectral_noise_shape(frequency,profile)
    signal=spectrum*np.exp(-2j*np.pi*frequency*delay/CHIP_RATE+1j*phase)
    noise=rng.normal(size=len(frequency))+1j*rng.normal(size=len(frequency))
    observation=signal+noise*np.sqrt(shape/(2*10**(cn0/10)*.200))
    return observation,delay


def evaluate(task):
    cn0,profile,replica,gate_data=task
    observation,delay=synthesize(cn0,profile,replica)
    fit=fit_bandlimited_paths(observation,3.6*CHIP_RATE)
    gate=PathSelectionGate(**gate_data)
    selected=_select_default_fit(fit,gate)
    def error(value): return (value-delay)*LIGHT_SPEED/CHIP_RATE if value is not None else None
    return dict(reference_cn0_dbhz=cn0,profile=profile,replica=replica,true_delay_chips=delay,
        fit=asdict(fit),calibrated_fit=asdict(selected),old_error_m=error(fit.selected_delay_chips),
        new_error_m=error(selected.selected_delay_chips),score_exceeds_gate=fit.bic_improvement>gate.score,
        observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest())


def summarize(records):
    groups={}
    for row in records: groups.setdefault((row['reference_cn0_dbhz'],row['profile']),[]).append(row)
    report=[]
    for (cn0,profile),rows in groups.items():
        n=len(rows);score=sum(r['score_exceeds_gate'] for r in rows)
        false=sum(r['calibrated_fit']['selected_model']=='two_path' for r in rows)
        ci=binomtest(false,n).proportion_ci(confidence_level=.95,method='exact')
        p=float(binomtest(false,n,p=.01,alternative='greater').pvalue)
        report.append(dict(reference_cn0_dbhz=cn0,profile=profile,attempts=n,
            score_exceedances=score,false_two_path_selections=false,false_selection_exact_95=list(ci),
            above_one_percent_p=p,bonferroni_12_p=min(1.,12*p),
            excess_over_one_percent_supported=12*p<.05,
            old_false_two_paths=sum(r['fit']['selected_model']=='two_path' for r in rows),
            old_under_10m=sum(r['old_error_m'] is not None and abs(r['old_error_m'])<10 for r in rows),
            new_under_10m=sum(r['new_error_m'] is not None and abs(r['new_error_m'])<10 for r in rows),
            new_unavailable=sum(r['new_error_m'] is None for r in rows)))
    return report


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-calibrated-path-selection/protocol.json').read_text())
    sources=dict(previous['source_sha256'])
    for path,digest in sources.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    for path in (Path(__file__).resolve(),Path('python/gnss_gpu/spectral_noise_cpu.py')):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    prior={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in GATE.parent.iterdir() if p.is_file()}
    gate=json.loads(GATE.read_text())
    tasks=[(c,p,r,gate) for c in (31.,37.,43.,55.) for p in ('white','center','edge') for r in range(128)]
    protocol=dict(tasks=tasks,source_sha256=sources,prior_artifact_sha256=prior,
        frozen_gate=gate,frozen_gate_sha256=hashlib.sha256(GATE.read_bytes()).hexdigest(),
        conditions='Single path only; PRN1, B=3.6 chip rates, 200 ms ideal carrier/nav coherence, default assistance',
        power='Reference C/N0 uses mean passband noise PSD; colored noise has frequency-dependent PSD; equal total expected noise energy at fixed reference C/N0',
        profiles='white=1; center=1+9*exp(-.5*(12*f/cutoff)^2); edge=1+9*(f/cutoff)^4; each normalized to mean 1',
        inference='12 predeclared one-sided exact tests of selected false-path rate >1%, Bonferroni family alpha .05',
        confidence='Pointwise exact 95% binomial intervals, not simultaneous; non-rejection is not proof of rate <=1%',
        seed_domain=0x53545253,replicas_per_cell=128,
        independence='128 independent replica IDs per cell shared across power/profile cells; not 1536 independent replicas',
        scope='Distribution-shift stress of a fixed conditional gate; no new calibration or acquisition/PVT guarantee')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    with (OUTPUT/'records.jsonl').open('w') as stream,ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(evaluate,tasks):
            records.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(records)%128==0:
                stream.flush();print(f'{len(records)}/1536 fits',flush=True)
    report=summarize(records)
    (OUTPUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    for path,digest in {**sources,**prior}.items(): assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    fig,ax=plt.subplots(figsize=(9,4.5))
    for j,profile in enumerate(('white','center','edge')):
        rows=[r for r in report if r['profile']==profile]
        y=np.array([r['false_two_path_selections']/r['attempts'] for r in rows])
        bounds=np.array([r['false_selection_exact_95'] for r in rows]).T
        ax.errorbar(np.arange(4)+(j-1)*.16,y,yerr=[y-bounds[0],bounds[1]-y],fmt='o',capsize=3,label=profile)
    ax.axhline(.01,color='black',ls=':',label='Original conditional 1% budget')
    ax.set_xticks(np.arange(4),['31','37','43','55']);ax.set_xlabel('Reference C/N0 (dB-Hz)')
    ax.set_ylabel('False two-path selections / all attempts')
    ax.set_title('Frozen gate, no echoes; pointwise exact 95% intervals')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__': run()
