"""Paired mean-matched control isolates block-variance contamination."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE
from gnss_gpu.noise_aware_multipath_cpu import estimate_noise_weights, _fit_weighted_spectrum
from experiments.run_cpu_dynamic_reflection import echo_coefficients, digest, select, GATE
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT=Path('output/cpu-matched-dynamic-reflection')
CONDITIONS=[(fd,phase) for fd in (.7,2.3,6.7) for phase in (0.,np.pi/2,np.pi)]


def synthesize(doppler, phase, replica):
    rng=np.random.default_rng(np.random.SeedSequence([20261026,0x4D415443,replica]))
    delay=float(rng.uniform(-.3,.3)); common=float(rng.uniform(-np.pi,np.pi))
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    direct=s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*common)
    coefficients=.8*np.exp(1j*phase)*echo_coefficients(doppler)
    signal=direct[None,:]*(1+coefficients[:,None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    variance=1/(10**3.7*.020)
    noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))*np.sqrt(variance/2)
    dynamic=signal+noise
    control=signal.mean(axis=0)[None,:]+noise
    return dynamic,control,delay,coefficients.mean()


def evaluate(task):
    dynamic,control,delay,coefficient=synthesize(*task)
    mean=dynamic.mean(axis=0)
    delta=float(np.max(np.abs(mean-control.mean(axis=0))))
    assert delta<1e-15
    fits={};weights={}
    for name,blocks in [('control',control),('dynamic',dynamic)]:
        weights[name]=estimate_noise_weights(blocks)
        # Same observed mean for both arms eliminates roundoff as a confound.
        fits[name]=dict(fit=asdict(_fit_weighted_spectrum(mean,3.6*CHIP_RATE,weights[name])),true_delay_chips=delay)
    import hashlib
    return dict(doppler_hz=task[0],phase_rad=task[1],replica=task[2],arms=fits,
        observation_sha256={name:hashlib.sha256(blocks.tobytes()).hexdigest() for name,blocks in [('control',control),('dynamic',dynamic)]},
        mean_max_difference=delta,coherent_echo_voltage=float(abs(coefficient)),
        coherent_echo_phase=float(np.angle(coefficient)),
        weight_difference_rms=float(np.sqrt(np.mean((weights['dynamic']-weights['control'])**2))))


def apply_gate(row,gate):
    return dict(row,arms={name:select(arm,gate) for name,arm in row['arms'].items()})


def summarize(rows):
    cells=[]
    for fd,phase in CONDITIONS:
        group=sorted([r for r in rows if (r['doppler_hz'],r['phase_rad'])==(fd,phase)],key=lambda r:r['replica'])
        paired=paired_summary([r['arms']['control']['success'] for r in group],[r['arms']['dynamic']['success'] for r in group])
        cells.append(dict(doppler_hz=fd,phase_rad=phase,**paired,adjusted_p=min(1.,9*paired['paired_exact_p']),
            two_paths={a:sum(r['arms'][a]['selected']['selected_model']=='two_path' for r in group) for a in ('control','dynamic')},
            unavailable={a:sum(r['arms'][a]['error_m'] is None for r in group) for a in ('control','dynamic')},
            mean_weight_difference_rms=float(np.mean([r['weight_difference_rms'] for r in group])),
            coherent_echo_voltage=group[0]['coherent_echo_voltage'],
            coherent_echo_phase=group[0]['coherent_echo_phase']))
    return dict(cells=cells)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()):raise ValueError('output must be empty')
    sources=dict(json.loads(Path('output/cpu-dynamic-reflection/protocol.json').read_text())['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    gate=json.loads(GATE.read_text())['combined']['ranked']
    tasks=[(fd,phase,520000+r) for fd,phase in CONDITIONS for r in range(64)]
    protocol=dict(tasks=tasks,source_sha256=sources,gate_path=str(GATE),gate_sha256=digest(GATE),gate=gate,
        scope='PRN1, white Gaussian noise, 37 reference dB-Hz, 10x20ms, one-sided B=3.6 chip rates, echo voltage .8, fixed .1-chip excess delay.',
        model='Same narrowband phase-ramp approximation as preceding dynamic study; block-averaged phase, no delay drift or inter-harmonic leakage.',
        control='Replace each noiseless dynamic block by its across-block mean; add identical noise. Receiver weights estimated independently per arm; both frozen fits use the identical observed dynamic mean to exclude roundoff.',
        inference='Simulator truth used only to construct counterfactual control and evaluate error; no true delay, Doppler, phase or noise variance supplied to fitting or selection.',
        primary='Dynamic vs mean-matched static control, success abs(range error)<10m; unavailable counts as failure; nine paired two-sided exact tests, Bonferroni .05.',
        pairing='64 new replicas shared across all nine conditions; 576 pairs, 1152 fits; do not pool conditions as independent.',
        limits='No recalibration, no real-data validation, no equivalence claim from non-significance. Counterfactual control is an experimental diagnostic, not a deployable correction.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool,(OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,tasks):
            row=apply_gate(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%64==0:stream.flush();print(f'Evaluation {len(rows)}/576 pairs',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    assert all(digest(p)==h for p,h in sources.items()) and digest(GATE)==protocol['gate_sha256']
    (OUTPUT/'completion.json').write_text(json.dumps(dict(pairs=len(rows),fits=2*len(rows),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
