"""Frozen receiver stress: block-integrated narrowband echo phase dynamics."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.stats import binomtest
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE, LIGHT_SPEED
from gnss_gpu.noise_aware_multipath_cpu import fit_noise_aware_paths, estimate_noise_weights
from gnss_gpu.calibrated_path_selection_cpu import PathSelectionGate, _select_default_fit
from experiments.validate_cpu_bandlimited_multipath import paired_summary

OUTPUT = Path('output/cpu-dynamic-reflection')
GATE = Path('output/cpu-ranked-calibration/frozen_gates.json')
CONDITIONS = [(0., 0.), (.8, 0.), (.8, 1.), (.8, 5.), (.8, 10.)]


def echo_coefficients(doppler_hz):
    """Exact rectangular block average of exp(2j*pi*fd*t), phase zero at t=0."""
    centers = (np.arange(10)+.5)*.020
    return np.sinc(doppler_hz*.020)*np.exp(2j*np.pi*doppler_hz*centers)


def synthesize(ratio, doppler, replica):
    rng = np.random.default_rng(np.random.SeedSequence([20261026, 0x44594E41, replica]))
    delay = float(rng.uniform(-.3, .3))
    phase = float(rng.uniform(-np.pi, np.pi))
    f, s = ca_passband_spectrum(3.6*CHIP_RATE)
    direct = s*np.exp(-2j*np.pi*f*delay/CHIP_RATE+1j*phase)
    signal = direct[None, :]*(1+ratio*echo_coefficients(doppler)[:, None]*np.exp(-2j*np.pi*f*.1/CHIP_RATE))
    variance = 1/(10**3.7*.020)
    noise = (rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))*np.sqrt(variance/2)
    return signal+noise, signal, delay, variance


def evaluate(task):
    ratio, doppler, replica = task
    blocks, signal, delay, variance = synthesize(*task)
    fit = fit_noise_aware_paths(blocks)
    return dict(echo_voltage=ratio, doppler_hz=doppler, replica=replica, true_delay_chips=delay,
        fit=asdict(fit), observation_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
        signal_variance_over_noise=float(np.mean(np.var(signal, axis=0, ddof=1))/variance),
        estimated_white_weight_rms=float(np.sqrt(np.mean((estimate_noise_weights(blocks)-1)**2))),
        coherent_echo_voltage=float(ratio*abs(echo_coefficients(doppler).mean())))


def select(row, gate):
    from gnss_gpu.noise_aware_multipath_cpu import BandlimitedFit
    fit = _select_default_fit(BandlimitedFit(**row['fit']), PathSelectionGate(**gate))
    error = None if fit.selected_delay_chips is None else (fit.selected_delay_chips-row['true_delay_chips'])*LIGHT_SPEED/CHIP_RATE
    return dict(row, selected=asdict(fit), error_m=error, success=error is not None and abs(error)<10)


def summarize(rows):
    cells=[]
    for ratio, doppler in CONDITIONS:
        group=sorted([r for r in rows if (r['echo_voltage'],r['doppler_hz'])==(ratio,doppler)],key=lambda r:r['replica'])
        n=len(group); wins=sum(r['success'] for r in group)
        cells.append(dict(echo_voltage=ratio,doppler_hz=doppler,attempts=n,successes=wins,
            success_exact_95=list(binomtest(wins,n).proportion_ci(method='exact')),
            two_paths=sum(r['selected']['selected_model']=='two_path' for r in group),
            unavailable=sum(r['error_m'] is None for r in group),
            mean_signal_variance_over_noise=float(np.mean([r['signal_variance_over_noise'] for r in group])),
            mean_estimated_white_weight_rms=float(np.mean([r['estimated_white_weight_rms'] for r in group])),
            coherent_echo_voltage=group[0]['coherent_echo_voltage']))
    base={r['replica']:r['success'] for r in rows if r['echo_voltage']==.8 and r['doppler_hz']==0}
    contrasts=[]
    for fd in (1.,5.,10.):
        group=sorted([r for r in rows if r['echo_voltage']==.8 and r['doppler_hz']==fd],key=lambda r:r['replica'])
        assert set(base)=={r['replica'] for r in group}
        p=paired_summary([base[r['replica']] for r in group],[r['success'] for r in group])
        contrasts.append(dict(doppler_hz=fd,**p,adjusted_p=min(1.,3*p['paired_exact_p'])))
    return dict(cells=cells,contrasts=contrasts)


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    previous=json.loads(Path('output/cpu-ranked-calibration/protocol.json').read_text())
    sources=dict(previous['source_sha256'])
    sources[str(Path(__file__).resolve())]=digest(__file__)
    assert all(digest(p)==h for p,h in sources.items())
    gate=json.loads(GATE.read_text())['combined']['ranked']
    tasks=[(ratio,fd,510000+r) for ratio,fd in CONDITIONS for r in range(128)]
    protocol=dict(tasks=tasks,source_sha256=sources,gate_path=str(GATE),gate_sha256=digest(GATE),gate=gate,
        context='PRN1; white Gaussian noise; 37 reference dB-Hz; 10x20ms; B=3.6 chip rates; direct path static; echo gap .1 chips; initial relative phase zero',
        model='Narrowband frozen code-delay approximation; exact 20ms average of echo complex phase ramp. No inter-harmonic Doppler leakage or geometric code-delay evolution.',
        physical_scale='At L1, 1/5/10 Hz correspond to .190/.951/1.903 m/s excess-path rate; maximum omitted delay drift .381 m in 200ms.',
        comparison='Preserved initial ranked gate chosen before evaluation; no recalibration. Dynamic vs static paired success <10m, three two-sided exact tests Bonferroni .05.',
        diagnostics='True signal variance and echo attenuation are analysis only; fitter receives only blocks.',
        limits='Synthetic stress, not measured forest/rain or positioning. Dynamics change both mean echo and noise estimate, so success differences do not isolate noise-estimation contamination.')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool, (OUTPUT/'evaluation.jsonl').open('w') as stream:
        for raw in pool.map(evaluate,tasks):
            row=select(raw,gate);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(rows)%128==0: stream.flush();print(f'Evaluation {len(rows)}/640',flush=True)
    (OUTPUT/'report.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    assert all(digest(p)==h for p,h in sources.items()) and digest(GATE)==protocol['gate_sha256']
    (OUTPUT/'completion.json').write_text(json.dumps(dict(evaluation=len(rows),evaluation_sha256=digest(OUTPUT/'evaluation.jsonl'),report_sha256=digest(OUTPUT/'report.json')),indent=2))

if __name__=='__main__':run()
