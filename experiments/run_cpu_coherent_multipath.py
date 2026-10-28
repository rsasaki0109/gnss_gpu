"""Nested 20/80/200 ms ideal coherent integration with frozen v2 fitting."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from experiments.validate_cpu_bandlimited_multipath import paired_summary
from gnss_gpu.bandlimited_multipath_v2_cpu import fit_bandlimited_paths
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE, LIGHT_SPEED

OUTPUT = Path('output/cpu-coherent-multipath')
KEYS = ('bandwidth_chip_rates', 'echo_voltage', 'gap_chips', 'phase_rad')


def synthesize_blocks(bandwidth, ratio, gap, phase, replica):
    """Ten independent-noise 20 ms blocks after PERFECT carrier/nav wipeoff.

    The static signal is identical in each block. Finite-window RF/filter
    transients, motion, phase noise and uncorrected bit changes are excluded.
    Return one noiseless spectrum, ten observations and direct delay for scoring.
    """
    rng = np.random.default_rng(np.random.SeedSequence([20261026,0x434F4845,replica]))
    delay = float(rng.uniform(-.3,.3))
    common_phase = float(rng.uniform(-np.pi,np.pi))
    frequency, spectrum = ca_passband_spectrum(3.6*CHIP_RATE)
    noise = rng.normal(size=(10,len(frequency)))+1j*rng.normal(size=(10,len(frequency)))
    mask = abs(frequency) <= bandwidth*CHIP_RATE
    frequency, spectrum = frequency[mask], spectrum[mask]
    signal = spectrum*np.exp(-2j*np.pi*frequency*delay/CHIP_RATE)*np.exp(1j*common_phase)
    signal *= 1+ratio*np.exp(1j*phase-2j*np.pi*frequency*gap/CHIP_RATE)
    observations = signal[None,:]+noise[:,mask]/np.sqrt(2*10**3.7*.020)
    return signal, observations, delay


def evaluate(task):
    bandwidth, ratio, gap, phase, replica = task
    _, blocks, delay = synthesize_blocks(*task)
    records = []
    for count in (1,4,10):
        observation = blocks[:count].mean(axis=0)
        fit = fit_bandlimited_paths(observation,bandwidth*CHIP_RATE)
        def error(d): return (d-delay)*LIGHT_SPEED/CHIP_RATE if d is not None else None
        records.append(dict(bandwidth_chip_rates=bandwidth, echo_voltage=ratio,
            gap_chips=gap,phase_rad=phase,replica=replica,duration_ms=count*20,
            true_delay_chips=delay,fit=asdict(fit),single_error_m=error(fit.single_delay_chips),
            selected_error_m=error(fit.selected_delay_chips),
            observation_sha256=hashlib.sha256(observation.tobytes()).hexdigest()))
    return records


def summarize(records):
    groups = {}
    for row in records:
        groups.setdefault(tuple(row[k] for k in KEYS),{}).setdefault(row['duration_ms'],[]).append(row)
    summaries, comparisons = [], []
    for key, durations in groups.items():
        success = {}
        for duration, rows in durations.items():
            rows = sorted(rows,key=lambda r:r['replica'])
            success[duration] = [r['selected_error_m'] is not None and abs(r['selected_error_m'])<10 for r in rows]
            total, wins = len(rows), sum(success[duration])
            ci = binomtest(wins,total).proportion_ci(confidence_level=.95,method='exact')
            summaries.append(dict(zip(KEYS,key),duration_ms=duration,attempts=total,
                selected_successes=wins,selected_exact_95=list(ci),
                single_successes=sum(abs(r['single_error_m'])<10 for r in rows),
                unavailable=sum(r['selected_error_m'] is None for r in rows),
                two_path_selected=sum(r['fit']['selected_model']=='two_path' for r in rows)))
        for duration in (80,200):
            paired = paired_summary(success[20],success[duration])
            primary = key[0]==3.6 and key[1]==.8 and key[2]==.1 and duration==200
            comparisons.append(dict(zip(KEYS,key),duration_ms=duration,**paired,primary=primary,
                adjusted_primary_p=min(1.,2*paired['paired_exact_p']) if primary else None,
                primary_improvement_supported=(paired['difference']>0 and paired['paired_exact_p']<.025) if primary else None))
    return summaries, comparisons


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if any(OUTPUT.iterdir()): raise ValueError('output must be empty')
    prior = json.loads(Path('output/cpu-bandlimited-v2/protocol.json').read_text())
    model_path = Path('python/gnss_gpu/bandlimited_multipath_v2_cpu.py')
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==prior['source_and_prior_sha256'][str(model_path)]
    sources = [Path(__file__).resolve(), model_path,
        Path('python/gnss_gpu/multipath_information_cpu.py'),Path('python/gnss_gpu/input_validation.py'),
        Path('python/gnss_gpu/validation/tracking_residuals.py'),
        Path('experiments/validate_cpu_bandlimited_multipath.py')]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    prior_hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path('output/cpu-bandlimited-v2').iterdir() if p.is_file()}
    cells = [(0.,0.,0.)]+[(.8,g,p) for g in (.1,.5) for p in (0.,np.pi/2)]
    tasks = [(b,a,g,p,r) for b in (.9,3.6) for a,g,p in cells for r in range(64)]
    protocol = dict(tasks=tasks,durations_ms=[20,80,200],cn0_dbhz=37.,replicas_per_cell=64,
        seed_domain=0x434F4845,source_sha256=hashes,prior_artifact_sha256=prior_hashes,
        scope='Static isolated PRN1, perfect carrier/nav wipeoff, external fixed coarse delay assistance',
        noise='Independent proper complex spectral noise per 20 ms block; variance 1/(C/N0*.020)',
        pairing='First 1/4/10 blocks of same 200 ms observation; common harmonics shared across bandwidths',
        primary='200 vs 20 ms selected success; wide bandwidth, .1-chip echo, phases 0 and pi/2',
        inference='Two exact paired McNemar tests, Bonferroni family alpha=.05; other comparisons exploratory',
        success='Available absolute direct range error <10 m; failures retained',
        limits='64 independent replica IDs shared across cells, not 1920 independent trials; no real coherence validation')
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    with (OUTPUT/'records.jsonl').open('w') as stream, ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(evaluate,tasks):
            records.extend(result)
            for row in result: stream.write(json.dumps(row,allow_nan=False)+'\n')
            if len(records)%192==0:
                stream.flush();print(f'{len(records)}/1920 estimates',flush=True)
    summaries,comparisons = summarize(records)
    (OUTPUT/'report.json').write_text(json.dumps(dict(cells=summaries,comparisons=comparisons),indent=2,allow_nan=False))
    for path,digest in {**hashes,**prior_hashes}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    fig,axes = plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,b in zip(axes,(.9,3.6)):
        for j,duration in enumerate((20,80,200)):
            rows = [r for r in summaries if r['bandwidth_chip_rates']==b and r['duration_ms']==duration]
            y=np.array([r['selected_successes']/r['attempts'] for r in rows])
            bounds=np.array([r['selected_exact_95'] for r in rows]).T
            ax.errorbar(np.arange(5)+(j-1)*.17,y,yerr=[y-bounds[0],bounds[1]-y],fmt='o',capsize=3,label=f'{duration} ms')
        ax.set_xticks(np.arange(5),['Direct only','.1 chip\nphase 0','.1 chip\npi/2','.5 chip\nphase 0','.5 chip\npi/2'])
        ax.set_title(f'One-sided B={b*1.023:.3f} MHz')
        ax.legend();ax.grid(axis='y',alpha=.2);ax.set_ylim(-.04,1.04)
    axes[0].set_ylabel('Available range error <10 m: proportion')
    fig.suptitle('37 dB-Hz, ideal carrier/nav coherence; pointwise exact 95% intervals')
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__': run()
