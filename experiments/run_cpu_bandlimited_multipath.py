"""Paired ideal spectral-noise experiment; local estimation, not acquisition."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt

from gnss_gpu.bandlimited_multipath_cpu import fit_bandlimited_paths
from gnss_gpu.multipath_information_cpu import (
    CHIP_RATE, LIGHT_SPEED, ca_passband_spectrum, two_path_information,
)


def evaluate(task):
    bandwidth, cn0, ratio, gap, phase, replica = task
    # Same random initial delay, common phase and spectral noise across cells.
    rng = np.random.default_rng(np.random.SeedSequence([20261026, 0x424C4D50, replica]))
    delay = float(rng.uniform(-.3, .3))
    common_phase = float(rng.uniform(-np.pi, np.pi))
    full_frequency, full_spectrum = ca_passband_spectrum(3.6*CHIP_RATE)
    noise = rng.normal(size=len(full_frequency))+1j*rng.normal(size=len(full_frequency))
    mask = abs(full_frequency) <= bandwidth*CHIP_RATE
    frequency, spectrum = full_frequency[mask], full_spectrum[mask]
    # Analytic Fourier synthesis; no fitted delays or CRLB supplied to inference.
    y = spectrum*np.exp(-2j*np.pi*frequency*delay/CHIP_RATE)*np.exp(1j*common_phase)
    y *= 1+ratio*np.exp(1j*phase-2j*np.pi*frequency*gap/CHIP_RATE)
    y += noise[mask]/np.sqrt(2*10**(cn0/10)*.020)
    fit = fit_bandlimited_paths(y, bandwidth*CHIP_RATE)
    bound = (two_path_information(sampling_freq=16*CHIP_RATE,
             bandwidth_hz=bandwidth*CHIP_RATE, excess_chips=gap,
             echo_voltage=ratio, phase_rad=phase, cn0_dbhz=cn0).direct_std_bound_m
             if ratio else None)
    def error(value):
        return (value-delay)*LIGHT_SPEED/CHIP_RATE if value is not None else None
    return dict(bandwidth_chip_rates=bandwidth, cn0_dbhz=cn0, echo_voltage=ratio,
        gap_chips=gap, phase_rad=phase, replica=replica, true_delay_chips=delay,
        fit=asdict(fit), single_error_m=error(fit.single_delay_chips),
        raw_two_path_error_m=error(fit.first_delay_chips),
        selected_error_m=error(fit.selected_delay_chips), local_std_bound_m=bound,
        observation_sha256=hashlib.sha256(y.tobytes()).hexdigest())


def run():
    output = Path('output/cpu-bandlimited-multipath')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty')
    cells = [(0., 0., 0.)]+[(.8, gap, phase) for gap in (.1, .5) for phase in (0., np.pi/2)]
    tasks = [(b, cn0, a, g, p, r) for b in (.9, 3.6) for cn0 in (37., 55.)
             for a, g, p in cells for r in range(8)]
    sources = [Path(__file__).resolve()]+[Path('python/gnss_gpu')/p for p in
        ('bandlimited_multipath_cpu.py', 'multipath_information_cpu.py',
         'input_validation.py', 'validation/tracking_residuals.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol = dict(tasks=tasks, replicas=8, integration_s=.020,
        seed_domain=0x424C4D50, source_sha256=hashes,
        assistance='Fixed center 0, earliest delay +/-1 chip, gap [.01,1.2] chips',
        selection='BIC plus convergence/interior/local std <= .1 chip; not calibrated',
        scope='Known carrier/nav; isolated PRN1 periodic spectral Gaussian model. No acquisition/PVT.',
        noise='Independent proper complex coefficients variance 1/(C/N0*T); shared harmonics paired',
        power='Direct pre-filter amplitude 1; no passband power normalization',
        denominator='All 8 attempts per cell; unavailable selected estimates count as failures')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for record in pool.map(evaluate, tasks):
            records.append(record)
            if len(records) % 16 == 0:
                print(f'{len(records)}/{len(tasks)} records', flush=True)
    summaries = []
    for b in (.9, 3.6):
        for cn0 in (37., 55.):
            for a, g, p in cells:
                rows = [r for r in records if (r['bandwidth_chip_rates'], r['cn0_dbhz'],
                    r['echo_voltage'], r['gap_chips'], r['phase_rad']) == (b, cn0, a, g, p)]
                selected = [r for r in rows if r['selected_error_m'] is not None]
                summaries.append(dict(bandwidth_chip_rates=b, cn0_dbhz=cn0, echo_voltage=a,
                    gap_chips=g, phase_rad=p, attempts=len(rows), available=len(selected),
                    two_path_selected=sum(r['fit']['selected_model']=='two_path' for r in rows),
                    ambiguous=sum(r['fit']['selected_model']=='ambiguous' for r in rows),
                    single_under_10m=sum(abs(r['single_error_m'])<10 for r in rows),
                    selected_under_10m=sum(abs(r['selected_error_m'])<10 for r in selected),
                    single_rmse_m=float(np.sqrt(np.mean([r['single_error_m']**2 for r in rows]))),
                    raw_two_path_rmse_m=float(np.sqrt(np.mean([r['raw_two_path_error_m']**2 for r in rows]))),
                    selected_rmse_available_m=float(np.sqrt(np.mean([r['selected_error_m']**2 for r in selected]))) if selected else None,
                    local_std_bound_m=rows[0]['local_std_bound_m']))
    assert len(records) == 160 and len(summaries) == 20
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (output/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
    (output/'report.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, cn0 in zip(axes, (37., 55.)):
        x = np.arange(5)
        for j, b in enumerate((.9, 3.6)):
            rows = [r for r in summaries if r['cn0_dbhz']==cn0 and r['bandwidth_chip_rates']==b]
            ax.bar(x+(j-.5)*.36, [r['selected_under_10m'] for r in rows], width=.36,
                   label=f'B={b*1.023:.3f} MHz')
        ax.set_xticks(x, ['Direct only', '.1 chip\n0 phase', '.1 chip\npi/2', '.5 chip\n0 phase', '.5 chip\npi/2'])
        ax.set_title(f'{cn0:g} dB-Hz; selected estimate')
        ax.set_ylim(0, 8.8)
        ax.legend()
    axes[0].set_ylabel('Available and absolute range error <10 m (out of 8)')
    fig.suptitle('Ideal bandlimited local fitting; failures retained; no acquisition test')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'comparison.{extension}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    run()
