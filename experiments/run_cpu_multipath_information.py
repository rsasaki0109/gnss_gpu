"""Separate sampling rate from physical bandwidth in an ideal local CRLB study."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnss_gpu.multipath_information_cpu import CHIP_RATE, two_path_information


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output must be empty; preserve previous experiments')
    sources = [Path(__file__).resolve(),
               Path('python/gnss_gpu/multipath_information_cpu.py'),
               Path('python/gnss_gpu/validation/tracking_residuals.py'),
               Path('python/gnss_gpu/input_validation.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    protocol = dict(
        samples_per_chip=[2, 4, 8, 16],
        one_sided_bandwidth_in_chip_rates=[.5, .9, 1.8, 3.6],
        excess_chips=[0., .01, .02, .05, .1, .2, .5, 1.],
        echo_voltage=[.3, .8, 1.5], phase_rad=[0., np.pi/2, np.pi],
        cn0_dbhz=[37., 55.], integration_s=.020, prn=1,
        noise='Ideal passband complex white noise PSD N0; correlated time samples',
        power='Direct C/N0 before filter; no post-filter power renormalization',
        nuisance='Both complex path amplitudes unknown; both delays unknown',
        scope='Local unbiased CRLB only; known carrier/nav; no acquisition or PVT',
        alias_policy='Skip fs <= 2*B; never silently alias the analog signal',
        source_sha256=hashes, numpy_version=np.__version__,
    )
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False))
    records, skipped = [], []
    for spc in protocol['samples_per_chip']:
        for bandwidth in protocol['one_sided_bandwidth_in_chip_rates']:
            if spc <= 2*bandwidth:
                skipped.append(dict(samples_per_chip=spc, bandwidth_chip_rates=bandwidth))
                continue
            for cn0 in protocol['cn0_dbhz']:
                for amplitude in protocol['echo_voltage']:
                    for phase in protocol['phase_rad']:
                        for gap in protocol['excess_chips']:
                            result = two_path_information(
                                sampling_freq=spc*CHIP_RATE, bandwidth_hz=bandwidth*CHIP_RATE,
                                excess_chips=gap, echo_voltage=amplitude, phase_rad=phase,
                                cn0_dbhz=cn0, integration_s=protocol['integration_s'])
                            records.append(dict(samples_per_chip=spc, bandwidth_chip_rates=bandwidth,
                                cn0_dbhz=cn0, echo_voltage=amplitude, phase_rad=phase,
                                excess_chips=gap, **asdict(result)))
    # Keep singular cases in the record set; plot only finite positive-gap bounds.
    (output/'records.json').write_text(json.dumps(records, indent=2, allow_nan=False))
    groups = {}
    for row in records:
        key = tuple(row[k] for k in ('bandwidth_chip_rates', 'cn0_dbhz',
                                    'echo_voltage', 'phase_rad', 'excess_chips'))
        groups.setdefault(key, []).append(row)
    spreads = [max(values)-min(values) for rows in groups.values()
               if (values := [r['direct_std_bound_m'] for r in rows
                              if r['direct_std_bound_m'] is not None])]
    report = dict(records=len(records), singular_records=sum(r['status'] != 'local_bound' for r in records),
                  skipped_sampling_bandwidth_pairs=skipped,
                  maximum_same_bandwidth_sampling_spread_m=max(spreads),
                  source_hashes_verified=all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                             for p, h in hashes.items()))
    assert report['source_hashes_verified']
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for i, cn0 in enumerate(protocol['cn0_dbhz']):
        for j, phase in enumerate((0., np.pi/2)):
            ax = axes[i, j]
            for bandwidth in protocol['one_sided_bandwidth_in_chip_rates']:
                rows = [r for r in records if r['samples_per_chip'] == 16
                        and r['cn0_dbhz'] == cn0 and r['echo_voltage'] == .8
                        and r['phase_rad'] == phase and r['bandwidth_chip_rates'] == bandwidth
                        and r['direct_std_bound_m'] is not None and r['excess_chips'] > 0]
                ax.loglog([r['excess_chips'] for r in rows],
                          [r['direct_std_bound_m'] for r in rows], 'o-',
                          label=f'B = {bandwidth*CHIP_RATE/1e6:.3f} MHz')
            ax.axhline(10, color='gray', ls=':', lw=1)
            ax.set_title(f'{cn0:g} dB-Hz, relative phase {phase/np.pi:g} pi')
            ax.grid(True, which='both', alpha=.2)
            if i == 1:
                ax.set_xlabel('Excess delay (chips)')
            if j == 0:
                ax.set_ylabel('Direct range local std lower bound (m)')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Ideal two-path local bound: 20 ms, echo/direct voltage 0.8\n'
                 'One-sided bandwidth B; unknown amplitudes; not achieved receiver error')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'comparison.{extension}', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/cpu-multipath-information'))
    run(parser.parse_args().output)
