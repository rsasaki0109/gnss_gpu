import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_echo_sign_reversal import OUTPUT, DOPPLERS


def run():
    rows = [json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    rates = (-2., 0., 2.)
    for ax, fd in zip(axes, DOPPLERS):
        for replica in sorted({r['replica'] for r in rows}):
            group = [next(r for r in rows if r['initial_doppler_hz']==fd and
                          r['relative_drift_hz_s']==rate and r['replica']==replica) for rate in rates]
            ax.plot(range(3), [r['raw_error_m'] for r in group], color='grey', alpha=.2, lw=.7)
        for i, rate in enumerate(rates):
            group = sorted([r for r in rows if r['initial_doppler_hz']==fd and
                            r['relative_drift_hz_s']==rate], key=lambda r: r['replica'])
            crossing = fd*(fd+rate*.8)<0
            for r in group:
                ax.scatter(i, r['raw_error_m'], color='tab:red' if crossing else 'tab:blue',
                           marker='o' if r['available'] else 'x', s=20, alpha=.7)
        ax.axhline(0, color='grey', ls='--', lw=.7)
        ax.set(xticks=range(3), xticklabels=rates, xlabel='Relative echo carrier drift (Hz/s)',
               title=f'Initial relative Doppler {fd:+.1f} Hz')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Direct range error (m)')
    fig.suptitle('16 shared replicas | Red: zero crossing; blue: no crossing\nFrozen receiver, 49 dB-Hz, 800 ms; lines connect paired replicas')
    fig.tight_layout()
    fig.savefig(OUTPUT/'summary.png', dpi=160)
    plt.close(fig)


if __name__=='__main__':
    run()
