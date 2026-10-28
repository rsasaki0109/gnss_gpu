import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_accelerating_echo_pilot import OUTPUT


def run():
    rows = [json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    report = json.loads((OUTPUT/'report.json').read_text())
    cells = sorted(report['cells'], key=lambda r: r['relative_drift_hz_s'])
    rates = [r['relative_drift_hz_s'] for r in cells]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, rate in enumerate(rates):
        group = sorted([r for r in rows if r['relative_drift_hz_s']==rate], key=lambda r: r['replica'])
        for j, r in enumerate(group):
            axes[0].scatter(i+(j-7.5)*.025, r['raw_error_m'], marker='o' if r['available'] else 'x',
                            color='tab:blue' if r['available'] else 'tab:red', s=24)
    axes[0].axhline(10, color='grey', ls='--'); axes[0].axhline(-10, color='grey', ls='--')
    axes[0].set(xticks=range(5), xticklabels=rates, ylabel='Direct range error (m)',
                xlabel='Relative echo carrier drift (Hz/s)', title='Dots: usable; crosses: unavailable (raw fits)')
    values = np.array([r['successes']/r['attempts'] for r in cells])
    ci = np.array([r['success_exact_95'] for r in cells])
    axes[1].errorbar(range(5), values, yerr=[values-ci[:, 0], ci[:, 1]-values], fmt='o', capsize=4)
    axes[1].set(xticks=range(5), xticklabels=rates, ylim=(-.03, 1.08), ylabel='Usable and |range error| < 10 m',
                xlabel='Relative echo carrier drift (Hz/s)', title='16 paired replicas/cell; pointwise exact 95% CI')
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle('Accelerating echo, frozen common-drift receiver | 49 dB-Hz, 800 ms, initial Doppler 0.7 Hz')
    fig.tight_layout()
    fig.savefig(OUTPUT/'summary.png', dpi=160)
    plt.close(fig)


if __name__=='__main__':
    run()
