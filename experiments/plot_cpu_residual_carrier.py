import json
import matplotlib.pyplot as plt
from experiments.stress_cpu_residual_carrier import OUTPUT


def run():
    cells = json.loads((OUTPUT/'report.json').read_text())['cells']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ratio in zip(axes, (0., .8)):
        group = [c for c in cells if c['echo_voltage'] == ratio]
        x = [c['residual_carrier_hz'] for c in group]
        for key, label, marker in [('successes', 'Range error <10 m', 'o'),
                                   ('two_paths', 'Two-path selections', 's'),
                                   ('unavailable', 'Unavailable', '^')]:
            ax.plot(x, [c[key]/c['attempts'] for c in group], marker=marker, label=label)
        ax.set(title='No reflection' if ratio == 0 else 'Echo voltage 0.8; relative Doppler 0.7 Hz',
               xlabel='Common residual carrier (Hz)', ylim=(-.04, 1.04))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Fraction (24 paired trials per condition)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen receiver and gate: residual carrier mismatch')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT/f'comparison.{ext}', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    run()
