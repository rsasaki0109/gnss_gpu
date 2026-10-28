"""Export fresh-measurement continuity and sequence-level null outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (directory/'sequences.jsonl').open()]
    report = json.loads((directory/'report.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for axis, outage in zip(axes[:2], (False, True)):
        for doppler in (170., 700.):
            selected = [r for r in records if r['target'] and r['doppler_hz'] == doppler and r['outage'] == outage]
            times = [r['available_after_ms'] for r in selected[0]['trace']]
            used = np.mean([[r['measurement_used'] for r in record['trace']] for record in selected], axis=0)
            axis.plot(times, used, marker='.', label=f'{doppler:g} Hz')
        if outage:
            axis.axvspan(80., 120., color='gray', alpha=.2, label='Target absent')
        axis.set(title='40 ms target outage' if outage else 'Continuous target', xlabel='Decision available after (ms)',
                 ylabel='Fraction using a fresh measurement', ylim=(-.03, 1.06))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    nulls = [r for r in report['groups'] if not r['target']]
    n = np.array([r['replicas'] for r in nulls])
    rates = np.array([r['published_sequences'] for r in nulls])/n
    bounds = np.array([r['published_sequence_wilson_95'] for r in nulls])
    axes[2].bar(range(3), rates*100, yerr=np.maximum(0., [rates-bounds[:, 0], bounds[:, 1]-rates])*100, capsize=3)
    for i, row in enumerate(nulls):
        axes[2].text(i, rates[i]*100+.3, f"{row['published_sequences']}/{row['replicas']}", ha='center')
    axes[2].set(title='Target absent for the full 240 ms', xticks=range(3),
                xticklabels=('White', 'Other PRNs', 'Others +20 dB'), ylabel='Sequences publishing any state (%)',
                ylim=(0., max(12., float(np.max(bounds))*110)))
    axes[2].grid(axis='y', alpha=.2)
    fig.suptitle('37 dB-Hz target, other satellites +20 dB; disjoint 20 ms windows\n'
                 'Coarse state following; pointwise sequence Wilson intervals, no track-level guarantee')
    fig.savefig(directory/'weak_tracking.png', dpi=160)
    fig.savefig(directory/'weak_tracking.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('output/cpu-weak-tracking'))
    plot(parser.parse_args().directory)
