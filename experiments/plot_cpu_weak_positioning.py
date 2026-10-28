"""Export coarse positioning error and availability with unavailable epochs retained."""

import argparse
import json
from pathlib import Path


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (directory/'positions.jsonl').open()]
    report = json.loads((directory/'report.json').read_text())
    selected = ('strong_55','weak_37','three_excess_paths_300m','common_clock_plus_300m')
    errors = [[r['position_error_m'] for r in records if r['scenario']==name and r['window']>0
               and r['position_error_m'] is not None] for name in selected]
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].boxplot(errors, tick_labels=('55 dB-Hz','37 dB-Hz','3 paths +300 m','Clock +300 m'), showmeans=True)
    axes[0].set(title='IQ-derived coarse position error', ylabel='3D error (m)')
    groups = report['summaries']
    fractions = [r['solved']/r['epochs'] for r in groups]
    axes[1].bar(range(len(groups)), fractions)
    for i, row in enumerate(groups):
        axes[1].text(i, fractions[i]+.025, f"{row['solved']}/{row['epochs']}", ha='center')
    axes[1].set(title='Post-confirmation position availability', xticks=range(len(groups)),
                xticklabels=('55 dB-Hz','37 dB-Hz','Only 3 PRNs','3 paths +300 m','Clock +300 m'),
                ylabel='Fraction solved', ylim=(0.,1.15))
    for axis in axes:
        axis.tick_params(axis='x',labelrotation=18)
        axis.grid(axis='y',alpha=.2)
    fig.suptitle('Isolated per-PRN IQ -> assisted code ranges -> position / clock\n'
                 '16 replicas, 3 correlated scored epochs each; synthetic effects, no forest/rain validation')
    fig.savefig(directory/'weak_positioning.png',dpi=160)
    fig.savefig(directory/'weak_positioning.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('output/cpu-weak-positioning'))
    plot(parser.parse_args().directory)
