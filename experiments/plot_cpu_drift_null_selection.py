import json
import matplotlib.pyplot as plt
from experiments.run_cpu_drift_null_selection import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for duration in (200,800):
        g=[c for c in cells if c['duration_ms']==duration]
        for ax,key in zip(axes,('false_two_paths','unavailable','successes')):
            ax.plot([c['drift_hz_s'] for c in g],[c[key]/32 for c in g],'o-',label=f'{duration} ms')
    for ax,title in zip(axes,('False two-path selection','Unavailable','Range error <10 m')):
        ax.set(title=title,xlabel='Common frequency drift (Hz/s)',ylim=(-.04,1.04),xticks=[0,.5,2]);ax.grid(alpha=.25)
    axes[0].set_ylabel('Fraction (32 shared null replicas)');axes[2].legend()
    fig.suptitle('No reflection: gates calibrated with zero drift only')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
