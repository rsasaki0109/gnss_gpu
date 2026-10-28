import json
import matplotlib.pyplot as plt
from experiments.stress_cpu_carrier_drift import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for duration in (200,800):
        g=[c for c in cells if c['duration_ms']==duration]
        for ax,key in zip(axes,('successes','unavailable')):
            ax.plot([c['drift_hz_s'] for c in g],[c[key]/16 for c in g],'o-',label=f'{duration} ms')
    for ax,title in zip(axes,('Range error <10 m','Unavailable')):
        ax.set(title=title,xlabel='Common frequency drift (Hz/s)',ylim=(-.04,1.04),xticks=[0,.5,2]);ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel('Fraction (16 shared replicas per cell)')
    fig.suptitle('Frozen constant-carrier fit; known two-path order')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
