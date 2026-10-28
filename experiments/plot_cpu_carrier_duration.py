import json
import matplotlib.pyplot as plt
from experiments.run_cpu_carrier_duration import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for fd in (0.,.1,.7):
        g=[c for c in cells if c['relative_doppler_hz']==fd]
        for ax,key in zip(axes,('successes','unavailable')):
            ax.plot([c['duration_ms'] for c in g],[c[key]/8 for c in g],'o-',label=f'Relative Doppler {fd:g} Hz')
    for ax,title in zip(axes,('Range error <10 m','Unavailable')):
        ax.set(title=title,xlabel='Observation duration (ms)',ylim=(-.04,1.04),xticks=[200,400,800]);ax.grid(alpha=.25)
    axes[0].set_ylabel('Fraction (8 paired replicas per cell)');axes[1].legend(fontsize=8)
    fig.suptitle('Initial reverse phase; known two-path order; ideal coherence')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
