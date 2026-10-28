import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_drift_model_pilot import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,kind in zip(axes,('null','echo')):
        for rate in (0.,.5,2.):
            g=[r for r in rows if r['kind']==kind and r['drift_hz_s']==rate]
            for r in g:ax.plot([rate-.07,rate+.07],[r['old_score'],r['uncalibrated_score']],color='gray',alpha=.35,lw=.8)
            ax.scatter([rate-.07]*len(g),[r['old_score'] for r in g],s=20,color='tab:blue',label='Constant carrier' if rate==0 else None)
            ax.scatter([rate+.07]*len(g),[r['uncalibrated_score'] for r in g],s=20,color='tab:orange',label='Estimated drift' if rate==0 else None)
        ax.set(title='No reflection' if kind=='null' else 'Reflection present',xlabel='Common drift (Hz/s)',xticks=[0,.5,2]);ax.grid(alpha=.2)
    axes[0].set_ylabel('Uncalibrated two-path ranking score');axes[1].legend(fontsize=8)
    fig.suptitle('800 ms development replay — no detection threshold applied')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
