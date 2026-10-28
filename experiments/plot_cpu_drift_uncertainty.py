import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.validate_cpu_drift_uncertainty import OUTPUT


def run():
    rows = [json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax, ratio, title in zip(axes,(0.,.2,.8),('No echo','Weak close echo','Strong close echo')):
        g = [r for r in rows if r['echo_voltage']==ratio]
        for model, color, marker in [('single_path','tab:blue','o'),('two_path','tab:orange','^')]:
            chosen = [r for r in g if r['selected']['model']==model]
            ax.scatter([r['uncertainty']['half_width_m'] for r in chosen],
                       [abs(r['error_m']) for r in chosen],c=color,marker=marker,s=27,label=model.replace('_',' '),alpha=.75)
        available = [r for r in g if r['uncertainty']['available']]
        limit = max([32]+[r['uncertainty']['half_width_m']*1.05 for r in available]+[abs(r['error_m'])*1.05 for r in available])
        ax.plot([0,limit],[0,limit],color='gray',ls=':',label='local interval boundary')
        ax.axhline(10,color='red',lw=.8);ax.axvline(10,color='red',lw=.8)
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Local 95% half-width (m)',
               title=f'{title}\n{len(g)-len(available)}/{len(g)} unavailable')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Absolute range error (m)')
    axes[0].legend(fontsize=7)
    fig.suptitle('Fresh replication: local curvature does not include model uncertainty')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
