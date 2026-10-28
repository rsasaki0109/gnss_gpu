import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.diagnose_cpu_common_carrier_boundary import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    x=np.arange(1,len(rows)+1)
    for name,label,marker in [('original','Original','o'),('same_bounds','Expanded starts','s'),('relaxed_floor','Gap floor 0.001','^')]:
        for ax,key in zip(axes,('loss','first_std_chips','cancellation_ratio')):
            values=[r[name][key] if r[name][key] is not None else np.nan for r in rows]
            if key=='loss':values=[(v-r['original']['loss'])/r['original']['loss'] for v,r in zip(values,rows)]
            ax.plot(x,values,marker=marker,label=label,alpha=.8)
    for ax,label in zip(axes,('Relative SSE change','Local delay std (chips)','Cancellation ratio')):
        ax.set(xlabel='Retrospective case index',ylabel=label);ax.grid(alpha=.2)
    axes[1].axhline(.1,color='black',ls='--',label='Local std guard')
    axes[1].set_yscale('log');axes[2].set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed-weight boundary diagnosis — minima are not accepted ranges')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
