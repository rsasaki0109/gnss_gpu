import json
import matplotlib.pyplot as plt
from experiments.diagnose_cpu_relative_drift_search import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True)
    for ax,(fd,rate) in zip(axes,((.7,-2.),(-.7,2.))):
        group=[r for r in rows if r['initial_doppler_hz']==fd and r['relative_drift_hz_s']==rate]
        for r in group:
            ax.plot([0,1],[r[k]['raw_error_m'] for k in ('original','broad')],color='grey',alpha=.3,lw=.7)
            for i,k in enumerate(('original','broad')):
                ax.scatter(i,r[k]['raw_error_m'],s=20,color=('tab:blue','tab:orange')[i],
                           marker='o' if r[k]['available'] else 'x')
        ax.axhline(0,color='grey',ls='--',lw=.7)
        ax.set(xticks=[0,1],xticklabels=['Previous search','Broader search'],xlim=(-.3,1.3),
               title=f'Initial {fd:+.1f} Hz; relative drift {rate:+.0f} Hz/s')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Direct range error (m)')
    fig.suptitle('Retrospective zero-crossing diagnosis | 16 paired cases per cell\nSame model, observations and weights; dots usable / crosses unavailable')
    fig.tight_layout();fig.savefig(OUTPUT/'summary.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
