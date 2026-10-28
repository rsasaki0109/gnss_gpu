import json
import matplotlib.pyplot as plt
from experiments.run_cpu_relative_drift_pilot import OUTPUT,DOPPLERS,RATES


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,3,figsize=(10,6),sharey=True)
    for i,fd in enumerate(DOPPLERS):
        for j,rate in enumerate(RATES):
            ax=axes[i,j];g=[r for r in rows if r['initial_doppler_hz']==fd and r['relative_drift_hz_s']==rate]
            for r in g:
                models=[r['models'][name] for name in ('common_drift','relative_drift')]
                ax.plot([0,1],[m['raw_error_m'] for m in models],color='grey',alpha=.3,lw=.7)
                for x,m in enumerate(models):
                    ax.scatter(x,m['raw_error_m'],s=15,marker='o' if m['available'] else 'x',color=('tab:blue','tab:orange')[x])
            ax.axhline(0,color='grey',ls='--',lw=.7)
            ax.set(xticks=[0,1],xticklabels=['Common drift','Echo drift'],xlim=(-.3,1.3),
                   title=f'Initial {fd:+.1f} Hz; drift {rate:+.0f} Hz/s')
            ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Direct range error (m)');axes[1,0].set_ylabel('Direct range error (m)')
    fig.suptitle('Independent echo drift refinement | 16 fresh paired replicas/cell\n49 dB-Hz, 800 ms, known two-path order; same noise weights')
    fig.tight_layout();fig.savefig(OUTPUT/'summary.png',dpi=160);plt.close(fig)


if __name__=='__main__':run()
