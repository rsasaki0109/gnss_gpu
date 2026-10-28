import json
import matplotlib.pyplot as plt
from experiments.run_cpu_drift_profile_alarm import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gate=json.loads((OUTPUT/'frozen_gate.json').read_text())['score']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True)
    for ax,ratio,title in zip(axes,(0.,.2,.8),('No echo','Weak echo','Strong echo')):
        g=[r for r in rows if r['echo_voltage']==ratio]
        for model,marker in [('single_path','o'),('two_path','^')]:
            chosen=[r for r in g if r['error_m'] is not None and r['selected']['model']==model]
            ax.scatter([r['profile_alarm']['score'] for r in chosen],[abs(r['error_m']) for r in chosen],
                marker=marker,s=25,alpha=.7,label=model.replace('_',' '))
        ax.axvline(gate,color='red',ls='--',label='Frozen warning gate')
        ax.axhline(10,color='gray',ls=':')
        ax.set(title=f'{title}; {sum(r["warning"] for r in g)}/{len(g)} warnings\n{sum(r["error_m"] is None for r in g)} unavailable',xlabel='Distant-fit warning score')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Absolute range error (m)');axes[0].legend(fontsize=7)
    fig.suptitle('Independent evaluation; diagnostic applied only to single-path selections')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
