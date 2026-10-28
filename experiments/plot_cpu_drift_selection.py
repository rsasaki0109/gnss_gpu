import json
import matplotlib.pyplot as plt
from experiments.run_cpu_drift_selection import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gate=json.loads((OUTPUT/'frozen_gate.json').read_text())['score']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for kind,label in [('null','No reflection'),('echo','Reflection')]:
        g=[r for r in rows if r['kind']==kind]
        axes[0].scatter([r['drift_hz_s'] for r in g],[r['uncalibrated_score'] for r in g],s=20,alpha=.7,label=label)
    axes[0].axhline(gate,color='black',ls='--',label='Frozen gate')
    axes[0].set(xlabel='True common drift (Hz/s)',ylabel='Two-path ranking score');axes[0].legend(fontsize=8)
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    for shift,key,label in [(-.18,'successes','Automatic selection'),(.18,'oracle_successes','Known-order diagnostic')]:
        axes[1].bar([i+shift for i in range(2)],[c[key]/c['attempts'] for c in cells],.36,label=label)
    axes[1].set(xticks=[0,1],xticklabels=['No reflection (64)','Reflection (32)'],ylim=(0,1.1),ylabel='Range error <10 m; unavailable counts as failure')
    axes[1].legend(fontsize=8)
    fig.suptitle('Drift-aware receiver: fresh signed-mixture evaluation,800 ms')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
