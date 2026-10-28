import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_common_carrier_selection import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    gate=json.loads((OUTPUT/'frozen_gate.json').read_text())['score']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ratio,label in [(0.,'No reflection (128)'),(.8,'Reflection (64)')]:
        g=[r for r in rows if r['echo_voltage']==ratio]
        axes[0].scatter([r['residual_carrier_hz'] for r in g],[r['uncalibrated_score'] for r in g],s=14,alpha=.7,label=label)
    axes[0].axhline(gate,color='black',linestyle='--',label='Frozen null gate')
    axes[0].set(xlabel='Common residual carrier (Hz)',ylabel='Selection score')
    axes[0].legend(fontsize=8)
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    x=np.arange(2)
    axes[1].bar(x-.18,[c['successes']/c['attempts'] for c in cells],.36,label='Automatic selection')
    axes[1].bar(x+.18,[c['oracle_successes']/c['attempts'] for c in cells],.36,label='Known-order diagnostic')
    axes[1].set(xticks=x,xticklabels=['No reflection','Reflection'],ylim=(0,1.08),ylabel='Range error <10 m (unavailable counts as failure)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Heldout evaluation: uniform [-1, 1] Hz carrier mixture')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
