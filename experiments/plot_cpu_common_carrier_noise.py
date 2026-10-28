import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.stress_cpu_common_carrier_noise import OUTPUT,PROFILES


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    x=np.arange(3)
    for ax,ratio in zip(axes,(0.,.8)):
        group=[next(c for c in cells if c['profile']==p and c['echo_voltage']==ratio) for p in PROFILES]
        for shift,key,label in [(-.25,'successes','Range error <10 m'),(0.,'two_paths','Two-path selection'),(.25,'unavailable','Unavailable')]:
            ax.bar(x+shift,[c[key]/c['attempts'] for c in group],.25,label=label)
        ax.set(xticks=x,xticklabels=PROFILES,ylim=(0,1.12),title='No reflection (64/profile)' if not ratio else 'Reflection (32/profile)')
    axes[0].set_ylabel('Fraction; profiles share random realizations')
    axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.1),ncol=3)
    fig.suptitle('Frozen common-carrier receiver: spectral-noise mismatch')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':run()
