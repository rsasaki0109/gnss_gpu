import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_matched_dynamic_reflection import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for i,ax in enumerate(axes):
        group=cells[3*i:3*i+3];x=np.arange(3)
        for offset,key,label in [(-.18,'baseline_successes','Matched static'),(.18,'selected_successes','Dynamic')]:
            ax.bar(x+offset,[100*c[key]/c['attempts'] for c in group],width=.36,label=label)
        ax.set(xticks=x,xticklabels=['0','pi/2','pi'],xlabel='Initial echo phase (rad)',title=f"{group[0]['doppler_hz']} Hz",ylim=(0,105))
    axes[0].set_ylabel('Range error <10 m (%)')
    axes[0].legend(loc='lower left')
    fig.suptitle('Same mean signal and noise; 64 paired trials per condition')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
