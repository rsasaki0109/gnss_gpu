import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_residual_ranked import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(3)
    for offset,method in [(-.18,'maximum'),(.18,'ranked')]:
        for ax,ratio,key in [(axes[0],0.,'two_paths'),(axes[1],.8,'successes')]:
            group=[c for c in cells if c['method']==method and c['echo_voltage']==ratio]
            ax.bar(x+offset,[c[key] for c in group],width=.36,label=method)
    for ax in axes:ax.set(xticks=x,xticklabels=['white','center','edge']);ax.legend()
    axes[0].set(ylabel='False two-path count (out of 128)',title='Heldout single-path control',ylim=(0,4),yticks=range(5))
    for i in range(3):axes[0].text(i,.15,'Both 0/128',ha='center',fontsize=9)
    axes[1].set(ylabel='Range error <10 m (out of 64)',title='Static same-phase echo',ylim=(0,70))
    fig.suptitle('Residual-weighted estimator; fresh three-profile rank calibration');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
