import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_residual_ablation import OUTPUT,ARMS


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(3)
    for i,arm in enumerate(ARMS):
        for ax,ratio,n,key in [(axes[0],0.,128,'two_paths'),(axes[1],.8,64,'successes')]:
            group=[c for c in cells if c['arm']==arm and c['echo_voltage']==ratio]
            ax.bar(x+(i-1.5)*.2,[100*c[key]/n for c in group],width=.2,label=arm)
    for ax in axes:ax.set(xticks=x,xticklabels=['white','center','edge'])
    axes[0].set(ylabel='False two-path selection (%)',title='128 null trials / noise profile')
    axes[1].set(ylabel='Range error <10 m (%)',title='64 static echo trials / noise profile',ylim=(0,105))
    axes[0].legend(fontsize=7);fig.suptitle('Same calibration samples and maximum rule; fresh evaluation');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
