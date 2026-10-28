import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_residual_repeat import OUTPUT


def run():
    report=json.loads((OUTPUT/'report.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(2)
    for method in ('maximum','ranked'):
        axes[0].plot(x,report['descriptive_variability'][method]['thresholds'],'o-',label=method)
    axes[0].set(xticks=x,xticklabels=['Original draw','Independent draw'],ylabel='Frozen score threshold',title='Two calibration draws, descriptive only')
    axes[0].legend()
    for i,(draw,method) in enumerate([(d,m) for d in ('original','new') for m in ('maximum','ranked')]):
        cells=[c for c in report['cells'] if c['draw']==draw and c['method']==method and c['echo_voltage']]
        axes[1].bar(np.arange(3)+(i-1.5)*.2,[c['successes'] for c in cells],width=.2,label=f'{draw} {method}')
    axes[1].set(xticks=range(3),xticklabels=['white','center','edge'],ylabel='Range error <10 m (out of 64)',title='Identical fresh echo observations',ylim=(0,70))
    axes[1].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
