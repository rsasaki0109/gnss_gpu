import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_residual_dynamic import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(3)
    for offset,method,label in [(-.18,'old','Old white-noise gate'),(.18,'weighted','Residual weights + new gate')]:
        group=[c for c in cells if c['method']==method and not c['echo_voltage']]
        axes[0].bar(x+offset,[100*c['two_paths']/c['attempts'] for c in group],width=.36,label=label)
    axes[0].set(xticks=x,xticklabels=['white','center','edge'],ylabel='False two-path selection (%)',title='128 null trials / profile')
    axes[0].legend(fontsize=8)
    for profile in ('white','center','edge'):
        group=[c for c in cells if c['method']=='weighted' and c['profile']==profile and c['echo_voltage']]
        axes[1].plot(x,[c['successes'] for c in group],'o-',label=profile)
    axes[1].set(xticks=x,xticklabels=['0','0.1','0.7'],xlabel='Echo Doppler (Hz)',ylabel='Range error <10 m (out of 16)',ylim=(0,17),title='Weighted + calibrated echo results')
    axes[1].legend();fig.suptitle('Joint weighting and recalibration change; synthetic profiles');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
