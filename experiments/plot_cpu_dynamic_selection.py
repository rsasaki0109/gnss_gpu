import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_dynamic_selection import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for i,ax in enumerate(axes):
        group=[c for c in cells if c['echo_voltage'] and c['phase_rad']==(0.,np.pi/2,np.pi)[i]]
        x=np.arange(len(group))
        ax.bar(x-.18,[c['forced_successes'] for c in group],width=.36,label='Forced two-path + local guards')
        ax.bar(x+.18,[c['successes'] for c in group],width=.36,label='Calibrated choice')
        ax.set(xticks=x,xticklabels=[str(c['doppler_hz']) for c in group],xlabel='Echo Doppler (Hz)',title=f'Initial phase {i}/2 pi' if i<2 else 'Initial phase pi',ylim=(0,18))
    axes[0].set_ylabel('Range error <10 m (out of 16)')
    axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Echo sensitivity after independent single-path-null calibration')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
