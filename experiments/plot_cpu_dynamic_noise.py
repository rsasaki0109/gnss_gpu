import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.stress_cpu_dynamic_noise import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    null=[c for c in cells if c['method']=='ranked' and c['echo_voltage']==0]
    y=np.array([100*c['two_paths']/c['attempts'] for c in null])
    ci=100*np.array([c['two_path_exact_95'] for c in null])
    axes[0].bar([c['profile'] for c in null],y)
    axes[0].errorbar([c['profile'] for c in null],y,yerr=np.maximum(0,np.vstack((y-ci[:,0],ci[:,1]-y))),fmt='none',color='black',capsize=3)
    axes[0].axhline(1,color='red',linestyle='--',label='1% target')
    axes[0].set(ylabel='False two-path selection (%)',title='Ranked gate; 128 null trials / profile')
    axes[0].legend()
    for profile in ('white','center','edge'):
        echo=[c for c in cells if c['profile']==profile and c['method']=='ranked' and c['echo_voltage']]
        axes[1].plot(range(3),[100*c['successes']/c['attempts'] for c in echo],'o-',label=profile)
    axes[1].set(xticks=range(3),xticklabels=['0','0.1','0.7'],xlabel='Echo Doppler (Hz)',ylabel='Range error <10 m (%)',title='32 paired echo trials / condition',ylim=(0,105))
    axes[1].legend();fig.suptitle('Fixed white-noise calibration under spectral mismatch');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
