import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_dynamic_reflection import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    labels=['Direct only','Static echo','1 Hz','5 Hz','10 Hz']
    y=np.array([c['successes']/c['attempts'] for c in cells])
    ci=np.array([c['success_exact_95'] for c in cells])
    axes[0].bar(labels,y*100,color=['gray','steelblue','teal','teal','teal'])
    axes[0].errorbar(labels,y*100,yerr=np.maximum(0,np.vstack((y-ci[:,0],ci[:,1]-y)))*100,fmt='none',color='black',capsize=3)
    axes[0].set(ylabel='Range error <10 m (%)',ylim=(0,105),title='128 paired trials / condition; pointwise 95% CI')
    axes[1].plot(labels,[c['mean_estimated_white_weight_rms'] for c in cells],'o-')
    axes[1].set(ylabel='RMS deviation of estimated weights from white',title='Observed noise-weight distortion')
    for ax in axes:ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Frozen ranked gate; synthetic narrowband phase dynamics')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
