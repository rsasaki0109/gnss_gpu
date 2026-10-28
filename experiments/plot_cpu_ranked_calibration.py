"""Compare maximum and ranked gates without dropping unavailable estimates."""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_ranked_calibration import OUTPUT,PROFILES


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,method in enumerate(('maximum','ranked')):
        for ax,ratio in zip(axes,(0.,.8)):
            rows=[r for r in cells if r['method']==method and r['echo_voltage']==ratio]
            field='two_paths' if ratio==0 else 'successes'
            ax.bar(np.arange(3)+(j-.5)*.35,[r[field]/r['attempts'] for r in rows],width=.35,label=method)
    axes[0].set_title('No echo: false two-path selections (299/profile)')
    axes[1].set_title('Same-phase close echo: error <10 m (128/profile)')
    axes[0].set_ylabel('Proportion of all attempts');axes[1].set_ylim(0,1.05)
    for ax in axes:
        ax.set_xticks(np.arange(3),PROFILES);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Identical fresh calibration/observations; frozen noise-aware estimator; 37 dB-Hz, 200 ms')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':run()
