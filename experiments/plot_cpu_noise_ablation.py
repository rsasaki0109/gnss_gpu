"""Plot false echoes and same-phase close-echo successes for all four arms."""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_noise_ablation import OUTPUT,ARMS,PROFILES


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for j,arm in enumerate(ARMS):
        label=arm.replace('_',' / ')
        for ax,ratio in zip(axes,(0.,.8)):
            rows=[r for r in cells if r['arm']==arm and r['echo_voltage']==ratio and r['phase_rad']==0]
            field='two_paths' if ratio==0 else 'successes'
            ax.bar(np.arange(3)+(j-1.5)*.2,[r[field]/r['attempts'] for r in rows],width=.2,label=label)
    axes[0].set_title('No echo: false two-path selection (128 per profile)')
    axes[1].set_title('0.1-chip same-phase echo: error <10 m (64 per profile)')
    axes[0].set_ylabel('Proportion of all attempts')
    axes[1].set_ylim(0,1.05)
    for ax in axes:
        ax.set_xticks(np.arange(3),PROFILES);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2)
    fig.suptitle('Fixed estimators; fresh paired calibration and evaluation; 37 reference dB-Hz, 200 ms')
    fig.tight_layout(rect=(0,.12,1,1))
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':run()
