"""Descriptive variability over three calibration draws on shared evaluation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_ranked_repeats import OUTPUT


def run():
    report=json.loads((OUTPUT/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,method in enumerate(('maximum','ranked')):
        row=report['descriptive_variability'][method]
        x=np.arange(3)+(j-.5)*.15
        axes[0].plot(x,row['thresholds'],'o',label=method)
        axes[1].bar(x,row['successes_on_shared_echo_set'],width=.15,label=method)
    axes[0].set_title('Frozen thresholds');axes[0].set_ylabel('Score gate')
    axes[1].set_title('White same-phase close echo: error <10 m');axes[1].set_ylabel('Successes out of 128');axes[1].set_ylim(0,128)
    for ax in axes:
        ax.set_xticks(np.arange(3),['Preserved draw','New draw 1','New draw 2']);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same held-out observations; frozen estimator; rank fixed at 4; only three calibration draws')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':run()
