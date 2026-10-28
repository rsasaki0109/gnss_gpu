"""Threshold and close-echo success variability across independent calibrations."""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_calibration_repeats import OUTPUT


def run():
    gates=json.loads((OUTPUT/'frozen_gates.json').read_text())
    report=json.loads((OUTPUT/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,coverage in enumerate(('white','multi')):
        x=np.arange(3)+(j-.5)*.16
        axes[0].plot(x,[gates[str(i)][coverage]['score'] for i in range(3)],'o',label=coverage+' calibration')
        cells=[r for r in report['cells'] if r['echo_voltage']!=0 and r['coverage']==coverage]
        axes[1].bar(x,[r['successes'] for r in cells],width=.16,label=coverage+' calibration')
    axes[0].set_ylabel('Frozen score threshold');axes[0].set_title('Same estimator; independent calibration data')
    axes[1].set_ylabel('Available error <10 m, out of 128');axes[1].set_ylim(0,128)
    axes[1].set_title('White noise, 0.1-chip same-phase echo')
    for ax in axes:
        ax.set_xticks(np.arange(3),['Draw 1','Draw 2','Draw 3']);ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('One shared held-out evaluation set; ideal 37 reference dB-Hz, 200 ms')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__':run()
