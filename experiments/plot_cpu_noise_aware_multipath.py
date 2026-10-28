"""Plot held-out errors and close-echo success, retaining all attempts."""
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_noise_aware_multipath import OUTPUT,PROFILES


def run():
    report=json.loads((OUTPUT/'report.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    null=[r for r in report if r['echo_voltage']==0]
    echoes=[r for r in report if r['echo_voltage']!=0]
    for j,(name,color) in enumerate((('Old','#d77c31'),('Noise-aware','#2979a4'))):
        field='old_two_paths' if j==0 else 'new_two_paths'
        axes[0].bar(np.arange(3)+(j-.5)*.36,[r[field]/r['attempts'] for r in null],width=.36,label=name,color=color)
        field='baseline_successes' if j==0 else 'selected_successes'
        axes[1].bar(np.arange(6)+(j-.5)*.36,[r[field]/r['attempts'] for r in echoes],width=.36,label=name,color=color)
    axes[0].set_xticks(np.arange(3),PROFILES)
    axes[0].set_title('No echo: false two-path selections (128 per profile)')
    axes[0].set_ylabel('Proportion of all attempts')
    axes[1].set_xticks(np.arange(6),[f'{r["profile"]}\n'+('phase 0' if r['phase_rad']==0 else 'pi/2') for r in echoes])
    axes[1].set_title('0.1-chip echo: range error <10 m (64 per cell)')
    axes[1].set_ylim(0,1.05)
    for ax in axes: ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Estimated noise weighting + recalibration; fixed 37 reference dB-Hz, ideal 200 ms coherence')
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(OUTPUT/f'comparison.{ext}',dpi=150)
    plt.close(fig)


if __name__=='__main__': run()
