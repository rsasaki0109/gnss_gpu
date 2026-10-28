import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.stress_cpu_drift_gap_amplitude import OUTPUT


def run():
    cells=[c for c in json.loads((OUTPUT/'report.json').read_text())['cells'] if c['kind']=='echo']
    fig,axes=plt.subplots(2,3,figsize=(11,6.5))
    for row,ratio in enumerate((.2,.8)):
        group=[c for c in cells if c['echo_voltage']==ratio]
        for ax,key,title in zip(axes[row],('successes','two_paths','unavailable'),('Range error <10 m','Two-path selections','Unavailable')):
            values=np.array([c[key] for c in group]).reshape(3,2)
            ax.imshow(values,vmin=0,vmax=16,cmap='Blues',aspect='auto')
            for (i,j),value in np.ndenumerate(values):ax.text(j,i,f'{value}/16',ha='center',va='center',color='white' if value>8 else 'black')
            ax.set(xticks=[0,1],xticklabels=['0','180'],yticks=[0,1,2],yticklabels=['.03','.1','.3'],xlabel='Initial phase (degrees)',title=f'{title}; voltage {ratio}')
        axes[row,0].set_ylabel('Excess delay (chips)')
    fig.suptitle('Frozen drift-aware receiver,800ms; relative Doppler0Hz')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
