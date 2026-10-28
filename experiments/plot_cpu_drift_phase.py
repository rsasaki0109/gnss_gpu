import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.stress_cpu_drift_phase import OUTPUT


def run():
    cells=[c for c in json.loads((OUTPUT/'report.json').read_text())['cells'] if c['kind']=='echo']
    fig,axes=plt.subplots(1,3,figsize=(11,3.7))
    for ax,key,title in zip(axes,('successes','two_paths','unavailable'),('Range error <10 m','Two-path selections','Unavailable')):
        values=np.array([c[key] for c in cells]).reshape(3,3)
        ax.imshow(values,vmin=0,vmax=16,cmap='Blues')
        for (i,j),value in np.ndenumerate(values):ax.text(j,i,f'{value}/16',ha='center',va='center',color='white' if value>8 else 'black')
        ax.set(xticks=range(3),xticklabels=['0','90','180'],yticks=range(3),yticklabels=['0','.1','.7'],xlabel='Initial phase (degrees)',title=title)
    axes[0].set_ylabel('Relative Doppler (Hz)')
    fig.suptitle('Frozen drift-aware receiver,800 ms;16 shared replicas per cell')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
