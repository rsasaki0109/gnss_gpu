import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.stress_cpu_wideband_phase_doppler import OUTPUT,LEVELS,FREQUENCIES,PHASES


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for row,cn0 in enumerate(LEVELS):
        group=[c for c in cells if c['cn0_dbhz']==cn0]
        for ax,key,title in zip(axes[row],('successes','unavailable','available_large_errors'),('Available and error <10m','Unavailable','Available but error >=10m')):
            values=np.array([c[key] for c in group]).reshape(2,3)
            ax.imshow(values,vmin=0,vmax=16,cmap='Blues',aspect='auto')
            for (i,j),v in np.ndenumerate(values):ax.text(j,i,f'{v}/16',ha='center',va='center',color='white' if v>8 else 'black')
            ax.set(xticks=[0,1,2],xticklabels=['0','90','180'],yticks=[0,1],yticklabels=['0','.7'],xlabel='Initial relative phase (degrees)',title=f'{title};{cn0:g}dB-Hz')
        axes[row,0].set_ylabel('Relative Doppler (Hz)')
    fig.suptitle('Frozen wideband receiver;known two-path order;16 shared replicas per cell')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
