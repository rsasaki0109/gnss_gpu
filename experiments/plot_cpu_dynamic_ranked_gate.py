import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_dynamic_ranked_gate import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    for i,(method,label) in enumerate([('maximum','Maximum, same 1000'),('ranked','Fifth largest, same 1000'),('prior299','Previous maximum, 299')]):
        group=[c for c in cells if c['method']==method and c['echo_voltage']]
        ax.bar(x+(i-1)*.25,[c['successes'] for c in group],width=.25,label=label)
    ax.set(xticks=x,xticklabels=['0 Hz','0.1 Hz','0.7 Hz'],ylim=(0,140),ylabel='Range error <10 m (out of 128)',xlabel='Echo Doppler, initial phase zero',title='Fixed dynamic estimator; independent ranked calibration')
    ax.legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
