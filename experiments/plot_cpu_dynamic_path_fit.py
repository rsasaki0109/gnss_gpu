import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_dynamic_path_fit import OUTPUT


def run():
    cells=[c for c in json.loads((OUTPUT/'report.json').read_text())['cells'] if c['panel']=='fresh']
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(cells))
    ax.bar(x-.18,[c['old_successes'] for c in cells],width=.36,label='Old calibrated selector')
    ax.bar(x+.18,[c['new_local_successes'] for c in cells],width=.36,label='New forced two-path fit + local guards')
    ax.set(xticks=x,xticklabels=[f"{c['doppler_hz']} Hz\n{c['phase_rad']/np.pi:g} pi" for c in cells],ylim=(0,9),ylabel='Range error <10 m (out of 8)',xlabel='Echo Doppler / initial phase',title='Fresh pilot; path-count false alarms not calibrated')
    ax.legend(loc='lower left');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)

if __name__=='__main__':run()
