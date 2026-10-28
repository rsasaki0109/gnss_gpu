import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_wideband_cn0 import OUTPUT,LEVELS


def run():
    report=json.loads((OUTPUT/'report.json').read_text())
    echo=[c for c in report['cells'] if c['kind']=='echo']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    values=np.array([c['successes']/c['attempts'] for c in echo]);ci=np.array([c['success_exact_95'] for c in echo])
    axes[0].errorbar(LEVELS,values,yerr=[values-ci[:,0],ci[:,1]-values],fmt='o-',capsize=4)
    for level,c,v in zip(LEVELS,echo,values):axes[0].text(level,v+.04,f'{c["successes"]}/32',ha='center')
    axes[0].set(ylim=(0,1.15),ylabel='Available and error <10m',title='Echo success;pointwise exact95% intervals')
    axes[1].bar(LEVELS,[c['unavailable'] for c in echo],width=2,color='tab:orange',label='Unavailable')
    axes[1].bar(LEVELS,[c['available_large_errors'] for c in echo],bottom=[c['unavailable'] for c in echo],width=2,color='tab:red',label='Available but error >=10m')
    axes[1].set(ylim=(0,34),ylabel='Echo observations',title='Failure composition');axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xticks(LEVELS);ax.set_xlabel('Prefilter direct C/N0 (dB-Hz)');ax.grid(alpha=.2)
    p=report['primary_37_vs_49']['exact_two_sided_p']
    fig.suptitle(f'Fixed7.37MHz,800ms,known two paths;prespecified37vs49 p={p:.3g}')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
