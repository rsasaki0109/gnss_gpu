import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.replicate_cpu_drift_bandwidth import OUTPUT


def run():
    report=json.loads((OUTPUT/'report.json').read_text());p=report['primary_echo_comparison']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind,shift,color in [('null',-.16,'tab:blue'),('echo',.16,'tab:orange')]:
        cells=[c for c in report['cells'] if c['kind']==kind]
        values=np.array([c['successes']/c['attempts'] for c in cells]);ci=np.array([c['success_exact_95'] for c in cells])
        x=np.arange(2)+shift
        axes[0].bar(x,values,width=.3,label=kind,color=color)
        axes[0].errorbar(x,values,yerr=[values-ci[:,0],ci[:,1]-values],fmt='none',color='black',capsize=3)
        for xx,c,v in zip(x,cells,values):axes[0].text(xx,v+.025,f'{c["successes"]}/{c["attempts"]}',ha='center',fontsize=9)
    axes[0].set(xticks=[0,1],xticklabels=['3.68 MHz','7.37 MHz'],ylim=(0,1.18),xlabel='One-sided bandwidth',ylabel='Available and error <10m',title='Known model order; pointwise exact95% intervals')
    axes[0].legend(loc='lower right')
    counts=[p[k] for k in ('both_failure','gains','losses','both_success')]
    axes[1].bar(range(4),counts,color=['gray','tab:green','tab:red','tab:blue'])
    for i,n in enumerate(counts):axes[1].text(i,n+.5,str(n),ha='center')
    axes[1].set(xticks=range(4),xticklabels=['Both fail','Recovered','Lost','Both pass'],ylabel='Echo pairs',ylim=(0,max(counts)+6),title=f'Prespecified paired contrast; p={p["exact_two_sided_p"]:.3g}')
    fig.suptitle('Independent replication;64 echo pairs and32 null pairs')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
