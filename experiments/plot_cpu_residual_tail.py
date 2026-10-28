import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.diagnose_cpu_residual_tail import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    supplement=json.loads((OUTPUT/'oracle_seed_check.json').read_text())['rows']
    for r,extra in zip(rows,supplement):
        assert (r['profile'],r['replica'])==(extra['profile'],extra['replica'])
        r['oracle_best_score']=extra['best_oracle_score']
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(rows))
    for key,label,marker in [('original_score','Original estimated weights','o'),('expanded_score','Expanded search, same weights','x'),('oracle_best_score','True PSD + extra seed (diagnostic)','+')]:
        ax.plot(x,[r[key] for r in rows],marker=marker,linestyle='none',label=label)
    ax.set(xticks=x,xticklabels=[f"{r['profile']}\nrank {r['selected_rank']}" for r in rows],ylabel='Path score',title='Selected calibration cases; retrospective diagnostic')
    ax.legend(fontsize=8);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'diagnosis.{ext}',dpi=160)

if __name__=='__main__':run()
