import json
import matplotlib.pyplot as plt
from experiments.diagnose_cpu_path_boundary import OUTPUT


def run():
    rows=[json.loads(line) for line in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for key,label,marker in [('original','Original','o'),('dense_same_bounds','More starts','x'),('relaxed_gap','Gap floor 0.001','+')]:
        axes[0].scatter(range(1,30),[r[key]['parameters'][1] for r in rows],label=label,marker=marker)
        axes[1].scatter([r[key]['parameters'][1] for r in rows],[r[key]['cancellation_ratio'] for r in rows],label=label,marker=marker)
    axes[0].set(yscale='log',xlabel='Unavailable case index (29 cases)',ylabel='Fitted path separation (chips)')
    axes[1].set(xscale='log',yscale='log',xlabel='Fitted path separation (chips)',ylabel='Sum of path norms / combined signal norm')
    axes[0].legend()
    fig.suptitle('Retrospective diagnostic; receiver guards unchanged')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'diagnosis.{ext}',dpi=160)

if __name__=='__main__':run()
