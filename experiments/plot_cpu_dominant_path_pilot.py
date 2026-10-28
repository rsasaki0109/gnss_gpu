import json
import matplotlib.pyplot as plt
from experiments.run_cpu_dominant_path_pilot import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,ratio in zip(axes,(.8,1.2)):
        group=[r for r in rows if r['echo_voltage']==ratio]
        for i,r in enumerate(group):
            color='tab:orange' if r['constrained']['amplitude_constraint_active'] else 'tab:blue'
            ax.plot([0,1],[abs(r['free_error_m']),abs(r['constrained_error_m'])],color=color,alpha=.5)
            for x,key in [(0,'free'),(1,'constrained')]:
                ax.scatter([x],[abs(r[key+'_error_m'])],color=color,marker='o' if r[key+'_eligible'] else 'x',s=22)
        ax.axhline(10,color='gray',ls='--')
        ax.set(xticks=[0,1],xticklabels=['Unconstrained','Dominant first'],xlim=(-.2,1.2),
            title=f'Echo/direct voltage {ratio}\n'+('Assumption holds' if ratio<1 else 'Assumption violated'))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Absolute range error (m)')
    fig.suptitle('Known-two-path pilot; orange: amplitude constraint active; x: ineligible')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
