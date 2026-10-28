import json
import matplotlib.pyplot as plt
from experiments.diagnose_cpu_drift_alarm_misses import OUTPUT,SCALE


def run():
    p=json.loads((OUTPUT/'protocol.json').read_text())
    rows=[json.loads(x) for x in (OUTPUT/'profiles.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(3,4,figsize=(14,9),sharex=True,sharey=True)
    for ax,r,c in zip(axes.ravel(),rows,p['cases']):
        center=c['selected']['delay_chips']
        for model,points in r['profiles'].items():
            ax.plot([(q['delay_chips']-center)*SCALE for q in points],[q['relative_loss'] for q in points],'.-',ms=2,label=model)
        true=(c['true_delay_chips']-center)*SCALE
        oracle=r['oracle_fixed_true_delay']
        import numpy as np
        ax.scatter([true],[2*r['complex_count']*np.log(oracle['loss']/r['reference_loss'])],marker='*',c='black',s=45,label='truth-fixed two-path')
        ax.axvline(true,color='gray',ls='--')
        ax.set(title=f'{r["replica"]}: {c["selected"]["model"]}',yscale='symlog',ylim=(-.1,1000),xlim=(-45,45))
        ax.grid(alpha=.2)
    axes.ravel()[-1].axis('off')
    axes[0,0].legend(fontsize=7)
    for ax in axes[-1,:3]:ax.set_xlabel('Offset from selected delay (m)')
    for ax in axes[:,0]:ax.set_ylabel('Uncalibrated relative loss')
    fig.suptitle('All 11 missed large errors: retrospective profiles and separate truth-fixed diagnostic')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
