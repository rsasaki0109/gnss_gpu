import json
import matplotlib.pyplot as plt
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE
from experiments.run_cpu_drift_delay_profile import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'profiles.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for ax,r in zip(axes.T.ravel(),rows):
        for model,points in r['profiles'].items():
            x=[(p['delay_chips']-r['selected_delay_chips'])*LIGHT_SPEED/CHIP_RATE for p in points]
            ax.plot(x,[p['relative_loss'] for p in points],'.-',label=model+' path',ms=3)
            bad=[i for i,p in enumerate(points) if not p['converged']]
            if bad:ax.scatter([x[i] for i in bad],[points[i]['relative_loss'] for i in bad],marker='x',c='red')
        truth=(r['true_delay_chips']-r['selected_delay_chips'])*LIGHT_SPEED/CHIP_RATE
        ax.axvline(truth,color='black',ls='--',label='true delay')
        h=r['local_half_width_m'];ax.axvspan(-h,h,color='gray',alpha=.15,label='local Gaussian interval')
        ax.set(title=f'Voltage {r["echo_voltage"]}; ID {r["replica"]}',yscale='symlog',ylim=(-.1,3000),xlim=(-45,45))
        ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Delay relative to selected estimate (m)')
    for ax in axes[:,0]:ax.set_ylabel('Relative loss (uncalibrated)')
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Retrospective delay profiles: frozen residual weights, finite multistart search')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
