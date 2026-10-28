import json
import matplotlib.pyplot as plt
from experiments.run_cpu_common_carrier import OUTPUT


def run():
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,ratio in zip(axes,(0.,.8)):
        group=[c for c in cells if c['echo_voltage']==ratio]
        x=[c['residual_carrier_hz'] for c in group]
        for key,label,style in [('old_oracle_successes','Old fit, known order','o--'),
                                ('new_oracle_successes','Common-carrier fit, known order','s-')]:
            ax.plot(x,[c[key]/c['attempts'] for c in group],style,label=label)
        ax.set(title='No reflection: single-path fit' if not ratio else 'Reflection: two-path fit',
               xlabel='Residual common carrier (Hz)',ylim=(-.04,1.04))
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Range error <10 m (24 paired trials/cell)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Development replay with known model order — NOT a calibrated detector')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
