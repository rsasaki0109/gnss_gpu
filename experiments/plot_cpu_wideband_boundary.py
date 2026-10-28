import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.diagnose_cpu_wideband_boundary import OUTPUT,CHIP_RATE,LIGHT_SPEED


def run():
    rows=[json.loads(x) for x in (OUTPUT/'diagnostics.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,label,log in zip(axes,('first_std_chips','cancellation_ratio','error_m'),
                               ('Local delay std (chips)','Component cancellation ratio','Signed range error (m)'),(True,True,False)):
        points=[(r['original'][key],r['relaxed_floor'][key]) for r in rows if r['original'][key] is not None and r['relaxed_floor'][key] is not None]
        if not points:
            ax.text(.5,.5,'No finite uncertainty pairs',ha='center',transform=ax.transAxes);ax.set_title(label);continue
        x,y=np.array(points).T
        ax.scatter(x,y,s=24)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],color='gray',ls=':')
        if log:ax.set_xscale('log');ax.set_yscale('log')
        ax.set(xlabel='Original floor .01 chip',ylabel='Relaxed floor .001 chip',title=label)
        ax.grid(alpha=.2)
    fig.suptitle('All29 wideband boundary failures;retrospective frozen-weight diagnosis')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
