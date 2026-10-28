import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_moving_echo_pilot import OUTPUT,DOPPLERS


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for mode,shift,color in [('fixed_code',-.16,'tab:blue'),('moving_code',.16,'tab:orange')]:
        g=[c for c in cells if c['mode']==mode]
        x=np.arange(4)+shift;values=[c['successes'] for c in g]
        axes[0].bar(x,values,width=.3,color=color,label=mode)
        for xx,n in zip(x,values):axes[0].text(xx,n+.3,str(n),ha='center')
    axes[0].set(xticks=range(4),xticklabels=[str(f) for f in DOPPLERS],ylim=(0,19),xlabel='Relative carrier Doppler (Hz)',ylabel='Successful /16',title='Known two paths;49dB-Hz')
    axes[0].legend(loc='lower right',fontsize=8)
    for j,fd in enumerate(DOPPLERS):
        fixed={r['replica']:r for r in rows if r['mode']=='fixed_code' and r['relative_doppler_hz']==fd}
        moving={r['replica']:r for r in rows if r['mode']=='moving_code' and r['relative_doppler_hz']==fd}
        delta=[moving[i]['raw_error_m']-fixed[i]['raw_error_m'] for i in fixed]
        axes[1].scatter(np.full(len(delta),j),delta,s=22,alpha=.7)
    axes[1].axhline(0,color='gray',ls=':');axes[1].set(xticks=range(4),xticklabels=[str(f) for f in DOPPLERS],xlabel='Relative carrier Doppler (Hz)',ylabel='Moving minus fixed direct-range estimate (m)',title='Paired estimate changes;includes unavailable fits')
    fig.suptitle('Consistent code/carrier motion;frozen-code receiver unchanged')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
