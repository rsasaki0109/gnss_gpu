import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_cpu_drift_bandwidth import OUTPUT,BANDS


def run():
    rows=[json.loads(x) for x in (OUTPUT/'evaluation.jsonl').read_text().splitlines()]
    cells=json.loads((OUTPUT/'report.json').read_text())['cells']
    timings=json.loads((OUTPUT/'timings.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for replica in sorted({r['replica'] for r in rows}):
        g=[next(r for r in rows if r['replica']==replica and r['bandwidth_chips']==b) for b in BANDS]
        axes[0].plot(BANDS,[abs(r['raw_error_m']) for r in g],color='tab:blue',alpha=.3)
        for r in g:axes[0].scatter([r['bandwidth_chips']],[abs(r['raw_error_m'])],color='tab:blue',marker='o' if r['available'] else 'x',s=20)
    axes[0].axhline(10,color='gray',ls='--');axes[0].set_ylabel('Absolute range error (m)')
    axes[0].set_title('Paired observations; x = unavailable')
    axes[1].bar(BANDS,[c['successes'] for c in cells],width=.8)
    for b,c in zip(BANDS,cells):axes[1].text(b,c['successes']+.3,f'{c["successes"]}/16',ha='center')
    axes[1].set(ylim=(0,18),ylabel='Available and error <10m',title='Known two-path order')
    for r in timings:axes[2].scatter(r['bandwidth_chips'],r['elapsed_s'],color='tab:orange')
    axes[2].set(ylabel='Sequential pipeline time (s)',title='Two observations per bandwidth')
    for ax in axes:
        ax.set_xticks(BANDS);ax.set_xlabel('One-sided bandwidth / chip rate');ax.grid(alpha=.2)
    fig.suptitle('Fixed prefilter C/N0 and noise PSD; exact shared in-band IQ')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
