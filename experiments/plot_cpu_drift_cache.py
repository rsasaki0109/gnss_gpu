import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.benchmark_cpu_drift_cache import OUTPUT


def run():
    rows=[json.loads(x) for x in (OUTPUT/'timings.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(1,9)
    axes[0].bar(x-.18,[r['baseline_seconds'] for r in rows],.36,label='Baseline')
    axes[0].bar(x+.18,[r['cached_seconds'] for r in rows],.36,label='Cached')
    axes[0].set(xlabel='Paired run (4 cases x2 repeats)',ylabel='Full pipeline wall time (seconds)');axes[0].legend()
    axes[1].plot(x,[r['speedup'] for r in rows],'o-');axes[1].axhline(1,color='black',ls='--')
    axes[1].set(xlabel='Paired run',ylabel='Baseline / cached wall time');axes[1].grid(alpha=.25)
    fig.suptitle('Same-host sequential timing;395 stored observations reproduced exactly')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(OUTPUT/f'comparison.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':run()
