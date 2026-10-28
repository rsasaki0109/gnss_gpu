"""Export the predeclared weak-capture endpoint and independent null outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((directory/'aggregate.json').read_text())
    report=json.loads((directory/'report.json').read_text())
    multiseed=any(row['method']=='multiseed_20' for row in rows)
    methods=(('single_seed_20',-.18,'Single coarse seed + local-delay coherent'),
             ('multiseed_20',.18,'Three coarse seeds + local-delay coherent')) if multiseed else (
             ('noncoherent_4',-.18,'4 ms noncoherent baseline'),
             ('bit_edge_20',.18,'20 ms bit-edge + local-delay model'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for method,offset,label in methods:
        selected=[next(row for row in rows if row['method']==method and row['profile']=='near_far_20db'
                       and row['nominal_primary_cn0_db_hz']==37. and row['primary_doppler_hz']==doppler) for doppler in (170.,700.)]
        y=np.array([row['criterion_fraction'] for row in selected]);bounds=np.array([row['criterion_wilson_95'] for row in selected])
        axes[0].bar(np.arange(2)+offset,y,width=.36,label=label,
                    yerr=np.maximum(0.,[y-bounds[:,0],bounds[:,1]-y]),capsize=3)
        for x,row in zip(np.arange(2)+offset,selected):
            axes[0].text(x,row['criterion_fraction']+.025,f"{row['criterion']}/{row['replicas']}",ha='center')
        profiles=('white','other_prns','near_far_20db')
        selected=[report['null_results'][profile][method] for profile in profiles]
        y=np.array([row['fraction'] for row in selected])*100;bounds=np.array([row['pointwise_wilson_95'] for row in selected])*100
        axes[1].bar(np.arange(3)+offset,y,width=.36,label=label,
                    yerr=np.maximum(0.,[y-bounds[:,0],bounds[:,1]-y]),capsize=3)
    axes[0].set(title='37 dB-Hz target; other satellites +20 dB',xticks=(0,1),xticklabels=('170 Hz','700 Hz'),
                ylabel='Detection + coarse carrier/phase criterion',ylim=(0.,1.12))
    axes[0].axhline(.75,color='gray',linestyle='--',label='Predeclared 75% event-fraction criterion')
    axes[0].legend(fontsize=7)
    null_count=selected[0]['records']
    null_upper=max(value['pointwise_wilson_95'][1]*100 for profile in report['null_results'].values() for value in profile.values())
    axes[1].set(title=f'Fresh target-absent nulls: {null_count} records per stratum',xticks=range(3),
                xticklabels=('White','Other PRNs','Other PRNs +20 dB'),ylabel='Passage rate (%)',ylim=(0.,max(1.4,null_upper*1.1)))
    axes[1].axhline(1.,color='gray',linestyle='--')
    for axis in axes:axis.grid(axis='y',alpha=.2)
    key='declared_improvement_criteria_met' if multiseed else 'declared_capture_criteria_met'
    endpoint='Multiseed improvement' if multiseed else 'Weak capture'
    status='met' if report[key] else 'not met'
    fig.suptitle(f'{endpoint} criterion {status} on fresh simulation data\nPointwise Wilson intervals; finite IID null family, not RF or tracking validation')
    fig.savefig(directory/'weak_acquisition.png',dpi=160);fig.savefig(directory/'weak_acquisition.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('output/cpu-weak-acquisition-refined'))
    plot(parser.parse_args().directory)
