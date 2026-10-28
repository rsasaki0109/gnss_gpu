"""Model-independent distant-delay ambiguity diagnostic, not a probability.

Uses the same two-path model on either side of baseline model selection.
Requires independent null calibration before it may be used as a warning.
"""
import numpy as np
from gnss_gpu.drift_delay_profile_cpu import delay_profile
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED,CHIP_RATE

OFFSETS=(-.15,-.1,-.05,-.035,.035,.05,.1,.15)


def ambiguity_score(single_loss,two_loss,distant_losses,complex_count):
    if not distant_losses:
        return 0.
    values=np.asarray([single_loss,two_loss,*distant_losses],dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values<=0) or complex_count<=0:
        raise ValueError('finite positive losses and sample count required')
    reference=float(values.min())
    penalty=max(0.,float(2*complex_count*np.log(min(distant_losses)/reference)))
    return 1/(1+penalty)


def candidate_grid(selected_delay,single,two):
    grid=np.unique(np.clip(np.r_[selected_delay+np.array(OFFSETS),single.first_delay_chips,two.first_delay_chips],-1,1))
    return grid[np.abs(grid-selected_delay)*LIGHT_SPEED/CHIP_RATE>10]


def diagnose_ambiguity(blocks,weights,selected_delay,single,two):
    grid=candidate_grid(selected_delay,single,two)
    points=delay_profile(blocks,weights,grid,single,two)['profiles']['two'] if len(grid) else []
    losses=[p['loss'] for p in points if p['converged']]
    return dict(score=ambiguity_score(single.loss,two.loss,losses,np.asarray(blocks).size),
        evaluated=True,unresolved=not bool(losses),points=points)
