"""Data-independent rank selection for a one-sided null tolerance gate."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom
from gnss_gpu.input_validation import positive_int


@dataclass(frozen=True)
class RankedPathGate:
    score: float
    calibration_count: int
    tail_budget: float
    confidence_failure: float
    rank_from_largest: int
    actual_failure_bound: float


def permissible_rank(count, tail_budget=.01, confidence_failure=.05/3):
    """Largest k satisfying BinomialCDF(k-1; n, alpha) <= delta.

    Depends only on sample size and requested budgets, never score values or
    echo outcomes. For iid null scores and strict exceedance, the kth-largest
    score has P(null tail probability > alpha) bounded by this CDF. Ties are
    conservative. Multiple nuisance strata require a separate union budget.
    """
    n=positive_int('count',count)
    alpha,delta=float(tail_budget),float(confidence_failure)
    if not 0<alpha<1 or not 0<delta<1:
        raise ValueError('budgets must be finite and in (0,1)')
    if binom.cdf(0,n,alpha)>delta:
        raise ValueError('insufficient calibration records even for maximum gate')
    low,high=1,n
    while low<high:
        mid=(low+high+1)//2
        if binom.cdf(mid-1,n,alpha)<=delta:low=mid
        else:high=mid-1
    return low


def calibrate_ranked_gate(scores, *, tail_budget=.01, confidence_failure=.05/3):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not np.all(np.isfinite(values)):
        raise ValueError('finite one-dimensional scores required')
    rank=permissible_rank(len(values),tail_budget,confidence_failure)
    score=float(np.partition(values,len(values)-rank)[len(values)-rank])
    return RankedPathGate(score,len(values),float(tail_budget),float(confidence_failure),rank,
                          float(binom.cdf(rank-1,len(values),tail_budget)))
