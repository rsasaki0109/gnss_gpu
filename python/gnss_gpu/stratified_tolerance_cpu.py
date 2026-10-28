"""A fixed maximum threshold covering a finite declared family of null strata."""

from dataclasses import dataclass

from gnss_gpu.input_validation import positive_float, positive_int
from gnss_gpu.noncoherent_acquisition_cpu import CaNoncoherentSearch, CaNoncoherentDetection, acquire_ca_noncoherent
from gnss_gpu.noncoherent_tolerance_cpu import calibrate_ca_noncoherent_tolerance


@dataclass(frozen=True)
class CaStratifiedToleranceThreshold:
    config: CaNoncoherentSearch
    score: float
    strata: tuple
    methods_in_family: int
    joint_failure_budget: float

    def __post_init__(self):
        score = positive_float("score", self.score)
        methods = positive_int("methods_in_family", self.methods_in_family)
        budget = positive_float("joint_failure_budget", self.joint_failure_budget)
        if budget >= 1:
            raise ValueError("joint_failure_budget must be below one")
        strata = tuple((name, value) for name, value in self.strata)
        if not strata or len({name for name, _ in strata}) != len(strata):
            raise ValueError("strata must be nonempty and uniquely named")
        if any(value.config != self.config or value.score > score for _, value in strata):
            raise ValueError("threshold must match and dominate every stratum")
        if sum(value.calibration_failure_probability for _, value in strata) > budget/methods+1e-15:
            raise ValueError("stratum confidence allocations exceed family budget")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "methods_in_family", methods)
        object.__setattr__(self, "joint_failure_budget", budget)
        object.__setattr__(self, "strata", strata)


def calibrate_ca_stratified_tolerance(scores_by_stratum, config, *,
                                      max_false_alarm_probability=.01, joint_failure_budget=.05,
                                      methods_in_family=2):
    """Calibrate each declared null stratum, then use one maximum threshold.

    Allocate delta = joint_budget/(method_count * stratum_count) per calibration.
    Each stratum's records must be IID representative draws from that stratum.
    Shared draws across strata/methods are allowed by the union bound.
    No stratum label or oracle selection is used during detection. The bound
    covers only declared stratum distributions, not arbitrary unseen interference.
    """
    methods = positive_int("methods_in_family", methods_in_family)
    budget = positive_float("joint_failure_budget", joint_failure_budget)
    if budget >= 1 or not scores_by_stratum:
        raise ValueError("require nonempty strata and failure budget below one")
    if any(not isinstance(name, str) or not name for name in scores_by_stratum):
        raise ValueError("strata must have nonempty string names")
    delta = budget/(methods*len(scores_by_stratum))
    strata = tuple((name, calibrate_ca_noncoherent_tolerance(scores_by_stratum[name], config,
                    max_false_alarm_probability=max_false_alarm_probability, failure_probability=delta))
                   for name in sorted(scores_by_stratum))
    return CaStratifiedToleranceThreshold(config, max(value.score for _, value in strata), strata, methods, budget)


def detect_ca_stratified_tolerance(iq, threshold):
    candidate = acquire_ca_noncoherent(iq, threshold.config)
    score = candidate.statistic if candidate is not None else 0.
    return CaNoncoherentDetection(candidate is not None and score > threshold.score, score, threshold.score, candidate)
