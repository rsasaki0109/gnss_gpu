"""Receiver-observable position publication gate; not a certified integrity bound.

Uses redundant pseudoranges, geometry, postfit residuals and recent published
positions. No reference/true position, weather flags or path visibility input.
"""

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np

from gnss_gpu.input_validation import finite_float, nonnegative_float, positive_float, positive_int
from gnss_gpu.positioning_cpu import PositionSolution, solve_pseudorange_position


@dataclass(frozen=True)
class PositionQuality:
    observations: int
    degrees_of_freedom: int
    pdop: float
    hdop: float
    vdop: float
    residual_rms_m: float
    postfit_sigma_m: Optional[float]


def position_quality(satellites, candidate: PositionSolution):
    """Equal-weight geometric DOP and postfit fit diagnostics at the estimate.

    DOP is dimensionless, not a metre uncertainty or protection level. HDOP/VDOP
    assume local x/y horizontal and z up; PDOP is rotation invariant. Four
    observations have zero residual degrees of freedom, not perfect integrity.
    Returns None for degenerate geometry. The candidate must come from these
    satellites and the associated pseudoranges, in the same order.
    """
    sat = np.asarray(satellites, dtype=float)
    position = np.asarray(candidate.position_m, dtype=float)
    residual = np.asarray(candidate.residuals_m, dtype=float)
    if (sat.ndim != 2 or sat.shape[1:] != (3,) or position.shape != (3,)
            or residual.shape != (len(sat),) or not np.all(np.isfinite(sat))
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(residual))):
        raise ValueError("finite (n,3) satellites, (3,) estimate and matching (n,) residuals required")
    if len(sat) < 4:
        return None
    delta = position-sat
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance == 0) or not np.all(np.isfinite(distance)):
        return None
    design = np.column_stack([delta/distance[:, None], np.ones(len(sat))])
    _, singular, right = np.linalg.svd(design, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps*max(design.shape)*singular[0]:
        return None
    covariance = (right.T/(singular**2))@right
    diagonal = np.maximum(np.diag(covariance), 0.)
    dof = len(sat)-4
    energy = float(residual@residual)
    return PositionQuality(len(sat), dof, float(np.sqrt(np.sum(diagonal[:3]))),
                           float(np.sqrt(np.sum(diagonal[:2]))), float(np.sqrt(diagonal[2])),
                           float(np.sqrt(energy/len(sat))), float(np.sqrt(energy/dof)) if dof else None)


@dataclass(frozen=True)
class PositionGateConfig:
    min_observations: int = 6
    max_pdop: float = 6.
    max_residual_rms_m: float = 15.
    jump_allowance_m: float = 20.
    speed_bound_mps: float = 15.
    max_anchor_age_s: float = 2.
    confirmation_blocks: int = 5

    def __post_init__(self):
        for name in ("min_observations", "confirmation_blocks"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))
        if self.min_observations < 5:
            raise ValueError("min_observations must provide redundancy beyond four unknowns")
        for name in ("max_pdop", "max_anchor_age_s"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        for name in ("max_residual_rms_m", "jump_allowance_m", "speed_bound_mps"):
            object.__setattr__(self, name, nonnegative_float(name, getattr(self, name)))


@dataclass(frozen=True)
class PositionGateResult:
    accepted: bool
    reason: str
    confirmation_count: int
    candidate: Optional[PositionSolution]
    quality: Optional[PositionQuality]
    position_m: Optional[np.ndarray]
    clock_bias_m: Optional[float]
    anchor_jump_m: Optional[float]
    anchor_age_s: Optional[float]
    recovery_count: int = 0
    recovery_radius_m: Optional[float] = None
    reanchored: bool = False
    anchor_source: Optional[str] = None


@dataclass(frozen=True)
class PositionRecoveryConfig:
    """Stricter, local candidate cluster for replacing a blocking old anchor.

    Serial samples are correlated. Radius and count are engineering stability
    checks, not independent evidence or an absolute-error confidence bound.
    """

    min_observations: int = 7
    max_pdop: float = 5.
    max_residual_rms_m: float = 5.
    window_blocks: int = 25
    min_span_s: float = .024
    max_gap_s: float = .005
    radius_m: float = 10.
    anchor_window_blocks: int = 15

    def __post_init__(self):
        for name in ("min_observations", "window_blocks", "anchor_window_blocks"):
            object.__setattr__(self, name, positive_int(name, getattr(self, name)))
        if self.min_observations < 5 or self.window_blocks < 2:
            raise ValueError("recovery requires redundant ranges and multiple candidate epochs")
        for name in ("max_pdop", "min_span_s", "max_gap_s", "radius_m"):
            object.__setattr__(self, name, positive_float(name, getattr(self, name)))
        object.__setattr__(self, "max_residual_rms_m", nonnegative_float("max_residual_rms_m", self.max_residual_rms_m))


class PositionPublicationGate:
    """Validate each candidate and confirm a fresh chain after any rejection.

    Jump allowance is a noise margin plus speed_bound*elapsed time, not a
    statistically calibrated motion test. By default the last published
    position anchors the test and rejected candidates never move it. Optional
    recovery uses a median of published positions and permits reanchoring to a
    sustained, stronger-quality candidate cluster before fresh confirmation.
    After max_anchor_age_s the anchor expires, allowing a fresh bootstrap.
    Persistent coherent bias, slow drift
    or an initially wrong consistent position can still pass. Confirmation
    counts serial observations, not independent evidence or a false-fix bound.
    Missing/rejected output is None, never a coasted or previously accepted fix.
    """

    def __init__(self, config=None, *, initial_position_m=(-25., 5., 0.), recovery_config=None):
        self.config = config if config is not None else PositionGateConfig()
        if not isinstance(self.config, PositionGateConfig):
            raise TypeError("config must be PositionGateConfig")
        if recovery_config is not None and not isinstance(recovery_config, PositionRecoveryConfig):
            raise TypeError("recovery_config must be PositionRecoveryConfig or None")
        self.recovery_config = recovery_config
        if recovery_config is not None and (
                recovery_config.min_observations < self.config.min_observations
                or recovery_config.max_pdop > self.config.max_pdop
                or recovery_config.max_residual_rms_m > self.config.max_residual_rms_m):
            raise ValueError("recovery quality checks must be at least as strict as ordinary checks")
        self.initial = np.array(initial_position_m, dtype=float, copy=True)
        if self.initial.shape != (3,) or not np.all(np.isfinite(self.initial)):
            raise ValueError("initial_position_m must be a finite (3,) vector")
        self.reset()

    def reset(self):
        self._last_time = None
        self._anchor = None
        self._anchor_time = None
        self._pending = None
        self._pending_time = None
        self._count = 0
        self._ready = False
        self._recovery = deque(maxlen=self.recovery_config.window_blocks if self.recovery_config else 1)
        self._published = deque(maxlen=self.recovery_config.anchor_window_blocks if self.recovery_config else 1)
        self._anchor_source = None

    def _try_recovery(self, candidate, quality, time):
        cfg = self.recovery_config
        if (cfg is None or candidate is None or quality is None
                or quality.observations < cfg.min_observations or quality.pdop > cfg.max_pdop
                or quality.residual_rms_m > cfg.max_residual_rms_m):
            self._recovery.clear()
            return None, None
        if self._recovery:
            last_time, last = self._recovery[-1]
            gap = time-last_time
            if gap > cfg.max_gap_s or np.linalg.norm(candidate.position_m-last) > self.config.jump_allowance_m+self.config.speed_bound_mps*gap:
                self._recovery.clear()
        self._recovery.append((time, candidate.position_m.copy()))
        points = np.asarray([position for _, position in self._recovery])
        center = np.median(points, axis=0)
        radius = float(np.quantile(np.linalg.norm(points-center, axis=1), .9))
        span = time-self._recovery[0][0]
        allowance = cfg.radius_m+self.config.speed_bound_mps*span
        if (len(self._recovery) < cfg.window_blocks or span+1e-12 < cfg.min_span_s
                or radius > allowance or np.linalg.norm(candidate.position_m-center) > allowance):
            return None, radius
        return center, radius

    def update(self, satellites, pseudoranges, *, time_s):
        time = finite_float("time_s", time_s)
        if self._last_time is not None and time <= self._last_time:
            raise ValueError("time_s must strictly increase; reset before a new sequence")
        sat, ranges = np.asarray(satellites, dtype=float), np.asarray(pseudoranges, dtype=float)
        if sat.size == 0 and ranges.size == 0:
            sat = np.empty((0, 3))
        if (sat.ndim != 2 or sat.shape[1:] != (3,) or ranges.shape != (len(sat),)
                or not np.all(np.isfinite(sat)) or not np.all(np.isfinite(ranges)) or np.any(ranges <= 0)):
            raise ValueError("finite (n,3) satellites and positive matching (n,) ranges required")
        candidate = solve_pseudorange_position(sat, ranges, initial_position_m=self.initial) if len(sat) >= 4 else None
        quality = position_quality(sat, candidate) if candidate is not None else None
        # Only now commit time/state, so malformed input can be retried.
        self._last_time = time
        if self._anchor_time is not None and time-self._anchor_time > self.config.max_anchor_age_s:
            self._anchor, self._anchor_time = None, None
            self._ready = False
            self._pending, self._pending_time, self._count = None, None, 0
            self._recovery.clear()
            self._published.clear()
            self._anchor_source = None
        age = time-self._anchor_time if self._anchor_time is not None else None
        jump = (float(np.linalg.norm(candidate.position_m-self._anchor))
                if candidate is not None and self._anchor is not None else None)
        reason = None
        recovery_count, radius, reanchored = 0, None, False
        if len(sat) < self.config.min_observations:
            reason = "too_few_satellites"
        elif candidate is None:
            reason = "no_solution"
        elif quality is None:
            reason = "rank_failure"
        elif quality.pdop > self.config.max_pdop:
            reason = "pdop"
        elif quality.residual_rms_m > self.config.max_residual_rms_m:
            reason = "residual_rms"
        elif jump is not None and age is not None and jump > self.config.jump_allowance_m+self.config.speed_bound_mps*age:
            reason = "anchor_jump"
        if reason == "anchor_jump" and self.recovery_config is not None:
            center, radius = self._try_recovery(candidate, quality, time)
            recovery_count = len(self._recovery)
            if center is not None:
                # Nothing is published yet: a stronger candidate cluster only
                # replaces the reference. Ordinary confirmation must restart.
                self._anchor, self._anchor_time = center.copy(), time
                self._anchor_source = "recovery_cluster"
                self._published.clear()
                self._published.extend(position.copy() for _, position in self._recovery)
                self._pending, self._pending_time, self._count, self._ready = None, None, 0, False
                self._recovery.clear()
                reason, reanchored = None, True
        else:
            self._recovery.clear()
        if reason is not None:
            self._pending, self._pending_time, self._count, self._ready = None, None, 0, False
            return PositionGateResult(False, reason, 0, candidate, quality, None, None, jump, age,
                                      recovery_count, radius, False, self._anchor_source)
        if candidate is None:
            raise RuntimeError("passing position checks produced no candidate")
        if not self._ready:
            consistent = False
            if self._pending is not None and self._pending_time is not None:
                allowance = self.config.jump_allowance_m+self.config.speed_bound_mps*(time-self._pending_time)
                consistent = np.linalg.norm(candidate.position_m-self._pending) <= allowance
            self._count = self._count+1 if consistent else 1
            self._pending, self._pending_time = candidate.position_m.copy(), time
            if self._count < self.config.confirmation_blocks:
                return PositionGateResult(False, "confirming", self._count, candidate, quality, None, None, jump, age,
                                          recovery_count, radius, reanchored, self._anchor_source)
            self._ready = True
        if self.recovery_config is None:
            self._anchor = candidate.position_m.copy()
            self._anchor_source = "published"
        else:
            self._published.append(candidate.position_m.copy())
            self._anchor = np.median(np.asarray(self._published), axis=0)
            self._anchor_source = "published_median"
        self._anchor_time = time
        return PositionGateResult(True, "accepted", self._count, candidate, quality,
                                  candidate.position_m.copy(), candidate.clock_bias_m, jump, age,
                                  recovery_count, radius, reanchored, self._anchor_source)
