"""Weighted pseudorange fitting and local linearized, model-dependent covariance."""

from dataclasses import dataclass

import numpy as np

from gnss_gpu.input_validation import positive_float, positive_int
from gnss_gpu.positioning_cpu import PositionSolution


@dataclass(frozen=True)
class WeightedPositionSolution:
    solution: PositionSolution
    covariance_m2: np.ndarray  # x, y, z, receiver clock in metres
    normalized_residual_energy: float
    degrees_of_freedom: int


def solve_weighted_pseudorange_position(satellites_m, pseudoranges_m, sigma_m, *,
                                       initial_position_m=(0., 0., 0.), max_iterations=20,
                                       tolerance_m=1e-4):
    """Fit independent ranges with supplied absolute standard deviations.

    Covariance is (H.T R^-1 H)^-1 at the final estimate, including clock coupling.
    It is NOT multiplied by postfit residual variance. Validity requires correct
    range variances and a local unbiased independent-error model. Four ranges
    have no redundant residual check. None denotes rank failure/nonconvergence.
    No reference position or weather/visibility fields are accepted.
    """
    sat = np.asarray(satellites_m, dtype=float)
    ranges, sigma = np.asarray(pseudoranges_m, dtype=float), np.asarray(sigma_m, dtype=float)
    position = np.array(initial_position_m, dtype=float, copy=True)
    iterations, tolerance = positive_int("max_iterations", max_iterations), positive_float("tolerance_m", tolerance_m)
    if (sat.ndim != 2 or sat.shape[1:] != (3,) or ranges.shape != (len(sat),)
            or sigma.shape != ranges.shape or position.shape != (3,)
            or not all(np.all(np.isfinite(x)) for x in (sat, ranges, sigma, position))
            or np.any(ranges <= 0) or np.any(sigma <= 0)):
        raise ValueError("finite (n,3) satellites, positive matching ranges/sigmas and finite (3,) initial position required")
    if len(sat) < 4:
        return None
    clock = 0.
    for index in range(iterations):
        delta = position-sat
        geometric = np.linalg.norm(delta, axis=1)
        if np.any(geometric == 0) or not np.all(np.isfinite(geometric)):
            return None
        design = np.column_stack([delta/geometric[:, None], np.ones(len(sat))])
        residual = ranges-geometric-clock
        update, _, rank, _ = np.linalg.lstsq(design/sigma[:, None], residual/sigma, rcond=None)
        if rank < 4 or not np.all(np.isfinite(update)):
            return None
        position += update[:3]
        clock += float(update[3])
        if np.linalg.norm(update) < tolerance:
            geometric = np.linalg.norm(position-sat, axis=1)
            if np.any(geometric == 0) or not np.all(np.isfinite(geometric)):
                return None
            design = np.column_stack([(position-sat)/geometric[:, None], np.ones(len(sat))])
            _, singular, right = np.linalg.svd(design/sigma[:, None], full_matrices=False)
            if singular[-1] <= np.finfo(float).eps*max(design.shape)*singular[0]:
                return None
            covariance = (right.T/(singular**2))@right
            residual = ranges-geometric-clock
            solution = PositionSolution(position.copy(), clock, residual, index+1)
            return WeightedPositionSolution(solution, covariance, float(np.sum((residual/sigma)**2)), len(sat)-4)
    return None
