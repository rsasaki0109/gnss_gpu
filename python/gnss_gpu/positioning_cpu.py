"""Small NumPy-only pseudorange/receiver-clock least-squares reference."""

from dataclasses import dataclass

import numpy as np

from gnss_gpu.input_validation import positive_int, positive_float


@dataclass(frozen=True)
class PositionSolution:
    position_m: np.ndarray
    clock_bias_m: float
    residuals_m: np.ndarray
    iterations: int


def solve_pseudorange_position(transmitter_positions_m, pseudoranges_m, *,
                               initial_position_m=(0., 0., 0.), max_iterations=20,
                               tolerance_m=1e-4):
    """Solve range_i=||sat_i-rx||+clock_m in a common Cartesian frame.

    At least four finite nondegenerate observations are required. Equal weights,
    no outlier rejection or truth-based selection. Returns None for rank failure
    or nonconvergence. Local coordinates are supported; this does not introduce
    orbit, Earth-rotation, atmosphere or code-period ambiguity resolution.
    """
    satellites = np.asarray(transmitter_positions_m, dtype=float)
    ranges = np.asarray(pseudoranges_m, dtype=float)
    position = np.array(initial_position_m, dtype=float, copy=True)
    iterations = positive_int("max_iterations", max_iterations)
    tolerance = positive_float("tolerance_m", tolerance_m)
    if (satellites.ndim != 2 or satellites.shape[1:] != (3,) or ranges.ndim != 1
            or satellites.shape[0] != ranges.size or position.shape != (3,)
            or not np.all(np.isfinite(satellites)) or not np.all(np.isfinite(ranges))
            or not np.all(np.isfinite(position)) or np.any(ranges <= 0)):
        raise ValueError("finite (n,3) transmitters, positive (n,) ranges and (3,) initial position required")
    if ranges.size < 4:
        return None
    clock = 0.
    for index in range(iterations):
        delta = position-satellites
        geometric = np.linalg.norm(delta, axis=1)
        if np.any(geometric == 0) or not np.all(np.isfinite(geometric)):
            return None
        design = np.column_stack([delta/geometric[:, None], np.ones(len(ranges))])
        residual = ranges-geometric-clock
        update, _, rank, _ = np.linalg.lstsq(design, residual, rcond=None)
        if rank < 4 or not np.all(np.isfinite(update)):
            return None
        position += update[:3]
        clock += float(update[3])
        if np.linalg.norm(update) < tolerance:
            return PositionSolution(position.copy(), clock,
                                     ranges-np.linalg.norm(satellites-position, axis=1)-clock, index+1)
    return None
