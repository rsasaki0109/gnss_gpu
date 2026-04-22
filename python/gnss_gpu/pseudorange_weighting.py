"""Pseudorange measurement weighting helpers for PF smoother runs."""

from __future__ import annotations

import numpy as np


def apply_pseudorange_weighting(
    measurements: list,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    spp_position: np.ndarray,
    pr_history: dict[int, list[float]],
    *,
    residual_downweight: bool,
    residual_threshold: float,
    pr_accel_downweight: bool,
    pr_accel_threshold: float,
) -> np.ndarray:
    updated = np.asarray(weights, dtype=np.float64).copy()
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pseudoranges, dtype=np.float64).ravel()

    if residual_downweight:
        spp_pos = np.asarray(spp_position, dtype=np.float64).ravel()[:3]
        if np.isfinite(spp_pos).all() and np.linalg.norm(spp_pos) > 1e6:
            n = min(len(updated), len(pr), sat.shape[0])
            ranges = np.linalg.norm(sat[:n] - spp_pos, axis=1)
            cb_est = float(np.median(pr[:n] - ranges))
            for i_m in range(n):
                residual = abs(pr[i_m] - ranges[i_m] - cb_est)
                updated[i_m] *= 1.0 / (1.0 + (residual / residual_threshold) ** 2)

    if pr_accel_downweight:
        n = min(len(updated), len(pr), len(measurements))
        for i_m in range(n):
            prn = int(getattr(measurements[i_m], "prn", 0))
            cur_pr = float(pr[i_m])
            hist = pr_history.get(prn, [])
            if len(hist) >= 2:
                accel = abs(cur_pr - 2.0 * hist[-1] + hist[-2])
                updated[i_m] *= 1.0 / (1.0 + (accel / pr_accel_threshold) ** 2)
            hist.append(cur_pr)
            if len(hist) > 2:
                hist.pop(0)
            pr_history[prn] = hist

    return updated
