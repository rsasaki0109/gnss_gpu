from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

from experiments.exp_gsdc2023_robust_wls import RobustWLSConfig, robust_wls_epoch


def test_robust_wls_downweights_single_pseudorange_outlier() -> None:
    rx_true = np.array([1.2e6, -4.7e6, 4.1e6], dtype=np.float64)
    directions = np.array([
        [0.82, 0.25, 0.52],
        [-0.32, 0.74, 0.59],
        [0.18, -0.91, 0.37],
        [-0.76, -0.20, 0.62],
        [0.53, -0.38, 0.76],
        [-0.12, 0.45, 0.88],
    ], dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sat_ecef = rx_true + directions * 20_200_000.0
    clock_bias = 42.0
    pseudoranges = np.linalg.norm(sat_ecef - rx_true, axis=1) + clock_bias
    pseudoranges[0] += 250.0
    init = rx_true + np.array([20.0, -10.0, 5.0])
    weights = np.ones(len(pseudoranges), dtype=np.float64)

    robust_pos, robust_stats = robust_wls_epoch(
        sat_ecef,
        pseudoranges,
        weights,
        init,
        RobustWLSConfig(huber_k_m=10.0, max_shift_m=500.0, prior_sigma_m=None),
    )
    plain_pos, plain_stats = robust_wls_epoch(
        sat_ecef,
        pseudoranges,
        weights,
        init,
        RobustWLSConfig(huber_k_m=1.0e9, max_shift_m=500.0, prior_sigma_m=None),
    )

    assert robust_stats["accepted"] is True
    assert plain_stats["accepted"] is True
    assert robust_stats["min_huber_weight"] < 0.1
    assert np.linalg.norm(robust_pos - rx_true) < np.linalg.norm(plain_pos - rx_true)
