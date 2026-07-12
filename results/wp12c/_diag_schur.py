"""Quick diagnostic: Schur front-block eigenvalues vs properly marginalized."""
from __future__ import annotations

import numpy as np

from gnss_gpu.local_fgo import LocalFgoConfig
from gnss_gpu.tc_fgo import (
    TcFgoConfig,
    TcFgoEpochObs,
    TcFgoNavState,
    TcFgoWindowProblem,
    build_window_hessian_gradient,
    compute_schur_marginal_from_window,
    enu_to_ecef,
    schur_complement_marginalize,
    solve_tc_fgo_window,
    state_dim,
)
from tests.test_tc_fgo import _dd_epoch_for_position, _make_segment, _synthetic_satellites

origin = np.array([4_000_000.0, 3_000_000.0, 2_000_000.0])
origin_lat, origin_lon = 0.35, 0.65
sats = _synthetic_satellites()
base_pos = origin + np.array([100.0, -50.0, 20.0])
n = 5
states, observations = [], []
for i in range(n):
    p = np.array([5.0 * 0.2 * i, 0.0, 0.0])
    states.append(
        TcFgoNavState(
            p_enu=p,
            v_enu=np.array([5.0, 0.0, 0.0]),
            q_body_to_enu=np.array([0.0, 0.0, 0.0, 1.0]),
            b_a=np.zeros(3),
            b_g=np.zeros(3),
        )
    )
    pos_ecef = enu_to_ecef(p, origin, origin_lat, origin_lon)
    observations.append(TcFgoEpochObs(dd_pseudorange=_dd_epoch_for_position(pos_ecef, sats, base_pos)))

problem = TcFgoWindowProblem(
    initial_states=states,
    imu_segments=[_make_segment() for _ in range(n - 1)],
    observations=observations,
    origin_ecef=origin,
    origin_lat=origin_lat,
    origin_lon=origin_lon,
)
cfg = TcFgoConfig(enable_schur_marginalization=True, pr_huber_k=0.0, max_iterations=30, window_epochs=n)
fgo = LocalFgoConfig()
result = solve_tc_fgo_window(problem, config=cfg, fgo_config=fgo)
schur = result.schur_marginal
assert schur is not None
sdim = state_dim(cfg)
front_block = schur.precision[:sdim, :sdim]
print("marginal size", schur.mean.size, "n_nav_epochs", schur.n_nav_epochs)
print("front block max eig", float(np.max(np.linalg.eigvalsh(front_block))))
print("full marginal max eig", float(np.max(np.linalg.eigvalsh(schur.precision))))

# proper front-only Schur from full marginal
m2 = schur_complement_marginalize(schur.precision, np.zeros(schur.mean.size), sdim)
if m2 is not None:
    print("proper front-only max eig", float(np.max(np.linalg.eigvalsh(m2))))
    print("ratio front_block / proper", float(np.max(np.linalg.eigvalsh(front_block))) / max(float(np.max(np.linalg.eigvalsh(m2))), 1e-12))
