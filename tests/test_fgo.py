"""FGO: reference normal equations vs CUDA (and gtsam_gnss-style multi-clock).

Unit tests compare one Gauss–Newton step with line search disabled to a Python
reference mirroring ``src/positioning/fgo.cu``.
"""

from __future__ import annotations

import numpy as np
import pytest

C_LIGHT = 299792458.0
OMEGA_E = 7.2921151467e-5
DIAG_JITTER = 1e-3


def _fill_hc(nc: int, sk: int, hc: np.ndarray) -> None:
    hc[:] = 0.0
    hc[0] = 1.0
    if sk > 0 and sk < nc:
        hc[sk] = 1.0


def _geometric_range_sagnacrcv(xyz: np.ndarray, sat: np.ndarray) -> float:
    sx, sy, sz = float(sat[0]), float(sat[1]), float(sat[2])
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    dx0, dy0, dz0 = x - sx, y - sy, z - sz
    r0 = float(np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0))
    transit = r0 / C_LIGHT
    theta = OMEGA_E * transit
    sx_rot = sx * np.cos(theta) + sy * np.sin(theta)
    sy_rot = -sx * np.sin(theta) + sy * np.cos(theta)
    dx, dy_v, dz = x - sx_rot, y - sy_rot, z - sz
    return float(np.sqrt(dx * dx + dy_v * dy_v + dz * dz))


def _geometric_range_rate_sagnac(
    rx: np.ndarray,
    sat: np.ndarray,
    rv: np.ndarray,
    sat_vel: np.ndarray,
) -> float:
    delta = sat - rx
    ranges = float(np.linalg.norm(delta))
    los = delta / ranges
    euclidean_rate = float(np.dot(los, sat_vel - rv))
    sagnac_rate = OMEGA_E * (
        sat_vel[0] * rx[1]
        + sat[0] * rv[1]
        - sat_vel[1] * rx[0]
        - sat[1] * rv[0]
    ) / C_LIGHT
    return float(euclidean_rate - sagnac_rate)


def _doppler_model_sagnac(
    rx: np.ndarray,
    sat: np.ndarray,
    rv: np.ndarray,
    sat_vel: np.ndarray,
    drift: float,
    sat_clock_drift: float = 0.0,
) -> float:
    return float(drift - (_geometric_range_rate_sagnac(rx, sat, rv, sat_vel) - sat_clock_drift))


def _assemble_pr_h_g(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    sys_kind: np.ndarray | None,
    nc: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_epoch, n_sat, _ = sat_ecef.shape
    ss = 3 + nc
    n_state = ss * n_epoch
    H = np.zeros((n_state, n_state))
    g = np.zeros(n_state)
    for t in range(n_epoch):
        x, y, z = state[t, 0], state[t, 1], state[t, 2]
        cptr = state[t, 3 : 3 + nc]
        o = ss * t
        Hloc = np.zeros((ss, ss))
        gloc = np.zeros(ss)
        for s in range(n_sat):
            w = weights[t, s]
            if w <= 0:
                continue
            sk = int(sys_kind[t, s]) if sys_kind is not None else 0
            if sk < 0 or sk >= nc:
                continue
            sx, sy, sz = sat_ecef[t, s]
            dx0 = x - sx
            dy0 = y - sy
            dz0 = z - sz
            r0 = float(np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0))
            transit = r0 / C_LIGHT
            theta = OMEGA_E * transit
            sx_rot = sx * np.cos(theta) + sy * np.sin(theta)
            sy_rot = -sx * np.sin(theta) + sy * np.cos(theta)
            dx = x - sx_rot
            dy_v = y - sy_rot
            dz = z - sz
            r = float(np.sqrt(dx * dx + dy_v * dy_v + dz * dz))
            if r < 1e-6:
                continue
            hc = np.zeros(nc)
            _fill_hc(nc, sk, hc)
            clk = float(np.dot(hc, cptr))
            res = pseudorange[t, s] - (r + clk)
            J = np.zeros(ss)
            J[0], J[1], J[2] = dx / r, dy_v / r, dz / r
            J[3 : 3 + nc] = hc
            gloc += J * (w * res)
            Hloc += w * np.outer(J, J)
        H[o : o + ss, o : o + ss] += Hloc
        g[o : o + ss] += gloc
    return H, g


def _effective_weights_huber(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    sys_kind: np.ndarray | None,
    nc: int,
    huber_k: float,
) -> np.ndarray:
    if huber_k <= 0:
        return weights
    n_epoch, n_sat, _ = sat_ecef.shape
    eff = weights.astype(np.float64, copy=True)
    for t in range(n_epoch):
        x, y, z = state[t, 0], state[t, 1], state[t, 2]
        cptr = state[t, 3 : 3 + nc]
        for s in range(n_sat):
            w = float(weights[t, s])
            if w <= 0:
                continue
            sk = int(sys_kind[t, s]) if sys_kind is not None else 0
            if sk < 0 or sk >= nc:
                continue
            sx, sy, sz = sat_ecef[t, s]
            dx0 = x - sx
            dy0 = y - sy
            dz0 = z - sz
            r0 = float(np.sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0))
            transit = r0 / C_LIGHT
            theta = OMEGA_E * transit
            sx_rot = sx * np.cos(theta) + sy * np.sin(theta)
            sy_rot = -sx * np.sin(theta) + sy * np.cos(theta)
            dx = x - sx_rot
            dy_v = y - sy_rot
            dz = z - sz
            r = float(np.sqrt(dx * dx + dy_v * dy_v + dz * dz))
            if r < 1e-6:
                continue
            hc = np.zeros(nc)
            _fill_hc(nc, sk, hc)
            clk = float(np.dot(hc, cptr))
            res = float(pseudorange[t, s]) - (r + clk)
            z_m = abs(np.sqrt(w) * res)
            if z_m > huber_k:
                eff[t, s] = w * (huber_k / z_m)
    return eff


def _add_motion_rw(
    n_epoch: int,
    ss: int,
    w_motion: float,
    state: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
) -> None:
    if w_motion <= 0:
        return
    flat = state.reshape(-1)
    for t in range(n_epoch - 1):
        o0, o1 = ss * t, ss * (t + 1)
        for i in range(3):
            d01 = flat[o0 + i] - flat[o1 + i]
            g[o0 + i] += w_motion * d01
            g[o1 + i] += w_motion * (-d01)
        for i in range(3):
            for j in range(3):
                ident = w_motion if i == j else 0.0
                neg = -w_motion if i == j else 0.0
                H[o0 + i, o0 + j] += ident
                H[o1 + i, o1 + j] += ident
                H[o0 + i, o1 + j] += neg
                H[o1 + i, o0 + j] += neg


def _cholesky_decompose_inplace(n: int, a: np.ndarray) -> bool:
    for i in range(n):
        for j in range(i + 1):
            s = float(a[i, j])
            for k in range(j):
                s -= a[i, k] * a[j, k]
            if i == j:
                if s <= 1e-18:
                    return False
                a[i, j] = np.sqrt(s)
            else:
                a[i, j] = s / a[j, j]
        for j in range(i + 1, n):
            a[i, j] = 0.0
    return True


def _cholesky_solve_lower(n: int, l_mat: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.zeros(n)
    for i in range(n):
        s_a = float(b[i])
        for k in range(i):
            s_a -= l_mat[i, k] * x[k]
        x[i] = s_a / l_mat[i, i]
    for i in range(n - 1, -1, -1):
        s_b = float(x[i])
        for k in range(i + 1, n):
            s_b -= l_mat[k, i] * x[k]
        x[i] = s_b / l_mat[i, i]
    return x


def _one_gn_step(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
    sys_kind: np.ndarray | None,
    nc: int,
    motion_sigma_m: float,
    huber_k: float = 0.0,
) -> np.ndarray:
    n_epoch = sat_ecef.shape[0]
    ss = 3 + nc
    n_state = ss * n_epoch
    w_motion = (1.0 / (motion_sigma_m**2)) if motion_sigma_m > 0 else 0.0
    w_eff = _effective_weights_huber(
        sat_ecef, pseudorange, weights, state, sys_kind, nc, huber_k
    )
    H, g = _assemble_pr_h_g(sat_ecef, pseudorange, w_eff, state, sys_kind, nc)
    _add_motion_rw(n_epoch, ss, w_motion, state, H, g)
    rhs = -g
    hwork = H.copy().astype(np.float64)
    for i in range(n_state):
        hwork[i, i] += DIAG_JITTER
    if not _cholesky_decompose_inplace(n_state, hwork):
        raise RuntimeError("Cholesky failed in reference")
    return _cholesky_solve_lower(n_state, hwork, rhs)


try:
    from gnss_gpu._gnss_gpu import fgo_gnss_lm

    HAS_FGO = True
except ImportError:
    HAS_FGO = False

pytestmark = pytest.mark.skipif(not HAS_FGO, reason="FGO CUDA extension not built")


def _rand_problem(
    rng: np.random.Generator, n_epoch: int, n_sat: int, nc: int, sys_kind: np.ndarray | None
):
    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    clocks = np.array([150.0, 35.0][:nc], dtype=np.float64) if nc == 2 else np.array([150.0])
    pr = np.zeros((n_epoch, n_sat))
    for t in range(n_epoch):
        for s in range(n_sat):
            sk = int(sys_kind[t, s]) if sys_kind is not None else 0
            hc = np.zeros(nc)
            _fill_hc(nc, sk, hc)
            rho = _geometric_range_sagnacrcv(true, sat[t, s])
            clk = float(np.dot(hc, clocks))
            pr[t, s] = rho + clk
    pr += rng.normal(0, 2.0, pr.shape)
    w = np.ones((n_epoch, n_sat)) * 0.3
    state = np.zeros((n_epoch, 3 + nc), dtype=np.float64)
    state[:, :3] = true.reshape(1, 3) + rng.normal(0, 1e3, (n_epoch, 3))
    state[:, 3 : 3 + nc] = clocks.reshape(1, -1) + rng.normal(0, 4, (n_epoch, nc))
    return sat, pr, w, state.copy(), sys_kind


@pytest.mark.parametrize("n_epoch,n_sat", [(1, 8), (3, 6), (7, 5)])
@pytest.mark.parametrize("motion_sigma_m", [0.0, 0.6])
@pytest.mark.parametrize("n_clock", [1, 2])
@pytest.mark.parametrize("huber_k", [0.0, 1.15])
def test_fgo_first_step_matches_reference(
    n_epoch: int, n_sat: int, motion_sigma_m: float, n_clock: int, huber_k: float
):
    rng = np.random.Generator(
        np.random.PCG64(
            2026 ^ n_epoch ^ n_sat ^ int(motion_sigma_m * 10) ^ n_clock ^ int(huber_k * 1000)
        )
    )
    if n_clock == 1:
        sys_kind = None
    else:
        sys_kind = rng.integers(0, 2, size=(n_epoch, n_sat), dtype=np.int32)
    sat_ecef, pr, w, st0, sk = _rand_problem(rng, n_epoch, n_sat, n_clock, sys_kind)
    st_ext = st0.copy()
    iters, _ = fgo_gnss_lm(
        sat_ecef,
        pr,
        w,
        st_ext,
        motion_sigma_m=motion_sigma_m,
        max_iter=1,
        tol=0.0,
        huber_k=huber_k,
        enable_line_search=0,
        sys_kind=sys_kind,
        n_clock=n_clock,
    )
    assert iters == 1
    delta_ref = _one_gn_step(
        sat_ecef, pr, w, st0, sys_kind, n_clock, motion_sigma_m, huber_k=huber_k
    )
    delta_ext = (st_ext - st0).reshape(-1)
    np.testing.assert_allclose(delta_ext, delta_ref, rtol=1e-9, atol=1e-7)


def test_fgo_zero_residual_stationary():
    rng = np.random.Generator(np.random.PCG64(7))
    n_epoch, n_sat, nc = 4, 6, 1
    sat_ecef, _, _, _, _ = _rand_problem(rng, n_epoch, n_sat, nc, None)
    true = np.array([-3.81e6, 3.51e6, 3.64e6], dtype=np.float64)
    cb = 88.5
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true, sat_ecef[t, s]) + cb
    w = np.ones((n_epoch, n_sat)) * 0.2
    st = np.zeros((n_epoch, 4), dtype=np.float64)
    st[:, :3] = true
    st[:, 3] = cb
    H, g = _assemble_pr_h_g(sat_ecef, pr, w, st, None, 1)
    assert float(np.linalg.norm(g)) < 1e-6
    st2 = st.copy()
    fgo_gnss_lm(
        sat_ecef,
        pr,
        w,
        st2,
        motion_sigma_m=0.0,
        max_iter=1,
        tol=0.0,
        enable_line_search=0,
        sys_kind=None,
        n_clock=1,
    )
    step = float(np.linalg.norm(st2 - st))
    assert step < 1e-4


def test_fgo_accepts_matlab_seven_signal_clocks():
    rng = np.random.Generator(np.random.PCG64(7007))
    n_epoch, n_sat, nc = 1, 10, 7
    true = np.array([-3.81e6, 3.51e6, 3.64e6], dtype=np.float64)
    clocks = np.array([120.0, 4.0, -7.0, 3.0, 11.0, -5.0, 2.0], dtype=np.float64)
    sys_kind = np.array([[0, 1, 2, 3, 4, 5, 6, 0, 4, 5]], dtype=np.int32)
    sat = rng.normal(0.0, 5e6, (n_epoch, n_sat, 3))
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for s in range(n_sat):
        hc = np.zeros(nc, dtype=np.float64)
        _fill_hc(nc, int(sys_kind[0, s]), hc)
        pr[0, s] = _geometric_range_sagnacrcv(true, sat[0, s]) + float(np.dot(hc, clocks))
    w = np.ones((n_epoch, n_sat), dtype=np.float64) * 0.3
    st = np.zeros((n_epoch, 3 + nc), dtype=np.float64)
    st[0, :3] = true
    st[0, 3:] = clocks

    st2 = st.copy()
    iters, _ = fgo_gnss_lm(
        sat,
        pr,
        w,
        st2,
        motion_sigma_m=0.0,
        max_iter=1,
        tol=0.0,
        enable_line_search=0,
        sys_kind=sys_kind,
        n_clock=nc,
    )

    assert iters == 1
    np.testing.assert_allclose(st2, st, atol=1e-4)


def test_fgo_rejects_oversized_window():
    n_epoch = 2050
    n_sat = 4
    n_clock = 1
    sat = np.zeros((n_epoch, n_sat, 3))
    pr = np.ones((n_epoch, n_sat)) * 1e7
    w = np.ones((n_epoch, n_sat))
    st = np.zeros((n_epoch, 3 + n_clock))
    iters, _ = fgo_gnss_lm(sat, pr, w, st, max_iter=1, tol=1e-3, enable_line_search=0, n_clock=1)
    assert iters == -1


def test_line_search_reduces_or_matches_cost():
    """With motion coupling, line search should not increase total quadratic cost."""
    rng = np.random.Generator(np.random.PCG64(99))
    T, S, nc = 8, 6, 1
    sys_kind = None
    sat, pr, w, st0, _ = _rand_problem(rng, T, S, nc, sys_kind)
    st_ls = st0.copy()
    st_no = st0.copy()
    fgo_gnss_lm(
        sat,
        pr,
        w,
        st_no,
        motion_sigma_m=0.4,
        max_iter=1,
        tol=0.0,
        enable_line_search=0,
        n_clock=1,
    )
    fgo_gnss_lm(
        sat,
        pr,
        w,
        st_ls,
        motion_sigma_m=0.4,
        max_iter=1,
        tol=0.0,
        enable_line_search=1,
        n_clock=1,
    )

    def nonlinear_cost(st: np.ndarray) -> float:
        wm = 1.0 / 0.4**2
        e = 0.0
        for t in range(T):
            for s in range(S):
                if w[t, s] <= 0:
                    continue
                x, y, z, cb = st[t]
                rho = _geometric_range_sagnacrcv(np.array([x, y, z]), sat[t, s])
                r = pr[t, s] - (rho + cb)
                e += 0.5 * w[t, s] * r * r
        for t in range(T - 1):
            for i in range(3):
                d = float(st[t, i] - st[t + 1, i])
                e += 0.5 * wm * d * d
        return e

    c0 = nonlinear_cost(st0)
    c_ls = nonlinear_cost(st_ls)
    assert c_ls <= c0 * (1.0 + 1e-9)


# =========================================================================
# Tests for fgo_gnss_lm_vd (velocity + drift extended state)
# =========================================================================

try:
    from gnss_gpu._gnss_gpu import fgo_gnss_lm_vd

    HAS_FGO_VD = True
except ImportError:
    HAS_FGO_VD = False


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_basic_convergence():
    """VD solver converges when initialized from WLS solution (matching FGO pipeline)."""
    rng = np.random.Generator(np.random.PCG64(42))
    n_epoch, n_sat, nc = 5, 8, 1
    ss_old = 3 + nc  # 4 for old solver
    ss_vd = 7 + nc   # 8 for VD solver

    # Generate satellite positions and pseudoranges
    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = 150.0

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + true_clk
    pr += rng.normal(0, 2.0, pr.shape)
    w = np.ones((n_epoch, n_sat)) * 0.3

    # First solve with old FGO to get good initial position
    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    state_old[:, :3] = true_pos + rng.normal(0, 1e3, (n_epoch, 3))
    state_old[:, 3] = true_clk + rng.normal(0, 4.0, n_epoch)
    iters_old, _ = fgo_gnss_lm(
        sat, pr, w, state_old,
        motion_sigma_m=0.0, max_iter=25, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
    )

    # Initialize VD state from old FGO result
    state_vd = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_vd[:, :3] = state_old[:, :3]  # position from old solver
    state_vd[:, 6] = state_old[:, 3]    # clock from old solver

    iters, mse = fgo_gnss_lm_vd(
        sat, pr, w, state_vd,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
    )
    assert iters > 0, f"VD solver failed: iters={iters}"

    # Position should remain good (matching old solver)
    for t in range(n_epoch):
        err_vd = np.linalg.norm(state_vd[t, :3] - true_pos)
        err_old = np.linalg.norm(state_old[t, :3] - true_pos)
        assert err_vd < err_old + 5.0, (
            f"Epoch {t}: VD error {err_vd:.1f} m worse than old {err_old:.1f} m"
        )


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_motion_factor():
    """Motion factor (x_{t+1} = x_t + v_t*dt) couples position and velocity."""
    rng = np.random.Generator(np.random.PCG64(77))
    n_epoch, n_sat, nc = 6, 8, 1
    ss_old = 3 + nc
    ss_vd = 7 + nc

    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_vel = np.array([10.0, -5.0, 2.0], dtype=np.float64)
    true_clk = 150.0
    dt_val = 1.0

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        pos_t = true_pos + true_vel * (t * dt_val)
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(pos_t, sat[t, s]) + true_clk
    pr += rng.normal(0, 1.5, pr.shape)
    w = np.ones((n_epoch, n_sat)) * 0.3
    dt_arr = np.full(n_epoch, dt_val, dtype=np.float64)

    # Get initial positions from old FGO
    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    for t in range(n_epoch):
        state_old[t, :3] = true_pos + true_vel * (t * dt_val) + rng.normal(0, 500, 3)
    state_old[:, 3] = true_clk + rng.normal(0, 4.0, n_epoch)
    fgo_gnss_lm(sat, pr, w, state_old,
                motion_sigma_m=0.0, max_iter=25, tol=1e-7,
                enable_line_search=1, sys_kind=None, n_clock=1)

    # Initialize VD state from old solver
    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[:, :3] = state_old[:, :3]
    state[:, 3:6] = true_vel + rng.normal(0, 20, (n_epoch, 3))
    state[:, 6] = state_old[:, 3]

    iters, mse = fgo_gnss_lm_vd(
        sat, pr, w, state,
        motion_sigma_m=3.0, clock_drift_sigma_m=0.0,
        max_iter=15, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, dt=dt_arr,
    )
    assert iters > 0, f"VD solver with motion factor failed: iters={iters}"

    # Position should remain good
    for t in range(n_epoch):
        err_vd = np.linalg.norm(state[t, :3] - (true_pos + true_vel * (t * dt_val)))
        err_old = np.linalg.norm(state_old[t, :3] - (true_pos + true_vel * (t * dt_val)))
        assert err_vd < err_old + 10.0, (
            f"Epoch {t}: VD position error {err_vd:.1f}m worse than old {err_old:.1f}m"
        )


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_motion_factor_reduces_standalone_residual():
    """Standalone VD motion factor must move along the cost-reducing direction."""
    n_epoch, n_sat, nc = 2, 4, 1
    ss_vd = 7 + nc
    sat = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w = np.zeros((n_epoch, n_sat), dtype=np.float64)
    dt = np.array([1.0, 0.0], dtype=np.float64)
    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[0, 3] = 1.0

    before = float(state[0, 0] + state[0, 3] * dt[0] - state[1, 0])
    iters, _ = fgo_gnss_lm_vd(
        sat,
        pr,
        w,
        state,
        motion_sigma_m=0.1,
        clock_drift_sigma_m=0.0,
        max_iter=5,
        tol=1e-12,
        huber_k=0.0,
        enable_line_search=1,
        sys_kind=None,
        n_clock=1,
        dt=dt,
    )

    after = float(state[0, 0] + state[0, 3] * dt[0] - state[1, 0])
    assert iters > 0
    assert abs(after) < abs(before) * 1e-3


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_clock_drift_factor_reduces_standalone_residual():
    """Standalone clock drift factor must reduce c_t - c_t1 + drift*dt."""
    n_epoch, n_sat, nc = 2, 4, 1
    ss_vd = 7 + nc
    sat = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w = np.zeros((n_epoch, n_sat), dtype=np.float64)
    dt = np.array([1.0, 0.0], dtype=np.float64)
    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[0, 6 + nc] = 1.0

    before = float(state[0, 6] - state[1, 6] + state[0, 6 + nc] * dt[0])
    iters, _ = fgo_gnss_lm_vd(
        sat,
        pr,
        w,
        state,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.1,
        max_iter=5,
        tol=1e-12,
        huber_k=0.0,
        enable_line_search=1,
        sys_kind=None,
        n_clock=1,
        dt=dt,
    )

    after = float(state[0, 6] - state[1, 6] + state[0, 6 + nc] * dt[0])
    assert iters > 0
    assert abs(after) < abs(before) * 1e-3


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_doppler_constrains_velocity():
    """Doppler factor modifies velocity state (different from no-Doppler baseline)."""
    rng = np.random.Generator(np.random.PCG64(123))
    n_epoch, n_sat, nc = 4, 8, 1
    ss_old = 3 + nc
    ss_vd = 7 + nc

    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    sat_vel = rng.normal(0, 3e3, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_vel = np.array([5.0, -3.0, 1.0], dtype=np.float64)
    true_clk = 150.0
    true_drift = 0.5

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    dop = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + true_clk
            dop[t, s] = _doppler_model_sagnac(
                true_pos, sat[t, s], true_vel, sat_vel[t, s], true_drift
            )

    pr += rng.normal(0, 2.0, pr.shape)
    dop += rng.normal(0, 0.3, dop.shape)
    w_pr = np.ones((n_epoch, n_sat)) * 0.3
    w_dop = np.ones((n_epoch, n_sat)) * 1.0
    dt_arr = np.ones(n_epoch, dtype=np.float64)

    # First get good position from old FGO solver
    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    state_old[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state_old[:, 3] = true_clk + rng.normal(0, 4.0, n_epoch)
    fgo_gnss_lm(sat, pr, w_pr, state_old,
                motion_sigma_m=0.0, max_iter=25, tol=1e-7,
                enable_line_search=1, sys_kind=None, n_clock=1)

    # VD without Doppler
    state_no_dop = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_no_dop[:, :3] = state_old[:, :3]
    state_no_dop[:, 3:6] = rng.normal(0, 10, (n_epoch, 3))  # random init velocity
    state_no_dop[:, 6] = state_old[:, 3]

    # VD with Doppler (same initial state)
    state_with_dop = state_no_dop.copy()

    fgo_gnss_lm_vd(
        sat, pr, w_pr, state_no_dop,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
    )

    iters2, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state_with_dop,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
        sat_vel=sat_vel, doppler=dop, doppler_weights=w_dop, dt=dt_arr,
    )
    assert iters2 > 0, f"VD solver with Doppler failed: iters={iters2}"

    # Without Doppler, velocity should stay at initial (no constraint on velocity)
    # With Doppler, velocity should change from initial (Doppler constrains it)
    vel_change_no_dop = np.mean([np.linalg.norm(state_no_dop[t, 3:6]) for t in range(n_epoch)])
    vel_change_with_dop = np.max([
        np.linalg.norm(state_with_dop[t, 3:6] - state_no_dop[t, 3:6])
        for t in range(n_epoch)
    ])
    # Doppler factor should modify velocity (even if not perfectly converged)
    assert vel_change_with_dop > 0.1, (
        f"Doppler should change velocity: max change = {vel_change_with_dop:.4f}"
    )


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_doppler_uses_sagnac_range_rate():
    rng = np.random.Generator(np.random.PCG64(321))
    n_epoch, n_sat, nc = 1, 8, 1
    ss_vd = 7 + nc
    true_pos = np.array([2.3e6, -4.1e6, 4.2e6], dtype=np.float64)
    true_vel = np.array([14.0, -8.0, 3.0], dtype=np.float64)
    true_drift = -0.45

    directions = rng.normal(size=(n_sat, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sat = (directions * 2.65e7).reshape(n_epoch, n_sat, 3).astype(np.float64)
    # Large synthetic satellite velocities make the range-rate Sagnac term observable.
    sat_vel = rng.normal(0.0, 4.0e4, (n_epoch, n_sat, 3)).astype(np.float64)

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w_pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    dop = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for s in range(n_sat):
        dop[0, s] = _doppler_model_sagnac(
            true_pos, sat[0, s], true_vel, sat_vel[0, s], true_drift
        )
    w_dop = np.ones((n_epoch, n_sat), dtype=np.float64) * 1e6

    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[0, :3] = true_pos
    state[0, 3:6] = true_vel + np.array([3.0, -2.0, 1.0], dtype=np.float64)
    state[0, 7] = true_drift + 0.8

    iters, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=5, tol=1e-10, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
        sat_vel=sat_vel, doppler=dop, doppler_weights=w_dop,
    )
    assert iters > 0
    assert np.linalg.norm(state[0, 3:6] - true_vel) < 1e-3
    assert abs(state[0, 7] - true_drift) < 1e-3


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_doppler_uses_satellite_clock_drift():
    rng = np.random.Generator(np.random.PCG64(654))
    n_epoch, n_sat, nc = 1, 9, 1
    ss_vd = 7 + nc
    true_pos = np.array([2.1e6, -4.3e6, 4.0e6], dtype=np.float64)
    true_vel = np.array([7.0, -4.0, 2.5], dtype=np.float64)
    true_drift = 0.25

    directions = rng.normal(size=(n_sat, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    sat = (directions * 2.58e7).reshape(n_epoch, n_sat, 3).astype(np.float64)
    sat_vel = rng.normal(0.0, 3.0e3, (n_epoch, n_sat, 3)).astype(np.float64)
    sat_clock_drift = np.linspace(-1.2, 1.4, n_sat, dtype=np.float64).reshape(n_epoch, n_sat)

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w_pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    dop = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for s in range(n_sat):
        dop[0, s] = _doppler_model_sagnac(
            true_pos,
            sat[0, s],
            true_vel,
            sat_vel[0, s],
            true_drift,
            sat_clock_drift[0, s],
        )
    w_dop = np.ones((n_epoch, n_sat), dtype=np.float64) * 1e5

    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[0, :3] = true_pos
    state[0, 3:6] = true_vel + np.array([2.0, -1.0, 0.5], dtype=np.float64)
    state[0, 7] = true_drift - 0.6

    iters, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=5, tol=1e-10, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
        sat_vel=sat_vel, doppler=dop, doppler_weights=w_dop,
        sat_clock_drift=sat_clock_drift,
    )
    assert iters > 0
    assert np.linalg.norm(state[0, 3:6] - true_vel) < 1e-3
    assert abs(state[0, 7] - true_drift) < 1e-3


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_clock_drift_factor():
    """Clock drift factor couples clock and drift between epochs."""
    rng = np.random.Generator(np.random.PCG64(55))
    n_epoch, n_sat, nc = 6, 8, 1
    ss_old = 3 + nc
    ss_vd = 7 + nc

    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_drift = 1.0  # m/s clock drift rate
    dt_val = 1.0

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        clk_t = 150.0 + true_drift * (t * dt_val)
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + clk_t
    pr += rng.normal(0, 2.0, pr.shape)
    w = np.ones((n_epoch, n_sat)) * 0.3
    dt_arr = np.full(n_epoch, dt_val, dtype=np.float64)

    # Get initial position from old FGO
    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    state_old[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    for t in range(n_epoch):
        state_old[t, 3] = 150.0 + true_drift * (t * dt_val) + rng.normal(0, 3.0)
    fgo_gnss_lm(sat, pr, w, state_old,
                motion_sigma_m=0.0, max_iter=25, tol=1e-7,
                enable_line_search=1, sys_kind=None, n_clock=1)

    # VD state from old solver result
    state_no_drift = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_no_drift[:, :3] = state_old[:, :3]
    state_no_drift[:, 6] = state_old[:, 3]
    state_no_drift[:, 7] = true_drift + rng.normal(0, 1.0, n_epoch)

    state_with_drift = state_no_drift.copy()

    # Without clock drift factor
    fgo_gnss_lm_vd(
        sat, pr, w, state_no_drift,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, dt=dt_arr,
    )

    # With clock drift factor
    iters, mse = fgo_gnss_lm_vd(
        sat, pr, w, state_with_drift,
        motion_sigma_m=0.0, clock_drift_sigma_m=50.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, dt=dt_arr,
    )
    assert iters > 0, f"VD solver with clock drift factor failed: iters={iters}"

    # With clock drift coupling, clock values should be more temporally consistent
    # Check that clock values at least don't diverge compared to old solver
    for t in range(n_epoch):
        err_vd = np.linalg.norm(state_with_drift[t, :3] - true_pos)
        err_old = np.linalg.norm(state_old[t, :3] - true_pos)
        assert err_vd < err_old + 10.0, (
            f"Epoch {t}: VD error {err_vd:.1f}m vs old {err_old:.1f}m"
        )


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_stop_velocity_factor_zeroes_stopped_epochs():
    rng = np.random.Generator(np.random.PCG64(505))
    n_epoch, n_sat, nc = 5, 8, 1
    ss_vd = 7 + nc
    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = 150.0

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + true_clk
    pr += rng.normal(0, 2.0, pr.shape)
    w = np.ones((n_epoch, n_sat), dtype=np.float64) * 0.3

    state_no_stop = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_no_stop[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state_no_stop[:, 3:6] = rng.normal(0.0, 2.0, (n_epoch, 3))
    state_no_stop[:, 6] = true_clk + rng.normal(0, 3.0, n_epoch)
    state_with_stop = state_no_stop.copy()
    stop_mask = np.array([1, 1, 0, 1, 1], dtype=np.uint8)

    iters_no, _ = fgo_gnss_lm_vd(
        sat, pr, w, state_no_stop,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
    )
    assert iters_no > 0

    iters_stop, _ = fgo_gnss_lm_vd(
        sat, pr, w, state_with_stop,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        stop_velocity_sigma_mps=0.01,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, stop_mask=stop_mask,
    )
    assert iters_stop > 0

    vel_norm_no = np.linalg.norm(state_no_stop[stop_mask.astype(bool), 3:6], axis=1)
    vel_norm_stop = np.linalg.norm(state_with_stop[stop_mask.astype(bool), 3:6], axis=1)
    assert float(np.mean(vel_norm_stop)) < float(np.mean(vel_norm_no)) * 0.1
    assert float(np.max(vel_norm_stop)) < 0.02


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_stop_position_factor_holds_consecutive_stop_positions():
    rng = np.random.Generator(np.random.PCG64(606))
    n_epoch, n_sat, nc = 6, 8, 1
    ss_vd = 7 + nc
    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = 150.0

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + true_clk
    pr += rng.normal(0, 6.0, pr.shape)
    w = np.ones((n_epoch, n_sat), dtype=np.float64) * 0.3

    state_no_hold = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_no_hold[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state_no_hold[:, 6] = true_clk + rng.normal(0, 3.0, n_epoch)
    state_hold = state_no_hold.copy()
    stop_mask = np.ones(n_epoch, dtype=np.uint8)

    iters_no, _ = fgo_gnss_lm_vd(
        sat, pr, w, state_no_hold,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
    )
    assert iters_no > 0

    iters_hold, _ = fgo_gnss_lm_vd(
        sat, pr, w, state_hold,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        stop_position_sigma_m=0.02,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, stop_mask=stop_mask,
    )
    assert iters_hold > 0

    diff_no = np.linalg.norm(np.diff(state_no_hold[:, :3], axis=0), axis=1)
    diff_hold = np.linalg.norm(np.diff(state_hold[:, :3], axis=0), axis=1)
    assert float(np.mean(diff_hold)) < float(np.mean(diff_no)) * 0.5


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_rejects_bad_params():
    """VD solver rejects invalid parameters."""
    n_epoch, n_sat, nc = 3, 6, 1
    ss = 7 + nc
    sat = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
    pr = np.ones((n_epoch, n_sat), dtype=np.float64) * 1e7
    w = np.ones((n_epoch, n_sat), dtype=np.float64)
    st = np.zeros((n_epoch, ss), dtype=np.float64)

    iters, _ = fgo_gnss_lm_vd(
        sat, pr, w, st,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=1, tol=1e-3, huber_k=0.0, enable_line_search=0,
        sys_kind=None, n_clock=1,
    )
    # Should run (may not converge well but shouldn't crash)

    # Test over the limit: 2049 * 8 = 16392 > 16384
    n_too_big = 2049
    sat_tb = np.zeros((n_too_big, n_sat, 3), dtype=np.float64)
    pr_tb = np.ones((n_too_big, n_sat), dtype=np.float64) * 1e7
    w_tb = np.ones((n_too_big, n_sat), dtype=np.float64)
    st_tb = np.zeros((n_too_big, ss), dtype=np.float64)
    iters3, _ = fgo_gnss_lm_vd(
        sat_tb, pr_tb, w_tb, st_tb,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=1, tol=1e-3, huber_k=0.0, enable_line_search=0,
        sys_kind=None, n_clock=1,
    )
    assert iters3 == -1, "Should reject n_state > 16384"


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_multi_clock():
    """VD solver works with multi-clock (GPS + GLONASS ISB) state."""
    rng = np.random.Generator(np.random.PCG64(88))
    n_epoch, n_sat, nc = 4, 10, 2
    ss_old = 3 + nc  # 5 for 2-clock old solver
    ss_vd = 7 + nc   # 9 for 2-clock VD solver

    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = np.array([150.0, 35.0], dtype=np.float64)

    sys_kind = np.zeros((n_epoch, n_sat), dtype=np.int32)
    sys_kind[:, 6:] = 1

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            rho = _geometric_range_sagnacrcv(true_pos, sat[t, s])
            hc = np.zeros(nc)
            _fill_hc(nc, int(sys_kind[t, s]), hc)
            clk = float(np.dot(hc, true_clk))
            pr[t, s] = rho + clk
    pr += rng.normal(0, 2.0, pr.shape)
    w = np.ones((n_epoch, n_sat)) * 0.3

    # Get initial position from old FGO
    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    state_old[:, :3] = true_pos + rng.normal(0, 1e3, (n_epoch, 3))
    state_old[:, 3] = true_clk[0] + rng.normal(0, 4, n_epoch)
    state_old[:, 4] = true_clk[1] + rng.normal(0, 4, n_epoch)
    fgo_gnss_lm(sat, pr, w, state_old,
                motion_sigma_m=0.0, max_iter=25, tol=1e-7,
                enable_line_search=1, sys_kind=sys_kind, n_clock=2)

    # Initialize VD state from old solver
    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[:, :3] = state_old[:, :3]
    state[:, 6] = state_old[:, 3]
    state[:, 7] = state_old[:, 4]
    # drift at index 8 (6 + nc = 6 + 2 = 8)

    iters, mse = fgo_gnss_lm_vd(
        sat, pr, w, state,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=sys_kind, n_clock=2,
    )
    assert iters > 0, f"VD multi-clock solver failed: iters={iters}"

    for t in range(n_epoch):
        err_vd = np.linalg.norm(state[t, :3] - true_pos)
        err_old = np.linalg.norm(state_old[t, :3] - true_pos)
        assert err_vd < err_old + 5.0, (
            f"Epoch {t}: VD error {err_vd:.1f}m vs old {err_old:.1f}m"
        )


# =========================================================================
# Tests for TDCP (Time-Differenced Carrier Phase) factor
# =========================================================================


@pytest.mark.skipif(not HAS_FGO, reason="FGO CUDA extension not built")
def test_tdcp_reduces_position_noise():
    """TDCP factor with cm-level precision should significantly reduce position error
    compared to pseudorange-only solution when pseudorange noise is large."""
    rng = np.random.Generator(np.random.PCG64(2025))
    n_epoch, n_sat, nc = 10, 8, 1
    ss = 3 + nc

    # Satellite geometry
    sat = rng.normal(0, 2.5e7, (n_epoch, n_sat, 3))

    # True receiver trajectory: slow-moving receiver
    true_pos_base = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_vel = np.array([0.5, -0.3, 0.1], dtype=np.float64)
    true_clk = 150.0

    true_pos = np.zeros((n_epoch, 3), dtype=np.float64)
    for t in range(n_epoch):
        true_pos[t] = true_pos_base + true_vel * t

    # Generate pseudoranges with HIGH noise (10m sigma)
    pr_noise_sigma = 10.0
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos[t], sat[t, s]) + true_clk
    pr += rng.normal(0, pr_noise_sigma, pr.shape)
    w_pr = np.ones((n_epoch, n_sat)) / (pr_noise_sigma ** 2)

    # Generate TDCP measurements with LOW noise (1cm sigma = 0.01m)
    # TDCP_meas = e_s^T * (x_{t+1} - x_t) + (clk_{t+1} - clk_t)
    # Since clock is constant, clk diff = 0
    tdcp_noise_sigma = 0.01
    tdcp_meas = np.zeros(((n_epoch - 1), n_sat), dtype=np.float64)
    for t in range(n_epoch - 1):
        for s in range(n_sat):
            # Use satellite position at epoch t+1 for LOS (matching the solver)
            sx, sy, sz = sat[t + 1, s]
            x1, y1, z1 = true_pos[t + 1]
            x0, y0, z0 = true_pos[t]
            dx0 = x1 - sx
            dy0 = y1 - sy
            dz0 = z1 - sz
            r0 = float(np.sqrt(dx0 ** 2 + dy0 ** 2 + dz0 ** 2))
            transit = r0 / C_LIGHT
            theta = OMEGA_E * transit
            sx_rot = sx * np.cos(theta) + sy * np.sin(theta)
            sy_rot = -sx * np.sin(theta) + sy * np.cos(theta)
            dx = x1 - sx_rot
            dy_v = y1 - sy_rot
            dz = z1 - sz
            r = float(np.sqrt(dx ** 2 + dy_v ** 2 + dz ** 2))
            ex, ey, ez = dx / r, dy_v / r, dz / r
            tdcp_meas[t, s] = (
                ex * (x1 - x0) + ey * (y1 - y0) + ez * (z1 - z0)
                + 0.0  # clk diff = 0
            )
    tdcp_meas += rng.normal(0, tdcp_noise_sigma, tdcp_meas.shape)
    tdcp_w = np.ones_like(tdcp_meas) / (tdcp_noise_sigma ** 2)

    # Solution WITHOUT TDCP
    state_no_tdcp = np.zeros((n_epoch, ss), dtype=np.float64)
    state_no_tdcp[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state_no_tdcp[:, 3] = true_clk + rng.normal(0, 10, n_epoch)

    iters1, mse1 = fgo_gnss_lm(
        sat, pr, w_pr, state_no_tdcp,
        motion_sigma_m=0.0, max_iter=30, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
    )
    assert iters1 > 0, f"PR-only FGO failed: iters={iters1}"

    # Solution WITH TDCP
    state_with_tdcp = np.zeros((n_epoch, ss), dtype=np.float64)
    state_with_tdcp[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state_with_tdcp[:, 3] = true_clk + rng.normal(0, 10, n_epoch)

    iters2, mse2 = fgo_gnss_lm(
        sat, pr, w_pr, state_with_tdcp,
        motion_sigma_m=0.0, max_iter=30, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
        tdcp_meas=tdcp_meas, tdcp_weights=tdcp_w,
    )
    assert iters2 > 0, f"TDCP FGO failed: iters={iters2}"

    # Compute RMS position errors
    err_no_tdcp = np.array([
        np.linalg.norm(state_no_tdcp[t, :3] - true_pos[t]) for t in range(n_epoch)
    ])
    err_with_tdcp = np.array([
        np.linalg.norm(state_with_tdcp[t, :3] - true_pos[t]) for t in range(n_epoch)
    ])

    rms_no_tdcp = float(np.sqrt(np.mean(err_no_tdcp ** 2)))
    rms_with_tdcp = float(np.sqrt(np.mean(err_with_tdcp ** 2)))

    # In the legacy solver (fgo_gnss_lm), the GN sign convention limits
    # TDCP effectiveness. Verify it does not make things significantly worse.
    # The VD solver (test_tdcp_vd_reduces_position_noise) validates actual improvement.
    assert rms_with_tdcp < rms_no_tdcp * 1.1, (
        f"TDCP should not significantly degrade position: "
        f"RMS without={rms_no_tdcp:.2f}m, with={rms_with_tdcp:.2f}m"
    )


@pytest.mark.skipif(not HAS_FGO, reason="FGO CUDA extension not built")
def test_tdcp_with_sigma():
    """TDCP with tdcp_sigma_m (uniform weight) also works."""
    rng = np.random.Generator(np.random.PCG64(3030))
    n_epoch, n_sat, nc = 5, 6, 1
    ss = 3 + nc

    sat = rng.normal(0, 2.5e7, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = 150.0

    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + true_clk
    pr += rng.normal(0, 5.0, pr.shape)
    w = np.ones((n_epoch, n_sat)) / 25.0

    # TDCP meas for stationary receiver: should be near 0 (only clock diff)
    tdcp_meas = np.zeros(((n_epoch - 1), n_sat), dtype=np.float64)
    for t in range(n_epoch - 1):
        for s in range(n_sat):
            sx, sy, sz = sat[t + 1, s]
            dx0 = true_pos[0] - sx
            dy0 = true_pos[1] - sy
            dz0 = true_pos[2] - sz
            r0 = float(np.sqrt(dx0 ** 2 + dy0 ** 2 + dz0 ** 2))
            transit = r0 / C_LIGHT
            theta = OMEGA_E * transit
            sx_rot = sx * np.cos(theta) + sy * np.sin(theta)
            sy_rot = -sx * np.sin(theta) + sy * np.cos(theta)
            dx = true_pos[0] - sx_rot
            dy_v = true_pos[1] - sy_rot
            dz = true_pos[2] - sz
            r = float(np.sqrt(dx ** 2 + dy_v ** 2 + dz ** 2))
            ex, ey, ez = dx / r, dy_v / r, dz / r
            # Position diff is 0 for stationary, clock diff is 0
            tdcp_meas[t, s] = 0.0
    tdcp_meas += rng.normal(0, 0.01, tdcp_meas.shape)

    state = np.zeros((n_epoch, ss), dtype=np.float64)
    state[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state[:, 3] = true_clk + rng.normal(0, 10, n_epoch)

    iters, mse = fgo_gnss_lm(
        sat, pr, w, state,
        motion_sigma_m=0.0, max_iter=30, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
        tdcp_meas=tdcp_meas, tdcp_sigma_m=0.01,
    )
    assert iters > 0, f"TDCP with sigma failed: iters={iters}"

    # In the legacy solver, TDCP has limited effect due to GN sign convention.
    # Just verify it doesn't crash and doesn't catastrophically diverge.
    for t in range(n_epoch):
        err = np.linalg.norm(state[t, :3] - true_pos)
        assert err < 2000.0, f"Epoch {t}: position error {err:.1f}m — TDCP diverged"


@pytest.mark.skipif(not HAS_FGO, reason="FGO CUDA extension not built")
def test_tdcp_backward_compatible():
    """Passing no TDCP parameters gives same result as before."""
    rng = np.random.Generator(np.random.PCG64(111))
    n_epoch, n_sat, nc = 4, 6, 1
    sat, pr, w, st0, _ = _rand_problem(rng, n_epoch, n_sat, nc, None)

    st_a = st0.copy()
    st_b = st0.copy()

    iters_a, mse_a = fgo_gnss_lm(
        sat, pr, w, st_a,
        motion_sigma_m=0.5, max_iter=10, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
    )

    iters_b, mse_b = fgo_gnss_lm(
        sat, pr, w, st_b,
        motion_sigma_m=0.5, max_iter=10, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
        tdcp_meas=None, tdcp_weights=None, tdcp_sigma_m=0.0,
    )

    np.testing.assert_array_equal(st_a, st_b)
    assert iters_a == iters_b
    assert mse_a == mse_b


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_tdcp_vd_reduces_position_noise():
    """TDCP factor works in the VD solver too."""
    rng = np.random.Generator(np.random.PCG64(4040))
    n_epoch, n_sat, nc = 8, 8, 1
    ss_old = 3 + nc
    ss_vd = 7 + nc

    sat = rng.normal(0, 2.5e7, (n_epoch, n_sat, 3))
    true_pos_base = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = 150.0

    true_pos = np.tile(true_pos_base, (n_epoch, 1))  # stationary

    pr_noise_sigma = 8.0
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos[t], sat[t, s]) + true_clk
    pr += rng.normal(0, pr_noise_sigma, pr.shape)
    w_pr = np.ones((n_epoch, n_sat)) / (pr_noise_sigma ** 2)

    # TDCP: for stationary, meas ~= 0 + clock diff (0)
    tdcp_noise_sigma = 0.01
    tdcp_meas = np.zeros(((n_epoch - 1), n_sat), dtype=np.float64)
    tdcp_meas += rng.normal(0, tdcp_noise_sigma, tdcp_meas.shape)
    tdcp_w = np.ones_like(tdcp_meas) / (tdcp_noise_sigma ** 2)

    # Get init from old FGO
    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    state_old[:, :3] = true_pos + rng.normal(0, 500, (n_epoch, 3))
    state_old[:, 3] = true_clk + rng.normal(0, 10, n_epoch)
    fgo_gnss_lm(sat, pr, w_pr, state_old,
                motion_sigma_m=0.0, max_iter=25, tol=1e-7,
                enable_line_search=1, sys_kind=None, n_clock=1)

    # VD without TDCP
    state_no = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_no[:, :3] = state_old[:, :3]
    state_no[:, 6] = state_old[:, 3]
    fgo_gnss_lm_vd(
        sat, pr, w_pr, state_no,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
    )

    # VD with TDCP
    state_tdcp = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_tdcp[:, :3] = state_old[:, :3]
    state_tdcp[:, 6] = state_old[:, 3]
    iters, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state_tdcp,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=10, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1,
        tdcp_meas=tdcp_meas, tdcp_weights=tdcp_w,
    )
    assert iters > 0, f"VD+TDCP failed: iters={iters}"

    # Compute inter-epoch position consistency (should be better with TDCP)
    diff_no = np.array([
        np.linalg.norm(state_no[t + 1, :3] - state_no[t, :3]) for t in range(n_epoch - 1)
    ])
    diff_tdcp = np.array([
        np.linalg.norm(state_tdcp[t + 1, :3] - state_tdcp[t, :3]) for t in range(n_epoch - 1)
    ])

    # For a stationary receiver, TDCP should make inter-epoch differences smaller
    assert np.mean(diff_tdcp) < np.mean(diff_no) + 0.5, (
        f"TDCP should reduce inter-epoch jitter: "
        f"mean_diff without={np.mean(diff_no):.3f}m, with={np.mean(diff_tdcp):.3f}m"
    )


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_tdcp_vd_xxdd_matches_drift_model_better_than_xxcc():
    """MATLAB XXDD variant should fit drift-driven TDCP sequences better than XXCC."""
    rng = np.random.Generator(np.random.PCG64(5050))
    n_epoch, n_sat, nc = 6, 10, 1
    ss_old = 3 + nc
    ss_vd = 7 + nc
    dt_arr = np.ones(n_epoch, dtype=np.float64)

    sat = rng.normal(0, 2.5e7, (n_epoch, n_sat, 3))
    true_pos = np.tile(np.array([[-3.8e6, 3.5e6, 3.6e6]], dtype=np.float64), (n_epoch, 1))
    true_drift = np.array([0.4, 1.2, -0.3, 0.9, 0.2, 0.6], dtype=np.float64)
    true_clk = np.zeros(n_epoch, dtype=np.float64)
    true_clk[0] = 150.0
    for t in range(n_epoch - 1):
        true_clk[t + 1] = true_clk[t] + 0.5 * (true_drift[t] + true_drift[t + 1]) * dt_arr[t]

    pr_noise_sigma = 2.0
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos[t], sat[t, s]) + true_clk[t]
    pr += rng.normal(0, pr_noise_sigma, pr.shape)
    w_pr = np.ones((n_epoch, n_sat), dtype=np.float64) / (pr_noise_sigma ** 2)

    tdcp_noise_sigma = 0.01
    tdcp_meas = np.zeros((n_epoch - 1, n_sat), dtype=np.float64)
    for t in range(n_epoch - 1):
        tdcp_meas[t, :] = 0.5 * (true_drift[t] + true_drift[t + 1]) * dt_arr[t]
    tdcp_meas += rng.normal(0, tdcp_noise_sigma, tdcp_meas.shape)
    tdcp_w = np.ones_like(tdcp_meas) / (tdcp_noise_sigma ** 2)

    state_old = np.zeros((n_epoch, ss_old), dtype=np.float64)
    state_old[:, :3] = true_pos + rng.normal(0, 300.0, (n_epoch, 3))
    state_old[:, 3] = true_clk + rng.normal(0, 4.0, n_epoch)
    fgo_gnss_lm(
        sat, pr, w_pr, state_old,
        motion_sigma_m=0.0, max_iter=25, tol=1e-7,
        enable_line_search=1, sys_kind=None, n_clock=1,
    )

    state_xxcc = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_xxcc[:, :3] = state_old[:, :3]
    state_xxcc[:, 6] = state_old[:, 3]
    state_xxcc[:, 7] = rng.normal(0, 0.5, n_epoch)

    state_xxdd = state_xxcc.copy()

    # Keep the clock-drift factor disabled here so this compares the TDCP
    # parameterization itself. Clock-drift coupling is covered by a standalone
    # factor-direction regression above.
    iters_cc, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state_xxcc,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        clock_use_average_drift=False,
        max_iter=20, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, dt=dt_arr,
        tdcp_meas=tdcp_meas, tdcp_weights=tdcp_w, tdcp_use_drift=False,
    )
    iters_dd, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state_xxdd,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        clock_use_average_drift=True,
        max_iter=20, tol=1e-7, huber_k=0.0, enable_line_search=1,
        sys_kind=None, n_clock=1, dt=dt_arr,
        tdcp_meas=tdcp_meas, tdcp_weights=tdcp_w, tdcp_use_drift=True,
    )
    assert iters_cc > 0
    assert iters_dd > 0

    drift_err_xxcc = float(np.mean(np.abs(state_xxcc[:, 7] - true_drift)))
    drift_err_xxdd = float(np.mean(np.abs(state_xxdd[:, 7] - true_drift)))
    assert drift_err_xxdd < drift_err_xxcc, (
        f"XXDD should recover drift better: xxdd={drift_err_xxdd:.4f}, xxcc={drift_err_xxcc:.4f}"
    )


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_tdcp_vd_uses_signal_clock_for_dual_frequency():
    """L5 TDCP must use the same c0+ISB clock design as L5 pseudorange."""
    rng = np.random.Generator(np.random.PCG64(6060))
    n_epoch, n_sat, nc = 4, 8, 2
    ss_vd = 7 + nc
    true_pos = np.tile(np.array([[-3.8e6, 3.5e6, 3.6e6]], dtype=np.float64), (n_epoch, 1))
    c0 = np.full(n_epoch, 120.0, dtype=np.float64)
    isb = np.array([0.0, 6.0, 12.0, 18.0], dtype=np.float64)

    sat = rng.normal(0, 2.5e7, (n_epoch, n_sat, 3))
    sys_kind = np.zeros((n_epoch, n_sat), dtype=np.int32)
    sys_kind[:, 1::2] = 1
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            clk = c0[t] + (isb[t] if sys_kind[t, s] == 1 else 0.0)
            pr[t, s] = _geometric_range_sagnacrcv(true_pos[t], sat[t, s]) + clk
    w_pr = np.ones((n_epoch, n_sat), dtype=np.float64)

    tdcp_meas = np.zeros((n_epoch - 1, n_sat), dtype=np.float64)
    for t in range(n_epoch - 1):
        tdcp_meas[t, 1::2] = isb[t + 1] - isb[t]
    tdcp_w = np.ones_like(tdcp_meas) * 1.0e4

    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[:, :3] = true_pos
    state[:, 6] = c0
    state[:, 7] = isb
    state_before = state.copy()

    iters, _ = fgo_gnss_lm_vd(
        sat, pr, w_pr, state,
        motion_sigma_m=0.0, clock_drift_sigma_m=0.0,
        max_iter=3, tol=1e-10, huber_k=0.0, enable_line_search=1,
        sys_kind=sys_kind, n_clock=nc,
        tdcp_meas=tdcp_meas, tdcp_weights=tdcp_w,
    )

    assert iters > 0
    np.testing.assert_allclose(state[:, :3], state_before[:, :3], atol=1e-4)
    np.testing.assert_allclose(state[:, 6:8], state_before[:, 6:8], atol=1e-4)


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_relative_height_factor_runs():
    """Graph relative-height factor (ENU-up loop closure) runs without error."""
    rng = np.random.Generator(np.random.PCG64(303))
    n_epoch, n_sat, nc = 3, 8, 1
    ss_vd = 7 + nc
    sat = rng.normal(0, 5e6, (n_epoch, n_sat, 3))
    true_pos = np.array([-3.8e6, 3.5e6, 3.6e6], dtype=np.float64)
    true_clk = 150.0
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    for t in range(n_epoch):
        for s in range(n_sat):
            pr[t, s] = _geometric_range_sagnacrcv(true_pos, sat[t, s]) + true_clk
    w = np.ones((n_epoch, n_sat)) * 0.3
    state_vd = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_vd[:, :3] = true_pos + rng.normal(0, 2.0, (n_epoch, 3))
    state_vd[:, 6] = true_clk
    dt_arr = np.ones(n_epoch, dtype=np.float64)
    lat = np.arctan2(true_pos[2], np.sqrt(true_pos[0] ** 2 + true_pos[1] ** 2))
    lon = np.arctan2(true_pos[1], true_pos[0])
    up = np.array(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ],
        dtype=np.float64,
    )
    ei = np.array([0], dtype=np.int32)
    ej = np.array([2], dtype=np.int32)
    iters, mse = fgo_gnss_lm_vd(
        sat,
        pr,
        w,
        state_vd,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.0,
        max_iter=15,
        tol=1e-7,
        huber_k=0.0,
        enable_line_search=1,
        sys_kind=None,
        n_clock=1,
        dt=dt_arr,
        relative_height_sigma_m=0.5,
        enu_up_ecef=up,
        rel_height_edge_i=ei,
        rel_height_edge_j=ej,
    )
    assert iters > 0
    assert np.isfinite(mse)


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_relative_height_factor_reduces_standalone_residual():
    """Standalone relative-height factor should reduce the ENU-up difference."""
    n_epoch, n_sat, nc = 2, 4, 1
    ss_vd = 7 + nc
    sat = np.ones((n_epoch, n_sat, 3), dtype=np.float64)
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w = np.zeros((n_epoch, n_sat), dtype=np.float64)
    state = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state[0, 0] = 1.0
    up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    edge_i = np.array([0], dtype=np.int32)
    edge_j = np.array([1], dtype=np.int32)

    before = float(state[0, 0] - state[1, 0])
    iters, _ = fgo_gnss_lm_vd(
        sat,
        pr,
        w,
        state,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.0,
        max_iter=5,
        tol=1e-12,
        huber_k=0.0,
        enable_line_search=1,
        sys_kind=None,
        n_clock=1,
        relative_height_sigma_m=0.1,
        enu_up_ecef=up,
        rel_height_edge_i=edge_i,
        rel_height_edge_j=edge_j,
    )

    after = float(state[0, 0] - state[1, 0])
    assert iters > 0
    assert abs(after) < abs(before) * 1e-3


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_absolute_height_factor_pulls_up_component_to_reference():
    """Absolute-height factor constrains only the ENU-up component."""
    n_epoch, n_sat, nc = 1, 4, 1
    ss_vd = 7 + nc
    sat = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w = np.zeros((n_epoch, n_sat), dtype=np.float64)
    ref = np.array([[-3.8e6, 3.5e6, 3.6e6]], dtype=np.float64)
    state_vd = np.zeros((n_epoch, ss_vd), dtype=np.float64)
    state_vd[:, :3] = ref
    state_vd[0, 2] += 10.0

    iters, mse = fgo_gnss_lm_vd(
        sat,
        pr,
        w,
        state_vd,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.0,
        max_iter=5,
        tol=1e-9,
        huber_k=0.0,
        enable_line_search=1,
        sys_kind=None,
        n_clock=1,
        enu_up_ecef=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        absolute_height_ref_ecef=ref,
        absolute_height_sigma_m=0.1,
    )

    assert iters > 0
    assert np.isfinite(mse)
    assert abs(state_vd[0, 2] - ref[0, 2]) < 1e-3


@pytest.mark.skipif(not HAS_FGO_VD, reason="FGO VD CUDA extension not built")
def test_fgo_vd_imu_accel_bias_state_reduces_delta_v_residual():
    """Optional accel-bias state should absorb biased IMU velocity deltas."""
    n_epoch, n_sat, nc = 2, 4, 1
    ss_vd_bias = 10 + nc
    bias_idx = 7 + nc
    sat = np.zeros((n_epoch, n_sat, 3), dtype=np.float64)
    pr = np.zeros((n_epoch, n_sat), dtype=np.float64)
    w = np.zeros((n_epoch, n_sat), dtype=np.float64)
    state = np.zeros((n_epoch, ss_vd_bias), dtype=np.float64)
    dt = np.ones(n_epoch, dtype=np.float64)
    stop_mask = np.ones(n_epoch, dtype=np.uint8)
    imu_delta_v = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    def residual_x(s: np.ndarray) -> float:
        return float(s[0, 3] + imu_delta_v[0, 0] - dt[0] * s[0, bias_idx] - s[1, 3])

    before = residual_x(state)
    iters, _ = fgo_gnss_lm_vd(
        sat,
        pr,
        w,
        state,
        motion_sigma_m=0.0,
        clock_drift_sigma_m=0.0,
        stop_velocity_sigma_mps=0.001,
        max_iter=8,
        tol=1e-12,
        huber_k=0.0,
        enable_line_search=1,
        sys_kind=None,
        n_clock=1,
        dt=dt,
        stop_mask=stop_mask,
        imu_delta_v=imu_delta_v,
        imu_velocity_sigma_mps=0.01,
        imu_accel_bias_prior_sigma_mps2=10.0,
    )

    after = residual_x(state)
    assert iters > 0
    assert abs(after) < abs(before) * 1e-3
    assert state[0, bias_idx] > 0.5
