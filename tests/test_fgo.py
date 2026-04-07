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
