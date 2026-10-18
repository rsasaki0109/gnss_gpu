"""WP61: single-bounce reflection paths on PLATEAU walls (CPU, numba).

Every vertical wall triangle near the receiver is a candidate specular
reflector (image method). The receiver's mirror image across the wall plane,
followed along the satellite direction, meets the plane at the reflection
point. The path counts when that point lies on the triangle and neither leg
(wall to satellite, wall to receiver) is blocked by the mesh. Its extra
length over the direct path is 2 s (u . n): s is the receiver's distance to
the wall plane, u the satellite direction and n the wall normal.

`ReflectionModel(tracer)` shares the triangles and the occlusion grid of a
`plateau_los.GridTracer`."""
import math

import numpy as np
from numba import njit, prange

from plateau_los import _enu_rot, _occluded


@njit(cache=True)
def _hit_t(o, d, v0, e1, e2):
    px = d[1] * e2[2] - d[2] * e2[1]
    py = d[2] * e2[0] - d[0] * e2[2]
    pz = d[0] * e2[1] - d[1] * e2[0]
    det = e1[0] * px + e1[1] * py + e1[2] * pz
    if abs(det) < 1e-12:
        return -1.0
    inv = 1.0 / det
    tx = o[0] - v0[0]; ty = o[1] - v0[1]; tz = o[2] - v0[2]
    u = (tx * px + ty * py + tz * pz) * inv
    if u < 0.0 or u > 1.0:
        return -1.0
    qx = ty * e1[2] - tz * e1[1]
    qy = tz * e1[0] - tx * e1[2]
    qz = tx * e1[1] - ty * e1[0]
    v = (d[0] * qx + d[1] * qy + d[2] * qz) * inv
    if v < 0.0 or u + v > 1.0:
        return -1.0
    return (e2[0] * qx + e2[1] * qy + e2[2] * qz) * inv


@njit(cache=True)
def _seg_occluded(o, d, tmin, tmax, V0, E1, E2, cstart, ctris, czmax, xmin, ymin, cell, nx, ny):
    """True when a triangle crosses the segment o + t d, tmin < t < tmax."""
    inf = 1e30
    t0 = 0.0; t1 = tmax
    bx0 = xmin; bx1 = xmin + nx * cell; by0 = ymin; by1 = ymin + ny * cell
    if d[0] != 0.0:
        ta = (bx0 - o[0]) / d[0]; tb = (bx1 - o[0]) / d[0]
        t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb))
    elif o[0] < bx0 or o[0] >= bx1:
        return False
    if d[1] != 0.0:
        ta = (by0 - o[1]) / d[1]; tb = (by1 - o[1]) / d[1]
        t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb))
    elif o[1] < by0 or o[1] >= by1:
        return False
    if t0 > t1:
        return False
    px = o[0] + d[0] * t0; py = o[1] + d[1] * t0
    cx = min(max(int(math.floor((px - xmin) / cell)), 0), nx - 1)  # noqa: RUF046 (numba)
    cy = min(max(int(math.floor((py - ymin) / cell)), 0), ny - 1)  # noqa: RUF046 (numba)
    if d[0] > 0:
        sx = 1; tmx = (xmin + (cx + 1) * cell - o[0]) / d[0]; tdx = cell / d[0]
    elif d[0] < 0:
        sx = -1; tmx = (xmin + cx * cell - o[0]) / d[0]; tdx = -cell / d[0]
    else:
        sx = 0; tmx = inf; tdx = inf
    if d[1] > 0:
        sy = 1; tmy = (ymin + (cy + 1) * cell - o[1]) / d[1]; tdy = cell / d[1]
    elif d[1] < 0:
        sy = -1; tmy = (ymin + cy * cell - o[1]) / d[1]; tdy = -cell / d[1]
    else:
        sy = 0; tmy = inf; tdy = inf
    t_in = t0
    while True:
        c = cy * nx + cx
        zlo = min(o[2] + d[2] * t_in, o[2] + d[2] * min(tmx, tmy, t1))
        if zlo <= czmax[c]:
            for k in range(cstart[c], cstart[c + 1]):
                j = ctris[k]
                t = _hit_t(o, d, V0[j], E1[j], E2[j])
                if tmin < t < tmax:
                    return True
        if tmx < tmy:
            t_in = tmx; tmx += tdx; cx += sx
        else:
            t_in = tmy; tmy += tdy; cy += sy
        if t_in > t1 or cx < 0 or cy < 0 or cx >= nx or cy >= ny:
            return False


@njit(cache=True)
def _reflect_one(o, U, valid, radius, W, WN, wstart, wtris, wxmin, wymin, wcell, wnx, wny,
                 V0, E1, E2, cstart, ctris, czmax, xmin, ymin, cell, nx, ny, zall,
                 out_min, out_n, out_wall):
    """Single-bounce paths from every satellite direction U[s] to receiver o."""
    S = U.shape[0]
    for s in range(S):
        out_min[s] = -1.0; out_n[s] = 0; out_wall[s] = -1
    r = int(math.ceil(radius / wcell))
    cx0 = int(math.floor((o[0] - wxmin) / wcell)); cy0 = int(math.floor((o[1] - wymin) / wcell))
    P = np.empty(3); d = np.empty(3); q = np.empty(3)
    for gy in range(max(cy0 - r, 0), min(cy0 + r + 1, wny)):
        for gx in range(max(cx0 - r, 0), min(cx0 + r + 1, wnx)):
            c = gy * wnx + gx
            for k in range(wstart[c], wstart[c + 1]):
                j = wtris[k]          # index into the full triangle arrays
                w = W[j]              # wall slot
                n0 = WN[w, 0]; n1 = WN[w, 1]; n2 = WN[w, 2]
                sd = (o[0] - V0[j, 0]) * n0 + (o[1] - V0[j, 1]) * n1 + (o[2] - V0[j, 2]) * n2
                if abs(sd) < 0.3 or abs(sd) > radius:
                    continue
                for s in range(S):
                    if not valid[s]:
                        continue
                    un = U[s, 0] * n0 + U[s, 1] * n1 + U[s, 2] * n2
                    if un * sd <= 1e-6:       # satellite behind the wall
                        continue
                    t = sd / un
                    # reflection point: image receiver o - 2 sd n, then + t U
                    P[0] = o[0] - 2.0 * sd * n0 + t * U[s, 0]
                    P[1] = o[1] - 2.0 * sd * n1 + t * U[s, 1]
                    P[2] = o[2] - 2.0 * sd * n2 + t * U[s, 2]
                    # on the triangle? (barycentric, in its plane)
                    q[0] = P[0] - V0[j, 0]; q[1] = P[1] - V0[j, 1]; q[2] = P[2] - V0[j, 2]
                    d00 = E1[j, 0] * E1[j, 0] + E1[j, 1] * E1[j, 1] + E1[j, 2] * E1[j, 2]
                    d01 = E1[j, 0] * E2[j, 0] + E1[j, 1] * E2[j, 1] + E1[j, 2] * E2[j, 2]
                    d11 = E2[j, 0] * E2[j, 0] + E2[j, 1] * E2[j, 1] + E2[j, 2] * E2[j, 2]
                    d20 = q[0] * E1[j, 0] + q[1] * E1[j, 1] + q[2] * E1[j, 2]
                    d21 = q[0] * E2[j, 0] + q[1] * E2[j, 1] + q[2] * E2[j, 2]
                    den = d00 * d11 - d01 * d01
                    if den <= 1e-12:
                        continue
                    bv = (d11 * d20 - d01 * d21) / den
                    bw = (d00 * d21 - d01 * d20) / den
                    if bv < 0.0 or bw < 0.0 or bv + bw > 1.0:
                        continue
                    extra = 2.0 * sd * un
                    if out_n[s] > 0 and extra >= out_min[s]:
                        continue
                    # leg 1: wall to satellite, starting just off the wall
                    sg = 1.0 if sd > 0 else -1.0
                    q[0] = P[0] + 0.05 * sg * n0; q[1] = P[1] + 0.05 * sg * n1; q[2] = P[2] + 0.05 * sg * n2
                    d[0] = U[s, 0]; d[1] = U[s, 1]; d[2] = U[s, 2]
                    if _occluded(q, d, V0, E1, E2, cstart, ctris, czmax, xmin, ymin, cell, nx, ny,
                                 zall, 0.01):
                        continue
                    # leg 2: receiver to the wall
                    d[0] = P[0] - o[0]; d[1] = P[1] - o[1]; d[2] = P[2] - o[2]
                    L = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                    d[0] /= L; d[1] /= L; d[2] /= L
                    if _seg_occluded(o, d, 3.0, L - 0.2, V0, E1, E2, cstart, ctris, czmax,
                                     xmin, ymin, cell, nx, ny):
                        continue
                    out_n[s] += 1
                    out_min[s] = extra
                    out_wall[s] = j


@njit(parallel=True, cache=True)
def _reflect_batch(O, U, valid, radius, W, WN, wstart, wtris, wxmin, wymin, wcell, wnx, wny,
                   V0, E1, E2, cstart, ctris, czmax, xmin, ymin, cell, nx, ny, zall):
    B, S = valid.shape
    out_min = np.full((B, S), -1.0)
    out_n = np.zeros((B, S), dtype=np.int64)
    out_wall = np.full((B, S), -1, dtype=np.int64)
    for b in prange(B):
        _reflect_one(O[b], U[b], valid[b], radius, W, WN, wstart, wtris, wxmin, wymin, wcell,
                     wnx, wny, V0, E1, E2, cstart, ctris, czmax, xmin, ymin, cell, nx, ny, zall,
                     out_min[b], out_n[b], out_wall[b])
    return out_min, out_n, out_wall


class ReflectionModel:
    """Shortest valid single-bounce path per (receiver, satellite)."""

    def __init__(self, tracer, wcell=10.0, max_nz=0.1, min_area=0.5):
        self.t = tracer
        n = np.cross(tracer.E1, tracer.E2)
        a = np.linalg.norm(n, axis=1)
        ok = a > 2.0 * min_area
        nn = np.where(ok[:, None], n / np.where(a > 0, a, 1.0)[:, None], 0.0)
        wall = ok & (np.abs(nn[:, 2]) < max_nz)
        idx = np.nonzero(wall)[0]
        self.n_walls = len(idx)
        self.W = np.full(len(n), -1, dtype=np.int64)
        self.W[idx] = np.arange(len(idx))
        self.WN = np.ascontiguousarray(nn[idx])
        cen = tracer.V0[idx] + (tracer.E1[idx] + tracer.E2[idx]) / 3.0
        self.wcell = float(wcell)
        self.wxmin = float(cen[:, 0].min()) - wcell; self.wymin = float(cen[:, 1].min()) - wcell
        self.wnx = math.ceil((cen[:, 0].max() - self.wxmin) / wcell) + 2
        self.wny = math.ceil((cen[:, 1].max() - self.wymin) / wcell) + 2
        cx = np.floor((cen[:, 0] - self.wxmin) / wcell).astype(np.int64)
        cy = np.floor((cen[:, 1] - self.wymin) / wcell).astype(np.int64)
        c = cy * self.wnx + cx
        order = np.argsort(c, kind='stable')
        cnt = np.bincount(c, minlength=self.wnx * self.wny)
        self.wstart = np.zeros(self.wnx * self.wny + 1, dtype=np.int64)
        self.wstart[1:] = np.cumsum(cnt)
        self.wtris = idx[order].astype(np.int64)

    def local(self, x_ecef):
        return (np.asarray(x_ecef, dtype=np.float64) - self.t.o) @ self.t.R.T

    def paths(self, rx_ecef, u_ecef, radius=60.0):
        """rx_ecef (B, 3) receivers, u_ecef (B, S, 3) unit satellite
        directions (NaN = unused). Returns the extra length of the shortest
        valid single-bounce path (B, S) (NaN = none), how many times a
        shorter valid path was found (0 = none) and the wall triangle of the
        shortest one."""
        rx = np.asarray(rx_ecef, dtype=np.float64).reshape(-1, 3)
        u = np.asarray(u_ecef, dtype=np.float64).reshape(len(rx), -1, 3)
        U = np.where(np.isfinite(u), u, 0.0) @ self.t.R.T
        valid = np.all(np.isfinite(u), axis=2) & (U[..., 2] > 0.0)
        U = np.ascontiguousarray(np.where(valid[..., None], U, 0.0))
        O = np.ascontiguousarray(self.local(rx))
        t = self.t
        m, k, w = _reflect_batch(O, U, valid, float(radius), self.W, self.WN, self.wstart,
                                 self.wtris, self.wxmin, self.wymin, self.wcell, self.wnx,
                                 self.wny, t.V0, t.E1, t.E2, t.cstart, t.ctris, t.czmax, t.xmin,
                                 t.ymin, t.cell, t.nx, t.ny, t.zall)
        return np.where(m >= 0, m, np.nan), k, w



if __name__ == '__main__':
    # self-test: a 20 m wide street between two 30 m walls (x = -10, x = +10)
    from plateau_los import GridTracer
    o = np.array([-3.81e6, 3.57e6, 3.65e6]); R = _enu_rot(o)
    tri = []
    for x in (-10.0, 10.0):
        tri += [[[x, -100, 0], [x, 100, 0], [x, 100, 30]], [[x, -100, 0], [x, 100, 30], [x, -100, 30]]]
    g = GridTracer(np.array(tri) @ R + o, cell=4.0)
    m = ReflectionModel(g)
    rx = o + np.array([0, 0, 1.5]) @ R
    def u_at(az, el):
        a, e = math.radians(az), math.radians(el)
        return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)]) @ R
    U = np.array([[u_at(90, 60), u_at(90, 40), u_at(0, 30), u_at(270, 60)]])
    ext, n, w = m.paths(rx[None], U)
    print('walls', m.n_walls, 'LOS', g.check_los(rx, rx + 2e7 * U[0]))
    print('extra', np.round(ext[0], 3), 'expected [10. nan nan 10.]')
