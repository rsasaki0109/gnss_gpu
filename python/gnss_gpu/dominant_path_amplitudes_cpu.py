"""Two complex amplitudes subject to |echo| <= |first|.

When unconstrained least squares is infeasible, the optimum is on |a1|=|a0|.
Eliminate the common complex scale, then solve the stationary phases of a
ratio of two first-order trigonometric polynomials. No phase grid is used.
"""

import numpy as np


def dominant_amplitudes(gram, rhs):
    g = np.asarray(gram, dtype=complex)
    b = np.asarray(rhs, dtype=complex)
    if (
        g.shape != (2, 2)
        or b.shape != (2,)
        or not np.all(np.isfinite(g))
        or not np.all(np.isfinite(b))
    ):
        raise ValueError("finite 2x2 Gram matrix and two correlations required")
    if (
        not np.allclose(g, g.conj().T, rtol=1e-12, atol=1e-14)
        or np.linalg.eigvalsh(g)[0] <= 0
    ):
        raise ValueError("positive definite Hermitian Gram matrix required")
    free = np.linalg.solve(g, b)
    if abs(free[1]) <= abs(free[0]):
        return free
    # N(theta)=A+nc*cos(theta)+ns*sin(theta), likewise D(theta).
    cross = b[0] * b[1].conjugate()
    a = float(np.vdot(b, b).real)
    nc = 2 * cross.real
    ns = -2 * cross.imag
    c = float(np.trace(g).real)
    dc = 2 * g[0, 1].real
    ds = -2 * g[0, 1].imag
    p = a * dc - c * nc
    q = c * ns - a * ds
    r = ns * dc - nc * ds
    magnitude = np.hypot(p, q)
    phases = [0.0, np.pi]
    if magnitude > 0:
        angle = np.arcsin(np.clip(-r / magnitude, -1, 1))
        shift = np.arctan2(q, p)
        phases += [angle - shift, np.pi - angle - shift]
    candidates = []
    for phase in phases:
        u = np.array([1.0, np.exp(1j * phase)])
        denom = float(np.vdot(u, g @ u).real)
        candidates.append(u * (np.vdot(u, b) / denom))
    return min(
        candidates,
        key=lambda amp: float(np.vdot(amp, g @ amp).real - 2 * np.vdot(amp, b).real),
    )
