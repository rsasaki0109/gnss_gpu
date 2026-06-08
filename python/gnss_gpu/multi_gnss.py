"""Multi-GNSS WLS positioning with Inter-System Bias (ISB) estimation."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from gnss_gpu.range_model import geometric_ranges_sagnac, rotate_satellites_sagnac

SYSTEM_GPS = 0
SYSTEM_GLONASS = 1
SYSTEM_GALILEO = 2
SYSTEM_BEIDOU = 3
SYSTEM_QZSS = 4

_SYSTEM_NAMES = {
    SYSTEM_GPS: "GPS",
    SYSTEM_GLONASS: "GLONASS",
    SYSTEM_GALILEO: "Galileo",
    SYSTEM_BEIDOU: "BeiDou",
    SYSTEM_QZSS: "QZSS",
}
_SUPPORTED_SYSTEMS = frozenset(_SYSTEM_NAMES)

try:
    from gnss_gpu._gnss_gpu_multi_gnss import wls_multi_gnss as _wls_multi_gnss
    from gnss_gpu._gnss_gpu_multi_gnss import wls_multi_gnss_batch as _wls_multi_gnss_batch
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


def _validate_solver_config(systems, max_iter, tol):
    if systems is None:
        systems = [SYSTEM_GPS]
    elif isinstance(systems, Integral) and not isinstance(systems, (bool, np.bool_)):
        systems = [systems]
    else:
        systems = list(systems)

    if not systems:
        raise RuntimeError("MultiGNSSSolver: systems must contain at least one system")

    validated_systems = []
    for system in systems:
        if not isinstance(system, Integral) or isinstance(system, (bool, np.bool_)):
            raise RuntimeError("MultiGNSSSolver: systems must be integer system IDs")
        system = int(system)
        if system not in _SUPPORTED_SYSTEMS:
            raise RuntimeError("MultiGNSSSolver: systems must be supported GNSS system IDs")
        validated_systems.append(system)

    if len(set(validated_systems)) != len(validated_systems):
        raise RuntimeError("MultiGNSSSolver: systems must be unique")

    if not isinstance(max_iter, Integral) or isinstance(max_iter, (bool, np.bool_)):
        raise RuntimeError("MultiGNSSSolver: max_iter must be an integer")
    if int(max_iter) < 1:
        raise RuntimeError("MultiGNSSSolver: max_iter must be >= 1")
    try:
        tol = float(tol)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("MultiGNSSSolver: tol must be positive and finite") from exc
    if not np.isfinite(tol) or tol <= 0.0:
        raise RuntimeError("MultiGNSSSolver: tol must be positive and finite")

    return sorted(validated_systems), int(max_iter), tol


def _system_id_array(name, system_ids, shape, enabled_systems):
    values = np.asarray(system_ids)
    if values.shape != shape:
        expected = ", ".join(str(dim) for dim in shape)
        raise RuntimeError(f"{name}: system_ids must have shape ({expected})")

    try:
        values_float = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name}: system_ids must be integer-valued") from exc

    if not np.isfinite(values_float).all() or not np.all(values_float == np.floor(values_float)):
        raise RuntimeError(f"{name}: system_ids must be integer-valued")

    invalid_mask = ~np.isin(values_float, np.asarray(enabled_systems, dtype=np.float64))
    if invalid_mask.any():
        raise RuntimeError(f"{name}: system_ids must be one of the enabled systems")

    return np.ascontiguousarray(values_float.astype(np.int32), dtype=np.int32)


def _sat_matrix(name, sat_ecef, n_sat):
    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
    if sat_ecef.ndim == 1 and sat_ecef.size == n_sat * 3:
        sat_ecef = sat_ecef.reshape(n_sat, 3)
    elif not (sat_ecef.ndim == 2 and sat_ecef.shape == (n_sat, 3)):
        raise RuntimeError(
            f"{name}: sat_ecef must have shape (n_sat, 3) or flat length n_sat*3"
        )
    return sat_ecef


def _validate_single_inputs(name, systems, sat_ecef, pseudoranges, system_ids, weights):
    pseudoranges = np.asarray(pseudoranges, dtype=np.float64)
    if pseudoranges.ndim != 1:
        raise RuntimeError(f"{name}: pseudoranges must have shape (n_sat,)")
    n_sat = pseudoranges.size

    sat_ecef = _sat_matrix(name, sat_ecef, n_sat)
    system_ids = _system_id_array(name, system_ids, (n_sat,), systems)

    if weights is None:
        weights = np.ones(n_sat, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size != n_sat:
            raise RuntimeError(f"{name}: weights must have shape (n_sat,)")

    if not np.isfinite(sat_ecef).all():
        raise RuntimeError(f"{name}: satellite positions must be finite")
    if not np.isfinite(pseudoranges).all():
        raise RuntimeError(f"{name}: pseudoranges must be finite")
    if not (np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError(f"{name}: weights must be finite and nonnegative")

    return (
        np.ascontiguousarray(sat_ecef, dtype=np.float64),
        np.ascontiguousarray(pseudoranges, dtype=np.float64),
        system_ids,
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _validate_batch_inputs(name, systems, sat_ecef, pseudoranges, system_ids, weights):
    sat_ecef = np.asarray(sat_ecef, dtype=np.float64)
    if sat_ecef.ndim != 3 or sat_ecef.shape[2] != 3:
        raise RuntimeError(f"{name}: sat_ecef must have shape (n_epoch, n_sat, 3)")
    n_epoch, n_sat, _ = sat_ecef.shape
    if n_epoch < 1:
        raise RuntimeError(f"{name}: n_epoch must be >= 1")

    pseudoranges = np.asarray(pseudoranges, dtype=np.float64)
    if pseudoranges.ndim != 2 or pseudoranges.shape != (n_epoch, n_sat):
        raise RuntimeError(f"{name}: pseudoranges must have shape (n_epoch, n_sat)")

    system_ids = _system_id_array(name, system_ids, (n_epoch, n_sat), systems)

    if weights is None:
        weights = np.ones((n_epoch, n_sat), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 2 or weights.shape != (n_epoch, n_sat):
            raise RuntimeError(f"{name}: weights must have shape (n_epoch, n_sat)")

    if not np.isfinite(sat_ecef).all():
        raise RuntimeError(f"{name}: satellite positions must be finite")
    if not np.isfinite(pseudoranges).all():
        raise RuntimeError(f"{name}: pseudoranges must be finite")
    if not (np.isfinite(weights).all() and np.all(weights >= 0.0)):
        raise RuntimeError(f"{name}: weights must be finite and nonnegative")

    return (
        np.ascontiguousarray(sat_ecef, dtype=np.float64),
        np.ascontiguousarray(pseudoranges, dtype=np.float64),
        system_ids,
        np.ascontiguousarray(weights, dtype=np.float64),
    )


def _remap_system_ids(system_ids, sys_to_idx):
    return np.array([sys_to_idx[int(s)] for s in system_ids.ravel()], dtype=np.int32).reshape(
        system_ids.shape
    )


class MultiGNSSSolver:
    """Multi-GNSS Weighted Least Squares solver with ISB estimation.

    Estimates position [x, y, z] plus per-constellation clock biases,
    enabling multi-GNSS positioning with GPS, GLONASS, Galileo, BeiDou, and QZSS.
    """

    def __init__(self, systems: list[int] | None = None,
                 max_iter: int = 10, tol: float = 1e-4):
        """Initialize multi-GNSS solver.

        Args:
            systems: List of enabled systems, e.g., [SYSTEM_GPS, SYSTEM_GALILEO].
                     Defaults to [SYSTEM_GPS].
            max_iter: Maximum Gauss-Newton iterations.
            tol: Convergence tolerance [m].
        """
        self.systems, self.max_iter, self.tol = _validate_solver_config(
            systems, max_iter, tol
        )
        # Build mapping from system enum to contiguous index
        self._sys_to_idx = {s: i for i, s in enumerate(self.systems)}
        self.n_systems = len(self.systems)

    def solve(self, sat_ecef: np.ndarray, pseudoranges: np.ndarray,
              system_ids: np.ndarray, weights: np.ndarray | None = None
              ) -> tuple[np.ndarray, dict[int, float], int]:
        """Single-epoch multi-GNSS WLS positioning.

        Args:
            sat_ecef: [n_sat, 3] satellite ECEF positions [m].
            pseudoranges: [n_sat] observed pseudoranges [m].
            system_ids: [n_sat] system identifier per satellite (SYSTEM_GPS, etc.).
            weights: [n_sat] observation weights (1/sigma^2). Defaults to uniform.

        Returns:
            position: [3] ECEF position (x, y, z) [m].
            clock_biases: dict {system_id: bias_m} for each enabled system.
            n_iter: Number of iterations used.
        """
        sat_ecef, pseudoranges, system_ids, weights = _validate_single_inputs(
            "MultiGNSSSolver.solve",
            self.systems,
            sat_ecef,
            pseudoranges,
            system_ids,
            weights,
        )
        n_sat = len(pseudoranges)

        n_state = 3 + self.n_systems
        if n_sat < n_state:
            return np.zeros(3, dtype=np.float64), {
                system: 0.0 for system in self.systems
            }, -1

        # Remap system_ids to contiguous indices
        mapped_ids = _remap_system_ids(system_ids, self._sys_to_idx)

        if _HAS_GPU:
            result, n_iter = _wls_multi_gnss(
                sat_ecef.ravel(), pseudoranges, weights, mapped_ids,
                self.n_systems, self.max_iter, self.tol)
        else:
            result, n_iter = self._solve_cpu(
                sat_ecef.ravel(), pseudoranges, weights, mapped_ids)

        position = result[:3]
        clock_biases = {self.systems[k]: float(result[3 + k])
                        for k in range(self.n_systems)}
        return position, clock_biases, int(n_iter)

    def solve_batch(self, sat_ecef: np.ndarray, pseudoranges: np.ndarray,
                    system_ids: np.ndarray, weights: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch multi-GNSS WLS positioning.

        Args:
            sat_ecef: [n_epoch, n_sat, 3] satellite ECEF positions [m].
            pseudoranges: [n_epoch, n_sat] pseudoranges [m].
            system_ids: [n_epoch, n_sat] system IDs.
            weights: [n_epoch, n_sat] weights. Defaults to uniform.

        Returns:
            positions: [n_epoch, 3] ECEF positions.
            clock_biases: [n_epoch, n_systems] per-system clock biases.
            n_iters: [n_epoch] iterations used.
        """
        sat_ecef, pseudoranges, system_ids, weights = _validate_batch_inputs(
            "MultiGNSSSolver.solve_batch",
            self.systems,
            sat_ecef,
            pseudoranges,
            system_ids,
            weights,
        )
        n_epoch, n_sat = pseudoranges.shape

        n_state = 3 + self.n_systems
        if n_sat < n_state:
            return (
                np.zeros((n_epoch, 3), dtype=np.float64),
                np.zeros((n_epoch, self.n_systems), dtype=np.float64),
                np.full(n_epoch, -1, dtype=np.int32),
            )

        # Remap system_ids to contiguous indices
        mapped_ids = _remap_system_ids(system_ids, self._sys_to_idx)

        if _HAS_GPU:
            results, n_iters = _wls_multi_gnss_batch(
                sat_ecef, pseudoranges, weights,
                np.ascontiguousarray(mapped_ids),
                self.n_systems, self.max_iter, self.tol)
        else:
            # CPU fallback: iterate over epochs
            results = np.zeros((n_epoch, n_state))
            n_iters = np.zeros(n_epoch, dtype=np.int32)
            for i in range(n_epoch):
                res, it = self._solve_cpu(
                    sat_ecef[i].ravel(), pseudoranges[i], weights[i],
                    mapped_ids[i])
                results[i] = res
                n_iters[i] = it

        positions = results[:, :3]
        clock_biases_arr = results[:, 3:]
        return positions, clock_biases_arr, n_iters

    def _solve_cpu(self, sat_ecef_flat, pseudoranges, weights, mapped_ids):
        """Pure numpy CPU fallback for single epoch."""
        WGS84_A = 6378137.0  # WGS84 semi-major axis [m]
        n_sat = len(pseudoranges)
        n_state = 3 + self.n_systems

        if n_sat < n_state:
            return np.zeros(n_state), -1

        # Initial guess
        sat = sat_ecef_flat.reshape(-1, 3)
        centroid = sat.mean(axis=0)
        cn = np.linalg.norm(centroid)
        if cn < 1e-6:
            return np.zeros(n_state), -1
        pos = centroid * (WGS84_A / cn)

        cb = np.zeros(self.n_systems)
        for k in range(self.n_systems):
            mask = mapped_ids == k
            if mask.any():
                diffs = geometric_ranges_sagnac(pos, sat[mask])
                cb[k] = np.mean(pseudoranges[mask] - diffs)

        state = np.concatenate([pos, cb])

        for it in range(self.max_iter):
            pos = state[:3]
            cb = state[3:]
            sat_rot = rotate_satellites_sagnac(pos, sat)
            dx_all = pos - sat_rot  # [n_sat, 3]
            r = np.linalg.norm(dx_all, axis=1)
            pr_pred = r + cb[mapped_ids]
            residual = pseudoranges - pr_pred

            H = np.zeros((n_sat, n_state))
            H[:, 0] = dx_all[:, 0] / r
            H[:, 1] = dx_all[:, 1] / r
            H[:, 2] = dx_all[:, 2] / r
            for s in range(n_sat):
                H[s, 3 + mapped_ids[s]] = 1.0

            W = np.diag(weights)
            HTWH = H.T @ W @ H
            HTWdy = H.T @ W @ residual

            try:
                delta = np.linalg.solve(HTWH, HTWdy)
            except np.linalg.LinAlgError:
                break

            state += delta
            if np.linalg.norm(delta) < self.tol:
                return state, it + 1

        return state, self.max_iter

    @staticmethod
    def prn_to_system(prn_str: str) -> tuple[int, int]:
        """Convert PRN string like 'G01', 'R05', 'E12', 'C03' to (system_id, prn_num).

        Args:
            prn_str: Satellite PRN string (e.g., 'G01' for GPS PRN 1).

        Returns:
            (system_id, prn_number) tuple.
        """
        mapping = {
            'G': SYSTEM_GPS,
            'R': SYSTEM_GLONASS,
            'E': SYSTEM_GALILEO,
            'C': SYSTEM_BEIDOU,
            'J': SYSTEM_QZSS,
        }
        return mapping.get(prn_str[0], SYSTEM_GPS), int(prn_str[1:])

    @staticmethod
    def system_name(system_id: int) -> str:
        """Get human-readable name for a GNSS system ID."""
        return _SYSTEM_NAMES.get(system_id, f"Unknown({system_id})")
