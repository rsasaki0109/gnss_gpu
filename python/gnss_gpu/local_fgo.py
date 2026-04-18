"""Local factor-graph rescue for weak double-difference GNSS windows.

This module keeps the main particle filter as the primary estimator.  It only
solves a small post-process position graph over a selected window, then returns a
replacement trajectory for that window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalFgoWindow:
    """Inclusive local-FGO window over trajectory indices."""

    start: int
    end: int

    def normalized(self, n_epochs: int) -> "LocalFgoWindow":
        start = max(0, int(self.start))
        end = min(int(n_epochs) - 1, int(self.end))
        if end < start:
            raise ValueError(f"empty FGO window after clipping: {self.start}:{self.end}")
        return LocalFgoWindow(start, end)

    @property
    def size(self) -> int:
        return int(self.end) - int(self.start) + 1


@dataclass
class DDCarrierEpoch:
    """DD carrier observations for one epoch."""

    dd_carrier_cycles: np.ndarray
    sat_ecef_k: np.ndarray
    sat_ecef_ref: np.ndarray
    base_range_k: np.ndarray
    base_range_ref: np.ndarray
    wavelengths_m: np.ndarray
    weights: np.ndarray | None = None

    @classmethod
    def from_result(cls, result: Any) -> "DDCarrierEpoch":
        return cls(
            dd_carrier_cycles=np.asarray(result.dd_carrier_cycles, dtype=np.float64),
            sat_ecef_k=np.asarray(result.sat_ecef_k, dtype=np.float64),
            sat_ecef_ref=np.asarray(result.sat_ecef_ref, dtype=np.float64),
            base_range_k=np.asarray(result.base_range_k, dtype=np.float64),
            base_range_ref=np.asarray(result.base_range_ref, dtype=np.float64),
            wavelengths_m=np.asarray(result.wavelengths_m, dtype=np.float64),
            weights=np.asarray(result.dd_weights, dtype=np.float64),
        )

    @property
    def n(self) -> int:
        return int(np.asarray(self.dd_carrier_cycles).size)


@dataclass
class DDPseudorangeEpoch:
    """DD pseudorange observations for one epoch."""

    dd_pseudorange_m: np.ndarray
    sat_ecef_k: np.ndarray
    sat_ecef_ref: np.ndarray
    base_range_k: np.ndarray
    base_range_ref: np.ndarray
    weights: np.ndarray | None = None

    @classmethod
    def from_result(cls, result: Any) -> "DDPseudorangeEpoch":
        return cls(
            dd_pseudorange_m=np.asarray(result.dd_pseudorange_m, dtype=np.float64),
            sat_ecef_k=np.asarray(result.sat_ecef_k, dtype=np.float64),
            sat_ecef_ref=np.asarray(result.sat_ecef_ref, dtype=np.float64),
            base_range_k=np.asarray(result.base_range_k, dtype=np.float64),
            base_range_ref=np.asarray(result.base_range_ref, dtype=np.float64),
            weights=np.asarray(result.dd_weights, dtype=np.float64),
        )

    @property
    def n(self) -> int:
        return int(np.asarray(self.dd_pseudorange_m).size)


@dataclass
class UndiffPseudorangeEpoch:
    """Undifferenced pseudorange rows with a fixed clock-bias estimate."""

    sat_ecef: np.ndarray
    pseudoranges_m: np.ndarray
    clock_bias_m: float
    weights: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(np.asarray(self.pseudoranges_m).size)


@dataclass
class LocalFgoProblem:
    """Inputs for a local position-only FGO solve."""

    initial_positions_ecef: np.ndarray
    window: LocalFgoWindow
    motion_deltas_ecef: np.ndarray | None = None
    motion_sigmas_m: np.ndarray | None = None
    dd_carrier: Sequence[DDCarrierEpoch | None] | None = None
    dd_pseudorange: Sequence[DDPseudorangeEpoch | None] | None = None
    undiff_pseudorange: Sequence[UndiffPseudorangeEpoch | None] | None = None
    prior_positions_ecef: np.ndarray | None = None


@dataclass
class LocalFgoConfig:
    """Noise and optimizer settings for local FGO."""

    prior_sigma_m: float = 0.5
    motion_sigma_m: float = 1.0
    dd_sigma_cycles: float = 0.20
    dd_pr_sigma_m: float = 5.0
    undiff_pr_sigma_m: float = 5.0
    dd_huber_k: float = 1.5
    pr_huber_k: float = 1.5
    max_iterations: int = 50
    relative_error_tol: float = 1e-5
    min_weight: float = 1e-3


@dataclass
class LocalFgoGraph:
    graph: Any
    values: Any
    keys: list[int]
    window: LocalFgoWindow
    factor_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class LocalFgoResult:
    positions_ecef: np.ndarray
    window: LocalFgoWindow
    factor_counts: dict[str, int]
    initial_error: float
    final_error: float


def parse_window_spec(spec: str, n_epochs: int) -> LocalFgoWindow:
    """Parse ``N:M`` inclusive window syntax."""

    if ":" not in str(spec):
        raise ValueError(f"FGO window must use N:M syntax, got {spec!r}")
    left, right = str(spec).split(":", 1)
    if not left or not right:
        raise ValueError(f"FGO window must include both bounds, got {spec!r}")
    return LocalFgoWindow(int(left), int(right)).normalized(n_epochs)


def detect_weak_dd_window(
    epoch_diagnostics: Sequence[dict[str, Any]],
    *,
    min_epochs: int = 100,
    dd_max_pairs: int = 4,
) -> LocalFgoWindow | None:
    """Return the longest weak-DD run from per-epoch diagnostics.

    An epoch is weak when the kept DD carrier pair count is at or below
    ``dd_max_pairs`` or when DD carrier was not used despite DD input being
    present.
    """

    best: tuple[int, int] | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    for i, row in enumerate(epoch_diagnostics):
        kept = _finite_int(row.get("dd_cp_kept_pairs"))
        input_pairs = _finite_int(row.get("dd_cp_input_pairs"))
        used = bool(row.get("used_dd_carrier"))
        weak = kept is not None and kept <= int(dd_max_pairs)
        weak = weak or (input_pairs is not None and input_pairs > 0 and not used)
        if weak:
            if cur_start is None:
                cur_start = i
            cur_end = i
            continue
        if cur_start is not None and cur_end is not None:
            best = _prefer_longer(best, (cur_start, cur_end))
        cur_start = None
        cur_end = None
    if cur_start is not None and cur_end is not None:
        best = _prefer_longer(best, (cur_start, cur_end))
    if best is None or best[1] - best[0] + 1 < int(min_epochs):
        return None
    return LocalFgoWindow(best[0], best[1])


def build_factor_graph(
    problem: LocalFgoProblem,
    config: LocalFgoConfig | None = None,
) -> LocalFgoGraph:
    """Build a GTSAM factor graph for a local position-only window."""

    gtsam = _require_gtsam()
    config = LocalFgoConfig() if config is None else config
    initial_positions = _as_positions(problem.initial_positions_ecef)
    window = problem.window.normalized(len(initial_positions))
    prior_positions = (
        initial_positions
        if problem.prior_positions_ecef is None
        else _as_positions(problem.prior_positions_ecef)
    )
    if len(prior_positions) != len(initial_positions):
        raise ValueError("prior_positions_ecef must match initial_positions_ecef")

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    keys = [_key_for_index(i) for i in range(window.start, window.end + 1)]
    key_by_index = dict(zip(range(window.start, window.end + 1), keys))
    counts = {
        "prior": 0,
        "between": 0,
        "dd_carrier": 0,
        "dd_pseudorange": 0,
        "undiff_pseudorange": 0,
    }

    for i in range(window.start, window.end + 1):
        values.insert(key_by_index[i], _point3(gtsam, initial_positions[i]))

    prior_model = gtsam.noiseModel.Isotropic.Sigma(3, float(config.prior_sigma_m))
    for i in (window.start, window.end):
        graph.add(
            gtsam.PriorFactorPoint3(
                key_by_index[i],
                _point3(gtsam, prior_positions[i]),
                prior_model,
            )
        )
        counts["prior"] += 1

    motion_deltas = problem.motion_deltas_ecef
    motion_sigmas = problem.motion_sigmas_m
    for i in range(window.start, window.end):
        if motion_deltas is None:
            delta = initial_positions[i + 1] - initial_positions[i]
        else:
            delta = np.asarray(motion_deltas[i], dtype=np.float64).ravel()[:3]
        if not np.isfinite(delta).all():
            continue
        sigma = float(config.motion_sigma_m)
        if motion_sigmas is not None and i < len(motion_sigmas):
            sigma_i = float(motion_sigmas[i])
            if np.isfinite(sigma_i) and sigma_i > 0.0:
                sigma = sigma_i
        graph.add(
            gtsam.BetweenFactorPoint3(
                key_by_index[i],
                key_by_index[i + 1],
                _point3(gtsam, delta),
                gtsam.noiseModel.Isotropic.Sigma(3, sigma),
            )
        )
        counts["between"] += 1

    for i in range(window.start, window.end + 1):
        key = key_by_index[i]
        if problem.dd_carrier is not None and i < len(problem.dd_carrier):
            obs = problem.dd_carrier[i]
            if obs is not None:
                counts["dd_carrier"] += _add_dd_carrier_factors(
                    gtsam,
                    graph,
                    key,
                    obs,
                    config,
                )
        if problem.dd_pseudorange is not None and i < len(problem.dd_pseudorange):
            obs_pr = problem.dd_pseudorange[i]
            if obs_pr is not None:
                counts["dd_pseudorange"] += _add_dd_pseudorange_factors(
                    gtsam,
                    graph,
                    key,
                    obs_pr,
                    config,
                )
        if problem.undiff_pseudorange is not None and i < len(problem.undiff_pseudorange):
            obs_ud = problem.undiff_pseudorange[i]
            if obs_ud is not None:
                counts["undiff_pseudorange"] += _add_undiff_pr_factors(
                    gtsam,
                    graph,
                    key,
                    obs_ud,
                    config,
                )

    return LocalFgoGraph(
        graph=graph,
        values=values,
        keys=keys,
        window=window,
        factor_counts=counts,
    )


def solve_fgo(
    graph: LocalFgoGraph,
    config: LocalFgoConfig | None = None,
) -> LocalFgoResult:
    """Optimize a local graph and return the solved ECEF positions."""

    gtsam = _require_gtsam()
    config = LocalFgoConfig() if config is None else config
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(int(config.max_iterations))
    params.setRelativeErrorTol(float(config.relative_error_tol))
    initial_error = float(graph.graph.error(graph.values))
    optimized = gtsam.LevenbergMarquardtOptimizer(
        graph.graph,
        graph.values,
        params,
    ).optimize()
    final_error = float(graph.graph.error(optimized))
    positions = np.vstack([np.asarray(optimized.atPoint3(k), dtype=np.float64) for k in graph.keys])
    return LocalFgoResult(
        positions_ecef=positions,
        window=graph.window,
        factor_counts=dict(graph.factor_counts),
        initial_error=initial_error,
        final_error=final_error,
    )


def inject_into_pf(
    pf_result: np.ndarray | dict[str, Any],
    fgo_trajectory: np.ndarray,
    window: LocalFgoWindow,
) -> np.ndarray | dict[str, Any]:
    """Return a copy of a trajectory-like PF result with the FGO window replaced."""

    if isinstance(pf_result, dict):
        updated = dict(pf_result)
        for key in ("positions_ecef", "trajectory_ecef", "smoothed_positions_ecef"):
            if key in updated:
                updated[key] = inject_into_pf(
                    np.asarray(updated[key], dtype=np.float64),
                    fgo_trajectory,
                    window,
                )
                return updated
        raise KeyError("pf_result dict must contain a known trajectory key")

    traj = np.asarray(pf_result, dtype=np.float64).copy()
    if traj.ndim != 2 or traj.shape[1] < 3:
        raise ValueError("pf_result trajectory must have shape [n, >=3]")
    win = window.normalized(len(traj))
    fgo = np.asarray(fgo_trajectory, dtype=np.float64)
    if fgo.shape[0] != win.size or fgo.shape[1] < 3:
        raise ValueError("fgo_trajectory length must match the target window")
    traj[win.start : win.end + 1, :3] = fgo[:, :3]
    return traj


def solve_local_fgo(
    problem: LocalFgoProblem,
    config: LocalFgoConfig | None = None,
) -> LocalFgoResult:
    """Convenience wrapper for ``build_factor_graph`` followed by ``solve_fgo``."""

    return solve_fgo(build_factor_graph(problem, config), config)


def _add_dd_carrier_factors(
    gtsam: Any,
    graph: Any,
    key: int,
    obs: DDCarrierEpoch,
    config: LocalFgoConfig,
) -> int:
    dd = np.asarray(obs.dd_carrier_cycles, dtype=np.float64).ravel()
    sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
    base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
    wavelengths = np.asarray(obs.wavelengths_m, dtype=np.float64).ravel()
    weights = _weights(obs.weights, len(dd))
    n_added = 0
    for j in range(len(dd)):
        if not _valid_dd_row(dd[j], sat_k[j], sat_ref[j], base_k[j], base_ref[j], wavelengths[j]):
            continue
        sigma = _weighted_sigma(config.dd_sigma_cycles, weights[j], config.min_weight)
        model = _robust_model(gtsam, 1, sigma, config.dd_huber_k)
        factor = gtsam.CustomFactor(
            model,
            [key],
            _dd_carrier_error_fn(
                sat_k[j],
                sat_ref[j],
                float(base_k[j]),
                float(base_ref[j]),
                float(wavelengths[j]),
                float(dd[j]),
            ),
        )
        graph.add(factor)
        n_added += 1
    return n_added


def _add_dd_pseudorange_factors(
    gtsam: Any,
    graph: Any,
    key: int,
    obs: DDPseudorangeEpoch,
    config: LocalFgoConfig,
) -> int:
    dd = np.asarray(obs.dd_pseudorange_m, dtype=np.float64).ravel()
    sat_k = np.asarray(obs.sat_ecef_k, dtype=np.float64).reshape(-1, 3)
    sat_ref = np.asarray(obs.sat_ecef_ref, dtype=np.float64).reshape(-1, 3)
    base_k = np.asarray(obs.base_range_k, dtype=np.float64).ravel()
    base_ref = np.asarray(obs.base_range_ref, dtype=np.float64).ravel()
    weights = _weights(obs.weights, len(dd))
    n_added = 0
    for j in range(len(dd)):
        if not _valid_dd_row(dd[j], sat_k[j], sat_ref[j], base_k[j], base_ref[j], 1.0):
            continue
        sigma = _weighted_sigma(config.dd_pr_sigma_m, weights[j], config.min_weight)
        graph.add(
            gtsam.CustomFactor(
                _robust_model(gtsam, 1, sigma, config.pr_huber_k),
                [key],
                _dd_pr_error_fn(
                    sat_k[j],
                    sat_ref[j],
                    float(base_k[j]),
                    float(base_ref[j]),
                    float(dd[j]),
                ),
            )
        )
        n_added += 1
    return n_added


def _add_undiff_pr_factors(
    gtsam: Any,
    graph: Any,
    key: int,
    obs: UndiffPseudorangeEpoch,
    config: LocalFgoConfig,
) -> int:
    sat = np.asarray(obs.sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(obs.pseudoranges_m, dtype=np.float64).ravel()
    weights = _weights(obs.weights, len(pr))
    cb = float(obs.clock_bias_m)
    n_added = 0
    for j in range(len(pr)):
        if not (np.isfinite(pr[j]) and np.isfinite(cb) and np.isfinite(sat[j]).all()):
            continue
        sigma = _weighted_sigma(config.undiff_pr_sigma_m, weights[j], config.min_weight)
        graph.add(
            gtsam.CustomFactor(
                _robust_model(gtsam, 1, sigma, config.pr_huber_k),
                [key],
                _undiff_pr_error_fn(sat[j], float(pr[j]), cb),
            )
        )
        n_added += 1
    return n_added


def _dd_carrier_error_fn(
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
    wavelength: float,
    observed_cycles: float,
):
    def error(_factor: Any, values: Any, jacobians: list[np.ndarray] | None) -> np.ndarray:
        x = np.asarray(values.atPoint3(_factor.keys()[0]), dtype=np.float64)
        expected_m, jac_m = _dd_expected_and_jacobian_m(x, sat_k, sat_ref, base_k, base_ref)
        raw_residual = observed_cycles - expected_m / wavelength
        residual = raw_residual - np.round(raw_residual)
        if jacobians is not None:
            jacobians[0] = np.asarray(-jac_m / wavelength, dtype=np.float64).reshape(1, 3, order="C")
        return np.asarray([residual], dtype=np.float64)

    return error


def _dd_pr_error_fn(
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
    observed_m: float,
):
    def error(_factor: Any, values: Any, jacobians: list[np.ndarray] | None) -> np.ndarray:
        x = np.asarray(values.atPoint3(_factor.keys()[0]), dtype=np.float64)
        expected_m, jac_m = _dd_expected_and_jacobian_m(x, sat_k, sat_ref, base_k, base_ref)
        if jacobians is not None:
            jacobians[0] = np.asarray(jac_m, dtype=np.float64).reshape(1, 3, order="C")
        return np.asarray([expected_m - observed_m], dtype=np.float64)

    return error


def _undiff_pr_error_fn(sat: np.ndarray, pseudorange_m: float, clock_bias_m: float):
    def error(_factor: Any, values: Any, jacobians: list[np.ndarray] | None) -> np.ndarray:
        x = np.asarray(values.atPoint3(_factor.keys()[0]), dtype=np.float64)
        rng, jac = _range_and_jacobian(x, sat)
        if jacobians is not None:
            jacobians[0] = np.asarray(jac, dtype=np.float64).reshape(1, 3, order="C")
        return np.asarray([rng + clock_bias_m - pseudorange_m], dtype=np.float64)

    return error


def _dd_expected_m(
    x: np.ndarray,
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
) -> float:
    return float(np.linalg.norm(sat_k - x) - np.linalg.norm(sat_ref - x) - base_k + base_ref)


def _dd_expected_and_jacobian_m(
    x: np.ndarray,
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
) -> tuple[float, np.ndarray]:
    rng_k, jac_k = _range_and_jacobian(x, sat_k)
    rng_ref, jac_ref = _range_and_jacobian(x, sat_ref)
    return float(rng_k - rng_ref - base_k + base_ref), jac_k - jac_ref


def _range_and_jacobian(x: np.ndarray, sat: np.ndarray) -> tuple[float, np.ndarray]:
    vec = np.asarray(x, dtype=np.float64) - np.asarray(sat, dtype=np.float64)
    rng = float(np.linalg.norm(vec))
    if not np.isfinite(rng) or rng <= 1e-9:
        return rng, np.zeros(3, dtype=np.float64)
    return rng, vec / rng


def _robust_model(gtsam: Any, dim: int, sigma: float, huber_k: float):
    base = gtsam.noiseModel.Isotropic.Sigma(int(dim), float(sigma))
    if huber_k <= 0.0:
        return base
    return gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(float(huber_k)),
        base,
    )


def _weighted_sigma(base_sigma: float, weight: float, min_weight: float) -> float:
    w = max(float(weight), float(min_weight))
    if not np.isfinite(w):
        w = float(min_weight)
    return float(base_sigma) / float(np.sqrt(w))


def _weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(int(n), dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64).ravel()
    if len(arr) != int(n):
        raise ValueError(f"weights length {len(arr)} does not match observation length {n}")
    return arr


def _valid_dd_row(
    observed: float,
    sat_k: np.ndarray,
    sat_ref: np.ndarray,
    base_k: float,
    base_ref: float,
    wavelength: float,
) -> bool:
    return bool(
        np.isfinite(observed)
        and np.isfinite(sat_k).all()
        and np.isfinite(sat_ref).all()
        and np.isfinite(base_k)
        and np.isfinite(base_ref)
        and np.isfinite(wavelength)
        and float(wavelength) > 0.0
    )


def _as_positions(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("positions must have shape [n, >=3]")
    return np.asarray(arr[:, :3], dtype=np.float64)


def _point3(gtsam: Any, value: np.ndarray):
    arr = np.asarray(value, dtype=np.float64).ravel()
    return gtsam.Point3(float(arr[0]), float(arr[1]), float(arr[2]))


def _key_for_index(index: int) -> int:
    gtsam = _require_gtsam()
    return int(gtsam.symbol("x", int(index)))


def _require_gtsam():
    try:
        import gtsam  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "local FGO requires the Python GTSAM bindings; install with "
            "`pip install -r requirements.txt`"
        ) from exc
    return gtsam


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _prefer_longer(
    current: tuple[int, int] | None,
    candidate: tuple[int, int],
) -> tuple[int, int]:
    if current is None:
        return candidate
    cur_len = current[1] - current[0]
    cand_len = candidate[1] - candidate[0]
    if cand_len > cur_len:
        return candidate
    return current
