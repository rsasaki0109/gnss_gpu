from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np

from gnss_gpu.validation.residuals import ks_statistic, wasserstein1


def _as_1d_float(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def score(sim_values, target_values, ks_weight: float = 0.0) -> float:
    sim = _as_1d_float(sim_values)
    target = _as_1d_float(target_values)
    if sim.size == 0 or target.size == 0:
        return float("inf")
    return float(wasserstein1(sim, target) + ks_weight * ks_statistic(sim, target))


def evaluate(sim_values, target_values, ks_weight: float = 0.0) -> dict:
    sim = _as_1d_float(sim_values)
    target = _as_1d_float(target_values)
    if sim.size == 0 or target.size == 0:
        return {"wasserstein": float("inf"), "ks": float("inf"), "score": float("inf")}

    w = float(wasserstein1(sim, target))
    ks = float(ks_statistic(sim, target))
    return {"wasserstein": w, "ks": ks, "score": float(w + ks_weight * ks)}


def grid_search(
    residual_fn: Callable[[dict[str, float]], np.ndarray],
    target_values,
    param_grid: dict[str, list[float]],
    ks_weight: float = 0.0,
) -> dict:
    if not param_grid or any(len(values) == 0 for values in param_grid.values()):
        raise ValueError("param_grid must contain at least one non-empty value list")

    names = list(param_grid.keys())
    values = [param_grid[name] for name in names]

    results = []
    best_params = None
    best_score = float("inf")

    for combo in product(*values):
        params = {name: float(value) for name, value in zip(names, combo)}
        current_score = score(residual_fn(dict(params)), target_values, ks_weight=ks_weight)
        results.append((dict(params), float(current_score)))

        if current_score < best_score:
            best_score = float(current_score)
            best_params = dict(params)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "results": results,
    }


def coordinate_descent(
    residual_fn: Callable[[dict[str, float]], np.ndarray],
    target_values,
    init_params: dict[str, float],
    bounds: dict[str, tuple[float, float]],
    step: dict[str, float] | None = None,
    n_iter: int = 20,
    shrink: float = 0.5,
    ks_weight: float = 0.0,
) -> dict:
    params = {name: float(value) for name, value in init_params.items()}

    for name, (lo, hi) in bounds.items():
        if lo > hi:
            raise ValueError(f"invalid bounds for {name}: lower bound exceeds upper bound")
        if name in params:
            params[name] = float(np.clip(params[name], lo, hi))

    if step is None:
        steps = {
            name: float(abs(hi - lo) * 0.25)
            for name, (lo, hi) in bounds.items()
        }
    else:
        steps = {name: float(value) for name, value in step.items()}

    best_score = score(residual_fn(dict(params)), target_values, ks_weight=ks_weight)
    history = [float(best_score)]

    for _ in range(int(n_iter)):
        improved_this_round = False

        for name in params:
            if name not in bounds:
                continue

            current_step = steps.get(name, 0.0)
            if current_step <= 0.0:
                continue

            lo, hi = bounds[name]
            base_value = params[name]
            best_local_value = base_value
            best_local_score = best_score

            for direction in (-1.0, 1.0):
                candidate = dict(params)
                candidate[name] = float(np.clip(base_value + direction * current_step, lo, hi))
                candidate_score = score(
                    residual_fn(candidate),
                    target_values,
                    ks_weight=ks_weight,
                )

                if candidate_score < best_local_score:
                    best_local_score = float(candidate_score)
                    best_local_value = candidate[name]

            if best_local_score < best_score:
                params[name] = best_local_value
                best_score = best_local_score
                improved_this_round = True

        if not improved_this_round:
            for name in steps:
                steps[name] *= shrink

        history.append(float(best_score))

    return {
        "best_params": dict(params),
        "best_score": float(best_score),
        "history": history,
    }
