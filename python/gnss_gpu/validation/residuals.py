from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualSample:
    epoch: int
    prn: str
    residual_m: float
    elevation_rad: float
    azimuth_rad: float
    cn0_dbhz: float | None
    is_los: bool


_DEFAULT_PERCENTILES = (50.0, 68.0, 95.0, 99.0)
_NAN = float("nan")


def _as_float_array(values) -> np.ndarray:
    if isinstance(values, np.ndarray):
        arr = values.astype(np.float64, copy=False)
    else:
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            arr = np.asarray(list(values), dtype=np.float64)

    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.reshape(-1)


def _percentile_label(p: float) -> str:
    p = float(p)
    if p.is_integer():
        return f"p{int(p)}"
    return "p" + f"{p:g}".replace(".", "_")


def residual_array(samples: Iterable[ResidualSample]) -> np.ndarray:
    return np.asarray([sample.residual_m for sample in samples], dtype=np.float64)


def percentiles(values, ps=(50, 68, 95, 99)) -> dict[float, float]:
    arr = _as_float_array(values)
    ps_arr = _as_float_array(ps)

    if ps_arr.size == 0:
        return {}

    if arr.size == 0:
        return {float(p): _NAN for p in ps_arr}

    qs = np.asarray(np.percentile(arr, ps_arr), dtype=np.float64).reshape(-1)
    return {float(p): float(q) for p, q in zip(ps_arr, qs)}


def summarize(values) -> dict[str, float | int]:
    arr = _as_float_array(values)
    count = int(arr.size)

    if count == 0:
        return {
            "count": 0,
            "mean": _NAN,
            "rms": _NAN,
            "mae": _NAN,
            "p50": _NAN,
            "p68": _NAN,
            "p95": _NAN,
            "p99": _NAN,
            "abs_p50": _NAN,
            "abs_p95": _NAN,
        }

    pct = percentiles(arr, _DEFAULT_PERCENTILES)
    abs_pct = percentiles(np.abs(arr), (50.0, 95.0))

    return {
        "count": count,
        "mean": float(np.mean(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "mae": float(np.mean(np.abs(arr))),
        "p50": pct[50.0],
        "p68": pct[68.0],
        "p95": pct[95.0],
        "p99": pct[99.0],
        "abs_p50": abs_pct[50.0],
        "abs_p95": abs_pct[95.0],
    }


def empirical_cdf(values) -> tuple[np.ndarray, np.ndarray]:
    sorted_x = np.sort(_as_float_array(values))
    if sorted_x.size == 0:
        return sorted_x, np.empty(0, dtype=np.float64)

    cdf_y = np.arange(1, sorted_x.size + 1, dtype=np.float64) / sorted_x.size
    return sorted_x, cdf_y


def wasserstein1(a, b) -> float:
    a_arr = np.sort(_as_float_array(a))
    b_arr = np.sort(_as_float_array(b))

    n_a = a_arr.size
    n_b = b_arr.size
    if n_a == 0 or n_b == 0:
        return _NAN

    v = np.unique(np.concatenate((a_arr, b_arr)))
    if v.size < 2:
        return 0.0

    cdf_a = np.searchsorted(a_arr, v[:-1], side="right") / n_a
    cdf_b = np.searchsorted(b_arr, v[:-1], side="right") / n_b
    widths = np.diff(v)

    return float(np.sum(np.abs(cdf_a - cdf_b) * widths))


def ks_statistic(a, b) -> float:
    a_arr = np.sort(_as_float_array(a))
    b_arr = np.sort(_as_float_array(b))

    n_a = a_arr.size
    n_b = b_arr.size
    if n_a == 0 or n_b == 0:
        return _NAN

    v = np.unique(np.concatenate((a_arr, b_arr)))
    if v.size == 0:
        return _NAN

    cdf_a = np.searchsorted(a_arr, v, side="right") / n_a
    cdf_b = np.searchsorted(b_arr, v, side="right") / n_b

    return float(np.max(np.abs(cdf_a - cdf_b)))


def compare_distributions(sim_values, real_values) -> dict[str, float | int]:
    sim = _as_float_array(sim_values)
    real = _as_float_array(real_values)

    out: dict[str, float | int] = {
        "n_sim": int(sim.size),
        "n_real": int(real.size),
    }

    sim_pct = percentiles(sim, _DEFAULT_PERCENTILES)
    real_pct = percentiles(real, _DEFAULT_PERCENTILES)

    for p in _DEFAULT_PERCENTILES:
        label = _percentile_label(p)
        sim_value = sim_pct[p]
        real_value = real_pct[p]
        out[f"{label}_sim"] = sim_value
        out[f"{label}_real"] = real_value
        out[f"{label}_delta"] = sim_value - real_value

    bias_sim = float(np.mean(sim)) if sim.size else _NAN
    bias_real = float(np.mean(real)) if real.size else _NAN

    out["bias_sim"] = bias_sim
    out["bias_real"] = bias_real
    out["bias_delta"] = bias_sim - bias_real
    out["wasserstein"] = wasserstein1(sim, real)
    out["ks"] = ks_statistic(sim, real)

    return out


def _edge_label(value: float) -> str:
    return f"{float(value):g}"


def bin_by_elevation(
    samples: Iterable[ResidualSample],
    edges_deg,
) -> dict[str, list[ResidualSample]]:
    edges = _as_float_array(edges_deg)
    if edges.size < 2:
        return {}

    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("edges_deg must be strictly increasing")

    labels = [
        f"[{_edge_label(lo)},{_edge_label(hi)})"
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    bins: dict[str, list[ResidualSample]] = {label: [] for label in labels}

    # Bin directly in radians: converting the sample elevation back to degrees
    # is a lossy round-trip (e.g. degrees(radians(30)) == 29.999999999999996),
    # which makes samples sitting exactly on a half-open boundary fall into the
    # wrong bin. Comparing radians against radian edges keeps boundaries exact.
    edges_rad = np.radians(edges)

    for sample in samples:
        elev_rad = float(sample.elevation_rad)
        idx = int(np.searchsorted(edges_rad, elev_rad, side="right") - 1)
        if 0 <= idx < len(labels):
            bins[labels[idx]].append(sample)

    return bins


def bin_by_los(samples: Iterable[ResidualSample]) -> dict[str, list[ResidualSample]]:
    bins: dict[str, list[ResidualSample]] = {"los": [], "nlos": []}
    for sample in samples:
        bins["los" if sample.is_los else "nlos"].append(sample)
    return bins
