"""GPU area-sweep GNSS coverage / accuracy prediction maps.

Sweeps a grid of receiver cells over a city and predicts, per cell, GNSS
positioning quality metrics (visible/LOS satellite counts, availability,
DOP-derived expected horizontal error) over a time window. This is the
Phase 2 "predict positioning accuracy/availability at any place & time"
feature: :mod:`gnss_gpu.scenario` simulates one receiver point/route, this
module batches that same building-model/ephemeris machinery across many
receiver points (cells) so a whole city block can be swept in one pass.

This module is pure wiring on top of existing building blocks; it does not
reimplement any of the underlying physics:

- :mod:`gnss_gpu.scenario` for its (private) epoch-time resolution, PLATEAU
  mesh loading, and GPS-time helpers -- reused directly rather than
  duplicated.
- :mod:`gnss_gpu.io.nav_rinex` for RINEX NAV parsing and
  :class:`gnss_gpu.ephemeris.Ephemeris` for broadcast satellite position.
- :mod:`gnss_gpu.io.plateau` for the optional PLATEAU city mesh and
  :class:`gnss_gpu.bvh.BVHAccelerator` for the batched GPU line-of-sight
  check across (cells x satellites) in a single CUDA launch per epoch --
  the ray-tracing differentiator this module exists to exploit.  When a
  batched call is unavailable (no CUDA device, or the BVH build fails) this
  degrades to a per-cell loop over
  :meth:`gnss_gpu.raytrace.BuildingModel.check_los`, still vectorized over
  satellites within each cell.

Degrades gracefully exactly like :func:`gnss_gpu.scenario.run_scenario`: no
PLATEAU mesh (or no CUDA) simply means every cell is treated as open sky
(``is_los`` all ``True``, no building-footprint masking), with a
``UserWarning`` explaining the fallback.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import read_nav_rinex_multi
from gnss_gpu.scenario import (
    _gps_week_sow,
    _lla_deg_to_ecef,
    _load_building_model,
    _normalize_constellations,
    _resolve_epoch_times,
)

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


# ---------------------------------------------------------------------------
# Small vectorized geometry helpers not already exported elsewhere
# ---------------------------------------------------------------------------


def _enu_basis(lat_rad: float, lon_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East/North/Up unit vectors (ECEF) for a local tangent plane at (lat, lon)."""
    sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
    sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=np.float64)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=np.float64)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=np.float64)
    return east, north, up


def _ecef_to_lla_deg_vec(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ECEF -> (lat_deg, lon_deg); ``ecef`` has shape (..., 3).

    Same iterative Bowring-style formula as
    :func:`gnss_gpu.validation.real_residuals._ecef_to_geodetic_lat_lon`,
    vectorized over an arbitrary batch of points (that helper is scalar-only).
    """
    x = ecef[..., 0]
    y = ecef[..., 1]
    z = ecef[..., 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(8):
        sin_lat = np.sin(lat)
        n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        lat = np.arctan2(z + _WGS84_E2 * n * sin_lat, p)
    return np.degrees(lat), np.degrees(lon)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoverageMapConfig:
    """Configuration for :func:`run_coverage_map`.

    The receiver grid is a square-cell lattice centered on
    (``center_lat_deg``, ``center_lon_deg``): ``extent_east_m`` x
    ``extent_north_m`` metres, tiled with ``cell_size_m`` cells, all sitting
    ``receiver_height_m`` above an assumed-flat ground at ``ground_alt_m``
    (WGS-84 ellipsoidal height; there is no per-cell DEM, mirroring the
    constant-offset simplification :mod:`gnss_gpu.scenario` uses for the
    PLATEAU geoid fallback).

    The epoch time grid is either ``epoch_times`` (explicit list) or derived
    from ``start_time`` + (``end_time`` or ``duration_s``) + ``step_s`` --
    same convention as :class:`gnss_gpu.scenario.ScenarioConfig`.
    """

    nav_file: str

    # --- Grid: center + extent + resolution --------------------------------
    center_lat_deg: float
    center_lon_deg: float
    extent_east_m: float = 300.0
    extent_north_m: float = 300.0
    cell_size_m: float = 10.0
    ground_alt_m: float = 0.0
    receiver_height_m: float = 1.5

    # --- Time window: explicit epochs OR start/end/step ---------------------
    start_time: str | datetime | None = None
    end_time: str | datetime | None = None
    duration_s: float | None = None
    step_s: float = 30.0
    epoch_times: Sequence | None = None

    # --- Constellations + NAV/PLATEAU inputs -------------------------------
    constellations: Sequence[str] = field(default_factory=lambda: ["G"])
    plateau_dir: str | None = None
    plateau_zone: int = 9
    plateau_geoid_correction: object = "egm96"

    # --- Modeling knobs -------------------------------------------------------
    elevation_mask_deg: float = 10.0
    uere_m: float = 5.0

    def __post_init__(self) -> None:
        if self.extent_east_m <= 0.0 or self.extent_north_m <= 0.0:
            raise ValueError("CoverageMapConfig: extent_east_m/extent_north_m must be positive")
        if self.cell_size_m <= 0.0:
            raise ValueError("CoverageMapConfig: cell_size_m must be positive")

        if self.epoch_times is None and self.start_time is None:
            raise ValueError("CoverageMapConfig: one of epoch_times or start_time is required")
        if self.epoch_times is None and self.end_time is None and self.duration_s is None:
            raise ValueError("CoverageMapConfig: start_time requires end_time or duration_s")
        if self.step_s <= 0.0:
            raise ValueError("CoverageMapConfig: step_s must be positive")

        self.constellations = _normalize_constellations(self.constellations)
        if not self.constellations:
            raise ValueError("CoverageMapConfig: constellations must not be empty")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CoverageMapResult:
    """Per-cell GNSS coverage/accuracy prediction, shape ``(n_north, n_east)``."""

    mean_visible: np.ndarray
    mean_los: np.ndarray
    los_fraction: np.ndarray
    availability: np.ndarray
    hdop: np.ndarray
    vdop: np.ndarray
    gdop: np.ndarray
    expected_hpe_m: np.ndarray
    cell_lat_deg: np.ndarray
    cell_lon_deg: np.ndarray
    epoch_times: list
    config: CoverageMapConfig

    @property
    def shape(self) -> tuple[int, int]:
        return self.mean_visible.shape


_METRIC_META: dict[str, tuple[str, bool]] = {
    # metric -> (display label, "lower is better")
    "mean_visible": ("Mean visible satellites", False),
    "mean_los": ("Mean LOS satellites", False),
    "los_fraction": ("LOS fraction", False),
    "availability": (">=4 LOS satellite availability", False),
    "hdop": ("HDOP", True),
    "vdop": ("VDOP", True),
    "gdop": ("GDOP", True),
    "expected_hpe_m": ("Expected horizontal position error [m]", True),
}


# ---------------------------------------------------------------------------
# LOS batching (GPU BVH batched call when available, else a per-cell loop)
# ---------------------------------------------------------------------------


def _satellite_los_grid(bvh, building_model, rx_flat: np.ndarray, sat_flat: np.ndarray, warn_once) -> np.ndarray:
    """LOS for every (cell, satellite) pair, one shared satellite set per epoch.

    Uses :meth:`gnss_gpu.bvh.BVHAccelerator.check_los_batch` (one CUDA launch
    for the whole ``n_cells x n_sat`` batch) when a BVH is available; falls
    back to a per-cell loop over :meth:`BuildingModel.check_los` (still
    vectorized over satellites within each cell) otherwise.
    """
    n_cells = rx_flat.shape[0]
    n_sat = sat_flat.shape[0]
    if n_sat == 0:
        return np.zeros((n_cells, 0), dtype=bool)

    if bvh is not None:
        try:
            sat_tiled = np.ascontiguousarray(
                np.broadcast_to(sat_flat[None, :, :], (n_cells, n_sat, 3))
            )
            return np.asarray(bvh.check_los_batch(rx_flat, sat_tiled), dtype=bool)
        except Exception as exc:
            warn_once(
                f"batched GPU LOS check unavailable ({exc}); falling back to a "
                "per-cell loop (vectorized over satellites)"
            )

    if building_model is None:
        return np.ones((n_cells, n_sat), dtype=bool)

    out = np.empty((n_cells, n_sat), dtype=bool)
    for c in range(n_cells):
        try:
            out[c] = np.asarray(building_model.check_los(rx_flat[c], sat_flat), dtype=bool)
        except Exception as exc:
            warn_once(f"LOS check unavailable ({exc}); treating all satellites as LOS")
            out[c] = True
    return out


def _occupancy_mask(bvh, building_model, ground_flat: np.ndarray, up_vec: np.ndarray, warn_once) -> np.ndarray:
    """True for cells whose ground point sits inside a building footprint.

    Detected with a vertical up-ray: a cell is "occupied" when the segment
    from just above the ground to well above the tallest plausible building
    is blocked by the mesh (the ray starts inside the building volume and
    hits a wall/roof triangle from the inside).
    """
    n_cells = ground_flat.shape[0]
    if building_model is None or getattr(building_model, "triangles", None) is None:
        return np.zeros(n_cells, dtype=bool)
    if np.asarray(building_model.triangles).shape[0] == 0:
        return np.zeros(n_cells, dtype=bool)

    rx = ground_flat + 0.2 * up_vec[None, :]
    target = ground_flat + 500.0 * up_vec[None, :]

    if bvh is not None:
        try:
            is_los = np.asarray(bvh.check_los_batch(rx, target[:, None, :]), dtype=bool)[:, 0]
            return ~is_los
        except Exception as exc:
            warn_once(
                f"batched GPU occupancy check unavailable ({exc}); falling back "
                "to a per-cell loop"
            )

    occupied = np.zeros(n_cells, dtype=bool)
    for c in range(n_cells):
        try:
            is_los = np.asarray(building_model.check_los(rx[c], target[c][None, :]), dtype=bool)
            occupied[c] = not bool(is_los[0])
        except Exception as exc:
            warn_once(f"occupancy LOS check unavailable ({exc}); assuming open ground")
    return occupied


# ---------------------------------------------------------------------------
# DOP
# ---------------------------------------------------------------------------


def _dop_from_mask(unit_enu: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HDOP/VDOP/GDOP per cell from LOS satellites' ENU unit vectors.

    Standard ``(H^T H)^-1`` DOP formula. ``unit_enu`` is ``(n_cells, n_sat, 3)``
    (east, north, up unit line-of-sight components); ``mask`` is
    ``(n_cells, n_sat)`` -- True where a satellite counts (above the
    elevation mask AND LOS). Cells with fewer than 4 counted satellites (or a
    singular geometry matrix) get NaN.
    """
    n_cells, n_sat, _ = unit_enu.shape
    hdop = np.full(n_cells, np.nan, dtype=np.float64)
    vdop = np.full(n_cells, np.nan, dtype=np.float64)
    gdop = np.full(n_cells, np.nan, dtype=np.float64)
    if n_sat == 0:
        return hdop, vdop, gdop

    ones = np.ones((n_cells, n_sat, 1), dtype=np.float64)
    design = np.concatenate([unit_enu, ones], axis=-1)  # (n_cells, n_sat, 4)
    design_masked = np.where(mask[..., None], design, 0.0)
    gram = np.einsum("csi,csj->cij", design_masked, design_masked)  # (n_cells, 4, 4)

    n_counted = mask.sum(axis=1)
    valid_idx = np.where(n_counted >= 4)[0]
    if valid_idx.size == 0:
        return hdop, vdop, gdop

    for c in valid_idx:
        try:
            q = np.linalg.inv(gram[c])
        except np.linalg.LinAlgError:
            continue
        diag = np.diagonal(q)
        if not np.all(np.isfinite(diag)):
            continue
        hdop[c] = math.sqrt(max(diag[0] + diag[1], 0.0))
        vdop[c] = math.sqrt(max(diag[2], 0.0))
        gdop[c] = math.sqrt(max(float(diag.sum()), 0.0))
    return hdop, vdop, gdop


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_coverage_map(config: CoverageMapConfig) -> CoverageMapResult:
    """Sweep a grid of receiver cells and predict per-cell GNSS quality.

    See :class:`CoverageMapConfig` for accepted inputs and
    :class:`CoverageMapResult` for the output schema.
    """
    warned: set[str] = set()

    def warn_once(msg: str) -> None:
        if msg not in warned:
            warned.add(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

    epoch_times = _resolve_epoch_times(config)
    n_epochs = len(epoch_times)

    n_east = max(1, int(round(config.extent_east_m / config.cell_size_m)))
    n_north = max(1, int(round(config.extent_north_m / config.cell_size_m)))

    east_off = (np.arange(n_east, dtype=np.float64) + 0.5 - n_east / 2.0) * config.cell_size_m
    north_off = (np.arange(n_north, dtype=np.float64) + 0.5 - n_north / 2.0) * config.cell_size_m
    east_grid, north_grid = np.meshgrid(east_off, north_off)  # both (n_north, n_east)

    center_ecef = _lla_deg_to_ecef(config.center_lat_deg, config.center_lon_deg, config.ground_alt_m)
    east_vec, north_vec, up_vec = _enu_basis(
        math.radians(config.center_lat_deg), math.radians(config.center_lon_deg)
    )

    ground_ecef = (
        center_ecef[None, None, :]
        + east_grid[..., None] * east_vec[None, None, :]
        + north_grid[..., None] * north_vec[None, None, :]
    )
    rx_ecef_grid = ground_ecef + float(config.receiver_height_m) * up_vec[None, None, :]

    cell_lat_deg, cell_lon_deg = _ecef_to_lla_deg_vec(rx_ecef_grid)

    n_cells = n_north * n_east
    rx_flat = rx_ecef_grid.reshape(n_cells, 3)
    ground_flat = ground_ecef.reshape(n_cells, 3)
    lat_rad_flat = np.radians(cell_lat_deg).reshape(n_cells)
    lon_rad_flat = np.radians(cell_lon_deg).reshape(n_cells)

    nav_messages = read_nav_rinex_multi(config.nav_file, systems=config.constellations)
    eph = Ephemeris(nav_messages)
    available_prns = eph.available_prns

    building_model = _load_building_model(config, warn_once)
    if config.plateau_dir is not None and building_model is None:
        warn_once(
            "no PLATEAU mesh available; treating all cells as open-sky LOS "
            "with no building-footprint masking"
        )

    bvh = None
    if building_model is not None and np.asarray(building_model.triangles).shape[0] > 0:
        try:
            from gnss_gpu.bvh import BVHAccelerator

            bvh = BVHAccelerator.from_building_model(building_model)
        except Exception as exc:
            warn_once(f"BVH build failed ({exc}); using a per-cell linear-scan LOS loop")

    el_mask_rad = math.radians(float(config.elevation_mask_deg))

    sum_visible = np.zeros(n_cells, dtype=np.float64)
    sum_los = np.zeros(n_cells, dtype=np.float64)
    avail_count = np.zeros(n_cells, dtype=np.float64)
    hdop_epochs = np.full((max(n_epochs, 1), n_cells), np.nan, dtype=np.float64)
    vdop_epochs = np.full((max(n_epochs, 1), n_cells), np.nan, dtype=np.float64)
    gdop_epochs = np.full((max(n_epochs, 1), n_cells), np.nan, dtype=np.float64)

    for ei, t in enumerate(epoch_times):
        _week, sow = _gps_week_sow(t)
        sat_ecef, _sat_clk_s, used_prns = eph.compute(sow, prn_list=available_prns)
        if not used_prns:
            continue
        sat_ecef = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)

        diff = sat_ecef[None, :, :] - rx_flat[:, None, :]  # (n_cells, n_sat, 3)
        sin_lat = np.sin(lat_rad_flat)[:, None]
        cos_lat = np.cos(lat_rad_flat)[:, None]
        sin_lon = np.sin(lon_rad_flat)[:, None]
        cos_lon = np.cos(lon_rad_flat)[:, None]
        dx, dy, dz = diff[..., 0], diff[..., 1], diff[..., 2]
        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        rng = np.sqrt(dx * dx + dy * dy + dz * dz)
        el = np.arctan2(up, np.hypot(east, north))

        above_mask = el >= el_mask_rad
        is_los = _satellite_los_grid(bvh, building_model, rx_flat, sat_ecef, warn_once)
        counted = above_mask & is_los

        sum_visible += above_mask.sum(axis=1)
        n_los_cell = counted.sum(axis=1)
        sum_los += n_los_cell
        avail_count += (n_los_cell >= 4).astype(np.float64)

        rng_safe = np.where(rng > 1e-6, rng, 1.0)
        unit_enu = np.stack([east / rng_safe, north / rng_safe, up / rng_safe], axis=-1)
        hd, vd, gd = _dop_from_mask(unit_enu, counted)
        hdop_epochs[ei] = hd
        vdop_epochs[ei] = vd
        gdop_epochs[ei] = gd

    denom = max(n_epochs, 1)
    mean_visible = (sum_visible / denom).reshape(n_north, n_east)
    mean_los = (sum_los / denom).reshape(n_north, n_east)
    mean_visible_safe = np.where(mean_visible > 0.0, mean_visible, np.nan)
    los_fraction = (mean_los / mean_visible_safe)
    availability = (avail_count / denom).reshape(n_north, n_east)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hdop = np.nanmean(hdop_epochs, axis=0).reshape(n_north, n_east)
        vdop = np.nanmean(vdop_epochs, axis=0).reshape(n_north, n_east)
        gdop = np.nanmean(gdop_epochs, axis=0).reshape(n_north, n_east)
    expected_hpe_m = hdop * float(config.uere_m)

    building_mask = _occupancy_mask(bvh, building_model, ground_flat, up_vec, warn_once).reshape(
        n_north, n_east
    )
    if np.any(building_mask):
        for arr in (
            mean_visible, mean_los, los_fraction, availability,
            hdop, vdop, gdop, expected_hpe_m,
        ):
            arr[building_mask] = np.nan

    return CoverageMapResult(
        mean_visible=mean_visible,
        mean_los=mean_los,
        los_fraction=los_fraction,
        availability=availability,
        hdop=hdop,
        vdop=vdop,
        gdop=gdop,
        expected_hpe_m=expected_hpe_m,
        cell_lat_deg=cell_lat_deg,
        cell_lon_deg=cell_lon_deg,
        epoch_times=epoch_times,
        config=config,
    )


# ---------------------------------------------------------------------------
# Rendering: PNG (matplotlib, lazy import) + self-contained deck.gl HTML
# ---------------------------------------------------------------------------


def _ramp_colors(values: np.ndarray, vmin: float, vmax: float, invert: bool) -> np.ndarray:
    """Red -> yellow -> green ramp; ``invert`` swaps which end is "good"."""
    span = max(vmax - vmin, 1e-12)
    norm = np.clip((np.nan_to_num(values, nan=vmin) - vmin) / span, 0.0, 1.0)
    goodness = (1.0 - norm) if invert else norm
    stops = np.array(
        [[214.0, 39.0, 40.0], [255.0, 221.0, 87.0], [44.0, 160.0, 44.0]], dtype=np.float64
    )
    seg = goodness * 2.0
    idx0 = np.clip(seg.astype(np.int64), 0, 1)
    frac = np.clip(seg - idx0, 0.0, 1.0)
    c0 = stops[idx0]
    c1 = stops[idx0 + 1]
    colors = c0 + (c1 - c0) * frac[..., None]
    return np.clip(colors, 0.0, 255.0).astype(np.uint8)


def _metric_array(result: CoverageMapResult, metric: str) -> tuple[np.ndarray, str, bool]:
    if metric not in _METRIC_META:
        raise ValueError(f"unknown metric {metric!r}; choose one of {sorted(_METRIC_META)}")
    label, lower_is_better = _METRIC_META[metric]
    values = getattr(result, metric)
    return values, label, lower_is_better


def to_png(result: CoverageMapResult, path, metric: str = "expected_hpe_m") -> None:
    """Render ``metric`` as a docs-quality heatmap PNG (matplotlib, Agg backend)."""
    values, label, lower_is_better = _metric_array(result, metric)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = "viridis_r" if lower_is_better else "viridis"
    lon = result.cell_lon_deg
    lat = result.cell_lat_deg

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(
        values,
        origin="lower",
        cmap=cmap,
        extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())],
        aspect="auto",
    )
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"GNSS coverage map -- {label}")
    fig.tight_layout()

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def to_deckgl_html(result: CoverageMapResult, path, metric: str = "expected_hpe_m") -> None:
    """Render ``metric`` as a self-contained deck.gl + MapLibre heatmap HTML.

    Follows the same maplibre-gl (OSM raster basemap, no API key) + deck.gl
    ``MapboxOverlay`` (CDN, dark HUD panel) pattern as
    ``experiments/build_deckgl_video.py`` (source of
    ``docs/assets/media/los-nlos/los_nlos_deckgl.html``), simplified to a
    static grid heatmap: colors are precomputed in Python (numpy) and
    embedded as a flat JSON array, so the page needs no client-side color
    scale logic.
    """
    values, label, lower_is_better = _metric_array(result, metric)
    lat = result.cell_lat_deg
    lon = result.cell_lon_deg
    n_north, n_east = values.shape

    finite = np.isfinite(values)
    if finite.any():
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    colors = _ramp_colors(values, vmin, vmax, invert=lower_is_better)

    rows = []
    for j in range(n_north):
        for i in range(n_east):
            if not finite[j, i]:
                continue
            r, g, b = (int(c) for c in colors[j, i])
            rows.append(
                [round(float(lon[j, i]), 7), round(float(lat[j, i]), 7),
                 round(float(values[j, i]), 4), r, g, b]
            )

    lo_r, lo_g, lo_b = (int(c) for c in _ramp_colors(np.array([vmin]), vmin, vmax, lower_is_better)[0])
    hi_r, hi_g, hi_b = (int(c) for c in _ramp_colors(np.array([vmax]), vmin, vmax, lower_is_better)[0])

    data_json = json.dumps(rows)
    center_lat = float(np.nanmean(lat))
    center_lon = float(np.nanmean(lon))
    cell_size_m = float(result.config.cell_size_m)
    n_epochs = len(result.epoch_times)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GNSS Coverage Map -- {label}</title>
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" />
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/deck.gl@9.1.12/dist.min.js"></script>
<script src="https://unpkg.com/@deck.gl/mapbox@9.1.12/dist.min.js"></script>
<style>
  body {{ margin:0; padding:0; background:#0a0f1e; overflow:hidden; font-family:monospace; }}
  #map {{ width:100vw; height:100vh; }}
  #hud {{
    position:absolute; top:12px; left:12px; z-index:10;
    background:rgba(10,15,30,0.92); color:#e0e6f0; padding:14px 18px;
    border-radius:10px; border:1px solid #334455; min-width:260px;
    box-shadow:0 2px 8px rgba(0,0,0,0.4);
  }}
  #hud h2 {{ margin:0 0 6px 0; font-size:15px; color:#ffffff; }}
  #hud .metric {{ font-size:13px; color:#ffd93d; margin-bottom:6px; }}
  #hud .meta {{ color:#9cb0c8; font-size:11px; line-height:1.5; }}
  #legend {{ margin-top:8px; height:12px; border-radius:2px;
             background:linear-gradient(to right, rgb({lo_r},{lo_g},{lo_b}), rgb({hi_r},{hi_g},{hi_b})); }}
  #legend-labels {{ display:flex; justify-content:space-between; font-size:10px; color:#9cb0c8; margin-top:2px; }}
  .maplibregl-ctrl-bottom-right {{ opacity: 0.9; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="hud">
  <h2>GNSS Coverage Map</h2>
  <div class="metric">{label}</div>
  <div class="meta">
    grid: {n_north}x{n_east} cells @ {cell_size_m:.0f} m<br>
    epochs: {n_epochs}<br>
    cells shown: {len(rows)} / {n_north * n_east}
  </div>
  <div id="legend"></div>
  <div id="legend-labels"><span>{vmin:.2f}</span><span>{vmax:.2f}</span></div>
</div>
<script>
const cells = {data_json};
const CELL_SIZE_M = {cell_size_m};

const map = new maplibregl.Map({{
  container: 'map',
  style: {{
    version: 8,
    sources: {{
      osm: {{
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'],
        tileSize: 256,
        maxzoom: 19,
        attribution: '&copy; OpenStreetMap contributors',
      }},
    }},
    layers: [
      {{ id: 'bg', type: 'background', paint: {{ 'background-color': '#0a0f1e' }} }},
      {{ id: 'osm', type: 'raster', source: 'osm', paint: {{ 'raster-opacity': 1.0 }} }},
    ],
  }},
  center: [{center_lon}, {center_lat}],
  zoom: 16,
  pitch: 0,
  attributionControl: true,
}});

const overlay = new deck.MapboxOverlay({{
  interleaved: false,
  layers: [
    new deck.GridCellLayer({{
      id: 'coverage-grid',
      data: cells,
      getPosition: d => [d[0], d[1]],
      getFillColor: d => [d[3], d[4], d[5], 205],
      cellSize: CELL_SIZE_M,
      extruded: false,
      pickable: true,
    }}),
  ],
  getTooltip: ({{object}}) => object ? {json.dumps(label)} + ': ' + object[2].toFixed(3) : null,
}});
map.addControl(overlay);
</script>
</body>
</html>"""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


__all__ = [
    "CoverageMapConfig",
    "CoverageMapResult",
    "run_coverage_map",
    "to_png",
    "to_deckgl_html",
]
