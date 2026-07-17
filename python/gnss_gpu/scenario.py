"""Unified scenario engine.

Turns ``(receiver location or route, time window, constellations, optional
PLATEAU city mesh)`` into simulated per-epoch GNSS observables: visible
satellites, elevation/azimuth, LOS/NLOS flag, pseudorange (with clock /
ionosphere / troposphere / multipath-excess-delay errors), C/N0 estimate, and
Doppler.

This module is pure wiring -- it reuses the existing building blocks and does
not reimplement any of the underlying physics:

- :mod:`gnss_gpu.io.nav_rinex` for RINEX NAV parsing (multi-constellation) and
  Klobuchar header extraction.
- :class:`gnss_gpu.ephemeris.Ephemeris` for broadcast satellite position /
  clock.
- :class:`gnss_gpu.atmosphere.AtmosphereCorrection` for Klobuchar iono +
  Saastamoinen tropo.
- :mod:`gnss_gpu.io.plateau` / :mod:`gnss_gpu.raytrace` for the optional
  PLATEAU mesh + GPU line-of-sight check.
- :mod:`gnss_gpu.diffraction` / :mod:`gnss_gpu.utd_diffraction` for the
  NLOS excess-delay + attenuation of a diffracted replica, and
  :class:`gnss_gpu.raytrace.BuildingModel` for first-order specular
  reflections.

All of the above are optional in the sense that this module degrades
gracefully when they are unavailable (no CUDA GPU, no PLATEAU mesh, no
``experiments`` package on ``sys.path`` for the edge extractor): satellites
are then treated as open-sky LOS with zero multipath, and a ``UserWarning``
is emitted once per :func:`run_scenario` call explaining the fallback.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from gnss_gpu.antenna import AntennaPattern
from gnss_gpu.atmosphere import AtmosphereCorrection
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.io.nav_rinex import (
    _datetime_to_gps_seconds_of_week,
    _datetime_to_gps_week,
    read_gps_klobuchar_from_nav_header,
    read_nav_rinex_multi,
)
from gnss_gpu.validation.real_residuals import elevation_azimuth

C_LIGHT = 299_792_458.0
# GPS L1 C/A wavelength -- used for every constellation's Doppler conversion.
# This is a deliberate simplification (an L1/E1/B1 style receiver would use a
# per-system carrier), acceptable for a scenario-simulation Doppler estimate.
L1_WAVELENGTH_M = 0.19029367279836488
GPS_WEEK_SEC = 604800.0

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


# ---------------------------------------------------------------------------
# Small geometry helpers not already exported elsewhere
# ---------------------------------------------------------------------------


def _lla_deg_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Geodetic (deg, deg, m) -> ECEF [m], WGS-84."""
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def _parse_time(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _gps_week_sow(dt: datetime) -> tuple[int, float]:
    """Datetime (UTC, GPS-time-scale) -> (gps_week, seconds_of_week)."""
    return _datetime_to_gps_week(dt), _datetime_to_gps_seconds_of_week(dt)


def _normalize_constellations(value) -> tuple[str, ...]:
    if isinstance(value, str):
        chars = value.replace(",", " ").split()
        if len(chars) == 1 and len(chars[0]) > 1:
            # A bare string like "GEJ" -- treat each character as one system.
            chars = list(chars[0])
        return tuple(sorted({c.strip().upper() for c in chars if c.strip()}))
    return tuple(sorted({str(c).strip().upper() for c in value if str(c).strip()}))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    """Configuration for :func:`run_scenario`.

    Receiver position is either a single fixed point (``lat_deg``/``lon_deg``/
    ``alt_m``) or a route: an ``(N, 4)`` array-like of
    ``(time, lat_deg, lon_deg, alt_m)`` rows, where ``time`` is a datetime,
    ISO-8601 string, or a raw GPS seconds-of-week float. Route lookups use
    linear interpolation on seconds-of-week and assume the whole route sits
    inside a single GPS week (fine for short recorded runs).

    The epoch time grid is either ``epoch_times`` (explicit list) or derived
    from ``start_time`` + (``end_time`` or ``duration_s``) + ``step_s``.

    ``antenna`` selects the receiver antenna gain pattern added into the
    C/N0 computation: ``None`` (default) keeps the original isotropic-ish
    ``cn0_zenith_dbhz * sin(elevation)`` model unchanged, a preset name
    string (``"isotropic"``, ``"patch"``, ``"helix"``, ``"smartphone"``,
    see :meth:`gnss_gpu.antenna.AntennaPattern.preset`) resolves to a
    built-in pattern, or an :class:`~gnss_gpu.antenna.AntennaPattern`
    instance may be passed directly for a custom pattern.
    """

    nav_file: str

    # --- Receiver position: fixed point OR route (mutually exclusive) -----
    lat_deg: float | None = None
    lon_deg: float | None = None
    alt_m: float | None = None
    route: np.ndarray | Sequence[Sequence] | None = None

    # --- Time window: explicit epochs OR start/end/step --------------------
    start_time: str | datetime | None = None
    end_time: str | datetime | None = None
    duration_s: float | None = None
    step_s: float = 1.0
    epoch_times: Sequence | None = None

    # --- Constellations + NAV/PLATEAU inputs -------------------------------
    constellations: Sequence[str] = field(default_factory=lambda: ["G"])
    plateau_dir: str | None = None
    plateau_zone: int = 9
    plateau_geoid_correction: object = "egm96"

    # --- Modeling knobs -----------------------------------------------------
    elevation_mask_deg: float = 10.0
    diffraction_model: str | None = "knife_edge"
    utd_mode: str = "absorbing"
    cn0_zenith_dbhz: float = 45.0
    antenna: str | AntennaPattern | None = None
    rx_clock_bias_m: float = 0.0
    nlos_attenuation_db: float = 15.0
    nlos_excess_fallback_m: float = 20.0
    max_reflection_paths: int = 2
    max_diffraction_paths: int = 2
    reflection_cull_radius_m: float = 150.0
    pr_noise_sigma_zenith_m: float = 0.3
    pr_noise_sigma_horizon_m: float = 3.0
    seed: int | None = None

    def __post_init__(self) -> None:
        has_point = self.lat_deg is not None or self.lon_deg is not None or self.alt_m is not None
        has_route = self.route is not None
        if has_point and has_route:
            raise ValueError("ScenarioConfig: pass either lat/lon/alt or route, not both")
        if not has_point and not has_route:
            raise ValueError("ScenarioConfig: one of lat/lon/alt or route is required")
        if has_point and (self.lat_deg is None or self.lon_deg is None or self.alt_m is None):
            raise ValueError("ScenarioConfig: lat_deg, lon_deg and alt_m must all be given")

        if self.epoch_times is None and self.start_time is None:
            raise ValueError("ScenarioConfig: one of epoch_times or start_time is required")
        if self.epoch_times is None and self.end_time is None and self.duration_s is None:
            raise ValueError("ScenarioConfig: start_time requires end_time or duration_s")
        if self.step_s <= 0.0:
            raise ValueError("ScenarioConfig: step_s must be positive")

        self.constellations = _normalize_constellations(self.constellations)
        if not self.constellations:
            raise ValueError("ScenarioConfig: constellations must not be empty")

        if self.diffraction_model not in (None, "knife_edge", "utd"):
            raise ValueError(
                f"ScenarioConfig: diffraction_model must be None, 'knife_edge' or "
                f"'utd', got {self.diffraction_model!r}"
            )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class EpochRecord:
    """One epoch's worth of per-satellite simulated observables."""

    time_gps_week: int
    time_sow: float
    time_utc: datetime
    rx_ecef: np.ndarray

    sat_id: np.ndarray
    elevation_rad: np.ndarray
    azimuth_rad: np.ndarray
    is_los: np.ndarray
    pseudorange_m: np.ndarray
    range_geometric_m: np.ndarray
    sat_clock_bias_m: np.ndarray
    iono_m: np.ndarray
    tropo_m: np.ndarray
    multipath_excess_m: np.ndarray
    cn0_dbhz: np.ndarray
    doppler_hz: np.ndarray

    @property
    def n_sat(self) -> int:
        return int(self.sat_id.size)


_EPOCH_ARRAY_FIELDS = (
    "sat_id",
    "elevation_rad",
    "azimuth_rad",
    "is_los",
    "pseudorange_m",
    "range_geometric_m",
    "sat_clock_bias_m",
    "iono_m",
    "tropo_m",
    "multipath_excess_m",
    "cn0_dbhz",
    "doppler_hz",
)


@dataclass
class ScenarioResult:
    """Container for :func:`run_scenario` output."""

    epochs: list[EpochRecord]
    config: ScenarioConfig

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    def to_arrays(self) -> dict[str, np.ndarray]:
        """Flatten every epoch into parallel arrays with an ``epoch_index`` column."""
        n_rows = sum(ep.n_sat for ep in self.epochs)

        epoch_index = np.empty(n_rows, dtype=np.int64)
        gps_week = np.empty(n_rows, dtype=np.int64)
        time_sow = np.empty(n_rows, dtype=np.float64)
        time_utc = np.empty(n_rows, dtype=object)
        rx_ecef_x = np.empty(n_rows, dtype=np.float64)
        rx_ecef_y = np.empty(n_rows, dtype=np.float64)
        rx_ecef_z = np.empty(n_rows, dtype=np.float64)
        per_field = {name: [] for name in _EPOCH_ARRAY_FIELDS}

        pos = 0
        for idx, ep in enumerate(self.epochs):
            n = ep.n_sat
            sl = slice(pos, pos + n)
            epoch_index[sl] = idx
            gps_week[sl] = ep.time_gps_week
            time_sow[sl] = ep.time_sow
            time_utc[sl] = ep.time_utc.isoformat()
            rx_ecef_x[sl] = ep.rx_ecef[0]
            rx_ecef_y[sl] = ep.rx_ecef[1]
            rx_ecef_z[sl] = ep.rx_ecef[2]
            for name in _EPOCH_ARRAY_FIELDS:
                per_field[name].append(getattr(ep, name))
            pos += n

        def _cat(name: str, dtype) -> np.ndarray:
            parts = per_field[name]
            if not parts:
                return np.array([], dtype=dtype)
            return np.concatenate(parts).astype(dtype, copy=False)

        return {
            "epoch_index": epoch_index,
            "gps_week": gps_week,
            "time_sow": time_sow,
            "time_utc": time_utc,
            "rx_ecef_x_m": rx_ecef_x,
            "rx_ecef_y_m": rx_ecef_y,
            "rx_ecef_z_m": rx_ecef_z,
            "sat_id": _cat("sat_id", "<U3"),
            "elevation_rad": _cat("elevation_rad", np.float64),
            "azimuth_rad": _cat("azimuth_rad", np.float64),
            "is_los": _cat("is_los", bool),
            "pseudorange_m": _cat("pseudorange_m", np.float64),
            "range_geometric_m": _cat("range_geometric_m", np.float64),
            "sat_clock_bias_m": _cat("sat_clock_bias_m", np.float64),
            "iono_m": _cat("iono_m", np.float64),
            "tropo_m": _cat("tropo_m", np.float64),
            "multipath_excess_m": _cat("multipath_excess_m", np.float64),
            "cn0_dbhz": _cat("cn0_dbhz", np.float64),
            "doppler_hz": _cat("doppler_hz", np.float64),
        }

    def to_rinex(
        self,
        path: str | Path,
        marker_name: str = "SIM",
        receiver_type: str = "gnss_gpu simulator",
    ) -> None:
        """Export this scenario as a RINEX 3.04 observation file.

        Maps ``pseudorange_m`` -> ``C1C``, ``doppler_hz`` -> ``D1C`` and
        ``cn0_dbhz`` -> ``S1C``. No carrier-phase observable is simulated,
        so ``L1C`` is omitted entirely (not written as all-blank).
        """
        # Imported lazily so importing :mod:`gnss_gpu.scenario` does not
        # pull in the RINEX writer module for callers who never export.
        from gnss_gpu.io.rinex_writer import (
            EpochRecord as _RinexEpochRecord,
            RinexObsHeader,
            write_rinex_obs,
        )

        codes = ("C1C", "D1C", "S1C")
        systems = sorted({ep.sat_id[i][0] for ep in self.epochs for i in range(ep.n_sat)})
        obs_types = {sys: list(codes) for sys in systems}

        rinex_epochs = [
            _RinexEpochRecord(
                time=ep.time_utc,
                sat_ids=list(ep.sat_id),
                obs={
                    "C1C": ep.pseudorange_m,
                    "D1C": ep.doppler_hz,
                    "S1C": ep.cn0_dbhz,
                },
            )
            for ep in self.epochs
        ]

        approx_position_ecef = self.epochs[0].rx_ecef if self.epochs else np.zeros(3)
        time_first_obs = self.epochs[0].time_utc if self.epochs else None
        interval_s = float(self.config.step_s)
        if len(self.epochs) >= 2:
            interval_s = abs(self.epochs[1].time_sow - self.epochs[0].time_sow) or interval_s

        header = RinexObsHeader(
            marker_name=marker_name,
            receiver_type=receiver_type,
            approx_position_ecef=np.asarray(approx_position_ecef, dtype=np.float64),
            obs_types=obs_types,
            interval_s=interval_s,
            time_first_obs=time_first_obs,
        )

        write_rinex_obs(path, header, rinex_epochs)


# ---------------------------------------------------------------------------
# Epoch time / receiver position resolution
# ---------------------------------------------------------------------------


def _resolve_epoch_times(config: ScenarioConfig) -> list[datetime]:
    if config.epoch_times is not None:
        return [_parse_time(t) for t in config.epoch_times]

    start = _parse_time(config.start_time)
    step = float(config.step_s)
    if config.end_time is not None:
        end = _parse_time(config.end_time)
        n = int(round((end - start).total_seconds() / step)) + 1
    else:
        n = int(round(float(config.duration_s) / step)) + 1
    n = max(n, 0)
    return [start + timedelta(seconds=i * step) for i in range(n)]


def _route_sow(entry_time, route_week: int) -> float:
    if isinstance(entry_time, (int, float)) and not isinstance(entry_time, bool):
        return float(entry_time)
    dt = _parse_time(entry_time)
    week, sow = _gps_week_sow(dt)
    sow += (week - route_week) * GPS_WEEK_SEC
    return sow


def _resolve_receiver_lla(
    config: ScenarioConfig, epoch_times: list[datetime]
) -> np.ndarray:
    """Return an ``(n_epoch, 3)`` array of (lat_deg, lon_deg, alt_m)."""
    n_epoch = len(epoch_times)
    if config.route is None:
        row = np.array([config.lat_deg, config.lon_deg, config.alt_m], dtype=np.float64)
        return np.tile(row, (n_epoch, 1))

    route = list(config.route)
    if not route:
        raise ValueError("ScenarioConfig.route must not be empty")

    route_week, _ = _gps_week_sow(epoch_times[0])
    route_sow = np.array([_route_sow(r[0], route_week) for r in route], dtype=np.float64)
    order = np.argsort(route_sow)
    route_sow = route_sow[order]
    lat = np.array([float(route[i][1]) for i in order], dtype=np.float64)
    lon = np.array([float(route[i][2]) for i in order], dtype=np.float64)
    alt = np.array([float(route[i][3]) for i in order], dtype=np.float64)

    epoch_sow = np.empty(n_epoch, dtype=np.float64)
    for i, t in enumerate(epoch_times):
        week, sow = _gps_week_sow(t)
        epoch_sow[i] = sow + (week - route_week) * GPS_WEEK_SEC

    out = np.empty((n_epoch, 3), dtype=np.float64)
    out[:, 0] = np.interp(epoch_sow, route_sow, lat)
    out[:, 1] = np.interp(epoch_sow, route_sow, lon)
    out[:, 2] = np.interp(epoch_sow, route_sow, alt)
    return out


# ---------------------------------------------------------------------------
# Optional PLATEAU mesh + diffraction edges (best-effort, graceful degrade)
# ---------------------------------------------------------------------------


def _load_building_model(config: ScenarioConfig, warn_once):
    if config.plateau_dir is None:
        return None
    from gnss_gpu.io.plateau import load_plateau

    try:
        return load_plateau(
            config.plateau_dir,
            zone=config.plateau_zone,
            geoid_correction=config.plateau_geoid_correction,
        )
    except Exception as exc:
        # Covers both "pyproj not installed" (ImportError) and "pyproj
        # installed but missing the egm96_15.gtx grid data" (raised deeper
        # inside pyproj as a ProjError/RuntimeError) -- same fallback as
        # examples/demo_diffraction_benchmark.py: a constant Tokyo-area datum
        # offset is adequate for LoS ray tracing (see io/plateau.py docstring).
        warn_once(
            f"plateau_geoid_correction={config.plateau_geoid_correction!r} failed "
            f"({exc}); falling back to a constant +36.7 m Tokyo-area offset"
        )
        try:
            return load_plateau(config.plateau_dir, zone=config.plateau_zone, geoid_correction=36.7)
        except Exception as exc2:
            warn_once(f"failed to load PLATEAU mesh from {config.plateau_dir}: {exc2}")
            return None


def _extract_edges(building_model, route_ecef: np.ndarray, warn_once):
    """Best-effort UTD/knife-edge diffraction edge extraction.

    The edge extractor lives in ``experiments/utd_edge_features.py`` (outside
    the installed package), mirroring how
    :meth:`gnss_gpu.urban_signal_sim.UrbanSignalSimulator._get_diffraction_edges`
    reaches it. Returns ``None`` (no diffraction candidates) if unavailable.
    """
    triangles = getattr(building_model, "triangles", None)
    if triangles is None or np.asarray(triangles).shape[0] == 0:
        return None
    try:
        import sys

        exp_dir = str(Path(__file__).resolve().parents[2] / "experiments")
        if exp_dir not in sys.path:
            sys.path.insert(0, exp_dir)
        from utd_edge_features import extract_diffraction_edges

        return extract_diffraction_edges(
            triangles, route_ecef=route_ecef, route_margin_m=250.0
        )
    except Exception as exc:
        warn_once(f"diffraction edge extraction unavailable ({exc}); NLOS excess "
                   f"delay will use the fallback constant instead")
        return None


# ---------------------------------------------------------------------------
# Per-epoch satellite geometry / ephemeris
# ---------------------------------------------------------------------------


def _finite_difference_velocity(
    eph: Ephemeris, gps_sow: float, sat_ids: list, positions: np.ndarray
) -> np.ndarray:
    """Satellite ECEF velocity via central finite difference over +-0.5 s."""
    vel = np.zeros_like(positions)
    if not sat_ids:
        return vel

    pos_minus, _, used_minus = eph.compute(gps_sow - 0.5, prn_list=sat_ids)
    pos_plus, _, used_plus = eph.compute(gps_sow + 0.5, prn_list=sat_ids)
    idx_minus = {sid: i for i, sid in enumerate(used_minus)}
    idx_plus = {sid: i for i, sid in enumerate(used_plus)}

    for i, sid in enumerate(sat_ids):
        im = idx_minus.get(sid)
        ip = idx_plus.get(sid)
        if im is None or ip is None:
            continue
        vel[i] = (pos_plus[ip] - pos_minus[im]) / 1.0
    return vel


def _resolve_antenna(antenna: str | AntennaPattern | None) -> AntennaPattern | None:
    """Resolve ``ScenarioConfig.antenna`` into an :class:`AntennaPattern` or ``None``.

    ``None`` is passed straight through (today's isotropic-ish behavior,
    unchanged bit-for-bit); an :class:`AntennaPattern` instance is passed
    straight through too; a string is resolved via
    :meth:`AntennaPattern.preset`.
    """
    if antenna is None or isinstance(antenna, AntennaPattern):
        return antenna
    return AntennaPattern.preset(antenna)


def _pr_noise_sigma_m(el_rad: np.ndarray, sigma_zenith: float, sigma_horizon: float) -> np.ndarray:
    sin_el = np.clip(np.sin(el_rad), 1.0e-3, 1.0)
    sigma = sigma_zenith / sin_el
    cap = max(float(sigma_horizon), float(sigma_zenith))
    return np.minimum(sigma, cap)


def _reflection_model_near(building_model, rx_ecef: np.ndarray, radius_m: float):
    """A small BuildingModel culled to triangles within ``radius_m`` of ``rx_ecef``.

    ``BuildingModel.compute_reflection_paths`` is a pure-Python O(n_tri^2)
    image-method search (every candidate reflection point is occlusion-tested
    against every other triangle); calling it against a full PLATEAU mesh
    (tens of thousands of triangles) is far too slow to run per epoch. Real
    specular reflections only involve nearby facades, so -- mirroring
    ``examples/demo_diffraction_benchmark.py``'s ``refl_cull_radius_m``
    pattern -- we cull to a small local mesh first.
    """
    triangles = getattr(building_model, "triangles", None)
    if triangles is None or triangles.shape[0] == 0:
        return None
    centroids = triangles.mean(axis=1)
    near = triangles[np.linalg.norm(centroids - rx_ecef[None, :], axis=1) < radius_m]
    if near.shape[0] == 0:
        return None

    from gnss_gpu.raytrace import BuildingModel

    return BuildingModel(near)


def _multipath_for_epoch(
    config: ScenarioConfig,
    building_model,
    edges,
    rx_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    is_los: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-satellite (multipath_excess_m, attenuation_db)."""
    n = sat_ecef.shape[0]
    excess = np.zeros(n, dtype=np.float64)
    atten_db = np.zeros(n, dtype=np.float64)
    if building_model is None or n == 0:
        return excess, atten_db

    refl_model = None
    if config.max_reflection_paths > 0:
        refl_model = _reflection_model_near(building_model, rx_ecef, config.reflection_cull_radius_m)

    los_idx = np.where(is_los)[0]
    if los_idx.size and refl_model is not None:
        try:
            paths = refl_model.compute_reflection_paths(
                rx_ecef, sat_ecef[los_idx], max_paths=config.max_reflection_paths
            )
        except Exception:
            paths = None
        if paths is not None:
            for k, sat_i in enumerate(los_idx):
                if paths[k]:
                    excess[sat_i] = float(min(p.excess_delay for p in paths[k]))

    nlos_idx = np.where(~is_los)[0]
    if nlos_idx.size == 0:
        return excess, atten_db

    handled = np.zeros(nlos_idx.size, dtype=bool)

    if config.diffraction_model is not None and edges is not None and config.max_diffraction_paths > 0:
        try:
            if config.diffraction_model == "utd":
                from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths

                dpaths = compute_utd_diffraction_paths(
                    rx_ecef, sat_ecef[nlos_idx], edges,
                    max_paths=config.max_diffraction_paths, mode=config.utd_mode,
                )
            else:
                from gnss_gpu.diffraction import compute_diffraction_paths

                dpaths = compute_diffraction_paths(
                    rx_ecef, sat_ecef[nlos_idx], edges,
                    max_paths=config.max_diffraction_paths,
                )
        except Exception:
            dpaths = None
        if dpaths is not None:
            for k, sat_i in enumerate(nlos_idx):
                if dpaths[k]:
                    best = min(dpaths[k], key=lambda p: p.excess_delay)
                    excess[sat_i] = float(best.excess_delay)
                    atten_db[sat_i] = float(best.attenuation_db)
                    handled[k] = True

    remaining = nlos_idx[~handled]
    if remaining.size and refl_model is not None:
        try:
            paths = refl_model.compute_reflection_paths(
                rx_ecef, sat_ecef[remaining], max_paths=config.max_reflection_paths
            )
        except Exception:
            paths = None
        if paths is not None:
            for k, sat_i in enumerate(remaining):
                if paths[k]:
                    excess[sat_i] = float(min(p.excess_delay for p in paths[k]))
                    atten_db[sat_i] = float(config.nlos_attenuation_db)
                    handled[np.where(nlos_idx == sat_i)[0][0]] = True

    fallback = nlos_idx[~handled]
    excess[fallback] = float(config.nlos_excess_fallback_m)
    atten_db[fallback] = float(config.nlos_attenuation_db)
    return excess, atten_db


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_scenario(config: ScenarioConfig) -> ScenarioResult:
    """Simulate per-epoch GNSS observables for ``config``.

    See :class:`ScenarioConfig` for the accepted inputs and
    :class:`ScenarioResult` / :class:`EpochRecord` for the output schema.
    """
    warned: set[str] = set()

    def warn_once(msg: str) -> None:
        if msg not in warned:
            warned.add(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

    epoch_times = _resolve_epoch_times(config)
    rx_lla_deg = _resolve_receiver_lla(config, epoch_times)

    nav_messages = read_nav_rinex_multi(config.nav_file, systems=config.constellations)
    eph = Ephemeris(nav_messages)

    alpha, beta = read_gps_klobuchar_from_nav_header(config.nav_file)
    atmo = AtmosphereCorrection(iono_alpha=alpha, iono_beta=beta)

    building_model = _load_building_model(config, warn_once)
    if config.plateau_dir is not None and building_model is None:
        warn_once(
            "no PLATEAU mesh available; treating all satellites as open-sky LOS"
        )

    edges = None
    if building_model is not None and config.diffraction_model is not None:
        rx_ecef_all = np.array(
            [_lla_deg_to_ecef(*row) for row in rx_lla_deg], dtype=np.float64
        )
        edges = _extract_edges(building_model, rx_ecef_all, warn_once)

    el_mask_rad = math.radians(float(config.elevation_mask_deg))
    rng = np.random.default_rng(config.seed)
    available_prns = eph.available_prns
    antenna_pattern = _resolve_antenna(config.antenna)

    epochs: list[EpochRecord] = []
    for i, t in enumerate(epoch_times):
        week, sow = _gps_week_sow(t)
        lat_deg, lon_deg, alt_m = rx_lla_deg[i]
        rx_ecef = _lla_deg_to_ecef(lat_deg, lon_deg, alt_m)
        rx_lla_rad = np.array([math.radians(lat_deg), math.radians(lon_deg), alt_m])

        sat_ecef, sat_clk_s, used_prns = eph.compute(sow, prn_list=available_prns)
        if not used_prns:
            epochs.append(_empty_epoch(week, sow, t, rx_ecef))
            continue

        el, az = elevation_azimuth(rx_ecef, sat_ecef)
        mask = el >= el_mask_rad
        if not np.any(mask):
            epochs.append(_empty_epoch(week, sow, t, rx_ecef))
            continue

        mask_idx = np.where(mask)[0]
        sat_id = np.array([used_prns[j] for j in mask_idx], dtype="<U3")
        sat_ecef_m = sat_ecef[mask_idx]
        sat_clk_m = sat_clk_s[mask_idx] * C_LIGHT
        el_m = el[mask_idx]
        az_m = az[mask_idx]
        range_geom = np.linalg.norm(sat_ecef_m - rx_ecef[None, :], axis=1)

        tropo_m = np.asarray(atmo.tropo(rx_lla_rad, el_m), dtype=np.float64)
        iono_m = np.asarray(atmo.iono(rx_lla_rad, az_m, el_m, sow), dtype=np.float64)

        is_los = np.ones(sat_id.size, dtype=bool)
        if building_model is not None:
            try:
                is_los = np.asarray(building_model.check_los(rx_ecef, sat_ecef_m), dtype=bool)
            except Exception as exc:
                warn_once(
                    f"LOS check unavailable ({exc}); treating all satellites as LOS"
                )
                is_los = np.ones(sat_id.size, dtype=bool)

        multipath_excess_m, atten_db = _multipath_for_epoch(
            config, building_model, edges, rx_ecef, sat_ecef_m, is_los
        )

        sigma = _pr_noise_sigma_m(
            el_m, config.pr_noise_sigma_zenith_m, config.pr_noise_sigma_horizon_m
        )
        noise = np.where(sigma > 0.0, rng.normal(0.0, np.maximum(sigma, 1e-300)), 0.0)

        pseudorange_m = (
            range_geom
            + float(config.rx_clock_bias_m)
            - sat_clk_m
            + iono_m
            + tropo_m
            + multipath_excess_m
            + noise
        )

        cn0_dbhz = config.cn0_zenith_dbhz * np.sin(el_m) - atten_db
        if antenna_pattern is not None:
            cn0_dbhz = cn0_dbhz + antenna_pattern.gain_db(el_m, az_m)
        cn0_dbhz = np.maximum(cn0_dbhz, 0.0)

        sat_vel = _finite_difference_velocity(eph, sow, list(sat_id), sat_ecef_m)
        los_unit = (sat_ecef_m - rx_ecef[None, :]) / range_geom[:, None]
        rel_vel = np.einsum("ij,ij->i", sat_vel, los_unit)
        doppler_hz = -rel_vel / L1_WAVELENGTH_M

        epochs.append(
            EpochRecord(
                time_gps_week=week,
                time_sow=sow,
                time_utc=t,
                rx_ecef=rx_ecef,
                sat_id=sat_id,
                elevation_rad=el_m,
                azimuth_rad=az_m,
                is_los=is_los,
                pseudorange_m=pseudorange_m,
                range_geometric_m=range_geom,
                sat_clock_bias_m=sat_clk_m,
                iono_m=iono_m,
                tropo_m=tropo_m,
                multipath_excess_m=multipath_excess_m,
                cn0_dbhz=cn0_dbhz,
                doppler_hz=doppler_hz,
            )
        )

    return ScenarioResult(epochs=epochs, config=config)


def _empty_epoch(week: int, sow: float, t: datetime, rx_ecef: np.ndarray) -> EpochRecord:
    def z_f() -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    return EpochRecord(
        time_gps_week=week,
        time_sow=sow,
        time_utc=t,
        rx_ecef=rx_ecef,
        sat_id=np.zeros(0, dtype="<U3"),
        elevation_rad=z_f(),
        azimuth_rad=z_f(),
        is_los=np.zeros(0, dtype=bool),
        pseudorange_m=z_f(),
        range_geometric_m=z_f(),
        sat_clock_bias_m=z_f(),
        iono_m=z_f(),
        tropo_m=z_f(),
        multipath_excess_m=z_f(),
        cn0_dbhz=z_f(),
        doppler_hz=z_f(),
    )


__all__ = [
    "ScenarioConfig",
    "EpochRecord",
    "ScenarioResult",
    "run_scenario",
]
