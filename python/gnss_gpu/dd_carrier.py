"""Double-Differenced (DD) carrier phase computation.

DD eliminates receiver clock bias (both rover and base) and substantially
reduces atmospheric (ionospheric + tropospheric) errors, making the
carrier phase AFV likelihood much more effective for the MUPF algorithm.

Algorithm
---------
For each epoch, given common satellites between rover and base:

1. For each supported constellation, pick a reference satellite
   (highest elevation = most likely LOS).
2. For each non-reference common satellite *k* in that constellation:

   DD_carrier[k] = (rover_L1[k] - rover_L1[ref]) - (base_L1[k] - base_L1[ref])

The DD observation eliminates receiver clock bias from both rover and base,
and largely cancels spatially-correlated atmospheric delays (since base and
rover see similar ionosphere/troposphere for the same satellite). This module
supports per-system wavelengths for GPS/Galileo/QZSS/BeiDou. GLONASS is
currently skipped because FDMA slot-dependent wavelengths are not modeled yet.

DD-AFV residual per particle:
   dd_expected = (range_rover_k - range_rover_ref - range_base_k + range_base_ref) / wavelength
   dd_residual = DD_carrier[k] - dd_expected
   afv = dd_residual - round(dd_residual)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gnss_gpu.io.rinex import read_rinex_obs

C_LIGHT = 299792458.0
GPS_L1_WAVELENGTH = C_LIGHT / 1575.42e6
GALILEO_E1_WAVELENGTH = C_LIGHT / 1575.42e6
QZSS_L1_WAVELENGTH = C_LIGHT / 1575.42e6
BEIDOU_B1I_WAVELENGTH = C_LIGHT / 1561.098e6

_SYS_MAP = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}
_SYSTEM_WAVELENGTHS = {
    "G": GPS_L1_WAVELENGTH,
    "E": GALILEO_E1_WAVELENGTH,
    "J": QZSS_L1_WAVELENGTH,
    "C": BEIDOU_B1I_WAVELENGTH,
}
_CARRIER_CODE_PREFERENCES = {
    "G": ("L1C", "L1W", "L1X", "L1P", "L1S", "L1L", "L1Z"),
    "E": ("L1X", "L1C", "L1A", "L1B", "L1Z"),
    "J": ("L1C", "L1X", "L1Z", "L1S", "L1L"),
    "C": ("L1I", "L1D", "L1X", "L1P"),
}


@dataclass
class DDResult:
    """Result of double-differenced carrier phase computation for one epoch."""

    dd_carrier_cycles: np.ndarray  # [n_dd] DD carrier phase observations in cycles
    sat_ecef_k: np.ndarray  # [n_dd, 3] ECEF positions of non-ref satellites
    sat_ecef_ref: np.ndarray  # [n_dd, 3] ECEF positions of reference satellites
    base_range_k: np.ndarray  # [n_dd] base-to-sat_k geometric ranges [m]
    base_range_ref: np.ndarray  # [n_dd] base-to-ref geometric ranges [m]
    dd_weights: np.ndarray  # [n_dd] per-DD-pair weights
    wavelengths_m: np.ndarray  # [n_dd] carrier wavelengths [m]
    ref_sat_ids: tuple[str, ...]  # [n_dd] reference satellite IDs per pair
    n_dd: int  # number of DD pairs
    sat_ids: tuple[str, ...] = ()  # [n_dd] non-reference satellite IDs per pair


def _datetime_to_tow(epoch_time) -> float:
    dow = epoch_time.weekday()
    gps_dow = (dow + 1) % 7
    sod = (
        epoch_time.hour * 3600
        + epoch_time.minute * 60
        + epoch_time.second
        + epoch_time.microsecond * 1e-6
    )
    return gps_dow * 86400.0 + sod


def _meas_key(m: Any) -> tuple[int, int]:
    return (int(getattr(m, "system_id", 0)), int(getattr(m, "prn", 0)))


def _normalize_sat_id(sat_id: str) -> str:
    sat_id = sat_id.strip()
    if not sat_id:
        return sat_id
    sys_char = sat_id[0]
    prn_str = sat_id[1:].strip()
    if not prn_str:
        return sat_id
    try:
        return f"{sys_char}{int(prn_str):02d}"
    except ValueError:
        return sat_id


def _one_row_per_satellite(measurements: Sequence[Any]) -> dict[tuple[int, int], Any]:
    by_key: dict[tuple[int, int], list[Any]] = {}
    for m in measurements:
        by_key.setdefault(_meas_key(m), []).append(m)

    out: dict[tuple[int, int], Any] = {}
    eps = 0.05
    for k, rows in by_key.items():
        if len(rows) == 1:
            out[k] = rows[0]
            continue
        rows_sorted = sorted(rows, key=lambda m: float(getattr(m, "snr", 0.0)), reverse=True)
        best_snr = float(getattr(rows_sorted[0], "snr", 0.0))
        n_top = sum(
            1 for m in rows_sorted if abs(float(getattr(m, "snr", 0.0)) - best_snr) <= eps
        )
        if n_top != 1:
            continue
        out[k] = rows_sorted[0]
    return out


def _valid_carrier_obs(obs: dict[str, float]) -> dict[str, float]:
    valid: dict[str, float] = {}
    for code, value in obs.items():
        if not code.startswith("L"):
            continue
        val = float(value)
        if np.isfinite(val) and abs(val) > 1e3:
            valid[code] = val
    return valid


def _ordered_obs_codes(
    sys_char: str,
    system_sats: Sequence[str],
    rover_obs: dict[str, dict[str, float]],
    base_obs: dict[str, dict[str, float]],
    preferred_code: str | None,
    primary_prefix: str,
) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()

    def _append(code: str) -> None:
        if code and code not in seen:
            seen.add(code)
            codes.append(code)

    if preferred_code is not None:
        _append(preferred_code)
    for code in _CARRIER_CODE_PREFERENCES.get(sys_char, ()):
        _append(code)

    extras = set()
    for sat_id in system_sats:
        extras.update(rover_obs.get(sat_id, {}).keys())
        extras.update(base_obs.get(sat_id, {}).keys())
    for code in sorted(c for c in extras if c.startswith(primary_prefix)):
        _append(code)
    for code in sorted(c for c in extras if c.startswith(primary_prefix[:1])):
        _append(code)
    return codes


def _select_common_obs_code(
    sys_char: str,
    system_sats: Sequence[str],
    rover_obs: dict[str, dict[str, float]],
    base_obs: dict[str, dict[str, float]],
    preferred_code: str | None,
) -> tuple[str | None, list[str]]:
    best_code: str | None = None
    best_sats: list[str] = []
    for code in _ordered_obs_codes(
        sys_char,
        system_sats,
        rover_obs,
        base_obs,
        preferred_code,
        primary_prefix="L1",
    ):
        sats = [
            sat_id
            for sat_id in system_sats
            if code in rover_obs.get(sat_id, {}) and code in base_obs.get(sat_id, {})
        ]
        if len(sats) > len(best_sats):
            best_code = code
            best_sats = sats
    return best_code, best_sats


def _pick_single_obs_value(
    sys_char: str,
    sat_obs: dict[str, float],
    preferred_code: str | None,
) -> float:
    fake_sat = f"{sys_char}00"
    for code in _ordered_obs_codes(
        sys_char,
        (fake_sat,),
        {fake_sat: sat_obs},
        {fake_sat: sat_obs},
        preferred_code,
        primary_prefix="L1",
    ):
        if code in sat_obs:
            return float(sat_obs[code])
    return 0.0


class DDCarrierComputer:
    """Compute Double-Differenced carrier phase observations.

    Loads base station RINEX observations and indexes them by TOW for
    fast lookup during epoch-by-epoch processing.

    Parameters
    ----------
    base_obs_path : str or Path
        Path to base station RINEX 3.x observation file.
    base_position : array_like, shape (3,)
        Base station ECEF position [m]. If None, read from RINEX header.
    carrier_obs_code : str
        RINEX observation code for L1 carrier phase. Default ``"L1C"``.
    """

    def __init__(
        self,
        base_obs_path: str | Path,
        rover_obs_path: str | Path | None = None,
        base_position: np.ndarray | None = None,
        carrier_obs_code: str | None = None,
        allowed_systems: Sequence[str] = ("G", "E", "J", "C"),
        interpolate_base_epochs: bool = False,
    ):
        base_obs_path = Path(base_obs_path)
        if not base_obs_path.exists():
            raise FileNotFoundError(f"Base RINEX not found: {base_obs_path}")

        self._obs = read_rinex_obs(base_obs_path)
        self._carrier_code = carrier_obs_code
        self._allowed_systems = tuple(allowed_systems)
        self._interpolate_base_epochs = bool(interpolate_base_epochs)

        if base_position is not None:
            self._base_pos = np.asarray(base_position, dtype=np.float64).ravel()
        else:
            self._base_pos = self._obs.header.approx_position.copy()

        if np.linalg.norm(self._base_pos) < 1e6:
            raise ValueError(
                f"Base position looks invalid (norm={np.linalg.norm(self._base_pos):.1f}m): "
                f"{self._base_pos}"
            )

        self._base_by_tow: dict[float, dict[str, dict[str, float]]] = {}
        for ep in self._obs.epochs:
            tow_key = round(_datetime_to_tow(ep.time), 1)

            carrier_obs: dict[str, dict[str, float]] = {}
            for sat_id, obs in ep.observations.items():
                sat_id_norm = _normalize_sat_id(sat_id)
                if not sat_id_norm or sat_id_norm[0] not in self._allowed_systems:
                    continue
                if sat_id_norm[0] not in _SYSTEM_WAVELENGTHS:
                    continue
                valid_obs = _valid_carrier_obs(obs)
                if valid_obs:
                    carrier_obs[sat_id_norm] = valid_obs

            if carrier_obs:
                self._base_by_tow[tow_key] = carrier_obs
        self._base_tow_keys = np.array(sorted(self._base_by_tow.keys()), dtype=np.float64)

        self._rover_by_tow: dict[float, dict[str, dict[str, float]]] | None = None
        if rover_obs_path is not None:
            rover_obs_path = Path(rover_obs_path)
            if not rover_obs_path.exists():
                raise FileNotFoundError(f"Rover RINEX not found: {rover_obs_path}")
            rover_obs = read_rinex_obs(rover_obs_path)
            self._rover_by_tow = {}
            for ep in rover_obs.epochs:
                tow_key = round(_datetime_to_tow(ep.time), 1)
                carrier_obs: dict[str, dict[str, float]] = {}
                for sat_id, obs in ep.observations.items():
                    sat_id_norm = _normalize_sat_id(sat_id)
                    if not sat_id_norm or sat_id_norm[0] not in self._allowed_systems:
                        continue
                    if sat_id_norm[0] not in _SYSTEM_WAVELENGTHS:
                        continue
                    valid_obs = _valid_carrier_obs(obs)
                    if valid_obs:
                        carrier_obs[sat_id_norm] = valid_obs
                if carrier_obs:
                    self._rover_by_tow[tow_key] = carrier_obs

        print(
            f"  [DD] Loaded base station: {len(self._base_by_tow)} epochs with "
            f"{self._carrier_code or 'system-preferred L1/E1/B1'} carrier phase"
        )

    @property
    def base_position(self) -> np.ndarray:
        """Base station ECEF position [m]."""
        return self._base_pos

    def _interpolate_base_obs(self, tow_key: float) -> dict[str, dict[str, float]] | None:
        if self._base_tow_keys.size < 2:
            return None

        idx = int(np.searchsorted(self._base_tow_keys, tow_key))
        if idx <= 0 or idx >= self._base_tow_keys.size:
            return None

        t0 = float(self._base_tow_keys[idx - 1])
        t1 = float(self._base_tow_keys[idx])
        if not (t0 < tow_key < t1):
            return None
        if (t1 - t0) > 1.5:
            return None

        alpha = (tow_key - t0) / (t1 - t0)
        obs0 = self._base_by_tow.get(t0)
        obs1 = self._base_by_tow.get(t1)
        if obs0 is None or obs1 is None:
            return None

        interp: dict[str, dict[str, float]] = {}
        for sat_id in set(obs0.keys()) & set(obs1.keys()):
            obs_interp: dict[str, float] = {}
            for code in set(obs0[sat_id].keys()) & set(obs1[sat_id].keys()):
                obs_interp[code] = (1.0 - alpha) * obs0[sat_id][code] + alpha * obs1[sat_id][code]
            if obs_interp:
                interp[sat_id] = obs_interp
        return interp or None

    def compute_dd(
        self,
        tow: float,
        rover_measurements,
        rover_position_approx: np.ndarray | None = None,
        min_common_sats: int = 4,
    ) -> DDResult | None:
        """Compute DD carrier phase for current epoch.

        Parameters
        ----------
        tow : float
            GPS Time of Week [s] for the current epoch.
        rover_measurements : list
            gnssplusplus measurement objects with attributes:
            ``satellite_ecef``, ``carrier_phase``, ``elevation``, ``snr``,
            and ``prn`` or satellite ID derivable from ``system``/``prn``.
        rover_position_approx : array_like, shape (3,), optional
            Approximate rover position for elevation computation (not critical).
        min_common_sats : int
            Minimum common satellites needed (including ref). Default 4.

        Returns
        -------
        DDResult or None
            DD carrier phase data, or None if insufficient common satellites.
        """
        tow_key = round(tow, 1)
        rover_obs = None
        if self._rover_by_tow is not None:
            base_obs = self._base_by_tow.get(tow_key)
            if base_obs is None and self._interpolate_base_epochs:
                base_obs = self._interpolate_base_obs(tow_key)
            rover_obs = self._rover_by_tow.get(tow_key)
            if base_obs is None or rover_obs is None:
                return None
        else:
            base_obs = self._base_by_tow.get(tow_key)
            if base_obs is None:
                # Legacy path: corrected rover measurements only, so keep loose matching.
                for offset in (0.1, -0.1, 0.2, -0.2):
                    base_obs = self._base_by_tow.get(round(tow + offset, 1))
                    if base_obs is not None:
                        break
            if base_obs is None:
                return None

        rover_cp: dict[str, float] = {}
        base_cp: dict[str, float] = {}
        rover_sat_ecef: dict[str, np.ndarray] = {}
        rover_elev: dict[str, float] = {}
        rover_wavelength: dict[str, float] = {}

        rover_rows = _one_row_per_satellite(rover_measurements)
        for (system_id, prn), m in rover_rows.items():
            sys_char = _SYS_MAP.get(system_id, "G")
            if sys_char not in self._allowed_systems:
                continue
            wavelength = _SYSTEM_WAVELENGTHS.get(sys_char)
            if wavelength is None:
                continue
            sat_id = f"{sys_char}{prn:02d}"

            if rover_obs is None:
                cp = float(getattr(m, "carrier_phase", 0.0))
                if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
                    continue

            sat_pos = np.asarray(m.satellite_ecef, dtype=np.float64).ravel()[:3]
            if sat_pos.size != 3 or not np.all(np.isfinite(sat_pos)):
                continue

            if rover_obs is None:
                rover_cp[sat_id] = cp
            rover_sat_ecef[sat_id] = sat_pos
            rover_elev[sat_id] = float(getattr(m, "elevation", 0.0))
            rover_wavelength[sat_id] = wavelength

        if rover_obs is not None:
            sats_by_system: dict[str, list[str]] = {}
            for sat_id in rover_sat_ecef:
                if sat_id in rover_obs and sat_id in base_obs:
                    sats_by_system.setdefault(sat_id[0], []).append(sat_id)
            for sys_char, sys_sats in sats_by_system.items():
                _code, selected_sats = _select_common_obs_code(
                    sys_char,
                    sys_sats,
                    rover_obs,
                    base_obs,
                    self._carrier_code,
                )
                if len(selected_sats) < 2:
                    continue
                for sat_id in selected_sats:
                    rover_cp[sat_id] = float(rover_obs[sat_id][_code])
                    base_cp[sat_id] = float(base_obs[sat_id][_code])
        else:
            for sat_id, sat_obs in base_obs.items():
                cp = _pick_single_obs_value(sat_id[0], sat_obs, self._carrier_code)
                if cp != 0.0:
                    base_cp[sat_id] = cp

        # Find common satellites
        common_sats = sorted(set(rover_cp.keys()) & set(base_cp.keys()))
        if len(common_sats) < min_common_sats:
            return None

        dd_carrier_list = []
        sat_ecef_k_list = []
        sat_ecef_ref_list = []
        base_range_k_list = []
        base_range_ref_list = []
        dd_weight_list = []
        wavelengths_m_list = []
        ref_sat_ids = []
        sat_ids = []

        sats_by_system: dict[str, list[str]] = {}
        for sat_id in common_sats:
            sats_by_system.setdefault(sat_id[0], []).append(sat_id)

        for sys_char, sys_sats in sorted(sats_by_system.items()):
            if len(sys_sats) < 2:
                continue

            ref_sat = max(sys_sats, key=lambda s: rover_elev.get(s, 0.0))
            non_ref = [s for s in sys_sats if s != ref_sat]
            if not non_ref:
                continue

            rover_cp_ref = rover_cp[ref_sat]
            base_cp_ref = base_cp[ref_sat]
            ref_ecef = rover_sat_ecef[ref_sat]
            base_range_ref = float(np.linalg.norm(ref_ecef - self._base_pos))
            wavelength = rover_wavelength[ref_sat]
            elev_ref = rover_elev.get(ref_sat, 0.3)

            for sat_id in non_ref:
                rover_cp_k = rover_cp[sat_id]
                base_cp_k = base_cp[sat_id]
                sat_k_ecef = rover_sat_ecef[sat_id]
                base_range_k = float(np.linalg.norm(sat_k_ecef - self._base_pos))

                dd = (rover_cp_k - rover_cp_ref) - (base_cp_k - base_cp_ref)

                elev_k = rover_elev.get(sat_id, 0.3)
                w = min(np.sin(max(elev_k, 0.05)), np.sin(max(elev_ref, 0.05)))

                dd_carrier_list.append(dd)
                sat_ecef_k_list.append(sat_k_ecef)
                sat_ecef_ref_list.append(ref_ecef.copy())
                base_range_k_list.append(base_range_k)
                base_range_ref_list.append(base_range_ref)
                dd_weight_list.append(w)
                wavelengths_m_list.append(wavelength)
                ref_sat_ids.append(ref_sat)
                sat_ids.append(sat_id)

        if not dd_carrier_list:
            return None

        n_dd = len(dd_carrier_list)
        return DDResult(
            dd_carrier_cycles=np.array(dd_carrier_list, dtype=np.float64),
            sat_ecef_k=np.array(sat_ecef_k_list, dtype=np.float64).reshape(n_dd, 3),
            sat_ecef_ref=np.array(sat_ecef_ref_list, dtype=np.float64).reshape(n_dd, 3),
            base_range_k=np.array(base_range_k_list, dtype=np.float64),
            base_range_ref=np.array(base_range_ref_list, dtype=np.float64),
            dd_weights=np.array(dd_weight_list, dtype=np.float64),
            wavelengths_m=np.array(wavelengths_m_list, dtype=np.float64),
            ref_sat_ids=tuple(ref_sat_ids),
            n_dd=n_dd,
            sat_ids=tuple(sat_ids),
        )
