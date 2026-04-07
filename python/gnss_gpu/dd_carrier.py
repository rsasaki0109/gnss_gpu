"""Double-Differenced (DD) carrier phase computation.

DD eliminates receiver clock bias (both rover and base) and substantially
reduces atmospheric (ionospheric + tropospheric) errors, making the
carrier phase AFV likelihood much more effective for the MUPF algorithm.

Algorithm
---------
For each epoch, given common satellites between rover and base:

1. Pick a reference satellite (highest elevation = most likely LOS).
2. For each non-reference common satellite *k*:

   DD_carrier[k] = (rover_L1[k] - rover_L1[ref]) - (base_L1[k] - base_L1[ref])

The DD observation eliminates receiver clock bias from both rover and base,
and largely cancels spatially-correlated atmospheric delays (since base and
rover see similar ionosphere/troposphere for the same satellite).

DD-AFV residual per particle:
   dd_expected = (range_rover_k - range_rover_ref - range_base_k + range_base_ref) / wavelength
   dd_residual = DD_carrier[k] - dd_expected
   afv = dd_residual - round(dd_residual)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnss_gpu.io.rinex import read_rinex_obs

# GPS L1 wavelength [m]
GPS_L1_WAVELENGTH = 0.190293673


@dataclass
class DDResult:
    """Result of double-differenced carrier phase computation for one epoch."""

    dd_carrier_cycles: np.ndarray  # [n_dd] DD carrier phase observations in cycles
    sat_ecef_k: np.ndarray  # [n_dd, 3] ECEF positions of non-ref satellites
    sat_ecef_ref: np.ndarray  # [3] ECEF position of reference satellite
    base_range_k: np.ndarray  # [n_dd] base-to-sat_k geometric ranges [m]
    base_range_ref: float  # base-to-ref geometric range [m]
    dd_weights: np.ndarray  # [n_dd] per-DD-pair weights
    ref_sat_id: str  # reference satellite identifier
    n_dd: int  # number of DD pairs


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
        base_position: np.ndarray | None = None,
        carrier_obs_code: str = "L1C",
    ):
        base_obs_path = Path(base_obs_path)
        if not base_obs_path.exists():
            raise FileNotFoundError(f"Base RINEX not found: {base_obs_path}")

        self._obs = read_rinex_obs(base_obs_path)
        self._carrier_code = carrier_obs_code

        if base_position is not None:
            self._base_pos = np.asarray(base_position, dtype=np.float64).ravel()
        else:
            self._base_pos = self._obs.header.approx_position.copy()

        if np.linalg.norm(self._base_pos) < 1e6:
            raise ValueError(
                f"Base position looks invalid (norm={np.linalg.norm(self._base_pos):.1f}m): "
                f"{self._base_pos}"
            )

        # Index base observations by rounded TOW for fast lookup.
        # Convert epoch datetimes to GPS TOW (seconds of week).
        self._base_by_tow: dict[float, dict[str, float]] = {}
        for ep in self._obs.epochs:
            # Compute TOW from datetime: day-of-week * 86400 + seconds-of-day
            dow = ep.time.weekday()  # Monday=0 ... Sunday=6
            # GPS week starts on Sunday, so adjust:
            gps_dow = (dow + 1) % 7
            sod = ep.time.hour * 3600 + ep.time.minute * 60 + ep.time.second + ep.time.microsecond * 1e-6
            tow = gps_dow * 86400.0 + sod
            tow_key = round(tow, 1)

            carrier_obs: dict[str, float] = {}
            for sat_id, obs in ep.observations.items():
                if not sat_id.startswith("G"):
                    continue  # GPS only for now
                cp = obs.get(carrier_obs_code, 0.0)
                if cp != 0.0 and abs(cp) > 1e3:
                    carrier_obs[sat_id] = cp

            if carrier_obs:
                self._base_by_tow[tow_key] = carrier_obs

        print(
            f"  [DD] Loaded base station: {len(self._base_by_tow)} epochs with "
            f"{carrier_obs_code} carrier phase"
        )

    @property
    def base_position(self) -> np.ndarray:
        """Base station ECEF position [m]."""
        return self._base_pos

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
        # Match base epoch by TOW
        tow_key = round(tow, 1)
        base_obs = self._base_by_tow.get(tow_key)
        if base_obs is None:
            # Try +/- 0.1s
            for offset in (0.1, -0.1, 0.2, -0.2):
                base_obs = self._base_by_tow.get(round(tow + offset, 1))
                if base_obs is not None:
                    break
        if base_obs is None:
            return None

        # Collect rover carrier phase + satellite info, indexed by sat_id
        rover_cp: dict[str, float] = {}
        rover_sat_ecef: dict[str, np.ndarray] = {}
        rover_elev: dict[str, float] = {}
        rover_snr: dict[str, float] = {}

        _SYS_MAP = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}

        for m in rover_measurements:
            # Derive GPS satellite ID (e.g., "G01")
            system_id = int(getattr(m, "system_id", 0))
            prn = int(getattr(m, "prn", 0))
            sys_char = _SYS_MAP.get(system_id, "G")
            if sys_char != "G":  # GPS only for DD with GPS base
                continue
            sat_id = f"G{prn:02d}"

            cp = float(getattr(m, "carrier_phase", 0.0))
            if cp == 0.0 or not np.isfinite(cp) or abs(cp) < 1e3:
                continue

            sat_pos = np.asarray(m.satellite_ecef, dtype=np.float64)
            if not np.all(np.isfinite(sat_pos)):
                continue

            rover_cp[sat_id] = cp
            rover_sat_ecef[sat_id] = sat_pos
            rover_elev[sat_id] = float(getattr(m, "elevation", 0.0))
            rover_snr[sat_id] = float(getattr(m, "snr", 0.0))

        # Find common satellites
        common_sats = sorted(set(rover_cp.keys()) & set(base_obs.keys()))
        if len(common_sats) < min_common_sats:
            return None

        # Pick reference satellite: highest elevation (most likely LOS)
        ref_sat = max(common_sats, key=lambda s: rover_elev.get(s, 0.0))

        # Compute DD for each non-ref satellite
        non_ref = [s for s in common_sats if s != ref_sat]
        if len(non_ref) < 1:
            return None

        rover_cp_ref = rover_cp[ref_sat]
        base_cp_ref = base_obs[ref_sat]
        ref_ecef = rover_sat_ecef[ref_sat]

        # Base-to-reference geometric range
        base_range_ref = float(np.linalg.norm(ref_ecef - self._base_pos))

        dd_carrier_list = []
        sat_ecef_k_list = []
        base_range_k_list = []
        dd_weight_list = []

        for sat_id in non_ref:
            rover_cp_k = rover_cp[sat_id]
            base_cp_k = base_obs[sat_id]
            sat_k_ecef = rover_sat_ecef[sat_id]

            # DD carrier phase [cycles]
            dd = (rover_cp_k - rover_cp_ref) - (base_cp_k - base_cp_ref)

            # Base-to-satellite_k geometric range
            base_range_k = float(np.linalg.norm(sat_k_ecef - self._base_pos))

            # Weight based on elevation (sin(elev) for both sats)
            elev_k = rover_elev.get(sat_id, 0.3)
            elev_ref = rover_elev.get(ref_sat, 0.3)
            w = min(np.sin(max(elev_k, 0.05)), np.sin(max(elev_ref, 0.05)))

            dd_carrier_list.append(dd)
            sat_ecef_k_list.append(sat_k_ecef)
            base_range_k_list.append(base_range_k)
            dd_weight_list.append(w)

        n_dd = len(dd_carrier_list)
        return DDResult(
            dd_carrier_cycles=np.array(dd_carrier_list, dtype=np.float64),
            sat_ecef_k=np.array(sat_ecef_k_list, dtype=np.float64).reshape(n_dd, 3),
            sat_ecef_ref=ref_ecef.copy(),
            base_range_k=np.array(base_range_k_list, dtype=np.float64),
            base_range_ref=base_range_ref,
            dd_weights=np.array(dd_weight_list, dtype=np.float64),
            ref_sat_id=ref_sat,
            n_dd=n_dd,
        )
