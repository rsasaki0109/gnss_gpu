"""Double-Differenced (DD) pseudorange computation.

DD eliminates receiver clock bias from both rover and base and greatly reduces
spatially correlated atmospheric errors on short baselines.

For each epoch, given common satellites between rover and base:

1. Pick a reference satellite (highest elevation = most likely LOS).
2. For each non-reference common satellite ``k``:

   DD_pr[k] = (rover_PR[k] - rover_PR[ref]) - (base_PR[k] - base_PR[ref])

The expected DD range for a particle at position ``x`` is:

   dd_expected = (range_rover_k - range_rover_ref) - (range_base_k - range_base_ref)

which is equivalent to:

   dd_expected = range_rover_k - range_rover_ref - range_base_k + range_base_ref
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gnss_gpu.io.rinex import read_rinex_obs

_SYS_MAP = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}
_PSEUDORANGE_CODE_PREFERENCES = {
    "G": ("C1C", "C1W", "C1X", "C1P", "C1S", "C1L", "C1Z"),
    "E": ("C1X", "C1C", "C1A", "C1B", "C1Z"),
    "J": ("C1C", "C1X", "C1Z", "C1S", "C1L"),
    "C": ("C1I", "C1D", "C1X", "C1P"),
    "R": ("C1C", "C1P"),
}


@dataclass
class DDPseudorangeResult:
    """Result of double-differenced pseudorange computation for one epoch."""

    dd_pseudorange_m: np.ndarray  # [n_dd] DD pseudorange observations [m]
    sat_ecef_k: np.ndarray  # [n_dd, 3] ECEF positions of non-ref satellites
    sat_ecef_ref: np.ndarray  # [n_dd, 3] ECEF positions of reference satellites per DD pair
    base_range_k: np.ndarray  # [n_dd] base-to-sat_k geometric ranges [m]
    base_range_ref: np.ndarray  # [n_dd] base-to-ref geometric ranges [m]
    dd_weights: np.ndarray  # [n_dd] per-DD-pair weights
    ref_sat_ids: tuple[str, ...]  # [n_dd] reference satellite IDs per DD pair
    n_dd: int  # number of DD pairs


def _datetime_to_tow(epoch_time) -> float:
    dow = epoch_time.weekday()  # Monday=0 ... Sunday=6
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


def _one_row_per_satellite(
    measurements: Sequence[Any], rover_weights: Sequence[float] | None = None
) -> dict[tuple[int, int], tuple[Any, float]]:
    """Pick one rover row per (system, prn), preferring the highest-SNR row.

    gnssplusplus can expose multiple rows per satellite (e.g. multiple signals).
    Prefer the highest-SNR row. If several rows tie for best SNR, omit that
    satellite to avoid mixing frequencies.
    """

    by_key: dict[tuple[int, int], list[tuple[Any, float]]] = {}
    for i, m in enumerate(measurements):
        k = _meas_key(m)
        w = (
            float(rover_weights[i])
            if rover_weights is not None and i < len(rover_weights)
            else float(getattr(m, "weight", 1.0))
        )
        by_key.setdefault(k, []).append((m, w))

    out: dict[tuple[int, int], tuple[Any, float]] = {}
    eps = 0.05
    for k, rows in by_key.items():
        if len(rows) == 1:
            out[k] = rows[0]
            continue
        rows_sorted = sorted(rows, key=lambda p: float(getattr(p[0], "snr", 0.0)), reverse=True)
        best_snr = float(getattr(rows_sorted[0][0], "snr", 0.0))
        n_top = sum(
            1
            for m, _w in rows_sorted
            if abs(float(getattr(m, "snr", 0.0)) - best_snr) <= eps
        )
        if n_top != 1:
            continue
        out[k] = rows_sorted[0]
    return out


def _valid_pseudorange_obs(obs: dict[str, float]) -> dict[str, float]:
    valid: dict[str, float] = {}
    for code, value in obs.items():
        if not code.startswith("C"):
            continue
        val = float(value)
        if np.isfinite(val) and abs(val) > 1e6:
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
    for code in _PSEUDORANGE_CODE_PREFERENCES.get(sys_char, ()):
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
        primary_prefix="C1",
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
        primary_prefix="C1",
    ):
        if code in sat_obs:
            return float(sat_obs[code])
    return 0.0


class DDPseudorangeComputer:
    """Compute Double-Differenced pseudorange observations.

    Parameters
    ----------
    base_obs_path : str or Path
        Path to base station RINEX observation file.
    base_position : array_like, shape (3,), optional
        Base station ECEF position [m]. If omitted, use the RINEX header.
    pseudorange_obs_code : str
        Preferred RINEX observation code. Default ``"C1C"``.
    allowed_systems : sequence of str
        Constellations to use. Default is GPS only, matching the current
        DD carrier AFV path.
    """

    def __init__(
        self,
        base_obs_path: str | Path,
        rover_obs_path: str | Path | None = None,
        base_position: np.ndarray | None = None,
        pseudorange_obs_code: str | None = None,
        allowed_systems: Sequence[str] = ("G", "E", "J", "C", "R"),
        interpolate_base_epochs: bool = False,
    ):
        base_obs_path = Path(base_obs_path)
        if not base_obs_path.exists():
            raise FileNotFoundError(f"Base RINEX not found: {base_obs_path}")

        self._obs = read_rinex_obs(base_obs_path)
        self._pseudorange_code = pseudorange_obs_code
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
            pr_obs: dict[str, dict[str, float]] = {}
            for sat_id, obs in ep.observations.items():
                sat_id_norm = _normalize_sat_id(sat_id)
                if not sat_id_norm or sat_id_norm[0] not in self._allowed_systems:
                    continue
                valid_obs = _valid_pseudorange_obs(obs)
                if valid_obs:
                    pr_obs[sat_id_norm] = valid_obs
            if pr_obs:
                self._base_by_tow[tow_key] = pr_obs
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
                pr_obs: dict[str, dict[str, float]] = {}
                for sat_id, obs in ep.observations.items():
                    sat_id_norm = _normalize_sat_id(sat_id)
                    if not sat_id_norm or sat_id_norm[0] not in self._allowed_systems:
                        continue
                    valid_obs = _valid_pseudorange_obs(obs)
                    if valid_obs:
                        pr_obs[sat_id_norm] = valid_obs
                if pr_obs:
                    self._rover_by_tow[tow_key] = pr_obs

        print(
            f"  [DD] Loaded base station: {len(self._base_by_tow)} epochs with "
            f"{self._pseudorange_code or 'system-preferred C1/E1/B1'} pseudorange"
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
        rover_measurements: Sequence[Any],
        rover_position_approx: np.ndarray | None = None,
        min_common_sats: int = 4,
        rover_weights: Sequence[float] | None = None,
    ) -> DDPseudorangeResult | None:
        """Compute DD pseudorange for one epoch.

        Parameters
        ----------
        tow : float
            GPS Time of Week [s] for the current epoch.
        rover_measurements : sequence
            gnssplusplus measurement rows exposing ``corrected_pseudorange``,
            ``satellite_ecef``, ``elevation``, ``snr``, ``system_id`` and ``prn``.
        rover_position_approx : ignored
            Reserved for future use. Present to mirror the DD carrier API.
        min_common_sats : int
            Minimum number of common satellites including the reference.
        rover_weights : sequence of float, optional
            Per-row pseudorange weights aligned with ``rover_measurements``.
        """
        del rover_position_approx

        tow_key = round(tow, 1)
        rover_obs = None
        if self._rover_by_tow is not None:
            # When using raw rover/base RINEX, require the exact same epoch key
            # on both sides. Reusing a 1 Hz raw epoch for nearby 10 Hz solver
            # epochs injects tens of meters of deterministic bias.
            base_obs = self._base_by_tow.get(tow_key)
            if base_obs is None and self._interpolate_base_epochs:
                base_obs = self._interpolate_base_obs(tow_key)
            rover_obs = self._rover_by_tow.get(tow_key)
            if base_obs is None or rover_obs is None:
                return None
        else:
            base_obs = self._base_by_tow.get(tow_key)
            if base_obs is None:
                for offset in (0.1, -0.1, 0.2, -0.2):
                    base_obs = self._base_by_tow.get(round(tow + offset, 1))
                    if base_obs is not None:
                        break
            if base_obs is None:
                return None

        rover_rows = _one_row_per_satellite(rover_measurements, rover_weights)
        rover_pr: dict[str, float] = {}
        base_pr: dict[str, float] = {}
        rover_sat_ecef: dict[str, np.ndarray] = {}
        rover_elev: dict[str, float] = {}
        rover_w: dict[str, float] = {}

        for (system_id, prn), (m, row_weight) in rover_rows.items():
            sys_char = _SYS_MAP.get(system_id, "G")
            if sys_char not in self._allowed_systems:
                continue

            sat_pos = np.asarray(getattr(m, "satellite_ecef"), dtype=np.float64).ravel()[:3]
            if sat_pos.size != 3 or not np.all(np.isfinite(sat_pos)):
                continue

            sat_id = f"{sys_char}{prn:02d}"
            if rover_obs is None:
                pr = float(getattr(m, "corrected_pseudorange", 0.0))
                if pr == 0.0 or not np.isfinite(pr) or abs(pr) < 1e6:
                    continue
                rover_pr[sat_id] = pr
            rover_sat_ecef[sat_id] = sat_pos
            rover_elev[sat_id] = float(getattr(m, "elevation", 0.0))
            rover_w[sat_id] = float(row_weight) if np.isfinite(row_weight) and row_weight > 0 else 1.0

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
                    self._pseudorange_code,
                )
                if len(selected_sats) < 2:
                    continue
                for sat_id in selected_sats:
                    rover_pr[sat_id] = float(rover_obs[sat_id][_code])
                    base_pr[sat_id] = float(base_obs[sat_id][_code])
        else:
            for sat_id, sat_obs in base_obs.items():
                pr = _pick_single_obs_value(sat_id[0], sat_obs, self._pseudorange_code)
                if pr != 0.0:
                    base_pr[sat_id] = pr

        common_sats = sorted(set(rover_pr.keys()) & set(base_pr.keys()))
        if len(common_sats) < min_common_sats:
            return None

        dd_pr_list = []
        sat_ecef_k_list = []
        sat_ecef_ref_list = []
        base_range_k_list = []
        base_range_ref_list = []
        dd_weight_list = []
        ref_sat_ids = []

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

            rover_pr_ref = rover_pr[ref_sat]
            base_pr_ref = base_pr[ref_sat]
            ref_ecef = rover_sat_ecef[ref_sat]
            base_range_ref = float(np.linalg.norm(ref_ecef - self._base_pos))
            ref_weight = rover_w.get(ref_sat, 1.0)
            ref_elev = rover_elev.get(ref_sat, 0.3)

            for sat_id in non_ref:
                rover_pr_k = rover_pr[sat_id]
                base_pr_k = base_pr[sat_id]
                sat_k_ecef = rover_sat_ecef[sat_id]
                base_range_k = float(np.linalg.norm(sat_k_ecef - self._base_pos))

                dd_pr = (rover_pr_k - rover_pr_ref) - (base_pr_k - base_pr_ref)

                meas_weight = float(
                    np.sqrt(
                        max(rover_w.get(sat_id, 1.0), 1.0e-12)
                        * max(ref_weight, 1.0e-12)
                    )
                )
                elev_k = rover_elev.get(sat_id, 0.3)
                elev_weight = min(np.sin(max(elev_k, 0.05)), np.sin(max(ref_elev, 0.05)))

                dd_pr_list.append(dd_pr)
                sat_ecef_k_list.append(sat_k_ecef)
                sat_ecef_ref_list.append(ref_ecef.copy())
                base_range_k_list.append(base_range_k)
                base_range_ref_list.append(base_range_ref)
                dd_weight_list.append(meas_weight * elev_weight)
                ref_sat_ids.append(ref_sat)

        if not dd_pr_list:
            return None

        n_dd = len(dd_pr_list)
        return DDPseudorangeResult(
            dd_pseudorange_m=np.array(dd_pr_list, dtype=np.float64),
            sat_ecef_k=np.array(sat_ecef_k_list, dtype=np.float64).reshape(n_dd, 3),
            sat_ecef_ref=np.array(sat_ecef_ref_list, dtype=np.float64).reshape(n_dd, 3),
            base_range_k=np.array(base_range_k_list, dtype=np.float64),
            base_range_ref=np.array(base_range_ref_list, dtype=np.float64),
            dd_weights=np.array(dd_weight_list, dtype=np.float64),
            ref_sat_ids=tuple(ref_sat_ids),
            n_dd=n_dd,
        )
