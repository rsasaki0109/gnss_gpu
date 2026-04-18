"""Wide-lane integer ambiguity helpers for dual-frequency GNSS.

This module keeps the wide-lane logic usable from two levels:

* small pure functions for Melbourne-Wuebbena float ambiguities, LAMBDA fixing,
  and fixed wide-lane carrier pseudoranges;
* a DD pseudorange computer that turns fixed rover/base L1-L2 wide-lane pairs
  into the existing :class:`gnss_gpu.dd_pseudorange.DDPseudorangeResult` shape.

Only GPS/QZSS L1-L2 is enabled by default.  Galileo E1/E5 needs a different
frequency pair and is intentionally excluded from the Odaiba L1-L2 path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

from gnss_gpu.dd_pseudorange import DDPseudorangeResult
from gnss_gpu.io.rinex import read_rinex_obs
from gnss_gpu.lambda_ambiguity import solve_lambda

C_LIGHT = 299792458.0
L1_FREQ = 1575.42e6
L2_FREQ = 1227.60e6
LAMBDA_1 = C_LIGHT / L1_FREQ
LAMBDA_2 = C_LIGHT / L2_FREQ
LAMBDA_WL = C_LIGHT / (L1_FREQ - L2_FREQ)

_SYS_MAP = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}
_SUPPORTED_L12_SYSTEMS = frozenset({"G", "J"})

_L1_CARRIER_PREFS = {
    "G": ("L1C", "L1W", "L1X", "L1P", "L1S", "L1L", "L1Z"),
    "J": ("L1C", "L1X", "L1Z", "L1S", "L1L"),
}
_L2_CARRIER_PREFS = {
    "G": ("L2W", "L2L", "L2X", "L2S", "L2P", "L2C"),
    "J": ("L2X", "L2L", "L2S", "L2C", "L2W"),
}
_L1_CODE_PREFS = {
    "G": ("C1C", "C1W", "C1X", "C1P", "C1S", "C1L", "C1Z"),
    "J": ("C1C", "C1X", "C1Z", "C1S", "C1L"),
}
_L2_CODE_PREFS = {
    "G": ("C2W", "C2L", "C2X", "C2S", "C2P", "C2C"),
    "J": ("C2X", "C2L", "C2S", "C2C", "C2W"),
}


@dataclass(frozen=True)
class WidelaneObservation:
    """One satellite's dual-frequency code and carrier observations."""

    sat_id: str
    l1_carrier_cycles: float
    l2_carrier_cycles: float
    p1_m: float
    p2_m: float


@dataclass(frozen=True)
class WidelaneCandidate:
    """Float wide-lane ambiguity candidate."""

    key: Hashable
    float_ambiguity_cycles: float
    variance_cycles2: float = 0.25
    n_epochs: int = 1


@dataclass(frozen=True)
class WidelaneFix:
    """Accepted integer wide-lane ambiguity."""

    key: Hashable
    integer: int
    ratio: float
    residual_cycles: float
    n_epochs: int
    std_cycles: float


@dataclass(frozen=True)
class WidelaneDDStats:
    """Per-epoch stats from wide-lane DD pseudorange construction."""

    n_candidate_pairs: int = 0
    n_fixed_pairs: int = 0
    n_ratio_rejected: int = 0
    n_dd: int = 0
    fix_rate: float = 0.0
    reason: str = "no_candidates"


@dataclass(frozen=True)
class _RawWidelaneObs:
    l1_carrier_cycles: float
    l2_carrier_cycles: float
    p1_m: float
    p2_m: float


@dataclass
class _Track:
    estimates: list[float]
    fixed: WidelaneFix | None = None
    last_value: float | None = None


def compute_n_wl_float(
    l1_carrier_cycles: float,
    l2_carrier_cycles: float,
    p1_m: float,
    p2_m: float,
) -> float:
    """Compute a single Melbourne-Wuebbena float L1-L2 wide-lane ambiguity."""

    p_narrow_m = (L1_FREQ * float(p1_m) + L2_FREQ * float(p2_m)) / (L1_FREQ + L2_FREQ)
    return float(l1_carrier_cycles) - float(l2_carrier_cycles) - p_narrow_m / LAMBDA_WL


def wide_lane_phase_m(l1_carrier_cycles: float, l2_carrier_cycles: float) -> float:
    """Return the L1-L2 wide-lane carrier phase combination in meters."""

    phi1_m = float(l1_carrier_cycles) * LAMBDA_1
    phi2_m = float(l2_carrier_cycles) * LAMBDA_2
    return (L1_FREQ * phi1_m - L2_FREQ * phi2_m) / (L1_FREQ - L2_FREQ)


def wl_fixed_pseudorange(obs: WidelaneObservation, fixed_amb: int) -> float:
    """Return a fixed wide-lane carrier pseudorange in meters."""

    return wide_lane_phase_m(obs.l1_carrier_cycles, obs.l2_carrier_cycles) - int(fixed_amb) * LAMBDA_WL


def detect_wl_candidates(
    obs_L1: Iterable[WidelaneObservation] | Mapping[str, Any],
    obs_L2: Mapping[str, Any] | None = None,
) -> tuple[WidelaneCandidate, ...]:
    """Build float wide-lane candidates from observations.

    When ``obs_L2`` is omitted, ``obs_L1`` must be an iterable of
    :class:`WidelaneObservation`.  With two mappings, each mapping is keyed by
    satellite id and values may be ``{"carrier": ..., "pseudorange": ...}`` or
    two-element ``(carrier, pseudorange)`` tuples.
    """

    observations: list[WidelaneObservation] = []
    if obs_L2 is None:
        observations = list(obs_L1)  # type: ignore[arg-type]
    else:
        for sat_id in sorted(set(obs_L1.keys()) & set(obs_L2.keys())):  # type: ignore[union-attr]
            l1_carrier, p1_m = _carrier_and_code(obs_L1[sat_id])  # type: ignore[index]
            l2_carrier, p2_m = _carrier_and_code(obs_L2[sat_id])
            observations.append(
                WidelaneObservation(
                    sat_id=str(sat_id),
                    l1_carrier_cycles=l1_carrier,
                    l2_carrier_cycles=l2_carrier,
                    p1_m=p1_m,
                    p2_m=p2_m,
                )
            )

    out: list[WidelaneCandidate] = []
    for obs in observations:
        n_wl = compute_n_wl_float(
            obs.l1_carrier_cycles,
            obs.l2_carrier_cycles,
            obs.p1_m,
            obs.p2_m,
        )
        if np.isfinite(n_wl):
            out.append(WidelaneCandidate(key=obs.sat_id, float_ambiguity_cycles=n_wl))
    return tuple(out)


def fix_wl_ambiguities(
    candidates: Sequence[WidelaneCandidate],
    cov: np.ndarray | None = None,
    *,
    ratio_threshold: float = 3.0,
) -> dict[Hashable, WidelaneFix]:
    """Fix a group of float wide-lane candidates using the LAMBDA helper."""

    if not candidates:
        return {}
    float_amb = np.asarray([c.float_ambiguity_cycles for c in candidates], dtype=np.float64)
    if cov is None:
        variances = np.asarray(
            [max(float(c.variance_cycles2), 1.0e-6) for c in candidates],
            dtype=np.float64,
        )
        cov_arr = np.diag(variances)
    else:
        cov_arr = np.asarray(cov, dtype=np.float64)
    fixed, ok, solution = solve_lambda(float_amb, cov_arr, ratio_threshold=ratio_threshold)
    if not ok or fixed is None:
        return {}
    fixes: dict[Hashable, WidelaneFix] = {}
    for cand, integer in zip(candidates, fixed):
        residual = float(cand.float_ambiguity_cycles - int(integer))
        fixes[cand.key] = WidelaneFix(
            key=cand.key,
            integer=int(integer),
            ratio=float(solution.ratio),
            residual_cycles=residual,
            n_epochs=int(cand.n_epochs),
            std_cycles=float(np.sqrt(max(cand.variance_cycles2, 0.0))),
        )
    return fixes


class WidelaneAmbiguityResolver:
    """Track and fix scalar wide-lane ambiguities for stable keys."""

    def __init__(
        self,
        *,
        min_epochs: int = 5,
        max_std_cycles: float = 0.75,
        ratio_threshold: float = 3.0,
        cycle_slip_threshold_cycles: float = 8.0,
        window_size: int = 60,
        min_variance_cycles2: float = 0.01,
    ) -> None:
        self.min_epochs = int(min_epochs)
        self.max_std_cycles = float(max_std_cycles)
        self.ratio_threshold = float(ratio_threshold)
        self.cycle_slip_threshold_cycles = float(cycle_slip_threshold_cycles)
        self.window_size = int(window_size)
        self.min_variance_cycles2 = float(min_variance_cycles2)
        self._tracks: dict[Hashable, _Track] = defaultdict(lambda: _Track(estimates=[]))

    def update(self, key: Hashable, float_ambiguity_cycles: float) -> WidelaneFix | None:
        """Add one float ambiguity sample and return an accepted fix if available."""

        value = float(float_ambiguity_cycles)
        if not np.isfinite(value):
            return None
        track = self._tracks[key]
        if track.last_value is not None and abs(value - track.last_value) > self.cycle_slip_threshold_cycles:
            track.estimates.clear()
            track.fixed = None
        if track.fixed is not None and abs(value - track.fixed.integer) > self.cycle_slip_threshold_cycles:
            track.estimates.clear()
            track.fixed = None
        track.last_value = value
        track.estimates.append(value)
        if len(track.estimates) > self.window_size:
            del track.estimates[: len(track.estimates) - self.window_size]
        if track.fixed is None:
            track.fixed = self._try_fix(key, track.estimates)
        return track.fixed

    def get(self, key: Hashable) -> WidelaneFix | None:
        """Return the current fixed ambiguity for *key*, if one is accepted."""

        track = self._tracks.get(key)
        return None if track is None else track.fixed

    def reset(self, key: Hashable | None = None) -> None:
        """Reset one ambiguity track, or all tracks when *key* is omitted."""

        if key is None:
            self._tracks.clear()
        else:
            self._tracks.pop(key, None)

    def _try_fix(self, key: Hashable, estimates: Sequence[float]) -> WidelaneFix | None:
        if len(estimates) < self.min_epochs:
            return None
        arr = np.asarray(estimates, dtype=np.float64)
        if not np.isfinite(arr).all():
            return None
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        if std > self.max_std_cycles:
            return None
        mean = float(np.mean(arr))
        variance_mean = max((std * std) / max(arr.size, 1), self.min_variance_cycles2)
        candidate = WidelaneCandidate(
            key=key,
            float_ambiguity_cycles=mean,
            variance_cycles2=variance_mean,
            n_epochs=int(arr.size),
        )
        return fix_wl_ambiguities([candidate], ratio_threshold=self.ratio_threshold).get(key)


class WidelaneDDPseudorangeComputer:
    """Build DD pseudorange rows from fixed L1-L2 wide-lane carrier phase."""

    def __init__(
        self,
        base_obs_path: str | Path,
        rover_obs_path: str | Path,
        *,
        base_position: np.ndarray | None = None,
        allowed_systems: Sequence[str] = ("G", "J"),
        interpolate_base_epochs: bool = False,
        min_epochs: int = 5,
        max_std_cycles: float = 0.75,
        ratio_threshold: float = 3.0,
        min_fix_rate: float = 0.3,
    ) -> None:
        base_obs_path = Path(base_obs_path)
        rover_obs_path = Path(rover_obs_path)
        if not base_obs_path.exists():
            raise FileNotFoundError(f"Base RINEX not found: {base_obs_path}")
        if not rover_obs_path.exists():
            raise FileNotFoundError(f"Rover RINEX not found: {rover_obs_path}")

        self._allowed_systems = tuple(s for s in allowed_systems if s in _SUPPORTED_L12_SYSTEMS)
        self._interpolate_base_epochs = bool(interpolate_base_epochs)
        self._min_fix_rate = float(min_fix_rate)
        self._resolver = WidelaneAmbiguityResolver(
            min_epochs=min_epochs,
            max_std_cycles=max_std_cycles,
            ratio_threshold=ratio_threshold,
        )

        self._base_obs = read_rinex_obs(base_obs_path)
        self._rover_obs = read_rinex_obs(rover_obs_path)
        self._base_pos = (
            np.asarray(base_position, dtype=np.float64).ravel()
            if base_position is not None
            else self._base_obs.header.approx_position.copy()
        )
        if np.linalg.norm(self._base_pos) < 1e6:
            raise ValueError(f"Base position looks invalid: {self._base_pos}")

        self._base_by_tow = self._index_obs(self._base_obs)
        self._rover_by_tow = self._index_obs(self._rover_obs)
        self._base_tow_keys = np.array(sorted(self._base_by_tow.keys()), dtype=np.float64)
        print(
            f"  [WL] Loaded rover/base L1-L2 obs: rover={len(self._rover_by_tow)} "
            f"base={len(self._base_by_tow)} epochs systems={','.join(self._allowed_systems) or 'none'}"
        )

    @property
    def base_position(self) -> np.ndarray:
        """Base station ECEF position [m]."""

        return self._base_pos

    def compute_dd(
        self,
        tow: float,
        rover_measurements: Sequence[Any],
        rover_position_approx: np.ndarray | None = None,
        *,
        min_common_sats: int = 4,
        rover_weights: Sequence[float] | None = None,
        min_fix_rate: float | None = None,
        min_ratio: float | None = None,
    ) -> tuple[DDPseudorangeResult | None, WidelaneDDStats]:
        """Compute fixed wide-lane DD pseudorange rows for one epoch."""

        del rover_position_approx
        tow_key = round(float(tow), 1)
        rover_obs = self._rover_by_tow.get(tow_key)
        base_obs = self._base_by_tow.get(tow_key)
        if base_obs is None and self._interpolate_base_epochs:
            base_obs = self._interpolate_base_obs(tow_key)
        if rover_obs is None or base_obs is None:
            return None, WidelaneDDStats(reason="missing_epoch")

        rover_rows = _one_row_per_satellite(rover_measurements, rover_weights)
        common_sats = sorted(set(rover_rows.keys()) & set(rover_obs.keys()) & set(base_obs.keys()))
        if not common_sats:
            return None, WidelaneDDStats(reason="no_common_sats")

        sats_by_system: dict[str, list[str]] = {}
        for sat_id in common_sats:
            sys_char = sat_id[0]
            if sys_char in self._allowed_systems:
                sats_by_system.setdefault(sys_char, []).append(sat_id)

        dd_pr_list: list[float] = []
        sat_ecef_k_list: list[np.ndarray] = []
        sat_ecef_ref_list: list[np.ndarray] = []
        base_range_k_list: list[float] = []
        base_range_ref_list: list[float] = []
        dd_weight_list: list[float] = []
        ref_sat_ids: list[str] = []
        n_candidate_pairs = 0
        n_fixed_pairs = 0
        n_ratio_rejected = 0
        min_ratio_value = None if min_ratio is None else float(min_ratio)

        for sys_char, sys_sats in sorted(sats_by_system.items()):
            if len(sys_sats) < 2:
                continue
            ref_sat = max(sys_sats, key=lambda s: float(getattr(rover_rows[s][0], "elevation", 0.0)))
            ref_row, ref_weight = rover_rows[ref_sat]
            ref_ecef = np.asarray(getattr(ref_row, "satellite_ecef"), dtype=np.float64).ravel()[:3]
            if ref_ecef.size != 3 or not np.isfinite(ref_ecef).all():
                continue
            ref_elev = float(getattr(ref_row, "elevation", 0.3))
            base_range_ref = float(np.linalg.norm(ref_ecef - self._base_pos))

            for sat_id in sys_sats:
                if sat_id == ref_sat:
                    continue
                row, row_weight = rover_rows[sat_id]
                sat_k_ecef = np.asarray(getattr(row, "satellite_ecef"), dtype=np.float64).ravel()[:3]
                if sat_k_ecef.size != 3 or not np.isfinite(sat_k_ecef).all():
                    continue
                n_candidate_pairs += 1
                dd_float = _dd_wl_float(
                    rover_obs[sat_id],
                    rover_obs[ref_sat],
                    base_obs[sat_id],
                    base_obs[ref_sat],
                )
                pair_key = (sys_char, sat_id, ref_sat)
                fix = self._resolver.update(pair_key, dd_float)
                if fix is None:
                    continue
                if min_ratio_value is not None and float(fix.ratio) < min_ratio_value:
                    n_ratio_rejected += 1
                    continue
                n_fixed_pairs += 1
                dd_phase_m = _dd_wl_phase_m(
                    rover_obs[sat_id],
                    rover_obs[ref_sat],
                    base_obs[sat_id],
                    base_obs[ref_sat],
                )
                dd_pr_m = dd_phase_m - float(fix.integer) * LAMBDA_WL
                if not np.isfinite(dd_pr_m):
                    continue

                elev_k = float(getattr(row, "elevation", 0.3))
                meas_weight = float(np.sqrt(max(row_weight, 1.0e-12) * max(ref_weight, 1.0e-12)))
                elev_weight = min(np.sin(max(elev_k, 0.05)), np.sin(max(ref_elev, 0.05)))

                dd_pr_list.append(float(dd_pr_m))
                sat_ecef_k_list.append(sat_k_ecef)
                sat_ecef_ref_list.append(ref_ecef.copy())
                base_range_k_list.append(float(np.linalg.norm(sat_k_ecef - self._base_pos)))
                base_range_ref_list.append(base_range_ref)
                dd_weight_list.append(meas_weight * elev_weight)
                ref_sat_ids.append(ref_sat)

        fix_rate = float(n_fixed_pairs) / float(n_candidate_pairs) if n_candidate_pairs else 0.0
        min_rate = self._min_fix_rate if min_fix_rate is None else float(min_fix_rate)
        if n_candidate_pairs == 0:
            return None, WidelaneDDStats(reason="no_candidate_pairs")
        if fix_rate < min_rate:
            return None, WidelaneDDStats(
                n_candidate_pairs=n_candidate_pairs,
                n_fixed_pairs=n_fixed_pairs,
                n_ratio_rejected=n_ratio_rejected,
                fix_rate=fix_rate,
                reason="low_fix_rate",
            )
        min_dd = max(1, int(min_common_sats) - 1)
        if len(dd_pr_list) < min_dd:
            return None, WidelaneDDStats(
                n_candidate_pairs=n_candidate_pairs,
                n_fixed_pairs=n_fixed_pairs,
                n_ratio_rejected=n_ratio_rejected,
                n_dd=len(dd_pr_list),
                fix_rate=fix_rate,
                reason="too_few_fixed_pairs",
            )

        n_dd = len(dd_pr_list)
        result = DDPseudorangeResult(
            dd_pseudorange_m=np.asarray(dd_pr_list, dtype=np.float64),
            sat_ecef_k=np.asarray(sat_ecef_k_list, dtype=np.float64).reshape(n_dd, 3),
            sat_ecef_ref=np.asarray(sat_ecef_ref_list, dtype=np.float64).reshape(n_dd, 3),
            base_range_k=np.asarray(base_range_k_list, dtype=np.float64),
            base_range_ref=np.asarray(base_range_ref_list, dtype=np.float64),
            dd_weights=np.asarray(dd_weight_list, dtype=np.float64),
            ref_sat_ids=tuple(ref_sat_ids),
            n_dd=n_dd,
        )
        return result, WidelaneDDStats(
            n_candidate_pairs=n_candidate_pairs,
            n_fixed_pairs=n_fixed_pairs,
            n_ratio_rejected=n_ratio_rejected,
            n_dd=n_dd,
            fix_rate=fix_rate,
            reason="ok",
        )

    def _index_obs(self, rinex_obs) -> dict[float, dict[str, _RawWidelaneObs]]:
        by_tow: dict[float, dict[str, _RawWidelaneObs]] = {}
        for ep in rinex_obs.epochs:
            tow_key = round(_datetime_to_tow(ep.time), 1)
            epoch_rows: dict[str, _RawWidelaneObs] = {}
            for sat_id, obs in ep.observations.items():
                sat_norm = _normalize_sat_id(sat_id)
                if not sat_norm or sat_norm[0] not in self._allowed_systems:
                    continue
                raw = _select_dual_frequency_observation(sat_norm[0], obs)
                if raw is not None:
                    epoch_rows[sat_norm] = raw
            if epoch_rows:
                by_tow[tow_key] = epoch_rows
        return by_tow

    def _interpolate_base_obs(self, tow_key: float) -> dict[str, _RawWidelaneObs] | None:
        if self._base_tow_keys.size < 2:
            return None
        idx = int(np.searchsorted(self._base_tow_keys, tow_key))
        if idx <= 0 or idx >= self._base_tow_keys.size:
            return None
        t0 = float(self._base_tow_keys[idx - 1])
        t1 = float(self._base_tow_keys[idx])
        if not (t0 < tow_key < t1) or (t1 - t0) > 1.5:
            return None
        obs0 = self._base_by_tow.get(t0)
        obs1 = self._base_by_tow.get(t1)
        if obs0 is None or obs1 is None:
            return None
        alpha = float((tow_key - t0) / (t1 - t0))
        out: dict[str, _RawWidelaneObs] = {}
        for sat_id in set(obs0.keys()) & set(obs1.keys()):
            a = obs0[sat_id]
            b = obs1[sat_id]
            out[sat_id] = _RawWidelaneObs(
                l1_carrier_cycles=(1.0 - alpha) * a.l1_carrier_cycles + alpha * b.l1_carrier_cycles,
                l2_carrier_cycles=(1.0 - alpha) * a.l2_carrier_cycles + alpha * b.l2_carrier_cycles,
                p1_m=(1.0 - alpha) * a.p1_m + alpha * b.p1_m,
                p2_m=(1.0 - alpha) * a.p2_m + alpha * b.p2_m,
            )
        return out or None


def _carrier_and_code(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        return float(value["carrier"]), float(value["pseudorange"])
    return float(value[0]), float(value[1])


def _dd_wl_float(
    rover_k: _RawWidelaneObs,
    rover_ref: _RawWidelaneObs,
    base_k: _RawWidelaneObs,
    base_ref: _RawWidelaneObs,
) -> float:
    return (
        _n_wl_raw(rover_k)
        - _n_wl_raw(rover_ref)
        - _n_wl_raw(base_k)
        + _n_wl_raw(base_ref)
    )


def _dd_wl_phase_m(
    rover_k: _RawWidelaneObs,
    rover_ref: _RawWidelaneObs,
    base_k: _RawWidelaneObs,
    base_ref: _RawWidelaneObs,
) -> float:
    return (
        _wl_phase_raw_m(rover_k)
        - _wl_phase_raw_m(rover_ref)
        - _wl_phase_raw_m(base_k)
        + _wl_phase_raw_m(base_ref)
    )


def _n_wl_raw(obs: _RawWidelaneObs) -> float:
    return compute_n_wl_float(obs.l1_carrier_cycles, obs.l2_carrier_cycles, obs.p1_m, obs.p2_m)


def _wl_phase_raw_m(obs: _RawWidelaneObs) -> float:
    return wide_lane_phase_m(obs.l1_carrier_cycles, obs.l2_carrier_cycles)


def _datetime_to_tow(epoch_time) -> float:
    dow = epoch_time.weekday()
    gps_dow = (dow + 1) % 7
    return (
        gps_dow * 86400.0
        + epoch_time.hour * 3600
        + epoch_time.minute * 60
        + epoch_time.second
        + epoch_time.microsecond * 1.0e-6
    )


def _normalize_sat_id(sat_id: str) -> str:
    sat_id = sat_id.strip()
    if not sat_id:
        return sat_id
    try:
        return f"{sat_id[0]}{int(sat_id[1:].strip()):02d}"
    except ValueError:
        return sat_id


def _select_dual_frequency_observation(sys_char: str, obs: Mapping[str, float]) -> _RawWidelaneObs | None:
    if sys_char not in _SUPPORTED_L12_SYSTEMS:
        return None
    l1 = _pick_valid(obs, _L1_CARRIER_PREFS.get(sys_char, ()), "L1", min_abs=1.0e3)
    l2 = _pick_valid(obs, _L2_CARRIER_PREFS.get(sys_char, ()), "L2", min_abs=1.0e3)
    p1 = _pick_valid(obs, _L1_CODE_PREFS.get(sys_char, ()), "C1", min_abs=1.0e6)
    p2 = _pick_valid(obs, _L2_CODE_PREFS.get(sys_char, ()), "C2", min_abs=1.0e6)
    if l1 is None or l2 is None or p1 is None or p2 is None:
        return None
    return _RawWidelaneObs(
        l1_carrier_cycles=float(l1),
        l2_carrier_cycles=float(l2),
        p1_m=float(p1),
        p2_m=float(p2),
    )


def _pick_valid(
    obs: Mapping[str, float],
    preferred_codes: Sequence[str],
    prefix: str,
    *,
    min_abs: float,
) -> float | None:
    codes: list[str] = []
    seen: set[str] = set()
    for code in preferred_codes:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    for code in sorted(k for k in obs if k.startswith(prefix)):
        if code not in seen:
            seen.add(code)
            codes.append(code)
    for code in codes:
        if code not in obs:
            continue
        value = float(obs[code])
        if np.isfinite(value) and abs(value) >= min_abs:
            return value
    return None


def _one_row_per_satellite(
    measurements: Sequence[Any],
    rover_weights: Sequence[float] | None = None,
) -> dict[str, tuple[Any, float]]:
    by_sat: dict[str, list[tuple[Any, float]]] = {}
    for i, m in enumerate(measurements):
        sys_char = _SYS_MAP.get(int(getattr(m, "system_id", 0)), "G")
        sat_id = f"{sys_char}{int(getattr(m, 'prn', 0)):02d}"
        if sys_char not in _SUPPORTED_L12_SYSTEMS:
            continue
        weight = (
            float(rover_weights[i])
            if rover_weights is not None and i < len(rover_weights)
            else float(getattr(m, "weight", 1.0))
        )
        by_sat.setdefault(sat_id, []).append((m, weight))

    out: dict[str, tuple[Any, float]] = {}
    eps = 0.05
    for sat_id, rows in by_sat.items():
        if len(rows) == 1:
            out[sat_id] = rows[0]
            continue
        rows_sorted = sorted(rows, key=lambda p: float(getattr(p[0], "snr", 0.0)), reverse=True)
        best_snr = float(getattr(rows_sorted[0][0], "snr", 0.0))
        n_top = sum(1 for m, _w in rows_sorted if abs(float(getattr(m, "snr", 0.0)) - best_snr) <= eps)
        if n_top == 1:
            out[sat_id] = rows_sorted[0]
    return out
