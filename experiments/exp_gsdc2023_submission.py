#!/usr/bin/env python3
"""Generate Kaggle GSDC 2023 submission CSV using GPU Particle Filter.

v3: Forward-backward smoother + bounded Hatch filter + TDCP velocity.
Built on v1 baseline with careful divergence protection.

Usage
-----
  cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
  PYTHONPATH=python python3 experiments/exp_gsdc2023_submission.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for p in [
    str(_PROJECT_ROOT / "python"),
    str(_PROJECT_ROOT / "third_party" / "gnssplusplus" / "build" / "python"),
    str(_PROJECT_ROOT / "third_party" / "gnssplusplus" / "python"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate import ecef_to_lla

# ---------------------------------------------------------------------------
# Config (v1 baseline params unless noted)
# ---------------------------------------------------------------------------
TEST_DIR = Path("/tmp/gsdc_data/gsdc2023/sdc2023/test")
SAMPLE_SUB = Path("/tmp/gsdc_data/gsdc2023/sdc2023/sample_submission.csv")
OUTPUT_CSV = _SCRIPT_DIR / "results" / "gsdc2023_submission_v2.csv"

N_PARTICLES = 100_000
SIGMA_POS = 10.0
SIGMA_CB = 300.0
SIGMA_PR = 12.0               # slightly tighter than v1 (Hatch helps)
POS_UPDATE_SIGMA = 3.0        # v1 WLS constraint
ELEV_THRESHOLD = 15.0

SIGMA_POS_TDCP = 4.0          # moderate TDCP tightness
HATCH_MAX_WINDOW = 50         # shorter window to avoid iono divergence
HATCH_DIVERGE_THRESHOLD = 10.0
CN0_THRESHOLD = 20.0

SIGNAL_TYPES = ["GPS_L1_CA", "GPS_L5_Q", "GAL_E1_C_P", "GAL_E5A_Q"]

# GPS time -> Unix time
GPS_UNIX_OFFSET_NS = 315964800 * 1_000_000_000
LEAP_SECONDS_NS = 18 * 1_000_000_000


# ---------------------------------------------------------------------------
# GNSS parser (v1 + ADR fields for TDCP/Hatch)
# ---------------------------------------------------------------------------
def parse_gnss_data_v2(gnss_path: Path) -> list[dict]:
    """Parse device_gnss.csv with ADR, satellite IDs, and quality filtering."""
    df = pd.read_csv(gnss_path, low_memory=False)
    df = df[df["SignalType"].isin(SIGNAL_TYPES)].copy()

    # Quality: Cn0 threshold
    df = df[df["Cn0DbHz"] >= CN0_THRESHOLD].copy()

    # Quality: remove multipath-flagged measurements
    if "MultipathIndicator" in df.columns:
        mp = df["MultipathIndicator"].fillna(0)
        df = df[mp != 1].copy()

    critical_cols = [
        "RawPseudorangeMeters", "SvClockBiasMeters",
        "SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters",
    ]
    df = df.dropna(subset=critical_cols)

    grouped = df.groupby("ArrivalTimeNanosSinceGpsEpoch")
    epochs = []

    for arrival_ns, group in sorted(grouped):
        n_sv = len(group)
        if n_sv < 4:
            continue

        sat_ecef = np.column_stack([
            group["SvPositionXEcefMeters"].values,
            group["SvPositionYEcefMeters"].values,
            group["SvPositionZEcefMeters"].values,
        ]).astype(np.float64)

        sv_vel = np.column_stack([
            group["SvVelocityXEcefMetersPerSecond"].values,
            group["SvVelocityYEcefMetersPerSecond"].values,
            group["SvVelocityZEcefMetersPerSecond"].values,
        ]).astype(np.float64)

        raw_pr = group["RawPseudorangeMeters"].values.astype(np.float64)
        sv_clock_bias = group["SvClockBiasMeters"].values.astype(np.float64)
        iono_delay = np.nan_to_num(
            group["IonosphericDelayMeters"].values.astype(np.float64), nan=0.0)
        tropo_delay = np.nan_to_num(
            group["TroposphericDelayMeters"].values.astype(np.float64), nan=0.0)
        isrb = np.nan_to_num(
            group["IsrbMeters"].values.astype(np.float64), nan=0.0)

        corrected_pr = raw_pr + sv_clock_bias - iono_delay - tropo_delay - isrb

        elevations = group["SvElevationDegrees"].values.astype(np.float64)
        pr_rate = group["PseudorangeRateMetersPerSecond"].values.astype(np.float64)
        cn0 = group["Cn0DbHz"].values.astype(np.float64)

        wls_ecef = np.array([
            group["WlsPositionXEcefMeters"].iloc[0],
            group["WlsPositionYEcefMeters"].iloc[0],
            group["WlsPositionZEcefMeters"].iloc[0],
        ], dtype=np.float64)

        if np.any(np.isnan(wls_ecef)) or np.linalg.norm(wls_ecef) < 1e6:
            continue

        # ADR fields
        adr = np.nan_to_num(
            group["AccumulatedDeltaRangeMeters"].values.astype(np.float64), nan=0.0)
        adr_state = np.nan_to_num(
            group["AccumulatedDeltaRangeState"].values.astype(np.float64), nan=0.0
        ).astype(np.int32)
        adr_valid = (adr_state & 1) == 1       # bit0: ADR_STATE_VALID
        # bit1: ADR_STATE_RESET, bit2: ADR_STATE_CYCLE_SLIP — both = discontinuity
        cycle_slip = (adr_state & 6) != 0

        sv_clock_drift = np.nan_to_num(
            group["SvClockDriftMetersPerSecond"].values.astype(np.float64), nan=0.0)

        constellation = group["ConstellationType"].values.astype(np.int32)
        svid = group["Svid"].values.astype(np.int32)
        signal_type = group["SignalType"].values
        sat_ids = [
            (int(constellation[j]), int(svid[j]), str(signal_type[j]))
            for j in range(n_sv)
        ]

        epochs.append({
            "arrival_ns": arrival_ns,
            "sat_ecef": sat_ecef,
            "sv_vel": sv_vel,
            "pseudoranges": corrected_pr,
            "elevations": elevations,
            "pr_rate": pr_rate,
            "cn0": cn0,
            "wls_ecef": wls_ecef,
            "n_sv": n_sv,
            "adr": adr,
            "adr_valid": adr_valid,
            "cycle_slip": cycle_slip,
            "sv_clock_drift": sv_clock_drift,
            "sat_ids": sat_ids,
        })

    return epochs


# ---------------------------------------------------------------------------
# Bounded Hatch filter
# ---------------------------------------------------------------------------
class HatchFilter:
    """Carrier-smoothed pseudorange with divergence protection.

    Resets filter for a satellite when:
    - ADR cycle slip or reset detected
    - |smoothed - raw_PR| exceeds diverge_threshold
    """

    def __init__(self, max_window: int = 100, diverge_threshold: float = 15.0):
        self.max_window = max_window
        self.diverge_threshold = diverge_threshold
        self._prev_adr: dict[tuple, float] = {}
        self._prev_smoothed: dict[tuple, float] = {}
        self._counts: dict[tuple, int] = {}

    def smooth(
        self,
        sat_ids: list[tuple],
        pseudoranges: np.ndarray,
        adr: np.ndarray,
        adr_valid: np.ndarray,
        cycle_slip: np.ndarray,
    ) -> np.ndarray:
        smoothed = pseudoranges.copy()
        for i, sid in enumerate(sat_ids):
            if not adr_valid[i] or adr[i] == 0.0:
                continue

            if cycle_slip[i] or sid not in self._prev_adr:
                self._prev_adr[sid] = adr[i]
                self._prev_smoothed[sid] = pseudoranges[i]
                self._counts[sid] = 1
                continue

            delta_adr = adr[i] - self._prev_adr[sid]

            # Sanity: reject huge delta_adr (broken tracking)
            if abs(delta_adr) > 50000.0:
                self._prev_adr[sid] = adr[i]
                self._prev_smoothed[sid] = pseudoranges[i]
                self._counts[sid] = 1
                continue

            n = min(self._counts[sid] + 1, self.max_window)
            sm = (1.0 / n) * pseudoranges[i] + (1.0 - 1.0 / n) * (
                self._prev_smoothed[sid] + delta_adr
            )

            # Divergence check: reset if too far from raw
            if abs(sm - pseudoranges[i]) > self.diverge_threshold:
                self._prev_adr[sid] = adr[i]
                self._prev_smoothed[sid] = pseudoranges[i]
                self._counts[sid] = 1
                continue

            self._prev_adr[sid] = adr[i]
            self._prev_smoothed[sid] = sm
            self._counts[sid] = n
            smoothed[i] = sm

        return smoothed


# ---------------------------------------------------------------------------
# Doppler velocity
# ---------------------------------------------------------------------------
def compute_doppler_velocity(epoch: dict, rx_ecef: np.ndarray) -> np.ndarray:
    sat_ecef = epoch["sat_ecef"]
    sv_vel = epoch["sv_vel"]
    pr_rate = epoch["pr_rate"]
    n_sv = len(pr_rate)

    if n_sv < 4:
        return np.zeros(3)

    los = sat_ecef - rx_ecef
    ranges = np.linalg.norm(los, axis=1, keepdims=True)
    ranges = np.maximum(ranges, 1.0)
    los_unit = los / ranges

    H = np.column_stack([-los_unit, np.ones(n_sv)])
    sv_radial = np.sum(sv_vel * los_unit, axis=1)
    obs = pr_rate - sv_radial

    try:
        x, _, _, _ = np.linalg.lstsq(H, obs, rcond=None)
        return x[:3]
    except np.linalg.LinAlgError:
        return np.zeros(3)


# ---------------------------------------------------------------------------
# TDCP velocity
# ---------------------------------------------------------------------------
def compute_tdcp_velocity(
    cur_ep: dict, prev_ep: dict, rx_ecef: np.ndarray, dt: float,
) -> np.ndarray | None:
    """Compute receiver velocity from Time-Differenced Carrier Phase."""
    prev_lookup: dict[tuple, int] = {}
    for j, sid in enumerate(prev_ep["sat_ids"]):
        if prev_ep["adr_valid"][j] and prev_ep["adr"][j] != 0.0:
            prev_lookup[sid] = j

    matched = []
    for i, sid in enumerate(cur_ep["sat_ids"]):
        if not cur_ep["adr_valid"][i] or cur_ep["adr"][i] == 0.0:
            continue
        if cur_ep["cycle_slip"][i]:
            continue
        if sid in prev_lookup:
            matched.append((i, prev_lookup[sid]))

    if len(matched) < 5:
        return None

    n = len(matched)
    H = np.zeros((n, 4))
    b = np.zeros(n)
    valid = np.ones(n, dtype=bool)

    for k, (i_cur, i_prev) in enumerate(matched):
        delta_adr = cur_ep["adr"][i_cur] - prev_ep["adr"][i_prev]

        sat_cur = cur_ep["sat_ecef"][i_cur]
        sat_prev = prev_ep["sat_ecef"][i_prev]

        range_cur = np.linalg.norm(sat_cur - rx_ecef)
        range_prev = np.linalg.norm(sat_prev - rx_ecef)
        if range_cur < 1.0:
            valid[k] = False
            continue
        range_sat_change = range_cur - range_prev

        los_unit = (sat_cur - rx_ecef) / range_cur

        obs = delta_adr - range_sat_change
        H[k] = [-los_unit[0], -los_unit[1], -los_unit[2], 1.0]
        b[k] = obs

    mask = valid
    if mask.sum() < 5:
        return None

    try:
        x, residuals, rank, sv = np.linalg.lstsq(H[mask], b[mask], rcond=None)
    except np.linalg.LinAlgError:
        return None

    velocity = x[:3] / dt

    # Sanity: reject unreasonable speed
    if np.linalg.norm(velocity) > 80.0:
        return None

    # Outlier rejection: remove satellites with large residuals and re-solve
    pred = H[mask] @ x
    resid = np.abs(b[mask] - pred)
    threshold = max(3.0 * np.median(resid), 0.5)  # 3x median, min 0.5m
    good = resid < threshold
    if good.sum() >= 5 and good.sum() < mask.sum():
        Hg = H[mask][good]
        bg = b[mask][good]
        try:
            x2, _, _, _ = np.linalg.lstsq(Hg, bg, rcond=None)
            velocity = x2[:3] / dt
            if np.linalg.norm(velocity) > 80.0:
                return None
        except np.linalg.LinAlgError:
            pass

    return velocity


# ---------------------------------------------------------------------------
# Elevation + Cn0 weights
# ---------------------------------------------------------------------------
def elevation_weights(
    elevations: np.ndarray, cn0: np.ndarray | None = None, threshold: float = 20.0,
) -> np.ndarray:
    weights = np.ones(len(elevations))
    for i, el in enumerate(elevations):
        if el < threshold:
            weights[i] = max(0.1, math.sin(math.radians(max(el, 1.0))) ** 2)
    if cn0 is not None:
        cn0_w = np.clip((cn0 - 20.0) / 25.0, 0.3, 1.0)
        weights *= cn0_w
    return weights


def epochs_to_unix_ms(epochs: list[dict]) -> np.ndarray:
    arr = np.array([ep["arrival_ns"] for ep in epochs], dtype=np.float64)
    unix_ns = arr + GPS_UNIX_OFFSET_NS - LEAP_SECONDS_NS
    return unix_ns / 1_000_000


# ---------------------------------------------------------------------------
# PF pipeline
# ---------------------------------------------------------------------------
def run_pf_test(data_dir: Path, label: str) -> tuple[np.ndarray, np.ndarray] | None:
    gnss_path = data_dir / "device_gnss.csv"
    if not gnss_path.exists():
        print(f"  [SKIP] {label}: missing device_gnss.csv")
        return None

    try:
        epochs = parse_gnss_data_v2(gnss_path)
    except Exception as e:
        print(f"  [ERROR] {label}: parse failed: {e}")
        return None

    if len(epochs) < 4:
        print(f"  [SKIP] {label}: too few epochs ({len(epochs)})")
        return None

    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    n_epochs = len(epochs)

    pf = ParticleFilterDevice(
        n_particles=N_PARTICLES,
        sigma_pos=SIGMA_POS,
        sigma_cb=SIGMA_CB,
        sigma_pr=SIGMA_PR,
        resampling="megopolis",
        ess_threshold=0.5,
        seed=42,
    )

    init_pos = epochs[0]["wls_ecef"]
    init_ranges = np.linalg.norm(epochs[0]["sat_ecef"] - init_pos, axis=1)
    init_cb = float(np.median(epochs[0]["pseudoranges"] - init_ranges))
    pf.initialize(init_pos, clock_bias=init_cb, spread_pos=50.0, spread_cb=500.0)

    # Enable smoother
    pf.enable_smoothing()

    # Hatch filter with divergence bound
    hatch = HatchFilter(max_window=HATCH_MAX_WINDOW,
                        diverge_threshold=HATCH_DIVERGE_THRESHOLD)

    pf_positions = np.zeros((n_epochs, 3))
    prev_arrival_ns = epochs[0]["arrival_ns"]
    tdcp_count = 0

    for i, ep in enumerate(epochs):
        dt = (ep["arrival_ns"] - prev_arrival_ns) / 1e9 if i > 0 else 1.0
        dt = max(dt, 0.01)
        prev_arrival_ns = ep["arrival_ns"]

        # Hatch filter (bounded)
        smoothed_pr = hatch.smooth(
            ep["sat_ids"], ep["pseudoranges"],
            ep["adr"], ep["adr_valid"], ep["cycle_slip"],
        )

        # TDCP or Doppler velocity
        rx_ecef = pf_positions[i - 1] if i > 0 else init_pos
        velocity = None
        sigma_pos = SIGMA_POS

        if i > 0:
            velocity = compute_tdcp_velocity(ep, epochs[i - 1], rx_ecef, dt)
        if velocity is not None:
            sigma_pos = SIGMA_POS_TDCP
            tdcp_count += 1
        else:
            velocity = compute_doppler_velocity(ep, rx_ecef)

        # PF steps (v1 params)
        pf.predict(velocity=velocity, dt=dt, sigma_pos=sigma_pos)
        pf.correct_clock_bias(ep["sat_ecef"], smoothed_pr)

        weights = elevation_weights(ep["elevations"], ep["cn0"], ELEV_THRESHOLD)
        pf.update_gmm(
            ep["sat_ecef"], smoothed_pr, weights=weights,
            sigma_pr=SIGMA_PR, w_los=0.8, mu_nlos=15.0, sigma_nlos=30.0,
        )
        pf.position_update(ep["wls_ecef"], sigma_pos=POS_UPDATE_SIGMA)

        est = pf.estimate()
        pf_positions[i] = est[:3]

        # Store for smoother
        pf.store_epoch(
            ep["sat_ecef"], smoothed_pr, weights,
            velocity, dt, spp_ref=ep["wls_ecef"],
        )

        # Divergence reset
        if np.linalg.norm(est[:3] - ep["wls_ecef"]) > 500.0:
            cb = float(np.median(
                ep["pseudoranges"]
                - np.linalg.norm(ep["sat_ecef"] - ep["wls_ecef"], axis=1)
            ))
            pf.initialize(
                ep["wls_ecef"], clock_bias=cb,
                spread_pos=50.0, spread_cb=500.0,
            )
            pf_positions[i] = ep["wls_ecef"]

    # Forward-backward smoother
    try:
        smoothed, forward = pf.smooth(position_update_sigma=POS_UPDATE_SIGMA)
        final_positions = smoothed
    except Exception:
        final_positions = pf_positions

    unix_ms = epochs_to_unix_ms(epochs)
    tdcp_pct = 100 * tdcp_count / max(n_epochs - 1, 1) if n_epochs > 1 else 0
    print(f"  {label}: {n_epochs} ep, TDCP {tdcp_pct:.0f}%")
    return final_positions, unix_ms


# ---------------------------------------------------------------------------
# Submission matching
# ---------------------------------------------------------------------------
def match_submission_times(
    sub_times_ms: np.ndarray, pf_ecef: np.ndarray, pf_times_ms: np.ndarray,
) -> np.ndarray:
    n = len(sub_times_ms)
    result = np.zeros((n, 2))
    for i in range(n):
        diffs = np.abs(pf_times_ms - sub_times_ms[i])
        idx = int(np.argmin(diffs))
        x, y, z = pf_ecef[idx]
        lat_rad, lon_rad, _ = ecef_to_lla(x, y, z)
        result[i, 0] = math.degrees(lat_rad)
        result[i, 1] = math.degrees(lon_rad)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("  GSDC 2023 Submission v3 (Smoother + bounded Hatch + TDCP)")
    print(f"  Particles: {N_PARTICLES:,}  SIGMA_PR: {SIGMA_PR}")
    print("=" * 80)

    sub_df = pd.read_csv(SAMPLE_SUB)
    trip_ids = sub_df["tripId"].unique()
    print(f"\n  {len(trip_ids)} tripIds, {len(sub_df)} total rows\n")

    out_rows = []
    t0_total = time.perf_counter()

    for trip_idx, trip_id in enumerate(trip_ids):
        run_name, phone_name = trip_id.split("/")
        data_dir = TEST_DIR / run_name / phone_name
        label = f"[{trip_idx + 1:2d}/{len(trip_ids)}] {trip_id}"

        t0 = time.perf_counter()
        result = run_pf_test(data_dir, label)

        if result is None:
            sub_trip = sub_df[sub_df["tripId"] == trip_id]
            for _, row in sub_trip.iterrows():
                out_rows.append({
                    "tripId": trip_id,
                    "UnixTimeMillis": int(row["UnixTimeMillis"]),
                    "LatitudeDegrees": row["LatitudeDegrees"],
                    "LongitudeDegrees": row["LongitudeDegrees"],
                })
            print(f"  {label}: FALLBACK")
            continue

        pf_ecef, pf_times_ms = result
        sub_trip = sub_df[sub_df["tripId"] == trip_id]
        sub_times = sub_trip["UnixTimeMillis"].values.astype(np.float64)

        latlon = match_submission_times(sub_times, pf_ecef, pf_times_ms)
        elapsed = time.perf_counter() - t0

        for j, (_, row) in enumerate(sub_trip.iterrows()):
            out_rows.append({
                "tripId": trip_id,
                "UnixTimeMillis": int(row["UnixTimeMillis"]),
                "LatitudeDegrees": latlon[j, 0],
                "LongitudeDegrees": latlon[j, 1],
            })

        print(f"    {len(sub_trip)} rows, {elapsed:.1f}s")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.perf_counter() - t0_total
    print(f"\n  Output: {OUTPUT_CSV}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
