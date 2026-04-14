#!/usr/bin/env python3
"""Generate Kaggle GSDC 2023 submission CSV using GPU Particle Filter.

v3: Forward-backward smoother + bounded Hatch filter + TDCP velocity.
Built on v1 baseline with careful divergence protection.

Usage
-----
  cd /workspace/ai_coding_ws/gnss_gpu
  PYTHONPATH=python python3 experiments/exp_gsdc2023_submission.py
  PYTHONPATH=python python3 experiments/exp_gsdc2023_submission.py \
      --label v12-smoother-only \
      --disable-hatch --disable-tdcp \
      --output experiments/results/gsdc2023_submission_v12.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
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
DEFAULT_OUTPUT_CSV = _SCRIPT_DIR / "results" / "gsdc2023_submission_v3.csv"

N_PARTICLES = 100_000
SIGMA_POS = 10.0              # v1 baseline
SIGMA_CB = 300.0
SIGMA_PR = 15.0               # v1 baseline
POS_UPDATE_SIGMA = 3.0        # v1 baseline
ELEV_THRESHOLD = 15.0

SIGMA_POS_TDCP = 5.0          # moderate
HATCH_MAX_WINDOW = 20
HATCH_DIVERGE_THRESHOLD = 10.0
CN0_THRESHOLD = 20.0
DIVERGENCE_RESET_THRESHOLD_M = 500.0

SIGNAL_TYPES = ["GPS_L1_CA", "GPS_L5_Q", "GAL_E1_C_P", "GAL_E5A_Q"]

# GPS time -> Unix time
GPS_UNIX_OFFSET_NS = 315964800 * 1_000_000_000
LEAP_SECONDS_NS = 18 * 1_000_000_000


@dataclass(frozen=True)
class SubmissionConfig:
    output_csv: Path
    use_hatch: bool = True
    use_tdcp: bool = True
    enable_smoother: bool = True
    label: str = "v3"


def parse_args() -> SubmissionConfig:
    parser = argparse.ArgumentParser(
        description="Generate Kaggle GSDC 2023 submission CSV using GPU PF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output submission CSV path.",
    )
    parser.add_argument(
        "--label",
        default="v3",
        help="Short label shown in logs for this submission variant.",
    )
    parser.add_argument(
        "--disable-hatch",
        action="store_true",
        help="Skip Hatch carrier smoothing and use raw corrected pseudorange.",
    )
    parser.add_argument(
        "--disable-tdcp",
        action="store_true",
        help="Skip TDCP velocity and use Doppler-only predict.",
    )
    parser.add_argument(
        "--disable-smoother",
        action="store_true",
        help="Disable backward smoothing and output forward PF only.",
    )
    args = parser.parse_args()
    return SubmissionConfig(
        output_csv=args.output,
        use_hatch=not args.disable_hatch,
        use_tdcp=not args.disable_tdcp,
        enable_smoother=not args.disable_smoother,
        label=args.label,
    )


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
        # bit1: ADR_STATE_RESET, bit2: ADR_STATE_CYCLE_SLIP
        # NOTE: bit4 (HALF_CYCLE_REPORTED) is NOT a slip — it's set for ALL measurements
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
# Cycle slip detector (Doppler-carrier + ADR jump + state bits)
# ---------------------------------------------------------------------------
class CycleSlipDetector:
    """Enhanced cycle slip detection using multiple criteria.

    1. ADR state bits (reset, cycle_slip, half_cycle_reported) — from parser
    2. Doppler-carrier consistency: |(-pr_rate_avg * dt) - delta_adr| > 1.5m
    3. ADR jump: |delta_adr| > 50000m
    """

    def __init__(self):
        self._prev_adr: dict[tuple, float] = {}
        self._prev_pr_rate: dict[tuple, float] = {}
        self._prev_arrival_ns: dict[tuple, int] = {}

    def detect(self, epoch: dict) -> np.ndarray:
        n_sv = epoch["n_sv"]
        slip = epoch["cycle_slip"].copy()
        arrival_ns = epoch["arrival_ns"]

        for i in range(n_sv):
            sid = epoch["sat_ids"][i]

            if not epoch["adr_valid"][i] or epoch["adr"][i] == 0.0:
                slip[i] = True
                continue

            adr_cur = epoch["adr"][i]
            pr_rate_cur = epoch["pr_rate"][i]

            if sid in self._prev_adr:
                dt = (arrival_ns - self._prev_arrival_ns[sid]) / 1e9
                if 0 < dt < 10:
                    adr_prev = self._prev_adr[sid]

                    # ADR jump check
                    if abs(adr_cur - adr_prev) > 50000.0:
                        slip[i] = True

                    # Doppler-carrier consistency: pr_rate*dt should match delta_adr
                    pr_rate_prev = self._prev_pr_rate.get(sid, pr_rate_cur)
                    pr_rate_avg = (pr_rate_cur + pr_rate_prev) / 2.0
                    predicted = pr_rate_avg * dt  # same sign convention as delta_adr
                    actual = adr_cur - adr_prev
                    if abs(predicted - actual) > 1.5:
                        slip[i] = True
                elif dt >= 10:
                    slip[i] = True

            self._prev_adr[sid] = adr_cur
            self._prev_pr_rate[sid] = pr_rate_cur
            self._prev_arrival_ns[sid] = arrival_ns

        return slip


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

    # Iterative outlier rejection: postfit residual > 1.0m
    H_sub = H[mask]
    b_sub = b[mask]
    for _ in range(3):
        pred = H_sub @ x
        resid = np.abs(b_sub - pred)
        good = resid <= 1.0
        if good.sum() >= 5 and good.sum() < len(b_sub):
            H_sub = H_sub[good]
            b_sub = b_sub[good]
            try:
                x, _, _, _ = np.linalg.lstsq(H_sub, b_sub, rcond=None)
            except np.linalg.LinAlgError:
                break
        else:
            break

    velocity = x[:3] / dt

    # Sanity: reject unreasonable speed
    if np.linalg.norm(velocity) > 80.0:
        return None

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


def compute_robust_wls(sat_ecef: np.ndarray, pseudoranges: np.ndarray,
                       weights: np.ndarray, init_pos: np.ndarray,
                       n_iter: int = 5) -> np.ndarray:
    """Iteratively Reweighted Least Squares (IRLS) with Huber loss.

    Returns ECEF position [3] or init_pos if solve fails.
    """
    pos = init_pos.copy()
    n_sat = len(pseudoranges)
    if n_sat < 5:
        return init_pos.copy()

    # Initialize clock bias from init_pos
    init_ranges = np.linalg.norm(sat_ecef - pos, axis=1)
    cb = float(np.median(pseudoranges - init_ranges))

    for iteration in range(n_iter):
        ranges = np.linalg.norm(sat_ecef - pos, axis=1)
        ranges = np.maximum(ranges, 1.0)
        residuals = pseudoranges - ranges - cb

        # Huber weighting: k = 20m threshold
        k = 20.0
        huber_w = np.where(np.abs(residuals) <= k, 1.0, k / np.abs(residuals))

        w = weights * huber_w

        los = (sat_ecef - pos) / ranges[:, np.newaxis]
        H = np.column_stack([-los, np.ones(n_sat)])
        W = np.diag(w)

        try:
            HtWH = H.T @ W @ H
            HtWy = H.T @ W @ residuals
            dx = np.linalg.solve(HtWH, HtWy)
        except np.linalg.LinAlgError:
            return init_pos.copy()

        pos += dx[:3]
        cb += dx[3]

        if np.linalg.norm(dx[:3]) < 0.01:
            break

    # Safety: if diverged far from init, fall back
    if np.linalg.norm(pos - init_pos) > 100.0:
        return init_pos.copy()

    return pos


def epochs_to_unix_ms(epochs: list[dict]) -> np.ndarray:
    arr = np.array([ep["arrival_ns"] for ep in epochs], dtype=np.float64)
    unix_ns = arr + GPS_UNIX_OFFSET_NS - LEAP_SECONDS_NS
    return unix_ns / 1_000_000


def estimate_clock_bias_from_reference(
    sat_ecef: np.ndarray, pseudoranges: np.ndarray, ref_ecef: np.ndarray,
) -> float:
    ranges = np.linalg.norm(sat_ecef - ref_ecef, axis=1)
    return float(np.median(pseudoranges - ranges))


def reinitialize_pf_at_reference(
    pf,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    ref_ecef: np.ndarray,
    spread_pos: float = 50.0,
    spread_cb: float = 500.0,
) -> None:
    cb = estimate_clock_bias_from_reference(sat_ecef, pseudoranges, ref_ecef)
    pf.initialize(ref_ecef, clock_bias=cb, spread_pos=spread_pos, spread_cb=spread_cb)


def capture_smoother_epoch(
    estimate: np.ndarray,
    sat_ecef: np.ndarray,
    pseudoranges: np.ndarray,
    weights: np.ndarray,
    velocity: np.ndarray | None,
    dt: float,
    spp_ref: np.ndarray,
    predict_sigma_pos: float,
    segment_start: bool,
) -> dict:
    return {
        "estimate": np.asarray(estimate, dtype=np.float64).copy(),
        "sat_ecef": np.asarray(sat_ecef, dtype=np.float64).copy(),
        "pseudoranges": np.asarray(pseudoranges, dtype=np.float64).copy(),
        "weights": np.asarray(weights, dtype=np.float64).copy(),
        "velocity": (
            np.asarray(velocity, dtype=np.float64).copy()
            if velocity is not None else None
        ),
        "dt": float(dt),
        "spp_ref": np.asarray(spp_ref, dtype=np.float64).copy(),
        "predict_sigma_pos": float(predict_sigma_pos),
        "segment_start": bool(segment_start),
    }


def smooth_segmented_epochs(
    stored_epochs: list[dict], position_update_sigma: float | None,
) -> np.ndarray:
    """Run backward smoothing on contiguous reset-free segments only.

    GSDC occasionally needs forward divergence resets back to the dataset WLS.
    A single smoother over the whole trip will incorrectly propagate pre-reset
    state across those discontinuities. Split the trip into independent segments
    and smooth each segment separately.
    """
    if not stored_epochs:
        return np.zeros((0, 3), dtype=np.float64)

    from gnss_gpu.particle_filter_device import ParticleFilterDevice

    segment_starts = [
        i for i, ep in enumerate(stored_epochs) if ep.get("segment_start", False)
    ]
    if not segment_starts or segment_starts[0] != 0:
        segment_starts.insert(0, 0)

    smoothed = np.zeros((len(stored_epochs), 3), dtype=np.float64)

    for seg_idx, start in enumerate(segment_starts):
        end = (
            segment_starts[seg_idx + 1]
            if seg_idx + 1 < len(segment_starts)
            else len(stored_epochs)
        )
        segment = stored_epochs[start:end]
        forward_pos = np.array([ep["estimate"] for ep in segment], dtype=np.float64)
        if len(segment) == 1:
            smoothed[start:end] = forward_pos
            continue

        last = segment[-1]
        bwd_pf = ParticleFilterDevice(
            n_particles=N_PARTICLES,
            sigma_pos=SIGMA_POS,
            sigma_cb=SIGMA_CB,
            sigma_pr=SIGMA_PR,
            resampling="megopolis",
            ess_threshold=0.5,
            seed=42 + seg_idx + 1,
        )
        reinitialize_pf_at_reference(
            bwd_pf,
            last["sat_ecef"],
            last["pseudoranges"],
            last["estimate"],
            spread_pos=10.0,
            spread_cb=100.0,
        )

        backward_pos = np.zeros_like(forward_pos)
        backward_pos[-1] = last["estimate"]

        for i in range(len(segment) - 2, -1, -1):
            next_ep = segment[i + 1]
            ep = segment[i]

            vel = -next_ep["velocity"] if next_ep["velocity"] is not None else None
            bwd_pf.predict(
                velocity=vel,
                dt=next_ep["dt"],
                sigma_pos=next_ep["predict_sigma_pos"],
            )

            sat = ep["sat_ecef"].reshape(-1, 3)
            pr = ep["pseudoranges"]
            w = ep["weights"]
            bwd_pf.correct_clock_bias(sat, pr)
            bwd_pf.update_gmm(
                sat,
                pr,
                weights=w,
                sigma_pr=SIGMA_PR,
                w_los=0.7,
                mu_nlos=15.0,
                sigma_nlos=30.0,
            )

            if position_update_sigma is not None:
                bwd_pf.position_update(ep["spp_ref"][:3], sigma_pos=position_update_sigma)

            est = bwd_pf.estimate()[:3]
            if np.linalg.norm(est - ep["spp_ref"][:3]) > DIVERGENCE_RESET_THRESHOLD_M:
                reinitialize_pf_at_reference(
                    bwd_pf,
                    sat,
                    pr,
                    ep["spp_ref"][:3],
                    spread_pos=10.0,
                    spread_cb=100.0,
                )
                est = ep["spp_ref"][:3]

            backward_pos[i] = est

        smoothed[start:end] = (forward_pos + backward_pos) / 2.0

    return smoothed


# ---------------------------------------------------------------------------
# PF pipeline
# ---------------------------------------------------------------------------
def run_pf_test(
    data_dir: Path,
    label: str,
    config: SubmissionConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
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

    need_cycle_slip = config.use_hatch or config.use_tdcp
    cycle_slip_detector = CycleSlipDetector() if need_cycle_slip else None
    hatch = None
    if config.use_hatch:
        hatch = HatchFilter(
            max_window=HATCH_MAX_WINDOW,
            diverge_threshold=HATCH_DIVERGE_THRESHOLD,
        )

    pf_positions = np.zeros((n_epochs, 3))
    smooth_epochs: list[dict] = []
    prev_arrival_ns = epochs[0]["arrival_ns"]
    tdcp_count = 0
    reset_count = 0

    for i, ep in enumerate(epochs):
        dt = (ep["arrival_ns"] - prev_arrival_ns) / 1e9 if i > 0 else 1.0
        dt = max(dt, 0.01)
        prev_arrival_ns = ep["arrival_ns"]

        if cycle_slip_detector is not None:
            ep["cycle_slip"] = cycle_slip_detector.detect(ep)

        if hatch is not None:
            smoothed_pr = hatch.smooth(
                ep["sat_ids"], ep["pseudoranges"], ep["adr"],
                ep["adr_valid"], ep["cycle_slip"],
            )
        else:
            smoothed_pr = ep["pseudoranges"]

        # TDCP velocity (preferred), Doppler fallback
        rx_ecef = pf_positions[i - 1] if i > 0 else init_pos
        sigma_pos = SIGMA_POS
        velocity = None
        if i > 0 and config.use_tdcp:
            velocity = compute_tdcp_velocity(ep, epochs[i - 1], rx_ecef, dt)
            if velocity is not None:
                sigma_pos = SIGMA_POS_TDCP
                tdcp_count += 1
        if velocity is None:
            velocity = compute_doppler_velocity(ep, rx_ecef)

        # PF steps (v1 params)
        pf.predict(velocity=velocity, dt=dt, sigma_pos=sigma_pos)
        pf.correct_clock_bias(ep["sat_ecef"], smoothed_pr)

        weights = elevation_weights(ep["elevations"], ep["cn0"], ELEV_THRESHOLD)
        pf.update_gmm(
            ep["sat_ecef"], smoothed_pr, weights=weights,
            sigma_pr=SIGMA_PR, w_los=0.7, mu_nlos=15.0, sigma_nlos=30.0,
        )
        # Use dataset WLS for position update (Sagnac-corrected)
        pf.position_update(ep["wls_ecef"], sigma_pos=POS_UPDATE_SIGMA)

        est = pf.estimate()
        est_pos = est[:3].copy()
        segment_start = (i == 0)

        # Divergence reset
        if np.linalg.norm(est_pos - ep["wls_ecef"]) > DIVERGENCE_RESET_THRESHOLD_M:
            reinitialize_pf_at_reference(
                pf,
                ep["sat_ecef"],
                ep["pseudoranges"],
                ep["wls_ecef"],
                spread_pos=50.0,
                spread_cb=500.0,
            )
            est_pos = ep["wls_ecef"].copy()
            segment_start = True
            reset_count += 1

        pf_positions[i] = est_pos
        smooth_epochs.append(capture_smoother_epoch(
            estimate=est_pos,
            sat_ecef=ep["sat_ecef"],
            pseudoranges=smoothed_pr,
            weights=weights,
            velocity=velocity,
            dt=dt,
            spp_ref=ep["wls_ecef"],
            predict_sigma_pos=sigma_pos,
            segment_start=segment_start,
        ))

    if config.enable_smoother:
        try:
            final_positions = smooth_segmented_epochs(
                smooth_epochs,
                position_update_sigma=POS_UPDATE_SIGMA,
            )
        except Exception:
            final_positions = pf_positions
    else:
        final_positions = pf_positions

    unix_ms = epochs_to_unix_ms(epochs)
    n_segments = sum(1 for ep in smooth_epochs if ep["segment_start"])
    velocity_label = "TDCP+Doppler" if config.use_tdcp else "Doppler"
    pr_label = "Hatch" if config.use_hatch else "rawPR"
    smooth_label = "seg-smth" if config.enable_smoother else "forward"
    print(
        f"  {label}: {n_epochs} ep, {pr_label}, {velocity_label}, {smooth_label}, "
        f"TDCP {tdcp_count}/{n_epochs - 1}, resets {reset_count}, segments {n_segments}"
    )
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
    config = parse_args()
    print("=" * 80)
    print(f"  GSDC 2023 Submission {config.label}")
    print(
        "  Features:"
        f" hatch={'on' if config.use_hatch else 'off'}"
        f" tdcp={'on' if config.use_tdcp else 'off'}"
        f" smoother={'on' if config.enable_smoother else 'off'}"
    )
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
        result = run_pf_test(data_dir, label, config)

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

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(config.output_csv, index=False)

    total_elapsed = time.perf_counter() - t0_total
    print(f"\n  Output: {config.output_csv}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
