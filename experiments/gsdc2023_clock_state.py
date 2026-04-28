"""Clock policy and segmentation helpers for GSDC2023 raw bridge."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.gsdc2023_observation_matrix import interpolate_series
from experiments.gsdc2023_residual_model import gradient_with_dt


MULTI_GNSS_BLOCKLIST_PHONES = {"mi8", "xiaomimi8"}
CLOCK_DRIFT_BLOCKLIST_PHONES = {
    "sm-a226b",
    "sm-a505g",
    "sm-a600t",
    "sm-a325f",
    "sm-a217m",
    "sm-a205u",
    "samsunga325g",
    "samsunga32",
    "sm-a505u",
}
CLOCK_DRIFT_SEED_BLOCKLIST_PHONES = {"sm-a505u"}
CLOCK_AID_PHONES = {"pixel4"}


def effective_multi_gnss_enabled(trip: str, requested: bool) -> bool:
    if not requested:
        return False
    phone = Path(trip).name.lower()
    return phone not in MULTI_GNSS_BLOCKLIST_PHONES


def effective_position_source(trip: str, requested: str) -> str:
    phone = Path(trip).name.lower()
    if phone in MULTI_GNSS_BLOCKLIST_PHONES and requested == "auto":
        return "raw_wls"
    return requested


def clock_aid_enabled(phone: str) -> bool:
    phone_l = phone.lower()
    return phone_l in CLOCK_AID_PHONES or phone_l in CLOCK_DRIFT_BLOCKLIST_PHONES


def clock_drift_seed_enabled(phone: str) -> bool:
    phone_l = phone.lower()
    return clock_aid_enabled(phone_l) and phone_l not in CLOCK_DRIFT_SEED_BLOCKLIST_PHONES


def clock_jump_threshold_m(phone: str) -> float:
    phone = phone.lower()
    if phone in {
        "sm-s908b",
        "sm-a226b",
        "samsungs22ultra",
        "sm-a325f",
        "samsunga325g",
        "samsunga32",
        "pixel7pro",
        "sm-a505u",
    }:
        return 2000.0
    if phone in {"sm-a205u", "sm-a217m", "sm-a505g", "sm-a600t", "sm-g988b", "pixel6pro"}:
        return 500.0
    if phone in {"pixel4xl", "pixel7"}:
        return 100.0
    return 50.0


def clean_clock_drift(
    times_ms: np.ndarray,
    clock_bias_m: np.ndarray | None,
    clock_drift_mps: np.ndarray | None,
    phone: str,
) -> np.ndarray | None:
    if clock_bias_m is None and clock_drift_mps is None:
        return None

    times_s = np.asarray(times_ms, dtype=np.float64) * 1e-3
    clock = None
    if clock_bias_m is not None:
        clock = np.asarray(clock_bias_m, dtype=np.float64).copy()
        if np.isfinite(clock).any():
            clock = interpolate_series(times_s, clock)
        else:
            clock = None

    if clock_drift_mps is None:
        drift = np.full(times_s.size, np.nan, dtype=np.float64)
    else:
        drift = np.asarray(clock_drift_mps, dtype=np.float64).copy()

    phone_l = phone.lower()
    if (phone_l in MULTI_GNSS_BLOCKLIST_PHONES or not np.isfinite(drift).any()) and clock is not None:
        drift = -gradient_with_dt(clock, times_s)

    if not np.isfinite(drift).any():
        return None

    drift[np.abs(drift) > 1e3] = np.nan
    if np.isfinite(drift).any():
        diff = np.abs(np.diff(drift))
        idx = np.flatnonzero(diff > 50.0)
        if idx.size > 0:
            bad = np.zeros(drift.size, dtype=bool)
            bad[idx] = True
            bad[idx + 1] = True
            drift[bad] = np.nan
    if np.isfinite(drift).any():
        drift = interpolate_series(times_s, drift)
    else:
        return None
    return drift


def detect_clock_jumps_from_clock_bias(clock_bias_m: np.ndarray, phone: str) -> np.ndarray:
    bias = np.asarray(clock_bias_m, dtype=np.float64)
    jumps = np.zeros(bias.size, dtype=bool)
    if bias.size <= 1:
        return jumps
    threshold_m = clock_jump_threshold_m(phone)
    diff = np.abs(np.diff(bias))
    jumps[1:] = np.isfinite(diff) & (diff > threshold_m)
    return jumps


def combine_clock_jump_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    combined: np.ndarray | None = None
    for mask in masks:
        if mask is None:
            continue
        arr = np.asarray(mask, dtype=bool)
        if combined is None:
            combined = arr.copy()
        else:
            combined |= arr
    return combined


def segment_ranges(start: int, end: int, clock_jump: np.ndarray | None) -> list[tuple[int, int]]:
    if end <= start:
        return []
    if clock_jump is None or end - start <= 1:
        return [(start, end)]
    split_points = np.flatnonzero(np.asarray(clock_jump[start + 1 : end], dtype=bool)) + start + 1
    boundaries = [start, *split_points.tolist(), end]
    segments: list[tuple[int, int]] = []
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        if seg_end > seg_start:
            segments.append((int(seg_start), int(seg_end)))
    return segments


def factor_break_mask(clock_jump: np.ndarray | None, dt: np.ndarray | None, n_epoch: int) -> np.ndarray | None:
    break_mask = None
    if clock_jump is not None:
        break_mask = np.asarray(clock_jump, dtype=bool).copy()
        if break_mask.size != n_epoch:
            padded = np.zeros(n_epoch, dtype=bool)
            n = min(n_epoch, break_mask.size)
            padded[:n] = break_mask[:n]
            break_mask = padded
    if dt is not None and n_epoch > 1:
        dt_arr = np.asarray(dt, dtype=np.float64).reshape(-1)
        n_dt = min(n_epoch - 1, dt_arr.size)
        invalid_dt = (~np.isfinite(dt_arr[:n_dt])) | (dt_arr[:n_dt] <= 0.0)
        if invalid_dt.any():
            if break_mask is None:
                break_mask = np.zeros(n_epoch, dtype=bool)
            break_mask[1 : n_dt + 1] |= invalid_dt
    return break_mask


__all__ = [
    "CLOCK_AID_PHONES",
    "CLOCK_DRIFT_BLOCKLIST_PHONES",
    "CLOCK_DRIFT_SEED_BLOCKLIST_PHONES",
    "MULTI_GNSS_BLOCKLIST_PHONES",
    "clean_clock_drift",
    "clock_aid_enabled",
    "clock_drift_seed_enabled",
    "clock_jump_threshold_m",
    "combine_clock_jump_masks",
    "detect_clock_jumps_from_clock_bias",
    "effective_multi_gnss_enabled",
    "effective_position_source",
    "factor_break_mask",
    "segment_ranges",
]
