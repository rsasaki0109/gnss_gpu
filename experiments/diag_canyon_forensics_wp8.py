"""WP8 work item 3: canyon (tow 188990-189070) float-divergence forensics.

Answers, with numbers, the three questions the task poses about the
tokyo/run1 urban-canyon segment that WP6/WP7 both flagged as the remaining
catastrophic failure mode (float error stays ~119 m under every tested
sigma-inflation NLOS mapping):

  (a) undetected cycle slip? -- summarizes the engine's own slip counters
      (gf_slip_count / doppler_slip_* / lli_slip_* / ambiguity_reset_*)
      from a ``--debug-epoch-log`` CSV across the divergence onset, and
      cross-references the raw rover RINEX LLI bits directly (independent
      of the engine's own slip detector, in case it misses something).
  (b) float covariance collapsing (overconfident) vs. wide-but-dragged?
      -- summarizes ``float_position_covariance_trace_m2`` (new WP8 debug
      column, see rtk.hpp EpochDebugTelemetry) and the float update's own
      prefit/post-suppression residual RMS + NIS-per-observation.
  (c) how many segment satellites are NLOS (phase-33 mask) and are the
      biggest raw pseudorange residuals on those? -- computes, independent
      of the RTK engine, per-satellite raw pseudorange residuals against
      the *known ground truth* position (reference.csv) with a robust
      per-epoch common-mode (receiver clock) removal, and cross-references
      each satellite's LOS/NLOS label from the phase-33 mask.

This is a read-only diagnostic script (no engine/solver changes); it feeds
WP8_REPORT.md's canyon mechanism verdict.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ppc_window_geometry import _read_obs_header  # noqa: E402

DEFAULT_DATA_ROOT = Path("E:/datasets/PPC-Dataset-data")
NLOS_MASK_DIR = _PROJECT_ROOT / "experiments" / "results" / "plateau_nlos_phase33"


# ---------------------------------------------------------------------------
# Debug-epoch-log summary (answers (a) slip counters, (b) covariance/NIS)
# ---------------------------------------------------------------------------

def load_debug_epoch_log(path: Path, tow_lo: float, tow_hi: float) -> list[dict]:
    """Loads ``--debug-epoch-log`` rows with tow in [tow_lo, tow_hi]."""
    rows = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tow = float(row["tow"])
            except (KeyError, ValueError):
                continue
            if tow_lo <= tow <= tow_hi:
                rows.append(row)
    return rows


def _to_float(row: dict, key: str) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def _to_int(row: dict, key: str) -> int:
    val = row.get(key, "")
    if val in ("", None):
        return 0
    try:
        return int(float(val))
    except ValueError:
        return 0


def summarize_debug_log(rows: list[dict]) -> dict:
    """Pure-function summary of a debug-epoch-log slice (unit-testable)."""
    if not rows:
        return {"n_epochs": 0}

    slip_fields = [
        "gf_slip_count", "doppler_slip_l1_count", "doppler_slip_l2_count",
        "code_slip_l1_count", "code_slip_l2_count",
        "lli_slip_l1_count", "lli_slip_l2_count",
        "ambiguity_reset_l1_count", "ambiguity_reset_l2_count",
    ]
    slip_totals = {f: sum(_to_int(r, f) for r in rows) for f in slip_fields}

    postfix_rms = [_to_float(r, "float_update_post_suppression_residual_rms_m") for r in rows]
    prefit_rms = [_to_float(r, "float_update_prefit_residual_rms_m") for r in rows]
    nis_per_obs = [_to_float(r, "float_update_nis_per_observation") for r in rows]
    cov_trace = [_to_float(r, "float_position_covariance_trace_m2") for r in rows]
    cov_trace_finite = np.array([v for v in cov_trace if np.isfinite(v)], dtype=np.float64)
    if cov_trace_finite.size:
        cov_regime = {
            "n": int(cov_trace_finite.size),
            # >500 m^2: resetPositionToSPP() fell back to its untrusted-seed
            # 900 m^2/axis prior this epoch (rtk.cpp resetPositionToSPP()), i.e.
            # the previous epoch's solution never refreshed "trusted" state.
            "frac_wide_untrusted_reset_gt500": float(np.mean(cov_trace_finite > 500.0)),
            "frac_partially_shrunk_50_500": float(
                np.mean((cov_trace_finite >= 50.0) & (cov_trace_finite <= 500.0))
            ),
            "frac_converged_lt50": float(np.mean(cov_trace_finite < 50.0)),
            "frac_fully_converged_lt1": float(np.mean(cov_trace_finite < 1.0)),
        }
    else:
        cov_regime = {"n": 0}
    num_sats = [_to_int(r, "num_sats") for r in rows]
    reject_reasons: dict[str, int] = {}
    for r in rows:
        reason = (r.get("reject_reason") or "").strip()
        if reason:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    def _finite_stats(values: list[float]) -> dict:
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return {"n": 0, "median": float("nan"), "min": float("nan"), "max": float("nan")}
        arr = np.asarray(finite, dtype=np.float64)
        return {
            "n": len(finite),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "n_epochs": len(rows),
        "tow_lo": min(_to_float(r, "tow") for r in rows),
        "tow_hi": max(_to_float(r, "tow") for r in rows),
        "slip_totals": slip_totals,
        "postfix_residual_rms_m": _finite_stats(postfix_rms),
        "prefit_residual_rms_m": _finite_stats(prefit_rms),
        "nis_per_observation": _finite_stats(nis_per_obs),
        "float_position_covariance_trace_m2": _finite_stats(cov_trace),
        "float_position_covariance_trace_regime": cov_regime,
        "num_sats": _finite_stats([float(v) for v in num_sats]),
        "reject_reasons": reject_reasons,
    }


# ---------------------------------------------------------------------------
# Raw LLI cross-reference (independent of the engine's own slip detector)
# ---------------------------------------------------------------------------

def scan_rover_lli(
    rover_obs_path: Path,
    *,
    start_tow: float,
    end_tow: float,
    obs_code_by_system: dict[str, str] | None = None,
) -> dict[str, list[tuple[float, int]]]:
    """Per-satellite list of (tow, lli) for a given phase obs code per epoch.

    Self-contained RINEX3 scan (does not modify python/gnss_gpu/io/rinex.py);
    only extracts the one phase channel's LLI digit per satellite, since
    that's all this diagnostic needs. Returns {sat_id: [(tow, lli), ...]}.
    """
    if obs_code_by_system is None:
        # First (primary) phase channel per system, matching the phase-33
        # mask's own single-frequency-classification convention.
        obs_code_by_system = {"G": "L1C", "E": "L1C", "R": "L1P", "C": "L1P", "J": "L1C"}

    with rover_obs_path.open() as f:
        lines = f.readlines()
    obs_types, _approx_position, idx = _read_obs_header(lines)

    from ppc_window_geometry import _datetime_to_tow, _looks_like_sat_id, _normalize_sat_id
    from datetime import datetime

    result: dict[str, list[tuple[float, int]]] = {}
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith(">"):
            idx += 1
            continue
        parts = line[2:].split()
        if len(parts) < 8:
            idx += 1
            continue
        try:
            sec = float(parts[5])
            sec_int = int(sec)
            usec = int(round((sec - sec_int) * 1e6))
            epoch_time = datetime(
                int(parts[0]), int(parts[1]), int(parts[2]),
                int(parts[3]), int(parts[4]), sec_int, usec,
            )
            epoch_flag = int(parts[6])
            n_sat = int(parts[7])
        except (ValueError, IndexError):
            idx += 1
            continue
        tow = round(_datetime_to_tow(epoch_time), 1)
        if tow > end_tow:
            break
        idx += 1
        if epoch_flag > 1 or tow < start_tow:
            idx += n_sat
            continue

        for _ in range(n_sat):
            if idx >= len(lines):
                break
            obs_line = lines[idx]
            sat_id = _normalize_sat_id(obs_line[:3])
            sys_char = sat_id[0] if sat_id else ""
            obs_codes = obs_types.get(sys_char, [])
            obs_record = obs_line.rstrip("\n")
            target_len = 3 + 16 * len(obs_codes)
            while len(obs_record) < target_len and idx + 1 < len(lines):
                next_line = lines[idx + 1]
                next_id = next_line[:3].strip()
                if next_line.startswith(">") or _looks_like_sat_id(next_id):
                    break
                idx += 1
                obs_record += lines[idx][3:].rstrip("\n")
            idx += 1

            wanted_code = obs_code_by_system.get(sys_char)
            if not sat_id or wanted_code is None or wanted_code not in obs_codes:
                continue
            col = obs_codes.index(wanted_code)
            pos = 3 + 16 * col
            value_field = obs_record[pos : pos + 14] if pos + 14 <= len(obs_record) else ""
            lli_field = obs_record[pos + 14 : pos + 15] if pos + 15 <= len(obs_record) else ""
            if not value_field.strip():
                continue  # no observation this epoch, LLI meaningless
            try:
                lli = int(lli_field) if lli_field.strip() else 0
            except ValueError:
                lli = 0
            result.setdefault(sat_id, []).append((tow, lli))
    return result


def summarize_lli(lli_by_sat: dict[str, list[tuple[float, int]]]) -> dict:
    """LLI bit 0 (value 1/3/5/7) marks possible loss-of-lock / slip."""
    summary = {}
    for sat_id, series in lli_by_sat.items():
        slip_events = [(tow, lli) for tow, lli in series if lli % 2 == 1]
        summary[sat_id] = {
            "n_obs": len(series),
            "n_slip_flagged": len(slip_events),
            "slip_tows": [tow for tow, _ in slip_events],
        }
    return summary


# ---------------------------------------------------------------------------
# Raw pseudorange residual vs ground truth, cross-referenced with NLOS mask
# ---------------------------------------------------------------------------

def load_nlos_mask(city: str, run: str, tow_lo: float, tow_hi: float) -> dict[tuple[float, str], bool]:
    """{(nearest_tow, sat_id): is_los} for the phase-33 mask in [tow_lo, tow_hi]."""
    path = NLOS_MASK_DIR / f"{city}_{run}_per_epoch_nlos.csv"
    result: dict[tuple[float, str], bool] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                tow = float(row["tow"])
            except (KeyError, ValueError):
                continue
            if not (tow_lo <= tow <= tow_hi):
                continue
            prn = row.get("prn", "").strip()
            if not prn:
                continue
            is_los = row.get("is_los", "1").strip() not in ("0", "0.0", "False", "false")
            result[(round(tow, 1), prn)] = is_los
    return result


def raw_pseudorange_residuals(
    run_dir: Path,
    *,
    start_tow: float,
    end_tow: float,
    systems: tuple[str, ...] = ("G", "R", "E", "C", "J"),
) -> list[dict]:
    """Per-(epoch, sat) raw pseudorange residual vs ground truth.

    residual_m = (pseudorange - geometric_range_to_truth) minus the
    per-epoch median across all tracked satellites (a robust, model-free
    proxy for the unknown receiver clock bias + any shared common-mode
    error) -- what's left over is dominated by each satellite's own
    multipath/NLOS bias, independent of the RTK engine's own float filter.
    """
    from ppc_window_geometry import load_ppc_window_geometry

    geometry = load_ppc_window_geometry(
        run_dir, start_tow=start_tow, end_tow=end_tow, systems=systems,
        transmit_time_iterations=2,
    )
    rows = []
    for tow, sat_ecef, pr, truth, sat_ids in zip(
        geometry["times"], geometry["sat_ecef"], geometry["pseudoranges"],
        geometry["ground_truth"], geometry["used_prns"],
    ):
        ranges = np.linalg.norm(sat_ecef - truth[None, :], axis=1)
        raw_residual = pr - ranges
        common_mode = float(np.median(raw_residual))
        for sat_id, res in zip(sat_ids, raw_residual):
            rows.append({
                "tow": float(tow),
                "sat_id": sat_id,
                "raw_residual_m": float(res),
                "common_mode_removed_residual_m": float(res - common_mode),
            })
    return rows


def summarize_residuals_by_los(
    residual_rows: list[dict], nlos_mask: dict[tuple[float, str], bool], tow_tolerance: float = 0.3
) -> dict:
    """Splits |common_mode_removed_residual_m| by LOS/NLOS label."""
    mask_tows = sorted({tow for tow, _ in nlos_mask})

    def _nearest_label(tow: float, sat_id: str) -> bool | None:
        best = None
        best_delta = tow_tolerance
        for mtow in mask_tows:
            delta = abs(mtow - tow)
            if delta <= best_delta and (mtow, sat_id) in nlos_mask:
                best = nlos_mask[(mtow, sat_id)]
                best_delta = delta
        return best

    los_abs = []
    nlos_abs = []
    unknown_abs = []
    for row in residual_rows:
        label = _nearest_label(row["tow"], row["sat_id"])
        magnitude = abs(row["common_mode_removed_residual_m"])
        if label is None:
            unknown_abs.append(magnitude)
        elif label:
            los_abs.append(magnitude)
        else:
            nlos_abs.append(magnitude)

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0, "median": float("nan"), "mean": float("nan"), "max": float("nan")}
        arr = np.asarray(values, dtype=np.float64)
        return {"n": len(values), "median": float(np.median(arr)), "mean": float(np.mean(arr)), "max": float(np.max(arr))}

    return {"los": _stats(los_abs), "nlos": _stats(nlos_abs), "unknown": _stats(unknown_abs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", default="tokyo")
    parser.add_argument("--run", default="run1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--debug-epoch-log", type=Path, required=True)
    parser.add_argument("--tow-lo", type=float, default=188925.0, help="60s lead-in before 188985")
    parser.add_argument("--tow-hi", type=float, default=189075.0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    run_dir = args.data_root / args.city / args.run

    debug_rows = load_debug_epoch_log(args.debug_epoch_log, args.tow_lo, args.tow_hi)
    debug_summary = summarize_debug_log(debug_rows)
    print(f"[debug-log] {len(debug_rows)} epochs in [{args.tow_lo}, {args.tow_hi}]")
    print(f"[debug-log] slip totals: {debug_summary.get('slip_totals')}")
    print(f"[debug-log] postfix_residual_rms_m: {debug_summary.get('postfix_residual_rms_m')}")
    print(f"[debug-log] float_position_covariance_trace_m2: {debug_summary.get('float_position_covariance_trace_m2')}")
    print(f"[debug-log] float_position_covariance_trace_regime: {debug_summary.get('float_position_covariance_trace_regime')}")
    print(f"[debug-log] nis_per_observation: {debug_summary.get('nis_per_observation')}")
    print(f"[debug-log] reject_reasons: {debug_summary.get('reject_reasons')}")

    lli_by_sat = scan_rover_lli(run_dir / "rover.obs", start_tow=args.tow_lo, end_tow=args.tow_hi)
    lli_summary = summarize_lli(lli_by_sat)
    n_slip_flagged_sats = sum(1 for s in lli_summary.values() if s["n_slip_flagged"] > 0)
    print(f"[lli] {len(lli_summary)} tracked satellites, {n_slip_flagged_sats} with >=1 LLI slip flag")

    nlos_mask = load_nlos_mask(args.city, args.run, args.tow_lo, args.tow_hi)
    n_nlos_sat_epochs = sum(1 for v in nlos_mask.values() if not v)
    print(f"[nlos] {len(nlos_mask)} (tow,sat) mask entries, {n_nlos_sat_epochs} NLOS")

    residual_rows = raw_pseudorange_residuals(run_dir, start_tow=args.tow_lo, end_tow=args.tow_hi)
    residual_by_los = summarize_residuals_by_los(residual_rows, nlos_mask)
    print(f"[residuals] |common-mode-removed residual| by LOS/NLOS: {residual_by_los}")

    if args.out_json:
        import json

        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as fh:
            json.dump(
                {
                    "debug_log_summary": debug_summary,
                    "lli_summary": {k: v for k, v in lli_summary.items()},
                    "n_nlos_mask_entries": len(nlos_mask),
                    "n_nlos_sat_epochs": n_nlos_sat_epochs,
                    "residual_by_los": residual_by_los,
                },
                fh,
                indent=2,
                default=str,
            )
        print(f"[out] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
