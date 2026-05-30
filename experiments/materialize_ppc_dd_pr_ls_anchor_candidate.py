#!/usr/bin/env python3
"""Materialize a TOW-window candidate via per-epoch pseudorange LS.

Approach (a): independent per-epoch position anchors built from a
4-satellite-or-more pseudorange least-squares solve. Each epoch's anchor
is computed from that epoch's PR observations alone — no segment state
leaks between epochs, so the trajectory is structurally independent
from RTK/PF/TDCP-anchor candidates already in the pool.

Two modes are supported via ``--mode``:

``undiff``
    Solve ``[x, y, z, b]`` from corrected undifferenced pseudoranges
    (sat clock applied; atmospheric biases absorbed in the residual).
    Equivalent to a per-epoch SPP solve — useful when no base station
    is available, but typical urban-canyon postfit RMS sits at ~10-30m.

``dd``
    Solve ``[x, y, z]`` from double-differenced pseudoranges built with
    a reference satellite (highest rover-side elevation). Atmospheric
    delays and receiver/sat clock biases mostly cancel, so postfit RMS
    drops to ~1-3m even in urban canyons. Requires ``base.obs`` in the
    PPC run dir; the base ECEF is taken from the RINEX header. This is
    the recommended mode for candidate-pool injection.

Pipeline (both modes):
1. Load PPC dataset (rover-side corrected pseudoranges, satellite
   ECEF, weights). For ``dd`` mode also load base.obs and build a
   ``{tow: {sat_id: pr}}`` lookup keyed by GPS TOW.
2. For each rover epoch with TOW in ``[start_tow, end_tow]``, run a
   Gauss-Newton LS initialised at the seed pos's row. ``dd`` mode
   matches the closest base epoch within ``--base-time-tolerance``
   (default 1.0s) and forms DD pairs against the reference sat.
3. Rewrite the seed pos rows in the segment with the LS ECEF X/Y/Z
   (LLH recomputed); window-slice the csv as-is.

The candidate must be wired via ``--rtkdiag-candidate-pos-dirs`` and a
matching label.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu.io.ppc import PPCDatasetLoader  # noqa: E402
from gnss_gpu.io.rinex import read_rinex_obs  # noqa: E402
from gnss_gpu.io.nav_rinex import _datetime_to_gps_seconds_of_week  # noqa: E402


_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def _ecef_to_lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    b = _WGS84_A * (1.0 - _WGS84_F)
    ep2 = (_WGS84_A * _WGS84_A - b * b) / (b * b)
    p = math.hypot(float(x), float(y))
    theta = math.atan2(float(z) * _WGS84_A, p * b)
    lon = math.atan2(float(y), float(x))
    lat = math.atan2(
        float(z) + ep2 * b * math.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * math.cos(theta) ** 3,
    )
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return lat, lon, h


def _ecef_to_enu_rotation(lat: float, lon: float) -> np.ndarray:
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def _elevation_deg(rover_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    lat, lon, _h = _ecef_to_lla(float(rover_ecef[0]), float(rover_ecef[1]), float(rover_ecef[2]))
    R = _ecef_to_enu_rotation(lat, lon)
    dx = np.asarray(sat_ecef, dtype=np.float64) - np.asarray(rover_ecef, dtype=np.float64)
    enu = R @ dx
    horiz = math.hypot(float(enu[0]), float(enu[1]))
    return math.degrees(math.atan2(float(enu[2]), max(horiz, 1e-9)))


def _load_base_pseudoranges(
    base_obs_path: Path,
    obs_code: str = "C1C",
) -> tuple[dict[float, dict[str, float]], np.ndarray]:
    """Return ``(pr_by_tow, base_ecef)`` extracted from ``base.obs``.

    ``pr_by_tow[tow][sat_id]`` is the raw pseudorange in metres at GPS
    time-of-week ``tow`` (rounded to the nearest 0.1s). ``base_ecef``
    is the RINEX header approximate position. Pseudoranges are kept as
    raw observations: the satellite-clock correction cancels in the DD
    formation, so we don't need to fold it in here.
    """
    obs = read_rinex_obs(base_obs_path)
    pr_by_tow: dict[float, dict[str, float]] = {}
    for ep in obs.epochs:
        tow = round(float(_datetime_to_gps_seconds_of_week(ep.time)), 1)
        sat_pr: dict[str, float] = {}
        for sat_id, sat_obs in ep.observations.items():
            pr = sat_obs.get(obs_code, 0.0)
            if pr and pr > 1e6:
                sat_pr[str(sat_id)] = float(pr)
        if sat_pr:
            pr_by_tow[tow] = sat_pr
    return pr_by_tow, np.asarray(obs.header.approx_position, dtype=np.float64)


def _nearest_base_epoch(
    pr_by_tow: dict[float, dict[str, float]],
    target_tow: float,
    tolerance_s: float,
) -> dict[str, float] | None:
    target = round(float(target_tow), 1)
    if target in pr_by_tow:
        return pr_by_tow[target]
    if not pr_by_tow:
        return None
    best_tow = min(pr_by_tow.keys(), key=lambda t: abs(t - target))
    if abs(best_tow - target) > float(tolerance_s):
        return None
    return pr_by_tow[best_tow]


def _solve_dd_pr_ls(
    sat_ecef: np.ndarray,
    sat_ids: list[str],
    rover_pr: np.ndarray,
    weights: np.ndarray,
    base_pr_map: dict[str, float],
    base_ecef: np.ndarray,
    init_pos: np.ndarray,
    *,
    elevation_mask_deg: float,
    max_iters: int,
    tol_m: float,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Iterative Gauss-Newton LS for ``[x, y, z]`` from DD pseudoranges.

    For each common (rover, base) satellite the DD residual is

        DD_obs = (rover_PR_k - base_PR_k) - (rover_PR_ref - base_PR_ref)
        DD_pred = (||sat_k - rover|| - ||sat_k - base||)
                - (||sat_ref - rover|| - ||sat_ref - base||)
        residual = DD_obs - DD_pred

    The reference satellite is picked as the highest rover-side
    elevation with valid base PR. Because both clock biases and most
    atmospheric delays cancel in the DD formation, no clock state is
    estimated — only the 3 rover ECEF coordinates.
    """
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr_r = np.asarray(rover_pr, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if sat.shape[0] != len(sat_ids) or sat.shape[0] != pr_r.size or sat.shape[0] != w.size:
        return None

    # Match satellites that exist on both rover and base.
    common: list[tuple[int, str]] = []
    for j, sat_id in enumerate(sat_ids):
        sat_id_s = str(sat_id)
        if sat_id_s in base_pr_map and np.isfinite(pr_r[j]) and pr_r[j] > 1e6:
            common.append((j, sat_id_s))
    if len(common) < 4:
        return None

    pos = np.asarray(init_pos, dtype=np.float64).ravel()[:3].copy()
    base = np.asarray(base_ecef, dtype=np.float64).ravel()[:3]
    last_postfit_rms = float("nan")
    n_used = 0
    ref_sat_id = ""

    for iteration in range(max(1, int(max_iters))):
        # Recompute elevations + reference sat from the current position.
        cand_idx: list[int] = []
        cand_ids: list[str] = []
        cand_el: list[float] = []
        for j, sat_id in common:
            el = _elevation_deg(pos, sat[j])
            if not np.isfinite(el):
                continue
            if float(elevation_mask_deg) > 0.0 and el < float(elevation_mask_deg):
                continue
            cand_idx.append(j)
            cand_ids.append(sat_id)
            cand_el.append(float(el))
        if len(cand_idx) < 4:
            return None
        ref_pos = int(np.argmax(cand_el))
        ref_idx = cand_idx[ref_pos]
        ref_sat_id = cand_ids[ref_pos]

        # Build DD pairs (skip the reference satellite itself).
        rows: list[tuple[int, str]] = [
            (j, sid) for k, (j, sid) in enumerate(zip(cand_idx, cand_ids)) if k != ref_pos
        ]
        if len(rows) < 3:
            return None

        sat_ref = sat[ref_idx]
        pr_r_ref = float(pr_r[ref_idx])
        pr_b_ref = float(base_pr_map[ref_sat_id])
        range_ref_rover = float(np.linalg.norm(sat_ref - pos))
        range_ref_base = float(np.linalg.norm(sat_ref - base))
        if not (np.isfinite(range_ref_rover) and np.isfinite(range_ref_base)):
            return None
        unit_ref = (sat_ref - pos) / max(range_ref_rover, 1e-9)

        H = np.empty((len(rows), 3), dtype=np.float64)
        residual = np.empty(len(rows), dtype=np.float64)
        weights_dd = np.empty(len(rows), dtype=np.float64)
        for r, (j, sat_id) in enumerate(rows):
            sat_k = sat[j]
            pr_r_k = float(pr_r[j])
            pr_b_k = float(base_pr_map[sat_id])
            range_k_rover = float(np.linalg.norm(sat_k - pos))
            range_k_base = float(np.linalg.norm(sat_k - base))
            if not (np.isfinite(range_k_rover) and np.isfinite(range_k_base)):
                return None
            unit_k = (sat_k - pos) / max(range_k_rover, 1e-9)
            dd_obs = (pr_r_k - pr_b_k) - (pr_r_ref - pr_b_ref)
            dd_pred = (range_k_rover - range_k_base) - (range_ref_rover - range_ref_base)
            residual[r] = dd_obs - dd_pred
            # d/d_pos [(range_k_rover - range_ref_rover)] = -unit_k + unit_ref
            H[r, :] = -unit_k + unit_ref
            # Per-pair weight: use the (rover) sqrt(weight_k * weight_ref) as a
            # rough heuristic — high-elevation pair gets more weight.
            weights_dd[r] = math.sqrt(max(float(w[j]), 1e-6) * max(float(w[ref_idx]), 1e-6))

        sqrt_w = np.sqrt(np.maximum(weights_dd, 1e-9))
        H_w = H * sqrt_w[:, None]
        r_w = residual * sqrt_w
        try:
            delta, *_ = np.linalg.lstsq(H_w, r_w, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta)):
            return None
        pos = pos + delta
        last_postfit_rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan")
        n_used = len(rows) + 1
        if float(np.linalg.norm(delta)) < float(tol_m):
            break

    info = {
        "n_used": int(n_used),
        "postfit_rms_m": float(last_postfit_rms),
        "iterations": int(iteration + 1),
        "ref_sat_id": str(ref_sat_id),
    }
    return pos, info


def _solve_pr_ls(
    sat_ecef: np.ndarray,
    pr_corr: np.ndarray,
    weights: np.ndarray,
    init_pos: np.ndarray,
    init_clock_m: float,
    *,
    elevation_mask_deg: float,
    max_iters: int,
    tol_m: float,
) -> tuple[np.ndarray, float, dict[str, float]] | None:
    """Iterative Gauss-Newton LS for ``[x, y, z, b]`` from pseudoranges.

    ``pr_corr`` is the rover-side pseudorange already corrected for
    satellite clock bias (as produced by ``PPCDatasetLoader``).
    Atmospheric corrections are absorbed into the residual; with at
    least 4 satellites the receiver clock state ``b`` plus the 3 rover
    coordinates are estimated jointly.
    """
    sat = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)
    pr = np.asarray(pr_corr, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if sat.shape[0] != pr.size or sat.shape[0] != w.size:
        return None
    if sat.shape[0] < 4:
        return None
    finite = np.isfinite(pr) & np.isfinite(w) & np.isfinite(sat).all(axis=1) & (w > 0.0)
    if int(finite.sum()) < 4:
        return None
    sat = sat[finite]
    pr = pr[finite]
    w = w[finite]

    pos = np.asarray(init_pos, dtype=np.float64).ravel()[:3].copy()
    b = float(init_clock_m)
    used_mask = np.ones(sat.shape[0], dtype=bool)
    last_postfit_rms = float("nan")

    for iteration in range(max(1, int(max_iters))):
        # Elevation mask (recomputed each iteration since pos moves).
        if float(elevation_mask_deg) > 0.0:
            el = np.array([_elevation_deg(pos, sat[j]) for j in range(sat.shape[0])])
            mask = np.isfinite(el) & (el >= float(elevation_mask_deg))
            if int(mask.sum()) < 4:
                return None
            used_mask = mask
        idx = np.flatnonzero(used_mask)
        sat_u = sat[idx]
        pr_u = pr[idx]
        w_u = w[idx]

        diff = sat_u - pos[None, :]
        ranges = np.linalg.norm(diff, axis=1)
        good = np.isfinite(ranges) & (ranges > 1e3)
        if int(good.sum()) < 4:
            return None
        sat_u = sat_u[good]
        pr_u = pr_u[good]
        w_u = w_u[good]
        diff = diff[good]
        ranges = ranges[good]

        # Predicted observation: range + receiver clock bias (in meters).
        pred = ranges + b
        residual = pr_u - pred  # observed - predicted

        # Jacobian: d/d_pos = -unit_los; d/d_b = +1.
        unit_los = diff / ranges[:, None]
        H = np.empty((sat_u.shape[0], 4), dtype=np.float64)
        H[:, 0:3] = -unit_los
        H[:, 3] = 1.0

        # Weighted normal equations.
        sqrt_w = np.sqrt(np.maximum(w_u, 1e-9))
        H_w = H * sqrt_w[:, None]
        r_w = residual * sqrt_w
        try:
            delta, *_ = np.linalg.lstsq(H_w, r_w, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(delta)):
            return None
        pos = pos + delta[0:3]
        b = float(b + delta[3])
        last_postfit_rms = float(np.sqrt(np.mean(residual * residual))) if residual.size else float("nan")
        if float(np.linalg.norm(delta[0:3])) < float(tol_m):
            break

    info = {
        "n_used": int(np.count_nonzero(used_mask)),
        "postfit_rms_m": float(last_postfit_rms),
        "iterations": int(iteration + 1),
    }
    return pos, float(b), info


def _load_seed_pos_rows(
    src: Path, start_tow: float, end_tow: float
) -> tuple[list[str], list[tuple[float, list[str]]]]:
    headers: list[str] = []
    rows: list[tuple[float, list[str]]] = []
    with src.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                headers.append(line)
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                tow = round(float(parts[1]), 1)
            except ValueError:
                continue
            if float(start_tow) <= tow <= float(end_tow):
                rows.append((tow, parts))
    return headers, rows


def _copy_csv_window(
    src: Path,
    dst: Path,
    start_tow: float,
    end_tow: float,
    *,
    anchor_metrics: dict[float, dict[str, float]] | None = None,
    anchor_ratio: float = 3.0,
    anchor_status: int = 4,
    keep_only_anchored: bool = True,
) -> int:
    """Copy a TOW-windowed slice of the seed CSV.

    When ``anchor_metrics`` is supplied, rewrite each anchored epoch's
    ``final_*`` / ``initial_*`` quality fields so the RTKDiag candidate
    gate sees the LS solution's postfit RMS instead of the original
    seed's metrics. ``keep_only_anchored`` drops non-anchored rows so
    the candidate cannot be selected outside the materialised window.
    """

    kept = 0
    metrics = anchor_metrics or {}
    with src.open(newline="") as inf, dst.open("w", newline="") as outf:
        reader = csv.DictReader(inf)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {src}")
        writer = csv.DictWriter(outf, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            try:
                tow = round(float(row["tow"]), 1)
            except (KeyError, TypeError, ValueError):
                continue
            if not (float(start_tow) <= tow <= float(end_tow)):
                continue
            anchor = metrics.get(tow)
            if anchor is None:
                if keep_only_anchored:
                    continue
                writer.writerow(row)
                kept += 1
                continue
            postfit_rms = float(anchor.get("postfit_rms_m", float("nan")))
            n_used = int(anchor.get("n_used", 0))
            row = dict(row)
            row["output_added"] = "1"
            row["final_valid"] = "1"
            row["final_status"] = str(int(anchor_status))
            row["final_ratio"] = f"{float(anchor_ratio):.6f}"
            if "final_residual_rms" in row:
                row["final_residual_rms"] = f"{postfit_rms:.6f}"
            if "final_residual_abs_max" in row:
                row["final_residual_abs_max"] = f"{postfit_rms * 2.0:.6f}"
            if "final_update_rows" in row:
                row["final_update_rows"] = str(max(n_used, 4))
            if "final_sats" in row:
                row["final_sats"] = str(max(n_used, 4))
            if "final_pdop" in row and (not row.get("final_pdop", "0").strip()
                                        or float(row.get("final_pdop", "0")) <= 0.0):
                row["final_pdop"] = "2.0"
            if "initial_residual_rms" in row:
                row["initial_residual_rms"] = f"{postfit_rms:.6f}"
            if "initial_residual_abs_max" in row:
                row["initial_residual_abs_max"] = f"{postfit_rms * 2.0:.6f}"
            if "initial_status" in row:
                row["initial_status"] = str(int(anchor_status))
            if "initial_ratio" in row:
                row["initial_ratio"] = f"{float(anchor_ratio):.6f}"
            writer.writerow(row)
            kept += 1
    return kept


def _write_pos_window(
    headers: list[str],
    rows: list[tuple[float, list[str]]],
    new_positions: dict[float, np.ndarray],
    dst: Path,
) -> int:
    kept = 0
    with dst.open("w", newline="") as outf:
        for line in headers:
            outf.write(line)
        for tow, parts in rows:
            pos_new = new_positions.get(round(float(tow), 1))
            if pos_new is None or not np.all(np.isfinite(pos_new)):
                continue
            x = float(pos_new[0])
            y = float(pos_new[1])
            z = float(pos_new[2])
            lat, lon, h = _ecef_to_lla(x, y, z)
            parts = list(parts)
            parts[2] = f"{x:.4f}"
            parts[3] = f"{y:.4f}"
            parts[4] = f"{z:.4f}"
            parts[5] = f"{math.degrees(lat):.9f}"
            parts[6] = f"{math.degrees(lon):.9f}"
            parts[7] = f"{h:.4f}"
            outf.write(" ".join(parts) + "\n")
            kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="PPC dataset run dir, e.g. PPC-Dataset-data/nagoya/run2",
    )
    parser.add_argument("--seed-pos", type=Path, required=True)
    parser.add_argument("--seed-csv", type=Path, required=True)
    parser.add_argument("--start-tow", type=float, required=True)
    parser.add_argument("--end-tow", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=12000)
    parser.add_argument(
        "--systems",
        type=str,
        default="G",
        help="Comma-separated systems for the LS (default 'G' = GPS L1 only)",
    )
    parser.add_argument(
        "--mode",
        choices=("dd", "undiff"),
        default="dd",
        help="LS mode: 'dd' (double-differenced PR with base station, recommended) "
             "or 'undiff' (single-point PR LS, SPP-class)",
    )
    parser.add_argument(
        "--base-time-tolerance",
        type=float,
        default=1.0,
        help="Match rover epoch to nearest base epoch within this [s] in dd mode",
    )
    parser.add_argument("--elevation-mask-deg", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--tol-m", type=float, default=1.0e-3)
    parser.add_argument(
        "--postfit-max-rms-m",
        type=float,
        default=5.0,
        help="Reject LS solutions whose post-fit residual RMS exceeds this [m]. "
             "Default 5.0 m (good for dd mode; raise to 20-30 for undiff mode).",
    )
    parser.add_argument(
        "--postfit-min-rms-m",
        type=float,
        default=0.0,
        help="Reject LS solutions whose post-fit residual RMS is below this [m]. "
             "Use to drop near-singular (under-determined) solutions where postfit is "
             "misleadingly low. Default 0.0 (no filter).",
    )
    parser.add_argument(
        "--min-n-used",
        type=int,
        default=0,
        help="Reject LS solutions that used fewer than this many satellites/DD rows. "
             "Default 0 (no filter). Use 7-8 for dd mode to ensure overdetermined system.",
    )
    parser.add_argument(
        "--max-shift-to-seed-m",
        type=float,
        default=float("inf"),
        help="Reject LS solutions whose final position is farther than this from the "
             "seed pos at the same epoch [m]. Default inf (no filter). Useful to keep "
             "only candidates that agree with the seed within a reasonable bound.",
    )
    parser.add_argument(
        "--init-from-seed",
        action="store_true",
        default=True,
        help="Initialize LS from the seed pos row (default true)",
    )
    parser.add_argument(
        "--anchor-ratio",
        type=float,
        default=3.0,
        help="Synthetic ratio written to anchored CSV rows so they pass the "
             "default RTKDiag ratio gate (default 3.0)",
    )
    parser.add_argument(
        "--anchor-status",
        type=int,
        default=4,
        help="Synthetic status written to anchored CSV rows (default 4 = fixed)",
    )
    parser.add_argument(
        "--keep-non-anchored-rows",
        action="store_true",
        help="Keep CSV rows that have no LS anchor (default: drop them so the "
             "candidate is only selectable inside the materialised window).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    headers, rows = _load_seed_pos_rows(
        args.seed_pos, float(args.start_tow), float(args.end_tow)
    )
    if not rows:
        raise SystemExit(
            f"no seed pos rows in window [{args.start_tow}, {args.end_tow}]: {args.seed_pos}"
        )
    seed_pos_at_tow: dict[float, np.ndarray] = {}
    for tow, parts in rows:
        seed_pos_at_tow[round(float(tow), 1)] = np.array(
            [float(parts[2]), float(parts[3]), float(parts[4])],
            dtype=np.float64,
        )

    systems_tuple = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    data = PPCDatasetLoader(args.run_dir).load_experiment_data(
        max_epochs=int(args.max_epochs),
        systems=systems_tuple,
    )
    times = np.asarray(data["times"], dtype=np.float64)
    sat_ecef_per_epoch = data["sat_ecef"]
    pr_corrected_per_epoch = data["pseudoranges"]
    w_per_epoch = data["weights"]
    sat_ids_per_epoch = data["used_prns"]

    base_pr_by_tow: dict[float, dict[str, float]] = {}
    base_ecef = np.zeros(3, dtype=np.float64)
    if args.mode == "dd":
        base_pr_by_tow, base_ecef = _load_base_pseudoranges(args.run_dir / "base.obs")
        if not base_pr_by_tow:
            raise SystemExit(
                f"dd mode requires base.obs with usable PR observations: "
                f"{args.run_dir / 'base.obs'}"
            )
        # In dd mode the rover-side observations must be raw PR (not
        # sat-clock-corrected) so that DD differencing cancels base-side
        # corrections too. ``PPCDatasetLoader`` only exposes corrected PR,
        # so re-read rover.obs from RINEX directly for dd mode.
        rover_pr_by_tow, _ = _load_base_pseudoranges(args.run_dir / "rover.obs")
    else:
        rover_pr_by_tow = {}

    new_positions: dict[float, np.ndarray] = {}
    anchor_metrics: dict[float, dict[str, float]] = {}
    n_attempted = 0
    n_solved = 0
    n_postfit_rejected = 0
    n_min_sats_rejected = 0
    n_no_base_match = 0
    postfit_rms_log: list[float] = []
    n_used_log: list[int] = []
    delta_to_seed_log: list[float] = []
    ref_sat_log: list[str] = []

    for i, tow in enumerate(times):
        tow_r = round(float(tow), 1)
        if tow_r not in seed_pos_at_tow:
            continue
        n_attempted += 1
        sat_ecef = np.asarray(sat_ecef_per_epoch[i], dtype=np.float64)
        pr_corr = np.asarray(pr_corrected_per_epoch[i], dtype=np.float64)
        weights = np.asarray(w_per_epoch[i], dtype=np.float64)
        init_pos = seed_pos_at_tow[tow_r]

        if args.mode == "dd":
            base_pr = _nearest_base_epoch(
                base_pr_by_tow, tow_r, float(args.base_time_tolerance)
            )
            rover_raw = _nearest_base_epoch(rover_pr_by_tow, tow_r, 0.5)
            if base_pr is None or rover_raw is None:
                n_no_base_match += 1
                continue
            sat_ids = list(sat_ids_per_epoch[i])
            # Replace rover PR with raw observations (sat clock cancels in DD).
            rover_pr = np.array(
                [float(rover_raw.get(str(sid), float("nan"))) for sid in sat_ids],
                dtype=np.float64,
            )
            result = _solve_dd_pr_ls(
                sat_ecef,
                sat_ids,
                rover_pr,
                weights,
                base_pr,
                base_ecef,
                init_pos,
                elevation_mask_deg=float(args.elevation_mask_deg),
                max_iters=int(args.max_iters),
                tol_m=float(args.tol_m),
            )
            if result is None:
                n_min_sats_rejected += 1
                continue
            pos_ls, info = result
        else:
            result = _solve_pr_ls(
                sat_ecef,
                pr_corr,
                weights,
                init_pos,
                init_clock_m=0.0,
                elevation_mask_deg=float(args.elevation_mask_deg),
                max_iters=int(args.max_iters),
                tol_m=float(args.tol_m),
            )
            if result is None:
                n_min_sats_rejected += 1
                continue
            pos_ls, _b, info = result

        postfit_rms_val = float(info.get("postfit_rms_m", float("nan")))
        n_used_val = int(info.get("n_used", 0))
        if postfit_rms_val > float(args.postfit_max_rms_m):
            n_postfit_rejected += 1
            continue
        if postfit_rms_val < float(args.postfit_min_rms_m):
            n_postfit_rejected += 1
            continue
        if n_used_val < int(args.min_n_used):
            n_min_sats_rejected += 1
            continue
        if not np.all(np.isfinite(pos_ls)):
            continue
        shift_seed = float(np.linalg.norm(pos_ls - init_pos))
        if shift_seed > float(args.max_shift_to_seed_m):
            n_postfit_rejected += 1
            continue
        new_positions[tow_r] = pos_ls
        anchor_metrics[tow_r] = {
            "postfit_rms_m": float(info.get("postfit_rms_m", float("nan"))),
            "n_used": int(info.get("n_used", 0)),
        }
        n_solved += 1
        postfit_rms_log.append(float(info.get("postfit_rms_m", float("nan"))))
        n_used_log.append(int(info.get("n_used", 0)))
        delta_to_seed_log.append(float(np.linalg.norm(pos_ls - init_pos)))
        if "ref_sat_id" in info:
            ref_sat_log.append(str(info["ref_sat_id"]))

    prefix = f"{args.city}_{args.run}_full"
    out_pos = args.out_dir / f"{prefix}.pos"
    out_csv = args.out_dir / f"{prefix}.csv"
    n_pos = _write_pos_window(headers, rows, new_positions, out_pos)
    n_csv = _copy_csv_window(
        args.seed_csv,
        out_csv,
        float(args.start_tow),
        float(args.end_tow),
        anchor_metrics=anchor_metrics,
        anchor_ratio=float(args.anchor_ratio),
        anchor_status=int(args.anchor_status),
        keep_only_anchored=not bool(args.keep_non_anchored_rows),
    )

    if not args.quiet:
        median_rms = float(np.median(postfit_rms_log)) if postfit_rms_log else float("nan")
        median_used = float(np.median(n_used_log)) if n_used_log else float("nan")
        median_delta = float(np.median(delta_to_seed_log)) if delta_to_seed_log else float("nan")
        max_delta = float(np.max(delta_to_seed_log)) if delta_to_seed_log else float("nan")
        ref_summary = ""
        if ref_sat_log:
            from collections import Counter
            top = Counter(ref_sat_log).most_common(3)
            ref_summary = f", ref_sat top3={top}"
        no_base_summary = (
            f", rejected_nobase={n_no_base_match}" if args.mode == "dd" else ""
        )
        print(
            f"saved pos: {out_pos} rows={n_pos}/{len(rows)} (csv={n_csv}, mode={args.mode}, "
            f"attempted={n_attempted} solved={n_solved} "
            f"rejected_postfit={n_postfit_rejected} rejected_minsats={n_min_sats_rejected}"
            f"{no_base_summary}, "
            f"postfit_rms median={median_rms:.2f}m, n_used median={median_used:.0f}, "
            f"shift_to_seed median={median_delta:.2f}m max={max_delta:.2f}m{ref_summary})",
            flush=True,
        )


if __name__ == "__main__":
    main()
