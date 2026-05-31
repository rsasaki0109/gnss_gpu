#!/usr/bin/env python3
"""Materialize a TOW-window candidate via TDCP-anchor reset.

Use case: a segment where every gated RTK candidate has the right *shape*
but a constant ECEF bias (the relative-bias oracle pattern). We rebuild
the segment trajectory by integrating GPS L1 TDCP velocity from anchor
positions just outside the segment, then blend forward / backward
integrations to spread anchor uncertainty across the window.

Pipeline:
1. Load a seed candidate's ``{city}_{run}_full.pos`` (RTK output to keep
   diagnostics coherent) and ``.csv`` (gate fields).
2. Compute the TDCP velocity series for the whole run using the same
   helper the height-prior script uses.
3. Pick anchor positions from the hybrid pos file:
   ``anchor_left`` = median of Status=4 hybrid in ``[start_tow -
   margin*dt_est, start_tow)``;
   ``anchor_right`` = median of Status=4 hybrid in ``(end_tow, end_tow +
   margin*dt_est]``.  Falls back to seed pos boundary if Status=4 is
   missing.
4. Forward-integrate from ``anchor_left``; backward-integrate from
   ``anchor_right``.  Linear blend ``alpha`` from 0 (start) to 1 (end).
5. Rewrite seed pos rows in the segment with the blended ECEF X/Y/Z
   (LLH recomputed); window-slice the csv as-is.

The candidate must be wired via ``--rtkdiag-candidate-pos-dirs`` and a
matching label, e.g. ``xd_anchor_reset_n2_6637_6660`` in the
``rtkdiag_candidate_labels`` list of the consuming run.
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
from gnss_gpu.tdcp_velocity import estimate_velocity_from_tdcp_with_metrics  # noqa: E402
from exp_ppc_ctrbpf_fgo import _build_tdcp_measurements, _load_hybrid_pos_file  # noqa: E402


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


def _parse_header_approx_position(obs_path: Path) -> np.ndarray:
    with obs_path.open(errors="replace") as fh:
        for line in fh:
            if "APPROX POSITION XYZ" in line:
                return np.array([float(line[0:14]), float(line[14:28]), float(line[28:42])])
            if "END OF HEADER" in line:
                break
    raise ValueError(f"APPROX POSITION XYZ not found: {obs_path}")


def _compute_tdcp_velocity_series(
    run_dir: Path,
    max_epochs: int,
    postfit_max_m: float,
    velocity_sign: float,
) -> tuple[dict[float, np.ndarray], dict[float, float]]:
    """Return per-epoch (TOW -> velocity_ECEF, TOW -> dt_to_prev) maps."""
    header_pos = _parse_header_approx_position(run_dir / "rover.obs")
    data = PPCDatasetLoader(run_dir).load_experiment_data(
        max_epochs=max_epochs,
        include_sat_velocity=True,
        systems=("G",),
    )
    times = np.asarray(data["times"], dtype=np.float64)
    vel_at: dict[float, np.ndarray] = {}
    dt_at: dict[float, float] = {}
    pos = header_pos.astype(np.float64).copy()
    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        prev_meas = _build_tdcp_measurements(
            np.asarray(data["sat_ecef"][i], dtype=np.float64),
            np.asarray(data["system_ids"][i], dtype=np.int32),
            list(data["used_prns"][i]),
            np.asarray(data["carrier_phase"][i], dtype=np.float64),
            np.asarray(data["sat_velocity"][i], dtype=np.float64),
            np.asarray(data["clock_drift"][i], dtype=np.float64),
            pos,
        )
        cur_meas = _build_tdcp_measurements(
            np.asarray(data["sat_ecef"][i + 1], dtype=np.float64),
            np.asarray(data["system_ids"][i + 1], dtype=np.int32),
            list(data["used_prns"][i + 1]),
            np.asarray(data["carrier_phase"][i + 1], dtype=np.float64),
            np.asarray(data["sat_velocity"][i + 1], dtype=np.float64),
            np.asarray(data["clock_drift"][i + 1], dtype=np.float64),
            pos,
        )
        velocity, _rms = estimate_velocity_from_tdcp_with_metrics(
            pos,
            prev_meas,
            cur_meas,
            dt,
            min_sats=4,
            max_postfit_rms_m=float(postfit_max_m),
            elevation_weight=True,
        )
        if velocity is not None:
            velocity = np.asarray(velocity, dtype=np.float64) * float(velocity_sign)
            pos = pos + np.asarray(velocity, dtype=np.float64) * dt
            tow = round(float(times[i + 1]), 1)
            vel_at[tow] = np.asarray(velocity, dtype=np.float64)
            dt_at[tow] = float(dt)
    return vel_at, dt_at


def _load_seed_pos_rows(
    src: Path, start_tow: float, end_tow: float
) -> tuple[list[str], list[tuple[float, list[str]]]]:
    """Return (header_lines, rows) for the seed pos file, restricted to the window.

    ``rows`` is a list of ``(tow, parts)`` where ``parts`` is the
    whitespace-split row preserved verbatim aside from later position
    rewrites.
    """
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


def _seed_position_for_tow(
    rows: list[tuple[float, list[str]]], target_tow: float
) -> np.ndarray | None:
    for tow, parts in rows:
        if abs(tow - float(target_tow)) <= 0.05:
            return np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
    return None


def _hybrid_status4_anchor(
    hybrid_pos: dict[float, np.ndarray],
    hybrid_status: dict[float, int],
    *,
    side_tow: float,
    direction: int,
    margin_epochs: int,
    epoch_dt_s: float,
    seed_fallback: np.ndarray | None,
    seed_fallback_tow: float | None = None,
) -> tuple[np.ndarray, float, str]:
    """Pick an anchor as the *closest* Status=4 hybrid epoch on the requested
    side of the segment. ``direction=-1`` searches before ``side_tow`` (the
    closest earlier Status=4 epoch is preferred); ``direction=+1`` searches
    after.

    Returns ``(position, tow, source_label)``. The TOW is the actual epoch
    of the chosen anchor, so the caller knows where the integration chain
    starts.
    """
    span_s = float(margin_epochs) * float(epoch_dt_s)
    if direction < 0:
        lo = float(side_tow) - span_s
        hi = float(side_tow)
        in_range = lambda t: (t >= lo) and (t < hi)  # noqa: E731
        sort_key = lambda t: -float(t)  # noqa: E731 (closest from below = largest tow)
    else:
        lo = float(side_tow)
        hi = float(side_tow) + span_s
        in_range = lambda t: (t > lo) and (t <= hi)  # noqa: E731
        sort_key = lambda t: float(t)  # noqa: E731 (closest from above = smallest tow)
    candidates: list[float] = []
    for tow, _pos in hybrid_pos.items():
        st = hybrid_status.get(tow)
        if st is None or int(st) != 4:
            continue
        if in_range(float(tow)):
            candidates.append(float(tow))
    if candidates:
        candidates.sort(key=sort_key)
        chosen_tow = float(candidates[0])
        return (
            np.asarray(hybrid_pos[round(chosen_tow, 1)], dtype=np.float64),
            chosen_tow,
            f"hybrid_status4_at_{chosen_tow:.1f}",
        )
    if seed_fallback is not None:
        fallback_tow = (
            float(seed_fallback_tow)
            if seed_fallback_tow is not None
            else float(side_tow)
        )
        return (
            np.asarray(seed_fallback, dtype=np.float64),
            fallback_tow,
            "seed_fallback",
        )
    raise ValueError(
        f"no Status=4 hybrid in [{lo:.1f}, {hi:.1f}] and no seed fallback"
    )


def _integrate_chain(
    anchor: np.ndarray,
    anchor_tow: float,
    rows_tows: list[float],
    vel_at: dict[float, np.ndarray],
    dt_at: dict[float, float],
    *,
    backward: bool = False,
    default_dt_s: float = 0.2,
) -> list[np.ndarray | None]:
    """Integrate TDCP velocity from ``anchor_tow`` (where position is ``anchor``)
    through every epoch on the TDCP grid until past the segment, then return
    the positions at ``rows_tows``.

    ``backward=False`` walks forward in time: requires ``anchor_tow <
    min(rows_tows)``.  ``backward=True`` walks backward in time: requires
    ``anchor_tow > max(rows_tows)``.

    Missing TDCP velocity for an intermediate epoch is treated as zero
    (the chain holds position) which biases long sparse segments; rely on
    the forward/backward blend to spread the bias.
    """
    rows_tows_rounded = [round(float(t), 1) for t in rows_tows]
    rows_tows_set = set(rows_tows_rounded)
    out_by_tow: dict[float, np.ndarray] = {}
    cur = np.asarray(anchor, dtype=np.float64).copy()

    # Build the path: union of TDCP grid epochs and the segment row epochs
    # (so missing-velocity rows still get an output by holding ``cur``).
    grid_tows = set(round(float(t), 1) for t in vel_at.keys())
    grid_tows.update(rows_tows_set)
    if backward:
        rows_min = float(min(rows_tows))
        path = sorted(
            (t for t in grid_tows if rows_min - 0.05 <= t < float(anchor_tow)),
            reverse=True,
        )
    else:
        rows_max = float(max(rows_tows))
        path = sorted(t for t in grid_tows if float(anchor_tow) < t <= rows_max + 0.05)

    for tow in path:
        v = vel_at.get(round(float(tow), 1))
        dt = dt_at.get(round(float(tow), 1), float(default_dt_s))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = float(default_dt_s)
        if v is not None:
            step = np.asarray(v, dtype=np.float64) * float(dt)
            if backward:
                # ``v`` is forward velocity (prev -> tow). Stepping
                # backward in time means moving from tow back to prev, i.e.
                # subtracting v*dt from the position at tow.
                cur = cur - step
            else:
                cur = cur + step
        # ``v is None`` means TDCP was rejected for this epoch — hold
        # ``cur`` and continue (a single missing velocity costs ~v*dt of
        # bias, partially absorbed by the forward/backward blend).
        if round(float(tow), 1) in rows_tows_set:
            out_by_tow[round(float(tow), 1)] = cur.copy()

    return [out_by_tow.get(round(float(t), 1)) for t in rows_tows]


def _blend_chains(
    fwd: list[np.ndarray | None],
    bwd: list[np.ndarray | None],
) -> list[np.ndarray | None]:
    n = len(fwd)
    if n == 0:
        return []
    if n == 1:
        a = fwd[0]
        b = bwd[0]
        if a is None and b is None:
            return [None]
        if a is None:
            return [b]
        if b is None:
            return [a]
        return [0.5 * (a + b)]
    out: list[np.ndarray | None] = []
    for i in range(n):
        alpha = float(i) / float(n - 1)
        a = fwd[i]
        b = bwd[i]
        if a is None and b is None:
            out.append(None)
        elif a is None:
            out.append(b)
        elif b is None:
            out.append(a)
        else:
            out.append((1.0 - alpha) * a + alpha * b)
    return out


def _write_pos_window(
    headers: list[str],
    rows: list[tuple[float, list[str]]],
    blended: list[np.ndarray | None],
    dst: Path,
) -> int:
    kept = 0
    with dst.open("w", newline="") as outf:
        for line in headers:
            outf.write(line)
        for (tow, parts), pos_new in zip(rows, blended, strict=True):
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


def _copy_csv_window(src: Path, dst: Path, start_tow: float, end_tow: float) -> int:
    kept = 0
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
            if float(start_tow) <= tow <= float(end_tow):
                writer.writerow(row)
                kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="PPC dataset run dir, e.g. PPC-Dataset-data/nagoya/run2")
    parser.add_argument("--seed-pos", type=Path, required=True,
                        help="Seed candidate's pos file ({city}_{run}_full.pos)")
    parser.add_argument("--seed-csv", type=Path, required=True,
                        help="Seed candidate's csv file ({city}_{run}_full.csv)")
    parser.add_argument("--hybrid-pos", type=Path, required=True,
                        help="Hybrid pos file ({city}_{run}_full.pos)")
    parser.add_argument("--start-tow", type=float, required=True)
    parser.add_argument("--end-tow", type=float, required=True)
    parser.add_argument("--anchor-margin-epochs", type=int, default=20,
                        help="Search this many epochs on each side for Status=4 hybrid anchors (default 20)")
    parser.add_argument("--epoch-dt-s", type=float, default=0.2,
                        help="Nominal epoch spacing for anchor margin (default 0.2 s = 5 Hz)")
    parser.add_argument("--max-epochs", type=int, default=12000)
    parser.add_argument("--tdcp-postfit-max-m", type=float, default=2.0)
    parser.add_argument("--tdcp-velocity-sign", type=float, default=-1.0,
                        help="Sign applied to TDCP velocity from current API (default -1.0, matching the historical script)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Phase-pool dir name, e.g. results/libgnss_diag_phase10/anchor_reset_n2_6637_6660")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load seed window.
    headers, rows = _load_seed_pos_rows(
        args.seed_pos, float(args.start_tow), float(args.end_tow)
    )
    if not rows:
        raise SystemExit(
            f"no seed pos rows in window [{args.start_tow}, {args.end_tow}]: {args.seed_pos}"
        )
    rows_tows = [float(t) for t, _ in rows]
    seed_left = np.array(
        [float(rows[0][1][2]), float(rows[0][1][3]), float(rows[0][1][4])],
        dtype=np.float64,
    )
    seed_right = np.array(
        [float(rows[-1][1][2]), float(rows[-1][1][3]), float(rows[-1][1][4])],
        dtype=np.float64,
    )

    # 2. TDCP velocity series.
    vel_at, dt_at = _compute_tdcp_velocity_series(
        args.run_dir,
        int(args.max_epochs),
        float(args.tdcp_postfit_max_m),
        float(args.tdcp_velocity_sign),
    )

    # 3. Anchors from hybrid Status=4 (closest epoch on each side).
    hybrid_pos, hybrid_status = _load_hybrid_pos_file(args.hybrid_pos)
    anchor_left, anchor_left_tow, anchor_left_src = _hybrid_status4_anchor(
        hybrid_pos,
        hybrid_status,
        side_tow=rows_tows[0],
        direction=-1,
        margin_epochs=int(args.anchor_margin_epochs),
        epoch_dt_s=float(args.epoch_dt_s),
        seed_fallback=seed_left,
        seed_fallback_tow=rows_tows[0] - float(args.epoch_dt_s),
    )
    anchor_right, anchor_right_tow, anchor_right_src = _hybrid_status4_anchor(
        hybrid_pos,
        hybrid_status,
        side_tow=rows_tows[-1],
        direction=+1,
        margin_epochs=int(args.anchor_margin_epochs),
        epoch_dt_s=float(args.epoch_dt_s),
        seed_fallback=seed_right,
        seed_fallback_tow=rows_tows[-1] + float(args.epoch_dt_s),
    )

    # 4. Forward + backward integration from the actual anchor TOWs.
    fwd = _integrate_chain(
        anchor_left,
        float(anchor_left_tow),
        rows_tows,
        vel_at,
        dt_at,
        backward=False,
        default_dt_s=float(args.epoch_dt_s),
    )
    bwd = _integrate_chain(
        anchor_right,
        float(anchor_right_tow),
        rows_tows,
        vel_at,
        dt_at,
        backward=True,
        default_dt_s=float(args.epoch_dt_s),
    )
    blended = _blend_chains(fwd, bwd)

    # 5. Write outputs.
    prefix = f"{args.city}_{args.run}_full"
    out_pos = args.out_dir / f"{prefix}.pos"
    out_csv = args.out_dir / f"{prefix}.csv"
    n_pos = _write_pos_window(headers, rows, blended, out_pos)
    n_csv = _copy_csv_window(args.seed_csv, out_csv, float(args.start_tow), float(args.end_tow))

    # 6. Diagnostics.
    if not args.quiet:
        deltas_to_seed = []
        for (tow, parts), pos_new in zip(rows, blended, strict=True):
            if pos_new is None:
                continue
            seed_pos = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
            deltas_to_seed.append(float(np.linalg.norm(np.asarray(pos_new) - seed_pos)))
        median_delta = float(np.median(deltas_to_seed)) if deltas_to_seed else float("nan")
        max_delta = float(np.max(deltas_to_seed)) if deltas_to_seed else float("nan")
        n_blended = sum(1 for p in blended if p is not None)
        print(
            f"saved pos: {out_pos} rows={n_pos}/{len(rows)} (csv={n_csv}, "
            f"blended={n_blended}, "
            f"anchor_left tow={anchor_left_tow:.1f} src={anchor_left_src}, "
            f"anchor_right tow={anchor_right_tow:.1f} src={anchor_right_src}, "
            f"shift_to_seed median={median_delta:.2f}m max={max_delta:.2f}m, "
            f"tdcp_pairs={len(vel_at)})",
            flush=True,
        )


if __name__ == "__main__":
    main()
