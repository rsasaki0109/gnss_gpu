#!/usr/bin/env python3
# ruff: noqa: E402
"""Render a fix/float-status figure for a libgnss++ PPC RTK result.

Six-panel layout:

  +-----------------------+----------+--------+
  | traj: FIXED only vs   | 2D err   | 2D err |
  | truth (cm tight)      | timeline | CDF    |
  +-----------------------+          |        |
  | traj: FLOAT only vs   |          |        |
  | truth (scatter)       |          |        |
  +-----------------------+----------+--------+

The FIXED-only panel makes it obvious when integer ambiguities are solid
(green stays on the truth line at cm level). The FLOAT-only panel exposes
where the solver is "giving up" and emitting meter-level positions.
"""

from __future__ import annotations

import argparse
import csv
from math import atan2, cos, hypot, sin, sqrt
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))


def _ecef_to_lat_lon(ecef: np.ndarray) -> tuple[float, float]:
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    lon = atan2(y, x)
    p = hypot(x, y)
    e2 = 6.694379990141316e-3
    lat = atan2(z, p * (1.0 - e2))
    for _ in range(6):
        sl = sin(lat)
        n = 6_378_137.0 / sqrt(1.0 - e2 * sl * sl)
        lat = atan2(z + e2 * n * sl, p)
    return lat, lon


def _enu_rotation(lat: float, lon: float) -> np.ndarray:
    sl, cl = sin(lat), cos(lat)
    so, co = sin(lon), cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def _load_reference(path: Path) -> dict[float, np.ndarray]:
    out: dict[float, np.ndarray] = {}
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            tow = round(float(row[0]), 1)
            out[tow] = np.array(
                [float(row[5]), float(row[6]), float(row[7])], dtype=np.float64,
            )
    return out


def _load_pos(path: Path) -> list[tuple[float, np.ndarray, int, float]]:
    rows: list[tuple[float, np.ndarray, int, float]] = []
    with path.open() as fh:
        for line in fh:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            tow = round(float(parts[1]), 1)
            ecef = np.array(
                [float(parts[2]), float(parts[3]), float(parts[4])],
                dtype=np.float64,
            )
            status = int(parts[8])
            ratio = float(parts[11])
            rows.append((tow, ecef, status, ratio))
    return rows


_FIXED_COLOR = "#1a9850"
_FLOAT_COLOR = "#e65100"
_TRUTH_COLOR = "#202020"


def _plot_trajectory(
    ax: plt.Axes,
    title: str,
    truth_enu: np.ndarray,
    pts_enu: np.ndarray,
    color: str,
    label: str,
) -> None:
    ax.plot(
        truth_enu[:, 0],
        truth_enu[:, 1],
        color=_TRUTH_COLOR,
        linewidth=1.2,
        alpha=0.9,
        label="truth",
        zorder=1,
    )
    if pts_enu.size:
        ax.scatter(
            pts_enu[:, 0],
            pts_enu[:, 1],
            s=8,
            c=color,
            edgecolor="none",
            alpha=0.75,
            label=label,
            zorder=2,
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East from centroid (m)")
    ax.set_ylabel("North from centroid (m)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render libgnss++ fix-status figure for a PPC segment",
    )
    parser.add_argument("--pos", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    pos_rows = _load_pos(args.pos)
    ref = _load_reference(args.reference)

    matched: list[tuple[float, np.ndarray, np.ndarray, int, float]] = []
    for tow, ecef, status, ratio in pos_rows:
        truth = ref.get(tow)
        if truth is None:
            continue
        matched.append((tow, ecef, truth, status, ratio))
    matched.sort(key=lambda r: r[0])
    if not matched:
        raise SystemExit(f"no TOW matches between pos and reference: {args.pos}")

    tows = np.array([r[0] for r in matched])
    fused = np.array([r[1] for r in matched])
    truth = np.array([r[2] for r in matched])
    status = np.array([r[3] for r in matched])

    center = np.mean(truth, axis=0)
    lat0, lon0 = _ecef_to_lat_lon(center)
    R = _enu_rotation(lat0, lon0)

    truth_enu = np.array([R @ (t - center) for t in truth])
    fused_enu = np.array([R @ (f - center) for f in fused])
    diff_enu = np.array([R @ (f - t) for f, t in zip(fused, truth)])

    e2d = np.linalg.norm(diff_enu[:, :2], axis=1)
    e3d = np.linalg.norm(fused - truth, axis=1)
    t_rel = tows - tows[0]

    fixed_mask = status == 4
    float_mask = status == 3
    n_fixed = int(fixed_mask.sum())
    n_float = int(float_mask.sum())

    fig = plt.figure(figsize=(15.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.0, 0.9], height_ratios=[1, 1])

    ax_fix = fig.add_subplot(gs[0, 0])
    _plot_trajectory(
        ax_fix,
        f"FIXED solutions vs truth ({n_fixed} pts, cm accuracy)",
        truth_enu,
        fused_enu[fixed_mask],
        _FIXED_COLOR,
        "FIXED",
    )

    ax_flt = fig.add_subplot(gs[1, 0])
    _plot_trajectory(
        ax_flt,
        f"FLOAT solutions vs truth ({n_float} pts, meter-level scatter)",
        truth_enu,
        fused_enu[float_mask],
        _FLOAT_COLOR,
        "FLOAT",
    )

    # 2D error timeline, split by status
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.axhspan(0.0, 0.5, color="#dfffe0", alpha=0.5, label="PPC 0.5 m pass band")
    ax_err.scatter(
        t_rel[float_mask],
        e2d[float_mask],
        s=6,
        c=_FLOAT_COLOR,
        edgecolor="none",
        alpha=0.85,
        label=f"FLOAT (n={n_float})",
        zorder=2,
    )
    ax_err.scatter(
        t_rel[fixed_mask],
        e2d[fixed_mask],
        s=6,
        c=_FIXED_COLOR,
        edgecolor="none",
        alpha=0.9,
        label=f"FIXED (n={n_fixed})",
        zorder=3,
    )
    ax_err.axhline(0.5, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax_err.set_ylabel("2D error from truth (m)")
    ax_err.set_yscale("symlog", linthresh=0.05)
    ax_err.set_title("Per-epoch 2D error (symlog @ 0.05 m)")
    ax_err.grid(alpha=0.25)
    ax_err.legend(loc="upper right", framealpha=0.9)
    ax_err.set_xlabel("Time from segment start (s)")
    ax_err.set_xlim(t_rel.min(), t_rel.max())

    # CDF of 2D error per status
    ax_cdf = fig.add_subplot(gs[0, 2])
    for mask, color, label in [
        (fixed_mask, _FIXED_COLOR, f"FIXED (n={n_fixed})"),
        (float_mask, _FLOAT_COLOR, f"FLOAT (n={n_float})"),
    ]:
        vals = np.sort(e2d[mask])
        if vals.size == 0:
            continue
        cdf = np.linspace(1.0 / vals.size, 1.0, vals.size)
        ax_cdf.plot(vals, cdf, color=color, linewidth=2, alpha=0.9, label=label)
    ax_cdf.axvline(0.5, color="green", linestyle="--", alpha=0.7, label="PPC threshold 0.5 m")
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel("2D error (m, log)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.set_title("2D error CDF")
    ax_cdf.grid(alpha=0.25, which="both")
    ax_cdf.legend(loc="lower right", framealpha=0.9)

    # Zoomed inset: most active area showing cm tracking
    ax_zoom = fig.add_subplot(gs[1, 1])
    # Pick the section with most FIXED density (~10% window of trajectory)
    if n_fixed > 50:
        # use centroid of FIXED points as zoom center
        fz_center = fused_enu[fixed_mask].mean(axis=0)
        # pick dynamic window size based on local scale
        window = 50.0  # 50 m radius
    else:
        fz_center = truth_enu.mean(axis=0)
        window = 50.0
    ax_zoom.plot(
        truth_enu[:, 0],
        truth_enu[:, 1],
        color=_TRUTH_COLOR,
        linewidth=1.4,
        alpha=0.9,
        label="truth",
        zorder=1,
    )
    near_fixed = fixed_mask & (
        (np.abs(fused_enu[:, 0] - fz_center[0]) < window)
        & (np.abs(fused_enu[:, 1] - fz_center[1]) < window)
    )
    near_float = float_mask & (
        (np.abs(fused_enu[:, 0] - fz_center[0]) < window)
        & (np.abs(fused_enu[:, 1] - fz_center[1]) < window)
    )
    if near_fixed.any():
        ax_zoom.scatter(
            fused_enu[near_fixed, 0], fused_enu[near_fixed, 1],
            s=28, c=_FIXED_COLOR, alpha=0.8,
            edgecolor="white", linewidth=0.3,
            label=f"FIXED ({int(near_fixed.sum())})", zorder=3,
        )
    if near_float.any():
        ax_zoom.scatter(
            fused_enu[near_float, 0], fused_enu[near_float, 1],
            s=28, c=_FLOAT_COLOR, alpha=0.8,
            edgecolor="white", linewidth=0.3,
            label=f"FLOAT ({int(near_float.sum())})", zorder=2,
        )
    ax_zoom.set_xlim(fz_center[0] - window, fz_center[0] + window)
    ax_zoom.set_ylim(fz_center[1] - window, fz_center[1] + window)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("East (m)")
    ax_zoom.set_ylabel("North (m)")
    ax_zoom.set_title(f"Zoom {2 * window:.0f} m window at FIXED centroid")
    ax_zoom.grid(alpha=0.25)
    ax_zoom.legend(loc="best", framealpha=0.9)

    # Coverage bar: rover TOW vs which ones have a solution
    ax_cov = fig.add_subplot(gs[1, 2])
    ref_tows = np.array(sorted(ref.keys()))
    total_ref = len(ref_tows)
    fixed_set = set(tows[fixed_mask].tolist())
    float_set = set(tows[float_mask].tolist())
    n_fixed_ref = sum(1 for t in ref_tows if t in fixed_set)
    n_float_ref = sum(1 for t in ref_tows if t in float_set)
    n_missing = total_ref - n_fixed_ref - n_float_ref

    coverage_counts = [n_fixed_ref, n_float_ref, n_missing]
    coverage_labels = [
        f"FIXED\n{n_fixed_ref} ({100 * n_fixed_ref / total_ref:.1f}%)",
        f"FLOAT\n{n_float_ref} ({100 * n_float_ref / total_ref:.1f}%)",
        f"NO SOL\n{n_missing} ({100 * n_missing / total_ref:.1f}%)",
    ]
    colors = [_FIXED_COLOR, _FLOAT_COLOR, "#bdbdbd"]
    ax_cov.bar(range(3), coverage_counts, color=colors, edgecolor="black", linewidth=0.6)
    ax_cov.set_xticks(range(3))
    ax_cov.set_xticklabels(coverage_labels)
    ax_cov.set_ylabel("Reference rover epochs")
    ax_cov.set_title(f"Epoch coverage (total={total_ref})")
    ax_cov.grid(alpha=0.25, axis="y")

    fixed_pct = 100.0 * fixed_mask.mean()
    ppc_epoch_pass = 100.0 * (e3d <= 0.5).mean()
    title_lines = [
        args.title or args.pos.stem,
        (
            f"n={len(matched)} sols (FIXED {n_fixed}, FLOAT {n_float})  |  "
            f"fix={fixed_pct:.1f}%  |  PPC 3D pass epochs={ppc_epoch_pass:.1f}%  |  "
            f"2D median={np.median(e2d):.3f} m  |  2D p95={np.percentile(e2d, 95):.3f} m"
        ),
    ]
    fig.suptitle("\n".join(title_lines), fontsize=12)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out} ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
