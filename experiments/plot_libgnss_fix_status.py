#!/usr/bin/env python3
# ruff: noqa: E402
"""Render a fix/float-status figure for a libgnss++ PPC RTK result.

Produces a 3-panel image:

1. ENU trajectory coloured by solution status (FIXED / FLOAT / REF), with
   truth overlaid.
2. Per-epoch 2D error timeline with PPC 0.5 m pass band and colour bars
   indicating fix status.
3. Solution status vs LAMBDA ratio timeline.

The .pos file is the output of ``experiments/exp_ppc_libgnss_rtk.py``
(which wraps ``third_party/gnssplusplus/build/apps/gnss_solve``).
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


def _status_label(status: int) -> str:
    return {1: "SINGLE", 2: "DGPS", 3: "FLOAT", 4: "FIXED"}.get(status, f"ST{status}")


_STATUS_COLOR = {
    4: "#1a9850",
    3: "#fdae61",
    2: "#9970ab",
    1: "#d73027",
}


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
    ratios = np.array([r[4] for r in matched])

    center = np.mean(truth, axis=0)
    lat0, lon0 = _ecef_to_lat_lon(center)
    R = _enu_rotation(lat0, lon0)

    truth_enu = np.array([R @ (t - center) for t in truth])
    fused_enu = np.array([R @ (f - center) for f in fused])
    diff_enu = np.array([R @ (f - t) for f, t in zip(fused, truth)])

    e2d = np.linalg.norm(diff_enu[:, :2], axis=1)
    e3d = np.linalg.norm(fused - truth, axis=1)
    t_rel = tows - tows[0]

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])

    # Panel 1: trajectory
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(
        truth_enu[:, 0],
        truth_enu[:, 1],
        color="black",
        linewidth=1.0,
        alpha=0.55,
        label="truth",
    )
    for st, col in _STATUS_COLOR.items():
        mask = status == st
        if not mask.any():
            continue
        ax_traj.scatter(
            fused_enu[mask, 0],
            fused_enu[mask, 1],
            s=18,
            c=col,
            edgecolor="none",
            alpha=0.9,
            label=f"{_status_label(st)} ({mask.sum()})",
        )
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.set_xlabel("East from segment centroid (m)")
    ax_traj.set_ylabel("North from segment centroid (m)")
    ax_traj.set_title("libgnss++ solution trajectory (ENU)")
    ax_traj.grid(alpha=0.25)
    ax_traj.legend(loc="best", framealpha=0.9)

    # Panel 2: 2D error timeline
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.axhspan(0.0, 0.5, color="#dfffe0", alpha=0.5, label="PPC 0.5 m pass band")
    for st, col in _STATUS_COLOR.items():
        mask = status == st
        if not mask.any():
            continue
        ax_err.scatter(
            t_rel[mask],
            e2d[mask],
            s=14,
            c=col,
            edgecolor="none",
            alpha=0.95,
            label=_status_label(st),
        )
    ax_err.axhline(0.5, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax_err.set_ylabel("2D error from truth (m)")
    ax_err.set_yscale("symlog", linthresh=0.2)
    ax_err.set_title("Per-epoch 2D error (log-symmetric at 0.2 m)")
    ax_err.grid(alpha=0.25)
    ax_err.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax_err.set_xlim(t_rel.min(), t_rel.max())

    # Panel 3: LAMBDA ratio timeline
    ax_ratio = fig.add_subplot(gs[1, 1], sharex=ax_err)
    for st, col in _STATUS_COLOR.items():
        mask = status == st
        if not mask.any():
            continue
        ax_ratio.scatter(
            t_rel[mask],
            ratios[mask],
            s=14,
            c=col,
            edgecolor="none",
            alpha=0.95,
        )
    ax_ratio.axhline(2.0, color="gray", linestyle=":", linewidth=1, alpha=0.8,
                     label="hold-ratio threshold 2.0")
    ax_ratio.axhline(3.0, color="green", linestyle="--", linewidth=1, alpha=0.8,
                     label="fix ratio 3.0")
    ax_ratio.set_xlabel("Time from segment start (s)")
    ax_ratio.set_ylabel("LAMBDA ratio")
    ax_ratio.set_title("Integer ambiguity search ratio")
    ax_ratio.grid(alpha=0.25)
    ax_ratio.legend(loc="upper right", framealpha=0.9)

    fixed_pct = 100.0 * (status == 4).mean()
    ppc_pass_mask = e3d <= 0.5
    ppc_epoch_pass = 100.0 * ppc_pass_mask.mean()
    title_lines = [
        args.title or args.pos.stem,
        (
            f"n={len(matched)} solutions  |  fix={fixed_pct:.1f}%  |  "
            f"PPC 3D pass epochs={ppc_epoch_pass:.1f}%  |  "
            f"2D median={np.median(e2d):.3f} m  |  2D p95={np.percentile(e2d, 95):.3f} m"
        ),
    ]
    fig.suptitle("\n".join(title_lines), fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"wrote {args.out} ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
