#!/usr/bin/env python3
"""Diagnostic: overlay Tokyo run2 trajectory with PLATEAU LOD2 building
footprints, with failure-window epochs highlighted.

Used to investigate whether the PLATEAU 3D model coverage in
/tmp/plateau_segment_cache misses any structures (highways, large
canopies) at the §7.16 failure points (Tokyo run2 w7/w9 false-high,
w23-w27 hidden-high).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd


CACHE_ROOT = Path("/tmp/plateau_segment_cache")
NS = {
    "gml": "http://www.opengis.net/gml",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "core": "http://www.opengis.net/citygml/2.0",
}


def _read_pos(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            t = s.split()
            if len(t) < 6:
                continue
            rows.append(t)
    df = pd.DataFrame(rows)
    df.columns = ["date", "time", "lat", "lon", "h", "q", "ns", "sdn", "sde", "sdu", "sdne", "sdeu", "sdun", "age", "ratio"][: df.shape[1]]
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["q"] = df["q"].astype(int)
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M:%S.%f")
    df["elapsed_s"] = (ts - ts.iloc[0]).dt.total_seconds()
    return df[["elapsed_s", "lat", "lon", "q"]]


def _extract_footprints_from_gml(path: Path, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> list[np.ndarray]:
    """Extract building roof / footprint polygons in lon/lat (deg) coordinates within the bbox.

    PLATEAU CityGML stores coordinates as `lat lon height lat lon height ...`
    in EPSG:6697 (JGD2011 + height).  We extract LOD1Solid surfaces.
    """
    polygons: list[np.ndarray] = []
    try:
        tree = ET.parse(path)
    except Exception:
        return polygons
    root = tree.getroot()
    # Iterate every gml:LinearRing and extract its posList (or pos children).
    for ring in root.iter("{http://www.opengis.net/gml}LinearRing"):
        pos_list = ring.find("gml:posList", NS)
        if pos_list is None or not pos_list.text:
            continue
        toks = pos_list.text.split()
        if len(toks) < 9:
            continue
        try:
            arr = np.array([float(x) for x in toks], dtype=np.float64)
        except ValueError:
            continue
        # PLATEAU posList: lat lon height triples
        if arr.size % 3 != 0:
            continue
        pts = arr.reshape(-1, 3)
        lats, lons = pts[:, 0], pts[:, 1]
        if not (lat_range[0] - 0.005 <= lats.mean() <= lat_range[1] + 0.005
                and lon_range[0] - 0.005 <= lons.mean() <= lon_range[1] + 0.005):
            continue
        # We only need a 2D footprint; the surface may be a roof or wall.
        polygons.append(np.column_stack([lons, lats]))
    return polygons


def _failure_window_epochs(pos: pd.DataFrame, w_indices: list[int], window_duration_s: float = 30.0) -> pd.DataFrame:
    out = []
    for w in w_indices:
        t0 = w * window_duration_s
        t1 = (w + 1) * window_duration_s
        mask = (pos["elapsed_s"] >= t0) & (pos["elapsed_s"] < t1)
        sub = pos.loc[mask].copy()
        sub["window_index"] = w
        out.append(sub)
    if not out:
        return pos.iloc[:0].copy()
    return pd.concat(out, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=Path, default=Path("/media/sasaki/aiueo/ai_coding_ws/gnss_cuda_sim_ws/gnss_gpu/experiments/results/demo5_pos/tokyo_run2/rtklib.pos"))
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "internal_docs" / "product_deliverable" / "plots" / "tokyo_run2_building_coverage.png")
    parser.add_argument("--gml-prefix", default="53393")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pos = _read_pos(args.pos)
    print(f"loaded {len(pos)} epochs from {args.pos}")
    lat_range = (pos["lat"].min(), pos["lat"].max())
    lon_range = (pos["lon"].min(), pos["lon"].max())
    print(f"lat {lat_range}  lon {lon_range}")

    # Find Tokyo PLATEAU GML files in cache (mesh code prefix)
    gml_files: list[Path] = []
    for d in sorted(args.cache_root.glob("*")):
        for gml in d.glob(f"{args.gml_prefix}*_bldg*.gml"):
            gml_files.append(gml)
    # De-duplicate by filename
    seen = set()
    unique = []
    for g in gml_files:
        if g.name not in seen:
            seen.add(g.name)
            unique.append(g)
    gml_files = unique
    print(f"found {len(gml_files)} unique Tokyo PLATEAU GML tiles")

    polygons: list[np.ndarray] = []
    for g in gml_files:
        polys = _extract_footprints_from_gml(g, lat_range, lon_range)
        polygons.extend(polys)
    print(f"extracted {len(polygons)} polygons inside route bbox")

    # Failure windows from §7.16 plan
    false_high_w = [7, 9]
    hidden_high_w = [23, 24, 25, 26, 27]

    fh_epochs = _failure_window_epochs(pos, false_high_w)
    hh_epochs = _failure_window_epochs(pos, hidden_high_w)

    fig, ax = plt.subplots(figsize=(11, 10))
    if polygons:
        patches = [Polygon(p, closed=True) for p in polygons]
        coll = PatchCollection(patches, facecolor="#bdc3c7", edgecolor="#7f8c8d", linewidth=0.2, alpha=0.55)
        ax.add_collection(coll)

    # Trajectory coloured by Q
    q_colors = {1: "#2ecc71", 2: "#f39c12", 3: "#3498db", 4: "#9b59b6", 5: "#95a5a6", 6: "#1abc9c"}
    for q, color in q_colors.items():
        mask = pos["q"] == q
        if mask.any():
            ax.scatter(pos.loc[mask, "lon"], pos.loc[mask, "lat"], c=color, s=2, alpha=0.6,
                       label=f"Q={q} ({int(mask.sum())})")

    # Highlight failure windows
    if not fh_epochs.empty:
        ax.scatter(fh_epochs["lon"], fh_epochs["lat"], c="#e74c3c", s=20, marker="X",
                   edgecolor="black", linewidth=0.4,
                   label=f"false-high w7/w9 ({len(fh_epochs)})", zorder=4)
    if not hh_epochs.empty:
        ax.scatter(hh_epochs["lon"], hh_epochs["lat"], c="#8e44ad", s=20, marker="P",
                   edgecolor="black", linewidth=0.4,
                   label=f"hidden-high w23-w27 ({len(hh_epochs)})", zorder=4)

    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(f"Tokyo run2 — PLATEAU LOD1/LOD2 footprints + trajectory + §7.16 failure windows\n"
                 f"({len(polygons)} buildings, {len(pos)} epochs)")
    ax.legend(loc="best", fontsize=9, markerscale=3)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lon_range[0] - 0.001, lon_range[1] + 0.001)
    ax.set_ylim(lat_range[0] - 0.001, lat_range[1] + 0.001)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
