#!/usr/bin/env python3
"""Compare Phase62 span candidates against OSM road centerlines."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import osmnx as ox
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, unary_union

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase43_span_oracle import _load_pool, _split_csv_values  # noqa: E402

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return float("nan")


def _ecef_to_llh(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))
    h = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z, p * (1.0 - _WGS84_E2 * n / (n + h)))
    return math.degrees(lat), math.degrees(lon), h


def _xyz_from_row(row: dict[str, str]) -> np.ndarray:
    return np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)


def _median(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return float(median(vals)) if vals else ""


def _mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else ""


def _p95(values: list[float]) -> float | str:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return ""
    idx = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return float(vals[idx])


def _project_lines(lines: list[LineString], transformer: Transformer) -> list[LineString]:
    out: list[LineString] = []
    for geom in lines:
        parts = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
        for part in parts:
            coords = [transformer.transform(float(lon), float(lat)) for lon, lat in part.coords]
            if len(coords) >= 2:
                out.append(LineString(coords))
    return out


def _road_union_from_osm(*, north: float, south: float, east: float, west: float, epsg: int) -> Any:
    bbox = (west, south, east, north)
    graph = ox.graph_from_bbox(bbox, network_type="drive", simplify=True, truncate_by_edge=True)
    _nodes, edges = ox.graph_to_gdfs(graph)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    geoms: list[LineString] = []
    for geom in edges.geometry:
        if geom is None:
            continue
        if isinstance(geom, (LineString, MultiLineString)):
            geoms.extend(_project_lines([geom], transformer))
    if not geoms:
        raise RuntimeError("OSM query returned no road geometries")
    return unary_union(geoms), transformer, len(geoms)


def _nearest_distance_and_point(point: Point, road_union: Any) -> tuple[float, Point]:
    nearest = nearest_points(point, road_union)[1]
    return float(point.distance(nearest)), nearest


def analyze(
    *,
    internal_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    labels: list[str],
    start_epoch: int,
    end_epoch: int,
    bbox_margin_deg: float,
    epsg: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    span_rows = internal_rows[start_epoch : end_epoch + 1]
    truth_ll = [_ecef_to_llh(_xyz_from_row(row)) for row in span_rows]
    north = max(lat for lat, _lon, _h in truth_ll) + bbox_margin_deg
    south = min(lat for lat, _lon, _h in truth_ll) - bbox_margin_deg
    east = max(lon for _lat, lon, _h in truth_ll) + bbox_margin_deg
    west = min(lon for _lat, lon, _h in truth_ll) - bbox_margin_deg
    road_union, transformer, n_edges = _road_union_from_osm(north=north, south=south, east=east, west=west, epsg=epsg)

    cand_by_label = {str(c["label"]): c for c in candidates}
    epoch_rows: list[dict[str, Any]] = []
    per_label_dist: dict[str, list[float]] = {"truth": []}
    per_label_signed: dict[str, list[float]] = {"truth": []}
    per_label_nearest_xy: dict[str, list[tuple[float, float]]] = {"truth": []}
    per_label_available: dict[str, int] = {"truth": 0}
    for label in labels:
        per_label_dist[label] = []
        per_label_signed[label] = []
        per_label_nearest_xy[label] = []
        per_label_available[label] = 0

    for row in span_rows:
        epoch = int(float(row["epoch"]))
        tow = round(float(row["tow"]), 1)
        truth_xyz = _xyz_from_row(row)
        truth_lat, truth_lon, _h = _ecef_to_llh(truth_xyz)
        tx, ty = transformer.transform(truth_lon, truth_lat)
        truth_point = Point(tx, ty)
        truth_dist, truth_nearest = _nearest_distance_and_point(truth_point, road_union)
        per_label_dist["truth"].append(truth_dist)
        per_label_nearest_xy["truth"].append((truth_nearest.x, truth_nearest.y))
        per_label_available["truth"] += 1
        rec: dict[str, Any] = {
            "epoch": epoch,
            "tow": tow,
            "truth_lat": truth_lat,
            "truth_lon": truth_lon,
            "truth_road_dist_m": truth_dist,
        }
        for label in labels:
            cand = cand_by_label[label]
            pos = cand["pos"].get(tow)
            if pos is None or not np.all(np.isfinite(pos)):
                rec[f"{label}_available"] = False
                rec[f"{label}_road_dist_m"] = ""
                rec[f"{label}_minus_truth_road_dist_m"] = ""
                continue
            lat, lon, _h = _ecef_to_llh(np.asarray(pos, dtype=np.float64))
            x, y = transformer.transform(lon, lat)
            point = Point(x, y)
            dist, nearest = _nearest_distance_and_point(point, road_union)
            per_label_dist[label].append(dist)
            per_label_nearest_xy[label].append((nearest.x, nearest.y))
            per_label_available[label] += 1
            rec[f"{label}_available"] = True
            rec[f"{label}_lat"] = lat
            rec[f"{label}_lon"] = lon
            rec[f"{label}_road_dist_m"] = dist
            rec[f"{label}_minus_truth_road_dist_m"] = dist - truth_dist
        epoch_rows.append(rec)

    summary_rows: list[dict[str, Any]] = [
        {
            "label": "osm_fetch",
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "bbox_north": north,
            "bbox_south": south,
            "bbox_east": east,
            "bbox_west": west,
            "epsg": epsg,
            "road_edge_geometries": n_edges,
        }
    ]
    for label in ["truth", *labels]:
        dists = per_label_dist[label]
        summary_rows.append(
            {
                "label": label,
                "available_epochs": per_label_available[label],
                "median_road_dist_m": _median(dists),
                "mean_road_dist_m": _mean(dists),
                "p95_road_dist_m": _p95(dists),
            }
        )
    truth_dist_by_epoch = [float(row["truth_road_dist_m"]) for row in epoch_rows]
    for label in labels:
        deltas = [
            float(row[f"{label}_minus_truth_road_dist_m"])
            for row in epoch_rows
            if row.get(f"{label}_minus_truth_road_dist_m") not in ("", None)
        ]
        summary_rows.append(
            {
                "label": f"{label}_minus_truth",
                "available_epochs": len(deltas),
                "median_delta_road_dist_m": _median(deltas),
                "mean_delta_road_dist_m": _mean(deltas),
                "p95_abs_delta_road_dist_m": _p95([abs(v) for v in deltas]),
                "candidate_closer_than_truth_epochs": sum(1 for v in deltas if v < 0.0),
                "candidate_farther_than_truth_epochs": sum(1 for v in deltas if v > 0.0),
                "truth_median_road_dist_m": _median(truth_dist_by_epoch),
            }
        )
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        pair = [
            float(row[f"{a}_road_dist_m"]) - float(row[f"{b}_road_dist_m"])
            for row in epoch_rows
            if row.get(f"{a}_road_dist_m") not in ("", None) and row.get(f"{b}_road_dist_m") not in ("", None)
        ]
        summary_rows.append(
            {
                "label": f"{a}_minus_{b}",
                "available_epochs": len(pair),
                "median_delta_road_dist_m": _median(pair),
                "mean_delta_road_dist_m": _mean(pair),
                "p95_abs_delta_road_dist_m": _p95([abs(v) for v in pair]),
            }
        )
    return summary_rows, epoch_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--compare-labels", default="xd_gici_hs,xd_gici_c4")
    parser.add_argument("--start-epoch", type=int, default=2520)
    parser.add_argument("--end-epoch", type=int, default=2881)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--epsg", type=int, default=32653)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase67_osm_road_centerline_feasibility"))
    args = parser.parse_args()

    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    summary_rows, epoch_rows = analyze(
        internal_rows=_read_csv(args.internal_epochs_csv),
        candidates=candidates,
        labels=_split_csv_values(args.compare_labels),
        start_epoch=int(args.start_epoch),
        end_epoch=int(args.end_epoch),
        bbox_margin_deg=float(args.bbox_margin_deg),
        epsg=int(args.epsg),
    )
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary_rows)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_epochs.csv"), epoch_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
