#!/usr/bin/env python3
"""Replay OSM road-centerline lateral corrections for a PPC span."""

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
from analyze_phase67_osm_road_centerline_feasibility import _ecef_to_llh  # noqa: E402
from experiments.evaluate import lla_to_ecef  # noqa: E402


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


def _xyz_from_row(row: dict[str, str]) -> np.ndarray:
    return np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [(_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")) for row in rows]
    weights: list[float] = []
    for i, pos in enumerate(xyz):
        if i == 0:
            weights.append(0.0)
            continue
        prev = xyz[i - 1]
        if any(not math.isfinite(v) for v in (*prev, *pos)):
            weights.append(0.0)
        else:
            weights.append(float(math.dist(prev, pos)))
    return weights


def _base_label(row: dict[str, str]) -> str:
    label = row.get("rtkdiag_selected_base_label", "")
    if label:
        return label
    return row.get("rtkdiag_selected_label", "").removesuffix("+rnk")


def _median(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return float(median(vals)) if vals else ""


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


def _road_union_from_osm(*, north: float, south: float, east: float, west: float, epsg: int) -> tuple[Any, Transformer, Transformer, int]:
    bbox = (west, south, east, north)
    graph = ox.graph_from_bbox(bbox, network_type="drive", simplify=True, truncate_by_edge=True)
    _nodes, edges = ox.graph_to_gdfs(graph)
    fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inv = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    geoms: list[LineString] = []
    for geom in edges.geometry:
        if geom is None:
            continue
        if isinstance(geom, (LineString, MultiLineString)):
            geoms.extend(_project_lines([geom], fwd))
    if not geoms:
        raise RuntimeError("OSM query returned no road geometries")
    return unary_union(geoms), fwd, inv, len(geoms)


def _candidate_by_label(candidates: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for cand in candidates:
        if str(cand["label"]) == label:
            return cand
    raise SystemExit(f"candidate label not loaded: {label}")


def _correct_xy(point: Point, road_union: Any, mode: str, alpha: float, target_distance_m: float) -> tuple[float, float, float]:
    nearest = nearest_points(point, road_union)[1]
    vx = nearest.x - point.x
    vy = nearest.y - point.y
    dist = math.hypot(vx, vy)
    if dist <= 1e-9:
        return point.x, point.y, 0.0
    if mode == "alpha":
        move = alpha * dist
    elif mode == "target-distance":
        move = max(0.0, dist - target_distance_m)
    else:
        raise ValueError(f"unknown mode: {mode}")
    move = min(move, dist)
    return point.x + vx / dist * move, point.y + vy / dist * move, dist


def analyze(
    *,
    internal_rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    selected_label: str,
    start_epoch: int,
    end_epoch: int,
    bbox_margin_deg: float,
    epsg: int,
    modes: list[str],
    alphas: list[float],
    target_distances: list[float],
    road_dist_min_values: list[float],
    min_contiguous_epochs_values: list[int],
    pass_m: float,
    require_runtime_selected: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    span_rows = internal_rows[start_epoch : end_epoch + 1]
    truth_ll = [_ecef_to_llh(_xyz_from_row(row)) for row in span_rows]
    north = max(lat for lat, _lon, _h in truth_ll) + bbox_margin_deg
    south = min(lat for lat, _lon, _h in truth_ll) - bbox_margin_deg
    east = max(lon for _lat, lon, _h in truth_ll) + bbox_margin_deg
    west = min(lon for _lat, lon, _h in truth_ll) - bbox_margin_deg
    road_union, fwd, inv, n_edges = _road_union_from_osm(north=north, south=south, east=east, west=west, epsg=epsg)
    cand = _candidate_by_label(candidates, selected_label)
    weights = _distance_weights(internal_rows)

    prepared: list[dict[str, Any]] = []
    for row in span_rows:
        epoch = int(float(row["epoch"]))
        tow = round(float(row["tow"]), 1)
        runtime_selected = _base_label(row)
        if require_runtime_selected and runtime_selected != selected_label:
            continue
        truth = _xyz_from_row(row)
        pos = cand["pos"].get(tow)
        if pos is None or not np.all(np.isfinite(pos)):
            continue
        lat, lon, h = _ecef_to_llh(np.asarray(pos, dtype=np.float64))
        x, y = fwd.transform(lon, lat)
        point = Point(x, y)
        road_dist = float(point.distance(road_union))
        selected_error = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - truth))
        prepared.append(
            {
                "epoch": epoch,
                "tow": tow,
                "truth": truth,
                "height_m": h,
                "point": point,
                "road_dist_m": road_dist,
                "selected_error": selected_error,
                "selected_pass": selected_error <= pass_m,
                "weight": weights[epoch],
                "runtime_selected": runtime_selected,
            }
        )

    eligible_by_guard: dict[tuple[float, int], set[int]] = {}
    for road_min in road_dist_min_values:
        for min_len in min_contiguous_epochs_values:
            eligible: set[int] = set()
            current: list[int] = []
            for idx, rec in enumerate(prepared):
                ok = float(rec["road_dist_m"]) >= road_min
                contiguous = bool(current) and int(rec["epoch"]) == int(prepared[current[-1]]["epoch"]) + 1
                if ok and (not current or contiguous):
                    current.append(idx)
                    continue
                if len(current) >= min_len:
                    eligible.update(current)
                current = [idx] if ok else []
            if len(current) >= min_len:
                eligible.update(current)
            eligible_by_guard[(road_min, min_len)] = eligible

    out: list[dict[str, Any]] = []
    epoch_debug: list[dict[str, Any]] = []
    configs: list[tuple[str, float, float, float, int]] = []
    for mode in modes:
        if mode == "alpha":
            configs.extend(
                (mode, alpha, 0.0, road_min, min_len)
                for alpha in alphas
                for road_min in road_dist_min_values
                for min_len in min_contiguous_epochs_values
            )
        elif mode == "target-distance":
            configs.extend(
                (mode, 0.0, target, road_min, min_len)
                for target in target_distances
                for road_min in road_dist_min_values
                for min_len in min_contiguous_epochs_values
            )
    for mode, alpha, target, road_min, min_len in configs:
        base_pass = 0.0
        replay_pass = 0.0
        total = 0.0
        corrected_errors: list[float] = []
        road_dists: list[float] = []
        good = 0
        bad = 0
        eligible_indices = eligible_by_guard[(road_min, min_len)]
        for idx, rec in enumerate(prepared):
            weight = float(rec["weight"])
            total += weight
            if bool(rec["selected_pass"]):
                base_pass += weight
            road_dist = float(rec["road_dist_m"])
            if idx in eligible_indices:
                x, y, road_dist = _correct_xy(rec["point"], road_union, mode, alpha, target)
                lon, lat = inv.transform(x, y)
                corrected = np.asarray(
                    lla_to_ecef(math.radians(lat), math.radians(lon), float(rec["height_m"])),
                    dtype=np.float64,
                )
                err = float(np.linalg.norm(corrected - np.asarray(rec["truth"], dtype=np.float64)))
            else:
                err = float(rec["selected_error"])
            corrected_errors.append(err)
            road_dists.append(float(rec["road_dist_m"]))
            chosen_pass = err <= pass_m
            if chosen_pass:
                replay_pass += weight
            if chosen_pass and not bool(rec["selected_pass"]):
                good += 1
            elif bool(rec["selected_pass"]) and not chosen_pass:
                bad += 1
            if (mode, alpha, target, road_min, min_len) == configs[0]:
                epoch_debug.append(
                    {
                        "epoch": rec["epoch"],
                        "tow": rec["tow"],
                        "road_dist_m": road_dist,
                        "triggered": idx in eligible_indices,
                        "selected_error_m": rec["selected_error"],
                        "corrected_error_m": err,
                    }
                )
        out.append(
            {
                "mode": mode,
                "alpha": alpha,
                "target_distance_m": target,
                "road_dist_min_m": road_min,
                "min_contiguous_epochs": min_len,
                "road_edge_geometries": n_edges,
                "epochs": len(prepared),
                "triggered_epochs": len(eligible_indices),
                "base_pass_m": base_pass,
                "replay_pass_m": replay_pass,
                "gain_m": replay_pass - base_pass,
                "total_m": total,
                "base_score_pct": 100.0 * base_pass / total if total > 0.0 else "",
                "replay_score_pct": 100.0 * replay_pass / total if total > 0.0 else "",
                "good_epochs": good,
                "bad_epochs": bad,
                "median_road_dist_m": _median(road_dists),
                "median_corrected_error_m": _median(corrected_errors),
                "p95_corrected_error_m": _p95(corrected_errors),
            }
        )
    out.sort(key=lambda row: float(row["gain_m"]), reverse=True)
    return out, epoch_debug


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--candidate-dirs", required=True)
    parser.add_argument("--selected-label", default="xd_gici_hs")
    parser.add_argument("--start-epoch", type=int, default=2520)
    parser.add_argument("--end-epoch", type=int, default=2881)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--epsg", type=int, default=32653)
    parser.add_argument("--modes", default="alpha,target-distance")
    parser.add_argument("--alphas", default="0,0.25,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument("--target-distances", default="0,0.5,1.0,1.2,1.5,2.0")
    parser.add_argument("--road-dist-min-values", default="0")
    parser.add_argument("--min-contiguous-epochs-values", default="1")
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--include-nonselected", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase68_osm_road_centerline_correction"))
    args = parser.parse_args()
    candidates = _load_pool(_split_csv_values(args.labels), _split_csv_values(args.candidate_dirs), args.city, args.run)
    summary, debug = analyze(
        internal_rows=_read_csv(args.internal_epochs_csv),
        candidates=candidates,
        selected_label=str(args.selected_label),
        start_epoch=int(args.start_epoch),
        end_epoch=int(args.end_epoch),
        bbox_margin_deg=float(args.bbox_margin_deg),
        epsg=int(args.epsg),
        modes=_split_csv_values(args.modes),
        alphas=[float(v) for v in _split_csv_values(args.alphas)],
        target_distances=[float(v) for v in _split_csv_values(args.target_distances)],
        road_dist_min_values=[float(v) for v in _split_csv_values(args.road_dist_min_values)],
        min_contiguous_epochs_values=[int(float(v)) for v in _split_csv_values(args.min_contiguous_epochs_values)],
        pass_m=float(args.pass_m),
        require_runtime_selected=not bool(args.include_nonselected),
    )
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_summary.csv"), summary)
    _write_csv(args.out_prefix.with_name(args.out_prefix.name + "_debug_epochs.csv"), debug)
    if summary:
        print("best:", summary[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
