#!/usr/bin/env python3
"""Sweep OSM road-centerline variants for the largest n/r2 residual span."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analyze_phase67_osm_road_centerline_feasibility import _ecef_to_llh  # noqa: E402
from experiments.evaluate import lla_to_ecef  # noqa: E402
from exp_ppc_ctrbpf_fgo import _load_hybrid_pos_file  # noqa: E402
from sim_phase68_osm_road_centerline_correction import _base_label, _float, _road_union_from_osm  # noqa: E402


DEFAULT_SOURCES = {
    "phase70_osm_current": "experiments/results/phase70_osm_road_hs_alpha05_triggered",
    "hs": "experiments/results/libgnss_diag_phase19/gici_full_hisnr",
    "hs45": "experiments/results/libgnss_diag_phase19/gici_full_hisnr45",
    "combo4": "experiments/results/libgnss_diag_phase19/gici_full_combo4",
    "tdcp2329": "experiments/results/libgnss_diag_phase10/tdcp_height_prior_n2_2329",
}


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
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {path}")


def _split_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _split_floats(text: str) -> list[float]:
    return [float(part) for part in _split_csv(text)]


def _xyz(row: dict[str, str]) -> np.ndarray:
    return np.array([_float(row, "ref_x"), _float(row, "ref_y"), _float(row, "ref_z")], dtype=np.float64)


def _distance_weights(rows: list[dict[str, str]]) -> list[float]:
    xyz = [_xyz(row) for row in rows]
    out: list[float] = []
    for idx, pos in enumerate(xyz):
        if idx == 0 or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(xyz[idx - 1])):
            out.append(0.0)
        else:
            out.append(float(np.linalg.norm(pos - xyz[idx - 1])))
    return out


def _p95(values: list[float]) -> float | str:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return ""
    idx = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return float(vals[idx])


def _median(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return float(median(vals)) if vals else ""


def _mean(values: list[float]) -> float | str:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else ""


def _load_sources(spec: str, *, city: str, run: str) -> dict[str, dict[float, np.ndarray]]:
    if spec:
        pairs = []
        for item in _split_csv(spec):
            label, raw_path = item.split("=", 1)
            pairs.append((label, Path(raw_path)))
    else:
        pairs = [(label, Path(path)) for label, path in DEFAULT_SOURCES.items()]

    out: dict[str, dict[float, np.ndarray]] = {}
    for label, path in pairs:
        pos_path = path / f"{city}_{run}_full.pos" if path.is_dir() else path
        if not pos_path.is_file():
            print(f"skip missing source: {label} {pos_path}")
            continue
        out[label], _statuses = _load_hybrid_pos_file(pos_path)
    return out


def _score_errors(errors: list[tuple[float, float]], total_m: float, pass_m: float) -> dict[str, Any]:
    vals = [error for error, _weight in errors if math.isfinite(error)]
    passed = sum(weight for error, weight in errors if math.isfinite(error) and error <= pass_m)
    return {
        "available_epochs": len(vals),
        "pass_m": passed,
        "fail_m": total_m - passed,
        "score_pct": 100.0 * passed / total_m if total_m > 0.0 else "",
        "mean_error_m": _mean(vals),
        "median_error_m": _median(vals),
        "p95_error_m": _p95(vals),
    }


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_csv(args.internal_epochs_csv)
    weights = _distance_weights(rows)
    span_rows = rows[args.start_epoch : args.end_epoch + 1]
    records = []
    for row in span_rows:
        if args.require_runtime_selected and _base_label(row) != args.selected_label:
            continue
        epoch = int(float(row["epoch"]))
        records.append(
            {
                "epoch": epoch,
                "tow": round(float(row["tow"]), 1),
                "truth": _xyz(row),
                "weight": weights[epoch],
            }
        )
    total_m = sum(float(rec["weight"]) for rec in records)

    truth_ll = [_ecef_to_llh(np.asarray(rec["truth"], dtype=np.float64)) for rec in records]
    north = max(lat for lat, _lon, _h in truth_ll) + args.bbox_margin_deg
    south = min(lat for lat, _lon, _h in truth_ll) - args.bbox_margin_deg
    east = max(lon for _lat, lon, _h in truth_ll) + args.bbox_margin_deg
    west = min(lon for _lat, lon, _h in truth_ll) - args.bbox_margin_deg
    road_union, fwd, inv, n_edges = _road_union_from_osm(
        north=north,
        south=south,
        east=east,
        west=west,
        epsg=args.epsg,
    )

    sources = _load_sources(args.sources, city=args.city, run=args.run)
    summary_rows: list[dict[str, Any]] = []
    residual_inputs: dict[str, list[tuple[int, float, float]]] = {}

    for source_label, lookup in sources.items():
        raw_errors: list[tuple[float, float]] = []
        prepared = []
        for rec in records:
            pos = lookup.get(float(rec["tow"]))
            if pos is None:
                continue
            error = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - np.asarray(rec["truth"], dtype=np.float64)))
            raw_errors.append((error, float(rec["weight"])))
            lat, lon, height_m = _ecef_to_llh(np.asarray(pos, dtype=np.float64))
            x, y = fwd.transform(lon, lat)
            point = Point(x, y)
            nearest = nearest_points(point, road_union)[1]
            prepared.append(
                {
                    "epoch": int(rec["epoch"]),
                    "tow": float(rec["tow"]),
                    "truth": np.asarray(rec["truth"], dtype=np.float64),
                    "weight": float(rec["weight"]),
                    "lat": lat,
                    "lon": lon,
                    "height_m": height_m,
                    "x": float(x),
                    "y": float(y),
                    "vx": float(nearest.x - x),
                    "vy": float(nearest.y - y),
                    "road_dist_m": float(point.distance(road_union)),
                }
            )
        raw_score = _score_errors(raw_errors, total_m, args.pass_m)
        summary_rows.append(
            {
                "mode": "raw",
                "source_label": source_label,
                "alpha": "",
                "height_offset_m": "",
                "road_edge_geometries": n_edges,
                "span_total_m": total_m,
                **raw_score,
            }
        )
        residual_inputs[f"raw:{source_label}"] = [
            (rec["epoch"], error, weight)
            for rec, (error, weight) in zip((r for r in records if lookup.get(float(r["tow"])) is not None), raw_errors)
        ]

        for alpha in args.alphas:
            for height_offset_m in args.height_offsets:
                errors: list[tuple[float, float]] = []
                for rec in prepared:
                    nx = float(rec["x"]) + float(rec["vx"]) * alpha
                    ny = float(rec["y"]) + float(rec["vy"]) * alpha
                    lon, lat = inv.transform(nx, ny)
                    corrected = np.asarray(
                        lla_to_ecef(math.radians(lat), math.radians(lon), float(rec["height_m"]) + height_offset_m),
                        dtype=np.float64,
                    )
                    error = float(np.linalg.norm(corrected - np.asarray(rec["truth"], dtype=np.float64)))
                    errors.append((error, float(rec["weight"])))
                score = _score_errors(errors, total_m, args.pass_m)
                summary_rows.append(
                    {
                        "mode": "alpha",
                        "source_label": source_label,
                        "alpha": alpha,
                        "height_offset_m": height_offset_m,
                        "road_edge_geometries": n_edges,
                        "span_total_m": total_m,
                        **score,
                    }
                )

    summary_rows.sort(key=lambda row: float(row["pass_m"]), reverse=True)
    residual_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(summary_rows[: args.residual_top_variants], start=1):
        if row["mode"] == "raw":
            key = f"raw:{row['source_label']}"
            values = residual_inputs.get(key, [])
        else:
            source_label = str(row["source_label"])
            lookup = sources[source_label]
            alpha = float(row["alpha"])
            height_offset_m = float(row["height_offset_m"])
            values = []
            for rec in records:
                pos = lookup.get(float(rec["tow"]))
                if pos is None:
                    continue
                lat0, lon0, h0 = _ecef_to_llh(np.asarray(pos, dtype=np.float64))
                x, y = fwd.transform(lon0, lat0)
                point = Point(x, y)
                nearest = nearest_points(point, road_union)[1]
                lon, lat = inv.transform(x + (nearest.x - x) * alpha, y + (nearest.y - y) * alpha)
                corrected = np.asarray(lla_to_ecef(math.radians(lat), math.radians(lon), h0 + height_offset_m), dtype=np.float64)
                error = float(np.linalg.norm(corrected - np.asarray(rec["truth"], dtype=np.float64)))
                values.append((int(rec["epoch"]), error, float(rec["weight"])))
        current: list[tuple[int, float, float]] = []
        for epoch, error, weight in values:
            if error <= args.pass_m:
                if current:
                    residual_rows.append(_residual_row(rank, row, current))
                    current = []
                continue
            if current and epoch != current[-1][0] + 1:
                residual_rows.append(_residual_row(rank, row, current))
                current = []
            current.append((epoch, error, weight))
        if current:
            residual_rows.append(_residual_row(rank, row, current))
    residual_rows.sort(key=lambda row: (int(row["variant_rank"]), -float(row["fail_m"])))
    return summary_rows, residual_rows


def _residual_row(rank: int, variant: dict[str, Any], items: list[tuple[int, float, float]]) -> dict[str, Any]:
    errors = [error for _epoch, error, _weight in items]
    return {
        "variant_rank": rank,
        "mode": variant["mode"],
        "source_label": variant["source_label"],
        "alpha": variant["alpha"],
        "height_offset_m": variant["height_offset_m"],
        "start_epoch": items[0][0],
        "end_epoch": items[-1][0],
        "n_epochs": len(items),
        "fail_m": sum(weight for _epoch, _error, weight in items),
        "mean_error_m": _mean(errors),
        "median_error_m": _median(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--selected-label", default="xd_gici_hs")
    parser.add_argument("--start-epoch", type=int, default=2520)
    parser.add_argument("--end-epoch", type=int, default=2881)
    parser.add_argument("--bbox-margin-deg", type=float, default=0.002)
    parser.add_argument("--epsg", type=int, default=32653)
    parser.add_argument("--pass-m", type=float, default=0.5)
    parser.add_argument("--sources", default="")
    parser.add_argument("--alphas", default="0,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.75,1.0")
    parser.add_argument("--height-offsets", default="-1.5,-1.25,-1.0,-0.5,0,0.5")
    parser.add_argument("--include-nonselected", action="store_true")
    parser.add_argument("--residual-top-variants", type=int, default=5)
    parser.add_argument("--out-prefix", type=Path, default=Path("experiments/results/phase74_nr2_span1_road_variant_sweep"))
    args = parser.parse_args()
    args.alphas = _split_floats(args.alphas)
    args.height_offsets = _split_floats(args.height_offsets)
    args.require_runtime_selected = not bool(args.include_nonselected)
    summary_rows, residual_rows = analyze(args)
    _write_csv(Path(f"{args.out_prefix}_summary.csv"), summary_rows)
    _write_csv(Path(f"{args.out_prefix}_residuals.csv"), residual_rows)
    if summary_rows:
        print("best:", summary_rows[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
