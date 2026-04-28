#!/usr/bin/env python3
"""Validate preprocessed GSDC2023 ``phone_data.mat`` bundles from Python.

This script is intentionally narrower than the upstream MATLAB pipeline. It
checks that the downloaded ``dataset_2023`` tree is readable from Python,
summarizes the stored observation/trajectory fields, and compares the provided
baseline / optional MATLAB result files against ground truth on training trips.

Expected layout:

    ref/gsdc2023/dataset_2023/
      train/<course>/<phone>/phone_data.mat
      train/<course>/<phone>/gt.mat
      test/<course>/<phone>/phone_data.mat

The original taroz.net zip URL currently returns 404. If the dataset is not
present locally, download it from Kaggle and unzip it under the path above.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gsdc2023_raw_bridge import resolve_gsdc2023_data_root
import zipfile

import numpy as np
from scipy.io import loadmat


_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = resolve_gsdc2023_data_root()
_WGS84_A = 6_378_137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F * _WGS84_F
_WGS84_EP2 = (_WGS84_A * _WGS84_A - _WGS84_B * _WGS84_B) / (_WGS84_B * _WGS84_B)
_EARTH_RADIUS = _WGS84_A
_FREQ_TYPES = ("L1", "L5")


@dataclass
class PositionMetrics:
    matched_epochs: int
    score_m: float | None
    p50_m: float | None
    p95_m: float | None
    rms_2d_m: float | None
    rms_3d_m: float | None


@dataclass
class TripValidationResult:
    trip_name: str
    phone_data_path: Path
    obs_epochs: int | None
    baseline_epochs: int
    gt_epochs: int
    nsat: int | None
    dt_s: float | None
    counts_by_freq: dict[str, dict[str, int]]
    baseline_metrics: PositionMetrics | None
    result_gnss_metrics: PositionMetrics | None
    result_gnss_imu_metrics: PositionMetrics | None


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.item()
    return value


def _field_names(value: Any) -> list[str]:
    value = _unwrap_singleton(value)
    if hasattr(value, "_fieldnames") and getattr(value, "_fieldnames"):
        return list(getattr(value, "_fieldnames"))
    dtype = getattr(value, "dtype", None)
    if dtype is not None and getattr(dtype, "names", None):
        return list(dtype.names)
    if isinstance(value, dict):
        return list(value)
    return []


def _get_field(value: Any, name: str) -> Any | None:
    value = _unwrap_singleton(value)
    if value is None:
        return None
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, dict):
        return value.get(name)
    dtype = getattr(value, "dtype", None)
    if dtype is not None and getattr(dtype, "names", None) and name in dtype.names:
        return value[name]
    return None


def _numeric_array(value: Any) -> np.ndarray | None:
    value = _unwrap_singleton(value)
    if value is None:
        return None
    if isinstance(value, str):
        return None
    if type(value).__name__ == "MatlabOpaque":
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object:
        if arr.size != 1:
            return None
        return _numeric_array(arr.item())
    try:
        return np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        return None


def _bool_array(value: Any) -> np.ndarray | None:
    arr = _numeric_array(value)
    if arr is None:
        return None
    return np.asarray(arr != 0).squeeze()


def _scalar_int(value: Any) -> int | None:
    arr = _numeric_array(value)
    if arr is None or arr.size == 0:
        return None
    return int(np.asarray(arr).reshape(-1)[0])


def _scalar_float(value: Any) -> float | None:
    arr = _numeric_array(value)
    if arr is None or arr.size == 0:
        return None
    return float(np.asarray(arr).reshape(-1)[0])


def _vector(value: Any) -> np.ndarray | None:
    arr = _numeric_array(value)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64).squeeze()
    return arr.reshape(-1) if arr.size else np.empty((0,), dtype=np.float64)


def _matrix_nx3(value: Any) -> np.ndarray | None:
    arr = _numeric_array(value)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64).squeeze()
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 3:
            return None
        return arr.reshape(1, 3)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    if arr.size % 3 != 0:
        return None
    return arr.reshape(-1, 3)


def _llh_to_ecef(llh_deg: np.ndarray) -> np.ndarray:
    llh_deg = np.asarray(llh_deg, dtype=np.float64)
    lat = np.deg2rad(llh_deg[:, 0])
    lon = np.deg2rad(llh_deg[:, 1])
    alt = llh_deg[:, 2]
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    N = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1.0 - _WGS84_E2) + alt) * sin_lat
    return np.column_stack((x, y, z))


def _ecef_to_llh(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    out = np.empty((xyz.shape[0], 3), dtype=np.float64)
    for i, (x, y, z) in enumerate(xyz):
        lon = math.atan2(y, x)
        p = math.hypot(x, y)
        theta = math.atan2(z * _WGS84_A, p * _WGS84_B)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        lat = math.atan2(
            z + _WGS84_EP2 * _WGS84_B * sin_theta ** 3,
            p - _WGS84_E2 * _WGS84_A * cos_theta ** 3,
        )
        sin_lat = math.sin(lat)
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - N
        out[i] = (math.degrees(lat), math.degrees(lon), alt)
    return out


def _haversine_dist_m(llh_a: np.ndarray, llh_b: np.ndarray) -> np.ndarray:
    lat1 = np.deg2rad(llh_a[:, 0])
    lon1 = np.deg2rad(llh_a[:, 1])
    lat2 = np.deg2rad(llh_b[:, 0])
    lon2 = np.deg2rad(llh_b[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def _nearest_indices(ref_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(ref_t, query_t)
    idx = np.clip(idx, 0, len(ref_t) - 1)
    prev_idx = np.clip(idx - 1, 0, len(ref_t) - 1)
    choose_prev = np.abs(ref_t[prev_idx] - query_t) <= np.abs(ref_t[idx] - query_t)
    return np.where(choose_prev, prev_idx, idx)


def _align_indices(
    est_n: int,
    gt_n: int,
    est_t: np.ndarray | None,
    gt_t: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        est_t is not None and gt_t is not None and len(est_t) == est_n and len(gt_t) == gt_n
        and est_n > 0 and gt_n > 0
    ):
        est_idx = np.arange(est_n, dtype=np.int64)
        gt_idx = _nearest_indices(np.asarray(gt_t, dtype=np.float64), np.asarray(est_t, dtype=np.float64))
        return est_idx, gt_idx.astype(np.int64)
    n = min(est_n, gt_n)
    idx = np.arange(n, dtype=np.int64)
    return idx, idx


def _extract_position_struct(mat: dict[str, Any], key: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    pos = mat.get(key)
    if pos is None:
        return None, None
    xyz = _matrix_nx3(_get_field(pos, "xyz"))
    llh = _matrix_nx3(_get_field(pos, "llh"))
    if xyz is None and llh is not None:
        xyz = _llh_to_ecef(llh)
    if llh is None and xyz is not None:
        llh = _ecef_to_llh(xyz)
    return xyz, llh


def _extract_time_struct(mat: dict[str, Any], key: str) -> np.ndarray | None:
    val = mat.get(key)
    if val is None:
        return None
    time_t = _vector(_get_field(val, "t"))
    if time_t is not None:
        return time_t
    return _vector(val)


def _load_csv_ground_truth(trip_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    gt_csv = trip_dir / "ground_truth.csv"
    if not gt_csv.is_file():
        return None, None
    llh_rows: list[list[float]] = []
    time_rows: list[float] = []
    with open(gt_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                llh_rows.append([
                    float(row["LatitudeDegrees"]),
                    float(row["LongitudeDegrees"]),
                    float(row["AltitudeMeters"]),
                ])
                time_rows.append(float(row["UnixTimeMillis"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not llh_rows:
        return None, None
    return _llh_to_ecef(np.asarray(llh_rows, dtype=np.float64)), np.asarray(time_rows, dtype=np.float64)


def _open_csv_reader(path: Path) -> csv.DictReader:
    if zipfile.is_zipfile(path):
        zf = zipfile.ZipFile(path)
        members = [name for name in zf.namelist() if not name.endswith("/")]
        if not members:
            zf.close()
            raise ValueError(f"Zip file is empty: {path}")
        member = members[0]
        fh = TextIOWrapper(zf.open(member, "r"), encoding="utf-8", newline="")
        reader = csv.DictReader(fh)
        reader._gnss_gpu_zipfile = zf  # type: ignore[attr-defined]
        reader._gnss_gpu_textio = fh  # type: ignore[attr-defined]
        return reader
    fh = path.open(newline="")
    reader = csv.DictReader(fh)
    reader._gnss_gpu_textio = fh  # type: ignore[attr-defined]
    return reader


def _close_csv_reader(reader: csv.DictReader) -> None:
    fh = getattr(reader, "_gnss_gpu_textio", None)
    if fh is not None:
        fh.close()
    zf = getattr(reader, "_gnss_gpu_zipfile", None)
    if zf is not None:
        zf.close()


def _load_raw_device_gnss_baseline(trip_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    device_gnss = trip_dir / "device_gnss.csv"
    if not device_gnss.is_file():
        return None, None

    reader = _open_csv_reader(device_gnss)
    seen: set[int] = set()
    time_rows: list[float] = []
    xyz_rows: list[list[float]] = []
    try:
        for row in reader:
            try:
                t_ms = int(float(row["utcTimeMillis"]))
                x = float(row["WlsPositionXEcefMeters"])
                y = float(row["WlsPositionYEcefMeters"])
                z = float(row["WlsPositionZEcefMeters"])
            except (KeyError, TypeError, ValueError):
                continue
            if t_ms in seen:
                continue
            if not np.isfinite([x, y, z]).all():
                continue
            seen.add(t_ms)
            time_rows.append(float(t_ms))
            xyz_rows.append([x, y, z])
    finally:
        _close_csv_reader(reader)

    if not xyz_rows:
        return None, None

    times = np.asarray(time_rows, dtype=np.float64)
    xyz = np.asarray(xyz_rows, dtype=np.float64)
    order = np.argsort(times)
    return xyz[order], times[order]


def _compute_metrics(
    est_xyz: np.ndarray | None,
    est_llh: np.ndarray | None,
    est_t: np.ndarray | None,
    gt_xyz: np.ndarray | None,
    gt_llh: np.ndarray | None,
    gt_t: np.ndarray | None,
) -> PositionMetrics | None:
    if est_xyz is None or gt_xyz is None or len(est_xyz) == 0 or len(gt_xyz) == 0:
        return None
    est_idx, gt_idx = _align_indices(len(est_xyz), len(gt_xyz), est_t, gt_t)
    if len(est_idx) == 0:
        return None

    est_xyz_sel = est_xyz[est_idx]
    gt_xyz_sel = gt_xyz[gt_idx]
    if est_llh is None:
        est_llh = _ecef_to_llh(est_xyz)
    if gt_llh is None:
        gt_llh = _ecef_to_llh(gt_xyz)
    est_llh_sel = est_llh[est_idx]
    gt_llh_sel = gt_llh[gt_idx]

    horizontal = _haversine_dist_m(est_llh_sel, gt_llh_sel)
    delta_xyz = est_xyz_sel - gt_xyz_sel
    delta_xy = delta_xyz[:, :2]

    return PositionMetrics(
        matched_epochs=len(est_idx),
        score_m=float((np.percentile(horizontal, 50) + np.percentile(horizontal, 95)) / 2.0),
        p50_m=float(np.percentile(horizontal, 50)),
        p95_m=float(np.percentile(horizontal, 95)),
        rms_2d_m=float(np.sqrt(np.mean(np.sum(delta_xy * delta_xy, axis=1)))),
        rms_3d_m=float(np.sqrt(np.mean(np.sum(delta_xyz * delta_xyz, axis=1)))),
    )


def _load_mat(path: Path) -> dict[str, Any]:
    raw = loadmat(path, struct_as_record=False, squeeze_me=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _observation_counts(obs: Any) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for freq in _FREQ_TYPES:
        band = _get_field(obs, freq)
        if band is None:
            continue
        band_counts: dict[str, int] = {}
        for field in ("P", "D", "L", "resPc", "resD", "resL"):
            arr = _numeric_array(_get_field(band, field))
            if arr is not None:
                band_counts[field] = int(np.isfinite(arr).sum())
        if band_counts:
            counts[freq] = band_counts
    return counts


def validate_trip(trip_dir: Path) -> TripValidationResult:
    phone_data_path = trip_dir / "phone_data.mat"
    phone_data = None
    obs = None
    posbl_xyz = posbl_llh = timebl_t = None
    if phone_data_path.is_file():
        phone_data = _load_mat(phone_data_path)
        obs = phone_data.get("obs")
        posbl_xyz, posbl_llh = _extract_position_struct(phone_data, "posbl")
        timebl_t = _extract_time_struct(phone_data, "timebl")
    else:
        posbl_xyz, timebl_t = _load_raw_device_gnss_baseline(trip_dir)
        if posbl_xyz is None:
            raise FileNotFoundError(
                f"Missing both phone_data.mat and usable raw baseline under: {trip_dir}",
            )
        posbl_llh = _ecef_to_llh(posbl_xyz)

    gt_mat_path = trip_dir / "gt.mat"
    gt_xyz = gt_llh = gt_t = None
    if gt_mat_path.is_file():
        gt_mat = _load_mat(gt_mat_path)
        gt_xyz, gt_llh = _extract_position_struct(gt_mat, "posgt")
        gt_t = _extract_time_struct(gt_mat, "timegt")
    else:
        gt_xyz, gt_t = _load_csv_ground_truth(trip_dir)
        gt_llh = _ecef_to_llh(gt_xyz) if gt_xyz is not None else None

    result_gnss_xyz = result_gnss_llh = None
    result_gnss_imu_xyz = result_gnss_imu_llh = None
    result_gnss_path = trip_dir / "result_gnss.mat"
    if result_gnss_path.is_file():
        result_gnss_xyz, result_gnss_llh = _extract_position_struct(_load_mat(result_gnss_path), "posest")
    result_gnss_imu_path = trip_dir / "result_gnss_imu.mat"
    if result_gnss_imu_path.is_file():
        result_gnss_imu_xyz, result_gnss_imu_llh = _extract_position_struct(_load_mat(result_gnss_imu_path), "posest")

    obs_epochs = _scalar_int(_get_field(obs, "n")) if obs is not None else None
    nsat = _scalar_int(_get_field(obs, "nsat")) if obs is not None else None
    dt_s = _scalar_float(_get_field(obs, "dt")) if obs is not None else None
    counts_by_freq = _observation_counts(obs) if obs is not None else {}
    if dt_s is None and timebl_t is not None and len(timebl_t) > 1:
        dt_s = float(np.median(np.diff(np.asarray(timebl_t, dtype=np.float64))) / 1000.0)

    baseline_metrics = _compute_metrics(posbl_xyz, posbl_llh, timebl_t, gt_xyz, gt_llh, gt_t)
    result_gnss_metrics = _compute_metrics(result_gnss_xyz, result_gnss_llh, timebl_t, gt_xyz, gt_llh, gt_t)
    result_gnss_imu_metrics = _compute_metrics(
        result_gnss_imu_xyz, result_gnss_imu_llh, timebl_t, gt_xyz, gt_llh, gt_t,
    )

    dataset = trip_dir.parent.parent.name
    course = trip_dir.parent.name
    phone = trip_dir.name
    trip_name = f"{dataset}/{course}/{phone}"

    return TripValidationResult(
        trip_name=trip_name,
        phone_data_path=phone_data_path,
        obs_epochs=obs_epochs,
        baseline_epochs=0 if posbl_xyz is None else int(len(posbl_xyz)),
        gt_epochs=0 if gt_xyz is None else int(len(gt_xyz)),
        nsat=nsat,
        dt_s=dt_s,
        counts_by_freq=counts_by_freq,
        baseline_metrics=baseline_metrics,
        result_gnss_metrics=result_gnss_metrics,
        result_gnss_imu_metrics=result_gnss_imu_metrics,
    )


def discover_trip_dirs(root: Path, dataset: str | None = None) -> list[Path]:
    search_roots: list[Path] = []
    if dataset is not None:
        search_roots.append(root / dataset)
    else:
        if (root / "train").is_dir():
            search_roots.append(root / "train")
        if (root / "test").is_dir():
            search_roots.append(root / "test")
        if not search_roots:
            search_roots.append(root)

    trips: list[Path] = []
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for mat_path in sorted(search_root.glob("*/*/phone_data.mat")):
            trips.append(mat_path.parent)
    return trips


def _fmt_float(value: float | None) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.2f}"


def _fmt_counts(counts: dict[str, int]) -> str:
    parts = []
    for key in ("P", "D", "L", "resPc", "resD", "resL"):
        if key in counts:
            parts.append(f"{key}={counts[key]}")
    return ",".join(parts) if parts else "-"


def _print_result(result: TripValidationResult) -> None:
    base = result.baseline_metrics
    gnss = result.result_gnss_metrics
    imu = result.result_gnss_imu_metrics
    l1 = _fmt_counts(result.counts_by_freq.get("L1", {}))
    l5 = _fmt_counts(result.counts_by_freq.get("L5", {}))
    print(
        f"{result.trip_name}: "
        f"obs={result.obs_epochs if result.obs_epochs is not None else '-'} "
        f"bl={result.baseline_epochs} gt={result.gt_epochs} "
        f"nsat={result.nsat if result.nsat is not None else '-'} "
        f"dt={_fmt_float(result.dt_s)}s "
        f"baseline_score={_fmt_float(None if base is None else base.score_m)} "
        f"gnss_score={_fmt_float(None if gnss is None else gnss.score_m)} "
        f"imu_score={_fmt_float(None if imu is None else imu.score_m)} "
        f"L1[{l1}] L5[{l5}]"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=_DEFAULT_ROOT, help="dataset_2023 root directory")
    p.add_argument("--dataset", choices=("train", "test"), default=None, help="limit to train or test")
    p.add_argument("--trip", type=str, default=None, help="relative trip path: <course>/<phone> or train/<course>/<phone>")
    p.add_argument("--all", action="store_true", help="validate all discovered trips")
    args = p.parse_args()

    root = args.data_root
    if not root.is_dir():
        print(f"GSDC2023 dataset root not found: {root}")
        print("Expected a local unzip of dataset_2023 with train/test trip directories.")
        print("The taroz.net zip URL currently returns 404; use Kaggle after accepting competition rules.")
        print("Example: kaggle competitions download -c smartphone-decimeter-2023")
        sys.exit(1)

    if args.trip:
        trip_arg = Path(args.trip)
        if len(trip_arg.parts) == 2 and args.dataset is not None:
            trip_dir = root / args.dataset / trip_arg
        else:
            trip_dir = root / trip_arg
        trip_dirs = [trip_dir]
    else:
        trip_dirs = discover_trip_dirs(root, dataset=args.dataset)

    if not trip_dirs:
        print(f"No phone_data.mat trips found under: {root}")
        sys.exit(1)

    if not args.all and args.trip is None:
        trip_dirs = trip_dirs[:1]

    any_error = False
    for trip_dir in trip_dirs:
        try:
            _print_result(validate_trip(trip_dir))
        except Exception as exc:
            any_error = True
            print(f"{trip_dir}: ERROR {exc}")

    if any_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
