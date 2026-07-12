#!/usr/bin/env python3
"""Score a trajectory against PPC ground truth with inuex35 and official PPC metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from gnss_gpu.ppc_score import score_ppc2024  # noqa: E402

_DATA_ROOT = Path("datasets/PPC-Dataset-data")
_TOW_TOLERANCE_S = 0.05

# Rover-epoch denominators from inuex35 README (full-run coverage baseline).
_ROVER_EPOCH_COUNTS: dict[tuple[str, str], int] = {
    ("tokyo", "run1"): 11928,
    ("tokyo", "run2"): 9151,
    ("tokyo", "run3"): 15301,
    ("nagoya", "run1"): 11928,
    ("nagoya", "run2"): 9151,
    ("nagoya", "run3"): 15301,
}


@dataclass(frozen=True)
class TrajectoryEpoch:
    tow: float
    ecef: np.ndarray
    is_fix: bool


@dataclass(frozen=True)
class ScoreResult:
    city: str
    run: str
    traj_path: str
    format: str
    n_scored: int
    n_rover_epochs: int
    coverage_pct: float
    n_fix: int
    all_rms_m: float
    fix_rms_m: float | None
    fix_pct: float
    lt50cm_pct: float
    lt50cm_full_pct: float
    ppc_official_pct: float
    ppc_note: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.fix_rms_m is None:
            payload["fix_rms_m"] = None
        return payload


def load_reference_grid(city: str, run: str, data_root: Path = _DATA_ROOT) -> dict[float, np.ndarray]:
    """Load reference ECEF indexed by rounded 5 Hz GPS TOW."""
    path = data_root / city / run / "reference.csv"
    out: dict[float, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                tow = round(float(row["GPS TOW (s)"]), 1)
                xyz = np.array(
                    [
                        float(row["ECEF X (m)"]),
                        float(row["ECEF Y (m)"]),
                        float(row["ECEF Z (m)"]),
                    ],
                    dtype=np.float64,
                )
            except (KeyError, ValueError):
                continue
            out[tow] = xyz
    if not out:
        raise FileNotFoundError(f"no reference epochs in {path}")
    return out


def _lookup_reference_tow(tow: float, ref_tows: np.ndarray) -> float | None:
    """Return nearest reference TOW within tolerance, else None."""
    rounded = round(tow, 1)
    idx = int(np.searchsorted(ref_tows, rounded))
    best: float | None = None
    best_dt = _TOW_TOLERANCE_S + 1.0
    for candidate_idx in (idx - 1, idx):
        if 0 <= candidate_idx < ref_tows.size:
            candidate = float(ref_tows[candidate_idx])
            dt = abs(candidate - rounded)
            if dt <= _TOW_TOLERANCE_S and dt < best_dt:
                best_dt = dt
                best = candidate
    return best


def _pos_quality_flag(parts: list[str]) -> int:
    """Pick the solution-quality column across RTKLIB and libgnss variants."""
    if len(parts) >= 10:
        try:
            quality = int(float(parts[8]))
            if quality in {1, 2, 3, 4, 5, 6}:
                return quality
        except ValueError:
            pass
    if len(parts) >= 6:
        return int(float(parts[5]))
    return 0


def _parse_fix_statuses(text: str) -> frozenset[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if token:
            values.add(int(token))
    if not values:
        raise ValueError("fix-statuses must list at least one integer")
    return frozenset(values)


def load_pos_trajectory(
    path: Path,
    *,
    fix_statuses: frozenset[int] = frozenset({4}),
) -> list[TrajectoryEpoch]:
    """Libgnss++/RTKLIB .pos: tow at parts[1], ECEF at parts[2:5], Status at index 8.

    Repo libgnss++ artifacts use Status==4 for RTK FIXED (default fix_statuses).
    RTKLIB Q-convention files need fix_statuses={1} explicitly.
    """
    epochs: list[TrajectoryEpoch] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("%") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                tow = float(parts[1])
                ecef = np.array(
                    [float(parts[2]), float(parts[3]), float(parts[4])],
                    dtype=np.float64,
                )
                quality = _pos_quality_flag(parts)
            except ValueError:
                continue
            epochs.append(
                TrajectoryEpoch(
                    tow=tow,
                    ecef=ecef,
                    is_fix=quality in fix_statuses,
                )
            )
    return epochs


def load_csv_trajectory(path: Path) -> list[TrajectoryEpoch]:
    """Simple CSV with header tow,ecef_x,ecef_y,ecef_z,fix."""
    epochs: list[TrajectoryEpoch] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                tow = float(row["tow"])
                ecef = np.array(
                    [float(row["ecef_x"]), float(row["ecef_y"]), float(row["ecef_z"])],
                    dtype=np.float64,
                )
                fix_raw = row["fix"].strip().lower()
                is_fix = fix_raw in {"1", "true", "t", "yes", "y"}
            except (KeyError, ValueError):
                continue
            epochs.append(TrajectoryEpoch(tow=tow, ecef=ecef, is_fix=is_fix))
    return epochs


def load_npz_trajectory(path: Path) -> list[TrajectoryEpoch]:
    """inuex35 SAVE_NPZ layout: sol_xyz (n,3), smode (n,), smode==4 is FIX."""
    payload = np.load(path)
    sol_xyz = np.asarray(payload["sol_xyz"], dtype=np.float64)
    if sol_xyz.ndim != 2 or sol_xyz.shape[1] < 3:
        raise ValueError("npz sol_xyz must have shape (n, 3+)")
    smode = np.asarray(payload["smode"]).reshape(-1)
    if "tow" in payload:
        tows = np.asarray(payload["tow"], dtype=np.float64).reshape(-1)
    elif "gps_tow" in payload:
        tows = np.asarray(payload["gps_tow"], dtype=np.float64).reshape(-1)
    else:
        raise ValueError("npz must include tow or gps_tow array")
    if tows.size != sol_xyz.shape[0] or smode.size != sol_xyz.shape[0]:
        raise ValueError("npz tow/sol_xyz/smode length mismatch")
    epochs: list[TrajectoryEpoch] = []
    for tow, xyz, mode in zip(tows, sol_xyz, smode, strict=True):
        epochs.append(
            TrajectoryEpoch(
                tow=float(tow),
                ecef=xyz[:3].copy(),
                is_fix=int(mode) == 4,
            )
        )
    return epochs


def load_trajectory(
    path: Path,
    fmt: str,
    *,
    fix_statuses: frozenset[int] = frozenset({4}),
) -> list[TrajectoryEpoch]:
    if fmt == "pos":
        return load_pos_trajectory(path, fix_statuses=fix_statuses)
    if fmt == "csv":
        return load_csv_trajectory(path)
    if fmt == "npz":
        return load_npz_trajectory(path)
    raise ValueError(f"unsupported format: {fmt}")


def rover_epoch_count(city: str, run: str, reference: dict[float, np.ndarray]) -> int:
    key = (city, run)
    if key in _ROVER_EPOCH_COUNTS:
        return _ROVER_EPOCH_COUNTS[key]
    return len(reference)


def score_trajectory(
    epochs: list[TrajectoryEpoch],
    reference: dict[float, np.ndarray],
    *,
    city: str,
    run: str,
    traj_path: Path,
    fmt: str,
) -> ScoreResult:
    ref_tows = np.asarray(sorted(reference), dtype=np.float64)
    est_list: list[np.ndarray] = []
    ref_list: list[np.ndarray] = []
    err_list: list[float] = []
    fix_flags: list[bool] = []

    for epoch in epochs:
        ref_tow = _lookup_reference_tow(epoch.tow, ref_tows)
        if ref_tow is None:
            continue
        ref_xyz = reference[ref_tow]
        if not np.all(np.isfinite(epoch.ecef)) or not np.all(np.isfinite(ref_xyz)):
            continue
        err3d = float(np.linalg.norm(epoch.ecef - ref_xyz))
        est_list.append(epoch.ecef)
        ref_list.append(ref_xyz)
        err_list.append(err3d)
        fix_flags.append(epoch.is_fix)

    n_scored = len(err_list)
    n_rover = rover_epoch_count(city, run, reference)
    coverage_pct = 100.0 * n_scored / n_rover if n_rover else 0.0

    if n_scored == 0:
        return ScoreResult(
            city=city,
            run=run,
            traj_path=str(traj_path),
            format=fmt,
            n_scored=0,
            n_rover_epochs=n_rover,
            coverage_pct=0.0,
            n_fix=0,
            all_rms_m=float("nan"),
            fix_rms_m=None,
            fix_pct=0.0,
            lt50cm_pct=0.0,
            lt50cm_full_pct=0.0,
            ppc_official_pct=float("nan"),
            ppc_note="no scored epochs",
        )

    err_arr = np.asarray(err_list, dtype=np.float64)
    all_rms = float(np.sqrt(np.mean(err_arr**2)))
    n_fix = int(sum(fix_flags))
    fix_pct = 100.0 * n_fix / n_scored
    n_lt50cm = int(np.sum(err_arr < 0.5))
    lt50cm_pct = 100.0 * n_lt50cm / n_scored
    lt50cm_full_pct = 100.0 * n_lt50cm / n_rover if n_rover else 0.0

    fix_err = err_arr[np.asarray(fix_flags, dtype=bool)]
    fix_rms = float(np.sqrt(np.mean(fix_err**2))) if fix_err.size else None

    ppc_note: str | None = None
    try:
        est_arr = np.asarray(est_list, dtype=np.float64)
        ref_arr = np.asarray(ref_list, dtype=np.float64)
        ppc = score_ppc2024(est_arr, ref_arr)
        ppc_pct = float(ppc.score_pct)
    except Exception as exc:  # noqa: BLE001
        ppc_pct = float("nan")
        ppc_note = str(exc)

    return ScoreResult(
        city=city,
        run=run,
        traj_path=str(traj_path),
        format=fmt,
        n_scored=n_scored,
        n_rover_epochs=n_rover,
        coverage_pct=coverage_pct,
        n_fix=n_fix,
        all_rms_m=all_rms,
        fix_rms_m=fix_rms,
        fix_pct=fix_pct,
        lt50cm_pct=lt50cm_pct,
        lt50cm_full_pct=lt50cm_full_pct,
        ppc_official_pct=ppc_pct,
        ppc_note=ppc_note,
    )


def format_summary(result: ScoreResult) -> str:
    fix_rms = "n/a" if result.fix_rms_m is None else f"{result.fix_rms_m:.3f}"
    ppc = "n/a" if not np.isfinite(result.ppc_official_pct) else f"{result.ppc_official_pct:.2f}"
    return (
        f"{result.city} {result.run}: n_scored={result.n_scored}/"
        f"{result.n_rover_epochs} coverage={result.coverage_pct:.1f}% "
        f"AllRMS={result.all_rms_m:.3f} (scored epochs only) "
        f"FixRMS={fix_rms} fix%={result.fix_pct:.1f} "
        f"<50cm%={result.lt50cm_pct:.1f} <50cm_full%={result.lt50cm_full_pct:.1f} "
        f"ppc={ppc}%"
    )


def write_csv_row(path: Path, result: ScoreResult) -> None:
    header = [
        "method",
        "city",
        "run",
        "n_scored",
        "n_rover_epochs",
        "coverage_pct",
        "all_rms_m",
        "fix_rms_m",
        "fix_pct",
        "lt50cm_pct",
        "lt50cm_full_pct",
        "ppc_official_pct",
        "traj_path",
    ]
    row = {
        "method": path.stem,
        "city": result.city,
        "run": result.run,
        "n_scored": result.n_scored,
        "n_rover_epochs": result.n_rover_epochs,
        "coverage_pct": f"{result.coverage_pct:.3f}",
        "all_rms_m": f"{result.all_rms_m:.6f}",
        "fix_rms_m": "" if result.fix_rms_m is None else f"{result.fix_rms_m:.6f}",
        "fix_pct": f"{result.fix_pct:.3f}",
        "lt50cm_pct": f"{result.lt50cm_pct:.3f}",
        "lt50cm_full_pct": f"{result.lt50cm_full_pct:.3f}",
        "ppc_official_pct": (
            "" if not np.isfinite(result.ppc_official_pct) else f"{result.ppc_official_pct:.3f}"
        ),
        "traj_path": result.traj_path,
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, required=True, help="trajectory file path")
    parser.add_argument("--city", default="tokyo", help="PPC city (tokyo|nagoya)")
    parser.add_argument("--run", default="run1", help="PPC run (run1|run2|run3)")
    parser.add_argument(
        "--format",
        choices=("pos", "csv", "npz"),
        default="pos",
        help="trajectory format",
    )
    parser.add_argument("--data-root", type=Path, default=_DATA_ROOT)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--fix-statuses",
        default=None,
        help="comma-separated .pos Status values counted as FIX (default for pos: 4; "
        "use 1 for RTKLIB Q-convention files)",
    )
    args = parser.parse_args()

    fix_statuses = (
        _parse_fix_statuses(args.fix_statuses)
        if args.fix_statuses is not None
        else (frozenset({4}) if args.format == "pos" else frozenset())
    )

    reference = load_reference_grid(args.city, args.run, data_root=args.data_root)
    epochs = load_trajectory(args.traj, args.format, fix_statuses=fix_statuses)
    result = score_trajectory(
        epochs,
        reference,
        city=args.city,
        run=args.run,
        traj_path=args.traj,
        fmt=args.format,
    )

    print(format_summary(result))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="utf-8")

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv_row(args.out_csv, result)


if __name__ == "__main__":
    main()
