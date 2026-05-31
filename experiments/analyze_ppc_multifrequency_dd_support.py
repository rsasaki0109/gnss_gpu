#!/usr/bin/env python3
"""Diagnose multi-frequency DD carrier support on PPC runs.

This is a read-only probe for the carrier-anchor path.  It answers a narrow
question before changing the solver: do rover/base RINEX observations contain
enough common non-L1 carrier tracks, on satellites that also exist in the PPC
epoch geometry, to build a multi-frequency DD ambiguity model?
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in (_PROJECT_ROOT / "python", _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA_ROOT = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
RESULTS = _SCRIPT_DIR / "results"

_SYS_MAP = {0: "G", 1: "R", 2: "E", 3: "C", 4: "J"}


@dataclass(frozen=True)
class CarrierFamily:
    name: str
    codes_by_system: dict[str, tuple[str, ...]]


FAMILIES = (
    CarrierFamily(
        "L1_E1_B1",
        {
            "G": ("L1C", "L1W", "L1X", "L1P", "L1S", "L1L", "L1Z"),
            "E": ("L1X", "L1C", "L1A", "L1B", "L1Z"),
            "J": ("L1C", "L1X", "L1Z", "L1S", "L1L"),
            "C": ("L2I", "L1I", "L1P", "L1D", "L1X"),
        },
    ),
    CarrierFamily(
        "L2_E5b_B2",
        {
            "G": ("L2W", "L2X", "L2L", "L2C", "L2P"),
            "J": ("L2X", "L2L", "L2C"),
            "C": ("L7I", "L7D", "L7P", "L7X"),
            "E": ("L7X", "L7Q", "L7I"),
        },
    ),
    CarrierFamily(
        "L5_E5a_B2a",
        {
            "G": ("L5X", "L5Q", "L5I"),
            "E": ("L5X", "L5Q", "L5I"),
            "J": ("L5X", "L5Q", "L5I"),
            "C": ("L5P", "L5D", "L5X"),
        },
    ),
    CarrierFamily(
        "E5ab_B3",
        {
            "E": ("L8X", "L8Q", "L8I"),
            "C": ("L6I", "L6D", "L6P", "L6X"),
        },
    ),
)


def _datetime_to_tow(epoch_time: datetime) -> float:
    dow = epoch_time.weekday()
    gps_dow = (dow + 1) % 7
    sod = (
        epoch_time.hour * 3600
        + epoch_time.minute * 60
        + epoch_time.second
        + epoch_time.microsecond * 1e-6
    )
    return gps_dow * 86400.0 + sod


def _normalize_sat_id(sat_id: str) -> str:
    sat_id = sat_id.strip()
    if not sat_id:
        return sat_id
    sys_char = sat_id[0]
    prn_str = sat_id[1:].strip()
    if not prn_str:
        return sat_id
    try:
        return f"{sys_char}{int(prn_str):02d}"
    except ValueError:
        return sat_id


def _valid_carrier_codes(obs: dict[str, float]) -> set[str]:
    out: set[str] = set()
    for code, value in obs.items():
        if not code.startswith("L"):
            continue
        val = float(value)
        if np.isfinite(val) and abs(val) > 1e3:
            out.add(code)
    return out


def _looks_like_sat_id(text: str) -> bool:
    text = text.strip()
    return len(text) >= 2 and text[0].isalpha() and text[1:].strip().isdigit()


def _read_rinex_obs_type_header(lines: list[str]) -> tuple[dict[str, list[str]], int]:
    obs_types: dict[str, list[str]] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        label = line[60:].strip() if len(line) > 60 else ""
        if label.startswith("SYS / # / OBS TYPES"):
            sys_char = line[0].strip()
            n_types = int(line[3:6])
            obs_list = line[7:60].split()
            while len(obs_list) < n_types:
                idx += 1
                obs_list.extend(lines[idx][7:60].split())
            obs_types[sys_char] = obs_list[:n_types]
        elif label == "END OF HEADER":
            return obs_types, idx + 1
        idx += 1
    raise ValueError("RINEX header END OF HEADER not found")


def _load_rinex_carrier_codes_window(
    path: Path,
    systems: tuple[str, ...],
    start_tow: float,
    end_tow: float,
) -> dict[float, dict[str, set[str]]]:
    with path.open() as f:
        lines = f.readlines()
    obs_types, idx = _read_rinex_obs_type_header(lines)
    by_tow: dict[float, dict[str, set[str]]] = {}
    while idx < len(lines):
        line = lines[idx]
        if not line.startswith(">"):
            idx += 1
            continue
        parts = line[2:].split()
        if len(parts) < 8:
            idx += 1
            continue
        try:
            sec = float(parts[5])
            sec_int = int(sec)
            usec = int(round((sec - sec_int) * 1e6))
            epoch_time = datetime(
                int(parts[0]),
                int(parts[1]),
                int(parts[2]),
                int(parts[3]),
                int(parts[4]),
                sec_int,
                usec,
            )
            epoch_flag = int(parts[6])
            n_sat = int(parts[7])
        except (ValueError, IndexError):
            idx += 1
            continue
        tow = round(_datetime_to_tow(epoch_time), 1)
        if tow > end_tow:
            break
        if epoch_flag > 1:
            idx += 1 + n_sat
            continue
        idx += 1
        if tow < start_tow:
            # Skip satellite records quickly.  Continuation lines are rare in
            # these PPC files; the robust path below is used inside the window.
            idx += n_sat
            continue
        rows: dict[str, set[str]] = {}
        for _ in range(n_sat):
            if idx >= len(lines):
                break
            obs_line = lines[idx]
            sat_id = _normalize_sat_id(obs_line[:3])
            sys_char = sat_id[0] if sat_id else ""
            obs_codes = obs_types.get(sys_char, [])
            obs_record = obs_line.rstrip("\n")
            target_len = 3 + 16 * len(obs_codes)
            while len(obs_record) < target_len and idx + 1 < len(lines):
                next_line = lines[idx + 1]
                next_id = next_line[:3].strip()
                if next_line.startswith(">") or _looks_like_sat_id(next_id):
                    break
                idx += 1
                obs_record += lines[idx][3:].rstrip("\n")
            idx += 1

            if not sat_id or sys_char not in systems:
                continue
            sat_obs: dict[str, float] = {}
            pos = 3
            for obs_code in obs_codes:
                val_str = obs_record[pos : pos + 14].strip() if pos + 14 <= len(obs_record) else ""
                try:
                    sat_obs[obs_code] = float(val_str) if val_str else 0.0
                except ValueError:
                    sat_obs[obs_code] = 0.0
                pos += 16
            codes = _valid_carrier_codes(sat_obs)
            if codes:
                rows[sat_id] = codes
        if rows:
            by_tow[tow] = rows
    return by_tow


def _interpolate_code_sets(
    by_tow: dict[float, dict[str, set[str]]],
    keys: np.ndarray,
    tow: float,
    max_gap_s: float,
) -> dict[str, set[str]] | None:
    idx = int(np.searchsorted(keys, tow))
    if idx <= 0 or idx >= keys.size:
        return None
    t0 = float(keys[idx - 1])
    t1 = float(keys[idx])
    if not (t0 < tow < t1) or (t1 - t0) > max_gap_s:
        return None
    obs0 = by_tow.get(t0)
    obs1 = by_tow.get(t1)
    if obs0 is None or obs1 is None:
        return None
    out: dict[str, set[str]] = {}
    for sat_id in set(obs0) & set(obs1):
        codes = obs0[sat_id] & obs1[sat_id]
        if codes:
            out[sat_id] = codes
    return out or None


def _family_codes(codes: Iterable[str], family: CarrierFamily, sys_char: str) -> set[str]:
    allowed = set(family.codes_by_system.get(sys_char, ()))
    return set(codes) & allowed


def _compatible_sats(
    *,
    sats: set[str],
    rover_obs: dict[str, set[str]],
    base_obs: dict[str, set[str]],
    family: CarrierFamily,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sat_id in sats:
        sys_char = sat_id[0]
        if sys_char not in family.codes_by_system:
            continue
        rover_codes = _family_codes(rover_obs.get(sat_id, set()), family, sys_char)
        base_codes = _family_codes(base_obs.get(sat_id, set()), family, sys_char)
        if rover_codes and base_codes:
            counts[sys_char] = counts.get(sys_char, 0) + 1
    return counts


def _dd_pairs_from_counts(counts: dict[str, int]) -> int:
    return int(sum(max(0, n - 1) for n in counts.values()))


def _available_sat_ids(
    *,
    system_ids: np.ndarray,
    used_prns: list,
    epoch_index: int,
    systems: tuple[str, ...],
) -> set[str]:
    sats: set[str] = set()
    for sid, prn in zip(np.asarray(system_ids[epoch_index], dtype=np.int32), used_prns[epoch_index]):
        sys_char = _SYS_MAP.get(int(sid))
        if sys_char is None or sys_char not in systems:
            continue
        try:
            sats.add(f"{sys_char}{int(prn):02d}")
        except (TypeError, ValueError):
            continue
    return sats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="nagoya/run2")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--start-tow", type=float, default=557046.2)
    parser.add_argument("--end-tow", type=float, default=557051.6)
    parser.add_argument("--max-epochs", type=int, default=12000)
    parser.add_argument("--systems", default="G,E,J,C")
    parser.add_argument("--base-interp", action="store_true")
    parser.add_argument("--base-interp-max-gap-s", type=float, default=1.5)
    parser.add_argument(
        "--use-ppc-geometry",
        action="store_true",
        help="Restrict counts to satellites present in PPC epoch geometry. Slower.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=RESULTS / "nr2_multifrequency_dd_support_summary.csv",
    )
    parser.add_argument(
        "--out-epochs",
        type=Path,
        default=RESULTS / "nr2_multifrequency_dd_support_epochs.csv",
    )
    args = parser.parse_args()

    city, run = str(args.run).split("/", 1)
    run_dir = args.data_root / city / run
    systems = tuple(s.strip() for s in str(args.systems).split(",") if s.strip())

    rover_by_tow = _load_rinex_carrier_codes_window(
        run_dir / "rover.obs",
        systems,
        start_tow=float(args.start_tow) - 0.1,
        end_tow=float(args.end_tow) + 0.1,
    )
    base_by_tow = _load_rinex_carrier_codes_window(
        run_dir / "base.obs",
        systems,
        start_tow=float(args.start_tow) - float(args.base_interp_max_gap_s),
        end_tow=float(args.end_tow) + float(args.base_interp_max_gap_s),
    )
    base_keys = np.array(sorted(base_by_tow), dtype=np.float64)

    if args.use_ppc_geometry:
        from gnss_gpu.io.ppc import PPCDatasetLoader

        data = PPCDatasetLoader(run_dir).load_experiment_data(
            max_epochs=int(args.max_epochs),
            include_sat_velocity=False,
            systems=systems,
        )
        times = np.asarray(data["times"], dtype=np.float64)
        system_ids = data["system_ids"]
        used_prns = data["used_prns"]
        epoch_items = [
            (
                i,
                round(float(tow), 1),
                _available_sat_ids(
                    system_ids=system_ids,
                    used_prns=used_prns,
                    epoch_index=i,
                    systems=systems,
                ),
            )
            for i, tow in enumerate(times)
            if float(args.start_tow) <= round(float(tow), 1) <= float(args.end_tow)
        ]
    else:
        epoch_items = [
            (-1, tow, set(rows.keys()))
            for tow, rows in sorted(rover_by_tow.items())
            if float(args.start_tow) <= tow <= float(args.end_tow)
        ]

    epoch_rows: list[dict[str, object]] = []
    for _i, tow_r, sats in epoch_items:
        rover_obs = rover_by_tow.get(tow_r)
        base_obs = base_by_tow.get(tow_r)
        base_exact = base_obs is not None
        if base_obs is None and args.base_interp:
            base_obs = _interpolate_code_sets(
                base_by_tow,
                base_keys,
                tow_r,
                max_gap_s=float(args.base_interp_max_gap_s),
            )
        row: dict[str, object] = {
            "run": str(args.run),
            "tow": tow_r,
            "use_ppc_geometry": int(bool(args.use_ppc_geometry)),
            "base_exact": int(base_exact),
            "has_rover_obs": int(rover_obs is not None),
            "has_base_obs": int(base_obs is not None),
            "geometry_sats": len(sats),
        }
        if rover_obs is None or base_obs is None:
            for family in FAMILIES:
                row[f"{family.name}_sats"] = 0
                row[f"{family.name}_dd_pairs"] = 0
                row[f"{family.name}_systems"] = ""
            row["dual_L1_L5_sats"] = 0
            row["dual_L1_L5_dd_pairs"] = 0
            epoch_rows.append(row)
            continue

        per_family_counts: dict[str, dict[str, int]] = {}
        for family in FAMILIES:
            counts = _compatible_sats(
                sats=sats,
                rover_obs=rover_obs,
                base_obs=base_obs,
                family=family,
            )
            per_family_counts[family.name] = counts
            row[f"{family.name}_sats"] = int(sum(counts.values()))
            row[f"{family.name}_dd_pairs"] = _dd_pairs_from_counts(counts)
            row[f"{family.name}_systems"] = ",".join(
                f"{sys_char}:{counts[sys_char]}" for sys_char in sorted(counts)
            )

        dual_counts = {
            sys_char: min(
                per_family_counts["L1_E1_B1"].get(sys_char, 0),
                per_family_counts["L5_E5a_B2a"].get(sys_char, 0),
            )
            for sys_char in systems
        }
        dual_counts = {k: v for k, v in dual_counts.items() if v > 0}
        row["dual_L1_L5_sats"] = int(sum(dual_counts.values()))
        row["dual_L1_L5_dd_pairs"] = _dd_pairs_from_counts(dual_counts)
        row["dual_L1_L5_systems"] = ",".join(
            f"{sys_char}:{dual_counts[sys_char]}" for sys_char in sorted(dual_counts)
        )
        epoch_rows.append(row)

    if not epoch_rows:
        raise SystemExit("no epochs matched requested TOW window")

    args.out_epochs.parent.mkdir(parents=True, exist_ok=True)
    with args.out_epochs.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    summary_rows: list[dict[str, object]] = []
    for key in [
        *(f"{family.name}_dd_pairs" for family in FAMILIES),
        "dual_L1_L5_dd_pairs",
    ]:
        vals = np.array([float(r[key]) for r in epoch_rows], dtype=np.float64)
        summary_rows.append(
            {
                "run": str(args.run),
                "start_tow": float(args.start_tow),
                "end_tow": float(args.end_tow),
                "systems": ",".join(systems),
                "base_interp": int(bool(args.base_interp)),
                "metric": key,
                "epochs": len(epoch_rows),
                "support_ge_1": int(np.count_nonzero(vals >= 1)),
                "support_ge_3": int(np.count_nonzero(vals >= 3)),
                "support_ge_5": int(np.count_nonzero(vals >= 5)),
                "median": float(np.median(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
                "max": float(np.max(vals)),
            }
        )

    with args.out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"rows={len(epoch_rows)}")
    for row in summary_rows:
        print(
            f"{row['metric']}: ge3={row['support_ge_3']}/{row['epochs']} "
            f"median={row['median']:.1f} p90={row['p90']:.1f} max={row['max']:.1f}"
        )
    print(f"wrote {args.out_summary}")
    print(f"wrote {args.out_epochs}")


if __name__ == "__main__":
    main()
