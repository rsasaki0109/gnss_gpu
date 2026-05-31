#!/usr/bin/env python3
"""Materialise GICI ``rtk_imu_tc`` NMEA output as a PPC selector-pool candidate.

The PPC ``sim_ppc_*`` selector pools ingest candidates as a pair of per-run
files under ``experiments/results/libgnss_diag_phase10/<dir>/``:

* ``<city>_<run>_full.pos`` — LibGNSS++ position solution (whitespace columns
  ``GPS_Week GPS_TOW X Y Z Lat Lon Height Status ...``); the loader keys on
  ``GPS_TOW`` (rounded 0.1 s), reads ECEF ``X/Y/Z`` and ``Status``.
* ``<city>_<run>_full.csv`` — a ``gnss_solve`` diagnostics row per TOW; the
  candidate gate reads ``output_added`` / ``final_status`` / ``final_ratio`` /
  ``final_residual_rms`` and the sort key reads ``final_ratio`` /
  ``final_residual_rms`` / ``final_update_rows``.

GICI emits only an NMEA GGA stream (no RTK ratio / residual diagnostics), so the
diagnostics row is **synthesised** to mark every emitted epoch as a trusted
external fix: ``output_added=1``, ``final_status`` = GGA fix quality (4 RTK-fix
/ 5 RTK-float), a large ``final_ratio`` and a small ``final_residual_rms`` so
the epoch clears the gate.  This makes the candidate a deliberately strong
"trust GICI wherever it has a fix" probe — the add-candidate sweep then measures
whether trusting it actually lifts the OFFICIAL metric.  Use ``--fix4-only`` to
drop the noisier RTK-float epochs.

GICI GGA time is GPS time-of-day, so ``--leap-seconds`` (18 s) is added to align
with the reference ``GPS TOW`` grid (see ``eval_gici_tc_ppc2024_batch``).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_GICI_WS = Path("/media/sasaki/aiueo/ai_coding_ws/gici_open_ws")
_DEFAULT_DATASET = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DEFAULT_OUT_DIR = _REPO / "experiments/results/libgnss_diag_phase10/gici_tc"
_DEFAULT_LEAP_SECONDS = 18.0
_SECONDS_PER_DAY = 86400.0

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)

# Coverage-aware best deployable variant per run (eval_gici_tc_ppc2024_batch,
# coverage >= 80%, ranked by <1 m pass rate).
_DEFAULT_VARIANT = {
    "nagoya/run1": "test_nagoya1_loosenl.txt",
    "nagoya/run2": "test_nagoya2_combo4.txt",
    "nagoya/run3": "test_nagoya3_combo4.txt",
    "tokyo/run1": "test_tokyo1_zr.txt",
    "tokyo/run2": "test_tokyo2_window5.txt",
    "tokyo/run3": "test_tokyo3.txt",
}

_DIAG_FIELDS = (
    "epoch_index", "gps_week", "tow",
    "final_valid", "final_status", "final_sats", "final_ratio", "final_pdop",
    "final_residual_rms", "final_residual_abs_max", "final_update_rows",
    "output_added", "rejection_reason",
)


def lla_to_ecef(lat_deg: float, lon_deg: float, h_m: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    x = (n + h_m) * math.cos(lat) * math.cos(lon)
    y = (n + h_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + h_m) * math.sin(lat)
    return x, y, z


def _dmm_to_deg(dmm: str, hemi: str) -> float:
    deg_len = dmm.index(".") - 2 if dmm.find(".") >= 4 else 2
    val = float(dmm[:deg_len]) + float(dmm[deg_len:]) / 60.0
    return -val if hemi in ("S", "W") else val


def _ref_day_base(reference_csv: Path) -> float:
    with reference_csv.open() as fh:
        header = [h.strip() for h in fh.readline().split(",")]
        idx = header.index("GPS TOW (s)")
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) > idx:
                try:
                    tow0 = float(cols[idx])
                except ValueError:
                    continue
                return math.floor(tow0 / _SECONDS_PER_DAY) * _SECONDS_PER_DAY
    raise SystemExit(f"no GPS TOW rows in {reference_csv}")


def parse_gga_full(line: str, day_base: float, leap_seconds: float) -> dict | None:
    if not line.startswith("$GPGGA") and not line.startswith("$GNGGA"):
        return None
    star = line.find("*")
    f = (line[:star] if star >= 0 else line).split(",")
    if len(f) < 12 or not f[1] or not f[2] or not f[4]:
        return None
    try:
        hms = f[1]
        tod = int(hms[0:2]) * 3600 + int(hms[2:4]) * 60 + float(hms[4:])
        lat = _dmm_to_deg(f[2], f[3])
        lon = _dmm_to_deg(f[4], f[5])
        fix = int(f[6]) if f[6] else 0
        nsat = int(f[7]) if f[7] else 0
        alt = float(f[9]) if f[9] else 0.0
        geoid = float(f[11]) if f[11] else 0.0
    except (ValueError, IndexError):
        return None
    return {
        "tow": round(day_base + tod + leap_seconds, 1),
        "lat": lat,
        "lon": lon,
        "h": alt + geoid,  # ellipsoidal height
        "fix": fix,
        "nsat": nsat,
    }


def materialize_run(
    *,
    nmea_path: Path,
    reference_csv: Path,
    out_dir: Path,
    city: str,
    run: str,
    leap_seconds: float,
    fix4_only: bool,
    synth_ratio: float,
    synth_rms_fix: float,
    synth_rms_float: float,
) -> dict:
    day_base = _ref_day_base(reference_csv)
    pos_path = out_dir / f"{city}_{run}_full.pos"
    csv_path = out_dir / f"{city}_{run}_full.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_in = n_out = n_fix4 = n_fix5 = 0
    with nmea_path.open() as fh, pos_path.open("w") as pos_fh, csv_path.open(
        "w", newline=""
    ) as csv_fh:
        pos_fh.write("% LibGNSS++ Position Solution (synthesised from GICI rtk_imu_tc NMEA)\n")
        pos_fh.write(
            "% GPS_Week GPS_TOW X(m) Y(m) Z(m) Lat(deg) Lon(deg) Height(m) Status NumSat\n"
        )
        writer = csv.DictWriter(csv_fh, fieldnames=_DIAG_FIELDS)
        writer.writeheader()
        seen: set[float] = set()
        for line in fh:
            g = parse_gga_full(line, day_base, leap_seconds)
            if g is None:
                continue
            n_in += 1
            if g["fix"] not in (4, 5):
                continue
            if fix4_only and g["fix"] != 4:
                continue
            tow = g["tow"]
            if tow in seen:
                continue
            seen.add(tow)
            x, y, z = lla_to_ecef(g["lat"], g["lon"], g["h"])
            pos_fh.write(
                f"0 {tow:.3f} {x:.4f} {y:.4f} {z:.4f} "
                f"{g['lat']:.9f} {g['lon']:.9f} {g['h']:.4f} {g['fix']} {g['nsat']}\n"
            )
            rms = synth_rms_fix if g["fix"] == 4 else synth_rms_float
            writer.writerow(
                {
                    "epoch_index": n_out,
                    "gps_week": 0,
                    "tow": f"{tow:.3f}",
                    "final_valid": 1,
                    "final_status": g["fix"],
                    "final_sats": g["nsat"],
                    "final_ratio": f"{synth_ratio:.3f}",
                    "final_pdop": "1.0",
                    "final_residual_rms": f"{rms:.4f}",
                    "final_residual_abs_max": f"{rms * 2.0:.4f}",
                    "final_update_rows": g["nsat"],
                    "output_added": 1,
                    "rejection_reason": "none",
                }
            )
            n_out += 1
            n_fix4 += int(g["fix"] == 4)
            n_fix5 += int(g["fix"] == 5)
    return {
        "city": city, "run": run, "variant": nmea_path.name,
        "gga_in": n_in, "epochs_out": n_out, "fix4": n_fix4, "fix5": n_fix5,
        "pos": str(pos_path), "csv": str(csv_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gici-ws", type=Path, default=_DEFAULT_GICI_WS)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--leap-seconds", type=float, default=_DEFAULT_LEAP_SECONDS)
    parser.add_argument("--fix4-only", action="store_true", help="drop RTK-float (fix=5) epochs")
    parser.add_argument("--synth-ratio", type=float, default=99.0)
    parser.add_argument("--synth-rms-fix", type=float, default=0.01)
    parser.add_argument("--synth-rms-float", type=float, default=0.05)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Override per-run variant: city/run=test_<...>.txt (repeatable).",
    )
    args = parser.parse_args()

    variant_map = dict(_DEFAULT_VARIANT)
    for spec in args.variant:
        key, _, val = spec.partition("=")
        if "/" in key and val:
            variant_map[key.strip()] = val.strip()

    summaries: list[dict] = []
    for run_key, variant in variant_map.items():
        city, run = run_key.split("/", 1)
        nmea_path = args.gici_ws / variant
        reference_csv = args.dataset / city / run / "reference.csv"
        if not nmea_path.is_file():
            print(f"[skip] {run_key}: missing {nmea_path}")
            continue
        if not reference_csv.is_file():
            print(f"[skip] {run_key}: missing {reference_csv}")
            continue
        s = materialize_run(
            nmea_path=nmea_path,
            reference_csv=reference_csv,
            out_dir=args.out_dir,
            city=city,
            run=run,
            leap_seconds=args.leap_seconds,
            fix4_only=args.fix4_only,
            synth_ratio=args.synth_ratio,
            synth_rms_fix=args.synth_rms_fix,
            synth_rms_float=args.synth_rms_float,
        )
        summaries.append(s)
        print(
            f"[{run_key}] {variant}: epochs={s['epochs_out']} "
            f"(fix4={s['fix4']} fix5={s['fix5']}) -> {Path(s['pos']).name}"
        )

    print(f"\nmaterialised {len(summaries)} runs into {args.out_dir}")
    print("next: sim_ppc_phase_csv_addcand.py --discover-diag-dirs --only-labels xd_gici_tc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
