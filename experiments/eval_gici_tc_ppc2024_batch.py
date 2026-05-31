#!/usr/bin/env python3
"""Batch-evaluate GICI ``rtk_imu_tc`` NMEA outputs against PPC2024 references.

The GICI ``forppc2024`` post-file estimator writes a GPGGA stream whose time
field is **GPS time-of-day**, not UTC, so it sits ``leap_seconds`` (18 s as of
the PPC2024 epoch) ahead of the reference ``GPS TOW``.  The decisive empirical
check (see ``project-gici-tc-nagoya1-96pct``) is that applying ``+18 s`` to the
GICI timestamp collapses nagoya/run1 horizontal error from p50 21.8 m to
0.12 m (<1 m pass rate 96.75 %).

This tool generalises the one-off nagoya/run1 evaluator to every
``test_<run>*.txt`` variant under the GICI workspace, matches each to its run's
``reference.csv`` (run-agnostic GPS day base derived from the reference), and
ranks the variants per run by <1 m pass rate.  Output feeds the PPC selector
pool candidate-injection decision.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

_DEFAULT_GICI_WS = Path("/media/sasaki/aiueo/ai_coding_ws/gici_open_ws")
_DEFAULT_DATASET = Path("/media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data")
_DEFAULT_LEAP_SECONDS = 18.0
_SECONDS_PER_DAY = 86400.0

_RUN_RE = re.compile(r"test_(nagoya|tokyo)([123])")


def run_key_from_filename(name: str) -> str | None:
    """Map ``test_nagoya1_combo.txt`` -> ``nagoya/run1`` (None if no match)."""
    m = _RUN_RE.match(name)
    if m is None:
        return None
    return f"{m.group(1)}/run{m.group(2)}"


def dmm_to_deg(dmm: str, hemi: str) -> float:
    if not dmm:
        return float("nan")
    deg_len = dmm.index(".") - 2 if dmm.find(".") >= 4 else 2
    deg = float(dmm[:deg_len])
    minutes = float(dmm[deg_len:])
    val = deg + minutes / 60.0
    return -val if hemi in ("S", "W") else val


def parse_gga(line: str, day_base: float) -> dict | None:
    if not line.startswith("$GPGGA") and not line.startswith("$GNGGA"):
        return None
    star = line.find("*")
    body = line[:star] if star >= 0 else line
    f = body.split(",")
    if len(f) < 10 or not f[1] or not f[2] or not f[4]:
        return None
    try:
        hms = f[1]
        tod = int(hms[0:2]) * 3600 + int(hms[2:4]) * 60 + float(hms[4:])
        lat = dmm_to_deg(f[2], f[3])
        lon = dmm_to_deg(f[4], f[5])
        fix = int(f[6]) if f[6] else 0
    except (ValueError, IndexError):
        return None
    return {"tow": day_base + tod, "lat": lat, "lon": lon, "fix": fix}


def load_reference(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        header = [h.strip() for h in fh.readline().split(",")]
        idx = {h: i for i, h in enumerate(header)}
        for line in fh:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < len(header):
                continue
            try:
                rows.append(
                    {
                        "tow": float(cols[idx["GPS TOW (s)"]]),
                        "lat": float(cols[idx["Latitude (deg)"]]),
                        "lon": float(cols[idx["Longitude (deg)"]]),
                    }
                )
            except (ValueError, KeyError):
                pass
    return rows


def interp_ref(ref: list[dict], tow: float) -> dict | None:
    if not ref or tow < ref[0]["tow"] or tow > ref[-1]["tow"]:
        return None
    lo, hi = 0, len(ref) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ref[mid]["tow"] <= tow:
            lo = mid
        else:
            hi = mid
    t0, t1 = ref[lo]["tow"], ref[hi]["tow"]
    if t1 == t0:
        return ref[lo]
    a = (tow - t0) / (t1 - t0)
    return {
        "lat": ref[lo]["lat"] + a * (ref[hi]["lat"] - ref[lo]["lat"]),
        "lon": ref[lo]["lon"] + a * (ref[hi]["lon"] - ref[lo]["lon"]),
    }


def horizontal_error_m(lat_e: float, lon_e: float, lat_r: float, lon_r: float) -> float:
    R = 6378137.0
    lat0 = math.radians(lat_r)
    n = R * math.radians(lat_e - lat_r)
    e = R * math.cos(lat0) * math.radians(lon_e - lon_r)
    return math.hypot(n, e)


def _pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[int(round(q * (len(sorted_vals) - 1)))]


def evaluate_file(nmea_path: Path, ref: list[dict], leap_seconds: float) -> dict:
    ref_rows = len(ref)
    day_base = math.floor(ref[0]["tow"] / _SECONDS_PER_DAY) * _SECONDS_PER_DAY
    errs: list[float] = []
    fix4: list[float] = []
    n_gga = 0
    with nmea_path.open() as fh:
        for line in fh:
            g = parse_gga(line, day_base)
            if g is None:
                continue
            n_gga += 1
            r = interp_ref(ref, g["tow"] + leap_seconds)
            if r is None:
                continue
            herr = horizontal_error_m(g["lat"], g["lon"], r["lat"], r["lon"])
            errs.append(herr)
            if g["fix"] == 4:
                fix4.append(herr)
    errs.sort()
    fix4.sort()
    n = len(errs)
    return {
        "variant": nmea_path.name,
        "n_gga": n_gga,
        "n_eval": n,
        "ref_rows": ref_rows,
        "coverage_pct": (n / ref_rows * 100) if ref_rows else 0.0,
        "mean_m": (sum(errs) / n) if n else float("nan"),
        "p50_m": _pct(errs, 0.50),
        "p95_m": _pct(errs, 0.95),
        "max_m": errs[-1] if errs else float("nan"),
        "pass_lt1m_pct": (sum(v < 1.0 for v in errs) / n * 100) if n else 0.0,
        "pass_lt2m_pct": (sum(v < 2.0 for v in errs) / n * 100) if n else 0.0,
        "fix4_count": len(fix4),
        "fix4_pass_lt1m_pct": (sum(v < 1.0 for v in fix4) / len(fix4) * 100) if fix4 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gici-ws", type=Path, default=_DEFAULT_GICI_WS)
    parser.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    parser.add_argument("--leap-seconds", type=float, default=_DEFAULT_LEAP_SECONDS)
    parser.add_argument("--glob", default="test_*.txt")
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--top", type=int, default=3, help="best variants per run to print")
    parser.add_argument(
        "--min-coverage-pct",
        type=float,
        default=80.0,
        help="coverage floor for the 'best deployable' per-run pick",
    )
    args = parser.parse_args()

    files = sorted(args.gici_ws.glob(args.glob))
    if not files:
        raise SystemExit(f"no GICI outputs match {args.gici_ws / args.glob}")

    ref_cache: dict[str, list[dict]] = {}
    rows: list[dict] = []
    for path in files:
        run = run_key_from_filename(path.name)
        if run is None:
            continue
        if run not in ref_cache:
            ref_path = args.dataset / run / "reference.csv"
            if not ref_path.is_file():
                continue
            ref_cache[run] = load_reference(ref_path)
        result = evaluate_file(path, ref_cache[run], args.leap_seconds)
        result["run"] = run
        rows.append(result)

    rows.sort(key=lambda r: (r["run"], -r["pass_lt1m_pct"]))

    print(f"# {len(rows)} variants over {len(ref_cache)} runs  (leap=+{args.leap_seconds:g}s)\n")
    print(f"## Top {args.top} variants per run (by <1 m pass rate)")
    hdr = f"{'run':<12} {'variant':<34} {'n':>5} {'cov':>6} {'p50':>7} {'p95':>7} {'<1m':>7} {'fix4<1m':>8}"
    print(hdr)
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    def _fmt(r: dict) -> str:
        return (
            f"{r['run']:<12} {r['variant']:<34} {r['n_eval']:>5} {r['coverage_pct']:>5.1f}% "
            f"{r['p50_m']:>7.3f} {r['p95_m']:>7.3f} {r['pass_lt1m_pct']:>6.2f}% "
            f"{r['fix4_pass_lt1m_pct']:>7.2f}%"
        )

    for run in sorted(by_run):
        for r in by_run[run][: args.top]:
            print(_fmt(r))
        print()

    print(f"## Best deployable per run (coverage >= {args.min_coverage_pct:g}%, then <1 m)")
    print(hdr)
    for run in sorted(by_run):
        eligible = [r for r in by_run[run] if r["coverage_pct"] >= args.min_coverage_pct]
        if eligible:
            print(_fmt(max(eligible, key=lambda r: r["pass_lt1m_pct"])))
        else:
            best_cov = max(by_run[run], key=lambda r: r["coverage_pct"])
            print(f"{run:<12} (none >= {args.min_coverage_pct:g}% coverage; best cov={best_cov['coverage_pct']:.1f}%)")
    print()

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "run", "variant", "n_gga", "n_eval", "ref_rows", "coverage_pct",
            "mean_m", "p50_m", "p95_m", "max_m",
            "pass_lt1m_pct", "pass_lt2m_pct", "fix4_count", "fix4_pass_lt1m_pct",
        ]
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fields})
        print(f"wrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
