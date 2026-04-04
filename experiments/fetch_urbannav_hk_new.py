#!/usr/bin/env python3
"""Fetch UrbanNav Hong Kong 2021 sequences and normalize for this repo.

These sequences (from IPNL-POLYU/UrbanNavDataset) differ from the 2019 pilot:
- GNSS RINEX is in a Dropbox shared folder (downloaded as ZIP)
- Ground truth is a space-delimited DMS text file (not CSV)
- Base station and navigation files are NOT bundled

Base station RINEX must be obtained separately from Hong Kong SatRef:
  https://www.geodetic.gov.hk/en/rinex/downv.aspx
Download HKSC station data for the collection date and place it as base_hksc.obs
in the output run directory.

Navigation ephemeris can be downloaded from BKG or IGS MGEX.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


SEQUENCES = {
    "tst": {
        "run_name": "HK_TST",
        "gnss_url": (
            "https://www.dropbox.com/sh/2haoy68xekg95zl/"
            "AAAkcN4FwhFxkPY1lXsxbJrxa?dl=1"
        ),
        "gt_url": (
            "https://www.dropbox.com/s/twsvwftucoytfpc/"
            "UrbanNav_TST_GT_raw.txt?dl=1"
        ),
        "date": "2021-05-17",
        "doy": 137,
        "rover_pattern": "f9p.splitter",
    },
    "whampoa": {
        "run_name": "HK_Whampoa",
        "gnss_url": (
            "https://www.dropbox.com/sh/7ox7718bzcjqtlf/"
            "AABH_Kjm65gHQ09K3antBRdua?dl=1"
        ),
        "gt_url": (
            "https://www.dropbox.com/s/ej2mkue2w3r36s2/"
            "UrbanNav_whampoa_raw.txt?dl=1"
        ),
        "date": "2021-05-21",
        "doy": 141,
        "rover_pattern": "f9p.splitter",
    },
}

# HKSC reference station approximate ECEF (WGS84), from prior RINEX headers.
HKSC_ECEF = (-2414266.9197, 5386768.9868, 2407460.0314)

# BKG mixed GNSS broadcast ephemeris (freely accessible, RINEX 3 multi-GNSS).
BKG_NAV_URL = (
    "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/"
    "BRDC00WRD_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
)
# Fallback: GPS-only RINEX 2.
BKG_NAV_URL_GPS = (
    "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{year}/{doy:03d}/"
    "brdc{doy:03d}0.{yy}n.gz"
)


def _download(url: str, path: Path, timeout: int = 300) -> None:
    print(f"  downloading {url[:80]}...")
    req = urllib.request.Request(url, headers={"User-Agent": "gnss_gpu/fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _find_rover_obs(names: list[str], pattern: str) -> str:
    candidates = sorted(
        n for n in names
        if n.lower().endswith(".obs") and pattern in n.lower()
    )
    if not candidates:
        all_obs = [n for n in names if n.lower().endswith(".obs")]
        if all_obs:
            return sorted(all_obs)[0]
        raise RuntimeError(f"no .obs file matching '{pattern}' in archive")
    return candidates[0]


def _find_nav_file(names: list[str]) -> str | None:
    for n in names:
        low = n.lower()
        if low.endswith((".nav", ".21n", ".21b", ".21p", ".rnx")) and "nav" in low:
            return n
    return None


def _dms_to_decimal(deg: float, minutes: float, seconds: float) -> float:
    sign = -1.0 if deg < 0 else 1.0
    return sign * (abs(deg) + minutes / 60.0 + seconds / 3600.0)


def _llh_to_ecef(lat_deg: float, lon_deg: float, alt: float) -> tuple[float, float, float]:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    x = (N + alt) * cos_lat * math.cos(lon)
    y = (N + alt) * cos_lat * math.sin(lon)
    z = (N * (1 - e2) + alt) * sin_lat
    return x, y, z


def _convert_gt_to_csv(gt_path: Path, csv_path: Path) -> int:
    """Convert SPAN-CPT DMS ground truth to ECEF CSV for UrbanNavLoader."""
    lines = gt_path.read_text().strip().splitlines()
    # Skip header lines (start with non-numeric characters).
    data_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
    if not data_lines:
        raise RuntimeError(f"no data lines in {gt_path}")

    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        gps_time = float(parts[2])
        lat_deg = _dms_to_decimal(float(parts[3]), float(parts[4]), float(parts[5]))
        lon_deg = _dms_to_decimal(float(parts[6]), float(parts[7]), float(parts[8]))
        alt = float(parts[9])
        x, y, z = _llh_to_ecef(lat_deg, lon_deg, alt)
        rows.append(f"{gps_time},{x:.6f},{y:.6f},{z:.6f}")

    with csv_path.open("w") as fh:
        fh.write("gps_time_s,ECEF X (m),ECEF Y (m),ECEF Z (m)\n")
        for row in rows:
            fh.write(row + "\n")
    return len(rows)


def _try_download_nav(year: int, doy: int, out_path: Path) -> bool:
    """Try to download broadcast nav from BKG (mixed GNSS first, GPS fallback)."""
    import gzip
    yy = year % 100
    urls = [
        BKG_NAV_URL.format(year=year, doy=doy),
        BKG_NAV_URL_GPS.format(year=year, doy=doy, yy=yy),
    ]
    for url in urls:
        try:
            gz_path = out_path.with_suffix(".rnx.gz")
            _download(url, gz_path, timeout=60)
            with gzip.open(gz_path, "rb") as gz, out_path.open("wb") as fh:
                shutil.copyfileobj(gz, fh)
            gz_path.unlink()
            return True
        except Exception:
            continue
    print("  warning: could not download nav from BKG")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch UrbanNav Hong Kong 2021 sequences"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        choices=tuple(SEQUENCES),
        required=True,
        help="Which HK sequence to fetch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for the normalized run directory",
    )
    args = parser.parse_args()

    seq = SEQUENCES[args.sequence]
    run_dir = args.output_dir / seq["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="urbannav_hk_new_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. Download GNSS RINEX (shared folder as ZIP).
        print(f"[1/3] Fetching GNSS RINEX for {args.sequence}...")
        gnss_zip = tmpdir_path / "gnss.zip"
        _download(seq["gnss_url"], gnss_zip)

        with zipfile.ZipFile(gnss_zip) as zf:
            names = zf.namelist()
            rover_name = _find_rover_obs(names, seq["rover_pattern"])
            print(f"  rover: {rover_name}")
            with zf.open(rover_name) as src, (run_dir / "rover_ublox.obs").open("wb") as dst:
                shutil.copyfileobj(src, dst)

            # Check if nav file is bundled (unlikely but worth checking).
            nav_name = _find_nav_file(names)
            if nav_name:
                print(f"  nav (bundled): {nav_name}")
                with zf.open(nav_name) as src, (run_dir / "base.nav").open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        # 2. Download and convert ground truth.
        print(f"[2/3] Fetching ground truth for {args.sequence}...")
        gt_raw = tmpdir_path / "gt_raw.txt"
        _download(seq["gt_url"], gt_raw)
        n_epochs = _convert_gt_to_csv(gt_raw, run_dir / "reference.csv")
        print(f"  converted {n_epochs} ground truth epochs to ECEF CSV")

        # 3. Try to download navigation ephemeris from BKG.
        print(f"[3/3] Fetching broadcast ephemeris...")
        year = int(seq["date"][:4])
        doy = seq["doy"]
        nav_path = run_dir / "base.nav"
        if not nav_path.exists():
            if not _try_download_nav(year, doy, nav_path):
                print(
                    f"  WARNING: no navigation file at {nav_path}\n"
                    f"  Download BRDC nav for {seq['date']} (DOY {doy}) manually from:\n"
                    f"    https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy:03d}/\n"
                    f"    or https://www.geodetic.gov.hk/en/rinex/downv.aspx"
                )

    # Check base station.
    base_obs = run_dir / "base_hksc.obs"
    if not base_obs.exists():
        # Write a minimal stub with HKSC position so the loader can read it.
        # The actual base obs data is not needed for standalone PF evaluation
        # since we use broadcast ephemeris directly.
        print(
            f"  NOTE: base station file not found at {base_obs}\n"
            f"  Creating a minimal HKSC stub with approximate position.\n"
            f"  For full RTK baseline, download HKSC RINEX from:\n"
            f"    https://www.geodetic.gov.hk/en/rinex/downv.aspx"
        )
        x, y, z = HKSC_ECEF
        base_obs.write_text(
            f"     3.02           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
            f"HKSC                                                        MARKER NAME\n"
            f"{x:14.4f}{y:14.4f}{z:14.4f}                  APPROX POSITION XYZ\n"
            f"                                                            END OF HEADER\n"
        )

    print(f"\nnormalized HK sequence written to {run_dir}")
    print("files:")
    for f in sorted(run_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name:30s} {size:>10,d} bytes")


if __name__ == "__main__":
    main()
