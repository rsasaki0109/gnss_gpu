#!/usr/bin/env python3
"""Fetch a minimal UrbanNav run subset from the remote Tokyo ZIP."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from fetch_plateau_subset import HTTPRangeReader, extract_entries


TOKYO_ZIP_URL = (
    "https://www.dropbox.com/scl/fi/b0d7jilvpjxf8h471iqvb/"
    "Tokyo_Data.zip?rlkey=z1qjhr84nb6ahxypvy6fn5ow6&dl=1"
)

PRESET_URLS = {
    "tokyo": TOKYO_ZIP_URL,
}

CORE_FILENAMES = (
    "base.nav",
    "base_trimble.obs",
    "reference.csv",
    "rover_trimble.obs",
    "rover_ublox.obs",
    "imu.csv",
)


def select_run_entries(
    zf: zipfile.ZipFile,
    run_name: str,
    include_imu: bool = True,
    include_lidar: bool = False,
) -> list[str]:
    """Select the minimal file set for one UrbanNav run."""
    prefix = f"Tokyo_Data/{run_name}/"
    wanted = set(CORE_FILENAMES)
    if not include_imu:
        wanted.discard("imu.csv")
    if include_lidar:
        wanted.add("lidar.bag")

    entries = []
    for name in zf.namelist():
        if not name.startswith(prefix):
            continue
        if Path(name).name in wanted:
            entries.append(name)
    return sorted(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a minimal UrbanNav run subset")
    parser.add_argument(
        "--run",
        type=str,
        choices=("Odaiba", "Shinjuku"),
        required=True,
        help="UrbanNav Tokyo run name",
    )
    parser.add_argument(
        "--zip-url",
        type=str,
        default="",
        help="Direct UrbanNav Tokyo ZIP URL",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESET_URLS),
        default="tokyo",
        help="Built-in UrbanNav ZIP preset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for extracted files",
    )
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="Skip imu.csv",
    )
    parser.add_argument(
        "--include-lidar",
        action="store_true",
        help="Also extract lidar.bag",
    )
    args = parser.parse_args()

    zip_url = args.zip_url or PRESET_URLS[args.preset]
    run_output_dir = args.output_dir / args.run
    run_output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(HTTPRangeReader(zip_url)) as zf:
        entries = select_run_entries(
            zf,
            args.run,
            include_imu=not args.no_imu,
            include_lidar=args.include_lidar,
        )
        if not entries:
            raise RuntimeError(f"no UrbanNav entries found for run: {args.run}")
        extract_entries(zf, entries, run_output_dir)

    print(f"extracted {len(entries)} files to {run_output_dir}")


if __name__ == "__main__":
    main()
