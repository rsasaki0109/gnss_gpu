#!/usr/bin/env python3
"""Fetch a minimal UrbanNav Hong Kong subset and normalize it for this repo."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


HK20190428_GNSS_URL = (
    "https://www.dropbox.com/s/25dsnx27wu8zgew/"
    "GNSS%20RINEX%20UrbanNav-HK-Data20190428.tar.gz?dl=1"
)
HK20190428_IMU_REF_URL = (
    "https://www.dropbox.com/s/g1xt89py1j0uea1/"
    "IMU-Reference%20UrbanNav-HK-Data20190428.zip?dl=1"
)


def select_hk20190428_gnss_members(names: list[str]) -> dict[str, str]:
    rover_candidates = sorted(
        name for name in names
        if name.lower().endswith(".obs") and "/com" in name.lower()
    )
    base_obs_candidates = sorted(
        name for name in names
        if name.lower().endswith((".19o", ".o")) or name.lower().endswith("base.obs")
    )
    base_nav_candidates = sorted(
        name for name in names
        if name.lower().endswith((".19n", ".n", ".nav"))
    )

    if not rover_candidates:
        raise RuntimeError("could not find Hong Kong rover observation file")
    if not base_obs_candidates:
        raise RuntimeError("could not find Hong Kong base observation file")
    if not base_nav_candidates:
        raise RuntimeError("could not find Hong Kong navigation file")

    return {
        "rover_ublox.obs": rover_candidates[0],
        "base_hksc.obs": base_obs_candidates[0],
        "base.nav": base_nav_candidates[0],
    }


def select_hk20190428_support_members(names: list[str]) -> dict[str, str]:
    lowered = {name.lower(): name for name in names}
    try:
        return {
            "imu.csv": lowered["imu.csv"],
            "reference.csv": lowered["reference.csv"],
        }
    except KeyError as exc:
        raise RuntimeError("could not find Hong Kong imu/reference CSVs") from exc


def _download(url: str, path: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, path.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a minimal UrbanNav Hong Kong subset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("hk20190428",),
        default="hk20190428",
        help="Hong Kong UrbanNav dataset preset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for the normalized run directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="HK_20190428",
        help="Name of the normalized run subdirectory",
    )
    args = parser.parse_args()

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="urbannav_hk_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        gnss_path = tmpdir_path / "hk_gnss.tar.gz"
        imu_ref_path = tmpdir_path / "hk_imu_ref.zip"

        _download(HK20190428_GNSS_URL, gnss_path)
        _download(HK20190428_IMU_REF_URL, imu_ref_path)

        with tarfile.open(gnss_path, "r:gz") as tf:
            gnss_map = select_hk20190428_gnss_members(tf.getnames())
            for out_name, member_name in gnss_map.items():
                member = tf.extractfile(member_name)
                if member is None:
                    raise RuntimeError(f"could not extract {member_name}")
                with (run_dir / out_name).open("wb") as fh:
                    shutil.copyfileobj(member, fh)

        with zipfile.ZipFile(imu_ref_path) as zf:
            support_map = select_hk20190428_support_members(zf.namelist())
            for out_name, member_name in support_map.items():
                with zf.open(member_name) as src, (run_dir / out_name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    print(f"normalized Hong Kong subset written to {run_dir}")


if __name__ == "__main__":
    main()
