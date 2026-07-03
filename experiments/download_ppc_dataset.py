#!/usr/bin/env python3
"""Install the taroz/PPC-Dataset into datasets/PPC-Dataset-data.

The official release is a ~155 MB zip on Microsoft OneDrive:
https://github.com/taroz/PPC-Dataset

Automated download often fails with HTTP 403/404 (guest auth required).
Use a browser to download the zip, then point this script at the file:

    PYTHONPATH=python python experiments/download_ppc_dataset.py --zip path/to/PPC-Dataset.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover - optional for --try-auto
    requests = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "datasets" / "PPC-Dataset-data"
OFFICIAL_ONEDRIVE = (
    "https://chibakoudai-my.sharepoint.com/:u:/g/personal/"
    "66lsm6_chibatech_ac_jp/ETmyNr1VrcpFjqxgjXJdgkQBkfTwnykhSPOVKClOUBNOMQ"
)
OFFICIAL_ONEDRIVE_DOWNLOAD = OFFICIAL_ONEDRIVE + "?download=1"
DEFAULT_SSD_DEST = Path("E:/datasets/PPC-Dataset-data")
REQUIRED_RUN_FILES = ("rover.obs", "base.obs", "base.nav", "reference.csv", "imu.csv")
EXPECTED_RUNS = (
    "tokyo/run1",
    "tokyo/run2",
    "tokyo/run3",
    "nagoya/run1",
    "nagoya/run2",
    "nagoya/run3",
)


def _find_run_root(extracted_root: Path) -> Path | None:
    for candidate in (extracted_root, extracted_root / "PPC-Dataset"):
        tokyo = candidate / "tokyo" / "run1" / "rover.obs"
        if tokyo.is_file():
            return candidate
    for rover in extracted_root.rglob("tokyo/run1/rover.obs"):
        return rover.parents[2]
    return None


def _validate_layout(data_root: Path) -> list[str]:
    missing: list[str] = []
    for run in EXPECTED_RUNS:
        run_dir = data_root / run
        for name in REQUIRED_RUN_FILES:
            if not (run_dir / name).is_file():
                missing.append(f"{run}/{name}")
    return missing


def _candidate_zip_paths() -> list[Path]:
    home = Path.home()
    names = (
        home / "Downloads" / "PPC-Dataset.zip",
        home / "Downloads" / "PPC-Dataset-data.zip",
        PROJECT_ROOT / "datasets" / "PPC-Dataset.zip",
    )
    return [path for path in names if path.is_file() and path.stat().st_size >= 1_000_000]


def try_auto_download(dest_zip: Path) -> bool:
    """Best-effort OneDrive fetch. Returns True when a real zip is written."""
    if requests is None:
        return False
    response = requests.get(
        OFFICIAL_ONEDRIVE_DOWNLOAD,
        stream=True,
        allow_redirects=True,
        timeout=300,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    if response.status_code != 200:
        return False
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest_zip.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
    if written < 1_000_000:
        dest_zip.unlink(missing_ok=True)
        return False
    if dest_zip.read_bytes()[:2] != b"PK":
        dest_zip.unlink(missing_ok=True)
        return False
    return True


def install_from_zip(
    zip_path: Path,
    dest: Path,
    *,
    force: bool,
    min_zip_bytes: int = 1_000_000,
) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    if zip_path.stat().st_size < int(min_zip_bytes):
        raise ValueError(
            f"zip looks too small ({zip_path.stat().st_size} bytes); "
            "OneDrive often returns an HTML login page instead of the archive. "
            "Download manually in a browser and retry."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if not force:
            missing = _validate_layout(dest)
            if not missing:
                print(f"[ok] existing dataset at {dest}")
                return
            print(f"[warn] incomplete dataset at {dest}; reinstalling")
        shutil.rmtree(dest)

    staging = dest.parent / ".ppc_dataset_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    print(f"[install] extracting {zip_path} -> {staging}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(staging)

    run_root = _find_run_root(staging)
    if run_root is None:
        shutil.rmtree(staging)
        raise ValueError(f"could not locate tokyo/run1 in extracted zip: {zip_path}")

    shutil.move(str(run_root), str(dest))
    shutil.rmtree(staging, ignore_errors=True)

    missing = _validate_layout(dest)
    if missing:
        raise ValueError(f"installed dataset is incomplete; missing: {missing[:8]}")
    print(f"[ok] installed PPC dataset at {dest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to PPC-Dataset.zip downloaded from the official OneDrive link",
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument(
        "--ssd-dest",
        type=Path,
        default=None,
        help=f"Install on mobile SSD (default when set: {DEFAULT_SSD_DEST})",
    )
    parser.add_argument("--force", action="store_true", help="Reinstall even if dest exists")
    parser.add_argument(
        "--try-auto",
        action="store_true",
        help="Attempt OneDrive API download before failing",
    )
    args = parser.parse_args(argv)
    dest = args.ssd_dest or args.dest
    if args.ssd_dest is None and DEFAULT_SSD_DEST.drive and Path(f"{DEFAULT_SSD_DEST.drive}/").exists():
        dest = DEFAULT_SSD_DEST

    zip_path = args.zip
    if zip_path is None:
        candidates = _candidate_zip_paths()
        if candidates:
            zip_path = candidates[0]
            print(f"[install] using local zip: {zip_path}")
        elif args.try_auto:
            auto_zip = (dest.parent / "PPC-Dataset.zip").resolve()
            print("[install] trying OneDrive auto-download...")
            if try_auto_download(auto_zip):
                zip_path = auto_zip
                print(f"[install] downloaded {auto_zip} ({auto_zip.stat().st_size} bytes)")
            else:
                print("[install] auto-download failed (link often requires browser auth)")

    if zip_path is None:
        print("Official PPC-Dataset download (manual):")
        print(f"  {OFFICIAL_ONEDRIVE}")
        print("")
        print("After downloading in a browser, run:")
        print(
            "  PYTHONPATH=python python experiments/download_ppc_dataset.py "
            "--zip <path/to/PPC-Dataset.zip>"
        )
        print("")
        print("Or place PPC-Dataset.zip in ~/Downloads and rerun without --zip.")
        return 1

    install_from_zip(zip_path.resolve(), dest.resolve(), force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
