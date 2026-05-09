#!/usr/bin/env python3
"""Download daily GPS broadcast ephemeris (RINEX nav) as ``brdc.<yy>n`` per GSDC course.

Kaggle / raw smartphone logs do not ship ``brdc.*`` in each course folder; MATLAB
``preprocessing.m`` expects ``path+"brdc.*"``. This script fills that gap using the
public NOAA CORS RINEX mirror (same pattern as RTKLIB ``URL_LIST.txt``)::

  https://noaa-cors-pds.s3.amazonaws.com/rinex/<year>/<doy>/brdc<doy>0.<yy>n.gz

Course directory names must start with ``YYYY-MM-DD-...`` (GSDC naming).

Example::

  PYTHONPATH=python python3 experiments/fetch_gsdc2023_brdc.py \\
    --data-root ../ref/gsdc2023/dataset_2023 --splits train test --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT

NOAA_BRDC_GPS_GZ = (
    "https://noaa-cors-pds.s3.amazonaws.com/rinex/{year}/{doy:03d}/brdc{doy:03d}0.{yy}n.gz"
)


def calendar_date_from_course(course: str) -> date:
    parts = course.split("-")
    if len(parts) < 3:
        raise ValueError(f"expected YYYY-MM-DD-... course name, got {course!r}")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def brdc_noaa_url(d: date) -> str:
    doy = d.timetuple().tm_yday
    yy = d.year % 100
    return NOAA_BRDC_GPS_GZ.format(year=d.year, doy=doy, yy=yy)


def _download_gunzip(url: str, dest_nav: Path, *, timeout: int = 180) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "gnss_gpu/fetch_gsdc2023_brdc"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    data = gzip.decompress(raw)
    dest_nav.write_bytes(data)


def fetch_brdc_for_course(
    data_root: Path,
    split: str,
    course: str,
    *,
    dry_run: bool,
    overwrite: bool,
) -> str:
    """Return status: skipped_present | skipped_dry_run | ok | error."""
    d = calendar_date_from_course(course)
    url = brdc_noaa_url(d)
    out_dir = data_root / split / course
    yy = d.year % 100
    dest = out_dir / f"brdc.{yy:02d}n"
    if not out_dir.is_dir():
        return f"error:missing_dir:{out_dir}"
    if dest.is_file() and not overwrite:
        return "skipped_present"
    if dry_run:
        print(f"dry {split}/{course} -> {dest.name} <= {url}")
        return "skipped_dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        _download_gunzip(url, dest)
    except urllib.error.HTTPError as e:
        return f"error:http:{e.code}:{url}"
    except OSError as e:
        return f"error:io:{e}"
    print(f"ok {split}/{course} -> {dest} ({url})")
    return "ok"


def _discover_courses(data_root: Path, splits: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for sp in splits:
        root = data_root / sp
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir():
                out.append((sp, p.name))
    return out


def _courses_from_settings(data_root: Path, splits: list[str]) -> list[tuple[str, str]]:
    import pandas as pd

    rows: list[tuple[str, str]] = []
    for sp in splits:
        path = data_root / f"settings_{sp}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "Course" not in df.columns:
            continue
        for c in df["Course"].astype(str).unique():
            rows.append((sp, c))
    return sorted(set(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--splits", nargs="*", default=["train", "test"])
    p.add_argument(
        "--from-settings",
        action="store_true",
        help="use settings_*.csv Course column only (skip dirs without a settings row)",
    )
    p.add_argument("--course", action="append", default=None, help="limit to course name (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    data_root = args.data_root.resolve()
    splits = list(args.splits)
    if args.from_settings:
        pairs = _courses_from_settings(data_root, splits)
    else:
        pairs = _discover_courses(data_root, splits)

    if args.course:
        allow = set(args.course)
        pairs = [(s, c) for s, c in pairs if c in allow]

    counts: dict[str, int] = {}
    for sp, course in pairs:
        st = fetch_brdc_for_course(
            data_root,
            sp,
            course,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        counts[st] = counts.get(st, 0) + 1
        if st.startswith("error"):
            print(st, file=sys.stderr)

    print("summary:", counts)


if __name__ == "__main__":
    main()
