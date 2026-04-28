#!/usr/bin/env python3
"""Download NOAA CORS Hatanaka base observations and install as ``<Base1>_rnx3.obs``.

``preprocessing.m`` expects, under each ``<split>/<course>/``::

  <Base1>_rnx3.obs

NOAA serves daily Hatanaka files::

  https://geodesy.noaa.gov/corsdata/rinex/<year>/<doy>/<site>/<site><doy>0.<yy>d.gz

This script downloads ``*.d.gz``, gunzips, runs **Hatanaka → RINEX obs** via the
``hatanaka`` package (``pip install hatanaka``), then copies the result to each
trip's course directory using ``settings_*.csv`` rows.

Example::

  pip install --user hatanaka
  PYTHONPATH=python python3 experiments/fetch_gsdc2023_base_obs.py \\
    --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.fetch_gsdc2023_brdc import calendar_date_from_course
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT


def _noaa_hatanaka_url(site4: str, d: date) -> str:
    doy = d.timetuple().tm_yday
    yy = d.year % 100
    s = site4.strip().lower()
    return (
        f"https://geodesy.noaa.gov/corsdata/rinex/{d.year}/{doy:03d}/{s}/"
        f"{s}{doy:03d}0.{yy:02d}d.gz"
    )


def _download(url: str, dest_gz: Path, *, timeout: int = 120) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "gnss_gpu/fetch_gsdc2023_base_obs"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest_gz.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def _gunzip_to(src_gz: Path, dest: Path) -> None:
    with gzip.open(src_gz, "rb") as f_in, dest.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _hatanaka_to_obs(hatak_path: Path) -> Path:
    from hatanaka import decompress_on_disk

    return Path(decompress_on_disk(hatak_path))


def ensure_base_obs_for_day(site4: str, d: date, cache_dir: Path) -> Path:
    """Return path to a cached RINEX observation file (``.obs``) for *site4* and day *d*."""

    doy = d.timetuple().tm_yday
    yy = d.year % 100
    s = site4.strip().lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{site4.upper()}_{d.year}_{doy:03d}.obs"
    if cached.is_file() and cached.stat().st_size > 0:
        return cached

    last_err: OSError | urllib.error.HTTPError | RuntimeError | None = None
    for delta in (0, -1, 1, -2, 2):
        d_try = d + timedelta(days=delta)
        doy_t = d_try.timetuple().tm_yday
        yy_t = d_try.year % 100
        url = _noaa_hatanaka_url(site4, d_try)
        gz_path = cache_dir / f"_{s}{doy_t:03d}0.{yy_t:02d}d.gz"
        hatak_path = cache_dir / f"{s}{doy_t:03d}0.{yy_t:02d}d"
        try:
            _download(url, gz_path)
            _gunzip_to(gz_path, hatak_path)
            obs_like = Path(_hatanaka_to_obs(hatak_path))
            if not obs_like.is_file():
                raise RuntimeError(f"hatanaka produced no obs next to {hatak_path}")
            shutil.copy2(obs_like, cached)
            obs_like.unlink(missing_ok=True)
            return cached
        except (urllib.error.HTTPError, OSError, RuntimeError) as e:
            last_err = e
        finally:
            gz_path.unlink(missing_ok=True)
            hatak_path.unlink(missing_ok=True)
    raise last_err if last_err else RuntimeError("no NOAA base obs")


def _iter_settings_rows(data_root: Path, splits: list[str]):
    import pandas as pd

    for sp in splits:
        p = data_root / f"settings_{sp}.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        for row in df.itertuples(index=False):
            course = str(getattr(row, "Course", "") or "").strip()
            phone = str(getattr(row, "Phone", "") or "").strip()
            base1 = str(getattr(row, "Base1", "") or "").strip()
            if not course or not phone or not base1:
                continue
            yield sp, course, phone, base1


def main() -> None:
    try:
        import hatanaka  # noqa: F401
    except ImportError:
        print("Install: pip install hatanaka", file=sys.stderr)
        raise SystemExit(1)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--splits", nargs="*", default=["train", "test"])
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="cache decompressed obs (default: <repo>/ref/gsdc2023/base_rinex_cache)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    data_root = args.data_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    cache_dir = args.cache_dir or (repo / "ref" / "gsdc2023" / "base_rinex_cache")
    cache_dir = cache_dir.resolve()

    # Unique (base, date) -> list of (split, course)
    from collections import defaultdict

    by_key: dict[tuple[str, date], set[tuple[str, str]]] = defaultdict(set)
    for sp, course, _phone, base1 in _iter_settings_rows(data_root, list(args.splits)):
        d = calendar_date_from_course(course)
        by_key[(base1, d)].add((sp, course))

    print(f"unique base-days: {len(by_key)} cache_dir={cache_dir}")

    ok = err = 0
    for (base1, d), destinations in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        try:
            if args.dry_run:
                print(f"dry-run {base1} {d} -> {len(destinations)} course(s)")
                continue
            src = ensure_base_obs_for_day(base1, d, cache_dir)
            for sp, course in sorted(destinations):
                dest_dir = data_root / sp / course
                dest = dest_dir / f"{base1}_rnx3.obs"
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                ok += 1
        except (urllib.error.HTTPError, OSError, RuntimeError) as e:
            print(f"error {base1} {d}: {e}", file=sys.stderr)
            err += 1

    if not args.dry_run:
        print(f"course files written: {ok} errors: {err}")


if __name__ == "__main__":
    main()
