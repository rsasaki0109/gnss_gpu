#!/usr/bin/env python3
"""Fetch CDDIS 1 Hz high-rate base observations for GSDC2023 course windows.

CDDIS high-rate GNSS observation data are archived as 15-minute, 1-second
RINEX/Hatanaka files.  This script mirrors the existing daily-base downloader,
but only enumerates and fetches slots covering each GSDC course time span.

Earthdata/CDDIS authentication is usually required.  Use ``--dry-run`` first.
"""

from __future__ import annotations

import argparse
import html
import netrc
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.fetch_gsdc2023_base_obs import _iter_settings_rows  # noqa: E402
from experiments.fetch_gsdc2023_brdc import calendar_date_from_course  # noqa: E402
from experiments.gsdc2023_raw_bridge import DEFAULT_ROOT  # noqa: E402


CDDIS_HIGHRATE_ROOT = "https://cddis.nasa.gov/archive/gnss/data/highrate"
DEFAULT_CACHE_DIR = _REPO.parent / "ref" / "gsdc2023" / "base_rinex_highrate_cache"


@dataclass(frozen=True)
class HighrateSlot:
    site4: str
    year: int
    doy: int
    hour: int
    minute: int


def hour_letter(hour: int) -> str:
    h = int(hour)
    if h < 0 or h > 23:
        raise ValueError(f"hour must be 0..23, got {hour!r}")
    return chr(ord("a") + h)


def cddis_highrate_url_candidates(slot: HighrateSlot, *, root: str = CDDIS_HIGHRATE_ROOT) -> list[str]:
    yy = slot.year % 100
    site = slot.site4.lower()
    hch = hour_letter(slot.hour)
    out: list[str] = []
    for rinex_type in ("d", "o"):
        stem = f"{site}{slot.doy:03d}{hch}{slot.minute:02d}.{yy:02d}{rinex_type}"
        # Current CDDIS standards use gzip, while older operational archives and
        # some documentation examples used UNIX .Z. Try gzip first.
        for extension in ("gz", "Z"):
            filename = f"{stem}.{extension}"
            # Current NASA Earthdata catalog documents YYYY/DDD/YYo/HH. Older
            # CDDIS reports also mention YYYY/DDD/HH/YYT, so try both layouts.
            out.append(f"{root}/{slot.year}/{slot.doy:03d}/{yy:02d}{rinex_type}/{slot.hour:02d}/{filename}")
            out.append(f"{root}/{slot.year}/{slot.doy:03d}/{slot.hour:02d}/{yy:02d}{rinex_type}/{filename}")
    return out


def rinex3_highrate_index_url(slot: HighrateSlot, *, root: str) -> str:
    return f"{root.rstrip('/')}/{slot.year}/{slot.doy:03d}/"


def parse_rinex3_highrate_index_urls(slot: HighrateSlot, index_html: str, *, index_url: str) -> list[str]:
    """Return matching RINEX3 15-minute high-rate CRX URLs from a directory index."""

    site = re.escape(slot.site4.upper())
    stamp = f"{slot.year}{slot.doy:03d}{slot.hour:02d}{slot.minute:02d}"
    pattern = re.compile(
        rf"(?i)\b({site}[A-Z0-9]{{2}}[A-Z0-9]{{3}}_R_{stamp}_15M_01S_MO\.crx\.gz)\b"
    )
    found: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(index_html):
        name = html.unescape(match.group(1))
        if name in seen:
            continue
        seen.add(name)
        found.append(urllib.parse.urljoin(index_url, name))
    return found


def fetch_rinex3_highrate_index_candidates(slot: HighrateSlot, *, root: str, timeout_s: int = 30) -> list[str]:
    index_url = rinex3_highrate_index_url(slot, root=root)
    req = urllib.request.Request(index_url, headers={"User-Agent": "gnss_gpu/fetch_gsdc2023_highrate_base_obs"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return parse_rinex3_highrate_index_urls(slot, body, index_url=index_url)


def highrate_url_candidates(slot: HighrateSlot, *, root: str, use_index: bool = True) -> list[str]:
    urls: list[str] = []
    if use_index:
        try:
            urls.extend(fetch_rinex3_highrate_index_candidates(slot, root=root))
        except (OSError, urllib.error.URLError):
            pass
    urls.extend(cddis_highrate_url_candidates(slot, root=root))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def course_time_span_utc_ms(data_root: Path, split: str, course: str) -> tuple[float, float]:
    times: list[float] = []
    course_dir = data_root / split / course
    if not course_dir.is_dir():
        raise FileNotFoundError(f"course directory missing: {course_dir}")
    for phone_dir in sorted(course_dir.iterdir()):
        gnss = phone_dir / "device_gnss.csv"
        if not gnss.is_file():
            continue
        try:
            frame = pd.read_csv(gnss, usecols=["utcTimeMillis"], low_memory=False)
        except (ValueError, OSError):
            continue
        arr = pd.to_numeric(frame["utcTimeMillis"], errors="coerce").to_numpy(dtype="float64")
        arr = arr[pd.notna(arr)]
        if arr.size:
            times.extend([float(arr.min()), float(arr.max())])
    if not times:
        raise RuntimeError(f"no finite utcTimeMillis under {course_dir}")
    return min(times), max(times)


def highrate_slots_for_span(
    site4: str,
    start_utc_ms: float,
    end_utc_ms: float,
    *,
    margin_s: float = 180.0,
) -> list[HighrateSlot]:
    start_s = float(start_utc_ms) * 1.0e-3 - float(margin_s)
    end_s = float(end_utc_ms) * 1.0e-3 + float(margin_s)
    start_dt = datetime.fromtimestamp(start_s, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_s, tz=timezone.utc)
    slots = []
    cur = start_dt.replace(minute=(start_dt.minute // 15) * 15, second=0, microsecond=0)
    end_slot = end_dt.replace(minute=(end_dt.minute // 15) * 15, second=0, microsecond=0)
    while cur <= end_slot:
        slots.append(
            HighrateSlot(
                site4=site4.upper(),
                year=cur.year,
                doy=cur.timetuple().tm_yday,
                hour=cur.hour,
                minute=cur.minute,
            )
        )
        cur += timedelta(minutes=15)
    return slots


def _earthdata_credentials() -> tuple[str, str] | None:
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if user and password:
        return user, password
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if auth is None:
        return None
    login, _account, password = auth
    if login and password:
        return login, password
    return None


def _looks_like_html(path: Path) -> bool:
    head = path.read_bytes()[:256].lstrip().lower() if path.is_file() else b""
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"earthdata login" in head


def installed_highrate_obs_ready(path: Path) -> bool:
    """Return whether an installed course-level high-rate obs file is usable."""

    return path.is_file() and path.stat().st_size > 0 and not _looks_like_html(path)


def download_with_curl(url: str, dest: Path, *, timeout_s: int = 180) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "2",
        "--connect-timeout",
        str(timeout_s),
        "--max-time",
        str(timeout_s),
        "-o",
        str(dest),
        url,
    ]
    creds = _earthdata_credentials()
    if creds is not None:
        cmd[1:1] = ["-u", f"{creds[0]}:{creds[1]}"]
    else:
        cmd[1:1] = ["-n"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if _looks_like_html(dest):
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            "CDDIS returned an Earthdata login page; configure ~/.netrc for "
            "urs.earthdata.nasa.gov or set EARTHDATA_USERNAME/EARTHDATA_PASSWORD"
        )


def download_first_available(urls: list[str], dest_dir: Path, *, dry_run: bool) -> tuple[Path | None, str | None]:
    last_error: Exception | None = None
    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        dest = dest_dir / filename
        if dry_run:
            print(f"dry {url}")
            return None, url
        if dest.is_file() and dest.stat().st_size > 0:
            if _looks_like_html(dest):
                dest.unlink(missing_ok=True)
            else:
                return dest, url
        if dest.is_file() and dest.stat().st_size > 0:
            return dest, url
        try:
            download_with_curl(url, dest)
            if dest.is_file() and dest.stat().st_size > 0:
                return dest, url
        except (subprocess.CalledProcessError, OSError) as exc:
            last_error = exc
            dest.unlink(missing_ok=True)
    if last_error is not None:
        raise RuntimeError(str(last_error))
    return None, None


def decompress_highrate_obs(src: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    current = src
    try:
        from hatanaka import decompress_on_disk
    except ImportError as exc:
        raise RuntimeError("install the optional hatanaka package to decode high-rate RINEX") from exc

    for _ in range(3):
        out = Path(decompress_on_disk(current))
        if out == current:
            break
        current = out
        if current.suffix.lower() in {".o", ".obs"} or current.name[-1:].lower() == "o":
            return current
    if current.suffix == ".Z":
        plain = work_dir / current.name[:-2]
        with plain.open("wb") as fh:
            subprocess.run(["uncompress", "-c", str(current)], check=True, stdout=fh)
        current = Path(decompress_on_disk(plain))
    if current.suffix.lower() in {".o", ".obs"} or current.name[-1:].lower() == "o":
        return current
    raise RuntimeError(f"could not decode high-rate observation file: {src}")


def merge_rinex_obs(files: list[Path], dest: Path) -> None:
    if not files:
        raise ValueError("no RINEX obs files to merge")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        for idx, path in enumerate(files):
            data = path.read_bytes()
            if idx == 0:
                out.write(data)
                if not data.endswith(b"\n"):
                    out.write(b"\n")
                continue
            marker = b"END OF HEADER"
            marker_idx = data.find(marker)
            if marker_idx < 0:
                raise RuntimeError(f"RINEX header marker missing: {path}")
            line_end = data.find(b"\n", marker_idx)
            if line_end < 0:
                raise RuntimeError(f"RINEX header line unterminated: {path}")
            out.write(data[line_end + 1 :])
            if not data.endswith(b"\n"):
                out.write(b"\n")


def _iter_course_base_windows(data_root: Path, splits: list[str]):
    by_course: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for split, course, _phone, base1 in _iter_settings_rows(data_root, splits):
        by_course[(split, course, base1)].add(base1)
    for split, course, base1 in sorted(by_course):
        start_ms, end_ms = course_time_span_utc_ms(data_root, split, course)
        yield split, course, base1, start_ms, end_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--root-url", default=CDDIS_HIGHRATE_ROOT)
    parser.add_argument("--course", action="append", default=[], help="limit to a course; repeatable")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--margin-s", type=float, default=180.0)
    parser.add_argument("--install-name-template", default="{base}_1hz.obs")
    parser.add_argument("--no-index", action="store_true", help="skip RINEX3 directory-index discovery")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    rows = list(_iter_course_base_windows(data_root, list(args.splits)))
    if args.course:
        allowed = set(args.course)
        rows = [row for row in rows if row[1] in allowed]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no course/base windows found")

    ok = err = total_slots = 0
    for split, course, base1, start_ms, end_ms in rows:
        slots = highrate_slots_for_span(base1, start_ms, end_ms, margin_s=args.margin_s)
        total_slots += len(slots)
        out_name = args.install_name_template.format(base=base1)
        out_path = data_root / split / course / out_name
        print(f"{split}/{course} {base1}: slots={len(slots)} -> {out_path}")
        if out_path.is_file() and not args.overwrite and not args.dry_run:
            if installed_highrate_obs_ready(out_path):
                print(f"  skip existing {out_path}")
                ok += 1
                continue
            print(f"  remove unusable existing high-rate obs {out_path}", file=sys.stderr)
            out_path.unlink(missing_ok=True)
        decoded: list[Path] = []
        try:
            for slot in slots:
                slot_dir = cache_dir / f"{slot.year}" / f"{slot.doy:03d}" / base1.upper()
                src, _url = download_first_available(
                    highrate_url_candidates(slot, root=args.root_url.rstrip("/"), use_index=not args.no_index),
                    slot_dir,
                    dry_run=args.dry_run,
                )
                if src is not None:
                    decoded.append(decompress_highrate_obs(src, slot_dir / "_decoded"))
            if args.dry_run:
                ok += 1
                continue
            merge_rinex_obs(decoded, out_path)
            ok += 1
        except (RuntimeError, OSError, subprocess.CalledProcessError, urllib.error.URLError) as exc:
            err += 1
            print(f"  error {split}/{course} {base1}: {exc}", file=sys.stderr)
    print(f"summary: courses={len(rows)} ok={ok} errors={err} slots={total_slots} dry_run={args.dry_run}")
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
