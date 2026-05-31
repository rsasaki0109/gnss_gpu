from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fetch_gsdc2023_highrate_base_obs import (
    HighrateSlot,
    cddis_highrate_url_candidates,
    course_time_span_utc_ms,
    highrate_slots_for_span,
    hour_letter,
    merge_rinex_obs,
    download_with_curl,
    parse_rinex3_highrate_index_urls,
    installed_highrate_obs_ready,
    rinex3_highrate_index_url,
)


def _ms(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp() * 1000.0


def test_hour_letter_and_cddis_highrate_url_candidates() -> None:
    slot = HighrateSlot(site4="SLAC", year=2023, doy=145, hour=19, minute=15)

    assert hour_letter(0) == "a"
    assert hour_letter(19) == "t"
    urls = cddis_highrate_url_candidates(slot, root="https://example.test/highrate")

    assert urls[0] == "https://example.test/highrate/2023/145/23d/19/slac145t15.23d.gz"
    assert urls[1] == "https://example.test/highrate/2023/145/19/23d/slac145t15.23d.gz"
    assert urls[2].endswith("/23d/19/slac145t15.23d.Z")
    assert urls[4].endswith("/23o/19/slac145t15.23o.gz")


def test_highrate_slots_for_span_rounds_to_15_minute_grid_with_margin() -> None:
    slots = highrate_slots_for_span(
        "SLAC",
        _ms(datetime(2023, 5, 25, 19, 10, 30)),
        _ms(datetime(2023, 5, 25, 19, 39, 5)),
        margin_s=180,
    )

    assert [(s.hour, s.minute) for s in slots] == [(19, 0), (19, 15), (19, 30)]
    assert all(s.doy == 145 for s in slots)


def test_highrate_slots_for_span_handles_utc_day_boundary() -> None:
    slots = highrate_slots_for_span(
        "VDCY",
        _ms(datetime(2023, 5, 25, 23, 58, 30)),
        _ms(datetime(2023, 5, 26, 0, 6, 5)),
        margin_s=0,
    )

    assert [(s.doy, s.hour, s.minute) for s in slots] == [(145, 23, 45), (146, 0, 0)]


def test_rinex3_highrate_index_candidates_match_station_and_slot() -> None:
    slot = HighrateSlot(site4="SLAC", year=2023, doy=145, hour=19, minute=15)
    index_url = rinex3_highrate_index_url(slot, root="https://example.test/highrate")
    index_html = """
    <a href="SLAC00USA_R_20231451915_15M_01S_MO.crx.gz">match</a>
    <a href="SLAC00USA_R_20231451930_15M_01S_MO.crx.gz">wrong slot</a>
    <a href="VDCY00USA_R_20231451915_15M_01S_MO.crx.gz">wrong station</a>
    """

    assert parse_rinex3_highrate_index_urls(slot, index_html, index_url=index_url) == [
        "https://example.test/highrate/2023/145/SLAC00USA_R_20231451915_15M_01S_MO.crx.gz"
    ]


def test_course_time_span_utc_ms_uses_all_phone_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "sdc2023"
    p1 = data_root / "train" / "course" / "pixel5"
    p2 = data_root / "train" / "course" / "pixel7"
    p1.mkdir(parents=True)
    p2.mkdir(parents=True)
    pd.DataFrame({"utcTimeMillis": [3000, 1000]}).to_csv(p1 / "device_gnss.csv", index=False)
    pd.DataFrame({"utcTimeMillis": [2000, 5000]}).to_csv(p2 / "device_gnss.csv", index=False)

    assert course_time_span_utc_ms(data_root, "train", "course") == (1000.0, 5000.0)


def test_merge_rinex_obs_keeps_first_header_only(tmp_path: Path) -> None:
    f1 = tmp_path / "a.obs"
    f2 = tmp_path / "b.obs"
    out = tmp_path / "merged.obs"
    f1.write_text("header1\nEND OF HEADER\n> epoch1\n", encoding="ascii")
    f2.write_text("header2\nEND OF HEADER\n> epoch2\n", encoding="ascii")

    merge_rinex_obs([f1, f2], out)

    text = out.read_text(encoding="ascii")
    assert text.count("END OF HEADER") == 1
    assert "> epoch1" in text
    assert "> epoch2" in text


def test_download_with_curl_rejects_earthdata_login_html(tmp_path: Path, monkeypatch) -> None:
    dest = tmp_path / "login.html"

    def fake_run(_cmd, check, stdout, stderr, text):
        dest.write_text("<!DOCTYPE html><title>Earthdata Login</title>", encoding="ascii")

    monkeypatch.setattr("subprocess.run", fake_run)

    try:
        download_with_curl("https://example.test/file.Z", dest)
    except RuntimeError as exc:
        assert "Earthdata login page" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert not dest.exists()


def test_installed_highrate_obs_ready_rejects_login_html_and_empty_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.obs"
    empty = tmp_path / "empty.obs"
    login = tmp_path / "login.obs"
    rinex = tmp_path / "rinex.obs"
    empty.write_text("", encoding="ascii")
    login.write_text("<html><title>Earthdata Login</title></html>", encoding="ascii")
    rinex.write_text("     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n", encoding="ascii")

    assert not installed_highrate_obs_ready(missing)
    assert not installed_highrate_obs_ready(empty)
    assert not installed_highrate_obs_ready(login)
    assert installed_highrate_obs_ready(rinex)
