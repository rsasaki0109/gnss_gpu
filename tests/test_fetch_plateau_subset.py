"""Tests for PLATEAU subset selection helpers."""

from pathlib import Path
import sys
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import fetch_plateau_subset
from fetch_plateau_subset import HTTPRangeReader, select_bldg_entries


class _FakeZip:
    def __init__(self, names):
        self._names = list(names)

    def namelist(self):
        return list(self._names)


def test_select_bldg_entries_matches_paths_without_leading_slash():
    zf = _FakeZip(
        [
            "udx/bldg/52365790_bldg_6697_op.gml",
            "udx/bldg/52365791_bldg_6697_op.gml",
            "udx/tran/52365790_tran_6697_op.gml",
        ]
    )

    selected = select_bldg_entries(zf, ["52365790"])

    assert selected == ["udx/bldg/52365790_bldg_6697_op.gml"]


def test_select_bldg_entries_matches_paths_with_leading_slash():
    zf = _FakeZip(
        [
            "foo/udx/bldg/53394613_bldg_6697_2_op.gml",
            "foo/udx/bldg/53394614_bldg_6697_2_op.gml",
        ]
    )

    selected = select_bldg_entries(zf, ["53394613"])

    assert selected == ["foo/udx/bldg/53394613_bldg_6697_2_op.gml"]


def test_http_range_reader_uses_head_content_length(monkeypatch):
    head = Mock()
    head.url = "https://example.com/archive.zip"
    head.headers = {"content-length": "1234"}
    head.raise_for_status = Mock()

    monkeypatch.setattr(fetch_plateau_subset.requests, "head", lambda *args, **kwargs: head)

    reader = HTTPRangeReader("https://example.com/archive.zip")

    assert reader.url == "https://example.com/archive.zip"
    assert reader.size == 1234


def test_http_range_reader_falls_back_to_range_probe(monkeypatch):
    head = Mock()
    head.url = "https://dropbox.example/archive.zip"
    head.headers = {}
    head.raise_for_status = Mock()

    probe = Mock()
    probe.url = "https://cdn.example/archive.zip"
    probe.headers = {"content-range": "bytes 0-0/4452177524"}
    probe.raise_for_status = Mock()

    monkeypatch.setattr(fetch_plateau_subset.requests, "head", lambda *args, **kwargs: head)
    monkeypatch.setattr(fetch_plateau_subset.requests, "get", lambda *args, **kwargs: probe)

    reader = HTTPRangeReader("https://dropbox.example/archive.zip")

    assert reader.url == "https://dropbox.example/archive.zip"
    assert reader.size == 4_452_177_524
