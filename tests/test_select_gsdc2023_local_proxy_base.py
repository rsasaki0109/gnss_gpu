from __future__ import annotations

import json

import pandas as pd

from experiments.select_gsdc2023_local_proxy_base import main, select_local_proxy_base


def _submission(offset: float = 0.0, *, bad: bool = False) -> pd.DataFrame:
    lat = [37.0 + offset, 37.1 + offset, 37.2 + offset]
    lon = [-122.0, -122.1, -122.2]
    if bad:
        lat[1] = 55.0
    return pd.DataFrame(
        {
            "tripId": ["trip-a/pixel5", "trip-a/pixel5", "trip-b/pixel4"],
            "UnixTimeMillis": [1000, 2000, 1000],
            "LatitudeDegrees": lat,
            "LongitudeDegrees": lon,
        },
    )


def test_select_local_proxy_base_picks_pairwise_medoid(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    _submission(0.0).to_csv(results_root / "gsdc2023_submission_a.csv", index=False)
    _submission(0.0001).to_csv(results_root / "gsdc2023_submission_b.csv", index=False)
    _submission(0.0002).to_csv(results_root / "gsdc2023_submission_c.csv", index=False)
    _submission(0.0, bad=True).to_csv(results_root / "gsdc2023_submission_bad.csv", index=False)

    payload = select_local_proxy_base(results_root=results_root, output_dir=tmp_path / "audit")

    assert payload["selected_proxy_base"] == "gsdc2023_submission_b.csv"
    assert payload["submit_allowed"] is False
    candidate_rows = pd.read_csv(tmp_path / "audit" / "local_proxy_base_candidates.csv")
    selected = candidate_rows[candidate_rows["selected_proxy_base"]]
    assert selected["filename"].tolist() == ["gsdc2023_submission_b.csv"]
    bad = candidate_rows[candidate_rows["filename"] == "gsdc2023_submission_bad.csv"].iloc[0]
    assert bad["eligible_proxy_base"] == False
    assert "coordinate_sanity_failed" in bad["rejection_reason"]


def test_select_local_proxy_base_can_reject_filename(tmp_path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    _submission(0.0).to_csv(results_root / "gsdc2023_submission_a.csv", index=False)
    _submission(0.0001).to_csv(results_root / "gsdc2023_submission_b.csv", index=False)
    _submission(0.0002).to_csv(results_root / "gsdc2023_submission_c.csv", index=False)

    payload = select_local_proxy_base(
        results_root=results_root,
        output_dir=tmp_path / "audit",
        reject_filenames={"gsdc2023_submission_b.csv"},
    )

    assert payload["selected_proxy_base"] == "gsdc2023_submission_a.csv"
    candidate_rows = pd.read_csv(tmp_path / "audit" / "local_proxy_base_candidates.csv")
    rejected = candidate_rows[candidate_rows["filename"] == "gsdc2023_submission_b.csv"].iloc[0]
    assert rejected["eligible_proxy_base"] == False
    assert "explicit_reject_filename" in rejected["rejection_reason"]


def test_select_local_proxy_base_cli(tmp_path, capsys) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    _submission(0.0).to_csv(results_root / "gsdc2023_submission_a.csv", index=False)
    _submission(0.0001).to_csv(results_root / "gsdc2023_submission_b.csv", index=False)
    output_dir = tmp_path / "audit"

    assert main(["--results-root", str(results_root), "--output-dir", str(output_dir)]) == 0

    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["selected_proxy_base"] == "gsdc2023_submission_a.csv"
    assert "local_proxy_base_selection.md" in capsys.readouterr().out
