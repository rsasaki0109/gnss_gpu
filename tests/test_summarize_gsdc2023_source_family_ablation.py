from __future__ import annotations

import csv
import json

from experiments.summarize_gsdc2023_source_family_ablation import (
    build_source_family_ranking,
    main,
)


def _write_manifest(path) -> None:
    rows = [
        {
            "family": "pixel6pro_fgo",
            "mode": "only",
            "sources": "fgo",
            "phones": "pixel6pro",
            "selected_rows": "10",
            "selected_trip_count": "2",
            "selected_vs_reference_score_m": "2.0",
            "selected_vs_reference_p50_m": "1.0",
            "selected_vs_reference_p95_m": "3.0",
            "selected_vs_reference_max_m": "4.0",
            "output": "only_pixel6.csv",
            "output_sha256": "sha-only-pixel6",
        },
        {
            "family": "pixel6pro_fgo",
            "mode": "revert",
            "sources": "fgo",
            "phones": "pixel6pro",
            "selected_rows": "10",
            "selected_trip_count": "2",
            "selected_vs_reference_score_m": "2.0",
            "selected_vs_reference_p50_m": "1.0",
            "selected_vs_reference_p95_m": "3.0",
            "selected_vs_reference_max_m": "4.0",
            "output": "revert_pixel6.csv",
            "output_sha256": "sha-revert-pixel6",
        },
        {
            "family": "pixel4_fgo",
            "mode": "only",
            "sources": "fgo",
            "phones": "pixel4",
            "selected_rows": "5",
            "selected_trip_count": "1",
            "selected_vs_reference_score_m": "5.0",
            "selected_vs_reference_p50_m": "4.0",
            "selected_vs_reference_p95_m": "6.0",
            "selected_vs_reference_max_m": "7.0",
            "output": "only_pixel4.csv",
            "output_sha256": "sha-only-pixel4",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_source_family_ranking_writes_outputs(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "out"
    _write_manifest(manifest)

    rows = build_source_family_ranking(manifest_path=manifest, output_dir=output_dir, tag="test")

    assert [row["family"] for row in rows] == ["pixel4_fgo", "pixel6pro_fgo"]
    assert rows[0]["recommended_action"] == "first_mtv700_probe"
    assert rows[1]["recommended_action"] == "risk_ablation"
    assert (output_dir / "source_family_ablation_ranking_test.csv").is_file()
    payload = json.loads((output_dir / "source_family_ablation_ranking_test.json").read_text())
    assert payload["submit_policy"]["first_probe_family"] == "pixel4_fgo"
    assert "`pixel4_fgo`" in (output_dir / "source_family_ablation_ranking_test.md").read_text()


def test_build_source_family_ranking_cli(tmp_path, capsys) -> None:
    manifest = tmp_path / "manifest.csv"
    output_dir = tmp_path / "out"
    _write_manifest(manifest)

    assert main(["--manifest", str(manifest), "--output-dir", str(output_dir), "--tag", "test"]) == 0
    assert "source_family_ablation_ranking_test.md" in capsys.readouterr().out
