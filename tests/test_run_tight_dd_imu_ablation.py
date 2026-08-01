from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import run_tight_dd_imu_ablation as runner  # noqa: E402


def test_summarize_slices_scope_and_parses_tight_diagnostics(tmp_path: Path) -> None:
    reference = tmp_path / "reference.csv"
    with reference.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tow", "a", "b", "c", "d", "x", "y", "z"])
        for tow in range(4):
            writer.writerow([tow, 0, 0, 0, 0, tow, 0, 0])
    positions = tmp_path / "result.pos"
    positions.write_text(
        "% header\n2200 1.00 1.0 0.0 0.0\n2200 2.00 3.0 0.0 0.0\n",
        encoding="utf-8",
    )
    log = """Tight DD/IMU epochs: 2 (accepted=1, innovation_rejected=1, heading_deferred=3)
Tight DD rows: 18
Partial-AR epochs/fixed ambiguities: 1/4
Innovation-gated soft resets: 1
"""

    summary = runner._summarize(positions, reference, log, 0.2, 1, 3)

    assert summary["requested_epochs"] == 2
    assert summary["coverage"] == 1.0
    assert summary["honest_ppc_score_pct"] == 0.0
    assert summary["pass_0_5m"] == 0.5
    assert summary["error_p50_m"] == 0.5
    assert summary["tight_dd_epochs"] == 2
    assert summary["tight_dd_accepted"] == 1
    assert summary["tight_dd_rejected"] == 1
    assert summary["tight_dd_heading_deferred"] == 3
    assert summary["tight_dd_rows"] == 18
    assert summary["partial_ar_epochs"] == 1
    assert summary["fixed_ambiguities"] == 4
    assert summary["tight_dd_soft_resets"] == 1

    sliced = runner._summarize(
        positions, reference, "", 0.1, 1, 3, include_diagnostics=False
    )
    assert sliced["coverage"] == 1.0
    assert sliced["tight_dd_epochs"] is None
    assert sliced["tight_dd_accepted"] is None
    assert sliced["tight_dd_heading_deferred"] is None
    assert sliced["partial_ar_epochs"] is None

    baseline = dict(
        summary,
        scope_id="span",
        city="tokyo",
        run="run1",
        evaluation_role="holdout",
        variant="baseline",
        binary_sha256="same-build",
    )
    baseline["honest_ppc_score_pct"] = 10.0
    tight = dict(baseline, variant="tight_dd_imu", honest_ppc_score_pct=12.5)
    comparison = tmp_path / "comparison.csv"
    runner._write_comparison(comparison, [baseline, tight])
    row = next(csv.DictReader(comparison.open(newline="", encoding="utf-8")))
    assert float(row["tight_minus_baseline_honest_ppc_score_pct"]) == 2.5
    assert row["binary_sha256_match"] == "True"
    assert row["comparison_status"] == "matched"

    tight["binary_sha256"] = "different-build"
    runner._write_comparison(comparison, [baseline, tight])
    mismatch = next(csv.DictReader(comparison.open(newline="", encoding="utf-8")))
    assert mismatch["comparison_status"] == "binary_mismatch"
    assert mismatch["tight_minus_baseline_honest_ppc_score_pct"] == ""


def test_binary_provenance_hashes_exact_executable(tmp_path: Path) -> None:
    binary = tmp_path / "gnss_fuse.exe"
    binary.write_bytes(b"same-binary-for-both-variants")
    first = runner._binary_provenance(str(binary), use_wsl=False)
    second = runner._binary_provenance(str(binary), use_wsl=False)
    assert first["binary_sha256"] == second["binary_sha256"]
    assert first["binary_size_bytes"] == len(b"same-binary-for-both-variants")

    with pytest.raises(FileNotFoundError):
        runner._binary_provenance(str(tmp_path / "missing.exe"), use_wsl=False)
