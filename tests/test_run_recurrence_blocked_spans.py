from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments/run_recurrence_blocked_spans.py"
)
SPEC = importlib.util.spec_from_file_location("run_recurrence_blocked_spans", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_safe_blocked_replay_rejects_legacy_policy_metadata(
    tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "span_id",
                "city",
                "run",
                "start_epoch",
                "end_epoch_exclusive",
                "evaluation_role",
                "recurrence_evaluation_role",
                "provenance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "span_id": "tokyo_run2_seg10",
                "city": "tokyo",
                "run": "run2",
                "start_epoch": 10,
                "end_epoch_exclusive": 12,
                "evaluation_role": "holdout",
                "recurrence_evaluation_role": "holdout",
                "provenance": "test",
            }
        )

    stem = "candidate_3dma_recurrence_blocked_tokyo_run2_seg10"
    summary = tmp_path / f"{stem}_summary.json"
    epochs = tmp_path / f"{stem}_epochs.csv"
    summary.write_text(
        json.dumps(
            {
                "start_epoch": 10,
                "requested_epochs": 2,
                "evaluated_epochs": 2,
                "skipped_epochs": 0,
            }
        ),
        encoding="utf-8",
    )
    epochs.write_text("epoch\n10\n11\n", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        summary.write_text(
            json.dumps(
                {
                    "start_epoch": 10,
                    "requested_epochs": 2,
                    "evaluated_epochs": 2,
                    "skipped_epochs": 0,
                    "recurrence_min_selected_probability": 0.05,
                    "recurrence_max_source_error_m": 20.0,
                    "recurrence_allow_boundary": False,
                    "runtime_s": 1.0,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        MODULE,
        "_summarize_epoch_files",
        lambda *_args, **_kwargs: {
            "coverage": 1.0,
            "honest_ppc_score_pct": 1.0,
            "selected_p50_m": 1.0,
            "selected_p95_m": 2.0,
            "selected_p99_m": 3.0,
            "recurrence_abstained_epochs": 2,
            "recurrence_acceptance_rate": 0.0,
            "runtime_s": 1.0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--data-root",
            str(tmp_path / "data"),
            "--source-pos-dir",
            str(tmp_path / "source"),
            "--triangle-cache-dir",
            str(tmp_path / "cache"),
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert MODULE.main() == 0
    assert len(commands) == 1
    assert commands[0][-4:] == [
        "--recurrence-max-source-error-m",
        "20.0",
        "--recurrence-min-selected-probability",
        "0.05",
    ]

    output = tmp_path / "candidate_3dma_recurrence_blocked_spans_summary.csv"
    row = next(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert row["recurrence_mode"] == "safe_gated"
    assert row["recurrence_min_selected_probability"] == "0.05"
    assert row["recurrence_max_source_error_m"] == "20.0"
    assert row["recurrence_allow_boundary"] == "False"
    assert row["coverage"] == "1.0"
    assert row["selected_p50_m"] == "1.0"
    assert row["selected_p95_m"] == "2.0"
    assert row["selected_p99_m"] == "3.0"
    assert row["recurrence_abstained_epochs"] == "2"
