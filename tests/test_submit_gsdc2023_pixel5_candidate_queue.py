from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import pytest

from experiments.submit_gsdc2023_pixel5_candidate_queue import (
    PENDING_QUEUE,
    assert_matlab_equivalence_gate,
    assert_pre_submit_manifest_gate,
    assert_ready_report_consistency,
    assert_submit_risk_gate,
    build_ready_report,
    candidate_submission_path,
    existing_queue_items,
    kaggle_submit_command,
    main,
    prepare_ready_report,
    pre_submit_manifest_payload,
    risk_report_payload,
    selected_queue,
    sha256_file,
    write_ready_report,
    write_submit_readiness_doc,
)


def test_pending_queue_prioritizes_sjc_r_then_mtv_sjc_then_lax_ebf() -> None:
    groups = [item.priority_group for item in PENDING_QUEUE]

    assert groups[:3] == ["sjc_r_scale_sweep"] * 3
    assert groups[3:6] == ["p6p0_clean_sjc_r_scale_sweep"] * 3
    assert groups[6:12] == ["mtv_sjc_trip_ablation"] * 6
    assert groups[12:] == ["lax_ebf_trip_ablation"] * 8


def test_candidate_submission_path_uses_scripted_output_layout() -> None:
    output_dir = Path("out")
    path = candidate_submission_path("pixel5phone_3p375_sjc_r0p84375", output_dir, "tag")

    assert path == (
        output_dir
        / "pixel5phone_3p375_sjc_r0p84375"
        / "submission_best_basecorr_posoffset_pixel5phone_3p375_sjc_r0p84375_plus_pixel5_patch_tag.csv"
    )


def test_selected_queue_filters_by_group() -> None:
    selected = selected_queue({"mtv_sjc_trip_ablation"})

    assert len(selected) == 6
    assert {item.priority_group for item in selected} == {"mtv_sjc_trip_ablation"}

    p6p0 = selected_queue({"p6p0_clean_sjc_r_scale_sweep"})
    assert [item.candidate for item in p6p0] == [
        "pixel5phone_3p375_sjc_r0p84375_p6p0",
        "pixel5phone_3p375_sjc_r1p6875_p6p0",
        "pixel5phone_3p375_sjc_r2p53125_p6p0",
    ]


def test_existing_queue_items_can_skip_missing(tmp_path) -> None:
    queue = selected_queue({"p6p0_clean_sjc_r_scale_sweep"})
    path = candidate_submission_path(queue[0].candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")

    assert existing_queue_items(queue, tmp_path, "tag", skip_missing=True) == [queue[0]]
    with pytest.raises(SystemExit, match="missing candidate CSV"):
        existing_queue_items(queue, tmp_path, "tag", skip_missing=False)


def test_kaggle_submit_command() -> None:
    command = kaggle_submit_command(Path("candidate.csv"), "message")

    assert command == [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        "smartphone-decimeter-2023",
        "-f",
        "candidate.csv",
        "-m",
        "message",
    ]


def test_dry_run_shell_quotes_submission_message() -> None:
    completed = subprocess.run(
        [
            "python3",
            "experiments/submit_gsdc2023_pixel5_candidate_queue.py",
            "--group",
            "mtv_sjc_trip_ablation",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    first_line = completed.stdout.splitlines()[0]
    assert "-m '20260501 pixel5 sjcr0 ablate mtv de1 20230523'" in first_line


def test_submit_risk_gate_requires_enabled_clean_report(tmp_path) -> None:
    with pytest.raises(SystemExit, match="missing risk report"):
        assert_submit_risk_gate(tmp_path)

    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": False}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="not enabled"):
        assert_submit_risk_gate(tmp_path)

    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "risky_chunks": 2}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="risky_chunks=2"):
        assert_submit_risk_gate(tmp_path)

    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "risky_chunks": 0}}),
        encoding="utf-8",
    )
    assert assert_submit_risk_gate(tmp_path) == {"enabled": True, "risky_chunks": 0}
    assert risk_report_payload(tmp_path) == {"enabled": True, "risky_chunks": 0}

    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "risky_chunks": 3, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    assert assert_submit_risk_gate(tmp_path)["candidate_actionable_risky_chunks"] == 0


def test_pre_submit_manifest_gate_requires_clean_p6p0_manifest(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    output = tmp_path / candidate / "candidate.csv"
    output.parent.mkdir(parents=True)
    output.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="missing pre-submit manifest"):
        assert_pre_submit_manifest_gate(tmp_path, [candidate])

    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(output),
                        "output_sha256": sha256_file(output),
                        "pixel6pro_scale": 0.0,
                        "risk_candidate_actionable_chunks": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    assert pre_submit_manifest_payload(tmp_path)["risk_report"]["candidate_actionable_risky_chunks"] == 0
    assert assert_pre_submit_manifest_gate(tmp_path, [candidate])["candidates"][0]["candidate"] == candidate

    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,1,0.5",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="pre-submit trip check failed"):
        assert_pre_submit_manifest_gate(tmp_path, [candidate])

    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m,previous_exists,previous_changed_rows,previous_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0,True,2,0.814",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="previous trip check failed"):
        assert_pre_submit_manifest_gate(tmp_path, [candidate])


def test_pre_submit_manifest_gate_rejects_any_candidate_that_moves_previous_safe_trip(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375"
    output = tmp_path / candidate / "candidate.csv"
    output.parent.mkdir(parents=True)
    output.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(output),
                        "output_sha256": sha256_file(output),
                        "pixel6pro_scale": 3.25,
                        "risk_candidate_actionable_chunks": 2,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m,previous_exists,previous_changed_rows,previous_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0,True,1,0.25",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="previous trip check failed"):
        assert_pre_submit_manifest_gate(tmp_path, [candidate])

    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m,previous_exists,previous_changed_rows,previous_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,2,0.75,True,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    assert assert_pre_submit_manifest_gate(tmp_path, [candidate])["candidates"][0]["candidate"] == candidate


def test_matlab_equivalence_gate_can_be_required_from_pre_submit_manifest(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    output = tmp_path / candidate / "candidate.csv"
    output.parent.mkdir(parents=True)
    output.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    manifest = {
        "risk_report": {"candidate_actionable_risky_chunks": 0},
        "candidates": [
            {
                "candidate": candidate,
                "output": str(output),
                "output_sha256": sha256_file(output),
                "pixel6pro_scale": 0.0,
                "risk_candidate_actionable_chunks": 0,
            },
        ],
    }
    (tmp_path / "pre_submit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing matlab_equivalence_gate"):
        assert_pre_submit_manifest_gate(tmp_path, [candidate], require_matlab_equivalence=True)

    manifest["matlab_equivalence_gate"] = {
        "passed": True,
        "equivalence_claim": "matlab_equivalent",
        "factor_mask_passed": True,
        "raw_bridge_counts_passed": True,
        "residual_values_passed": True,
        "residual_total_matlab_only": 0,
        "residual_total_bridge_only": 0,
        "residual_overall_max_abs_delta_m": 5.9e-5,
        "residual_max_abs_delta_threshold_m": 1.0e-4,
        "residual_internal_delta_failure_count": 0,
        "residual_internal_delta_failures": [],
        "residual_internal_delta_thresholds": {"model_delta": 1.0e-4},
    }
    (tmp_path / "pre_submit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        assert_pre_submit_manifest_gate(tmp_path, [candidate], require_matlab_equivalence=True)[
            "matlab_equivalence_gate"
        ]["equivalence_claim"]
        == "matlab_equivalent"
    )


def test_matlab_equivalence_gate_rejects_side_only_or_delta_failures() -> None:
    clean = {
        "matlab_equivalence_gate": {
            "passed": True,
            "equivalence_claim": "matlab_equivalent",
            "factor_mask_passed": True,
            "raw_bridge_counts_passed": True,
            "residual_values_passed": True,
            "residual_total_matlab_only": 0,
            "residual_total_bridge_only": 0,
            "residual_overall_max_abs_delta_m": 5.0e-5,
            "residual_max_abs_delta_threshold_m": 1.0e-4,
            "residual_internal_delta_failure_count": 0,
            "residual_internal_delta_failures": [],
            "residual_internal_delta_thresholds": {"model_delta": 1.0e-4},
        },
    }
    assert assert_matlab_equivalence_gate(clean)["equivalence_claim"] == "matlab_equivalent"

    dirty = json.loads(json.dumps(clean))
    dirty["matlab_equivalence_gate"]["residual_total_bridge_only"] = 1
    with pytest.raises(SystemExit, match="side-only"):
        assert_matlab_equivalence_gate(dirty)

    dirty = json.loads(json.dumps(clean))
    dirty["matlab_equivalence_gate"]["residual_overall_max_abs_delta_m"] = 2.0e-4
    with pytest.raises(SystemExit, match="max delta failed"):
        assert_matlab_equivalence_gate(dirty)

    dirty = json.loads(json.dumps(clean))
    del dirty["matlab_equivalence_gate"]["residual_internal_delta_failure_count"]
    with pytest.raises(SystemExit, match="missing residual internal delta failure count"):
        assert_matlab_equivalence_gate(dirty)

    dirty = json.loads(json.dumps(clean))
    dirty["matlab_equivalence_gate"]["residual_internal_delta_failure_count"] = 2
    with pytest.raises(SystemExit, match="internal delta failures"):
        assert_matlab_equivalence_gate(dirty)

    dirty = json.loads(json.dumps(clean))
    dirty["matlab_equivalence_gate"]["residual_internal_delta_thresholds"] = {}
    with pytest.raises(SystemExit, match="missing residual internal delta thresholds"):
        assert_matlab_equivalence_gate(dirty)


def test_submit_checks_risk_before_running_kaggle(monkeypatch, tmp_path) -> None:
    for item in selected_queue({"sjc_r_scale_sweep"}):
        path = candidate_submission_path(item.candidate, tmp_path, "tag")
        path.parent.mkdir(parents=True)
        path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "risky_chunks": 1}}),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="risky_chunks=1"):
        main(["--output-dir", str(tmp_path), "--tag", "tag", "--group", "sjc_r_scale_sweep", "--submit"])

    assert calls == []

    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--tag",
                "tag",
                "--group",
                "sjc_r_scale_sweep",
                "--submit",
                "--allow-risk",
                "--skip-missing",
            ],
        )
        == 0
    )
    assert len(calls) == 3


def test_check_ready_runs_gates_without_running_kaggle(monkeypatch, capsys, tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "risky_chunks": 3, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(path),
                        "output_sha256": sha256_file(path),
                        "pixel6pro_scale": 0.0,
                        "risk_candidate_actionable_chunks": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--tag",
                "tag",
                "--group",
                "p6p0_clean_sjc_r_scale_sweep",
                "--check-ready",
                "--skip-missing",
            ],
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "ready: 1 candidate(s)" in captured.out
    assert "kaggle competitions submit" in captured.out
    assert calls == []


def test_build_and_write_ready_report(tmp_path) -> None:
    queue = selected_queue({"p6p0_clean_sjc_r_scale_sweep"})[:1]
    path = candidate_submission_path(queue[0].candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    report_path = tmp_path / "ready_report.json"

    report = build_ready_report(
        output_dir=tmp_path,
        tag="tag",
        groups=["p6p0_clean_sjc_r_scale_sweep"],
        queue=queue,
        risk_report={"enabled": True, "candidate_actionable_risky_chunks": 0},
        pre_submit_manifest={"risk_report": {"candidate_actionable_risky_chunks": 0}},
        allow_risk=False,
    )
    write_ready_report(report_path, report)

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((tmp_path / "ready_report.csv").open(encoding="utf-8")))
    assert payload["ready_count"] == 1
    assert payload["groups"] == ["p6p0_clean_sjc_r_scale_sweep"]
    assert payload["risk_report"]["candidate_actionable_risky_chunks"] == 0
    assert payload["pre_submit_manifest"]["present"] is True
    assert payload["candidates"][0]["candidate"] == queue[0].candidate
    assert payload["candidates"][0]["sha256"] == sha256_file(path)
    assert csv_rows[0]["candidate"] == queue[0].candidate
    assert csv_rows[0]["sha256"] == sha256_file(path)
    assert csv_rows[0]["command"].startswith("kaggle competitions submit")


def test_ready_report_consistency_audits_json_csv_manifest_and_sha(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(path),
                        "output_sha256": sha256_file(path),
                        "pixel6pro_scale": 0.0,
                        "risk_candidate_actionable_chunks": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "ready.json"
    report = build_ready_report(
        output_dir=tmp_path,
        tag="tag",
        groups=["p6p0_clean_sjc_r_scale_sweep"],
        queue=selected_queue({"p6p0_clean_sjc_r_scale_sweep"})[:1],
        risk_report={"enabled": True, "candidate_actionable_risky_chunks": 0},
        pre_submit_manifest={"risk_report": {"candidate_actionable_risky_chunks": 0}},
        allow_risk=False,
    )
    write_ready_report(report_path, report)

    assert assert_ready_report_consistency(report_path)["ready_count"] == 1

    csv_path = tmp_path / "ready.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    rows[0]["sha256"] = "bad"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit, match="CSV sha256 mismatch"):
        assert_ready_report_consistency(report_path)


def test_ready_report_consistency_applies_pre_submit_manifest_to_non_p6p0_candidates(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(path),
                        "output_sha256": sha256_file(path),
                        "pixel6pro_scale": 3.25,
                        "risk_candidate_actionable_chunks": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m,previous_exists,previous_changed_rows,previous_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0,True,1,0.25",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "ready.json"
    report = build_ready_report(
        output_dir=tmp_path,
        tag="tag",
        groups=["sjc_r_scale_sweep"],
        queue=selected_queue({"sjc_r_scale_sweep"})[:1],
        risk_report={"enabled": True, "candidate_actionable_risky_chunks": 0},
        pre_submit_manifest={"risk_report": {"candidate_actionable_risky_chunks": 0}},
        allow_risk=False,
    )
    write_ready_report(report_path, report)

    with pytest.raises(SystemExit, match="previous trip check failed"):
        assert_ready_report_consistency(report_path)


def test_audit_ready_report_cli(capsys, tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(path),
                        "output_sha256": sha256_file(path),
                        "pixel6pro_scale": 0.0,
                        "risk_candidate_actionable_chunks": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "ready.json"
    report = build_ready_report(
        output_dir=tmp_path,
        tag="tag",
        groups=["p6p0_clean_sjc_r_scale_sweep"],
        queue=selected_queue({"p6p0_clean_sjc_r_scale_sweep"})[:1],
        risk_report={"enabled": True, "candidate_actionable_risky_chunks": 0},
        pre_submit_manifest={"risk_report": {"candidate_actionable_risky_chunks": 0}},
        allow_risk=False,
    )
    write_ready_report(report_path, report)

    assert main(["--audit-ready-report", str(report_path)]) == 0
    assert "audited: 1 candidate(s)" in capsys.readouterr().out


def test_prepare_ready_report_builds_manifest_report_and_audits(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    risky_trip = "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"
    output_dir = tmp_path / "out"
    candidate_path = candidate_submission_path(candidate, output_dir, "tag")
    base_path = tmp_path / "base.csv"
    build_summary_path = output_dir / "build_summary.json"
    ready_report_path = output_dir / "ready.json"
    rows = "\n".join(
        [
            "tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees",
            f"{risky_trip},1000,37.0,-122.0",
            f"{risky_trip},2000,37.00001,-121.99999",
        ],
    ) + "\n"
    base_path.write_text(rows, encoding="utf-8")
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(rows, encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_summary_path.write_text(
        json.dumps(
            {
                "input": str(base_path),
                "pr_proxy_risk_report": {
                    "enabled": True,
                    "risky_chunks": 3,
                    "candidate_actionable_risky_chunks": 0,
                },
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(candidate_path),
                        "output_sha256": sha256_file(candidate_path),
                        "effective_phone_scales": {"pixel6pro": 0.0},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    report = prepare_ready_report(
        output_dir=output_dir,
        tag="tag",
        groups=["p6p0_clean_sjc_r_scale_sweep"],
        ready_report_path=ready_report_path,
        build_summary_path=build_summary_path,
        risky_trips=(risky_trip,),
        skip_missing=True,
    )

    assert report["ready_count"] == 1
    assert (output_dir / "pre_submit_manifest.json").is_file()
    assert (output_dir / "pre_submit_trip_delta_checks.csv").is_file()
    assert (output_dir / "ready.json").is_file()
    assert (output_dir / "ready.csv").is_file()
    readiness = (output_dir / "submit_readiness.md").read_text(encoding="utf-8")
    assert "prepared: 1 candidate(s)" in readiness
    assert "Max risky Pixel6Pro input changed rows: `0`" in readiness


def test_prepare_ready_report_cli(capsys, tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    risky_trip = "2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro"
    output_dir = tmp_path / "out"
    candidate_path = candidate_submission_path(candidate, output_dir, "tag")
    base_path = tmp_path / "base.csv"
    build_summary_path = output_dir / "build_summary.json"
    rows = "\n".join(
        [
            "tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees",
            f"{risky_trip},1000,37.0,-122.0",
            f"{risky_trip},2000,37.00001,-121.99999",
        ],
    ) + "\n"
    base_path.write_text(rows, encoding="utf-8")
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(rows, encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_summary_path.write_text(
        json.dumps(
            {
                "input": str(base_path),
                "pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(candidate_path),
                        "output_sha256": sha256_file(candidate_path),
                        "effective_phone_scales": {"pixel6pro": 0.0},
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--output-dir",
                str(output_dir),
                "--tag",
                "tag",
                "--group",
                "p6p0_clean_sjc_r_scale_sweep",
                "--prepare-ready-report",
                str(output_dir / "ready.json"),
                "--build-summary",
                str(build_summary_path),
                "--risky-trip",
                risky_trip,
                "--skip-missing",
            ],
        )
        == 0
    )
    assert "prepared: 1 candidate(s)" in capsys.readouterr().out
    assert (output_dir / "submit_readiness.md").is_file()


def test_write_submit_readiness_doc_uses_report_values(tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "risk_report": {"candidate_actionable_risky_chunks": 0},
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "ready.json"
    report = build_ready_report(
        output_dir=tmp_path,
        tag="tag",
        groups=["p6p0_clean_sjc_r_scale_sweep"],
        queue=selected_queue({"p6p0_clean_sjc_r_scale_sweep"})[:1],
        risk_report={"enabled": True, "candidate_actionable_risky_chunks": 0},
        pre_submit_manifest={"risk_report": {"candidate_actionable_risky_chunks": 0}},
        allow_risk=False,
    )
    write_ready_report(report_path, report)

    doc_path = write_submit_readiness_doc(
        output_dir=tmp_path,
        ready_report_path=report_path,
        tag="tag",
        groups=["p6p0_clean_sjc_r_scale_sweep"],
        previous_output_dir=tmp_path / "previous",
        previous_tag="old",
        skip_missing=True,
    )

    doc = doc_path.read_text(encoding="utf-8")
    assert "--prepare-ready-report" in doc
    assert "--previous-tag old" in doc
    assert "Ready candidates: `1`" in doc
    assert candidate in doc


def test_check_ready_can_write_ready_report(monkeypatch, capsys, tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_manifest.json").write_text(
        json.dumps(
            {
                "risk_report": {"candidate_actionable_risky_chunks": 0},
                "candidates": [
                    {
                        "candidate": candidate,
                        "output": str(path),
                        "output_sha256": sha256_file(path),
                        "pixel6pro_scale": 0.0,
                        "risk_candidate_actionable_chunks": 0,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    (tmp_path / "pre_submit_trip_delta_checks.csv").write_text(
        "\n".join(
            [
                "candidate,tripId,rows,input_changed_rows,input_max_m",
                f"{candidate},2023-05-23-22-16-us-ca-mtv-ie2/pixel6pro,2,0,0.0",
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "ready.json"
    monkeypatch.setattr(subprocess, "run", lambda command, check: None)

    assert (
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--tag",
                "tag",
                "--group",
                "p6p0_clean_sjc_r_scale_sweep",
                "--check-ready",
                "--skip-missing",
                "--ready-report",
                str(report_path),
            ],
        )
        == 0
    )
    capsys.readouterr()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    csv_rows = list(csv.DictReader((tmp_path / "reports" / "ready.csv").open(encoding="utf-8")))
    assert payload["ready_count"] == 1
    assert payload["candidates"][0]["command"][:3] == ["kaggle", "competitions", "submit"]
    assert csv_rows[0]["candidate"] == candidate


def test_submit_p6p0_checks_pre_submit_manifest_before_running_kaggle(monkeypatch, tmp_path) -> None:
    candidate = "pixel5phone_3p375_sjc_r0p84375_p6p0"
    path = candidate_submission_path(candidate, tmp_path, "tag")
    path.parent.mkdir(parents=True)
    path.write_text("tripId,UnixTimeMillis,LatitudeDegrees,LongitudeDegrees\n", encoding="utf-8")
    (tmp_path / "build_summary.json").write_text(
        json.dumps({"pr_proxy_risk_report": {"enabled": True, "risky_chunks": 3, "candidate_actionable_risky_chunks": 0}}),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="missing pre-submit manifest"):
        main(
            [
                "--output-dir",
                str(tmp_path),
                "--tag",
                "tag",
                "--group",
                "p6p0_clean_sjc_r_scale_sweep",
                "--submit",
                "--skip-missing",
            ],
        )
    assert calls == []
