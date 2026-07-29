#!/usr/bin/env python3
"""Build and verify the deterministic gnss_gpu v0.3 reproducibility bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from gnss_gpu.evaluation_contract import (  # noqa: E402
    MANDATORY_NEGATIVE_HOLDOUTS,
    verify_locked_contract,
)


VERSION = "0.3.0"
BUNDLE_SCHEMA = "gnss_gpu_reproducibility_bundle_v1"
MANIFEST_SCHEMA = "gnss_gpu_release_manifest_v1"

CORE_EVIDENCE = (
    "internal_docs/urban_navigation_phase0_evaluation_contract.md",
    "internal_docs/urban_navigation_phase1_evidence_api.md",
    "internal_docs/phase1_holdout_detector_audit_2026_07_29.json",
    "internal_docs/urban_navigation_phase2_ddpr_profiles.md",
    "internal_docs/phase2_structural_audit_2026_07_29.json",
    "internal_docs/urban_navigation_phase3_multihypothesis.md",
    "internal_docs/phase3_outage_recovery_audit_2026_07_29.json",
    "internal_docs/urban_navigation_phase4_realtime.md",
    "internal_docs/phase4_realtime_benchmark_2026_07_29.json",
    "internal_docs/urban_navigation_phase5_cross_domain.md",
    "internal_docs/phase5_cross_domain_input_2026_07_29.json",
    "internal_docs/phase5_cross_domain_result_2026_07_29.json",
    "internal_docs/urban_navigation_phase6_ros2.md",
    "internal_docs/phase6_ros2_replay_input_2026_07_29.json",
    "internal_docs/phase6_ros2_replay_result_2026_07_29.json",
    "internal_docs/phase6_ros2_soak_result_2026_07_29.json",
    "configs/evaluation/v030_production_promotion.json",
    "internal_docs/wp30_m4_production_config.json",
    "internal_docs/wp30_m4_tokyo_evidence_ledger.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(repo_root: Path, relative: str) -> dict[str, Any]:
    value = json.loads((repo_root / relative).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return value


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assert_passed(name: str, value: Mapping[str, Any]) -> None:
    if value.get("passed") is not True:
        raise ValueError(f"{name} is not a passing locked artifact")


def _benchmark(
    phase4: Mapping[str, Any],
    phase5: Mapping[str, Any],
    phase6: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "gnss_gpu_v030_benchmark_summary_v1",
        "runtime": {
            "gpu": phase4["gpu"],
            "normal_particles": phase4["normal_particles"],
            "search_particles": phase4["search_particles"],
            **phase4["assessment"],
        },
        "cross_domain": {
            "coverage": phase5["coverage"],
            "campaigns": [
                {
                    "id": item["id"],
                    "primary_metric": item["primary_metric"],
                    "weighted": item["weighted"],
                    "passed": item["passed"],
                }
                for item in phase5["campaigns"]
            ],
        },
        "ros2_replay": {
            "event_count": phase6["event_count"],
            "restart_count": phase6["restart_count"],
            "dispositions": phase6["dispositions"],
            "replay_sha256": phase6["replay_sha256"],
        },
    }


def _ablation(
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase5: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "gnss_gpu_v030_ablation_summary_v1",
        "entries": [
            {
                "component": "truth_free_unsafe_acceptance_detector",
                "baseline": "minimum 2 historical rejections",
                "candidate": f"{phase1['rejected_holdouts']} historical rejections",
                "safety": "no accepted mandatory negative control",
            },
            {
                "component": "affine_ddpr_profile_and_arc_screen_v2",
                "baseline": "constant profile misses WP163 offset shape",
                "candidate": (
                    f"{phase2['wp163']['recovered_reference_ranks']} of 3 "
                    "reference ranks recovered"
                ),
                "safety": (
                    f"{phase2['wp164']['false_passing_hypotheses']} false WP164 passes"
                ),
            },
            {
                "component": "multi_hypothesis_outage_recovery",
                "baseline": f"{phase3['legacy_greedy_error_m']} m greedy error",
                "candidate": f"{phase3['multihypothesis_error_m']} m retained-branch error",
                "safety": (
                    f"FIX suppressed during outage; recovery in "
                    f"{phase3['recovery_epochs']} evidence epochs"
                ),
            },
            {
                "component": "cross_domain_safe_adaptive_policy",
                "baseline": phase5["campaigns"][0]["weighted"]["baseline"],
                "candidate": phase5["campaigns"][0]["weighted"]["candidate"],
                "safety": "all held-out primary and catastrophic metrics non-degraded",
            },
        ],
    }


def _failure_gallery(repo_root: Path) -> dict[str, Any]:
    entries = []
    for spec in MANDATORY_NEGATIVE_HOLDOUTS:
        entries.append(
            {
                "id": spec.holdout_id,
                "city": spec.city,
                "dataset": spec.dataset,
                "segment": list(spec.segment),
                "failure_category": spec.failure_category.value,
                "required_disposition": spec.expected_disposition,
                "evidence_path": spec.lock_path,
                "evidence_sha256": _sha256(repo_root / spec.lock_path),
                "lesson": {
                    "nagoya_wp53": "abstain when independent evidence supply is missing",
                    "tokyo_wp129": "reject an ambiguous basin identity",
                    "tokyo_wp156": "reject an unopposed zero-gain candidate",
                    "tokyo_wp168": "screening alone cannot justify an unsafe acceptance",
                }[spec.holdout_id],
            }
        )
    return {
        "schema": "gnss_gpu_v030_failure_gallery_v1",
        "entries": entries,
    }


def _report(
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
    phase3: Mapping[str, Any],
    phase4: Mapping[str, Any],
    phase5: Mapping[str, Any],
    phase6: Mapping[str, Any],
) -> str:
    runtime = phase4["assessment"]
    positioning = phase5["campaigns"][0]["weighted"]
    return f"""# gnss_gpu v{VERSION} technical report

## Scope

v0.3 turns the urban-navigation research workspace into a guarded GNSS/IMU/map/GPU
platform with immutable negative controls, truth-free acceptance evidence,
multi-hypothesis outage recovery, enforced real-time budgets, cross-domain
validation, and a ROS 2 lifecycle safety boundary.

## Audited results

- Evidence detector: {phase1["rejected_holdouts"]}/4 mandatory historical negative
  controls rejected.
- DDPR structure: {phase2["wp163"]["recovered_reference_ranks"]}/3 WP163 reference
  ranks recovered; {phase2["wp164"]["false_passing_hypotheses"]} WP164 false passes.
- Outage recovery: {phase3["recovery_epochs"]} evidence epochs; greedy
  {phase3["legacy_greedy_error_m"]:.1f} m versus retained-hypothesis
  {phase3["multihypothesis_error_m"]:.1f} m.
- GTX 1660 Ti runtime: normal maximum {runtime["normal_latency_max_ms"]:.3f} ms,
  search maximum {runtime["search_latency_max_ms"]:.3f} ms, recorded capacity
  {runtime["peak_gpu_memory_mb"]:.3f} MiB.
- Cross-domain positioning: epoch-weighted RMS {positioning["baseline"]:.3f} m to
  {positioning["candidate"]:.3f} m, with Tokyo non-degradation and Hong Kong gain.
- Coverage: {len(phase5["coverage"]["cities"])} cities,
  {len(phase5["coverage"]["sites"])} sites/routes,
  {len(phase5["coverage"]["receivers"])} receivers, and
  {len(phase5["coverage"]["dates"])} dates.
- ROS replay: {phase6["event_count"]} events, {phase6["restart_count"]} restart,
  canonical hash `{phase6["replay_sha256"]}`.

## Safety and promotion

Promotion remains fail closed. It requires truth-free production input, positive
full-denominator gain with zero loss, zero false FIX, all mandatory negative
holdouts rejected or abstained as declared, exact M4 preservation, multi-city
non-degradation, reproducible input/config hashes, and the Phase 4 latency/memory
limits.

## Limitations

- The final Tokyo sub-50 cm target of 45% is a program target, not demonstrated by
  the Phase 5 UrbanNav RMS campaign; no release claim upgrades it to achieved.
- Phase 3 city-scale accuracy is supported indirectly by later campaigns; its
  locked outage controller audit is synthetic.
- Hong Kong is reproduced from a tracked result summary because raw data is not
  in this checkout.
- The ROS lifecycle package is unit/replay tested here; real `rclpy`/colcon
  validation is performed by the ROS container build.
- The Windows Phase 4 memory value is a conservative capacity estimate because
  per-process `nvidia-smi` memory was unavailable.
"""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _prepare_output(output: Path) -> None:
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output exists and is not a directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)


def build_bundle(repo_root: Path, output: Path) -> dict[str, Any]:
    locked = verify_locked_contract(repo_root)
    _assert_passed("immutable contract", locked)
    phase1 = _read_json(
        repo_root, "internal_docs/phase1_holdout_detector_audit_2026_07_29.json"
    )
    phase2 = _read_json(repo_root, "internal_docs/phase2_structural_audit_2026_07_29.json")
    phase3 = _read_json(
        repo_root, "internal_docs/phase3_outage_recovery_audit_2026_07_29.json"
    )
    phase4 = _read_json(
        repo_root, "internal_docs/phase4_realtime_benchmark_2026_07_29.json"
    )
    phase5 = _read_json(
        repo_root, "internal_docs/phase5_cross_domain_result_2026_07_29.json"
    )
    phase6 = _read_json(
        repo_root, "internal_docs/phase6_ros2_replay_result_2026_07_29.json"
    )
    for name, artifact in (
        ("phase1", phase1),
        ("phase2", phase2),
        ("phase3", phase3),
        ("phase4", phase4),
        ("phase5", phase5),
    ):
        _assert_passed(name, artifact)

    _prepare_output(output)
    evidence_paths = list(CORE_EVIDENCE)
    evidence_paths.extend(spec.lock_path for spec in MANDATORY_NEGATIVE_HOLDOUTS)
    evidence_paths = sorted(set(evidence_paths))
    for relative in evidence_paths:
        source = repo_root / relative
        if not source.is_file():
            raise FileNotFoundError(relative)
        destination = output / "evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    _write_json(output / "benchmark.json", _benchmark(phase4, phase5, phase6))
    _write_json(output / "ablation.json", _ablation(phase1, phase2, phase3, phase5))
    _write_json(output / "failure_gallery.json", _failure_gallery(repo_root))
    (output / "TECHNICAL_REPORT.md").write_text(
        _report(phase1, phase2, phase3, phase4, phase5, phase6),
        encoding="utf-8",
        newline="\n",
    )

    tracked_status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    files = []
    for path in sorted(
        output.rglob("*"),
        key=lambda item: item.relative_to(output).as_posix(),
    ):
        if path.is_file() and path.name != "MANIFEST.json":
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "bundle_schema": BUNDLE_SCHEMA,
        "version": VERSION,
        "git_commit": _git(repo_root, "rev-parse", "HEAD"),
        "tracked_worktree_clean": tracked_status == "",
        "python": sys.version.split()[0],
        "entrypoint": (
            "python tools/build_release_bundle.py --output dist/reproducibility "
            "--archive dist/gnss_gpu-v0.3.0-reproducibility.zip"
        ),
        "files": files,
    }
    _write_json(output / "MANIFEST.json", manifest)
    return manifest


def verify_bundle(output: Path) -> dict[str, Any]:
    manifest_path = output / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for item in manifest.get("files", []):
        path = output / item["path"]
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "passed": actual == item["sha256"],
            }
        )
    passed = (
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("version") == VERSION
        and bool(checks)
        and all(item["passed"] for item in checks)
    )
    return {"passed": passed, "file_count": len(checks), "checks": checks}


def write_deterministic_zip(output: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(
            output.rglob("*"),
            key=lambda item: item.relative_to(output).as_posix(),
        ):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(
                f"gnss_gpu-v{VERSION}/{path.relative_to(output).as_posix()}",
                date_time=(2026, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            bundle.writestr(info, path.read_bytes(), compresslevel=9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if args.verify is not None:
        result = verify_bundle(args.verify.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 2
    if args.output is None:
        parser.error("--output is required when not using --verify")
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    manifest = build_bundle(repo_root, output)
    verification = verify_bundle(output)
    if not verification["passed"]:
        return 2
    if args.archive is not None:
        write_deterministic_zip(output, args.archive.resolve())
    print(
        json.dumps(
            {
                "manifest": manifest,
                "verification": verification,
                "archive": str(args.archive.resolve()) if args.archive else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
