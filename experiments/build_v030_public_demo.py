#!/usr/bin/env python3
"""Build the public v0.3 release snapshot from locked Phase 4-6 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def build_snapshot(repo_root: Path) -> dict[str, Any]:
    phase1 = _read(repo_root / "internal_docs/phase1_holdout_detector_audit_2026_07_29.json")
    phase2 = _read(repo_root / "internal_docs/phase2_structural_audit_2026_07_29.json")
    phase3 = _read(repo_root / "internal_docs/phase3_outage_recovery_audit_2026_07_29.json")
    phase4 = _read(repo_root / "internal_docs/phase4_realtime_benchmark_2026_07_29.json")
    wp172_runtime = _read(
        repo_root / "internal_docs/wp172_runtime_benchmark_2026_07_29.json"
    )
    phase5 = _read(repo_root / "internal_docs/phase5_cross_domain_result_2026_07_29.json")
    phase6 = _read(repo_root / "internal_docs/phase6_ros2_replay_result_2026_07_29.json")
    soak = _read(repo_root / "internal_docs/phase6_ros2_soak_result_2026_07_29.json")
    promotion = _read(
        repo_root / "internal_docs/v030_production_promotion_audit_2026_07_29.json"
    )
    tokyo = _read(
        repo_root / "internal_docs/wp173_tokyo_operational_audit_2026_07_29.json"
    )
    promotion_gates = {gate["id"]: gate for gate in promotion["gates"]}
    tokyo_gate = promotion_gates["tokyo_sub50cm_target"]
    gain_gate = promotion_gates["full_denominator_gain_without_loss"]
    return {
        "schema": "gnss_gpu_v030_public_snapshot_v1",
        "version": "0.3.0",
        "headline": "46.51% Tokyo sub-50 cm, 10.87% guarded LAMBDA FIX, false-FIX zero",
        "phases": [
            {"id": 0, "name": "Evaluation contract", "status": "complete"},
            {
                "id": 1,
                "name": "Truth-free Evidence API",
                "status": "complete",
                "metric": f"{phase1['rejected_holdouts']}/4 negative controls rejected",
            },
            {
                "id": 2,
                "name": "Affine DDPR + arc screen",
                "status": "complete",
                "metric": (
                    f"{phase2['wp163']['recovered_reference_ranks']}/3 ranks recovered; "
                    f"{phase2['wp164']['false_passing_hypotheses']} false passes"
                ),
            },
            {
                "id": 3,
                "name": "Multi-hypothesis outage recovery",
                "status": "complete",
                "metric": f"{phase3['recovery_epochs']} evidence epochs",
            },
            {
                "id": 4,
                "name": "Persistent real-time GPU",
                "status": "complete",
                "metric": (
                    f"{phase4['assessment']['normal_latency_max_ms']:.2f} ms normal / "
                    f"{phase4['assessment']['search_latency_max_ms']:.2f} ms search"
                ),
            },
            {
                "id": 5,
                "name": "Cross-domain validation",
                "status": "complete",
                "metric": (
                    f"{len(phase5['coverage']['cities'])} cities / "
                    f"{len(phase5['coverage']['receivers'])} receivers"
                ),
            },
            {
                "id": 6,
                "name": "ROS 2 lifecycle safety",
                "status": "complete",
                "metric": (
                    f"{phase6['event_count']}-event replay + "
                    f"{soak['simulated_duration_s'] / 3600:.0f} h soak"
                ),
            },
            {"id": 7, "name": "v0.3 release", "status": "complete"},
        ],
        "runtime": {
            "gpu": phase4["gpu"],
            **phase4["assessment"],
            "wp172_candidate_supply": wp172_runtime["measurement"],
        },
        "coverage": phase5["coverage"],
        "positioning": phase5["campaigns"][0]["weighted"],
        "replay": {
            "event_count": phase6["event_count"],
            "restart_count": phase6["restart_count"],
            "dispositions": phase6["dispositions"],
            "sha256": phase6["replay_sha256"],
            "timeline": [
                {
                    "index": step["event_index"],
                    "sensor": step["sensor"],
                    "disposition": step["disposition"],
                    "mode": step["mode"],
                }
                for step in phase6["steps"]
            ],
        },
        "negative_controls": phase1["results"],
        "promotion": {
            "allowed": promotion["promotion_allowed"],
            "passed_gates": promotion["passed_gate_count"],
            "gate_count": promotion["gate_count"],
            "failed_gates": promotion["failed_gates"],
            "tokyo_sub50cm_percent": tokyo_gate["actual"],
            "tokyo_target_percent": tokyo_gate["expected"],
            "tokyo_sub50cm_epochs": tokyo["after_sub50cm_epochs"],
            "tokyo_total_epochs": tokyo["full_denominator_epochs"],
            "tokyo_required_epochs": 5366,
            "tokyo_epoch_margin": tokyo["after_sub50cm_epochs"] - 5366,
            **gain_gate["actual"],
            "false_fix": promotion_gates["false_fix_zero"]["actual"],
            "lambda_fix_epochs": tokyo["fix_epochs"],
            "lambda_fix_percent": tokyo["fix_percent"],
        },
        "soak": {
            "passed": soak["passed"],
            "simulated_duration_s": soak["simulated_duration_s"],
            "ticks": soak["ticks"],
            "accepted_events": soak["dispositions"]["accepted"],
            "watchdog_trips": soak["final"]["counters"]["watchdog_trips"],
            "normal_recoveries": soak["normal_recoveries"],
            "final_mode": soak["final"]["navigation_mode"],
            "sha256": soak["state_digest_sha256"],
        },
        "limitations": [
            (
                "Tokyo sub-50 cm is 46.5112% (5,546/11,924); guarded LAMBDA "
                "declares FIX on 1,296 epochs (10.8688%) with false-FIX zero."
            ),
            (
                "Tokyo run1 was inspected by earlier campaign diagnostics; "
                "WP172 is an operational audit, not a virgin scientific holdout."
            ),
            "The Phase 3 outage audit is synthetic; city-scale evidence is reported separately.",
            "Hong Kong is locked from a tracked summary because raw data is not bundled.",
            "Windows GPU memory is a conservative capacity estimate, not nvidia-smi peak usage.",
        ],
    }


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>gnss_gpu v0.3 — release audit</title>
  <link rel="stylesheet" href="site.css">
  <style>
    :root { color-scheme: dark; --ink:#eaf4ff; --muted:#9db0c7; --panel:#101a29;
      --line:#263b55; --cyan:#47d7ff; --green:#57e39b; --amber:#ffcb6b; }
    html { overflow-anchor:none; }
    body { margin:0; background:#07101b; color:var(--ink); font:16px/1.55 system-ui,sans-serif; }
    main { width:min(1120px,calc(100% - 32px)); margin:auto; padding:48px 0 80px; }
    a { color:var(--cyan); } .eyebrow { color:var(--cyan); letter-spacing:.14em; text-transform:uppercase; }
    h1 { font-size:clamp(2.4rem,7vw,5.6rem); line-height:.95; margin:.2em 0; }
    .lede { max-width:780px; color:var(--muted); font-size:1.2rem; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin:32px 0; }
    .card,.phase,.event,.limit { background:var(--panel); border:1px solid var(--line);
      border-radius:14px; padding:18px; }
    .metric { color:var(--green); font-size:1.8rem; font-weight:750; }
    .label,.detail { color:var(--muted); } .phases { display:grid; gap:10px; }
    .phase { display:grid; grid-template-columns:48px 1fr auto; align-items:center; gap:12px; }
    .phase b:first-child { color:var(--cyan); font-size:1.3rem; }
    .status { color:var(--green); } .timeline { display:flex; gap:8px; overflow:auto; padding-bottom:8px; }
    .event { min-width:140px; } .event.safe_fallback { border-color:var(--amber); }
    code { overflow-wrap:anywhere; } ul { padding-left:22px; }
    footer { color:var(--muted); margin-top:48px; }
  </style>
</head>
<body><main>
  <p class="eyebrow">Audited release · v<span id="version"></span></p>
  <h1>Urban navigation,<br>safely bounded.</h1>
  <p class="lede" id="headline"></p>
  <p><a href="./">← Main results</a> · <a href="https://github.com/rsasaki0109/gnss_gpu/releases/tag/v0.3.0">Release assets</a></p>
  <section class="grid" id="metrics"></section>
  <h2>Production promotion</h2>
  <section class="grid" id="promotion"></section>
  <h2>Phase 0 → 7</h2><section class="phases" id="phases"></section>
  <h2>Deterministic anomaly replay</h2>
  <p class="detail">Recorded arrival order; integrity faults latch safe fallback until a valid newer event or restart.</p>
  <section class="timeline" id="timeline"></section>
  <p>Replay SHA-256: <code id="replay-hash"></code></p>
  <h2>Mandatory negative controls</h2><section class="grid" id="holdouts"></section>
  <h2>Limits, not footnotes</h2><ul id="limits"></ul>
  <footer>Generated from checked-in Phase 1–6 and fail-closed promotion artifacts. No live telemetry and no post-load metric rewriting.</footer>
</main>
<script>
fetch("assets/data/v030_release_snapshot.json").then(r => r.json()).then(d => {
  document.getElementById("version").textContent=d.version;
  document.getElementById("headline").textContent=d.headline;
  const metricRows=[
    [d.runtime.normal_latency_max_ms.toFixed(2)+" ms","normal max latency"],
    [d.runtime.search_latency_max_ms.toFixed(2)+" ms","search max latency"],
    [d.coverage.cities.length,"cities"],
    [d.coverage.receivers.length,"receiver families"],
    [d.positioning.baseline.toFixed(3)+" → "+d.positioning.candidate.toFixed(3)+" m","weighted positioning RMS"],
    [d.runtime.deadline_misses,"deadline misses"]
  ];
  metricRows.forEach(([v,l])=>metricsEl.insertAdjacentHTML("beforeend",`<article class="card"><div class="metric">${v}</div><div class="label">${l}</div></article>`));
  const promotionEl=document.getElementById("promotion");
  const promotionRows=[
    [d.promotion.passed_gates+"/"+d.promotion.gate_count,"promotion gates pass"],
    [d.promotion.tokyo_sub50cm_percent.toFixed(2)+"% / "+d.promotion.tokyo_target_percent.toFixed(0)+"%","Tokyo sub-50 cm"],
    [d.promotion.gained_epochs+" / "+d.promotion.lost_epochs,"gained / lost epochs"],
    [d.promotion.lambda_fix_percent.toFixed(2)+"%","guarded LAMBDA FIX"],
    [d.promotion.false_fix,"false FIX"],
    [(d.soak.simulated_duration_s/3600).toFixed(0)+" h","ROS 2 continuity soak"],
    [d.soak.final_mode,"final navigation mode"]
  ];
  promotionRows.forEach(([v,l])=>promotionEl.insertAdjacentHTML("beforeend",`<article class="card"><div class="metric">${v}</div><div class="label">${l}</div></article>`));
  const phasesEl=document.getElementById("phases");
  d.phases.forEach(p=>phasesEl.insertAdjacentHTML("beforeend",`<article class="phase"><b>${p.id}</b><div><b>${p.name}</b><div class="detail">${p.metric||""}</div></div><span class="status">${p.status}</span></article>`));
  const timelineEl=document.getElementById("timeline");
  d.replay.timeline.forEach(e=>timelineEl.insertAdjacentHTML("beforeend",`<article class="event ${e.mode}"><b>#${e.index} ${e.sensor}</b><div>${e.disposition}</div><small>${e.mode}</small></article>`));
  document.getElementById("replay-hash").textContent=d.replay.sha256;
  const holdoutsEl=document.getElementById("holdouts");
  d.negative_controls.forEach(h=>holdoutsEl.insertAdjacentHTML("beforeend",`<article class="card"><b>${h.holdout_id}</b><div class="metric">REJECT</div><div class="detail">${h.reason}</div></article>`));
  const limitsEl=document.getElementById("limits");
  d.limitations.forEach(x=>limitsEl.insertAdjacentHTML("beforeend",`<li class="limit">${x}</li>`));
}).catch(e=>document.body.textContent="Snapshot load failed: "+e);
const metricsEl=document.getElementById("metrics");
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    snapshot_path = root / "docs/assets/data/v030_release_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(build_snapshot(root), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "docs/v0.3.0.html").write_text(_html(), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
