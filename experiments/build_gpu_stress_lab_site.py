#!/usr/bin/env python3
"""Build the static GPU GNSS Stress Lab docs page."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
ASSETS_DIR = DOCS_DIR / "assets"
DATA_DIR = ASSETS_DIR / "data"
SNAPSHOT_JSON = ASSETS_DIR / "gpu_stress_lab_snapshot.json"
SNAPSHOT_JS = ASSETS_DIR / "gpu_stress_lab_snapshot.js"
PAGE_PATH = DOCS_DIR / "gpu_gnss_stress_lab.html"

RF_SUMMARY_CSV = RESULTS_DIR / "gnss_security_lab" / "gnss_security_lab_summary.csv"
RF_SUMMARY_JSON = RESULTS_DIR / "gnss_security_lab" / "gnss_security_lab_summary.json"
URBAN_SUMMARY_JSON = RESULTS_DIR / "urban_shadow_lab" / "urban_shadow_summary.json"
URBAN_EPOCH_CSV = RESULTS_DIR / "urban_shadow_lab" / "urban_shadow_epoch_summary.csv"
SWEEP_SUMMARY_JSON = RESULTS_DIR / "gpu_scenario_sweeper" / "gpu_scenario_sweep_summary.json"
SWEEP_SUMMARY_CSV = RESULTS_DIR / "gpu_scenario_sweeper" / "gpu_scenario_sweep_summary.csv"
GPU_FEATURE_CSV = RESULTS_DIR / "gpu_stress_selector_features.csv"
PPC_SHADOW_CSV = RESULTS_DIR / "ppc_gpu_urban_shadow_features_tokyo_run1_nav_smoke.csv"
PPC_SHADOW_SELECTOR_CSV = RESULTS_DIR / "ppc_gpu_urban_shadow_selector_features_tokyo_run1_nav_smoke.csv"
GPU_SHADOW_PROBE_SUMMARY_JSON = RESULTS_DIR / "gpu_shadow_selector_probe" / "gpu_shadow_selector_probe_summary.json"
GPU_SHADOW_PROBE_ROWS_CSV = RESULTS_DIR / "gpu_shadow_selector_probe" / "gpu_shadow_selector_probe_rows.csv"
GPU_SHADOW_PROBE_BUCKET_CSV = RESULTS_DIR / "gpu_shadow_selector_probe" / "gpu_shadow_selector_probe_by_risk_bucket.csv"
GPU_SHADOW_REAL_PROBE_DIR = RESULTS_DIR / "gpu_shadow_selector_probe_real_candidates_tokyo_run1"
GPU_SHADOW_REAL_PROBE_SUMMARY_JSON = GPU_SHADOW_REAL_PROBE_DIR / "gpu_shadow_selector_probe_summary.json"
GPU_SHADOW_REAL_PROBE_ROWS_CSV = GPU_SHADOW_REAL_PROBE_DIR / "gpu_shadow_selector_probe_rows.csv"
GPU_SHADOW_REAL_PROBE_BUCKET_CSV = GPU_SHADOW_REAL_PROBE_DIR / "gpu_shadow_selector_probe_by_risk_bucket.csv"
GPU_SHADOW_REAL_SWEEP_DIR = RESULTS_DIR / "gpu_shadow_selector_probe_real_candidates_tokyo_run1_sweep"
GPU_SHADOW_REAL_SWEEP_SUMMARY_JSON = GPU_SHADOW_REAL_SWEEP_DIR / "gpu_shadow_selector_probe_weight_sweep_summary.json"
GPU_SHADOW_REAL_SWEEP_CSV = GPU_SHADOW_REAL_SWEEP_DIR / "gpu_shadow_selector_probe_weight_sweep.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_data(path: Path, *, dst_name: str | None = None) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / (dst_name or path.name)
    shutil.copy2(path, dst)
    return str(dst.relative_to(DOCS_DIR)).replace("\\", "/")


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _b(row: dict, key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes"}


def _summarize_rf(rows: list[dict[str, str]]) -> dict:
    clean = next((row for row in rows if row.get("scenario") == "clean"), rows[0])
    false_locks = [row for row in rows if _b(row, "false_lock")]
    detected = [row for row in rows if _b(row, "interference_detected")]
    lowest_snr = min(rows, key=lambda row: _f(row, "target_snr"))
    return {
        "n_scenarios": len(rows),
        "backend": clean.get("acquisition_backend", "unknown"),
        "clean_snr": _f(clean, "target_snr"),
        "false_lock_count": len(false_locks),
        "interference_detected_count": len(detected),
        "lowest_snr_scenario": lowest_snr.get("scenario", ""),
        "lowest_snr": _f(lowest_snr, "target_snr"),
        "rows": rows,
    }


def _summarize_features(rows: list[dict[str, str]]) -> dict:
    labels: dict[str, int] = {}
    for row in rows:
        label = row.get("gpu_failure_label", "unknown")
        labels[label] = labels.get(label, 0) + 1
    return {
        "n_rows": len(rows),
        "n_features": max(0, len(rows[0]) - 6) if rows else 0,
        "target_bad": int(sum(_f(row, "gpu_stress_target_bad") for row in rows)),
        "label_counts": labels,
    }


def _summarize_ppc_shadow(rows: list[dict[str, str]], selector_rows: list[dict[str, str]]) -> dict:
    if not rows:
        return {
            "n_epochs": 0,
            "satellite_source": "unknown",
            "backend": "unknown",
            "n_sat": 0,
            "mean_blocked_ratio": 0.0,
            "max_blocked_ratio": 0.0,
            "mean_shadow_risk": 0.0,
            "max_shadow_risk": 0.0,
        }
    risks = [_f(row, "gpu_urban_shadow_risk_score") for row in selector_rows] if selector_rows else [0.0]
    return {
        "n_epochs": len(rows),
        "satellite_source": rows[0].get("gpu_urban_satellite_source", "unknown"),
        "backend": rows[0].get("gpu_urban_backend", "unknown"),
        "n_sat": int(_f(rows[0], "gpu_urban_n_sat")),
        "mean_blocked_ratio": sum(_f(row, "gpu_urban_mean_blocked_ratio") for row in rows) / len(rows),
        "max_blocked_ratio": max(_f(row, "gpu_urban_max_blocked_ratio") for row in rows),
        "mean_shadow_risk": sum(risks) / len(risks),
        "max_shadow_risk": max(risks),
    }


def _build_snapshot() -> dict:
    rf_rows = _read_csv(RF_SUMMARY_CSV)
    urban = _read_json(URBAN_SUMMARY_JSON)
    sweep = _read_json(SWEEP_SUMMARY_JSON)
    features = _read_csv(GPU_FEATURE_CSV)
    copied = {
        "rf_csv": _copy_data(RF_SUMMARY_CSV),
        "rf_json": _copy_data(RF_SUMMARY_JSON),
        "urban_csv": _copy_data(URBAN_EPOCH_CSV),
        "urban_json": _copy_data(URBAN_SUMMARY_JSON),
        "sweep_csv": _copy_data(SWEEP_SUMMARY_CSV),
        "sweep_json": _copy_data(SWEEP_SUMMARY_JSON),
        "features_csv": _copy_data(GPU_FEATURE_CSV),
    }
    ppc_shadow = None
    if PPC_SHADOW_CSV.exists() and PPC_SHADOW_SELECTOR_CSV.exists():
        ppc_rows = _read_csv(PPC_SHADOW_CSV)
        ppc_selector_rows = _read_csv(PPC_SHADOW_SELECTOR_CSV)
        ppc_shadow = _summarize_ppc_shadow(ppc_rows, ppc_selector_rows)
        copied["ppc_shadow_csv"] = _copy_data(PPC_SHADOW_CSV)
        copied["ppc_shadow_selector_csv"] = _copy_data(PPC_SHADOW_SELECTOR_CSV)
    selector_probe = None
    if (
        GPU_SHADOW_PROBE_SUMMARY_JSON.exists()
        and GPU_SHADOW_PROBE_ROWS_CSV.exists()
        and GPU_SHADOW_PROBE_BUCKET_CSV.exists()
    ):
        selector_probe = _read_json(GPU_SHADOW_PROBE_SUMMARY_JSON)
        copied["shadow_probe_rows_csv"] = _copy_data(GPU_SHADOW_PROBE_ROWS_CSV)
        copied["shadow_probe_bucket_csv"] = _copy_data(GPU_SHADOW_PROBE_BUCKET_CSV)
        copied["shadow_probe_summary_json"] = _copy_data(GPU_SHADOW_PROBE_SUMMARY_JSON)
    real_selector_probe = None
    if (
        GPU_SHADOW_REAL_PROBE_SUMMARY_JSON.exists()
        and GPU_SHADOW_REAL_PROBE_ROWS_CSV.exists()
        and GPU_SHADOW_REAL_PROBE_BUCKET_CSV.exists()
    ):
        real_selector_probe = _read_json(GPU_SHADOW_REAL_PROBE_SUMMARY_JSON)
        copied["real_shadow_probe_rows_csv"] = _copy_data(
            GPU_SHADOW_REAL_PROBE_ROWS_CSV,
            dst_name="real_gpu_shadow_selector_probe_rows.csv",
        )
        copied["real_shadow_probe_bucket_csv"] = _copy_data(
            GPU_SHADOW_REAL_PROBE_BUCKET_CSV,
            dst_name="real_gpu_shadow_selector_probe_by_risk_bucket.csv",
        )
        copied["real_shadow_probe_summary_json"] = _copy_data(
            GPU_SHADOW_REAL_PROBE_SUMMARY_JSON,
            dst_name="real_gpu_shadow_selector_probe_summary.json",
        )
    real_selector_sweep = None
    if GPU_SHADOW_REAL_SWEEP_SUMMARY_JSON.exists() and GPU_SHADOW_REAL_SWEEP_CSV.exists():
        real_selector_sweep = _read_json(GPU_SHADOW_REAL_SWEEP_SUMMARY_JSON)
        copied["real_shadow_probe_sweep_csv"] = _copy_data(
            GPU_SHADOW_REAL_SWEEP_CSV,
            dst_name="real_gpu_shadow_selector_probe_weight_sweep.csv",
        )
        copied["real_shadow_probe_sweep_summary_json"] = _copy_data(
            GPU_SHADOW_REAL_SWEEP_SUMMARY_JSON,
            dst_name="real_gpu_shadow_selector_probe_weight_sweep_summary.json",
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": "GPU GNSS Stress Lab",
        "subtitle": (
            "CUDA-backed RF acquisition stress, BVH urban shadow casting, "
            "scenario sweeps, and selector-ready GPU risk features."
        ),
        "rf": _summarize_rf(rf_rows),
        "urban": urban["summary"],
        "sweep": sweep["summary"],
        "features": _summarize_features(features),
        "ppc_shadow": ppc_shadow,
        "selector_probe": selector_probe,
        "real_selector_probe": real_selector_probe,
        "real_selector_sweep": real_selector_sweep,
        "artifacts": copied,
    }


def _write_snapshot(snapshot: dict) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_JSON.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    SNAPSHOT_JS.write_text(
        "window.GPU_STRESS_LAB_SNAPSHOT = "
        + json.dumps(snapshot, indent=2)
        + ";\n",
        encoding="utf-8",
    )


def _write_page() -> None:
    PAGE_PATH.write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GPU GNSS Stress Lab</title>
    <meta
      name="description"
      content="CUDA RF stress, BVH urban shadow, scenario sweeps, and selector feature artifacts for gnss_gpu."
    />
    <script src="assets/gpu_stress_lab_snapshot.js"></script>
    <style>
      :root {
        --bg: #f5f7f4;
        --panel: #ffffff;
        --ink: #172126;
        --muted: #5b676d;
        --line: #d9e0df;
        --teal: #0f766e;
        --rust: #9f4f2f;
        --gold: #bf8c32;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background: var(--bg);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .shell {
        width: min(1180px, calc(100vw - 32px));
        margin: 0 auto;
        padding: 28px 0 54px;
      }
      header {
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(260px, 0.75fr);
        gap: 16px;
        align-items: stretch;
      }
      .hero, .status, .card, .panel, .link-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
      }
      .hero, .status {
        padding: 24px;
      }
      .eyebrow, .label, .meta, .link-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--muted);
      }
      h1, h2, h3, p { margin: 0; }
      h1 {
        margin-top: 8px;
        font-size: clamp(2.3rem, 5vw, 4.7rem);
        line-height: 0.96;
        letter-spacing: 0;
      }
      h2 {
        font-size: 1.35rem;
      }
      h3 {
        font-size: 1rem;
      }
      .subtitle {
        margin-top: 14px;
        max-width: 64ch;
        color: var(--muted);
        line-height: 1.5;
      }
      .status {
        background: #16302f;
        color: #eef8f6;
      }
      .status .label, .status .meta {
        color: rgba(238, 248, 246, 0.72);
      }
      .status strong {
        display: block;
        margin-top: 10px;
        font-size: 2rem;
        line-height: 1;
      }
      main {
        display: grid;
        gap: 22px;
        margin-top: 22px;
      }
      .metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
      }
      .card, .panel, .link-card {
        padding: 18px;
      }
      .value {
        margin: 10px 0 8px;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1;
      }
      .detail {
        color: var(--muted);
        line-height: 1.45;
      }
      .two-col {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 14px;
      }
      .panel-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
        margin-bottom: 12px;
      }
      .bar-row {
        display: grid;
        grid-template-columns: 118px 1fr 86px;
        gap: 8px;
        align-items: center;
        margin: 9px 0;
        font-size: 0.9rem;
      }
      .bar-track {
        height: 10px;
        border-radius: 999px;
        background: #e8eeee;
        overflow: hidden;
      }
      .bar-fill {
        height: 100%;
        border-radius: 999px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
      }
      th, td {
        padding: 8px 9px;
        border-bottom: 1px solid var(--line);
        text-align: left;
        vertical-align: top;
      }
      th {
        background: #edf2f0;
        color: #4d5a60;
        font-size: 0.75rem;
        text-transform: uppercase;
      }
      .heat {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
      }
      .heat-cell {
        min-height: 70px;
        border-radius: 8px;
        padding: 9px;
        color: white;
        text-shadow: 0 1px 1px rgba(0, 0, 0, 0.36);
      }
      .heat-cell strong {
        display: block;
        font-size: 1.2rem;
      }
      .links {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
      }
      a {
        color: var(--teal);
        text-decoration: none;
        font-weight: 700;
      }
      a:hover { color: var(--rust); }
      code {
        background: #edf1ef;
        border-radius: 4px;
        padding: 2px 5px;
      }
      @media (max-width: 860px) {
        header, .two-col, .metrics, .links {
          grid-template-columns: 1fr;
        }
        .heat {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header>
        <section class="hero">
          <p class="eyebrow">gnss_gpu</p>
          <h1 id="title">GPU GNSS Stress Lab</h1>
          <p id="subtitle" class="subtitle"></p>
        </section>
        <aside class="status">
          <p class="label">Current Sweep</p>
          <strong id="sweep-score"></strong>
          <p id="sweep-detail" class="meta"></p>
        </aside>
      </header>
      <main>
        <section class="metrics" id="metrics"></section>
        <section class="two-col">
          <article class="panel">
            <div class="panel-head">
              <h2>RF Stress</h2>
              <p class="meta" id="rf-backend"></p>
            </div>
            <div id="rf-bars"></div>
          </article>
          <article class="panel">
            <div class="panel-head">
              <h2>Urban Shadow</h2>
              <p class="meta" id="urban-backend"></p>
            </div>
            <div id="urban-bars"></div>
          </article>
        </section>
        <section class="two-col">
          <article class="panel">
            <div class="panel-head">
              <h2>Failure Mix</h2>
              <p class="meta">Phase 3 scenario sweep</p>
            </div>
            <div id="failure-heat" class="heat"></div>
          </article>
          <article class="panel">
            <div class="panel-head">
              <h2>Worst Case</h2>
              <p class="meta">highest risk row</p>
            </div>
            <table>
              <tbody id="worst-case"></tbody>
            </table>
          </article>
        </section>
        <section>
          <div class="panel-head">
            <h2>Artifacts</h2>
            <p class="meta">Copied into docs/assets/data</p>
          </div>
          <div class="links" id="artifact-links"></div>
        </section>
      </main>
    </div>
    <script>
      const snapshot = window.GPU_STRESS_LAB_SNAPSHOT;

      function pct(v, digits = 1) {
        return `${(Number(v) * 100).toFixed(digits)}%`;
      }

      function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
      }

      function metric(label, value, detail) {
        const card = el("article", "card");
        card.append(el("p", "label", label));
        card.append(el("p", "value", value));
        card.append(el("p", "detail", detail));
        return card;
      }

      function bar(label, value, maxValue, text, color) {
        const row = el("div", "bar-row");
        row.append(el("span", "", label));
        const track = el("div", "bar-track");
        const fill = el("div", "bar-fill");
        fill.style.width = `${Math.max(0, Math.min(100, (Number(value) / Number(maxValue || 1)) * 100))}%`;
        fill.style.background = color;
        track.append(fill);
        row.append(track);
        row.append(el("strong", "", text));
        return row;
      }

      function heatColor(score) {
        const s = Math.max(0, Math.min(1, Number(score)));
        const hue = 120 - 120 * s;
        return `hsl(${hue} 58% 42%)`;
      }

      function render() {
        document.getElementById("title").textContent = snapshot.title;
        document.getElementById("subtitle").textContent = snapshot.subtitle;
        document.getElementById("sweep-score").textContent = `${snapshot.sweep.n_rows} rows`;
        document.getElementById("sweep-detail").textContent =
          `mean risk ${snapshot.sweep.mean_risk.toFixed(3)}, max risk ${snapshot.sweep.max_risk.toFixed(3)}`;

        const metrics = document.getElementById("metrics");
        metrics.append(metric("RF scenarios", snapshot.rf.n_scenarios, `${snapshot.rf.false_lock_count} false lock cases`));
        metrics.append(metric("Urban rays", snapshot.urban.particle_rays.toLocaleString(), `${snapshot.urban.particle_los_ms.toFixed(2)} ms particle BVH`));
        metrics.append(metric("Selector rows", snapshot.features.n_rows, `${snapshot.features.n_features} gpu feature columns`));
        metrics.append(metric("Bad targets", snapshot.features.target_bad, "synthetic stress labels for training probes"));
        if (snapshot.ppc_shadow) {
          metrics.append(metric("PPC nav epochs", snapshot.ppc_shadow.n_epochs, `${snapshot.ppc_shadow.n_sat} ${snapshot.ppc_shadow.satellite_source} satellites`));
        }
        if (snapshot.selector_probe) {
          metrics.append(metric("Probe flips", snapshot.selector_probe.changed_epochs, `${pct(snapshot.selector_probe.change_rate)} selector change rate`));
        }
        if (snapshot.real_selector_probe) {
          metrics.append(metric("Real flips", snapshot.real_selector_probe.changed_epochs, `${snapshot.real_selector_probe.source_rows.toLocaleString()} joined candidate rows`));
        }

        document.getElementById("rf-backend").textContent = snapshot.rf.backend;
        const rfBars = document.getElementById("rf-bars");
        rfBars.append(bar("clean SNR", snapshot.rf.clean_snr, snapshot.rf.clean_snr, snapshot.rf.clean_snr.toFixed(1), "var(--teal)"));
        rfBars.append(bar("lowest SNR", snapshot.rf.lowest_snr, snapshot.rf.clean_snr, snapshot.rf.lowest_snr.toFixed(1), "var(--rust)"));
        rfBars.append(bar("interference", snapshot.rf.interference_detected_count, snapshot.rf.n_scenarios, `${snapshot.rf.interference_detected_count}/${snapshot.rf.n_scenarios}`, "var(--gold)"));
        rfBars.append(bar("false lock", snapshot.rf.false_lock_count, snapshot.rf.n_scenarios, `${snapshot.rf.false_lock_count}/${snapshot.rf.n_scenarios}`, "var(--rust)"));

        document.getElementById("urban-backend").textContent = snapshot.urban.particle_backend;
        const urbanBars = document.getElementById("urban-bars");
        urbanBars.append(bar("mean blocked", snapshot.urban.mean_blocked_ratio, 1, pct(snapshot.urban.mean_blocked_ratio), "var(--teal)"));
        urbanBars.append(bar("max blocked", snapshot.urban.max_blocked_ratio, 1, pct(snapshot.urban.max_blocked_ratio), "var(--rust)"));
        urbanBars.append(bar("shadow contrast", snapshot.urban.mean_particle_shadow_contrast, 0.4, snapshot.urban.mean_particle_shadow_contrast.toFixed(3), "var(--gold)"));
        urbanBars.append(bar("route BVH", snapshot.urban.route_los_ms, 2, `${snapshot.urban.route_los_ms.toFixed(2)} ms`, "var(--teal)"));
        if (snapshot.ppc_shadow) {
          urbanBars.append(bar("PPC blocked", snapshot.ppc_shadow.mean_blocked_ratio, 1, pct(snapshot.ppc_shadow.mean_blocked_ratio), "var(--rust)"));
          urbanBars.append(bar("PPC risk", snapshot.ppc_shadow.mean_shadow_risk, 1, snapshot.ppc_shadow.mean_shadow_risk.toFixed(3), "var(--gold)"));
        }
        if (snapshot.selector_probe) {
          urbanBars.append(bar("probe flips", snapshot.selector_probe.change_rate, 1, pct(snapshot.selector_probe.change_rate), "var(--rust)"));
          urbanBars.append(bar("proxy gain", Math.abs(snapshot.selector_probe.mean_gpu_minus_baseline_truth_cost), 0.05, snapshot.selector_probe.mean_gpu_minus_baseline_truth_cost.toFixed(4), "var(--teal)"));
        }
        if (snapshot.real_selector_probe) {
          urbanBars.append(bar("real flips", snapshot.real_selector_probe.change_rate, 1, pct(snapshot.real_selector_probe.change_rate), "var(--rust)"));
          urbanBars.append(bar("real delta", Math.abs(snapshot.real_selector_probe.mean_gpu_minus_baseline_truth_cost), 0.01, snapshot.real_selector_probe.mean_gpu_minus_baseline_truth_cost.toFixed(6), "var(--teal)"));
        }
        if (snapshot.real_selector_sweep && snapshot.real_selector_sweep.best) {
          urbanBars.append(bar("best sweep", Math.abs(snapshot.real_selector_sweep.best.mean_gpu_minus_baseline_truth_cost), 0.01, snapshot.real_selector_sweep.best.mean_gpu_minus_baseline_truth_cost.toFixed(6), "var(--gold)"));
        }

        const failureHeat = document.getElementById("failure-heat");
        Object.entries(snapshot.sweep.label_counts).forEach(([label, count]) => {
          const cell = el("div", "heat-cell");
          const ratio = count / snapshot.sweep.n_rows;
          cell.style.background = heatColor(ratio);
          cell.append(el("strong", "", String(count)));
          cell.append(el("span", "", label));
          failureHeat.append(cell);
        });

        const worst = snapshot.sweep.worst;
        const worstRows = [
          ["label", worst.failure_label],
          ["risk", worst.total_risk_score.toFixed(3)],
          ["JNR", `${worst.jammer_jnr_db} dB`],
          ["spoof delay", `${worst.spoof_delay_samples} samples`],
          ["code error", `${worst.code_phase_error_samples} samples`],
          ["blocked", pct(worst.mean_blocked_ratio)],
          ["backend", `${worst.acquisition_backend} / ${worst.particle_backend}`],
        ];
        const worstBody = document.getElementById("worst-case");
        worstRows.forEach(([k, v]) => {
          const tr = document.createElement("tr");
          tr.append(el("th", "", k));
          tr.append(el("td", "", v));
          worstBody.append(tr);
        });

        const labels = {
          rf_csv: "RF CSV",
          rf_json: "RF JSON",
          urban_csv: "Urban CSV",
          urban_json: "Urban JSON",
          sweep_csv: "Sweep CSV",
          sweep_json: "Sweep JSON",
          features_csv: "Features CSV",
          ppc_shadow_csv: "PPC Shadow CSV",
          ppc_shadow_selector_csv: "PPC Selector CSV",
          shadow_probe_rows_csv: "Probe Rows CSV",
          shadow_probe_bucket_csv: "Probe Buckets CSV",
          shadow_probe_summary_json: "Probe Summary JSON",
          real_shadow_probe_rows_csv: "Real Probe Rows CSV",
          real_shadow_probe_bucket_csv: "Real Probe Buckets CSV",
          real_shadow_probe_summary_json: "Real Probe Summary JSON",
          real_shadow_probe_sweep_csv: "Real Sweep CSV",
          real_shadow_probe_sweep_summary_json: "Real Sweep JSON",
        };
        const links = document.getElementById("artifact-links");
        Object.entries(snapshot.artifacts).forEach(([key, href]) => {
          const card = el("article", "link-card");
          card.append(el("p", "link-label", labels[key] || key));
          const a = document.createElement("a");
          a.href = href;
          a.textContent = href.split("/").pop();
          card.append(a);
          links.append(card);
        });
      }

      render();
    </script>
  </body>
</html>
""",
        encoding="utf-8",
    )


def build() -> dict:
    snapshot = _build_snapshot()
    _write_snapshot(snapshot)
    _write_page()
    return snapshot


def main() -> int:
    snapshot = build()
    print(f"[gpu-stress-site] wrote {PAGE_PATH}")
    print(f"[gpu-stress-site] wrote {SNAPSHOT_JSON}")
    print(f"[gpu-stress-site] rows={snapshot['sweep']['n_rows']} features={snapshot['features']['n_features']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
