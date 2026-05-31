#!/usr/bin/env python3
"""GPU GNSS scenario sweeper.

Phase 3 MVP: combine RF stress conditions with urban shadow conditions and emit
a sweep table that is ready for heatmaps and selector-feature experiments.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def _load_neighbor(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RF = _load_neighbor("exp_gnss_security_lab")
URBAN = _load_neighbor("exp_urban_shadow_lab")


@dataclass(frozen=True)
class RFStressMetrics:
    jammer_jnr_db: float
    spoof_delay_samples: int
    acquisition_backend: str
    interference_backend: str
    acquisition_ms: float
    interference_ms: float
    target_acquired: bool
    target_snr: float
    code_phase_error_samples: float
    doppler_error_hz: float
    false_lock: bool
    interference_detected: bool
    interference_kind: str
    best_prn: int
    max_false_prn_snr: float


@dataclass(frozen=True)
class UrbanStressMetrics:
    building_height_scale: float
    particles_per_epoch: int
    route_backend: str
    particle_backend: str
    route_los_ms: float
    particle_los_ms: float
    mean_blocked_ratio: float
    max_blocked_ratio: float
    mean_particle_shadow_contrast: float
    worst_epoch: int


@dataclass(frozen=True)
class SweepRow:
    jammer_jnr_db: float
    spoof_delay_samples: int
    building_height_scale: float
    particles_per_epoch: int
    acquisition_backend: str
    interference_backend: str
    route_backend: str
    particle_backend: str
    target_acquired: bool
    target_snr: float
    code_phase_error_samples: float
    doppler_error_hz: float
    false_lock: bool
    acquisition_miss: bool
    interference_detected: bool
    interference_kind: str
    max_false_prn_snr: float
    mean_blocked_ratio: float
    max_blocked_ratio: float
    mean_particle_shadow_contrast: float
    worst_epoch: int
    acquisition_ms: float
    interference_ms: float
    route_los_ms: float
    particle_los_ms: float
    total_risk_score: float
    failure_label: str


def _parse_float_list(text: str) -> tuple[float, ...]:
    vals = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not vals:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return vals


def _parse_int_list(text: str) -> tuple[int, ...]:
    vals = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not vals:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return vals


def _rf_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        cpu_only=args.cpu_only,
        prn=args.prn,
        prn_search=args.prn_search,
        sampling_freq=args.sampling_freq,
        n_samples=max(1, int(round(args.sampling_freq * args.duration_ms * 1e-3))),
        code_phase_samples=args.code_phase_samples,
        doppler_hz=args.doppler_hz,
        snr_db=args.snr_db,
        spoof_doppler_offset_hz=args.spoof_doppler_offset_hz,
        spoof_jsr_db=args.spoof_jsr_db,
        doppler_range_hz=args.doppler_range_hz,
        doppler_step_hz=args.doppler_step_hz,
        acquisition_threshold=args.acquisition_threshold,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        interference_threshold_db=args.interference_threshold_db,
        false_lock_code_error_samples=args.false_lock_code_error_samples,
        false_lock_doppler_error_hz=args.false_lock_doppler_error_hz,
    )


def _rf_stress_signal(
    *,
    args: argparse.Namespace,
    jammer_jnr_db: float,
    spoof_delay_samples: int,
    seed: int,
) -> np.ndarray:
    rf_args = _rf_args(args)
    rng = np.random.default_rng(seed)
    signal = RF._synth_gps_ca_signal(
        prn=rf_args.prn,
        code_phase_samples=rf_args.code_phase_samples,
        doppler_hz=rf_args.doppler_hz,
        sampling_freq=rf_args.sampling_freq,
        n_samples=rf_args.n_samples,
        snr_db=rf_args.snr_db,
        rng=rng,
    )
    if jammer_jnr_db > 0.0:
        signal = RF._inject_wideband_jammer(signal, jsr_db=jammer_jnr_db, rng=rng)
    if spoof_delay_samples != 0:
        spoof = RF._synth_gps_ca_signal(
            prn=rf_args.prn,
            code_phase_samples=rf_args.code_phase_samples + int(spoof_delay_samples),
            doppler_hz=rf_args.doppler_hz + rf_args.spoof_doppler_offset_hz,
            sampling_freq=rf_args.sampling_freq,
            n_samples=rf_args.n_samples,
            snr_db=rf_args.snr_db + rf_args.spoof_jsr_db,
            rng=rng,
            amplitude=10.0 ** (rf_args.spoof_jsr_db / 20.0),
            carrier_phase_rad=0.7,
        )
        signal = (signal.astype(np.float64) + spoof.astype(np.float64)).astype(np.float32)
    return signal


def _run_rf_condition(
    *,
    args: argparse.Namespace,
    jammer_jnr_db: float,
    spoof_delay_samples: int,
) -> RFStressMetrics:
    rf_args = _rf_args(args)
    signal = _rf_stress_signal(
        args=args,
        jammer_jnr_db=jammer_jnr_db,
        spoof_delay_samples=spoof_delay_samples,
        seed=args.seed + int(round(jammer_jnr_db * 10.0)) * 1000 + int(spoof_delay_samples),
    )
    scenario = RF.Scenario(
        name=f"jnr{jammer_jnr_db:g}_delay{spoof_delay_samples}",
        attack_type="sweep",
        signal=signal,
        notes="Wideband jammer plus optional coherent delayed replica.",
    )
    metrics = RF._metrics_for_scenario(scenario, rf_args)
    return RFStressMetrics(
        jammer_jnr_db=float(jammer_jnr_db),
        spoof_delay_samples=int(spoof_delay_samples),
        acquisition_backend=metrics.acquisition_backend,
        interference_backend=metrics.interference_backend,
        acquisition_ms=float(metrics.acquisition_ms),
        interference_ms=float(metrics.interference_ms),
        target_acquired=bool(metrics.target_acquired),
        target_snr=float(metrics.target_snr),
        code_phase_error_samples=float(metrics.target_code_phase_error_samples),
        doppler_error_hz=float(metrics.target_doppler_error_hz),
        false_lock=bool(metrics.false_lock),
        interference_detected=bool(metrics.interference_detected),
        interference_kind=str(metrics.interference_kind),
        best_prn=int(metrics.best_prn),
        max_false_prn_snr=float(metrics.max_false_prn_snr),
    )


def _run_urban_condition(
    *,
    args: argparse.Namespace,
    building_height_scale: float,
    particles_per_epoch: int,
) -> UrbanStressMetrics:
    triangles, buildings = URBAN._build_canyon_mesh(
        length_m=args.length_m,
        block_depth_m=args.block_depth_m,
        road_half_width_m=args.road_half_width_m,
        building_width_m=args.building_width_m,
        base_height_m=args.base_height_m * building_height_scale,
        height_wave_m=args.height_wave_m * building_height_scale,
        n_blocks_per_side=args.n_blocks_per_side,
    )
    specs = URBAN._satellite_specs()
    directions = URBAN._satellite_directions(specs)
    route = URBAN._make_route(args.n_epochs, args.length_m, args.rx_height_m)
    sat_route = URBAN._sat_positions_for_receivers(route, directions, range_m=args.sat_range_m)
    route_backend, route_ms, route_los = URBAN._run_los_batch(
        route, sat_route, triangles, cpu_only=args.cpu_only
    )

    particles = URBAN._make_particles(
        route, particles_per_epoch=particles_per_epoch, seed=args.seed + int(building_height_scale * 100)
    )
    sat_particles = URBAN._sat_positions_for_receivers(
        particles, directions, range_m=args.sat_range_m
    )
    particle_backend, particle_ms, particle_los = URBAN._run_los_batch(
        particles, sat_particles, triangles, cpu_only=args.cpu_only
    )
    rows = URBAN._epoch_metrics(
        route=route,
        route_los=route_los,
        particle_los=particle_los,
        specs=specs,
        particles_per_epoch=particles_per_epoch,
        route_backend=route_backend,
        particle_backend=particle_backend,
    )
    summary = URBAN._summary(
        rows,
        route_backend=route_backend,
        particle_backend=particle_backend,
        n_sat=len(specs),
        n_triangles=triangles.shape[0],
        n_buildings=len(buildings),
        route_los_ms=route_ms,
        particle_los_ms=particle_ms,
        particles_per_epoch=particles_per_epoch,
    )
    return UrbanStressMetrics(
        building_height_scale=float(building_height_scale),
        particles_per_epoch=int(particles_per_epoch),
        route_backend=summary.route_backend,
        particle_backend=summary.particle_backend,
        route_los_ms=float(summary.route_los_ms),
        particle_los_ms=float(summary.particle_los_ms),
        mean_blocked_ratio=float(summary.mean_blocked_ratio),
        max_blocked_ratio=float(summary.max_blocked_ratio),
        mean_particle_shadow_contrast=float(summary.mean_particle_shadow_contrast),
        worst_epoch=int(summary.worst_epoch),
    )


def _risk_score(rf: RFStressMetrics, urban: UrbanStressMetrics) -> tuple[float, str]:
    miss = not rf.target_acquired
    false_lock = rf.false_lock
    snr_term = max(0.0, min(1.0, (8.0 - rf.target_snr) / 8.0))
    code_term = max(0.0, min(1.0, abs(rf.code_phase_error_samples) / 160.0))
    rf_term = (
        (0.35 if miss else 0.0)
        + (0.38 if false_lock else 0.0)
        + 0.15 * snr_term
        + 0.10 * code_term
        + (0.08 if rf.interference_detected else 0.0)
    )
    urban_term = (
        0.30 * max(0.0, min(1.0, urban.mean_blocked_ratio))
        + 0.15 * max(0.0, min(1.0, urban.max_blocked_ratio))
        + 0.10 * max(0.0, min(1.0, urban.mean_particle_shadow_contrast * 3.0))
    )
    score = min(1.0, rf_term + urban_term)
    if miss:
        label = "acquisition_miss"
    elif false_lock:
        label = "false_lock"
    elif score >= 0.72:
        label = "high_risk"
    elif score >= 0.48:
        label = "degraded"
    else:
        label = "nominal"
    return float(score), label


def run(args: argparse.Namespace) -> list[SweepRow]:
    if args.prn not in args.prn_search:
        args.prn_search = tuple([args.prn, *args.prn_search])

    rf_cache: dict[tuple[float, int], RFStressMetrics] = {}
    urban_cache: dict[tuple[float, int], UrbanStressMetrics] = {}

    for jnr in args.jammer_jnr_db:
        for delay in args.spoof_delay_samples:
            rf_cache[(float(jnr), int(delay))] = _run_rf_condition(
                args=args, jammer_jnr_db=float(jnr), spoof_delay_samples=int(delay)
            )

    for scale in args.building_height_scale:
        for particles in args.particles_per_epoch:
            urban_cache[(float(scale), int(particles))] = _run_urban_condition(
                args=args,
                building_height_scale=float(scale),
                particles_per_epoch=int(particles),
            )

    rows: list[SweepRow] = []
    for jnr in args.jammer_jnr_db:
        for delay in args.spoof_delay_samples:
            rf = rf_cache[(float(jnr), int(delay))]
            for scale in args.building_height_scale:
                for particles in args.particles_per_epoch:
                    urban = urban_cache[(float(scale), int(particles))]
                    score, label = _risk_score(rf, urban)
                    rows.append(
                        SweepRow(
                            jammer_jnr_db=rf.jammer_jnr_db,
                            spoof_delay_samples=rf.spoof_delay_samples,
                            building_height_scale=urban.building_height_scale,
                            particles_per_epoch=urban.particles_per_epoch,
                            acquisition_backend=rf.acquisition_backend,
                            interference_backend=rf.interference_backend,
                            route_backend=urban.route_backend,
                            particle_backend=urban.particle_backend,
                            target_acquired=rf.target_acquired,
                            target_snr=rf.target_snr,
                            code_phase_error_samples=rf.code_phase_error_samples,
                            doppler_error_hz=rf.doppler_error_hz,
                            false_lock=rf.false_lock,
                            acquisition_miss=not rf.target_acquired,
                            interference_detected=rf.interference_detected,
                            interference_kind=rf.interference_kind,
                            max_false_prn_snr=rf.max_false_prn_snr,
                            mean_blocked_ratio=urban.mean_blocked_ratio,
                            max_blocked_ratio=urban.max_blocked_ratio,
                            mean_particle_shadow_contrast=urban.mean_particle_shadow_contrast,
                            worst_epoch=urban.worst_epoch,
                            acquisition_ms=rf.acquisition_ms,
                            interference_ms=rf.interference_ms,
                            route_los_ms=urban.route_los_ms,
                            particle_los_ms=urban.particle_los_ms,
                            total_risk_score=score,
                            failure_label=label,
                        )
                    )
    return rows


def _write_csv(path: Path, rows: list[SweepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _summarize(rows: list[SweepRow], elapsed_s: float) -> dict:
    labels: dict[str, int] = {}
    for row in rows:
        labels[row.failure_label] = labels.get(row.failure_label, 0) + 1
    worst = max(rows, key=lambda row: row.total_risk_score)
    return {
        "n_rows": len(rows),
        "elapsed_s": elapsed_s,
        "label_counts": labels,
        "worst": asdict(worst),
        "backends": {
            "acquisition": sorted({row.acquisition_backend for row in rows}),
            "interference": sorted({row.interference_backend for row in rows}),
            "route": sorted({row.route_backend for row in rows}),
            "particle": sorted({row.particle_backend for row in rows}),
        },
        "mean_risk": float(np.mean([row.total_risk_score for row in rows])),
        "max_risk": float(worst.total_risk_score),
    }


def _write_json(path: Path, rows: list[SweepRow], summary: dict) -> None:
    payload = {
        "experiment": "gpu_gnss_scenario_sweeper_phase3_mvp",
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _risk_color(score: float) -> str:
    score = max(0.0, min(1.0, score))
    hue = 120.0 - 120.0 * score
    return f"hsl({hue:.0f} 58% 46%)"


def _write_html(path: Path, rows: list[SweepRow], summary: dict) -> None:
    baseline_scale = sorted({row.building_height_scale for row in rows})[len({row.building_height_scale for row in rows}) // 2]
    baseline_particles = min({row.particles_per_epoch for row in rows})
    heat_rows = [
        row
        for row in rows
        if row.building_height_scale == baseline_scale and row.particles_per_epoch == baseline_particles
    ]
    jnrs = sorted({row.jammer_jnr_db for row in heat_rows})
    delays = sorted({row.spoof_delay_samples for row in heat_rows})
    by_key = {(row.jammer_jnr_db, row.spoof_delay_samples): row for row in heat_rows}

    heat_body = []
    for jnr in jnrs:
        cells = [f"<th>{jnr:g} dB</th>"]
        for delay in delays:
            row = by_key[(jnr, delay)]
            color = _risk_color(row.total_risk_score)
            cells.append(
                f'<td style="background:{color}" title="{html.escape(row.failure_label)}">'
                f"{row.total_risk_score:.2f}<br><small>{html.escape(row.failure_label)}</small></td>"
            )
        heat_body.append("<tr>" + "".join(cells) + "</tr>")

    worst = sorted(rows, key=lambda row: row.total_risk_score, reverse=True)[:12]
    worst_rows = "".join(
        "<tr>"
        f"<td>{row.total_risk_score:.2f}</td>"
        f"<td>{html.escape(row.failure_label)}</td>"
        f"<td>{row.jammer_jnr_db:g}</td>"
        f"<td>{row.spoof_delay_samples}</td>"
        f"<td>{row.building_height_scale:g}</td>"
        f"<td>{row.particles_per_epoch}</td>"
        f"<td>{row.target_snr:.2f}</td>"
        f"<td>{row.code_phase_error_samples:+.1f}</td>"
        f"<td>{row.mean_blocked_ratio:.2f}</td>"
        "</tr>"
        for row in worst
    )
    label_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in sorted(summary["label_counts"].items())
    )
    delay_headers = "".join(f"<th>{delay}</th>" for delay in delays)

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GPU GNSS Scenario Sweeper</title>
  <style>
    :root {{
      --ink: #172126;
      --muted: #5c6870;
      --line: #d8dee1;
      --bg: #f5f6f2;
      --panel: #fff;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 32px 6vw 20px;
      background: #eaf0ed;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
    }}
    .sub {{
      max-width: 980px;
      margin: 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    main {{
      padding: 24px 6vw 42px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .stat, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .stat strong {{
      display: block;
      margin-top: 6px;
      font-size: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(300px, 0.95fr) minmax(320px, 1.05fr);
      gap: 14px;
      align-items: start;
    }}
    @media (max-width: 900px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #edf1ee;
      color: #4e5a60;
      font-size: 12px;
      text-transform: uppercase;
    }}
    .heat td {{
      color: white;
      font-weight: 700;
      text-shadow: 0 1px 1px rgba(0,0,0,.35);
      min-width: 74px;
      text-align: center;
    }}
    small {{
      font-weight: 600;
    }}
    .stack {{
      display: grid;
      gap: 14px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>GPU GNSS Scenario Sweeper</h1>
    <p class="sub">
      Phase 3 MVP: cross product of RF jammer/spoof stress and urban BVH shadow stress.
      Heatmap fixes building height scale {baseline_scale:g} and {baseline_particles} particles/epoch.
    </p>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><span>Rows</span><strong>{summary['n_rows']}</strong></div>
      <div class="stat"><span>Mean Risk</span><strong>{summary['mean_risk']:.2f}</strong></div>
      <div class="stat"><span>Max Risk</span><strong>{summary['max_risk']:.2f}</strong></div>
      <div class="stat"><span>Elapsed</span><strong>{summary['elapsed_s']:.1f}s</strong></div>
    </section>
    <section class="grid">
      <article class="panel">
        <h2>RF Risk Heatmap</h2>
        <table class="heat">
          <thead><tr><th>JNR \\ Delay</th>{delay_headers}</tr></thead>
          <tbody>{''.join(heat_body)}</tbody>
        </table>
      </article>
      <div class="stack">
        <article class="panel">
          <h2>Worst Cases</h2>
          <table>
            <thead><tr><th>Risk</th><th>Label</th><th>JNR</th><th>Delay</th><th>Height</th><th>Particles</th><th>SNR</th><th>Code Err</th><th>Blocked</th></tr></thead>
            <tbody>{worst_rows}</tbody>
          </table>
        </article>
        <article class="panel">
          <h2>Label Counts</h2>
          <table>
            <thead><tr><th>Label</th><th>Count</th></tr></thead>
            <tbody>{label_rows}</tbody>
          </table>
        </article>
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/gpu_scenario_sweeper"))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260526)

    parser.add_argument("--jammer-jnr-db", type=_parse_float_list, default=(0.0, 8.0, 16.0, 24.0))
    parser.add_argument("--spoof-delay-samples", type=_parse_int_list, default=(0, 48, 112, 192))
    parser.add_argument("--building-height-scale", type=_parse_float_list, default=(0.7, 1.0, 1.4))
    parser.add_argument("--particles-per-epoch", type=_parse_int_list, default=(16, 64))

    parser.add_argument("--prn", type=int, default=7)
    parser.add_argument("--prn-search", type=RF._parse_prn_search, default=RF.DEFAULT_PRN_SEARCH)
    parser.add_argument("--sampling-freq", type=float, default=4.092e6)
    parser.add_argument("--duration-ms", type=float, default=1.0)
    parser.add_argument("--code-phase-samples", type=int, default=96)
    parser.add_argument("--doppler-hz", type=float, default=1500.0)
    parser.add_argument("--snr-db", type=float, default=18.0)
    parser.add_argument("--spoof-doppler-offset-hz", type=float, default=0.0)
    parser.add_argument("--spoof-jsr-db", type=float, default=6.0)
    parser.add_argument("--doppler-range-hz", type=float, default=5000.0)
    parser.add_argument("--doppler-step-hz", type=float, default=500.0)
    parser.add_argument("--acquisition-threshold", type=float, default=2.2)
    parser.add_argument("--fft-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=128)
    parser.add_argument("--interference-threshold-db", type=float, default=13.0)
    parser.add_argument("--false-lock-code-error-samples", type=float, default=24.0)
    parser.add_argument("--false-lock-doppler-error-hz", type=float, default=750.0)

    parser.add_argument("--n-epochs", type=int, default=36)
    parser.add_argument("--length-m", type=float, default=360.0)
    parser.add_argument("--block-depth-m", type=float, default=44.0)
    parser.add_argument("--road-half-width-m", type=float, default=13.0)
    parser.add_argument("--building-width-m", type=float, default=26.0)
    parser.add_argument("--base-height-m", type=float, default=34.0)
    parser.add_argument("--height-wave-m", type=float, default=42.0)
    parser.add_argument("--n-blocks-per-side", type=int, default=7)
    parser.add_argument("--rx-height-m", type=float, default=1.6)
    parser.add_argument("--sat-range-m", type=float, default=20_200_000.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    start = time.perf_counter()
    rows = run(args)
    elapsed_s = time.perf_counter() - start
    summary = _summarize(rows, elapsed_s)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "gpu_scenario_sweep_summary.csv", rows)
    _write_json(args.out_dir / "gpu_scenario_sweep_summary.json", rows, summary)
    _write_html(args.out_dir / "gpu_scenario_sweep_report.html", rows, summary)

    print(f"[gpu-scenario-sweeper] wrote {args.out_dir / 'gpu_scenario_sweep_summary.csv'}")
    print(f"[gpu-scenario-sweeper] wrote {args.out_dir / 'gpu_scenario_sweep_summary.json'}")
    print(f"[gpu-scenario-sweeper] wrote {args.out_dir / 'gpu_scenario_sweep_report.html'}")
    print(
        "[gpu-scenario-sweeper] "
        f"rows={summary['n_rows']} mean_risk={summary['mean_risk']:.3f} "
        f"max_risk={summary['max_risk']:.3f} labels={summary['label_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
