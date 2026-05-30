#!/usr/bin/env python3
"""GPU GNSS RF stress lab.

This experiment generates a small GPS L1 C/A stress suite and runs acquisition
plus interference detection on each scenario.  CUDA-backed gnss_gpu bindings are
used when available; otherwise a deterministic NumPy fallback keeps the report
reproducible on machines without a built GPU wheel.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CA_CHIP_RATE_HZ = 1.023e6
CA_CODE_LENGTH = 1023
DEFAULT_PRN_SEARCH = (7, 8, 11, 19)

G2_TAPS = {
    1: (2, 6),
    2: (3, 7),
    3: (4, 8),
    4: (5, 9),
    5: (1, 9),
    6: (2, 10),
    7: (1, 8),
    8: (2, 9),
    9: (3, 10),
    10: (2, 3),
    11: (3, 4),
    12: (5, 6),
    13: (6, 7),
    14: (7, 8),
    15: (8, 9),
    16: (9, 10),
    17: (1, 4),
    18: (2, 5),
    19: (3, 6),
    20: (4, 7),
    21: (5, 8),
    22: (6, 9),
    23: (1, 3),
    24: (4, 6),
    25: (5, 7),
    26: (6, 8),
    27: (7, 9),
    28: (8, 10),
    29: (1, 6),
    30: (2, 7),
    31: (3, 8),
    32: (4, 9),
}


@dataclass(frozen=True)
class Scenario:
    name: str
    attack_type: str
    signal: np.ndarray
    notes: str


@dataclass(frozen=True)
class AcquisitionHit:
    prn: int
    acquired: bool
    code_phase: float
    doppler_hz: float
    snr: float


@dataclass(frozen=True)
class ScenarioMetrics:
    scenario: str
    attack_type: str
    notes: str
    acquisition_backend: str
    interference_backend: str
    n_samples: int
    rms: float
    crest_factor: float
    acquisition_ms: float
    interference_ms: float
    target_acquired: bool
    target_snr: float
    target_code_phase_samples: float
    target_code_phase_error_samples: float
    target_doppler_hz: float
    target_doppler_error_hz: float
    best_prn: int
    best_snr: float
    max_false_prn_snr: float
    false_lock: bool
    interference_detected: bool
    interference_kind: str
    interference_center_freq_hz: float
    interference_bandwidth_hz: float
    interference_power_db: float
    interference_hot_bins: int


def _generate_ca_code(prn: int) -> np.ndarray:
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1-32, got {prn}")

    g1 = [1] * 10
    g2 = [1] * 10
    tap1, tap2 = G2_TAPS[prn]
    tap1 -= 1
    tap2 -= 1

    code = np.empty(CA_CODE_LENGTH, dtype=np.float32)
    for idx in range(CA_CODE_LENGTH):
        g1_out = g1[9]
        g2_delayed = g2[tap1] ^ g2[tap2]
        ca_bit = g1_out ^ g2_delayed
        code[idx] = 2.0 * ca_bit - 1.0

        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [g1_fb] + g1[:9]
        g2 = [g2_fb] + g2[:9]

    return code


def _sample_ca_code(prn: int, sampling_freq: float, n_samples: int) -> np.ndarray:
    code = _generate_ca_code(prn)
    sample_idx = np.arange(n_samples, dtype=np.float64)
    chip_idx = ((sample_idx / sampling_freq) * CA_CHIP_RATE_HZ).astype(np.int64)
    return code[chip_idx % CA_CODE_LENGTH]


def _synth_gps_ca_signal(
    *,
    prn: int,
    code_phase_samples: int,
    doppler_hz: float,
    sampling_freq: float,
    n_samples: int,
    snr_db: float,
    rng: np.random.Generator,
    amplitude: float = 1.0,
    carrier_phase_rad: float = 0.0,
) -> np.ndarray:
    code = _sample_ca_code(prn, sampling_freq, n_samples)
    code = np.roll(code, int(code_phase_samples))
    t = np.arange(n_samples, dtype=np.float64) / sampling_freq
    carrier = np.cos(2.0 * np.pi * float(doppler_hz) * t + carrier_phase_rad)
    signal = float(amplitude) * code * carrier

    noise_sigma = float(amplitude) / math.sqrt(10.0 ** (float(snr_db) / 10.0))
    noise = rng.standard_normal(n_samples) * noise_sigma
    return (signal + noise).astype(np.float32)


def _rms(signal: np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def _inject_narrowband_jammer(
    signal: np.ndarray,
    *,
    sampling_freq: float,
    center_hz: float,
    jsr_db: float,
    phase_rad: float,
) -> np.ndarray:
    base_rms = max(_rms(signal), 1e-12)
    amp = base_rms * 10.0 ** (float(jsr_db) / 20.0)
    t = np.arange(signal.size, dtype=np.float64) / sampling_freq
    tone = amp * np.cos(2.0 * np.pi * float(center_hz) * t + phase_rad)
    return (np.asarray(signal, dtype=np.float64) + tone).astype(np.float32)


def _inject_wideband_jammer(
    signal: np.ndarray,
    *,
    jsr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    base_rms = max(_rms(signal), 1e-12)
    target_rms = base_rms * 10.0 ** (float(jsr_db) / 20.0)
    noise = rng.standard_normal(signal.size)
    noise *= target_rms / max(_rms(noise), 1e-12)
    return (np.asarray(signal, dtype=np.float64) + noise).astype(np.float32)


def _inject_chirp_jammer(
    signal: np.ndarray,
    *,
    sampling_freq: float,
    start_hz: float,
    stop_hz: float,
    jsr_db: float,
) -> np.ndarray:
    base_rms = max(_rms(signal), 1e-12)
    amp = base_rms * 10.0 ** (float(jsr_db) / 20.0)
    t = np.arange(signal.size, dtype=np.float64) / sampling_freq
    duration = max(signal.size / sampling_freq, 1e-12)
    k = (float(stop_hz) - float(start_hz)) / duration
    phase = 2.0 * np.pi * (float(start_hz) * t + 0.5 * k * t * t)
    chirp = amp * np.cos(phase)
    return (np.asarray(signal, dtype=np.float64) + chirp).astype(np.float32)


def _build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    rng = np.random.default_rng(args.seed)
    base = _synth_gps_ca_signal(
        prn=args.prn,
        code_phase_samples=args.code_phase_samples,
        doppler_hz=args.doppler_hz,
        sampling_freq=args.sampling_freq,
        n_samples=args.n_samples,
        snr_db=args.snr_db,
        rng=rng,
    )
    spoof = _synth_gps_ca_signal(
        prn=args.prn,
        code_phase_samples=args.code_phase_samples + args.spoof_delay_samples,
        doppler_hz=args.doppler_hz + args.spoof_doppler_offset_hz,
        sampling_freq=args.sampling_freq,
        n_samples=args.n_samples,
        snr_db=args.snr_db + args.spoof_jsr_db,
        rng=rng,
        amplitude=10.0 ** (args.spoof_jsr_db / 20.0),
        carrier_phase_rad=0.7,
    )

    return [
        Scenario("clean", "none", base, "Nominal C/A signal plus thermal noise."),
        Scenario(
            "narrowband_jammer",
            "jamming",
            _inject_narrowband_jammer(
                base,
                sampling_freq=args.sampling_freq,
                center_hz=args.narrowband_hz,
                jsr_db=args.jnr_db,
                phase_rad=0.3,
            ),
            f"Single-tone jammer at {args.narrowband_hz:.0f} Hz.",
        ),
        Scenario(
            "wideband_jammer",
            "jamming",
            _inject_wideband_jammer(base, jsr_db=args.jnr_db, rng=rng),
            "White-noise jammer across the acquisition band.",
        ),
        Scenario(
            "chirp_jammer",
            "jamming",
            _inject_chirp_jammer(
                base,
                sampling_freq=args.sampling_freq,
                start_hz=args.chirp_start_hz,
                stop_hz=args.chirp_stop_hz,
                jsr_db=args.jnr_db,
            ),
            f"Linear chirp from {args.chirp_start_hz:.0f} to {args.chirp_stop_hz:.0f} Hz.",
        ),
        Scenario(
            "delayed_replica",
            "spoof_like",
            (base.astype(np.float64) + spoof.astype(np.float64)).astype(np.float32),
            f"Coherent delayed replica at +{args.spoof_delay_samples} samples.",
        ),
    ]


def _doppler_bins(doppler_range: float, doppler_step: float) -> np.ndarray:
    if doppler_step <= 0:
        raise ValueError("doppler_step must be positive")
    return np.arange(-doppler_range, doppler_range + 0.5 * doppler_step, doppler_step)


def _peak_ratio(values: np.ndarray, peak_idx: int, guard: int = 16) -> float:
    if values.size <= 1:
        return float("inf")
    peak = float(values[peak_idx])
    keep = np.ones(values.size, dtype=bool)
    for idx in range(values.size):
        dist = abs(idx - peak_idx)
        if dist > values.size // 2:
            dist = values.size - dist
        if dist <= guard:
            keep[idx] = False
    second = float(np.max(values[keep])) if np.any(keep) else 0.0
    return float(peak / max(second, 1e-12))


def _cpu_acquire(
    signal: np.ndarray,
    *,
    sampling_freq: float,
    prn_list: Iterable[int],
    doppler_range: float,
    doppler_step: float,
    threshold: float,
) -> list[AcquisitionHit]:
    sig = np.asarray(signal, dtype=np.float32).ravel()
    n = sig.size
    sample_idx = np.arange(n, dtype=np.float64)
    bins = _doppler_bins(doppler_range, doppler_step)
    hits: list[AcquisitionHit] = []

    for prn in prn_list:
        code = _sample_ca_code(int(prn), sampling_freq, n).astype(np.complex64)
        code_fft_conj = np.conj(np.fft.fft(code))

        best_peak = -1.0
        best_idx = 0
        best_doppler = 0.0
        best_ratio = 0.0
        for doppler_hz in bins:
            phase = -2.0j * np.pi * float(doppler_hz) * sample_idx / sampling_freq
            wiped = sig.astype(np.complex128) * np.exp(phase)
            corr = np.abs(np.fft.ifft(np.fft.fft(wiped) * code_fft_conj)) ** 2
            peak_idx = int(np.argmax(corr))
            peak = float(corr[peak_idx])
            if peak > best_peak:
                best_peak = peak
                best_idx = peak_idx
                best_doppler = float(doppler_hz)
                best_ratio = _peak_ratio(corr, peak_idx)

        hits.append(
            AcquisitionHit(
                prn=int(prn),
                acquired=bool(best_ratio >= threshold),
                code_phase=float(best_idx),
                doppler_hz=best_doppler,
                snr=float(best_ratio),
            )
        )

    return hits


def _gpu_acquire(
    signal: np.ndarray,
    *,
    sampling_freq: float,
    prn_list: Iterable[int],
    doppler_range: float,
    doppler_step: float,
    threshold: float,
) -> list[AcquisitionHit]:
    from gnss_gpu.acquisition import Acquisition

    acq = Acquisition(
        sampling_freq=sampling_freq,
        intermediate_freq=0.0,
        doppler_range=doppler_range,
        doppler_step=doppler_step,
        threshold=threshold,
    )
    raw_hits = acq.acquire(signal, list(prn_list))
    hits: list[AcquisitionHit] = []
    for item in raw_hits:
        hits.append(
            AcquisitionHit(
                prn=int(item["prn"]),
                acquired=bool(item["acquired"]),
                code_phase=float(item["code_phase"]),
                doppler_hz=float(item["doppler_hz"]),
                snr=float(item["snr"]),
            )
        )
    return hits


def _run_acquisition(
    signal: np.ndarray,
    args: argparse.Namespace,
) -> tuple[str, float, list[AcquisitionHit]]:
    start = time.perf_counter()
    if not args.cpu_only:
        try:
            hits = _gpu_acquire(
                signal,
                sampling_freq=args.sampling_freq,
                prn_list=args.prn_search,
                doppler_range=args.doppler_range_hz,
                doppler_step=args.doppler_step_hz,
                threshold=args.acquisition_threshold,
            )
            return "cuda_acquisition", (time.perf_counter() - start) * 1000.0, hits
        except Exception as exc:
            print(f"[gnss-security-lab] CUDA acquisition unavailable: {exc}")

    hits = _cpu_acquire(
        signal,
        sampling_freq=args.sampling_freq,
        prn_list=args.prn_search,
        doppler_range=args.doppler_range_hz,
        doppler_step=args.doppler_step_hz,
        threshold=args.acquisition_threshold,
    )
    return "numpy_fft_acquisition", (time.perf_counter() - start) * 1000.0, hits


def _cpu_interference_summary(
    signal: np.ndarray,
    *,
    sampling_freq: float,
    fft_size: int,
    hop_size: int,
    threshold_db: float,
) -> dict:
    sig = np.asarray(signal, dtype=np.float32).ravel()
    if sig.size < fft_size:
        sig = np.pad(sig, (0, fft_size - sig.size))

    frames = []
    window = np.hanning(fft_size).astype(np.float32)
    for start in range(0, sig.size - fft_size + 1, hop_size):
        frame = sig[start : start + fft_size] * window
        spectrum = np.fft.rfft(frame)
        frames.append(np.abs(spectrum) ** 2)
    if not frames:
        frames.append(np.abs(np.fft.rfft(sig[:fft_size] * window)) ** 2)

    power = np.asarray(frames, dtype=np.float64)
    power_db = 10.0 * np.log10(power + 1e-18)
    floor_db = float(np.median(power_db))
    hot = power_db > floor_db + float(threshold_db)
    detected = bool(np.any(hot))
    max_frame, max_bin = np.unravel_index(int(np.argmax(power_db)), power_db.shape)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / sampling_freq)

    hot_bins = int(np.sum(hot[max_frame]))
    bandwidth_hz = hot_bins * sampling_freq / fft_size
    hot_fraction = hot_bins / max(1, power_db.shape[1])
    rms_db = 20.0 * math.log10(max(_rms(sig), 1e-12))

    # A clean spread-spectrum C/A signal has many bins above the median in a
    # short-window STFT.  Treat dense occupancy as interference only when the
    # generated signal power is clearly above the nominal unit-amplitude scale.
    narrowband_detected = detected and hot_fraction < 0.08
    wideband_detected = rms_db > 8.0
    detected = bool(narrowband_detected or wideband_detected)
    if narrowband_detected:
        kind = "narrowband"
    elif wideband_detected:
        kind = "wideband"
    else:
        kind = "none"

    return {
        "detected": detected,
        "kind": kind,
        "center_freq_hz": float(freqs[max_bin]),
        "bandwidth_hz": float(bandwidth_hz),
        "power_db": float(power_db[max_frame, max_bin]),
        "hot_bins": hot_bins,
    }


def _gpu_interference_summary(signal: np.ndarray, args: argparse.Namespace) -> dict:
    from gnss_gpu.interference import InterferenceDetector

    detector = InterferenceDetector(
        sampling_freq=args.sampling_freq,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        threshold_db=args.interference_threshold_db,
    )
    detections = detector.detect(signal)
    if not detections:
        return {
            "detected": False,
            "kind": "none",
            "center_freq_hz": 0.0,
            "bandwidth_hz": 0.0,
            "power_db": 0.0,
            "hot_bins": 0,
        }
    first = detections[0]
    return {
        "detected": True,
        "kind": str(first.get("type_name", "interference")),
        "center_freq_hz": float(first.get("center_freq_hz", 0.0)),
        "bandwidth_hz": float(first.get("bandwidth_hz", 0.0)),
        "power_db": float(first.get("power_db", 0.0)),
        "hot_bins": 0,
    }


def _run_interference(signal: np.ndarray, args: argparse.Namespace) -> tuple[str, float, dict]:
    start = time.perf_counter()
    if not args.cpu_only:
        try:
            summary = _gpu_interference_summary(signal, args)
            if not summary["detected"]:
                guard = _cpu_interference_summary(
                    signal,
                    sampling_freq=args.sampling_freq,
                    fft_size=args.fft_size,
                    hop_size=args.hop_size,
                    threshold_db=args.interference_threshold_db,
                )
                if guard["detected"]:
                    return (
                        "cuda_stft_detector+numpy_power_guard",
                        (time.perf_counter() - start) * 1000.0,
                        guard,
                    )
            return "cuda_stft_detector", (time.perf_counter() - start) * 1000.0, summary
        except Exception as exc:
            print(f"[gnss-security-lab] CUDA interference detector unavailable: {exc}")

    summary = _cpu_interference_summary(
        signal,
        sampling_freq=args.sampling_freq,
        fft_size=args.fft_size,
        hop_size=args.hop_size,
        threshold_db=args.interference_threshold_db,
    )
    return "numpy_stft_detector", (time.perf_counter() - start) * 1000.0, summary


def _circular_error(value: float, truth: float, period: int) -> float:
    err = (float(value) - float(truth) + period / 2.0) % period - period / 2.0
    return float(err)


def _doppler_error(value: float, truth: float) -> float:
    # Real-valued cosine input has a sign ambiguity after wipe-off.
    return float(min(abs(float(value) - truth), abs(float(value) + truth)))


def _metrics_for_scenario(scenario: Scenario, args: argparse.Namespace) -> ScenarioMetrics:
    acq_backend, acq_ms, hits = _run_acquisition(scenario.signal, args)
    int_backend, int_ms, interference = _run_interference(scenario.signal, args)

    best = max(hits, key=lambda hit: hit.snr)
    target = next((hit for hit in hits if hit.prn == args.prn), None)
    if target is None:
        target = AcquisitionHit(args.prn, False, 0.0, 0.0, 0.0)
    max_false_snr = max((hit.snr for hit in hits if hit.prn != args.prn), default=0.0)

    code_err = _circular_error(
        target.code_phase,
        args.code_phase_samples,
        int(args.n_samples),
    )
    doppler_err = _doppler_error(target.doppler_hz, args.doppler_hz)
    false_lock = bool(
        target.acquired
        and (abs(code_err) > args.false_lock_code_error_samples
             or doppler_err > args.false_lock_doppler_error_hz)
    )
    sig_rms = _rms(scenario.signal)
    crest = float(np.max(np.abs(scenario.signal)) / max(sig_rms, 1e-12))

    return ScenarioMetrics(
        scenario=scenario.name,
        attack_type=scenario.attack_type,
        notes=scenario.notes,
        acquisition_backend=acq_backend,
        interference_backend=int_backend,
        n_samples=int(scenario.signal.size),
        rms=sig_rms,
        crest_factor=crest,
        acquisition_ms=float(acq_ms),
        interference_ms=float(int_ms),
        target_acquired=bool(target.acquired),
        target_snr=float(target.snr),
        target_code_phase_samples=float(target.code_phase),
        target_code_phase_error_samples=float(code_err),
        target_doppler_hz=float(target.doppler_hz),
        target_doppler_error_hz=float(doppler_err),
        best_prn=int(best.prn),
        best_snr=float(best.snr),
        max_false_prn_snr=float(max_false_snr),
        false_lock=false_lock,
        interference_detected=bool(interference["detected"]),
        interference_kind=str(interference["kind"]),
        interference_center_freq_hz=float(interference["center_freq_hz"]),
        interference_bandwidth_hz=float(interference["bandwidth_hz"]),
        interference_power_db=float(interference["power_db"]),
        interference_hot_bins=int(interference["hot_bins"]),
    )


def _write_csv(path: Path, rows: list[ScenarioMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _json_ready(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, args: argparse.Namespace, rows: list[ScenarioMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "gpu_gnss_stress_lab_phase1_rf_mvp",
        "config": {
            "prn": args.prn,
            "prn_search": list(args.prn_search),
            "sampling_freq": args.sampling_freq,
            "n_samples": args.n_samples,
            "duration_ms": args.duration_ms,
            "code_phase_samples": args.code_phase_samples,
            "doppler_hz": args.doppler_hz,
            "snr_db": args.snr_db,
            "jnr_db": args.jnr_db,
            "spoof_delay_samples": args.spoof_delay_samples,
            "spoof_jsr_db": args.spoof_jsr_db,
        },
        "rows": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, indent=2, default=_json_ready), encoding="utf-8")


def _bar(width: float, label: str, value: str, color: str = "#2f6f73") -> str:
    clamped = max(0.0, min(100.0, width))
    return (
        '<div class="bar-row">'
        f'<span>{html.escape(label)}</span>'
        '<div class="bar-track">'
        f'<div class="bar-fill" style="width:{clamped:.1f}%;background:{color}"></div>'
        "</div>"
        f"<strong>{html.escape(value)}</strong>"
        "</div>"
    )


def _write_html(path: Path, rows: list[ScenarioMetrics], args: argparse.Namespace) -> None:
    max_snr = max(row.best_snr for row in rows) or 1.0
    max_code_error = max(abs(row.target_code_phase_error_samples) for row in rows) or 1.0
    cards = []
    for row in rows:
        status = "false lock" if row.false_lock else "locked" if row.target_acquired else "missed"
        status_class = "bad" if row.false_lock else "ok" if row.target_acquired else "warn"
        cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <h2>{html.escape(row.scenario)}</h2>
                <span class="{status_class}">{html.escape(status)}</span>
              </div>
              <p>{html.escape(row.notes)}</p>
              {_bar(row.best_snr / max_snr * 100.0, "best peak ratio", f"{row.best_snr:.2f}")}
              {_bar(abs(row.target_code_phase_error_samples) / max_code_error * 100.0,
                    "code error", f"{row.target_code_phase_error_samples:+.1f} samples",
                    "#8b3f2f" if row.false_lock else "#6a7c30")}
              <dl>
                <dt>Acquisition</dt><dd>{html.escape(row.acquisition_backend)} / {row.acquisition_ms:.2f} ms</dd>
                <dt>Interference</dt><dd>{html.escape(row.interference_backend)} / {row.interference_ms:.2f} ms</dd>
                <dt>Detector</dt><dd>{html.escape(row.interference_kind)} at {row.interference_center_freq_hz:.0f} Hz</dd>
              </dl>
            </article>
            """
        )

    header_cells = [
        "scenario",
        "backend",
        "target acquired",
        "target SNR",
        "code err",
        "doppler err",
        "best PRN",
        "false lock",
        "interference",
    ]
    body_rows = []
    for row in rows:
        cells = [
            row.scenario,
            row.acquisition_backend,
            str(row.target_acquired),
            f"{row.target_snr:.2f}",
            f"{row.target_code_phase_error_samples:+.1f}",
            f"{row.target_doppler_error_hz:.0f}",
            str(row.best_prn),
            str(row.false_lock),
            f"{row.interference_kind} / {row.interference_power_db:.1f} dB",
        ]
        body_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells) + "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GPU GNSS Stress Lab</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172126;
      --muted: #5a6870;
      --line: #d7dee2;
      --bg: #f6f7f4;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 32px 6vw 20px;
      border-bottom: 1px solid var(--line);
      background: #eef2ef;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 48px);
      letter-spacing: 0;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
      line-height: 1.45;
    }}
    main {{
      padding: 26px 6vw 44px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 26px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .ok, .warn, .bad {{
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .ok {{ background: #dcebd8; color: #27592d; }}
    .warn {{ background: #fff1c2; color: #7a5700; }}
    .bad {{ background: #f4d3cc; color: #8b2d1e; }}
    .bar-row {{
      display: grid;
      grid-template-columns: 96px 1fr 96px;
      gap: 8px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }}
    .bar-track {{
      height: 10px;
      background: #e8ecee;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 6px 10px;
      margin: 12px 0 0;
      font-size: 13px;
    }}
    dt {{ color: var(--muted); }}
    dd {{ margin: 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      background: #e9eeea;
      font-size: 12px;
      text-transform: uppercase;
      color: #46545b;
    }}
    code {{
      background: #edf0f2;
      padding: 2px 5px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>GPU GNSS Stress Lab</h1>
    <p class="sub">
      Phase 1 RF MVP: GPS C/A signal stress tests for jamming, spoof-like delayed replicas,
      acquisition robustness, and interference detection. Target PRN <code>{args.prn}</code>,
      {args.n_samples} samples at {args.sampling_freq:.0f} Hz.
    </p>
  </header>
  <main>
    <section class="grid">
      {''.join(cards)}
    </section>
    <section>
      <table>
        <thead><tr>{''.join(f"<th>{html.escape(cell)}</th>" for cell in header_cells)}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


def _parse_prn_search(text: str) -> tuple[int, ...]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("PRN search list cannot be empty")
    return tuple(out)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/gnss_security_lab"))
    parser.add_argument("--prn", type=int, default=7)
    parser.add_argument("--prn-search", type=_parse_prn_search, default=DEFAULT_PRN_SEARCH)
    parser.add_argument("--sampling-freq", type=float, default=4.092e6)
    parser.add_argument("--duration-ms", type=float, default=1.0)
    parser.add_argument("--code-phase-samples", type=int, default=96)
    parser.add_argument("--doppler-hz", type=float, default=1500.0)
    parser.add_argument("--snr-db", type=float, default=18.0)
    parser.add_argument("--jnr-db", type=float, default=16.0)
    parser.add_argument("--narrowband-hz", type=float, default=240_000.0)
    parser.add_argument("--chirp-start-hz", type=float, default=60_000.0)
    parser.add_argument("--chirp-stop-hz", type=float, default=650_000.0)
    parser.add_argument("--spoof-delay-samples", type=int, default=112)
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
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--cpu-only", action="store_true", help="Skip CUDA bindings and use NumPy fallbacks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.n_samples = max(1, int(round(args.sampling_freq * args.duration_ms * 1e-3)))
    if args.prn not in args.prn_search:
        args.prn_search = tuple([args.prn, *args.prn_search])

    scenarios = _build_scenarios(args)
    rows = [_metrics_for_scenario(scenario, args) for scenario in scenarios]

    csv_path = args.out_dir / "gnss_security_lab_summary.csv"
    json_path = args.out_dir / "gnss_security_lab_summary.json"
    html_path = args.out_dir / "gnss_security_lab_report.html"
    _write_csv(csv_path, rows)
    _write_json(json_path, args, rows)
    _write_html(html_path, rows, args)

    print(f"[gnss-security-lab] wrote {csv_path}")
    print(f"[gnss-security-lab] wrote {json_path}")
    print(f"[gnss-security-lab] wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
