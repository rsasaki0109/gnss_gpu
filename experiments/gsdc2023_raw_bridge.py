from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate import compute_metrics, ecef_to_lla, lla_to_ecef
from gnss_gpu import wls_position
from gnss_gpu.fgo import fgo_gnss_lm


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "ref" / "gsdc2023" / "dataset_2023"
POSITION_SOURCES = ("baseline", "raw_wls", "fgo", "auto", "gated")
RAW_GNSS_COLUMNS = [
    "utcTimeMillis",
    "Svid",
    "ConstellationType",
    "SignalType",
    "RawPseudorangeMeters",
    "IonosphericDelayMeters",
    "TroposphericDelayMeters",
    "SvClockBiasMeters",
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "SvElevationDegrees",
    "Cn0DbHz",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
]


@dataclass(frozen=True)
class TripArrays:
    times_ms: np.ndarray
    sat_ecef: np.ndarray
    pseudorange: np.ndarray
    weights: np.ndarray
    kaggle_wls: np.ndarray
    truth: np.ndarray
    max_sats: int
    has_truth: bool


GATED_BASELINE_THRESHOLD_DEFAULT = 500.0


@dataclass(frozen=True)
class BridgeConfig:
    motion_sigma_m: float = 3.0
    fgo_iters: int = 8
    signal_type: str = "GPS_L1_CA"
    constellation_type: int = 1
    weight_mode: str = "sin2el"
    position_source: str = "baseline"
    chunk_epochs: int = 0
    gated_baseline_threshold: float = GATED_BASELINE_THRESHOLD_DEFAULT

    def __post_init__(self) -> None:
        validate_position_source(self.position_source)


@dataclass
class BridgeResult:
    trip: str
    signal_type: str
    weight_mode: str
    selected_source_mode: str
    times_ms: np.ndarray
    kaggle_wls: np.ndarray
    raw_wls: np.ndarray
    fgo_state: np.ndarray
    selected_state: np.ndarray
    selected_sources: np.ndarray
    truth: np.ndarray | None
    max_sats: int
    fgo_iters: int
    failed_chunks: int
    selected_mse_pr: float
    baseline_mse_pr: float
    raw_wls_mse_pr: float
    fgo_mse_pr: float
    selected_source_counts: dict[str, int]
    metrics_selected: dict | None
    metrics_kaggle: dict | None
    metrics_raw_wls: dict | None
    metrics_fgo: dict | None

    @property
    def n_epochs(self) -> int:
        return int(self.times_ms.size)

    def positions_table(self) -> pd.DataFrame:
        selected_llh = ecef_to_llh_deg(self.selected_state[:, :3])
        kaggle_llh = ecef_to_llh_deg(self.kaggle_wls)
        raw_wls_llh = ecef_to_llh_deg(self.raw_wls[:, :3])
        fgo_llh = ecef_to_llh_deg(self.fgo_state[:, :3])
        if self.truth is not None:
            truth_llh = ecef_to_llh_deg(self.truth)
        else:
            truth_llh = np.full((self.times_ms.size, 3), np.nan, dtype=np.float64)
        return pd.DataFrame(
            {
                "UnixTimeMillis": self.times_ms.astype(np.int64),
                "SelectedSource": self.selected_sources.astype(str),
                "BaselineLatitudeDegrees": kaggle_llh[:, 0],
                "BaselineLongitudeDegrees": kaggle_llh[:, 1],
                "BaselineAltitudeMeters": kaggle_llh[:, 2],
                "RawWlsLatitudeDegrees": raw_wls_llh[:, 0],
                "RawWlsLongitudeDegrees": raw_wls_llh[:, 1],
                "RawWlsAltitudeMeters": raw_wls_llh[:, 2],
                "FgoLatitudeDegrees": fgo_llh[:, 0],
                "FgoLongitudeDegrees": fgo_llh[:, 1],
                "FgoAltitudeMeters": fgo_llh[:, 2],
                "LatitudeDegrees": selected_llh[:, 0],
                "LongitudeDegrees": selected_llh[:, 1],
                "AltitudeMeters": selected_llh[:, 2],
                "GroundTruthLatitudeDegrees": truth_llh[:, 0],
                "GroundTruthLongitudeDegrees": truth_llh[:, 1],
                "GroundTruthAltitudeMeters": truth_llh[:, 2],
            },
        )

    def metrics_payload(self) -> dict:
        return {
            "trip": self.trip,
            "signal_type": self.signal_type,
            "weight_mode": self.weight_mode,
            "selected_source_mode": self.selected_source_mode,
            "n_epochs": self.n_epochs,
            "max_sats": int(self.max_sats),
            "fgo_iters": int(self.fgo_iters),
            "failed_chunks": int(self.failed_chunks),
            "mse_pr": float(self.selected_mse_pr),
            "selected_mse_pr": float(self.selected_mse_pr),
            "baseline_mse_pr": float(self.baseline_mse_pr),
            "raw_wls_mse_pr": float(self.raw_wls_mse_pr),
            "fgo_mse_pr": float(self.fgo_mse_pr),
            "selected_source_counts": {k: int(v) for k, v in self.selected_source_counts.items()},
            "selected_score_m": score_from_metrics(self.metrics_selected),
            "kaggle_wls_score_m": score_from_metrics(self.metrics_kaggle),
            "raw_wls_score_m": score_from_metrics(self.metrics_raw_wls),
            "fgo_score_m": score_from_metrics(self.metrics_fgo),
            "selected_metrics": metrics_summary(self.metrics_selected),
            "kaggle_wls_metrics": metrics_summary(self.metrics_kaggle),
            "raw_wls_metrics": metrics_summary(self.metrics_raw_wls),
            "fgo_metrics": metrics_summary(self.metrics_fgo),
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"GSDC2023 raw validation: {self.trip}",
            f"  epochs      : {self.n_epochs}",
            f"  max sats/ep : {self.max_sats}",
            f"  signal      : {self.signal_type}",
            f"  weights      : {self.weight_mode}",
            f"  FGO iters   : {self.fgo_iters}",
            f"  output source: {self.selected_source_mode}",
            f"  wMSE pr     : {self.selected_mse_pr:.4f} (selected)",
            "  source mix  : "
            + ", ".join(f"{name}={count}" for name, count in self.selected_source_counts.items() if count > 0),
            (
                f"  candidate MSE: baseline={self.baseline_mse_pr:.4f} "
                f"raw={self.raw_wls_mse_pr:.4f} fgo={self.fgo_mse_pr:.4f}"
            ),
        ]
        if self.failed_chunks > 0:
            lines.append(f"  failed chunks: {self.failed_chunks} (raw WLS fallback)")
        if self.metrics_selected is not None:
            lines.extend(
                [
                    format_metrics_line("Selected", self.metrics_selected),
                    format_metrics_line("Kaggle WLS", self.metrics_kaggle),
                    format_metrics_line("Raw WLS", self.metrics_raw_wls),
                    format_metrics_line("FGO", self.metrics_fgo),
                ],
            )
            if self.metrics_selected["rms_2d"] < self.metrics_raw_wls["rms_2d"] - 1e-9:
                gain = (1.0 - self.metrics_selected["rms_2d"] / self.metrics_raw_wls["rms_2d"]) * 100.0
                lines.append(f"  -> selected output improves raw WLS by {gain:.1f}% on RMS2D")
        else:
            lines.append("  ground truth: unavailable (test/raw mode)")
        return lines


def validate_position_source(position_source: str) -> str:
    if position_source not in POSITION_SOURCES:
        raise ValueError(f"unsupported position source: {position_source}")
    return position_source


def nearest_index(sorted_times: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(sorted_times, t))
    if idx <= 0:
        return 0
    if idx >= len(sorted_times):
        return len(sorted_times) - 1
    prev_idx = idx - 1
    return idx if abs(sorted_times[idx] - t) < abs(sorted_times[prev_idx] - t) else prev_idx


def load_ground_truth_ecef(trip_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    gt_path = trip_dir / "ground_truth.csv"
    if not gt_path.is_file():
        return None, None
    gt_df = pd.read_csv(gt_path)
    gt_times = gt_df["UnixTimeMillis"].to_numpy(dtype=np.float64)
    gt_ecef = np.array(
        [
            lla_to_ecef(np.deg2rad(lat), np.deg2rad(lon), alt)
            for lat, lon, alt in gt_df[["LatitudeDegrees", "LongitudeDegrees", "AltitudeMeters"]].to_numpy(
                dtype=np.float64,
            )
        ],
        dtype=np.float64,
    )
    return gt_times, gt_ecef


def build_trip_arrays(
    trip_dir: Path,
    *,
    max_epochs: int,
    start_epoch: int,
    constellation_type: int,
    signal_type: str,
    weight_mode: str,
) -> TripArrays:
    raw_path = trip_dir / "device_gnss.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"device_gnss.csv not found: {raw_path}")

    gt_times, gt_ecef = load_ground_truth_ecef(trip_dir)
    df = pd.read_csv(raw_path, compression="zip", usecols=RAW_GNSS_COLUMNS)
    df = df[(df["ConstellationType"] == constellation_type) & (df["SignalType"] == signal_type)]
    df = df[
        np.isfinite(df["RawPseudorangeMeters"])
        & np.isfinite(df["SvPositionXEcefMeters"])
        & np.isfinite(df["SvElevationDegrees"])
        & np.isfinite(df["SvClockBiasMeters"])
        & np.isfinite(df["IonosphericDelayMeters"])
        & np.isfinite(df["TroposphericDelayMeters"])
    ]
    if df.empty:
        raise RuntimeError("No usable raw observations after filtering")

    df = df.sort_values(["utcTimeMillis", "Svid", "Cn0DbHz"]).groupby(
        ["utcTimeMillis", "Svid"], as_index=False,
    ).tail(1)

    epochs: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    usable_epoch_index = 0
    for tow_ms, group in df.groupby("utcTimeMillis"):
        if len(group) < 4:
            continue
        if usable_epoch_index < start_epoch:
            usable_epoch_index += 1
            continue

        sats = group[["SvPositionXEcefMeters", "SvPositionYEcefMeters", "SvPositionZEcefMeters"]].to_numpy(
            dtype=np.float64,
        )
        pseudorange = (
            group["RawPseudorangeMeters"]
            + group["SvClockBiasMeters"]
            - group["IonosphericDelayMeters"]
            - group["TroposphericDelayMeters"]
        ).to_numpy(dtype=np.float64)
        if weight_mode == "sin2el":
            weights = np.maximum(np.sin(np.deg2rad(group["SvElevationDegrees"].to_numpy(dtype=np.float64))), 0.1) ** 2
        else:
            weights = np.maximum(group["Cn0DbHz"].to_numpy(dtype=np.float64), 1.0)

        kaggle_wls = group[["WlsPositionXEcefMeters", "WlsPositionYEcefMeters", "WlsPositionZEcefMeters"]].iloc[
            0
        ].to_numpy(dtype=np.float64)
        if gt_times is not None and gt_ecef is not None:
            gt_idx = nearest_index(gt_times, float(tow_ms))
            truth = gt_ecef[gt_idx]
        else:
            truth = np.full(3, np.nan, dtype=np.float64)

        epochs.append((float(tow_ms), sats, pseudorange, weights, kaggle_wls, truth))
        usable_epoch_index += 1
        if len(epochs) >= max_epochs:
            break

    if not epochs:
        raise RuntimeError("No usable epochs found")

    max_sats = max(len(item[1]) for item in epochs)
    n_epoch = len(epochs)
    sat_ecef = np.zeros((n_epoch, max_sats, 3), dtype=np.float64)
    pseudorange = np.zeros((n_epoch, max_sats), dtype=np.float64)
    weights = np.zeros((n_epoch, max_sats), dtype=np.float64)
    kaggle_wls = np.zeros((n_epoch, 3), dtype=np.float64)
    truth = np.zeros((n_epoch, 3), dtype=np.float64)
    times_ms = np.zeros(n_epoch, dtype=np.float64)

    for i, (tow_ms, sats, pr, w, baseline, gt_xyz) in enumerate(epochs):
        n_sat = len(sats)
        sat_ecef[i, :n_sat] = sats
        pseudorange[i, :n_sat] = pr
        weights[i, :n_sat] = w
        kaggle_wls[i] = baseline
        truth[i] = gt_xyz
        times_ms[i] = tow_ms

    return TripArrays(
        times_ms=times_ms,
        sat_ecef=sat_ecef,
        pseudorange=pseudorange,
        weights=weights,
        kaggle_wls=kaggle_wls,
        truth=truth,
        max_sats=max_sats,
        has_truth=(gt_times is not None and gt_ecef is not None),
    )


def run_wls(sat_ecef: np.ndarray, pseudorange: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n_epoch = sat_ecef.shape[0]
    out = np.zeros((n_epoch, 4), dtype=np.float64)
    for i in range(n_epoch):
        idx = np.flatnonzero(weights[i] > 0)
        if idx.size < 4:
            continue
        state, _ = wls_position(
            sat_ecef[i, idx].reshape(-1),
            pseudorange[i, idx],
            weights[i, idx],
            25,
            1e-9,
        )
        out[i] = state
    return out


def fit_state_with_clock_bias(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    state = np.zeros((xyz.shape[0], 4), dtype=np.float64)
    state[:, :3] = xyz
    weighted_sse = 0.0
    weight_sum = 0.0
    per_epoch_wmse = np.full(xyz.shape[0], np.nan, dtype=np.float64)

    for i in range(xyz.shape[0]):
        idx = np.flatnonzero(weights[i] > 0)
        if idx.size < 4:
            continue
        rho = np.linalg.norm(sat_ecef[i, idx] - xyz[i], axis=1)
        resid0 = pseudorange[i, idx] - rho
        w = weights[i, idx]
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue
        bias = float(np.sum(w * resid0) / w_sum)
        resid = resid0 - bias
        sse = float(np.sum(w * resid * resid))
        state[i, 3] = bias
        weighted_sse += sse
        weight_sum += w_sum
        per_epoch_wmse[i] = sse / w_sum

    return state, weighted_sse, weight_sum, per_epoch_wmse


def weighted_mse(weighted_sse: float, weight_sum: float) -> float:
    if weight_sum <= 0.0:
        return float("inf")
    return float(weighted_sse / weight_sum)


def run_fgo_chunked(
    sat_ecef: np.ndarray,
    pseudorange: np.ndarray,
    weights: np.ndarray,
    kaggle_wls: np.ndarray,
    raw_wls: np.ndarray,
    *,
    motion_sigma_m: float,
    fgo_iters: int,
    tol: float,
    chunk_epochs: int,
) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray, dict[str, int]]:
    n_epoch = sat_ecef.shape[0]
    chunk_size = n_epoch if chunk_epochs <= 0 or n_epoch <= chunk_epochs else chunk_epochs
    stitched = raw_wls.copy()
    fgo_stitched = raw_wls.copy()
    total_iters = 0
    failed_chunks = 0
    selected_sources = np.empty(n_epoch, dtype=object)
    selected_source_counts = {"baseline": 0, "raw_wls": 0, "fgo": 0}

    for start in range(0, n_epoch, chunk_size):
        end = min(start + chunk_size, n_epoch)
        baseline_state, baseline_sse, baseline_weight_sum, _ = fit_state_with_clock_bias(
            sat_ecef[start:end],
            pseudorange[start:end],
            weights[start:end],
            kaggle_wls[start:end],
        )
        raw_state, raw_sse, raw_weight_sum, _ = fit_state_with_clock_bias(
            sat_ecef[start:end],
            pseudorange[start:end],
            weights[start:end],
            raw_wls[start:end, :3],
        )

        state_chunk = raw_state.copy()
        if start > 0 and state_chunk.shape[0] > 0:
            state_chunk[0] = stitched[start - 1]
        try:
            iters, _ = fgo_gnss_lm(
                sat_ecef[start:end],
                pseudorange[start:end],
                weights[start:end],
                state_chunk,
                motion_sigma_m=motion_sigma_m,
                max_iter=fgo_iters,
                tol=tol,
            )
        except RuntimeError:
            iters = -1

        if int(iters) < 0:
            failed_chunks += 1
            fgo_state = raw_state.copy()
        else:
            total_iters += int(iters)
            fgo_state, _, _, _ = fit_state_with_clock_bias(
                sat_ecef[start:end],
                pseudorange[start:end],
                weights[start:end],
                state_chunk[:, :3],
            )

        candidates = {
            "baseline": (baseline_state, weighted_mse(baseline_sse, baseline_weight_sum)),
            "raw_wls": (raw_state, weighted_mse(raw_sse, raw_weight_sum)),
        }
        if int(iters) >= 0:
            fgo_state, fgo_sse, fgo_weight_sum, _ = fit_state_with_clock_bias(
                sat_ecef[start:end],
                pseudorange[start:end],
                weights[start:end],
                fgo_state[:, :3],
            )
            candidates["fgo"] = (fgo_state, weighted_mse(fgo_sse, fgo_weight_sum))

        source = min(candidates, key=lambda name: candidates[name][1])
        stitched[start:end] = candidates[source][0]
        fgo_stitched[start:end] = fgo_state
        selected_sources[start:end] = source
        selected_source_counts[source] += end - start

    return stitched, fgo_stitched, total_iters, failed_chunks, selected_sources, selected_source_counts


def metrics_summary(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {
        "rms_2d_m": float(metrics["rms_2d"]),
        "rms_3d_m": float(metrics["rms_3d"]),
        "mean_2d_m": float(metrics["mean_2d"]),
        "mean_3d_m": float(metrics["mean_3d"]),
        "std_2d_m": float(metrics["std_2d"]),
        "p50_m": float(metrics["p50"]),
        "p67_m": float(metrics["p67"]),
        "p95_m": float(metrics["p95"]),
        "max_2d_m": float(metrics["max_2d"]),
        "n_epochs": int(metrics["n_epochs"]),
    }


def score_from_metrics(metrics: dict | None) -> float | None:
    if metrics is None:
        return None
    return 0.5 * (float(metrics["p50"]) + float(metrics["p95"]))


def ecef_to_llh_deg(ecef_xyz: np.ndarray) -> np.ndarray:
    ecef_xyz = np.asarray(ecef_xyz, dtype=np.float64).reshape(-1, 3)
    llh_deg = np.zeros((ecef_xyz.shape[0], 3), dtype=np.float64)
    for i, (x, y, z) in enumerate(ecef_xyz):
        lat_rad, lon_rad, alt_m = ecef_to_lla(float(x), float(y), float(z))
        llh_deg[i] = [np.rad2deg(lat_rad), np.rad2deg(lon_rad), alt_m]
    return llh_deg


def format_metrics_line(label: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"  {label:14s} unavailable"
    return (
        f"  {label:14s} "
        f"RMS2D={metrics['rms_2d']:.3f}m  "
        f"P50={metrics['p50']:.3f}m  "
        f"P95={metrics['p95']:.3f}m  "
        f"RMS3D={metrics['rms_3d']:.3f}m"
    )


def solve_trip(trip: str, batch: TripArrays, config: BridgeConfig) -> BridgeResult:
    raw_wls = run_wls(batch.sat_ecef, batch.pseudorange, batch.weights)
    auto_state, fgo_state, iters, failed_chunks, auto_sources, auto_source_counts = run_fgo_chunked(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
        raw_wls,
        motion_sigma_m=config.motion_sigma_m,
        fgo_iters=config.fgo_iters,
        tol=1e-7,
        chunk_epochs=config.chunk_epochs,
    )
    kaggle_state, kaggle_sse, kaggle_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        batch.kaggle_wls,
    )
    raw_state, raw_sse, raw_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        raw_wls[:, :3],
    )
    fgo_state, fgo_sse, fgo_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        fgo_state[:, :3],
    )
    auto_state, auto_sse, auto_weight_sum, _ = fit_state_with_clock_bias(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        auto_state[:, :3],
    )
    baseline_mse_pr = weighted_mse(kaggle_sse, kaggle_weight_sum)
    raw_wls_mse_pr = weighted_mse(raw_sse, raw_weight_sum)
    fgo_mse_pr = weighted_mse(fgo_sse, fgo_weight_sum)
    auto_mse_pr = weighted_mse(auto_sse, auto_weight_sum)

    source_states = {
        "baseline": kaggle_state,
        "raw_wls": raw_state,
        "fgo": fgo_state,
        "auto": auto_state,
    }
    source_counts = {
        "baseline": {"baseline": int(batch.times_ms.size), "raw_wls": 0, "fgo": 0},
        "raw_wls": {"baseline": 0, "raw_wls": int(batch.times_ms.size), "fgo": 0},
        "fgo": {"baseline": 0, "raw_wls": 0, "fgo": int(batch.times_ms.size)},
        "auto": auto_source_counts,
    }
    source_arrays = {
        "baseline": np.full(batch.times_ms.size, "baseline", dtype=object),
        "raw_wls": np.full(batch.times_ms.size, "raw_wls", dtype=object),
        "fgo": np.full(batch.times_ms.size, "fgo", dtype=object),
        "auto": auto_sources,
    }
    source_mse = {
        "baseline": baseline_mse_pr,
        "raw_wls": raw_wls_mse_pr,
        "fgo": fgo_mse_pr,
        "auto": auto_mse_pr,
    }

    # Gated source: use baseline unless its mse_pr exceeds the threshold,
    # in which case fall back to the best of raw_wls / fgo.
    if config.position_source == "gated":
        threshold = config.gated_baseline_threshold
        if baseline_mse_pr > threshold:
            fallback = "fgo" if fgo_mse_pr <= raw_wls_mse_pr else "raw_wls"
        else:
            fallback = "baseline"
        gated_state = source_states[fallback]
        gated_mse = source_mse[fallback]
        n_ep = int(batch.times_ms.size)
        source_states["gated"] = gated_state
        source_mse["gated"] = gated_mse
        source_arrays["gated"] = np.full(n_ep, fallback, dtype=object)
        source_counts["gated"] = {
            "baseline": n_ep if fallback == "baseline" else 0,
            "raw_wls": n_ep if fallback == "raw_wls" else 0,
            "fgo": n_ep if fallback == "fgo" else 0,
        }

    selected_state = source_states[config.position_source]
    selected_sources = source_arrays[config.position_source]
    selected_source_counts = source_counts[config.position_source]
    selected_mse_pr = source_mse[config.position_source]

    if batch.has_truth:
        truth = batch.truth
        metrics_selected = compute_metrics(selected_state[:, :3], truth)
        metrics_kaggle = compute_metrics(batch.kaggle_wls, truth)
        metrics_raw_wls = compute_metrics(raw_state[:, :3], truth)
        metrics_fgo = compute_metrics(fgo_state[:, :3], truth)
    else:
        truth = None
        metrics_selected = None
        metrics_kaggle = None
        metrics_raw_wls = None
        metrics_fgo = None

    return BridgeResult(
        trip=trip,
        signal_type=config.signal_type,
        weight_mode=config.weight_mode,
        selected_source_mode=config.position_source,
        times_ms=batch.times_ms,
        kaggle_wls=batch.kaggle_wls,
        raw_wls=raw_state,
        fgo_state=fgo_state,
        selected_state=selected_state,
        selected_sources=selected_sources,
        truth=truth,
        max_sats=batch.max_sats,
        fgo_iters=iters,
        failed_chunks=failed_chunks,
        selected_mse_pr=selected_mse_pr,
        baseline_mse_pr=baseline_mse_pr,
        raw_wls_mse_pr=raw_wls_mse_pr,
        fgo_mse_pr=fgo_mse_pr,
        selected_source_counts=selected_source_counts,
        metrics_selected=metrics_selected,
        metrics_kaggle=metrics_kaggle,
        metrics_raw_wls=metrics_raw_wls,
        metrics_fgo=metrics_fgo,
    )


def validate_raw_gsdc2023_trip(
    data_root: Path,
    trip: str,
    *,
    max_epochs: int = 200,
    start_epoch: int = 0,
    config: BridgeConfig | None = None,
) -> BridgeResult:
    cfg = config or BridgeConfig()
    trip_dir = data_root / trip
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")
    batch = build_trip_arrays(
        trip_dir,
        max_epochs=(max_epochs if max_epochs > 0 else 1_000_000_000),
        start_epoch=start_epoch,
        constellation_type=cfg.constellation_type,
        signal_type=cfg.signal_type,
        weight_mode=cfg.weight_mode,
    )
    return solve_trip(trip, batch, cfg)


def export_bridge_outputs(export_dir: Path, result: BridgeResult) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    result.positions_table().to_csv(export_dir / "bridge_positions.csv", index=False)
    (export_dir / "bridge_metrics.json").write_text(
        json.dumps(result.metrics_payload(), indent=2),
        encoding="utf-8",
    )


def load_bridge_metrics(trip_dir: Path) -> dict:
    return json.loads((trip_dir / "bridge_metrics.json").read_text(encoding="utf-8"))


def has_valid_bridge_outputs(trip_dir: Path) -> bool:
    metrics_path = trip_dir / "bridge_metrics.json"
    positions_path = trip_dir / "bridge_positions.csv"
    if not metrics_path.is_file() or not positions_path.is_file():
        return False
    if positions_path.stat().st_size <= 0:
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fgo_iters = int(metrics["fgo_iters"])
        mse_pr = float(metrics["mse_pr"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return fgo_iters >= 0 and np.isfinite(mse_pr)


def bridge_position_columns(position_source: str, available_columns: set[str] | list[str] | tuple[str, ...]) -> tuple[str, str]:
    validate_position_source(position_source)
    columns = set(available_columns)
    if position_source == "baseline":
        return "BaselineLatitudeDegrees", "BaselineLongitudeDegrees"
    if position_source == "raw_wls":
        return "RawWlsLatitudeDegrees", "RawWlsLongitudeDegrees"
    if position_source == "fgo" and {"FgoLatitudeDegrees", "FgoLongitudeDegrees"}.issubset(columns):
        return "FgoLatitudeDegrees", "FgoLongitudeDegrees"
    return "LatitudeDegrees", "LongitudeDegrees"


_build_trip_arrays = build_trip_arrays
_export_bridge_outputs = export_bridge_outputs
_fit_state_with_clock_bias = fit_state_with_clock_bias

__all__ = [
    "BridgeConfig",
    "BridgeResult",
    "DEFAULT_ROOT",
    "GATED_BASELINE_THRESHOLD_DEFAULT",
    "POSITION_SOURCES",
    "RAW_GNSS_COLUMNS",
    "TripArrays",
    "bridge_position_columns",
    "build_trip_arrays",
    "ecef_to_llh_deg",
    "export_bridge_outputs",
    "fit_state_with_clock_bias",
    "format_metrics_line",
    "has_valid_bridge_outputs",
    "load_bridge_metrics",
    "metrics_summary",
    "run_fgo_chunked",
    "run_wls",
    "score_from_metrics",
    "solve_trip",
    "validate_position_source",
    "validate_raw_gsdc2023_trip",
    "weighted_mse",
]
