"""Bind nuisance preprocessing and bit-edge detection into one calibrated API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from gnss_gpu.input_validation import positive_int
from gnss_gpu.ca_cancellation_cpu import CaCancellationConfig
from gnss_gpu.ca_local_delay_cpu import CaLocalDelayResult, cancel_other_ca_local_delay
from gnss_gpu.bit_edge_acquisition_cpu import (
    CaBitEdgeSearch,
    CaBitEdgeThreshold,
    CaBitEdgeDetection,
    CaBitEdgeStratumCalibration,
    acquire_ca_bit_edge,
    calibrate_ca_bit_edge,
    detect_ca_bit_edge,
)


def _validate_pipeline(cancellation, search):
    if (
        cancellation.sampling_freq != search.search.sampling_freq
        or cancellation.target_prn != search.search.prn
    ):
        raise ValueError(
            "cancellation and target search must match sampling frequency and target PRN"
        )
    if search.periods < max(3, cancellation.discovery_periods):
        raise ValueError(
            "target decision must contain enough nuisance-discovery periods"
        )


@dataclass(frozen=True)
class CaWeakAcquisitionThreshold:
    cancellation: CaCancellationConfig
    detection: CaBitEdgeThreshold
    delay_radius_samples: int = 1
    coarse_candidates: int = 1

    def __post_init__(self):
        _validate_pipeline(self.cancellation, self.detection.config)
        if self.delay_radius_samples not in (0, 1, 2):
            raise ValueError("delay_radius_samples must be 0, 1 or 2")
        object.__setattr__(
            self,
            "coarse_candidates",
            positive_int("coarse_candidates", self.coarse_candidates),
        )


@dataclass(frozen=True)
class CaWeakAcquisitionResult:
    detection: CaBitEdgeDetection
    cancellation: CaLocalDelayResult


def _cancel_weak(iq, cancellation, delay_radius_samples, coarse_candidates):
    if coarse_candidates == 1:
        return cancel_other_ca_local_delay(
            iq, cancellation, delay_radius_samples=delay_radius_samples
        )
    from gnss_gpu.ca_multiseed_cpu import cancel_other_ca_multiseed
    from gnss_gpu.ca_local_delay_cpu import _refine_local_delays

    discovery = cancel_other_ca_multiseed(
        iq, cancellation, coarse_candidates=coarse_candidates
    )
    return _refine_local_delays(iq, cancellation, discovery, delay_radius_samples)


def calibrate_weak_ca(
    null_records_by_stratum,
    cancellation: CaCancellationConfig,
    search: CaBitEdgeSearch,
    *,
    max_false_alarm_probability=0.01,
    joint_failure_budget=0.05,
    methods_in_family=1,
    delay_radius_samples=1,
    coarse_candidates=1,
):
    """Score each supplied null record through the entire frozen pipeline.

    Record collections may be iterators. Stratum names are calibration metadata,
    never inference inputs. Nulls must be representative IID records; neither
    this API nor an empirical gate establishes robustness to unknown RF scenes.
    """
    _validate_pipeline(cancellation, search)
    coarse_candidates = positive_int("coarse_candidates", coarse_candidates)
    if delay_radius_samples not in (0, 1, 2):
        raise ValueError("delay_radius_samples must be 0, 1 or 2")
    scores = {}
    for name, records in null_records_by_stratum.items():
        scores[name] = []
        for iq in records:
            residual = _cancel_weak(
                iq, cancellation, delay_radius_samples, coarse_candidates
            )
            candidate = acquire_ca_bit_edge(residual.iq, search)
            scores[name].append(candidate.statistic if candidate is not None else 0.0)
    detection = calibrate_ca_bit_edge(
        scores,
        search,
        max_false_alarm_probability=max_false_alarm_probability,
        joint_failure_budget=joint_failure_budget,
        methods_in_family=methods_in_family,
    )
    return CaWeakAcquisitionThreshold(
        cancellation, detection, delay_radius_samples, coarse_candidates
    )


def detect_weak_ca(iq, threshold: CaWeakAcquisitionThreshold):
    """Run the exact preprocessing/search configuration stored with calibration."""
    residual = _cancel_weak(
        iq,
        threshold.cancellation,
        threshold.delay_radius_samples,
        threshold.coarse_candidates,
    )
    return CaWeakAcquisitionResult(
        detect_ca_bit_edge(residual.iq, threshold.detection), residual
    )


def _model_hashes(coarse_candidates=1):
    from gnss_gpu import (
        ca_cancellation_cpu,
        ca_local_delay_cpu,
        bit_edge_acquisition_cpu,
    )

    modules = (ca_cancellation_cpu, ca_local_delay_cpu, bit_edge_acquisition_cpu)
    if coarse_candidates != 1:
        from gnss_gpu import ca_multiseed_cpu

        modules += (ca_multiseed_cpu,)
    return {
        "python/gnss_gpu/" + module.__name__.rsplit(".", 1)[1] + ".py": hashlib.sha256(
            Path(str(module.__file__)).read_bytes()
        ).hexdigest()
        for module in modules
    }


def load_weak_ca_experiment(directory):
    """Load a completed local-delay experiment's bound preprocessing and gate.

    Reject modified protocol/threshold files or changed model source versions.
    These checks identify a local artifact; they do not authenticate an untrusted
    report or expand the calibration's finite simulated-null scope.
    """
    from gnss_gpu.acquisition_cpu import CaSearch

    directory = Path(directory)
    protocol_path, frozen_path = (
        directory / "protocol.json",
        directory / "frozen_thresholds.json",
    )
    report = json.loads((directory / "report.json").read_text())
    for path, key in (
        (protocol_path, "protocol_sha256"),
        (frozen_path, "frozen_thresholds_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != report[key]:
            raise ValueError("calibration artifact hash mismatch")
    protocol = json.loads(protocol_path.read_text())
    coarse_candidates = protocol.get("coarse_candidates", 1)
    method = "multiseed_20" if coarse_candidates != 1 else "bit_edge_20"
    frozen = json.loads(frozen_path.read_text())[method]
    if frozen["config"] != protocol["detectors"][method]:
        raise ValueError("calibration search configuration differs")
    for name, digest in _model_hashes(coarse_candidates).items():
        if protocol["source_sha256"].get(name) != digest:
            raise ValueError(
                "calibrated model source differs; recalibrate the pipeline"
            )
    values = dict(frozen["config"])
    values["search"] = CaSearch(**values["search"])
    search = CaBitEdgeSearch(**values)
    strata = tuple(CaBitEdgeStratumCalibration(**row) for row in frozen["strata"])
    if not strata or frozen["score"] != max(row.score for row in strata):
        raise ValueError("invalid maximum stratum gate")
    detection = CaBitEdgeThreshold(
        search,
        frozen["score"],
        strata,
        frozen["max_false_alarm_probability"],
        frozen["joint_failure_budget"],
        frozen["methods_in_family"],
    )
    return CaWeakAcquisitionThreshold(
        CaCancellationConfig(**protocol["cancellation"]),
        detection,
        protocol["local_delay_radius_samples"],
        coarse_candidates,
    )
