"""3D city model validation using GNSS LOS/NLOS predictions vs observed C/N0.

Compares PLATEAU 3D model predictions (LOS/NLOS via BVH ray-tracing) against
actual received signal quality (C/N0 from RINEX observations) to detect:

  - Missing buildings: model predicts LOS but C/N0 is low (new construction)
  - Phantom buildings: model predicts NLOS but C/N0 is high (demolished building)
  - Consistent: prediction matches observation

Inspired by Kim et al. (Georgia Tech, 2008) "Localization and 3D Reconstruction
of Urban Scenes Using GPS" — which uses SNR drops to reconstruct buildings.
This module does the inverse: validates existing 3D models against GNSS signals.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from gnss_gpu.urban_signal_sim import ecef_to_lla, _sat_elevation_azimuth


# C/N0 thresholds for LOS/NLOS classification from observations
CN0_LOS_THRESHOLD = 30.0      # dB-Hz: above this → likely LOS
CN0_NLOS_THRESHOLD = 25.0     # dB-Hz: below this → likely NLOS
ELEVATION_MASK_DEG = 10.0     # ignore satellites below this


@dataclass
class SatValidation:
    """Validation result for one satellite at one epoch."""
    prn: str
    elevation_deg: float
    azimuth_deg: float
    predicted_los: bool        # from 3D model ray-tracing
    observed_cn0: float        # from RINEX [dB-Hz]
    status: str                # "consistent", "missing_building", "phantom_building", "ambiguous"


@dataclass
class EpochValidation:
    """Validation result for one epoch."""
    time: object
    rx_ecef: np.ndarray
    satellites: list = field(default_factory=list)
    n_consistent: int = 0
    n_missing: int = 0         # model says LOS but signal is NLOS
    n_phantom: int = 0         # model says NLOS but signal is LOS
    n_ambiguous: int = 0
    model_score: float = 0.0   # consistency score [0-1]


def classify_satellite(predicted_los, observed_cn0, elevation_deg):
    """Classify model-vs-observation consistency for one satellite.

    Returns one of:
      "consistent"       — prediction matches observation
      "missing_building" — model says LOS, but signal is weak (building not in model)
      "phantom_building" — model says NLOS, but signal is strong (building removed)
      "ambiguous"        — signal quality is borderline, can't decide
    """
    if elevation_deg < ELEVATION_MASK_DEG:
        return "ambiguous"

    if observed_cn0 != observed_cn0:  # NaN check
        return "ambiguous"

    if predicted_los:
        if observed_cn0 >= CN0_LOS_THRESHOLD:
            return "consistent"         # LOS predicted, strong signal ✓
        elif observed_cn0 < CN0_NLOS_THRESHOLD:
            return "missing_building"   # LOS predicted, weak signal ✗
        else:
            return "ambiguous"
    else:
        if observed_cn0 < CN0_NLOS_THRESHOLD:
            return "consistent"         # NLOS predicted, weak signal ✓
        elif observed_cn0 >= CN0_LOS_THRESHOLD:
            return "phantom_building"   # NLOS predicted, strong signal ✗
        else:
            return "ambiguous"


def validate_epoch(rx_ecef, sat_ecef, prn_labels, cn0_values,
                   building_model, time=None):
    """Validate 3D model at one epoch.

    Args:
        rx_ecef: [3] receiver ECEF position.
        sat_ecef: [n_sat, 3] satellite ECEF positions.
        prn_labels: list of PRN strings (e.g. ["G05", "G12", ...]).
        cn0_values: [n_sat] observed C/N0 in dB-Hz (NaN if unavailable).
        building_model: BuildingModel or BVHAccelerator with check_los().
        time: epoch timestamp (for reporting).

    Returns:
        EpochValidation
    """
    n_sat = len(prn_labels)
    rx = np.asarray(rx_ecef, dtype=np.float64).ravel()
    sats = np.asarray(sat_ecef, dtype=np.float64).reshape(-1, 3)

    # Elevation / azimuth
    el, az = _sat_elevation_azimuth(rx, sats)

    # LOS prediction from 3D model
    is_los = building_model.check_los(rx, sats)
    is_los = np.asarray(is_los, dtype=bool)

    result = EpochValidation(time=time, rx_ecef=rx)

    for i in range(n_sat):
        el_deg = math.degrees(el[i])
        az_deg = math.degrees(az[i])
        cn0 = float(cn0_values[i])
        pred_los = bool(is_los[i])

        status = classify_satellite(pred_los, cn0, el_deg)

        result.satellites.append(SatValidation(
            prn=prn_labels[i],
            elevation_deg=el_deg,
            azimuth_deg=az_deg,
            predicted_los=pred_los,
            observed_cn0=cn0,
            status=status,
        ))

        if status == "consistent":
            result.n_consistent += 1
        elif status == "missing_building":
            result.n_missing += 1
        elif status == "phantom_building":
            result.n_phantom += 1
        else:
            result.n_ambiguous += 1

    n_decided = result.n_consistent + result.n_missing + result.n_phantom
    result.model_score = result.n_consistent / max(1, n_decided)

    return result
