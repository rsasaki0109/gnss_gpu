# ruff: noqa: E402
import os
import sys

from gnss_gpu._version import __version__

_DLL_DIR_HANDLES = []


def _register_windows_dll_dirs():
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    package_dir = os.path.dirname(__file__)
    candidate_dirs = [package_dir]

    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env_var)
        if root:
            candidate_dirs.append(os.path.join(root, "bin"))

    seen = set()
    for dll_dir in candidate_dirs:
        dll_dir = os.path.abspath(dll_dir)
        if dll_dir in seen or not os.path.isdir(dll_dir):
            continue
        seen.add(dll_dir)
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(dll_dir))
        except OSError:
            pass


_register_windows_dll_dirs()

try:
    from gnss_gpu._gnss_gpu import (
        ecef_to_lla,
        lla_to_ecef,
        satellite_azel,
        wls_position,
        wls_batch,
    )
except ImportError:
    pass

from gnss_gpu.io import read_rinex_obs, parse_nmea

# Module imports
from gnss_gpu.raytrace import BuildingModel
from gnss_gpu.bvh import BVHAccelerator
from gnss_gpu.multipath import MultipathSimulator
from gnss_gpu.skyplot import VulnerabilityMap
from gnss_gpu.acquisition import Acquisition
from gnss_gpu.interference import InterferenceDetector
from gnss_gpu.tracking import ScalarTracker, VectorTracker
from gnss_gpu.particle_filter import ParticleFilter
from gnss_gpu.particle_filter_device import ParticleFilterDevice
from gnss_gpu.svgd import SVGDParticleFilter
from gnss_gpu.particle_filter_3d import ParticleFilter3D
from gnss_gpu.ephemeris import Ephemeris
from gnss_gpu.rtk import RTKSolver
from gnss_gpu.cycle_slip import detect_geometry_free, detect_melbourne_wubbena, detect_time_difference
from gnss_gpu.io.nmea_writer import NMEAWriter
from gnss_gpu.atmosphere import AtmosphereCorrection
from gnss_gpu.sbas import SBASCorrection, QZSSAugmentation
from gnss_gpu.multi_gnss import MultiGNSSSolver
from gnss_gpu.multi_gnss_quality import (
    MultiGNSSQualityDecision,
    MultiGNSSQualityMetrics,
    MultiGNSSQualityVetoConfig,
    accept_multi_gnss_solution,
    compute_multi_gnss_quality_metrics,
    select_multi_gnss_solution,
)
from gnss_gpu.ekf import EKFPositioner
from gnss_gpu.raim import raim_check, raim_fde
from gnss_gpu.doppler import doppler_velocity, doppler_velocity_batch
from gnss_gpu.fgo import fgo_gnss_lm, fgo_gnss_lm_vd
from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.urban_signal_sim import UrbanSignalSimulator
from gnss_gpu.e2e_helpers import (
    compute_e2e_wls_weights,
    acquisition_lag_to_code_phase_chips,
    code_phase_chips_to_acquisition_lag,
    refine_acquisition_code_lag_dll,
    refine_acquisition_code_lags_dll_batch,
    refine_acquisition_code_lags_diagnostic_batch,
    dump_e2e_diagnostics_csv,
    pseudorange_to_code_phase_chips,
    acquisition_code_phase_to_pseudorange,
)

__all__ = [
    # Core positioning
    "ecef_to_lla",
    "lla_to_ecef",
    "satellite_azel",
    "wls_position",
    "wls_batch",
    "fgo_gnss_lm",
    "fgo_gnss_lm_vd",
    # I/O
    "read_rinex_obs",
    "parse_nmea",
    # Ray tracing
    "BuildingModel",
    # BVH-accelerated ray tracing
    "BVHAccelerator",
    # Multipath
    "MultipathSimulator",
    # Vulnerability map
    "VulnerabilityMap",
    # Signal acquisition
    "Acquisition",
    # Interference detection
    "InterferenceDetector",
    # Tracking
    "ScalarTracker",
    "VectorTracker",
    # Particle filter
    "ParticleFilter",
    # SVGD particle filter
    "SVGDParticleFilter",
    # 3D particle filter
    "ParticleFilter3D",
    # Ephemeris
    "Ephemeris",
    # RTK
    "RTKSolver",
    # Cycle slip detection
    "detect_geometry_free",
    "detect_melbourne_wubbena",
    "detect_time_difference",
    # NMEA output
    "NMEAWriter",
    # Atmosphere corrections
    "AtmosphereCorrection",
    # SBAS / QZSS augmentation
    "SBASCorrection",
    "QZSSAugmentation",
    # Multi-GNSS positioning
    "MultiGNSSSolver",
    "MultiGNSSQualityDecision",
    "MultiGNSSQualityMetrics",
    "MultiGNSSQualityVetoConfig",
    "accept_multi_gnss_solution",
    "compute_multi_gnss_quality_metrics",
    "select_multi_gnss_solution",
    # EKF positioning
    "EKFPositioner",
    # RAIM / FDE integrity monitoring
    "raim_check",
    "raim_fde",
    # Doppler velocity estimation
    "doppler_velocity",
    "doppler_velocity_batch",
    # Signal simulation
    "SignalSimulator",
    "UrbanSignalSimulator",
    # E2E helpers (acquisition to pseudorange)
    "compute_e2e_wls_weights",
    "acquisition_lag_to_code_phase_chips",
    "code_phase_chips_to_acquisition_lag",
    "refine_acquisition_code_lag_dll",
    "refine_acquisition_code_lags_dll_batch",
    "refine_acquisition_code_lags_diagnostic_batch",
    "dump_e2e_diagnostics_csv",
    "pseudorange_to_code_phase_chips",
    "acquisition_code_phase_to_pseudorange",
]
