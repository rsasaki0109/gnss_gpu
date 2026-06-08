"""GPU-accelerated GNSS tracking loops (scalar and vector)."""

import numpy as np

try:
    from gnss_gpu._gnss_gpu_tracking import (
        TrackingConfig as _TrackingConfig,
        ChannelState as _ChannelState,
        batch_correlate as _batch_correlate,
        scalar_tracking_update as _scalar_tracking_update,
        vector_tracking_update as _vector_tracking_update,
        cn0_nwpr as _cn0_nwpr,
    )
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


# GPS L1 C/A constants
CA_CODE_RATE = 1.023e6      # chips/s
CA_CODE_LENGTH = 1023       # chips
GPS_L1_FREQ = 1575.42e6     # Hz


def _finite_float(name, value):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _positive_float(name, value):
    out = _finite_float(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative_float(name, value):
    out = _finite_float(name, value)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _validate_prn(name, value):
    try:
        prn = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if prn != value:
        raise ValueError(f"{name} must be an integer")
    if prn < 1 or prn > 32:
        raise ValueError(f"{name} must be in [1, 32]")
    return prn


def _as_1d_array(name, values, dtype):
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return arr


def _as_prn_array(prn_list):
    arr = np.asarray(prn_list)
    if arr.ndim != 1:
        raise ValueError("prn_list must be 1-D")
    if arr.size == 0:
        raise ValueError("prn_list must contain at least one PRN")
    if np.issubdtype(arr.dtype, np.integer):
        prns = arr.astype(np.int64, copy=False)
    else:
        try:
            numeric = arr.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("prn_list must contain integers") from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("prn_list must contain integers")
        if not np.all(numeric == np.floor(numeric)):
            raise ValueError("prn_list must contain integers")
        prns = numeric.astype(np.int64)
    for prn in prns:
        _validate_prn("prn_list values", prn)
    return prns


def _finite_1d_array(name, values, dtype, *, min_size=1):
    arr = _as_1d_array(name, values, dtype)
    if arr.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_signal_block(signal_block):
    arr = np.asarray(signal_block, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("signal_block must be 1-D")
    if arr.size == 0:
        raise ValueError("signal_block must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("signal_block must be finite")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _validate_config_values(config):
    sampling_freq = _positive_float("sampling_freq", config.sampling_freq)
    intermediate_freq = _finite_float("intermediate_freq", config.intermediate_freq)
    integration_time = _positive_float("integration_time", config.integration_time)
    dll_bandwidth = _nonnegative_float("dll_bandwidth", config.dll_bandwidth)
    pll_bandwidth = _nonnegative_float("pll_bandwidth", config.pll_bandwidth)
    correlator_spacing = _positive_float("correlator_spacing", config.correlator_spacing)
    return (
        sampling_freq,
        intermediate_freq,
        integration_time,
        dll_bandwidth,
        pll_bandwidth,
        correlator_spacing,
    )


def _native_config(config):
    if _HAS_GPU and isinstance(config, _TrackingConfig):
        _validate_config_values(config)
        return config
    values = _validate_config_values(config)
    if not _HAS_GPU:
        raise RuntimeError("GPU bindings not available")
    return _TrackingConfig(*values)


def _as_matrix(name, values, rows, cols):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (rows, cols):
        raise ValueError(f"{name} must have shape ({rows}, {cols})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(arr, dtype=np.float64)


class TrackingConfig:
    """Configuration for tracking loops."""

    def __init__(self, sampling_freq=4.092e6, intermediate_freq=4.092e6,
                 integration_time=1e-3, dll_bandwidth=2.0,
                 pll_bandwidth=15.0, correlator_spacing=0.5):
        self.sampling_freq = _positive_float("sampling_freq", sampling_freq)
        self.intermediate_freq = _finite_float("intermediate_freq", intermediate_freq)
        self.integration_time = _positive_float("integration_time", integration_time)
        self.dll_bandwidth = _nonnegative_float("dll_bandwidth", dll_bandwidth)
        self.pll_bandwidth = _nonnegative_float("pll_bandwidth", pll_bandwidth)
        self.correlator_spacing = _positive_float("correlator_spacing", correlator_spacing)


def ChannelState(prn, code_phase=0.0, code_freq=CA_CODE_RATE,
                 carrier_phase=0.0, carrier_freq=0.0, cn0=0.0,
                 dll_integrator=0.0, pll_integrator=0.0, locked=True):
    """Create a ChannelState using the C++ bound type if available."""
    prn = _validate_prn("prn", prn)
    code_phase = _finite_float("code_phase", code_phase)
    code_freq = _positive_float("code_freq", code_freq)
    carrier_phase = _finite_float("carrier_phase", carrier_phase)
    carrier_freq = _finite_float("carrier_freq", carrier_freq)
    cn0 = _finite_float("cn0", cn0)
    dll_integrator = _finite_float("dll_integrator", dll_integrator)
    pll_integrator = _finite_float("pll_integrator", pll_integrator)
    if _HAS_GPU:
        ch = _ChannelState()
        ch.prn = prn
        ch.code_phase = code_phase
        ch.code_freq = code_freq
        ch.carrier_phase = carrier_phase
        ch.carrier_freq = carrier_freq
        ch.cn0 = cn0
        ch.dll_integrator = dll_integrator
        ch.pll_integrator = pll_integrator
        ch.locked = locked
        return ch
    else:
        raise RuntimeError("GPU bindings not available")


class ScalarTracker:
    """Scalar tracking loop: independent DLL/PLL per channel."""

    def __init__(self, config):
        if isinstance(config, dict):
            self.config = TrackingConfig(**config)
        else:
            self.config = config
        self.channels = []
        self._corr_history = []
        self._max_hist = 20  # for CN0 estimation

    def initialize(self, prn_list, code_phases, doppler_freqs):
        """Initialize tracking channels.

        Args:
            prn_list: list of PRN numbers
            code_phases: initial code phases in chips
            doppler_freqs: initial Doppler frequencies in Hz
        """
        prns = _as_prn_array(prn_list)
        code_phases = _finite_1d_array("code_phases", code_phases, np.float64)
        doppler_freqs = _finite_1d_array("doppler_freqs", doppler_freqs, np.float64)
        if code_phases.size != prns.size:
            raise ValueError("code_phases length must match prn_list")
        if doppler_freqs.size != prns.size:
            raise ValueError("doppler_freqs length must match prn_list")

        self.channels = []
        for prn, cp, df in zip(prns, code_phases, doppler_freqs):
            ch = ChannelState(
                prn=prn,
                code_phase=cp,
                code_freq=CA_CODE_RATE + df * CA_CODE_RATE / GPS_L1_FREQ,
                carrier_phase=0.0,
                carrier_freq=self.config.intermediate_freq + df,
                locked=True,
            )
            self.channels.append(ch)
        self._corr_history = []

    def process(self, signal_block):
        """Process one integration period.

        Args:
            signal_block: numpy array of IF samples (float32)

        Returns:
            correlations: numpy array [n_channels, 6] (EI,EQ,PI,PQ,LI,LQ)
        """
        if len(self.channels) == 0:
            raise ValueError("tracker has no initialized channels")
        signal_block = _as_signal_block(signal_block)
        n_ch = len(self.channels)
        cfg = _native_config(self.config)

        correlations = _batch_correlate(
            signal_block, self.channels, n_ch, len(signal_block), cfg
        )

        _scalar_tracking_update(self.channels, correlations, n_ch, cfg)

        # Store history for CN0
        self._corr_history.append(correlations.copy())
        if len(self._corr_history) > self._max_hist:
            self._corr_history.pop(0)

        # Update CN0 if enough history
        if len(self._corr_history) >= 10:
            hist = np.array(self._corr_history)
            cn0_vals = _cn0_nwpr(hist, n_ch, len(self._corr_history),
                                 cfg.integration_time)
            for i, ch in enumerate(self.channels):
                ch.cn0 = cn0_vals[i]

        return correlations.reshape(n_ch, 6)


class VectorTracker(ScalarTracker):
    """Vector tracking loop: EKF-based navigation-aided tracking."""

    def __init__(self, config, initial_pos_ecef):
        super().__init__(config)
        initial_pos_ecef = _finite_1d_array(
            "initial_pos_ecef", initial_pos_ecef, np.float64)
        if initial_pos_ecef.size != 3:
            raise ValueError("initial_pos_ecef must have shape (3,)")
        self.nav_state = np.zeros(8, dtype=np.float64)
        self.nav_state[:3] = initial_pos_ecef
        self.nav_cov = np.eye(8, dtype=np.float64) * 100.0
        # Larger initial uncertainty for velocity and clock
        self.nav_cov[3, 3] = 10.0
        self.nav_cov[4, 4] = 10.0
        self.nav_cov[5, 5] = 10.0
        self.nav_cov[6, 6] = 1e6   # clock bias
        self.nav_cov[7, 7] = 1e4   # clock drift

    def process(self, signal_block, sat_ecef, sat_vel):
        """Process one block with vector tracking.

        Args:
            signal_block: numpy array of IF samples (float32)
            sat_ecef: numpy array [n_channels, 3] satellite ECEF positions
            sat_vel: numpy array [n_channels, 3] satellite ECEF velocities

        Returns:
            nav_solution: numpy array [8] (x,y,z,vx,vy,vz,cb,cd)
        """
        if len(self.channels) == 0:
            raise ValueError("tracker has no initialized channels")
        signal_block = _as_signal_block(signal_block)
        n_ch = len(self.channels)
        sat_ecef = _as_matrix("sat_ecef", sat_ecef, n_ch, 3)
        sat_vel = _as_matrix("sat_vel", sat_vel, n_ch, 3)
        cfg = _native_config(self.config)

        correlations = _batch_correlate(
            signal_block, self.channels, n_ch, len(signal_block), cfg
        )

        _vector_tracking_update(
            self.channels, correlations, sat_ecef, sat_vel,
            self.nav_state, self.nav_cov.ravel(),
            n_ch, cfg, cfg.integration_time
        )

        return self.nav_state.copy()
