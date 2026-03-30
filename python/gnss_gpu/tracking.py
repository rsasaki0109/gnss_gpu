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


class TrackingConfig:
    """Configuration for tracking loops."""

    def __init__(self, sampling_freq=4.092e6, intermediate_freq=4.092e6,
                 integration_time=1e-3, dll_bandwidth=2.0,
                 pll_bandwidth=15.0, correlator_spacing=0.5):
        self.sampling_freq = sampling_freq
        self.intermediate_freq = intermediate_freq
        self.integration_time = integration_time
        self.dll_bandwidth = dll_bandwidth
        self.pll_bandwidth = pll_bandwidth
        self.correlator_spacing = correlator_spacing


def ChannelState(prn, code_phase=0.0, code_freq=CA_CODE_RATE,
                 carrier_phase=0.0, carrier_freq=0.0, cn0=0.0,
                 dll_integrator=0.0, pll_integrator=0.0, locked=True):
    """Create a ChannelState using the C++ bound type if available."""
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
        self.channels = []
        for i, prn in enumerate(prn_list):
            cp = code_phases[i] if i < len(code_phases) else 0.0
            df = doppler_freqs[i] if i < len(doppler_freqs) else 0.0
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
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available")

        signal_block = np.asarray(signal_block, dtype=np.float32)
        n_ch = len(self.channels)

        correlations = _batch_correlate(
            signal_block, self.channels, n_ch, len(signal_block), self.config
        )

        _scalar_tracking_update(self.channels, correlations, n_ch, self.config)

        # Store history for CN0
        self._corr_history.append(correlations.copy())
        if len(self._corr_history) > self._max_hist:
            self._corr_history.pop(0)

        # Update CN0 if enough history
        if len(self._corr_history) >= 10:
            hist = np.array(self._corr_history)
            cn0_vals = _cn0_nwpr(hist, n_ch, len(self._corr_history),
                                 self.config.integration_time)
            for i, ch in enumerate(self.channels):
                ch.cn0 = cn0_vals[i]

        return correlations.reshape(n_ch, 6)


class VectorTracker(ScalarTracker):
    """Vector tracking loop: EKF-based navigation-aided tracking."""

    def __init__(self, config, initial_pos_ecef):
        super().__init__(config)
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
        if not _HAS_GPU:
            raise RuntimeError("GPU bindings not available")

        signal_block = np.asarray(signal_block, dtype=np.float32)
        sat_ecef = np.asarray(sat_ecef, dtype=np.float64).ravel()
        sat_vel = np.asarray(sat_vel, dtype=np.float64).ravel()
        n_ch = len(self.channels)

        correlations = _batch_correlate(
            signal_block, self.channels, n_ch, len(signal_block), self.config
        )

        _vector_tracking_update(
            self.channels, correlations, sat_ecef, sat_vel,
            self.nav_state, self.nav_cov.ravel(),
            n_ch, self.config, self.config.integration_time
        )

        return self.nav_state.copy()
