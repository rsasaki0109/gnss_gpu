"""GPU-accelerated GNSS signal simulation.

Supports multi-constellation: GPS, GLONASS, Galileo, BeiDou, QZSS.
"""

from pathlib import Path

import numpy as np

# GNSS system constants (must match C++ GnssSystem enum)
GNSS_GPS = 0
GNSS_GLONASS = 1
GNSS_GALILEO = 2
GNSS_BEIDOU = 3
GNSS_QZSS = 4

SYSTEM_NAMES = {
    GNSS_GPS: "GPS", GNSS_GLONASS: "GLONASS",
    GNSS_GALILEO: "Galileo", GNSS_BEIDOU: "BeiDou", GNSS_QZSS: "QZSS",
}

def prn_label_to_system(label):
    """Convert PRN label like 'G05' to (system_int, prn_int)."""
    if isinstance(label, int):
        return GNSS_GPS, label
    s = str(label).strip().upper()
    if not s:
        return GNSS_GPS, 1
    prefix = s[0]
    prn = int(s[1:]) if s[1:].strip().isdigit() else 1
    mapping = {"G": GNSS_GPS, "R": GNSS_GLONASS, "E": GNSS_GALILEO,
               "C": GNSS_BEIDOU, "J": GNSS_QZSS}
    return mapping.get(prefix, GNSS_GPS), prn


class SignalSimulator:
    """CUDA-accelerated GNSS IQ signal generator."""

    def __init__(self, sampling_freq=2.6e6, intermediate_freq=0,
                 noise_floor_db=-20, noise_seed=None):
        self.sampling_freq = float(sampling_freq)
        self.intermediate_freq = float(intermediate_freq)
        self.noise_floor_db = float(noise_floor_db)
        self.noise_seed = None if noise_seed is None else int(noise_seed)

    def generate_epoch(self, channels, n_samples=None):
        """Generate composite IQ signal for one epoch.

        Args:
            channels: List of dicts with keys:
                prn, code_phase, carrier_phase, doppler_hz, amplitude, nav_bit
            n_samples: Number of samples (default: 1ms worth).

        Returns:
            float32 array of shape [2*n_samples] with interleaved I/Q.
        """
        from gnss_gpu._gnss_gpu_signal_sim import generate_signal

        if n_samples is None:
            n_samples = int(self.sampling_freq * 1e-3)

        return generate_signal(
            self.sampling_freq, self.intermediate_freq,
            channels, int(n_samples), self.noise_floor_db,
            0 if self.noise_seed is None else self.noise_seed)

    def generate_test_signal(self, prn, code_phase=0, doppler=0,
                             cn0_dbhz=45, duration_s=1e-3):
        """Generate single-satellite test signal with noise.

        Args:
            prn: PRN number (1-32).
            code_phase: Code phase in chips.
            doppler: Doppler shift in Hz.
            cn0_dbhz: Carrier-to-noise ratio in dB-Hz.
            duration_s: Duration in seconds.

        Returns:
            float32 array of interleaved I/Q samples.
        """
        n_samples = max(1, int(self.sampling_freq * duration_s))
        channels = [{
            "prn": int(prn),
            "code_phase": float(code_phase),
            "carrier_phase": 0.0,
            "doppler_hz": float(doppler),
            "amplitude": 1.0,
            "nav_bit": 1,
        }]
        from gnss_gpu._gnss_gpu_signal_sim import generate_signal

        return generate_signal(
            self.sampling_freq, self.intermediate_freq,
            channels, n_samples, -float(cn0_dbhz),
            0 if self.noise_seed is None else self.noise_seed)

    @staticmethod
    def write_bin(iq_data, path, fmt="int8"):
        """Write IQ data to binary file.

        Args:
            iq_data: float32 array of interleaved I/Q.
            path: Output file path.
            fmt: 'int8' (HackRF), 'int16' (USRP), or 'float32' (GnuRadio).
        """
        arr = np.asarray(iq_data, dtype=np.float32).ravel()
        if fmt == "int8":
            data = np.clip(np.rint(arr * 127.0), -127, 127).astype(np.int8)
        elif fmt == "int16":
            data = np.clip(np.rint(arr * 32767.0), -32767, 32767).astype(np.int16)
        elif fmt == "float32":
            data = arr
        else:
            raise ValueError(f"Unknown format: {fmt}")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.tofile(str(path))
