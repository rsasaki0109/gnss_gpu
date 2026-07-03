"""GPU-accelerated GNSS signal simulation.

Supports multi-constellation: GPS, GLONASS, Galileo, BeiDou, QZSS.
"""

from pathlib import Path

import numpy as np

from gnss_gpu.input_validation import finite_float, positive_float, positive_int

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
    """Convert PRN label like 'G05' to (system_int, prn_int).

    Raises ValueError for malformed labels.
    """
    if isinstance(label, int):
        return GNSS_GPS, label
    s = str(label).strip().upper()
    if len(s) < 2:
        raise ValueError(f"Malformed PRN label: {label!r} (need e.g. 'G05')")
    prefix = s[0]
    mapping = {"G": GNSS_GPS, "R": GNSS_GLONASS, "E": GNSS_GALILEO,
               "C": GNSS_BEIDOU, "J": GNSS_QZSS}
    if prefix not in mapping:
        raise ValueError(f"Unknown GNSS system prefix: {prefix!r} in {label!r}")
    prn_str = s[1:].strip()
    if not prn_str.isdigit():
        raise ValueError(f"Invalid PRN number in label: {label!r}")
    return mapping[prefix], int(prn_str)


_REQUIRED_CHANNEL_KEYS = (
    "prn", "code_phase", "carrier_phase", "doppler_hz", "amplitude", "nav_bit",
)


def _validate_signal_sim_config(
    sampling_freq,
    intermediate_freq,
    noise_floor_db,
    noise_seed=None,
):
    positive_float("sampling_freq", sampling_freq)
    finite_float("intermediate_freq", intermediate_freq)
    finite_float("noise_floor_db", noise_floor_db)
    if noise_seed is not None:
        try:
            int(noise_seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("noise_seed must be an integer") from exc


def _validate_channel_dict(channel, index):
    if not isinstance(channel, dict):
        raise ValueError(f"channels[{index}] must be a dict")
    for key in _REQUIRED_CHANNEL_KEYS:
        if key not in channel:
            raise ValueError(f"channels[{index}] missing required key '{key}'")
    positive_int(f"channels[{index}].prn", channel["prn"])
    finite_float(f"channels[{index}].code_phase", channel["code_phase"])
    finite_float(f"channels[{index}].carrier_phase", channel["carrier_phase"])
    finite_float(f"channels[{index}].doppler_hz", channel["doppler_hz"])
    finite_float(f"channels[{index}].amplitude", channel["amplitude"])
    try:
        int(channel["nav_bit"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"channels[{index}].nav_bit must be an integer") from exc
    if "system" in channel:
        try:
            system = int(channel["system"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"channels[{index}].system must be an integer") from exc
        if system < 0:
            raise ValueError(f"channels[{index}].system must be non-negative")
    if "nav_bit_rate" in channel:
        positive_float(f"channels[{index}].nav_bit_rate", channel["nav_bit_rate"])


def _validate_channels(channels):
    if not isinstance(channels, list):
        raise ValueError("channels must be a list of dicts")
    for index, channel in enumerate(channels):
        _validate_channel_dict(channel, index)


class SignalSimulator:
    """CUDA-accelerated GNSS IQ signal generator."""

    def __init__(self, sampling_freq=2.6e6, intermediate_freq=0,
                 noise_floor_db=-20, noise_seed=None):
        _validate_signal_sim_config(
            sampling_freq, intermediate_freq, noise_floor_db, noise_seed,
        )
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
        _validate_channels(channels)
        if n_samples is None:
            n_samples = int(self.sampling_freq * 1e-3)
        else:
            positive_int("n_samples", n_samples)

        from gnss_gpu._gnss_gpu_signal_sim import generate_signal

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
        positive_int("prn", prn)
        finite_float("code_phase", code_phase)
        finite_float("doppler", doppler)
        finite_float("cn0_dbhz", cn0_dbhz)
        positive_float("duration_s", duration_s)

        n_samples = max(1, int(self.sampling_freq * duration_s))
        channels = [{
            "prn": int(prn),
            "code_phase": float(code_phase),
            "carrier_phase": 0.0,
            "doppler_hz": float(doppler),
            "amplitude": 1.0,
            "nav_bit": 1,
        }]
        _validate_channels(channels)
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
        if arr.size == 0:
            raise ValueError("iq_data must be non-empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("iq_data must be finite")
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
