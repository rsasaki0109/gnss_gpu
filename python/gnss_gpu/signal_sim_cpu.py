"""Explicit CPU GPS L1 C/A IQ synthesis with Numba sample parallelism."""

from functools import lru_cache
import math

import numpy as np

from gnss_gpu.input_validation import finite_float, positive_float, positive_int
from gnss_gpu.signal_sim import GNSS_GPS, SignalSimulator, _validate_channels
from gnss_gpu.validation.tracking_residuals import generate_ca_code

L1_HZ = 1575.42e6
CHIP_RATE = 1.023e6


@lru_cache(maxsize=1)
def _kernel():
    # Keep Numba optional for users importing the CUDA-only package.
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def synthesize(codes, params, bits, offsets, fs, intermediate, start, count):
        out = np.empty((count, 2), dtype=np.float32)
        chip_steps = np.empty(len(params), dtype=np.float64)
        angular_rates = np.empty(len(params), dtype=np.float64)
        for ch in range(len(params)):
            doppler = params[ch, 2]
            chip_steps[ch] = (CHIP_RATE * (1.0 + doppler / L1_HZ)) / fs
            angular_rates[ch] = 2.0 * math.pi * (intermediate + doppler)
        for sample in prange(count):
            t = (start + sample) / fs
            real, imag = 0.0, 0.0
            # Fixed channel order avoids parallel floating-point reductions.
            for ch in range(len(params)):
                code_phase, phase, doppler, amplitude, nav, rate = params[ch]
                chip_step = chip_steps[ch]
                chip = int(math.floor(code_phase + chip_step * (start + sample))) % 1023
                if offsets[ch + 1] > offsets[ch]:
                    nav = bits[offsets[ch] + int(math.floor(t * rate))]
                angle = phase + angular_rates[ch] * t
                value = amplitude * nav * codes[ch, chip]
                real += value * math.cos(angle)
                imag += value * math.sin(angle)
            out[sample, 0] = real
            out[sample, 1] = imag
        return out

    return synthesize


@lru_cache(maxsize=1)
def _batch_kernel():
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def synthesize(codes, code_ids, params, bits, nav_offsets, epoch_offsets, starts,
                   fs, intermediate, count):
        epochs = len(epoch_offsets)-1
        out = np.empty((epochs, 2*count), dtype=np.float32)
        chip_steps = np.empty(len(params), dtype=np.float64)
        angular_rates = np.empty(len(params), dtype=np.float64)
        for ch in range(len(params)):
            doppler = params[ch, 2]
            chip_steps[ch] = (CHIP_RATE * (1.0 + doppler / L1_HZ)) / fs
            angular_rates[ch] = 2.0 * math.pi * (intermediate + doppler)
        # Parallelize all independent output samples with a single dispatch.
        # Each epoch sums its own channels in input order. Absolute sample
        # offsets match repeated generate_epoch(..., start_sample=...) calls.
        tiles = (count+255)//256
        for index in prange(epochs*tiles):
            epoch = index//tiles
            first_sample = (index-epoch*tiles)*256
            for sample in range(first_sample, min(first_sample+256, count)):
                absolute_sample = starts[epoch]+sample
                t = absolute_sample/fs
                real, imag = 0.0, 0.0
                for ch in range(epoch_offsets[epoch], epoch_offsets[epoch+1]):
                    code_phase, phase, doppler, amplitude, nav, rate = params[ch]
                    chip = int(math.floor(code_phase+chip_steps[ch]*absolute_sample)) % 1023
                    if nav_offsets[ch+1] > nav_offsets[ch]:
                        nav = bits[nav_offsets[ch]+int(math.floor(t*rate))]
                    angle = phase+angular_rates[ch]*t
                    value = amplitude*nav*codes[code_ids[ch], chip]
                    real += value*math.cos(angle)
                    imag += value*math.sin(angle)
                out[epoch, 2*sample] = real
                out[epoch, 2*sample+1] = imag
        return out

    return synthesize


def _sample_index(name, value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        index = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if index != value or index < 0 or index > 2**53 - 1:
        raise ValueError(f"{name} must be a non-negative integer below 2**53")
    return index


def _noise_seed(value):
    if value is None:
        return None
    try:
        seed = int(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("noise_seed must be a uint64 integer") from exc
    if isinstance(value, (bool, np.bool_)) or seed != value or not 0 <= seed < 2**64:
        raise ValueError("noise_seed must be a uint64 integer")
    return seed


def _add_noise(out, noise_db, seed):
    """Apply the shared per-I/Q noise convention to a float32 output block."""
    sigma = 10.0 ** (noise_db / 20.0) if noise_db < 770 else float("inf")
    if not np.isfinite(sigma) or sigma > np.finfo(np.float32).max:
        raise ValueError("noise_floor_db exceeds float32 range")
    if sigma:
        rng = np.random.default_rng(None if seed in (None, 0) else seed)
        out += rng.normal(0.0, sigma, out.shape).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise ValueError("composite IQ exceeds float32 range")
    return out


class CpuSignalSimulator(SignalSimulator):
    """Generate GPS PRNs 1–32 on CPU using the SignalSimulator channel format.

    Positive code_phase advances the local code; carrier_phase is in radians.
    Carrier Doppler also scales the code rate by (1 + doppler / L1 frequency).
    nav_bits, when supplied, must cover the entire requested interval at
    nav_bit_rate (default 50 bit/s); otherwise nav_bit is held constant.

    start_sample addresses samples relative to the supplied phases/nav_bits at
    time zero. Unchanged channels therefore produce continuous noiseless blocks.
    Parameters stay constant within a block; this is not an orbit/clock model.
    noise_floor_db specifies per-component noise standard deviation in dB,
    matching SignalSimulator. Fixed nonzero seeds repeat noise per call, not
    across streaming block boundaries. Numba thread count is controlled with
    NUMBA_NUM_THREADS or numba.set_num_threads, without changing sample results.
    """

    def __init__(self, sampling_freq=2.6e6, intermediate_freq=0.0,
                 noise_floor_db=-20.0, noise_seed=None):
        super().__init__(sampling_freq, intermediate_freq, noise_floor_db, _noise_seed(noise_seed))

    def generate_epoch(self, channels, n_samples=None, *, start_sample=0):
        n = max(1, int(self.sampling_freq * 1e-3)) if n_samples is None else positive_int("n_samples", n_samples)
        start = _sample_index("start_sample", start_sample)
        return self._render(channels, n, start, self.noise_floor_db, self.noise_seed)

    def _generate_signal(self, channels, n_samples, noise_floor_db):
        return self._render(channels, n_samples, 0, noise_floor_db, self.noise_seed)

    def _prepare_channels(self, channels, n, start, *, code_bank=None):
        """Validate/pack one epoch; optionally return IDs into a shared PRN bank."""
        _validate_channels(channels)
        if start + n > 2**53 - 1:
            raise ValueError("sample interval must end below 2**53")
        codes = (np.empty((len(channels), 1023), dtype=np.float64) if code_bank is None
                 else np.empty(len(channels), dtype=np.int64))
        params = np.empty((len(channels), 6), dtype=np.float64)
        bits, offsets = [], [0]
        last_time = (start + n - 1) / self.sampling_freq
        for i, ch in enumerate(channels):
            if ch.get("system", GNSS_GPS) != GNSS_GPS:
                raise ValueError("CPU synthesis supports GPS L1 C/A only")
            prn = positive_int("prn", ch["prn"])
            if prn > 32:
                raise ValueError("prn must be in 1..32")
            nav = ch["nav_bit"]
            if nav not in (-1, 1):
                raise ValueError("nav_bit must be -1 or +1")
            rate = positive_float("nav_bit_rate", ch.get("nav_bit_rate", 50.0))
            sequence = np.asarray(ch.get("nav_bits", []))
            if sequence.ndim != 1:
                raise ValueError("nav_bits must be a 1-D sequence of -1 or +1")
            if sequence.size:
                # A two-value domain needs no sorting/set-membership machinery.
                # Most short-block callers omit nav_bits entirely; avoid array
                # membership work on that path while keeping coverage checks.
                if np.any((sequence != -1) & (sequence != 1)):
                    raise ValueError("nav_bits must be a 1-D sequence of -1 or +1")
                if not np.isfinite(last_time * rate) or math.floor(last_time * rate) >= sequence.size:
                    raise ValueError("nav_bits do not cover the requested sample interval")
                bits.extend(sequence.tolist())
            offsets.append(len(bits))
            if code_bank is None:
                codes[i] = generate_ca_code(prn)
            else:
                table, copied = code_bank
                code_id = prn-1
                if code_id not in copied:
                    table[code_id] = generate_ca_code(prn)
                    copied.add(code_id)
                codes[i] = code_id
            params[i] = [ch["code_phase"], ch["carrier_phase"], ch["doppler_hz"],
                         ch["amplitude"], nav, rate]
        # Reject overflow before entering JIT integer indexing or float32 output.
        with np.errstate(over="ignore", invalid="ignore"):
            chip_steps = (CHIP_RATE * (1.0 + params[:, 2] / L1_HZ)) / self.sampling_freq
            code_end = params[:, 0] + chip_steps * (start + n - 1)
            phase_end = params[:, 1] + 2 * np.pi * (self.intermediate_freq + params[:, 2]) * last_time
        if (np.any(np.abs(params[:, 0]) > 2**53 - 1)
                or not np.all(np.isfinite(chip_steps))
                or not np.all(np.isfinite(code_end)) or np.any(np.abs(code_end) > 2**53 - 1)
                or not np.all(np.isfinite(phase_end))):
            raise ValueError("channel phases exceed the supported numeric range")
        return codes, params, np.asarray(bits, dtype=np.int8), np.asarray(offsets, dtype=np.int64)

    def _render(self, channels, n, start, noise_db, seed):
        codes, params, bits, offsets = self._prepare_channels(channels, n, start)
        out = _kernel()(codes, params, bits, offsets, self.sampling_freq,
                        self.intermediate_freq, start, n)
        return _add_noise(out, noise_db, seed).ravel()

    def generate_epochs(self, channels_per_epoch, n_samples=None, noise_seeds=None, *, start_samples=None):
        """Generate a CPU batch, optionally at supplied absolute sample offsets.

        Epochs may have different channel counts, PRNs and navigation bits.
        Channel order and per-epoch noise-seed semantics match generate_epoch.
        Omitted start_samples preserves independent sample-zero epochs. For
        contiguous blocks supply consecutive offsets and channels whose phases
        and navigation sequences share the same sample-zero reference.
        """
        rows = list(channels_per_epoch)
        n = max(1, int(self.sampling_freq * 1e-3)) if n_samples is None else positive_int("n_samples", n_samples)
        seeds = [self.noise_seed] * len(rows) if noise_seeds is None else list(noise_seeds)
        if len(seeds) != len(rows):
            raise ValueError("noise_seeds must have one value per epoch")
        seeds = [_noise_seed(seed) for seed in seeds]
        starts = [0]*len(rows) if start_samples is None else list(start_samples)
        if len(starts) != len(rows):
            raise ValueError("start_samples must have one value per epoch")
        starts = [_sample_index("start_sample", start) for start in starts]
        if not rows:
            return np.empty((0, 2 * n), dtype=np.float32)
        if len(rows) == 1:
            return self._render(rows[0], n, starts[0], self.noise_floor_db, seeds[0])[None, :]
        from numba import get_num_threads
        # Packing a shared batch pays off for short epochs with parallel
        # workers. Keep the existing path for single-threaded execution and
        # long epochs, where measurements showed no benefit from batching.
        if get_num_threads() == 1 or n > 8192:
            return np.stack([self._render(ch, n, start, self.noise_floor_db, seed)
                             for ch, seed, start in zip(rows, seeds, starts)])
        # Share each PRN's spreading code instead of retaining a full code
        # matrix per epoch. Only parameters/navigation bits scale with channels.
        code_table = np.empty((32, 1023), dtype=np.float64)
        code_bank = (code_table, set())
        code_ids_rows, params_rows, bits_rows = [], [], []
        epoch_offsets, nav_offsets, bit_count = [0], [0], 0
        for row, start in zip(rows, starts):
            code_ids, params, bits, offsets = self._prepare_channels(row, n, start, code_bank=code_bank)
            code_ids_rows.append(code_ids)
            params_rows.append(params)
            bits_rows.append(bits)
            epoch_offsets.append(epoch_offsets[-1]+len(row))
            nav_offsets.extend((offsets[1:]+bit_count).tolist())
            bit_count += len(bits)
        out = _batch_kernel()(code_table, np.concatenate(code_ids_rows),
                              np.concatenate(params_rows), np.concatenate(bits_rows),
                              np.asarray(nav_offsets, dtype=np.int64), np.asarray(epoch_offsets, dtype=np.int64),
                              np.asarray(starts, dtype=np.int64), self.sampling_freq, self.intermediate_freq, n)
        for row, seed in zip(out, seeds):
            _add_noise(row.reshape(n, 2), self.noise_floor_db, seed)
        return out

    def generate_test_signal(self, prn, code_phase=0, doppler=0,
                             cn0_dbhz=45, duration_s=1e-3):
        """Unit signal with C/N0 calibrated for complex samples (Hz units).

        E[|noise|**2] = fs / (C/N0), hence variance per I/Q = fs/(2*C/N0).
        """
        cn0 = finite_float("cn0_dbhz", cn0_dbhz)
        duration = positive_float("duration_s", duration_s)
        n = max(1, int(self.sampling_freq * duration))
        channel = dict(prn=prn, code_phase=code_phase, carrier_phase=0.0,
                       doppler_hz=doppler, amplitude=1.0, nav_bit=1)
        noise_db = 10 * math.log10(self.sampling_freq / 2) - cn0
        return self._generate_signal([channel], n, noise_db)
