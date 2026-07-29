"""Unit tests for gnss_gpu_ros.filters (no ROS required)."""

import math

import numpy as np
import pytest

from gnss_gpu_ros.filters import (
    CausalHampel,
    CvKalman1D,
    NavSatTrajectoryFilter,
    _radii_at,
)


class TestCausalHampel:
    def test_passthrough_during_warmup(self):
        h = CausalHampel(window=11, min_samples=5)
        for v in [0.0, 100.0, -50.0, 3.0]:
            out, flagged = h.update(v)
            assert out == v
            assert not flagged

    def test_spike_is_replaced_by_median(self):
        h = CausalHampel(window=11, k=2.5)
        for i in range(10):
            h.update(float(i) * 0.1)
        out, flagged = h.update(50.0)
        assert flagged
        assert abs(out - 0.45) < 0.2  # near the window median, not the spike

    def test_smooth_motion_untouched(self):
        h = CausalHampel(window=11, k=2.5)
        flags = []
        for i in range(100):
            _, flagged = h.update(float(i) * 1.0)  # 1 m/s steady drive
            flags.append(flagged)
        assert not any(flags)

    def test_departure_from_stop_recovers(self):
        # Long stop, then accelerate away at 1 m/s^2 (1 Hz fixes). The gate
        # must not latch onto the stop position forever.
        h = CausalHampel(window=21, k=2.5, max_consecutive=5)
        for _ in range(60):
            h.update(0.0)
        suppressed = 0
        last_out = last_pos = None
        for k in range(1, 31):
            pos = 0.5 * 1.0 * k * k  # x = a t^2 / 2
            out, flagged = h.update(pos)
            if flagged:
                suppressed += 1
            else:
                last_out, last_pos = out, pos
        assert suppressed <= 5  # bounded by max_consecutive, no latch-up
        assert last_out == last_pos  # raw values flow once the gate stands down

    def test_reacquisition_jump_recovers(self):
        # GNSS outage in a tunnel: fixes reappear 300 m ahead. The gate must
        # follow the new position cluster, not freeze at the old one.
        h = CausalHampel(window=21, k=2.5, max_consecutive=5)
        for _ in range(40):
            h.update(0.0)
        outputs = [h.update(300.0 + i)[0] for i in range(15)]
        assert outputs[-1] > 290.0  # following the new cluster

    def test_single_spike_still_gated_with_escape_logic(self):
        # The escape hatch must not weaken isolated-spike rejection.
        h = CausalHampel(window=21, k=2.5, max_consecutive=5)
        for i in range(40):
            h.update(float(i))
        out, flagged = h.update(500.0)
        assert flagged
        assert out < 60.0
        out2, flagged2 = h.update(41.0)  # next genuine fix passes again
        assert not flagged2
        assert out2 == 41.0

    def test_stationary_noise_untouched(self):
        rng = np.random.default_rng(7)
        h = CausalHampel(window=21, k=2.5)
        n_flagged = 0
        for v in rng.normal(0.0, 0.02, size=200):  # 2 cm noise at a stop
            _, flagged = h.update(float(v))
            n_flagged += int(flagged)
        assert n_flagged == 0  # MAD floor prevents false positives

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            CausalHampel(window=2)
        with pytest.raises(ValueError):
            CausalHampel(k=0.0)


class TestCvKalman1D:
    def test_first_sample_passthrough(self):
        kf = CvKalman1D()
        assert kf.update(0.0, 5.0) == 5.0

    def test_reduces_noise_on_constant_velocity(self):
        rng = np.random.default_rng(11)
        kf = CvKalman1D(sigma_a=1.0, sigma_z=1.0)
        raw_err, filt_err = [], []
        for k in range(200):
            t = float(k)
            truth = 5.0 * t
            z = truth + float(rng.normal(0.0, 1.0))
            x = kf.update(t, z)
            if k >= 20:  # after convergence
                raw_err.append(abs(z - truth))
                filt_err.append(abs(x - truth))
        assert np.mean(filt_err) < 0.8 * np.mean(raw_err)

    def test_reinitializes_on_non_monotonic_time(self):
        kf = CvKalman1D()
        kf.update(10.0, 1.0)
        kf.update(11.0, 2.0)
        assert kf.update(5.0, 100.0) == 100.0  # time went backwards: reset

    def test_explicit_reset_discards_velocity_history(self):
        kf = CvKalman1D()
        kf.update(10.0, 1.0)
        kf.update(11.0, 20.0)
        kf.reset()
        assert kf.update(12.0, -5.0) == -5.0

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            CvKalman1D(sigma_a=0.0)


class TestNavSatTrajectoryFilter:
    LAT0, LON0 = 35.6804, 139.7690  # central Tokyo

    def _drive(self, f, n_epochs=120, speed_mps=5.0, spikes=()):
        """Drive east; inject given (epoch, metres) latitude spikes."""
        _, r_p = _radii_at(self.LAT0)
        r_m, _ = _radii_at(self.LAT0)
        results = []
        for k in range(n_epochs):
            lat = self.LAT0
            lon = self.LON0 + math.degrees(speed_mps * k / r_p)
            for at, size_m in spikes:
                if k == at:
                    lat += math.degrees(size_m / r_m)
            results.append((k, lat, lon, f.update(float(k), lat, lon)))
        return results

    def test_spike_suppressed(self):
        f = NavSatTrajectoryFilter()
        r_m, _ = _radii_at(self.LAT0)
        results = self._drive(f, spikes=[(60, 50.0)])
        k, lat_in, _, (lat_out, _, _, north, outlier) = results[60]
        spike_in_m = math.radians(lat_in - self.LAT0) * r_m
        assert spike_in_m > 49.0  # the spike really was injected
        assert outlier
        assert abs(north) < 5.0  # output stays near the true track (north=0)

    def test_clean_track_preserved(self):
        f = NavSatTrajectoryFilter()
        results = self._drive(f, n_epochs=100)
        for k, _, _, (_, _, east, north, outlier) in results[20:]:
            assert not outlier
            assert abs(east - 5.0 * k) < 1.0
            assert abs(north) < 1.0

    def test_stages_can_be_disabled(self):
        f = NavSatTrajectoryFilter(use_hampel=False, use_kalman=False)
        results = self._drive(f, n_epochs=30, spikes=[(20, 50.0)])
        _, lat_in, lon_in, (lat_out, lon_out, _, _, outlier) = results[20]
        assert not outlier
        assert lat_out == pytest.approx(lat_in, abs=1e-12)
        assert lon_out == pytest.approx(lon_in, abs=1e-12)

    def test_roundtrip_latlon_accuracy(self):
        f = NavSatTrajectoryFilter(use_hampel=False, use_kalman=False)
        # passthrough config: lat/lon -> EN -> lat/lon must be lossless
        lat, lon = self.LAT0 + 0.01, self.LON0 + 0.01
        lat_out, lon_out, _, _, _ = f.update(0.0, lat, lon)
        assert lat_out == pytest.approx(lat, abs=1e-12)
        assert lon_out == pytest.approx(lon, abs=1e-12)

    def test_rejects_missing_or_invalid_fix_fields(self):
        f = NavSatTrajectoryFilter()
        with pytest.raises(ValueError, match="finite"):
            f.update(0.0, float("nan"), self.LON0)
        with pytest.raises(ValueError, match="latitude"):
            f.update(0.0, 91.0, self.LON0)
        with pytest.raises(ValueError, match="longitude"):
            f.update(0.0, self.LAT0, 181.0)

    def test_long_missing_interval_resets_motion_state_but_keeps_frame(self):
        f = NavSatTrajectoryFilter(max_gap_s=5.0)
        _, r_p = _radii_at(self.LAT0)
        f.update(0.0, self.LAT0, self.LON0)
        f.update(1.0, self.LAT0, self.LON0 + math.degrees(5.0 / r_p))
        result = f.update(
            20.0, self.LAT0, self.LON0 + math.degrees(100.0 / r_p)
        )
        assert result[2] == pytest.approx(100.0)

    def test_out_of_order_bag_timestamp_is_deterministic(self):
        f = NavSatTrajectoryFilter(use_hampel=False)
        samples = [
            (10.0, self.LAT0, self.LON0),
            (11.0, self.LAT0, self.LON0 + 0.00001),
            (9.0, self.LAT0, self.LON0 + 0.00002),
        ]
        outputs = [f.update(*sample) for sample in samples]
        # CvKalman1D deliberately reinitializes on a backward timestamp.
        assert outputs[-1][1] == pytest.approx(samples[-1][2], abs=1e-12)

    def test_clock_epoch_offset_does_not_change_replay(self):
        fixes = [
            (self.LAT0, self.LON0 + k * 0.00001)
            for k in range(20)
        ]

        def replay(epoch_offset):
            stream_filter = NavSatTrajectoryFilter()
            return np.asarray(
                [
                    stream_filter.update(epoch_offset + k, lat, lon)[:4]
                    for k, (lat, lon) in enumerate(fixes)
                ]
            )

        assert np.allclose(replay(0.0), replay(1_700_000_000.0), atol=1e-9)

    def test_restart_establishes_fresh_local_frame(self):
        first_process = NavSatTrajectoryFilter()
        first_process.update(0.0, self.LAT0, self.LON0)
        first_process.update(1.0, self.LAT0, self.LON0 + 0.001)

        restarted = NavSatTrajectoryFilter()
        _, _, east, north, _ = restarted.update(
            2.0, self.LAT0 + 0.001, self.LON0 + 0.001
        )
        assert east == pytest.approx(0.0)
        assert north == pytest.approx(0.0)

    def test_bag_replay_same_input_same_output(self):
        samples = [
            (float(k), self.LAT0, self.LON0 + k * 0.00001)
            for k in range(20)
            if k not in {7, 8, 9}  # model dropped messages in a rosbag
        ]

        def replay():
            stream_filter = NavSatTrajectoryFilter(max_gap_s=2.0)
            return np.asarray(
                [stream_filter.update(*sample)[:4] for sample in samples]
            )

        assert np.array_equal(replay(), replay())
