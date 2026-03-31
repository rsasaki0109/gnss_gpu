"""Tests for gnss_gpu.viz visualization functions."""

import sys
import unittest
from unittest.mock import patch

import numpy as np

# Force non-interactive backend before any matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnss_gpu.viz.plots import (
    plot_particles,
    plot_skyplot,
    plot_vulnerability_map,
    plot_trajectory,
    plot_dop_timeline,
    plot_positioning_error,
    plot_spectrogram,
    plot_acquisition_grid,
    plot_multipath_scenario,
)


class TestPlotParticles(unittest.TestCase):
    def test_basic(self):
        particles = np.random.randn(500, 4)
        fig, ax = plot_particles(particles)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)

    def test_with_true_and_estimate(self):
        particles = np.random.randn(200, 3)
        fig, ax = plot_particles(particles, true_pos=[0, 0, 0],
                                 estimate=[0.5, 0.5, 0.5])
        plt.close(fig)

    def test_downsample(self):
        particles = np.random.randn(20000, 2)
        fig, ax = plot_particles(particles, max_particles=1000)
        plt.close(fig)

    def test_colorby_height(self):
        particles = np.random.randn(100, 4)
        fig, ax = plot_particles(particles, colorby="height")
        plt.close(fig)

    def test_colorby_weight(self):
        particles = np.random.randn(100, 3)
        weights = np.random.rand(100)
        fig, ax = plot_particles(particles, colorby="weight", weights=weights)
        plt.close(fig)

    def test_colorby_clockbias(self):
        particles = np.random.randn(100, 4)
        fig, ax = plot_particles(particles, colorby="clockbias")
        plt.close(fig)


class TestPlotSkyplot(unittest.TestCase):
    def test_basic(self):
        az = np.array([0, 90, 180, 270])
        el = np.array([30, 45, 60, 15])
        fig, ax = plot_skyplot(az, el)
        plt.close(fig)

    def test_with_prn(self):
        az = np.array([45, 135, 225, 315])
        el = np.array([50, 30, 70, 20])
        fig, ax = plot_skyplot(az, el, prn_list=[1, 5, 12, 31])
        plt.close(fig)

    def test_with_los(self):
        az = np.array([0, 90, 180, 270])
        el = np.array([30, 45, 60, 15])
        fig, ax = plot_skyplot(az, el, is_los=[True, False, True, False])
        plt.close(fig)

    def test_with_cn0(self):
        az = np.array([0, 90, 180, 270])
        el = np.array([30, 45, 60, 15])
        fig, ax = plot_skyplot(az, el, cn0=[40, 35, 45, 25])
        plt.close(fig)


class TestPlotVulnerabilityMap(unittest.TestCase):
    def test_basic(self):
        grid_e = np.linspace(-100, 100, 20)
        grid_n = np.linspace(-100, 100, 20)
        metric = np.random.rand(20, 20) * 10
        fig, ax = plot_vulnerability_map(grid_e, grid_n, metric)
        plt.close(fig)

    def test_with_buildings_and_trajectory(self):
        grid_e = np.linspace(-100, 100, 15)
        grid_n = np.linspace(-100, 100, 15)
        metric = np.random.rand(15, 15) * 8
        buildings = [(0, 0, 30, 20), (50, 50, 15, 25)]
        traj = np.column_stack([np.linspace(-80, 80, 30),
                                np.linspace(-50, 50, 30)])
        fig, ax = plot_vulnerability_map(grid_e, grid_n, metric,
                                         buildings=buildings,
                                         trajectory=traj)
        plt.close(fig)


class TestPlotTrajectory(unittest.TestCase):
    def test_single_trajectory(self):
        pos = np.column_stack([np.linspace(0, 100, 50),
                               np.sin(np.linspace(0, 4, 50)) * 20])
        fig, ax = plot_trajectory(pos)
        plt.close(fig)

    def test_multiple_trajectories(self):
        pos1 = np.column_stack([np.linspace(0, 100, 50),
                                np.sin(np.linspace(0, 4, 50)) * 20])
        pos2 = pos1 + np.random.randn(50, 2) * 2
        gt = np.column_stack([np.linspace(0, 100, 50),
                              np.sin(np.linspace(0, 4, 50)) * 20])
        fig, ax = plot_trajectory({"WLS": pos1, "PF": pos2},
                                  true_trajectory=gt)
        plt.close(fig)


class TestPlotDopTimeline(unittest.TestCase):
    def test_basic(self):
        t = np.arange(100)
        hdop = 1.5 + 0.5 * np.sin(t * 0.1)
        pdop = 2.0 + np.sin(t * 0.1)
        fig, ax = plot_dop_timeline(t, hdop=hdop, pdop=pdop)
        plt.close(fig)

    def test_with_satellite_count(self):
        t = np.arange(50)
        vdop = 2 + np.random.rand(50)
        n_vis = np.random.randint(4, 12, 50)
        fig, ax = plot_dop_timeline(t, vdop=vdop, n_visible=n_vis)
        plt.close(fig)


class TestPlotPositioningError(unittest.TestCase):
    def test_single_error(self):
        t = np.arange(100)
        err = np.abs(np.random.randn(100)) * 3
        fig, ax = plot_positioning_error(t, err)
        plt.close(fig)

    def test_multiple_errors(self):
        t = np.arange(80)
        errors = {
            "WLS": np.abs(np.random.randn(80)) * 5,
            "PF": np.abs(np.random.randn(80)) * 2,
        }
        fig, ax = plot_positioning_error(t, errors)
        plt.close(fig)


class TestPlotSpectrogram(unittest.TestCase):
    def test_basic(self):
        spec = np.random.randn(64, 128)
        fig, ax = plot_spectrogram(spec, sampling_freq=20e6,
                                   fft_size=256, hop_size=64)
        plt.close(fig)

    def test_with_detections(self):
        spec = np.random.randn(64, 128)
        dets = [{"frame_start": 10, "frame_end": 30,
                 "bin_start": 20, "bin_end": 60}]
        fig, ax = plot_spectrogram(spec, sampling_freq=20e6,
                                   fft_size=256, hop_size=64,
                                   detections=dets)
        plt.close(fig)


class TestPlotAcquisitionGrid(unittest.TestCase):
    def test_basic(self):
        corr = np.random.rand(41, 2046)
        corr[20, 500] = 10.0  # peak
        fig, ax = plot_acquisition_grid(corr, sampling_freq=4.092e6,
                                        doppler_range=5000,
                                        doppler_step=250)
        plt.close(fig)


try:
    from mpl_toolkits.mplot3d import Axes3D
    HAS_3D = True
except (ImportError, ModuleNotFoundError):
    HAS_3D = False


@unittest.skipUnless(HAS_3D, "mpl_toolkits.mplot3d not available")
class TestPlotMultipathScenario(unittest.TestCase):
    def test_basic(self):
        buildings = [(50, 0, 30, 20, 40), (-30, 30, 20, 25, 30)]
        receiver = np.array([0, 0, 1.5])
        sats = np.array([[100, 100, 200], [-150, 50, 300],
                         [50, -100, 250]])
        fig, ax = plot_multipath_scenario(buildings, receiver, sats)
        plt.close(fig)

    def test_with_los_and_reflections(self):
        buildings = [(50, 0, 30, 20, 40)]
        receiver = np.array([0, 0, 1.5])
        sats = np.array([[100, 100, 200], [-150, 50, 300]])
        los = [True, False]
        refl = np.array([[50, 20, 20], [np.nan, np.nan, np.nan]])
        fig, ax = plot_multipath_scenario(buildings, receiver, sats,
                                          los_mask=los,
                                          reflections=refl)
        plt.close(fig)


class TestMatplotlibMissing(unittest.TestCase):
    """Test graceful error when matplotlib is not available."""

    def test_import_error_message(self):
        import gnss_gpu.viz.plots as plots_mod
        original = plots_mod.HAS_MPL
        try:
            plots_mod.HAS_MPL = False
            with self.assertRaises(ImportError) as ctx:
                plot_particles(np.random.randn(10, 3))
            self.assertIn("matplotlib", str(ctx.exception))
        finally:
            plots_mod.HAS_MPL = original


if __name__ == "__main__":
    unittest.main()
