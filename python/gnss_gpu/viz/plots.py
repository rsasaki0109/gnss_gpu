"""Matplotlib-based visualization tools for gnss_gpu."""

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import FancyArrowPatch, Rectangle
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import mpl_toolkits.mplot3d  # noqa: F401 - register 3D projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_MPL3D = True
except ImportError:
    HAS_MPL3D = False


def _require_matplotlib():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install: pip install matplotlib"
        )


def _ensure_ax(ax, polar=False, projection=None):
    """Return (fig, ax), creating them if ax is None."""
    if ax is not None:
        return ax.figure, ax
    if polar:
        projection = "polar"
    fig, ax = plt.subplots(subplot_kw={"projection": projection} if projection else {})
    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_particles
# --------------------------------------------------------------------------- #

def plot_particles(particles, true_pos=None, estimate=None,
                   ax=None, title="Particle Distribution",
                   max_particles=10000, colorby="weight", weights=None):
    """Plot particle cloud in 2D (columns 0 and 1 treated as East/X and North/Y).

    Args:
        particles: [N, 4] array (x, y, z, cb) or [N, 3] or [N, 2].
        true_pos: [2] or [3] true position (optional, shown as red star).
        estimate: [2] or [3] estimated position (optional, shown as blue diamond).
        ax: matplotlib Axes (created if None).
        title: plot title.
        max_particles: downsample if more particles than this.
        colorby: 'weight', 'height', or 'clockbias'.
        weights: [N] particle weights (used when colorby='weight').

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()
    particles = np.asarray(particles)

    n = particles.shape[0]
    if n > max_particles:
        idx = np.random.choice(n, max_particles, replace=False)
        particles = particles[idx]
        if weights is not None:
            weights = np.asarray(weights)[idx]

    fig, ax = _ensure_ax(ax)

    # Determine color array
    if colorby == "height" and particles.shape[1] >= 3:
        c = particles[:, 2]
        clabel = "Height [m]"
    elif colorby == "clockbias" and particles.shape[1] >= 4:
        c = particles[:, 3]
        clabel = "Clock bias [m]"
    elif colorby == "weight" and weights is not None:
        c = np.asarray(weights)
        clabel = "Weight"
    else:
        c = None
        clabel = None

    scatter_kwargs = {"s": 1, "alpha": 0.3, "rasterized": True}
    if c is not None:
        scatter_kwargs["c"] = c
        scatter_kwargs["cmap"] = "viridis"
    sc = ax.scatter(particles[:, 0], particles[:, 1], **scatter_kwargs)
    if c is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(clabel)

    if true_pos is not None:
        true_pos = np.asarray(true_pos)
        ax.plot(true_pos[0], true_pos[1], "r*", markersize=14, label="True",
                zorder=10)

    if estimate is not None:
        estimate = np.asarray(estimate)
        ax.plot(estimate[0], estimate[1], "bD", markersize=8, label="Estimate",
                zorder=10)

    ax.set_xlabel("East / X [m]")
    ax.set_ylabel("North / Y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    if true_pos is not None or estimate is not None:
        ax.legend()

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_skyplot
# --------------------------------------------------------------------------- #

def plot_skyplot(az_deg, el_deg, prn_list=None, is_los=None,
                 cn0=None, ax=None, title="Sky Plot"):
    """Polar plot of satellite positions.

    Args:
        az_deg: [n_sat] azimuth in degrees.
        el_deg: [n_sat] elevation in degrees.
        prn_list: [n_sat] PRN numbers for labels.
        is_los: [n_sat] bool, True=LOS (green), False=NLOS (red).
        cn0: [n_sat] C/N0 values for color/size.
        ax: polar Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()
    az_deg = np.asarray(az_deg, dtype=float)
    el_deg = np.asarray(el_deg, dtype=float)

    theta = np.radians(az_deg)
    r = 90.0 - el_deg  # zenith at center

    fig, ax = _ensure_ax(ax, polar=True)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_yticklabels(["90", "75", "60", "45", "30", "15", "0"])

    # Determine colours
    if cn0 is not None:
        cn0 = np.asarray(cn0, dtype=float)
        cn0_range = cn0.max() - cn0.min()
        sizes = 20 + 80 * (cn0 - cn0.min()) / max(cn0_range, 1.0)
        sc = ax.scatter(theta, r, c=cn0, s=sizes, cmap="RdYlGn",
                        edgecolors="k", linewidths=0.5, zorder=5)
        fig.colorbar(sc, ax=ax, pad=0.1, label="C/N0 [dB-Hz]")
    elif is_los is not None:
        is_los = np.asarray(is_los, dtype=bool)
        colors = np.where(is_los, "green", "red")
        ax.scatter(theta, r, c=colors, s=60, edgecolors="k",
                   linewidths=0.5, zorder=5)
    else:
        ax.scatter(theta, r, s=60, c="steelblue", edgecolors="k",
                   linewidths=0.5, zorder=5)

    # PRN labels
    if prn_list is not None:
        for i, prn in enumerate(prn_list):
            ax.annotate(str(prn), (theta[i], r[i]), fontsize=7,
                        ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 5))

    ax.set_title(title, va="bottom", pad=20)
    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_vulnerability_map
# --------------------------------------------------------------------------- #

def plot_vulnerability_map(grid_e, grid_n, metric_2d,
                           metric_name="HDOP", buildings=None,
                           trajectory=None, ax=None,
                           title="GNSS Vulnerability Map"):
    """Plot 2D heatmap of GNSS quality metric.

    Args:
        grid_e: [n_e] east coordinates of grid.
        grid_n: [n_n] north coordinates of grid.
        metric_2d: [n_n, n_e] 2D array of metric values.
        metric_name: label for colorbar.
        buildings: list of (center_e, center_n, width, depth) for footprints.
        trajectory: [M, 2] receiver trajectory in ENU.
        ax: matplotlib Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()

    grid_e = np.asarray(grid_e)
    grid_n = np.asarray(grid_n)
    metric_2d = np.asarray(metric_2d)

    fig, ax = _ensure_ax(ax)

    ee, nn = np.meshgrid(grid_e, grid_n)
    pcm = ax.pcolormesh(ee, nn, metric_2d, cmap="RdYlGn_r", shading="auto")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(metric_name)

    # Building footprints
    if buildings is not None:
        for bld in buildings:
            ce, cn, w, d = bld[:4]
            rect = Rectangle((ce - w / 2.0, cn - d / 2.0), w, d,
                              facecolor="gray", edgecolor="black",
                              alpha=0.6, linewidth=0.8, zorder=3)
            ax.add_patch(rect)

    # Trajectory overlay
    if trajectory is not None:
        trajectory = np.asarray(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], "w-", linewidth=1.5,
                zorder=4, label="Trajectory")
        ax.plot(trajectory[0, 0], trajectory[0, 1], "wo", markersize=6,
                zorder=5)
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ws", markersize=6,
                zorder=5)
        ax.legend()

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    ax.set_aspect("equal")

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_trajectory
# --------------------------------------------------------------------------- #

def plot_trajectory(positions_enu, true_trajectory=None,
                    labels=None, ax=None, title="Positioning Trajectory"):
    """Plot positioning results vs ground truth.

    Args:
        positions_enu: dict of {name: [N, 2/3] array} or single [N, 2/3].
        true_trajectory: [N, 2/3] ground truth.
        labels: unused (kept for API compat; dict keys serve as labels).
        ax: matplotlib Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()

    fig, ax = _ensure_ax(ax)

    if not isinstance(positions_enu, dict):
        positions_enu = {"Estimated": np.asarray(positions_enu)}

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, pos) in enumerate(positions_enu.items()):
        pos = np.asarray(pos)
        c = colors[i % len(colors)]
        ax.plot(pos[:, 0], pos[:, 1], "-", color=c, linewidth=1.2, label=name)
        ax.plot(pos[0, 0], pos[0, 1], "o", color=c, markersize=7)
        ax.plot(pos[-1, 0], pos[-1, 1], "s", color=c, markersize=7)

    if true_trajectory is not None:
        true_trajectory = np.asarray(true_trajectory)
        ax.plot(true_trajectory[:, 0], true_trajectory[:, 1], "k--",
                linewidth=1.5, label="Ground Truth")
        ax.plot(true_trajectory[0, 0], true_trajectory[0, 1], "ko",
                markersize=7)
        ax.plot(true_trajectory[-1, 0], true_trajectory[-1, 1], "ks",
                markersize=7)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_dop_timeline
# --------------------------------------------------------------------------- #

def plot_dop_timeline(times, pdop=None, hdop=None, vdop=None, gdop=None,
                      n_visible=None, ax=None, title="DOP Timeline"):
    """Plot DOP values over time, optionally with satellite count.

    Args:
        times: [N] time values (seconds or datetime).
        pdop, hdop, vdop, gdop: [N] DOP time series.
        n_visible: [N] satellite count (plotted on secondary y-axis).
        ax: matplotlib Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()

    fig, ax = _ensure_ax(ax)
    times = np.asarray(times)

    any_plotted = False
    for arr, label, style in [
        (pdop, "PDOP", "-"),
        (hdop, "HDOP", "--"),
        (vdop, "VDOP", "-."),
        (gdop, "GDOP", ":"),
    ]:
        if arr is not None:
            ax.plot(times, np.asarray(arr), style, label=label)
            any_plotted = True

    ax.set_xlabel("Time")
    ax.set_ylabel("DOP")
    ax.set_title(title)
    if any_plotted:
        ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if n_visible is not None:
        ax2 = ax.twinx()
        ax2.bar(times, np.asarray(n_visible), alpha=0.2, color="gray",
                width=(times[1] - times[0]) * 0.8 if len(times) > 1 else 1.0,
                label="Visible SVs")
        ax2.set_ylabel("Visible satellites")
        ax2.legend(loc="upper right")

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_positioning_error
# --------------------------------------------------------------------------- #

def plot_positioning_error(times, errors, labels=None,
                           ax=None, title="Positioning Error"):
    """Plot positioning error over time for multiple methods.

    Args:
        times: [N] time values.
        errors: dict of {name: [N] error array} or single [N].
        labels: unused (dict keys serve as labels).
        ax: matplotlib Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()

    fig, ax = _ensure_ax(ax)
    times = np.asarray(times)

    if not isinstance(errors, dict):
        errors = {"Error": np.asarray(errors)}

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, err) in enumerate(errors.items()):
        err = np.asarray(err)
        c = colors[i % len(colors)]
        ax.plot(times, err, "-", color=c, linewidth=1.0, label=name)

        mean_val = np.mean(err)
        std_val = np.std(err)
        p95 = np.percentile(err, 95)
        stat_text = f"{name}: mean={mean_val:.2f}, std={std_val:.2f}, 95%={p95:.2f}"
        ax.axhline(mean_val, color=c, linestyle=":", alpha=0.5)
        ax.text(0.02, 0.98 - i * 0.06, stat_text, transform=ax.transAxes,
                fontsize=7, va="top", color=c)

    ax.set_xlabel("Time")
    ax.set_ylabel("Error [m]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_spectrogram
# --------------------------------------------------------------------------- #

def plot_spectrogram(spectrogram, sampling_freq, fft_size, hop_size,
                     detections=None, ax=None, title="Spectrogram"):
    """Plot time-frequency spectrogram with detected interference.

    Args:
        spectrogram: [n_frames, n_bins] power in dB.
        sampling_freq: sampling frequency in Hz.
        fft_size: FFT size used.
        hop_size: hop size used.
        detections: list of dicts with keys 'frame_start', 'frame_end',
                    'bin_start', 'bin_end' (overlay boxes).
        ax: matplotlib Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()

    spectrogram = np.asarray(spectrogram)
    n_frames, n_bins = spectrogram.shape

    fig, ax = _ensure_ax(ax)

    # Time and frequency axes
    time_axis = np.arange(n_frames + 1) * hop_size / sampling_freq
    freq_axis = np.arange(n_bins + 1) * sampling_freq / fft_size

    pcm = ax.pcolormesh(time_axis, freq_axis / 1e6, spectrogram.T,
                        cmap="inferno", shading="flat")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Power [dB]")

    # Overlay detection boxes
    if detections is not None:
        for det in detections:
            fs = det.get("frame_start", 0)
            fe = det.get("frame_end", n_frames)
            bs = det.get("bin_start", 0)
            be = det.get("bin_end", n_bins)
            t0 = fs * hop_size / sampling_freq
            t1 = fe * hop_size / sampling_freq
            f0 = bs * sampling_freq / fft_size / 1e6
            f1 = be * sampling_freq / fft_size / 1e6
            rect = Rectangle((t0, f0), t1 - t0, f1 - f0,
                              linewidth=2, edgecolor="red",
                              facecolor="none", zorder=5)
            ax.add_patch(rect)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [MHz]")
    ax.set_title(title)

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_acquisition_grid
# --------------------------------------------------------------------------- #

def plot_acquisition_grid(correlation_map, sampling_freq,
                          doppler_range, doppler_step,
                          ax=None, title="Acquisition Search Grid"):
    """Plot 2D correlation map (code phase vs Doppler).

    Args:
        correlation_map: [n_doppler, n_code_phase] correlation values.
        sampling_freq: sampling frequency in Hz.
        doppler_range: max Doppler shift in Hz (symmetric).
        doppler_step: Doppler step in Hz.
        ax: matplotlib Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()

    correlation_map = np.asarray(correlation_map)
    n_doppler, n_code_phase = correlation_map.shape

    fig, ax = _ensure_ax(ax)

    doppler_axis = np.arange(-doppler_range, doppler_range + doppler_step * 0.5,
                             doppler_step)
    if len(doppler_axis) > n_doppler:
        doppler_axis = doppler_axis[:n_doppler]
    elif len(doppler_axis) < n_doppler:
        doppler_axis = np.linspace(-doppler_range, doppler_range, n_doppler)

    code_chips = np.arange(n_code_phase) * 1023.0 / n_code_phase

    pcm = ax.pcolormesh(code_chips, doppler_axis, correlation_map,
                        cmap="hot", shading="auto")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label("Correlation")

    # Mark the peak
    peak_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
    ax.plot(code_chips[peak_idx[1]], doppler_axis[peak_idx[0]], "c+",
            markersize=14, markeredgewidth=2, zorder=5)

    ax.set_xlabel("Code phase [chips]")
    ax.set_ylabel("Doppler [Hz]")
    ax.set_title(title)

    return fig, ax


# --------------------------------------------------------------------------- #
#  plot_multipath_scenario
# --------------------------------------------------------------------------- #

def plot_multipath_scenario(buildings_enu, receiver, satellites_enu,
                            los_mask=None, reflections=None,
                            ax=None, title="Multipath Scenario"):
    """3D visualization of urban multipath scenario.

    Args:
        buildings_enu: list of (center_e, center_n, width, depth, height) tuples.
        receiver: [3] receiver ENU position.
        satellites_enu: [n_sat, 3] satellite ENU positions (or directions).
        los_mask: [n_sat] bool, True=LOS.
        reflections: [n_sat, 3] reflection points.
        ax: mplot3d Axes (created if None).
        title: plot title.

    Returns:
        (fig, ax) tuple.
    """
    _require_matplotlib()
    if not HAS_MPL3D:
        raise ImportError(
            "mpl_toolkits.mplot3d is required for 3D visualization. "
            "Install: pip install matplotlib"
        )

    receiver = np.asarray(receiver, dtype=float)
    satellites_enu = np.asarray(satellites_enu, dtype=float)
    n_sat = satellites_enu.shape[0]

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Draw buildings as 3D boxes
    for bld in buildings_enu:
        ce, cn, w, d, h = bld[:5]
        x0, x1 = ce - w / 2.0, ce + w / 2.0
        y0, y1 = cn - d / 2.0, cn + d / 2.0
        z0, z1 = 0.0, h

        # 6 faces
        verts = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],  # bottom
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
        ]
        poly = Poly3DCollection(verts, alpha=0.3, facecolor="gray",
                                edgecolor="black", linewidths=0.5)
        ax.add_collection3d(poly)

    # Draw receiver
    ax.scatter(*receiver, c="blue", s=100, marker="^", zorder=10,
               label="Receiver")

    # Draw satellite rays
    if los_mask is None:
        los_mask = np.ones(n_sat, dtype=bool)
    else:
        los_mask = np.asarray(los_mask, dtype=bool)

    for i in range(n_sat):
        color = "green" if los_mask[i] else "red"
        style = "-" if los_mask[i] else "--"
        ax.plot([receiver[0], satellites_enu[i, 0]],
                [receiver[1], satellites_enu[i, 1]],
                [receiver[2], satellites_enu[i, 2]],
                color=color, linestyle=style, linewidth=0.8, alpha=0.7)
        ax.scatter(*satellites_enu[i], c=color, s=40, marker="o")

    # Draw reflection paths
    if reflections is not None:
        reflections = np.asarray(reflections)
        for i in range(min(n_sat, len(reflections))):
            rp = reflections[i]
            if np.any(np.isnan(rp)):
                continue
            ax.plot([receiver[0], rp[0]], [receiver[1], rp[1]],
                    [receiver[2], rp[2]], "r:", linewidth=0.6, alpha=0.5)
            ax.plot([rp[0], satellites_enu[i, 0]],
                    [rp[1], satellites_enu[i, 1]],
                    [rp[2], satellites_enu[i, 2]],
                    "r:", linewidth=0.6, alpha=0.5)
            ax.scatter(*rp, c="orange", s=30, marker="x", zorder=8)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title(title)
    ax.legend()

    return fig, ax
