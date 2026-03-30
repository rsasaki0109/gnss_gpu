"""Optional plotly-based interactive visualizations for gnss_gpu."""

import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _require_plotly():
    if not HAS_PLOTLY:
        raise ImportError(
            "plotly is required for interactive visualization. "
            "Install: pip install plotly"
        )


def interactive_vulnerability_map(grid_e, grid_n, metrics, buildings=None):
    """Interactive plotly heatmap with hover info showing all metrics.

    Args:
        grid_e: [n_e] east coordinates of grid.
        grid_n: [n_n] north coordinates of grid.
        metrics: dict of {name: [n_n, n_e] 2D array}.
                 The first key is used as the displayed heatmap;
                 all values appear in hover text.
        buildings: list of (center_e, center_n, width, depth) for footprints.

    Returns:
        plotly Figure.
    """
    _require_plotly()

    grid_e = np.asarray(grid_e)
    grid_n = np.asarray(grid_n)

    metric_names = list(metrics.keys())
    primary = metric_names[0]
    z = np.asarray(metrics[primary])

    # Build custom hover text
    n_n, n_e = z.shape
    hover = np.empty((n_n, n_e), dtype=object)
    for r in range(n_n):
        for c in range(n_e):
            parts = [f"E: {grid_e[c]:.1f} m, N: {grid_n[r]:.1f} m"]
            for name in metric_names:
                val = float(metrics[name][r, c])
                parts.append(f"{name}: {val:.2f}")
            hover[r, c] = "<br>".join(parts)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=grid_e, y=grid_n, z=z,
        colorscale="RdYlGn_r",
        colorbar=dict(title=primary),
        text=hover,
        hoverinfo="text",
    ))

    # Building footprints as shapes
    if buildings is not None:
        for bld in buildings:
            ce, cn, w, d = bld[:4]
            fig.add_shape(
                type="rect",
                x0=ce - w / 2.0, y0=cn - d / 2.0,
                x1=ce + w / 2.0, y1=cn + d / 2.0,
                fillcolor="rgba(128,128,128,0.4)",
                line=dict(color="black", width=1),
            )

    fig.update_layout(
        title="GNSS Vulnerability Map (interactive)",
        xaxis_title="East [m]",
        yaxis_title="North [m]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def interactive_particles_3d(particles, true_pos=None):
    """3D interactive scatter of particle cloud.

    Args:
        particles: [N, 3+] array (x, y, z, ...).
        true_pos: [3] true position (optional).

    Returns:
        plotly Figure.
    """
    _require_plotly()

    particles = np.asarray(particles)
    n = particles.shape[0]

    # Downsample for browser performance
    max_pts = 50000
    if n > max_pts:
        idx = np.random.choice(n, max_pts, replace=False)
        particles = particles[idx]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=particles[:, 0], y=particles[:, 1], z=particles[:, 2],
        mode="markers",
        marker=dict(size=1, color=particles[:, 2], colorscale="Viridis",
                    opacity=0.3, colorbar=dict(title="Z [m]")),
        name="Particles",
    ))

    if true_pos is not None:
        true_pos = np.asarray(true_pos)
        fig.add_trace(go.Scatter3d(
            x=[true_pos[0]], y=[true_pos[1]], z=[true_pos[2]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="diamond"),
            name="True position",
        ))

    fig.update_layout(
        title="Particle Cloud (3D)",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="data",
        ),
    )

    return fig


def interactive_skyplot(az, el, prn_list, cn0=None, is_los=None):
    """Interactive polar plot of satellites.

    Args:
        az: [n_sat] azimuth in degrees.
        el: [n_sat] elevation in degrees.
        prn_list: [n_sat] PRN identifiers.
        cn0: [n_sat] C/N0 values.
        is_los: [n_sat] bool.

    Returns:
        plotly Figure.
    """
    _require_plotly()

    az = np.asarray(az, dtype=float)
    el = np.asarray(el, dtype=float)
    r = 90.0 - el

    # Determine colours
    if cn0 is not None:
        cn0 = np.asarray(cn0, dtype=float)
        marker = dict(
            size=10, color=cn0, colorscale="RdYlGn",
            colorbar=dict(title="C/N0 [dB-Hz]"),
            line=dict(color="black", width=0.5),
        )
    elif is_los is not None:
        is_los = np.asarray(is_los, dtype=bool)
        colors = ["green" if v else "red" for v in is_los]
        marker = dict(size=10, color=colors,
                      line=dict(color="black", width=0.5))
    else:
        marker = dict(size=10, color="steelblue",
                      line=dict(color="black", width=0.5))

    hover = [f"PRN {p}<br>Az: {a:.1f} deg<br>El: {e:.1f} deg"
             for p, a, e in zip(prn_list, az, el)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r, theta=az,
        mode="markers+text",
        marker=marker,
        text=[str(p) for p in prn_list],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=hover,
        hoverinfo="text",
    ))

    fig.update_layout(
        title="Sky Plot (interactive)",
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90),
            radialaxis=dict(range=[0, 90],
                            tickvals=[0, 15, 30, 45, 60, 75, 90],
                            ticktext=["90", "75", "60", "45", "30", "15", "0"]),
        ),
    )

    return fig
