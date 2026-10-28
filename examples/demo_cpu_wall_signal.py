"""Moving GPS receiver beside a finite wall: CPU IQ, acquisition and bias.

Run: PYTHONPATH=python python examples/demo_cpu_wall_signal.py
Outputs are written under output/cpu-wall-signal, outside tracked source files.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.validation.tracking_residuals import generate_ca_code
from gnss_gpu.wall_signal_sim import C_LIGHT, VerticalWall, simulate_wall_signal


def acquire_delay_m(iq, local_fft, sampling_freq):
    """Coarse FFT code delay, with a parabolic three-bin peak refinement.

    This is a single-PRN, near-zero residual-Doppler diagnostic, not tracking.
    The returned lag is modulo one C/A period, wrapped around zero.
    """
    data = iq.reshape(-1, 2)
    corr = np.abs(np.fft.ifft(np.fft.fft(data[:, 0]+1j*data[:, 1])*np.conj(local_fft)))
    peak = int(np.argmax(corr))
    if corr[peak] == 0:
        return float("nan")
    before, center, after = corr[(peak-1) % len(corr)], corr[peak], corr[(peak+1) % len(corr)]
    denominator = before-2*center+after
    fraction = 0. if denominator == 0 else .5*(before-after)/denominator
    lag = (peak+fraction+len(corr)/2) % len(corr)-len(corr)/2
    return lag*C_LIGHT/sampling_freq


def main(output_dir="output/cpu-wall-signal"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fs, count = 20*CHIP_RATE, 20460  # 1 ms; 14.65 m per coarse code-delay bin
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    wall = VerticalWall(0., -30., 5., 0., 100., material="concrete", polarization="rhcp")
    rx0, velocity = np.array([-20., -20., 1.5]), np.array([1., 5., 0.])
    satellite = np.array([-1e7, 1e7, 1e7])
    reference = float(np.linalg.norm(satellite-rx0))
    local_fft = np.fft.fft(np.repeat(generate_ca_code(1), 20))
    rows = []
    for time in np.linspace(0., 2., 101):
        kwargs = dict(start_sample=round(time*fs), reference_range_m=reference, integrate_samples=True)
        full = simulate_wall_signal(sim, wall, rx0, velocity, satellite, count, **kwargs)
        direct = simulate_wall_signal(sim, wall, rx0, velocity, satellite, count,
                                      include_reflection=False, **kwargs)
        reflected = simulate_wall_signal(sim, wall, rx0, velocity, satellite, count,
                                         include_direct=False, **kwargs)
        coarse_direct = acquire_delay_m(direct.iq, local_fft, fs)
        coarse_full = acquire_delay_m(full.iq, local_fft, fs)
        coarse_reflected = acquire_delay_m(reflected.iq, local_fft, fs)
        direct_fit = estimate_ca_delay(direct.iq, fs)
        full_fit = estimate_ca_delay(full.iq, fs)
        reflection_fit = estimate_ca_delay(reflected.iq, fs)
        assert direct_fit is not None and full_fit is not None
        direct_delay, full_delay = direct_fit.delay_m, full_fit.delay_m
        reflection_delay = float("nan") if reflection_fit is None else reflection_fit.delay_m
        geometric_direct = float(full.direct_range_m[count//2]-reference)
        geometric_reflected = float(full.reflected_range_m[count//2]-reference)
        rows.append(dict(time_s=float(time), receiver_x_m=float(rx0[0]+velocity[0]*time),
                         receiver_y_m=float(rx0[1]+velocity[1]*time),
                         direct_visible=bool(full.direct_visible[0]),
                         reflection_visible=bool(full.reflection_visible[0]),
                         geometric_excess_m=float(full.reflected_range_m[0]-full.direct_range_m[0]),
                         reflected_field_abs=float(abs(full.reflection_coefficient[0])),
                         reflected_field_phase_rad=(float(np.angle(full.reflection_coefficient[0]))
                                                    if full.reflection_visible[0] else float("nan")),
                         geometric_direct_delay_m=geometric_direct,
                         geometric_reflection_delay_m=geometric_reflected,
                         coarse_direct_delay_m=coarse_direct,
                         coarse_reflection_delay_m=coarse_reflected,
                         coarse_composite_delay_m=coarse_full,
                         acquired_direct_delay_m=direct_delay,
                         acquired_reflection_delay_m=reflection_delay,
                         acquired_composite_delay_m=full_delay,
                         incremental_multipath_bias_m=full_delay-direct_delay,
                         coarse_multipath_bias_m=coarse_full-coarse_direct,
                         direct_fit_error_m=direct_delay-geometric_direct,
                         direct_fit_residual_fraction=direct_fit.fit_residual_fraction,
                         composite_fit_residual_fraction=full_fit.fit_residual_fraction))
    with (output/"route.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    times = [row["time_s"] for row in rows]
    figure, axes = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True)
    axes[0].plot([row["receiver_y_m"] for row in rows], [row["receiver_x_m"] for row in rows], "o-", label="receiver, velocity (1,5,0) m/s")
    axes[0].plot([wall.y_min_m, wall.y_max_m], [wall.x_m]*2, "k-", linewidth=4, label="finite wall")
    axes[0].set(xlabel="local y [m]", ylabel="local x [m]", title="Moving receiver, stationary GPS transmitter")
    axes[0].legend()
    excess = np.array([row["geometric_excess_m"] if row["reflection_visible"] else np.nan for row in rows])
    axes[1].plot(times, excess, label="geometric reflected excess")
    axes[1].plot(times, [row["coarse_reflection_delay_m"]-row["coarse_direct_delay_m"] for row in rows], ".", label="three-bin parabola")
    axes[1].plot(times, [row["acquired_reflection_delay_m"]-row["acquired_direct_delay_m"] for row in rows], "--", label="aperture-template fit")
    axes[1].set(xlabel="time [s]", ylabel="excess delay [m]")
    axes[1].legend()
    axes[2].plot(times, [row["coarse_direct_delay_m"]-row["geometric_direct_delay_m"] for row in rows], label="three-bin parabola")
    axes[2].plot(times, [row["direct_fit_error_m"] for row in rows], label="aperture-template fit")
    axes[2].axhline(0, color="gray", linewidth=.5)
    axes[2].set(xlabel="time [s]", ylabel="direct-only error [m]", title="Estimator calibration against geometry at block midpoint")
    axes[2].legend()
    axes[3].plot(times, [row["coarse_multipath_bias_m"] for row in rows], label="three-bin parabola")
    axes[3].plot(times, [row["incremental_multipath_bias_m"] for row in rows], label="aperture-template fit")
    axes[3].axhline(0, color="gray", linewidth=.5)
    axes[3].set(xlabel="time [s]", ylabel="code-delay bias [m]", title="Composite minus direct-only; same integrated IQ and respective estimator")
    axes[3].legend()
    figure.savefig(output/"route.png", dpi=150)
    plt.close(figure)
    visible = sum(row["reflection_visible"] for row in rows)
    max_bias = max(abs(row["incremental_multipath_bias_m"]) for row in rows)
    max_error = max(abs(row["direct_fit_error_m"]) for row in rows)
    print(f"CPU GPS IQ: {len(rows)} poses, {visible} with a specular reflection")
    print(f"Maximum incremental acquired code-delay bias: {max_bias:.3f} m")
    print(f"Maximum direct-only aperture-fit error: {max_error:.6f} m")
    print(f"Acquisition grid: {C_LIGHT/fs:.2f} m per bin; the aperture fit refines its strongest peak.")
    print(f"CSV: {output/'route.csv'}\nFigure: {output/'route.png'}")
    print("Ideal wall, centered sample integration, constant nav bit, noiseless, no diffraction.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/cpu-wall-signal")
    main(parser.parse_args().output_dir)
