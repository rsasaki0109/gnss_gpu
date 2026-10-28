"""Paper-linked geometric Doppler validation from CPU-generated GPS IQ.

PYTHONPATH=python python examples/demo_cpu_signal_dynamics.py
No RF transmission; diagnostic frequency measurement uses BPSK squaring.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.signal_sim_cpu import CHIP_RATE
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal


def measure_bpsk_frequency(iq, sampling_freq):
    """Median adjacent squared phase increment, unambiguous within +/-fs/4.

    Removes BPSK polarity without input code or phase. Assumes one noiseless
    carrier path, no cycle ambiguity and approximately constant frequency over
    the short block. This is a diagnostic, not a GNSS acquisition/tracking loop.
    """
    data = iq.reshape(-1, 2).astype(np.float64)
    z = data[:, 0]+1j*data[:, 1]
    product = z[1:]*np.conj(z[:-1])
    return float(np.median(np.angle(product**2)*sampling_freq/(4*np.pi)))


def main(output_dir="output/cpu-signal-dynamics"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fs, count = 2*CHIP_RATE, 2046
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    wall = VerticalWall(0., -100., 100., 0., 100., antenna_lhcp_response=.1)
    receiver, transmitter = [-20., 0., 2.], [-20000020., 0., 2.]
    cases = [("receiver constant velocity", [0., 0., 0.]), ("receiver acceleration 200 m/s^2", [200., 0., 0.])]
    rows = []
    for label, acceleration in cases:
        for time in np.linspace(0., .2, 21):
            args = dict(start_sample=round(time*fs), reference_range_m=20000000.,
                        transmitter_velocity_mps=[1000., 0., 0.], rx_acceleration_mps2=acceleration,
                        transmitter_acceleration_mps2=[2., 0., 0.])
            direct = simulate_wall_signal(sim, wall, receiver, [5., 0., 0.], transmitter,
                                          count, include_reflection=False, **args)
            reflected = simulate_wall_signal(sim, wall, receiver, [5., 0., 0.], transmitter,
                                             count, include_direct=False, **args)
            for name, block, true_doppler in [
                ("direct", direct, direct.direct_doppler_hz),
                ("reflection", reflected, reflected.reflected_doppler_hz),
            ]:
                measured = measure_bpsk_frequency(block.iq, fs)
                expected = float(np.median(.5*(true_doppler[1:]+true_doppler[:-1])))
                rows.append(dict(scenario=label, path=name, time_s=float(time),
                                 theory_doppler_hz=expected, measured_doppler_hz=measured,
                                 error_hz=measured-expected))
    with (output/"doppler.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    root = Path(__file__).resolve().parents[1]
    sources = ["python/gnss_gpu/wall_signal_sim.py", "python/gnss_gpu/signal_sim_cpu.py",
               "python/gnss_gpu/fresnel.py", "examples/demo_cpu_signal_dynamics.py",
               "docs/gnss_signal_simulation_sources.json"]
    metadata = dict(paper="https://arxiv.org/abs/2605.01553v1", reviewed_equation="Eq. (8), Section II.A.2, page 3",
                    paper_status="arXiv preprint; journal template is not evidence of publication",
                    sampling_freq_hz=fs, observation_s=.001, receiver_m=receiver,
                    transmitter_m=transmitter, receiver_velocity_mps=[5., 0., 0.],
                    transmitter_velocity_mps=[1000., 0., 0.], transmitter_acceleration_mps2=[2., 0., 0.],
                    receiver_acceleration_cases_mps2=[case[1] for case in cases],
                    wall_bounds_m=[0., -100., 100., 0., 100.], antenna_lhcp_response=.1,
                    assumptions="Illustrative prescribed constant-acceleration trajectories, raw instantaneous samples, noiseless isolated paths, constant nav bit",
                    validation="Frequency measured from squared adjacent IQ products; no true code/carrier phase supplied to diagnostic",
                    limitations="Instantaneous shared-frame geometry; no orbital propagation, retarded transmit time, Sagnac, clock/atmospheric/relativistic corrections, RF/HIL or PVT verification",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for label, _ in cases:
        for path in ("direct", "reflection"):
            selected = [row for row in rows if row["scenario"] == label and row["path"] == path]
            t = [row["time_s"] for row in selected]
            line, = axes[0].plot(t, [row["theory_doppler_hz"] for row in selected], label=f"{label}, {path}")
            axes[0].plot(t, [row["measured_doppler_hz"] for row in selected], ".", color=line.get_color())
            axes[1].plot(t, [row["error_hz"] for row in selected], color=line.get_color())
    axes[0].set(ylabel="Doppler [Hz]", title="Sanwale et al. (2026) Eq. (8): lines=geometry, dots=generated IQ")
    axes[0].legend()
    axes[1].set(ylabel="measured minus theoretical [Hz]", title="Noiseless isolated-path diagnostic; not receiver performance")
    for axis in axes:
        axis.set_xlabel("block start time [s]")
        axis.grid(alpha=.2)
    figure.savefig(output/"doppler.png", dpi=150)
    plt.close(figure)
    print(f"Validated {len(rows)} IQ blocks; maximum Doppler discrepancy {max(abs(row['error_hz']) for row in rows):.6f} Hz")
    print(f"Saved CSV, figure and assumptions to {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/cpu-signal-dynamics")
    main(parser.parse_args().output_dir)
