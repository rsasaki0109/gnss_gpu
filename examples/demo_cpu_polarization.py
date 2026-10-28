"""Paper-linked circular polarization composition in the moving-wall IQ scene.

PYTHONPATH=python python examples/demo_cpu_polarization.py
Uses illustrative antenna voltage responses, not measured antenna patterns.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gnss_gpu import CpuSignalSimulator
from gnss_gpu.code_delay_cpu import estimate_ca_delay
from gnss_gpu.wall_signal_sim import VerticalWall, simulate_wall_signal


def main(output_dir="output/cpu-polarization"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fs, count = 20.46e6, 20460
    sim = CpuSignalSimulator(fs, noise_floor_db=-10000)
    rx, velocity, satellite = np.array([-20., -20., 1.5]), [1., 5., 0.], np.array([-1e7, 1e7, 1e7])
    reference = float(np.linalg.norm(satellite-rx))
    responses = [("ideal RHCP", 0j), ("LHCP -20 dB, 0 deg", .1+0j),
                 ("LHCP -20 dB, 90 deg", .1j), ("LHCP -20 dB, 180 deg", -.1+0j)]
    rows = []
    for time in np.linspace(0., 2., 101):
        args = dict(start_sample=round(time*fs), reference_range_m=reference, integrate_samples=True)
        ideal_wall = VerticalWall(0., -30., 5., 0., 100.)
        direct = simulate_wall_signal(sim, ideal_wall, rx, velocity, satellite, count,
                                      include_reflection=False, **args)
        direct_estimate = estimate_ca_delay(direct.iq, fs)
        assert direct_estimate is not None
        for label, response in responses:
            wall = VerticalWall(0., -30., 5., 0., 100., antenna_lhcp_response=response)
            full = simulate_wall_signal(sim, wall, rx, velocity, satellite, count, **args)
            estimate = estimate_ca_delay(full.iq, fs)
            assert estimate is not None
            coefficient = full.reflection_coefficient[count//2]
            rows.append(dict(time_s=float(time), antenna=label,
                             receiver_lhcp_real=response.real, receiver_lhcp_imag=response.imag,
                             reflection_visible=bool(full.reflection_visible[count//2]),
                             reflected_voltage_abs=float(abs(coefficient)),
                             reflected_voltage_phase_rad=float(np.angle(coefficient)) if coefficient else None,
                             direct_delay_m=direct_estimate.delay_m, composite_delay_m=estimate.delay_m,
                             incremental_code_bias_m=estimate.delay_m-direct_estimate.delay_m))
    with (output/"polarization.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    root = Path(__file__).resolve().parents[1]
    provenance_path = root/"docs/gnss_signal_simulation_sources.json"
    sources = ["python/gnss_gpu/fresnel.py", "python/gnss_gpu/wall_signal_sim.py",
               "python/gnss_gpu/code_delay_cpu.py", "examples/demo_cpu_polarization.py"]
    metadata = dict(sampling_freq_hz=fs, observation_s=.001, poses=101,
                    papers=["10.1007/s10291-013-0331-y", "10.1007/s10291-014-0370-z"],
                    verified_source=json.loads(provenance_path.read_text()),
                    incident_rhcp=1, incident_lhcp=0, receiver_rhcp=1,
                    antenna_assumption="Illustrative constant complex LHCP voltage responses; .1 voltage = -20 dB power, arbitrary phases, not measured antenna calibration",
                    scene=dict(receiver_m=rx.tolist(), velocity_mps=velocity, transmitter_m=satellite.tolist(),
                               reference_range_m=reference, wall_bounds_m=[0., -30., 5., 0., 100.], material="concrete"),
                    interpretation="Subset of author-published polarization composition; not a reproduction of a paper figure or full MPSim",
                    source_sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in sources})
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for label, _ in responses:
        selected = [row for row in rows if row["antenna"] == label]
        t = [row["time_s"] for row in selected]
        axes[0].plot(t, [row["reflected_voltage_abs"] if row["reflection_visible"] else np.nan for row in selected], label=label)
        axes[1].plot(t, [row["incremental_code_bias_m"] for row in selected], label=label)
    axes[0].set(ylabel="effective reflected voltage magnitude", title="Nievinski-Larson supplementary polarization model; illustrative receive responses")
    axes[1].set(ylabel="composite minus direct-only delay [m]", title="Sample-integrated CPU IQ; noiseless single-wall experiment")
    for axis in axes:
        axis.set_xlabel("time [s]")
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(output/"polarization.png", dpi=150)
    plt.close(figure)
    for label, _ in responses:
        peak = max(abs(row["incremental_code_bias_m"]) for row in rows if row["antenna"] == label)
        print(f"{label}: maximum incremental code-delay bias {peak:.3f} m")
    print(f"Saved {len(rows)} observations, plot and source provenance to {output}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output/cpu-polarization")
    main(parser.parse_args().output_dir)
