"""Demo: GPU GNSS signal simulation + acquisition round-trip."""

from pathlib import Path

import numpy as np

from gnss_gpu.signal_sim import SignalSimulator
from gnss_gpu.acquisition import Acquisition


def main():
    sim = SignalSimulator()

    prns = [3, 7, 13, 17, 23, 28]
    dopplers = np.linspace(-3000, 3000, len(prns))
    channels = [
        {
            "prn": prn,
            "code_phase": 0.0,
            "carrier_phase": 0.0,
            "doppler_hz": float(doppler),
            "amplitude": 1.0,
            "nav_bit": 1,
        }
        for prn, doppler in zip(prns, dopplers)
    ]

    # 10ms of signal
    n_samples = int(sim.sampling_freq * 10e-3)
    iq = sim.generate_epoch(channels, n_samples=n_samples)

    out_path = Path("output/demo_signal.bin")
    sim.write_bin(iq, out_path, fmt="int8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Acquire using I channel
    signal = iq[0::2].copy()
    acq = Acquisition(sampling_freq=sim.sampling_freq,
                      intermediate_freq=sim.intermediate_freq)
    results = acq.acquire(signal, prn_list=list(range(1, 33)))

    print(f"\n{'PRN':>3} | {'Acquired':>8} | {'Doppler':>10} | {'CodePhase':>10} | {'SNR':>8}")
    print("-" * 52)
    for r in results:
        if r["prn"] in prns or r["acquired"]:
            print(f"{r['prn']:>3} | {str(r['acquired']):>8} | "
                  f"{r['doppler_hz']:>10.1f} | "
                  f"{r['code_phase']:>10.1f} | "
                  f"{r['snr']:>8.2f}")


if __name__ == "__main__":
    main()
