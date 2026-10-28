# Continuous batch generation through acquisition and tracking

The CPU batch API now accepts keyword-only `start_samples`, with one absolute
sample offset per epoch. Omission preserves sample-zero independent epochs.
Explicit offsets let blocks share carrier/code/navigation time with individual
`generate_epoch(..., start_sample=...)` calls. The normal validation and the
single-thread/long-block fallback use the same offsets.

```python
from gnss_gpu import CpuSignalSimulator

sim = CpuSignalSimulator(noise_floor_db=-10000)
channels = [dict(prn=1, code_phase=-.5, carrier_phase=.7,
                 doppler_hz=170., amplitude=1., nav_bit=1,
                 nav_bits=[1, -1, 1])]
iq = sim.generate_epochs([channels]*30, n_samples=2600,
                         start_samples=[i*2600 for i in range(30)])
```

For unchanged channels, the concatenated noiseless output matches a single
long generation bit for bit, including navigation transitions. With noise,
use distinct `noise_seeds` per block for reproducible independent noise.
Fixed seeds retain the existing per-call convention; they do not reproduce
the RNG sequence of one long call. Channel phases/nav bits remain referenced
to sample zero, and nav sequences must cover the full requested interval.

## Integration experiment

[The experiment](../experiments/run_cpu_batch_receiver.py) compares individual
public-API calls against a batch with explicit contiguous offsets:

* 128 contiguous 1 ms blocks at 2.046 MHz, 12 composite GPS L1 C/A PRNs.
* Primary PRN 1 at 170 Hz Doppler, a fixed carrier grid −500/0/+500 Hz, and
  nominal primary thermal-noise C/N0 of 55 dB-Hz.
* Absolute code/carrier timing and illustrative block-aligned 50 bit/s nav bits.
* Signal amplitudes zero during blocks 48–63; receiver noise remains present.
* One threshold frozen from 199 separately seeded noise-only blocks, scored
  with the same full carrier grid, PRN, sampling and noise precision.
* Two confirming blocks, three unused blocks before state loss, fixed settings
  and one receiver search worker. The receiver sees IQ and timestamps only;
  outage labels and known signal parameters are used only for evaluation.

Both paths produce **bitwise-identical IQ**. Every receiver result is compared,
including coarse/refined candidates, complex fitted amplitudes, detection
scores, status, frequency/delay states, missed blocks and confirmation count.
The full result objects match exactly for all blocks and timed repetitions.

| Observed outcome | Both generation paths |
| --- | ---: |
| Input blocks | 128 |
| Signal-present blocks / fresh measurements used | 112 / 110 |
| Signal-absent blocks / measurements used | 16 / 0 |
| Confirming / acquired / tracking states | 2 / 2 / 108 |
| Coasting / lost / searching states | 2 / 1 / 13 |
| Initial acquisition decision, observation time | End of block 2, 2 ms |
| Reacquisition after signal returns, observation time | End of second returned block, 2 ms |

These times are observation-time decisions, not CPU wall-clock latency or a
new recovery-speed result. Predictions during coasting are not fresh measured
positions. Zero used noise-only blocks in this one recording does not establish
a false-lock probability or track-level integrity guarantee.

## Timing: receiver work dominates

Five timing repetitions alternate the two paths after warmup. Calibration,
JIT, equality assertions, scoring, plots and serialization are outside timing.
Numba uses four threads on the current four-core CPU quota; receiver updates
are sequential. The latest saved run reports these independent medians:

| Stage | Individual calls | Batch |
| --- | ---: | ---: |
| IQ generation | 112.25 ms | 51.95 ms |
| Receiver construction and updates | 2,442.54 ms | 2,855.93 ms |
| Complete pipeline | 2,537.99 ms | 2,904.92 ms |

Stage medians need not add to the total median. The receiver runs the exact
same code on identical inputs; its measured time varies substantially across
runs and paths. The latest total is slower for the batch path, whereas an
earlier execution showed a small improvement. **No repeatable whole-pipeline
speedup is established.** The local generation improvement must not be
extrapolated to end-to-end performance or real-time operation. Receiver work
accounts for roughly 96–98% of measured total time, so it is the next profiling
target. Timing repetitions reuse one recording and are not independent GNSS
performance trials.

A later [receiver-template reuse experiment](cpu_receiver_cache.md) keeps this
recording and threshold frozen while comparing old/new delay fitting. It
establishes receiver-side computational equivalence and reports its separate
timings; the generation-only measurements above remain historical results.

## Scope and reproduction

This is computational-equivalence validation. Signals are rectangular,
point-sampled C/A, while the receiver's fractional-delay model assumes centered
aperture averaging. This shared mismatch does not invalidate path equality but
prevents treating the test as calibrated physical ranging accuracy. Only PRN 1
is followed in the 12-PRN composite. This code follower does not implement a
PLL/FLL/DLL, bit synchronization, navigation decoding or PVT. Real RF, rain and
forest validity are not established here.

```bash
PYTHONPATH=python:. python experiments/run_cpu_batch_receiver.py
pytest -q tests/test_signal_sim_cpu.py tests/test_cpu_batch_receiver.py \
  tests/test_wall_signal_sim.py tests/test_wall_signal_dynamics.py \
  tests/test_acquisition_cpu.py tests/test_tracking_cpu.py
```

The selected suites pass **112 tests**. New checks cover whole-signal continuity,
permuted absolute offsets, ragged channels, per-block seeds, invalid intervals,
global nav-bit coverage and the receiver integration with loss/reacquisition.
`output/cpu-batch-receiver/` contains `metadata.json` (seeds, threshold, scores,
versions and source hashes), `summary.json`, complete `receiver_trace.json`,
`timings.csv`, and `receiver_equivalence.png` showing detection and state changes.
