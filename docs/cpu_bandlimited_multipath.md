# Bandlimited two-path estimation

The [information-bound study](cpu_multipath_information.md) now has a matching
spectral-domain estimator and a paired noisy-signal experiment. This is an
assisted local estimation study, **not end-to-end acquisition or positioning**.
No existing detector thresholds are reused on the new signal/noise model.

## Estimator and experiment

`fit_bandlimited_paths` accepts complex Fourier coefficients in the passband.
It fits two ordered delays and two unknown complex amplitudes, without requiring
the earlier path to be stronger. Linear least squares eliminates amplitudes;
twelve deterministic grid seeds initialize bounded two-dimensional optimization.
It also fits a single-path baseline to the same observation. Returned amplitude
magnitudes use an internal observation normalization; they are not physical volts.

All records use the same externally supplied coarse window: earliest delay
within [-1,1] chips, excess delay within [0.01,1.2] chips. The single-path search
covers [-1,2.2] chips to include a possible dominant echo. Neither search window
is centered on simulation truth. True direct delay varies uniformly in [-0.3,0.3]
chips; common complex phase is random. Carrier and navigation polarity are known,
and 20 ms integration is ideal/coherent. This assistance is not a blind search.

Selection uses a 3- versus 6-real-parameter BIC comparison. A preferred two-path
fit must converge, lie away from search boundaries, and have estimated local
direct-delay standard deviation <=0.1 chip. Otherwise it is ambiguous/unavailable.
The uncertainty estimate uses residual noise power and the projected delay
Jacobian. A single-path BIC preference can still publish a biased single-path
delay; these checks are **not calibrated detection, integrity or confidence
coverage guarantees**. Discrete model selection is nonregular near coincident or
vanishing paths. Local convergence does not guarantee a global optimum.

The 160 records span two one-sided cutoffs B (0.921/3.683 MHz), two direct-path
pre-filter C/N0 levels (37/55 dB-Hz), eight replicas, and five conditions:
direct-only, plus echo/direct voltage 0.8 at gaps 0.1/0.5 chip and relative phases
0/pi/2. The underlying model is the preceding ideal periodic spectral Gaussian
model: each coefficient has proper complex noise variance `1/(C/N0*T)`.
Signal power is not renormalized after filtering. Identical harmonic noise,
direct delay and common phase are paired across conditions and bandwidths.
These 160 records are not 160 independent replicas.

## Results

A success requires an available estimate with absolute direct-range error <10 m.
Every attempted record stays in the denominator, including ambiguous selections.
Each entry below is single-path successes -> selected-method successes, out of 8.

| C/N0 | B (one-sided) | Direct only | 0.1 chip, phase 0 | 0.1 chip, pi/2 | 0.5 chip, phase 0 | 0.5 chip, pi/2 |
| --- | --- | --- | --- | --- | --- | --- |
| 37 | 0.921 MHz | 6 -> 6 | 1 -> 1 | 3 -> 3 | 0 -> 0 | 0 -> 4 |
| 37 | 3.683 MHz | 7 -> 7 | 2 -> 2 | 5 -> 5 | 0 -> 7 | 6 -> 8 |
| 55 | 0.921 MHz | 8 -> 8 | 0 -> 0 | 0 -> 5 | 0 -> 8 | 0 -> 8 |
| 55 | 3.683 MHz | 8 -> 8 | 0 -> 8 | 0 -> 8 | 0 -> 8 | 8 -> 8 |

Two of eight selections at 55 dB-Hz, narrow bandwidth, 0.1-chip gap and pi/2
phase are ambiguous. Every other cell has eight available selected estimates;
availability alone does not imply correctness or two-path selection.

For 37 dB-Hz, wide bandwidth and a 0.5-chip gap, the raw two-path RMSE is
4.96 m at phase zero and 4.49 m at pi/2. The corresponding local theoretical
standard-deviation bounds are 5.67 m and 5.64 m. An eight-replica finite-sample
RMSE below a local unbiased bound does not violate the bound or establish
efficiency: the estimator is constrained, can be biased, and has model selection.
For the same wide bandwidth at 0.1-chip gap and phase zero, raw two-path RMSE
rises to 142.83 m, versus a 21.25 m local bound. The selected method succeeds in
only 2/8 records. Local information does not prevent gross path-selection errors.

The report preserves raw two-path errors even when that model is rejected.
Direct-only controls can produce arbitrary noise-fitting second paths and large
raw earliest-delay errors; the selected method retains the single-path solution
in those controls. Selected RMSE is explicitly conditional on availability and
must not be read without its denominator. No failed records are silently removed.

![Selected range successes](../output/cpu-bandlimited-multipath/comparison.png)

These results support bandwidth-dependent improvements for this small synthetic
grid, not general weak-signal mitigation. They do not validate forest/rain channel
physics, real RF noise, unknown carrier/nav, composite satellites, acquisition,
tracking or positioning. Next priorities are stronger independent replication
and full-pipeline acquisition calibration before PVT integration.

## Reproduction and checks

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_bandlimited_multipath_cpu.py tests/test_multipath_information_cpu.py
python experiments/run_cpu_bandlimited_multipath.py
python experiments/audit_cpu_bandlimited_multipath.py
```

The experiment requires an empty `output/cpu-bandlimited-multipath` directory
and uses four CPU processes. Existing artifacts must be preserved elsewhere
before any rerun. The protocol and source hashes are written before evaluation;
all 160 observations have hashes, fitted diagnostics and error records. Plots
are generated by the script. Twenty-three tests pass, including noiseless
two-path recovery with a stronger echo, single-path/noisy controls, phase/scale
invariance, boundary rejection and the preceding independent Fisher checks.
The audit verifies all saved denominators/RMSE and source hashes, then exactly
replays two records including synthesis and optimization. This limited replay
uses the same implementation; it is not independent physical validation.
