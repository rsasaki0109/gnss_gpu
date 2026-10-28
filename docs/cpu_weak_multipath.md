# Direct-plus-echo weak receiver stress

The new `fit_weak_ca_two_paths` wrapper coherently combines the acquired 20 ms
record, then applies the existing local two-path aperture fitter. Carrier and
bit-edge hypotheses come from acquisition, not simulation truth. Original
acquisition acceptance is unchanged. Ambiguous fits return no selected delay;
the receiver does not silently replace them with truth or a preferred path.

The study contains 448 records: two direct-path C/N0 levels (37/55 dB-Hz), eight
replicas, a direct-only control, and 27 echo conditions. Echo/direct voltage
ratios are 0.3, 0.8 and 1.5; excess delays are 0.1, 0.5 and 1 chip (approximately
29.3, 146.5 and 293.1 m); net relative phases are 0, pi/2 and pi. Unknown common
carrier phase and fractional direct delay vary by replica. The same noise and
initial parameters are paired across cells. This is exact centered-aperture
sampling of two static rectangular code waveforms, isolated PRN 1 at zero
Doppler, with constant navigation polarity over 20 ms.

The previously frozen PRN 1 white-null gate is reused with identical search and
white-noise processing. Signal synthesis changes, not the null search. The
post-gate two-path BIC/conditioning rules are diagnostics, not a newly calibrated
multipath alarm. The fitter assumes the earlier path is stronger: ratio 1.5 is
an intentional assumption-violation stress. No forest, rain or physical wall
reflection coefficient is inferred from these synthetic voltage/phase choices.

## Results

A successful range below means an available estimate with absolute circular
error below 10 m. All 72 replicas/cells per nonzero amplitude remain in the
success denominator, including acquisition failure and ambiguous estimates.
The selected method may retain a single-path model; it is not always a correction.

| Direct C/N0 | Echo voltage ratio | Single successes | Selected successes | Ambiguous fits |
| --- | ---: | ---: | ---: | ---: |
| 37 | 0.3 | 36/72 | 36/72 | 1 |
| 37 | 0.8 | 16/72 | 15/72 | 23 |
| 37 | 1.5 | 0/72 | 0/72 | 10 |
| 55 | 0.3 | 39/72 | 32/72 | 26 |
| 55 | 0.8 | 15/72 | 20/72 | 35 |
| 55 | 1.5 | 0/72 | 2/72 | 36 |

Direct-only controls produce identical success counts: 6/8 at 37 dB-Hz and
8/8 at 55 dB-Hz. These are descriptive results over a selected finite grid;
shared replicas across cells are not independent trials.

An illustrative favorable cell at 55 dB-Hz, ratio 0.8, half-chip delay and pi/2
phase retains all eight records and changes median range error from **35.85 m
to 1.66 m**. It is not representative of all phases. At the same amplitude/delay
with phase pi, six of eight fits are ambiguous; the two remaining records have
median selected error 41.00 m versus 74.23 m for single-path on those same two
records. Reporting only retained medians would hide the availability loss.

The current method therefore does **not** establish robust mitigation at
37 dB-Hz. At two samples/chip, nearby aperture templates can occupy the same
piecewise-affine interval and become nonidentifiable. More signal power alone
cannot resolve that degeneracy. Stronger echoes can also violate the
LOS-dominant assumption. A higher sampling-bandwidth/model study and better
ambiguity handling are needed before selecting this method for positioning.
No refined multipath ranges have been inserted into the existing PVT pipeline.

![Success-count change by echo condition](../output/cpu-weak-multipath/comparison.png)

## Reproduction and checks

Run `source /workspace/.gnss-env/activate.sh`, then
`python experiments/run_cpu_weak_multipath.py`. The output directory must be
empty. `output/cpu-weak-multipath` stores protocol/source and frozen-gate hashes,
all candidates, IQ hashes, failure reasons, per-record errors and 56 cell
summaries. Comparisons of medians use exactly the same selected subset; success
counts retain every attempted record. The preceding coarse-position and
fractional-position results are preserved.

20 relevant tests pass, including coherent edge handling, agreement with the
underlying fitter, consistency with the single-path estimator, input preservation,
physical aperture checks and rank-degenerate short-path controls. This is a
controlled range study, not a positioning or real-RF performance claim.
