# Independent random-data replication of the bandlimited estimator

This study repeats the [160-record development experiment](cpu_bandlimited_multipath.md)
with **128 fresh replicas per condition, 2,560 records total**. The estimator,
assistance windows, model-selection rules, bandwidths and signal conditions are
unchanged. Source hashes are checked against the development protocol before
evaluation; prior artifacts are hashed and verified unchanged at completion.

## Frozen evaluation

Each cell has replica IDs 10000–10127, disjoint from development IDs 0–7 under
the same NumPy SeedSequence construction. Direct delay, common phase and spectral
noise are paired across conditions. There are 128 independent random replicas
per cell, not 2,560 independent random realizations. This is new random data from
the same ideal synthesis implementation, not an independent physical model or
external dataset.

The complete task list and analysis plan were saved before evaluation. Twenty
cells cover 37/55 dB-Hz, one-sided bandwidths 0.921/3.683 MHz, direct-only controls,
and voltage-ratio-0.8 echoes with 0.1/0.5-chip delays at phases 0/pi/2. Each fit
has the same external coarse-delay assistance and known carrier/navigation
polarity as before. No acquisition or positioning is evaluated.

A success means an available direct-range estimate with absolute error <10 m.
All 128 attempts stay in each denominator; ambiguous selections count as failures.
The single-path baseline uses exactly the same noisy observation. Available
selected estimates can still be single-path fits rather than corrected two-path
estimates. Counts of model choice and unavailability are saved separately.

Two primary comparisons were fixed: selected versus single-path success at
37 dB-Hz, bandwidth 3.683 MHz, gap 0.5 chip, separately for phases zero and pi/2.
An exact two-sided McNemar test compares paired gains/losses. Bonferroni correction
controls the two-test family at 0.05; improvement requires a positive observed
difference and unadjusted p <0.025. This tests estimator selection at fixed
bandwidth, not the bandwidth difference itself. Other cell tests are exploratory.

Success proportions have pointwise 95% Clopper–Pearson binomial intervals.
Paired success differences also have descriptive percentile-bootstrap intervals
(20,000 draws from replica differences, fixed bootstrap seed). These intervals
are not simultaneous or family-adjusted. Bootstrap intervals can degenerate
when all observed differences agree; they do not establish zero uncertainty.
Zero observed failures likewise does not establish a zero population failure rate.

## Replication results

Both predeclared primary comparisons support improvement under this synthetic
model. At 37 dB-Hz, one-sided B = 3.683 MHz and excess delay 0.5 chip:

| Relative phase | Single successes | Selected successes | Paired gains / losses | Unavailable | Bonferroni-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 16/128 (12.5%) | 112/128 (87.5%) | 101 / 5 | 3/128 | 5.25e-24 |
| pi/2 | 70/128 (54.7%) | 108/128 (84.4%) | 48 / 10 | 6/128 | 9.03e-7 |

The selected success proportions have pointwise exact 95% intervals of
80.5–92.7% and 76.9–90.2%. Success-rate differences are +75.0 and +29.7 percentage
points; descriptive paired bootstrap intervals are +65.6 to +83.6 and +18.8 to
+39.8 points. These are changes in success probability on this fixed condition,
not a claim of that percentage improvement in positioning accuracy.

The complete grid follows. Entries are single -> selected success counts, all
out of 128; unavailable estimates remain failures.

| C/N0 | One-sided B | Direct only | 0.1 chip, phase 0 | 0.1 chip, pi/2 | 0.5 chip, phase 0 | 0.5 chip, pi/2 |
| --- | --- | --- | --- | --- | --- | --- |
| 37 | 0.921 MHz | 76 -> 76 | 42 -> 42 | 48 -> 48 | 0 -> 3 | 9 -> 49 |
| 37 | 3.683 MHz | 117 -> 117 | 30 -> 30 | 49 -> 47 | 16 -> 112 | 70 -> 108 |
| 55 | 0.921 MHz | 128 -> 128 | 0 -> 0 | 17 -> 86 | 0 -> 118 | 0 -> 122 |
| 55 | 3.683 MHz | 128 -> 128 | 0 -> 121 | 20 -> 126 | 0 -> 123 | 128 -> 122 |

The 37 dB-Hz, 0.1-chip cells show no robust correction: the selected method
largely retains the biased single-path solution. Wide bandwidth alone does not
make this close-path case reliable. At 55 dB-Hz, wide bandwidth, 0.5-chip gap and
pi/2 phase, all single-path estimates succeed but the selected method rejects
six records as ambiguous, reducing success from 128 to 122. All six report
optimizer nonconvergence; this is a numerical acceptance failure, not evidence
that the signal has no usable range information. This regression
remains in the report; the method must not be described as uniformly better.
All four direct-only controls select a single path in all 128 replicas.

![Independent replication success intervals](../output/cpu-bandlimited-multipath-replication/comparison.png)

The primary improvement survives new randomness without receiver tuning, which
is stronger evidence than the initial eight-replica examples. It remains evidence
within one ideal, assisted spectral model. The next development target is to
understand the avoidable availability losses and validate any changed selection
rule on another untouched replica set, while preserving this frozen result.

## Reproduction and audit

```bash
source /workspace/.gnss-env/activate.sh
pytest -q tests/test_bandlimited_replication.py
python experiments/validate_cpu_bandlimited_multipath.py
python experiments/audit_cpu_bandlimited_replication.py
```

The experiment requires an empty `output/cpu-bandlimited-multipath-replication`
directory; preserve existing artifacts before any rerun. It uses four CPU
processes and streams all fitted diagnostics, observation hashes and errors to
JSONL. Protocol, summary, audit and PNG/PDF figures are separate artifacts.
Six new tests check paired-discordance statistics, an analytically known exact
probability, nonzero binomial uncertainty at extreme counts, and invalid pairs.
The audit verifies all record identities, denominators, paired contingency tables,
reported statistics and hashes, and exactly replays two full synthesis/fit cases.
The replay uses the same implementation and covers only two records.

The frozen receiver retains the preceding 23 passing estimator/information tests.
Replication does not validate real forest/rain channels, receiver acquisition,
tracking, PVT, or robustness to wrong assistance and noise models.
