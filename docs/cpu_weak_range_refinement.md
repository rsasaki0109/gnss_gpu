# Fractional weak-signal range refinement

`refine_weak_ca_delay` coherently combines the acquired 20 ms record using its
estimated carrier and bit-edge hypothesis, omits the transition row, and applies
the existing analytic fractional aperture-delay fitter. It is a post-detection
estimator: its fitted power is never substituted into the acquisition gate.
Input must be centered integrate-and-dump samples. The prior instantaneous
rectangular samples can be identical for different fractional delays; interpolation
alone cannot recover missing information.

A fresh 64-replica paired study uses independent random fractional delays and
carrier phases, unknown within-row navigation reversals, and white noise. Each
pair uses exactly the same aperture IQ for integer and fractional estimation.
The unchanged PRN 1 white-null gate from the isolated-channel positioning study
accepts all 64 records at each signal level. Only the signal sampling model
changes relative to that older study; the white-null acquisition algorithm is
unchanged. This is not a direct comparison against the old point-sampled PVT.

| C/N0 | Integer range error median / p95 | Fractional range error median / p95 |
| --- | ---: | ---: |
| 37 dB-Hz | 34.97 / 74.85 m | 7.34 / 26.43 m |
| 55 dB-Hz | 33.85 / 69.68 m | 0.92 / 2.78 m |

These are absolute circular code-range errors, conditional on detection, for
one isolated static PRN at zero Doppler. They are not position errors, uncertainty
bounds, multipath results or forest/rain validation. Delay/carrier drift, imperfect
edge selection and real front-end filtering remain unvalidated. The aperture
model is idealized; numerical agreement is not an RF accuracy specification.

The new module does not change the frozen weak tracker or prior PVT results.
Applying this estimator to positioning requires aperture-compatible multi-PRN IQ
and a new paired position experiment. The receiver is never supplied the true
delay, phase or bit-edge location; those enter synthesis and offline scoring.

Reproduce with `source /workspace/.gnss-env/activate.sh` followed by
`python experiments/validate_cpu_weak_range_refinement.py`. Output is retained
in `output/cpu-weak-range-refinement`: protocol/source hashes, frozen-gate hash,
all 128 outcomes and IQ hashes, and aggregate statistics. Nonempty output is
rejected. 34 relevant tests pass, including unknown-edge acquisition, fractional
and wrapped delays, independent aperture integration and input preservation.
