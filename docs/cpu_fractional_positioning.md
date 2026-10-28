# Fractional range refinement connected to eight-satellite positioning

Fresh paired aperture-IQ trials now carry the fractional delay estimate into
assisted range resolution and position/clock least squares. The same IQ,
acquisition gates, confirmation states, geometry and assistance are used by
both methods. Integer candidates and fractional fits are both instantaneous
window estimates, avoiding a smoothing-versus-estimator comparison. The existing
tracker still supplies acceptance/confirmation; fractional estimates do not
change its gate or state history.

| Condition | Integer position error median / p95 | Fractional position error median / p95 |
| --- | ---: | ---: |
| 55 dB-Hz | 69.25 / 69.25 m | **2.08 / 4.65 m** |
| 37 dB-Hz | 69.25 / 101.52 m | **16.01 / 42.25 m** |
| Three paths +300 m | 324.31 / 386.23 m | 311.63 / 325.08 m |
| Common clock +300 m | 69.25 / 101.52 m | 16.17 / 34.47 m |

Each condition uses 16 independent replicas with three correlated scored windows
per replica; all 48 post-confirmation epochs solve for both methods. When only
three PRNs remain, both methods report **0/48** positions. All initial pending
windows and unavailable outcomes are retained. There are 320 physical epochs
and 640 method outcomes, not 640 independent trials. These descriptive quantiles
are not confidence bounds or significance tests. The repeated integer result
at high signal strength reflects a fixed geometry and quantized code delays.

The 300 m excess-path bias persists after refinement: improving resolution does
not remove propagation bias. Common clock changes are estimated separately.
These effects are illustrative synthetic conditions, not validated forest or
rain physics. This remains isolated per-PRN, static zero-Doppler IQ with known
code epochs and coarse position/clock assistance, not a composite RF receiver.

The generator exactly integrates the piecewise-constant C/A code and navigation
polarity over a centered sampling aperture, splitting apertures at bit boundaries.
Tests compare this implementation against independent dense numerical integration
and check noiseless IQ-to-assisted-range recovery for multiple PRNs. Arbitrary
carrier phases and channel noise are shared between paired conditions. This
sampling model differs from the old point-sampled positioning experiment, so
its approximately 102 m baseline must not be used for an estimator-only comparison.

New white-null gates use 599 records shared across eight PRNs, with the same
1% marginal budget and 5% family calibration-failure allocation. Another 199
records evaluate the gates. Fractional fitting occurs only after a fresh accepted
state; it supplies no new detection statistic or integrity guarantee. No claim
about near-far interference, dynamic tracking or unassisted ambiguity resolution
follows from the result.

Reproduction:

```bash
source /workspace/.gnss-env/activate.sh
python experiments/run_cpu_fractional_positioning.py
python experiments/audit_cpu_fractional_positioning.py
```

Use `--output` for another run; existing results cannot be overwritten. Outputs
in `output/cpu-fractional-positioning` include protocol/source hashes, frozen
per-PRN gates, calibration/evaluation records, IQ hashes, both methods' states,
ranges, positions and failures. The audit reproduces all eight gates and all
640 saved-state-to-position outcomes; this is not independent IQ replay evidence.
19 relevant tests pass. The paired methods are integrated in the experiment;
the production weak tracker retains its existing integer state representation.

![Paired position errors](../output/cpu-fractional-positioning/comparison.png)
