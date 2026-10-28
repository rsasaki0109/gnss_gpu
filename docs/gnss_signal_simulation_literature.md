# Literature-backed GNSS signal simulation

## Sources and review status

The implemented components use GNSS multipath supplementary material and a
full-text-reviewed trajectory-driven signal-simulation preprint:

| Source | Role in this implementation |
| --- | --- |
| F. G. Nievinski and K. M. Larson (2014), **Forward modeling of GPS multipath for near-surface reflectometry and positioning applications**, GPS Solutions 18(2), 309–322. [DOI](https://doi.org/10.1007/s10291-013-0331-y) | Physical composition of incident polarization, reflected polarization and receiver response. |
| F. G. Nievinski and K. M. Larson (2014), **An open source GPS multipath simulator in Matlab/Octave**, GPS Solutions 18(3), 473–481. [DOI](https://doi.org/10.1007/s10291-014-0370-z) | Author-published simulator and scripts that identify the paper figures and reproducible model components. |
| J. Sanwale, M. Kothari, H. B. Hablani and S. Dahiya (2026), **Physics Driven Digital Twin Model for Evaluation of GNSS User Receiver Equipment**, [arXiv:2605.01553v1](https://arxiv.org/abs/2605.01553v1) | Full PDF retrieved and relevant methods reviewed. Section II.A.2, Eq. (8), page 3 supplies the relative-velocity Doppler reference. This is a preprint; its journal-style layout does not establish peer-reviewed publication. |
| I. Iudice et al. (2024), **A real/fast-time simulator for impact assessment of spoofing & jamming attacks on GNSS receivers**, [arXiv:2405.17925v2](https://arxiv.org/abs/2405.17925v2), [DOI](https://doi.org/10.1109/MetroAeroSpace61015.2024.10591529) | Full accepted-author PDF retrieved; Section II models effects on measurements. Used to distinguish observation-domain simulation from IQ generation; no attack mechanism implemented. |

**Review boundary:** network access to the publisher and its identity host now
works. The official PDF requests for both 2014 articles return HTTP 200 **HTML
subscription previews**, not PDFs. Their public abstracts, metadata and reference
lists were reviewed, but full text requires publisher entitlement or a lawful
author/user-supplied copy. No additional network publication resolves that
subscription requirement. Their authors, journal, volume, issue, pages and
print/online dates were independently verified through Crossref. Their model
details below remain checked against the authors' public supplement. The 2026
and 2024 arXiv PDFs were retrieved at explicit version URLs and their relevant
methods read; PDF/text hashes and reviewed sections are recorded. Equation
numbers below refer only to a retrieved full text. No published figure or HIL
performance reproduction is claimed.

The supplement is pinned to MPSim commit
[`61be6278d49779d539bc7b3e19f6a13cefcc30a8`](https://github.com/ufrgs-gnss-lab/mpsim/tree/61be6278d49779d539bc7b3e19f6a13cefcc30a8).
Verified URLs and SHA-256 values are recorded in
[`gnss_signal_simulation_sources.json`](gnss_signal_simulation_sources.json).
The supplement is BSD-2-Clause licensed. No MATLAB files or article PDFs are
vendored in this repository; the extension uses our existing NumPy Fresnel
calculation and an explicit field/basis derivation.

## Why polarization comes first

The existing wall IQ model used one selected Fresnel coefficient. Its default
`polarization="rhcp"` implicitly assumed an ideal receiver insensitive to LHCP.
GNSS multipath changes polarization; a finite opposite-hand receive response
can change both reflected amplitude and phase. Merely attenuating NLOS paths,
or adding received powers, does not describe coherent interference.

This extension adds the missing receive-polarization composition without
changing the default waveform or introducing unverified surface parameters.
It targets one identifiable component of the published simulator, not its
complete surface/antenna/receiver pipeline.

## Verified model and implementation mapping

For complex parallel/perpendicular electric-field coefficients, use the circular
convention already present in this repository:

```text
R_same  = (R_parallel + R_perpendicular) / 2
R_cross = (R_parallel - R_perpendicular) / 2

E_ref_R = R_same * E_inc_R + R_cross * E_inc_L
E_ref_L = R_cross * E_inc_R + R_same * E_inc_L

V_ref = h_R * E_ref_R + h_L * E_ref_L
```

`E_inc_*` and `h_*` are complex **field/voltage** quantities. These are receive
responses defined by the final voltage equation, not analyzer vectors that
should be conjugated a second time. Power is formed only after summing fields:
`|V_ref|²`. A -20 dB opposite-hand power response corresponds to voltage magnitude
`10**(-20/20) = 0.1`, not 0.01. Its phase still affects interference.

| Verified supplementary source | Local implementation |
| --- | --- |
| [`get_reflection_coeff_linear.m`](https://github.com/ufrgs-gnss-lab/mpsim/blob/61be6278d49779d539bc7b3e19f6a13cefcc30a8/lib/snr/fwd/get_reflection_coeff_linear.m) | Existing `fresnel_coefficients`; incidence is measured from the normal, whereas the supplement takes grazing elevation. |
| [`get_reflection_coeff_linear2circ.m`](https://github.com/ufrgs-gnss-lab/mpsim/blob/61be6278d49779d539bc7b3e19f6a13cefcc30a8/lib/snr/fwd/get_reflection_coeff_linear2circ.m) | Circular same-/opposite-hand coefficients in `circular_reflection_voltage`. |
| [`snr_fwd_reflected.m`](https://github.com/ufrgs-gnss-lab/mpsim/blob/61be6278d49779d539bc7b3e19f6a13cefcc30a8/lib/snr/fwd/snr_fwd_reflected.m) | Complex incident fields mixed into two reflected circular fields, then coherently summed through receive responses. |
| `script/figA4.m`, `script/extra/figB4.m` | Inspected figure drivers identify roughness and antenna effects as separate model components; these figures are not reproduced by the new demo. |

Our material permittivity, finite-wall geometry, aperture sampling and
fractional-delay estimator remain this repository's models. They are not
represented as reproductions of MPSim's material, antenna or receiver models.
The roughness coefficient and surface-height definitions have not been
cross-checked against article text, so no roughness law was imported here.

## API and wall IQ connection

The general field helper supports arbitrary scalar incident circular fields
and scalar receiver voltage responses:

```python
from gnss_gpu.fresnel import circular_reflection_voltage

voltage = circular_reflection_voltage(
    incidence_angle_rad=0.7,
    material="concrete",
    incident_rhcp=1+0j,
    incident_lhcp=0j,
    receiver_rhcp=1+0j,
    receiver_lhcp=0.1j,
)
```

The finite-wall simulator uses the controlled subset `E_inc_R=1`, `E_inc_L=0`,
`h_R=1`, with a configurable `h_L`:

```python
from gnss_gpu.wall_signal_sim import VerticalWall

wall = VerticalWall(
    0., -30., 5., 0., 100.,
    material="concrete",
    polarization="rhcp",
    antenna_lhcp_response=0.1j,
)
```

This gives `R_effective = R_same + h_L*R_cross`. The direct incident RHCP voltage
stays normalized to one, so changes in reflected response do not rescale the
direct signal or noise. Zero LHCP response retains the previous ideal-RHCP
path. A nonzero response is rejected with linear or cross-only polarization
modes, whose meanings do not match this receive basis.

These parameters refer to the reflected ray's local circular basis. A constant
response is an illustrative approximation. There is no measured antenna
pattern, angle-dependent effective length, antenna orientation/phase-center
model or global-to-local polarization transport in this extension. A voltage
transfer is not a reflectance and need not be clipped to one for arbitrary
user-supplied receive responses.

## Validation and reproducible comparison

```bash
PYTHONPATH=python python -m pytest tests/test_polarization_voltage.py tests/test_fresnel.py tests/test_wall_signal_sim.py -q
PYTHONPATH=python python examples/demo_cpu_polarization.py
```

Tests check the circular composition against an independent unitary
circular/linear basis calculation and dielectric boundary coefficients, verify
the ideal RHCP limit, and verify the exact normal-incidence case `eps=4`, where
`R_same=0` and `R_cross=1/3`. They also check the -20 dB voltage/power conversion,
receive-phase interference, unchanged direct IQ and coherent sum of isolated
paths in the CPU simulator.

The demo uses the existing moving-wall scene, with 101 positions and four
illustrative receiver responses: ideal RHCP and LHCP response of magnitude 0.1
at phases 0, 90 and 180 degrees. These values are **assumptions for comparison**,
not measured antenna data or parameter values extracted from the papers. It
writes `output/cpu-polarization/polarization.csv`, `polarization.png` and
`metadata.json`, including the literature provenance and local source hashes.
The plots show effective reflected voltage and incremental code-delay bias
from sample-integrated IQ. They are new controlled experiments, not replicas
of published figures or claims about real receiver accuracy.

## Next literature checks

After legitimate full-text access to the 2014 articles becomes available, check the full physical assumptions
and notation against the supplement, including roughness and antenna
normalization. Then extend one component at a time with a reproducible reference
case. Useful subsequent targets are measured RHCP/LHCP antenna responses,
coherent versus diffuse rough-surface scattering, and a published receiver
correlator/front-end model. Do not retune previously calibrated detector
thresholds on their held-out evaluation data.

## Full-text-driven extension: moving transmitter and accelerating receiver

Sanwale et al. (2026), Section II.A.2, Eq. (8), page 3, specifies geometric
carrier Doppler from relative velocity:

```text
d(t)       = p_sat(t) - p_rx(t)
R(t)       = ||d(t)||
R_dot(t)   = dot(v_sat(t) - v_rx(t), d(t)) / R(t)
f_D(t)     = -f_L1 * R_dot(t) / c
```

The finite-wall simulator now accepts `transmitter_velocity_mps`,
`rx_acceleration_mps2`, and `transmitter_acceleration_mps2`. Both bodies follow
the explicitly supplied trajectory `p0 + v0*t + 0.5*a*t²`. The reflected path
uses the transmitter mirrored across the fixed wall; its normal velocity and
acceleration components are mirrored too. Differentiating the image-path
length yields that path's range rate and Doppler. Returned blocks expose
`direct_range_rate_mps`, `reflected_range_rate_mps`, `direct_doppler_hz` and
`reflected_doppler_hz`; the Doppler properties exclude intermediate frequency.

One geometric delay drives both spreading code and carrier:

```text
tau(t)   = (R(t) - reference_range_m) / c
chip(t)  = floor(f_chip * (t - tau(t))) mod 1023
phase(t) = 2*pi * (f_IF*t - f_L1*tau(t))
```

This preserves `d phase/dt / (2*pi) = f_IF + f_D(t)` and received code rate
`f_chip*(1 - R_dot/c)`. It does not approximate an evolving carrier phase with
`2*pi*f_D(t)*t`, whose derivative would add a spurious `t*d f_D/dt` term.
The same positions/velocities and sample-zero reference must be supplied to
every contiguous block. The existing sample-aperture approximation remains
locally affine during each aperture, with instantaneous range rate from these
trajectories.

This is a selected component of the preprint's framework. No orbital
propagator, retarded satellite transmit-time solution, Earth-rotation/Sagnac,
clock, atmospheric or relativistic corrections have been added. The paper's
SDR/HIL, navigation decoding, PVT and tracking-loop results have not been
reproduced. Its Section II.A.3 noise statement uses total complex variance
`N0*fs`, consistent with our `N0*fs/2` per real I/Q component; this is a convention
check, not validation against RF measurements.

```bash
PYTHONPATH=python python -m pytest tests/test_wall_signal_dynamics.py -q
PYTHONPATH=python python examples/demo_cpu_signal_dynamics.py
```

The demo compares constant-velocity and accelerated receiver scenarios with a
moving/accelerating transmitter, for both isolated direct and reflected paths.
The 200 m/s² acceleration and all trajectory/antenna values are illustrative
test settings, not values claimed to reproduce the paper's figures. It records
84 blocks of 1 ms noiseless GPS IQ and compares geometric Doppler with an
adjacent-sample BPSK-squaring diagnostic. The diagnostic removes unknown code
polarity without being given true code or carrier phase. It is only unambiguous
within +/-fs/4 and is not a general acquisition or tracking algorithm.
CSV, plot and assumptions are written to `output/cpu-signal-dynamics/`.

Tests use radial closed-form kinematics, independent finite differences for
oblique direct/image paths, Doppler measured from generated IQ, code/carrier
consistency, and exact contiguous-block equality with and without aperture
integration. These synthetic checks demonstrate internal consistency, not
real-world ranging accuracy or a complete GNSS simulator.

### Receiver-side carrier search following the dynamics check

`acquisition_cpu.py` now searches a fixed received-frequency grid jointly with
code delay, using the existing aperture fitter and an independently calibrated
maximum-over-grid detection gate. This receiver diagnostic is our implementation
choice for evaluating the generated dynamics; it is **not** presented as an
acquisition algorithm reproduced from Sanwale et al. or the 2014 papers.
`experiments/run_cpu_carrier_search.py` withholds true carrier and delay from
the search, then compares outputs against geometric range/range-rate truth.
It includes low-SNR false peaks and rejected reflected paths, rather than
assuming that acquisition always succeeds. The [CPU guide](cpu_signal_simulation.md#unknown-carrier-acquisition-on-cpu)
records assumptions, full counts and the observed false-alarm rate above the
nominal target. No tracking-loop, receiver hardware or PVT claim is added.

`tracking_cpu.py` further adds bounded carrier-profile refinement and temporal
frequency/code-state following, evaluated in `experiments/run_cpu_tracking.py`.
Its geometric truth still comes from the reviewed Eq. (8). Golden-section
optimization, exponential smoothing, carrier-aided code prediction and state
transition rules are our receiver-side engineering choices, not methods claimed
to reproduce this paper. The original coarse calibrated statistic decides
detection before refinement. The synthetic outage test retains a false noise
capture, and these results do not establish a phase-locked receiver, track-level
false-alarm control or RF accuracy. See the [implementation and limits](cpu_signal_simulation.md#carrier-refinement-and-contiguous-block-following).

Acquisition now also defaults to two consecutive frequency/code-consistent
blocks before publishing a state. This is a receiver engineering extension,
not an acquisition procedure attributed to the reviewed papers. A paired
single-/two-block comparison separates the old failure seed from two fresh
evaluation seeds, preserves coarse per-block calibration, and records the
additional acquisition latency and remaining ranging errors. The
[confirmation experiment](cpu_signal_simulation.md#multi-block-acquisition-confirmation)
does not claim sequence-level false-track guarantees from per-block statistics.

The [continuous wall-tracking experiment](cpu_signal_simulation.md#continuous-tracking-with-specular-multipath)
now combines the author-supplement-based complex polarization model with the
geometry-driven code/carrier model and a stateful receiver. Direct-only and
two polarization variants share the same noise while a specular point exits
the finite wall. A noiseless control separates multipath bias from additive
noise; two fresh noise seeds retain all measurement/confirmation counts.
The chosen geometry, antenna responses and receiver are our diagnostic setup,
not reproduced paper figures or measured antenna data. Successful temporal
confirmation still permits metre-scale code bias, and no mitigation, full
MPSim reproduction or new full-text access to the 2014 papers is claimed.

`multipath_fit_cpu.py` introduces an experimental two-path aperture fit using
linear amplitude elimination and local projected Gauss-Newton delay updates.
This is our receiver engineering work, **not** an algorithm attributed to the
reviewed 2014 supplements or 2026 preprint. The
[two-path fitting experiment](cpu_signal_simulation.md#experimental-two-path-waveform-fitting)
compares the same wall's IQ at two sampling rates, explicitly tests affine-
template nonidentifiability and preserves failed/noisy corrections. Strong-
signal high-rate improvement coexists with degraded errors at lower C/N0 and
false local certainty. Corrected estimates have not been integrated into the
tracker, and these results do not establish generally effective mitigation.
