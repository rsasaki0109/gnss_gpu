# Weak IQ to assisted code ranges and position

The new bridge connects fresh weak-tracker states to pseudoranges and a joint
Cartesian position/receiver-clock solution. A static eight-channel simulation
now reports end-to-end IQ-derived position error and unavailable epochs. It
also exposes the current precision bottleneck: integer-delay acquisition at
2.046 MHz leaves approximately 100 m median position error even without
unmodeled propagation effects. This is a coarse positioning baseline.

## Range assistance and epoch alignment

GPS L1 C/A repeats every millisecond, corresponding to **299,792.458 m** of
code-range. A measured lag alone cannot identify the absolute pseudorange.
`resolve_ca_range` enumerates period candidates inside a supplied coarse prior
interval and returns a range only if exactly one positive candidate remains.
Zero candidates give `outside_prior`; multiple candidates give `ambiguous`.
Neither case silently selects the nearest range. Numerical interval endpoints
receive 1 micrometre padding.

`position_from_weak_tracks` restores the known absolute receiver sample phase
to each window-local lag before resolving the period. It uses only fresh
`acquired`/`tracking` states; pending and coasting states do not produce ranges.
All included states must refer to the same window-start sample. At least four
resolved, nondegenerate ranges are needed for position and common clock.
Unavailable epochs retain their reason and rejected channels. Equal-weight
least squares performs no truth-based exclusion or outlier rejection.

This is **assisted** positioning: global code epoch alignment, coarse range
intervals and common-frame satellite coordinates are supplied. Navigation time,
ephemerides and receiver-clock bounds are not decoded from IQ. A wrong prior
can select a wrong period, and this API supplies no integrity guarantee.

In this experiment, the independent fixed prior position is `[0,0,0]` m and
prior clock is zero. Prior pseudoranges are distances from this prior to the
known synthetic satellites, with a ±6,000 m interval including clock uncertainty.
The simulated receiver is `[120,-80,20]` m with 1,500 m clock bias. These truth
values enter synthesis and offline scoring; they never choose an integer period
or enter the position solver's initial position. Satellites lie at different
21–22.4 Mm distances and different elevations/azimuths in a local Cartesian frame.

## Frozen receiver experiment

Eight isolated streams use actual PRNs 1–8, independent channel noise and unknown
carrier phase, zero Doppler, point-sampled rectangular C/A, and alternating
navigation bits. Each stream spans 80 ms and has four contiguous 20 ms windows.
The common start is offset by seven milliseconds plus 137 samples to test epoch
phase restoration. Each PRN has its own coherent carrier/delay/bit-edge search,
two-window confirmation and fresh-only position handoff. No other-PRN canceller
is needed in these isolated streams; this does not validate a composite receiver.

New per-PRN complete-pipeline gates are tolerance-calibrated on 599 independent
white-noise IQ records shared across PRNs. The 1% marginal upper false-alarm
budget uses a 5% joint calibration-failure allocation across eight gates. Shared
IQ creates correlated method scores; the family allocation does not require
independence between PRNs. The PRN 1 near-far gate from the earlier experiment
is **not** transferred to other PRNs. Calibration here covers white noise only.

After freezing, 199 separate white-null records per PRN produce passage counts
`[0,0,1,0,1,0,0,0]` for PRNs 1–8. These finite counts establish neither a 1%
position false-alarm rate nor a sequence-level false-track guarantee.

Sixteen new replicas compare five paired conditions with shared initial phases
and per-channel noise. Attenuation is represented by C/N0 55 vs 37 dB-Hz;
obstruction is represented by making five channels absent; excess propagation
is represented by delaying three channels by 300 m. A common +300 m code-clock
shift is a separate control. These are explicit sensitivity conditions, **not**
calibrated forest, rain or multipath-scattering models.

## Results and the precision limit

| Condition | Solved / scored epochs | 3D error median | 3D error p95 |
| --- | ---: | ---: | ---: |
| Strong, 55 dB-Hz | 48/48 | 98.9 m | 168.2 m |
| Weak, 37 dB-Hz | 48/48 | 102.3 m | 159.7 m |
| Only three weak channels | **0/48** | unavailable | unavailable |
| Three weak channels with +300 m excess path | 48/48 | **284.9 m** | 363.9 m |
| All weak channels with common +300 m clock shift | 48/48 | 148.6 m | 291.2 m |

The first confirmation window is unavailable in every replica. All **320**
epochs are retained, including these 80 pending epochs and the unavailable
three-channel epochs. The table separately scores the three post-confirmation
epochs per replica. The 48 rows are temporally correlated within 16 replicas;
they are not 48 independent trials and no superiority significance is claimed.

One sample corresponds to approximately **146.5 m** of code-range. The integer
detector's quantization dominates the open baseline. Thus the small difference
between strong and weak conditions is not evidence of better weak-signal
accuracy, and the common-clock control need not produce identical position:
the common delay also changes per-channel quantization/noise errors. Exact
range-domain unit controls separately verify that a common range shift goes
into the estimated clock. Detectability and precise ranging remain different
problems. Subsample range estimation is needed before small atmospheric or
wet-surface changes can be interpreted credibly.

![Position error and unavailable epochs](../output/cpu-weak-positioning/weak_positioning.png)

## API and reproduction

```python
from gnss_gpu.weak_positioning_cpu import position_from_weak_tracks

result = position_from_weak_tracks(
    states_by_prn, satellite_positions_m, range_priors_m, range_bounds_m,
    sampling_freq=2_046_000., initial_position_m=(0., 0., 0.),
)
if result.solution is not None:
    print(result.solution.position_m, result.solution.clock_bias_m)
```

Each mapping uses PRN keys. `states_by_prn` contains synchronized results from
`CpuWeakCaTracker`. Satellite positions use one common Cartesian frame in metres;
prior ranges and their bounds must include receiver-clock and unapplied model
correction uncertainty. Results refer to the observation window start and become
available only after window processing/confirmation; they are not current-time
extrapolated positions.

```bash
source /workspace/.gnss-env/activate.sh
python experiments/run_cpu_weak_positioning.py
python experiments/audit_cpu_weak_positioning.py
python experiments/plot_cpu_weak_positioning.py
```

Runs refuse a nonempty output directory; use `--output` for another run.
`output/cpu-weak-positioning` retains the protocol, source hashes, all per-PRN
calibration/evaluation scores, frozen gates, input IQ hashes, full states,
range ambiguity decisions, position/clock/residuals, failure reasons and figures.
The audit regenerates all eight gates and all 320 range/position outcomes,
checks source/artifact identities and preserves unavailable denominators. This
audit reuses saved states; it is not a second independent RF or IQ replay trial.

Unit checks cover wrapped and negative lags, unique/missing/ambiguous periods,
absolute sample-phase restoration, joint position/clock recovery, exclusion of
coasting and mixed-epoch rejection; 55 relevant tests pass. No orbit, Earth-rotation, ionosphere,
troposphere, real RF, precise positioning or research novelty claim follows
from this controlled baseline.
