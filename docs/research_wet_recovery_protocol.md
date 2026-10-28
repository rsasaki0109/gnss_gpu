# Research target: wet-transition recovery from signal to position

Status: exploratory protocol and executed synthetic diagnostic, 2026-10-24.
No superiority over published methods or first-of-its-kind claim is established.
The available hardware is CPU only; no physical receiver is required for the
public-data stage.

## Contribution to test

Test whether a receiver-state-aware recovery policy can restore position
availability after signal degradation **without increasing large-error output**,
compared with fixed policies and an independently reproduced published method.
The intended deliverable is a reproducible signal/receiver/position benchmark,
an explicit recovery intervention, and evaluation on unseen recordings.

An attenuation plot alone does not establish positioning performance. Separate
three mechanisms: propagation/antenna response, acquisition/code tracking, and
the decision to publish a position. Rain, wet foliage, wet ground and a wet
antenna/radome are distinct model inputs. Existing antenna coefficients and the
0.5 s schedule are illustrative sensitivity inputs, not rainfall calibration.

Candidate hypotheses, to be frozen before confirmatory evaluation:

1. At equal current observations, receiver/publication state can change output
   availability. A paired suffix intervention can identify which state causes it.
2. A policy using only past and current receiver observables can improve
   sustained recovery availability over fixed policies, subject to a predeclared
   large-error constraint. It must not use synthetic truth or wetness labels.
3. Effects remain on disjoint real recording episodes. Weather association in
   public road data is an external check, not proof of a wet-antenna mechanism.

## What the literature review currently supports

| Source | Material actually reviewed | Consequence for this project |
| --- | --- | --- |
| Nievinski & Larson (2014), [forward model](https://doi.org/10.1007/s10291-013-0331-y) and [MPSim](https://doi.org/10.1007/s10291-014-0370-z) | Publisher abstracts/metadata and pinned open source supplement; subscription article full texts unavailable | Coherent/polarized multipath is existing work. Extending its evaluation to recovery is a candidate question, not evidence that these papers omit it. |
| Iudice et al. (2024), [receiver impact simulator](https://arxiv.org/abs/2405.17925) | Accepted-author full text, especially observation-domain effects | Observation-to-navigation evaluation is existing work. Our IQ diagnostic cannot be called a novel end-to-end simulator on that basis. |
| Sanwale et al. (2026), [digital twin](https://arxiv.org/abs/2605.01553) | Preprint methods and validation sections | Physics-driven receiver evaluation already exists. This is a preprint; no reproduction of its full HIL pipeline has been performed. |
| [Tracking Canopy Water Content with GNSS-Derived VOD (2026)](https://doi.org/10.5194/egusphere-egu26-9965) | Crossref abstract only, conference abstract | GNSS attenuation and canopy water are already an active measurement topic. This is not a reviewed full journal study or a positioning baseline. |

Detailed review boundaries and hashes are in
[the existing source register](gnss_signal_simulation_sources.json).
The additional Crossref scoping responses are saved in
`output/research-scoping/`. Broad queries returned irrelevant hits and one
rate-limit response. This is not a systematic or exhaustive novelty search.
Before a submission, review wet-radome/foliage measurements, fading-channel
tracking, reacquisition, and correlated-error integrity methods, then select and
reproduce the closest published comparator. That comparator remains unresolved.

## Executed pilot: distinguish tracking from publication state

Run from the repository with the configured CPU environment:

```bash
PYTHONPATH=python:. python experiments/audit_wet_recovery_history.py
```

This is a **post-hoc development diagnostic**. All seeds 24683–24690 were
previously inspected. Earlier experiments' historical holdout labels do not
make these data a new holdout for this question.

The native-pair audit compares dry and three antenna stress scenarios across
eight seeds and two environments: 48 episode pairs, 4,800 jointly available
position epochs in blocks 400–499 (400–500 ms), when wetness is zero.

* The maximum paired raw 3-D position difference is **0.069875 m** over those
  comparisons. This is a difference between estimates, not absolute error.
  These data do not support a large persistent tracking-error claim.
* 36 sample-level checks compare dry and stressed IQ, including paired noise,
  at blocks 400/450/499, PRNs 1/12, open/wall, seed 24686. All are exactly
  equal. This is a spot-check, not an exhaustive IQ comparison.
* For 12 wet episodes from seeds 24685/24686, replay the frozen publication
  gate with its native wet prefix, then replace all channels from block 400
  onward with the **same saved dry channel rows**. Compare with a dry-prefix
  replay. Candidate raw positions are exactly identical throughout the shared
  suffix, including availability; the gate histories remain different.
* In **one of these 12 episodes** (24686, wall, 6 dB loss), the dry-prefix gate
  publishes **100/100** suffix positions, while the wet-prefix gate publishes
  **0/100**, even with the identical dry suffix. The other 11 episode pairs
  publish 100/100 in both interventions.

The intervention isolates a failure of the current engineering publication
policy from the current channel measurements. It does not demonstrate that
real rain causes such a failure, that receiver memory is newly discovered, or
that a replacement method beats published work. The episode frequency is not
a population failure probability; the 1 ms epochs are correlated.

Generated artifacts in `output/research-recovery-history/`:

* `raw_tail_pairs.csv`, `gate_state_intervention.csv`,
  `iq_suffix_spot_checks.csv`: complete episode and check counts.
* `recovery_history.png`: comparison figure.
* `provenance.json`: exact input/source hashes, frozen policy and limitations.

## Public data: acquisition stage

The user has no physical receiver. Public data are the intended empirical path.
The [source register](public_gnss_weather_data.json) records inspected sources,
versions and access limits.

| Candidate | Verified useful content | Missing evidence / role |
| --- | --- | --- |
| [Oxford RobotCar SDK](https://github.com/ori-mrg/robotcar-dataset-sdk) | Official metadata for 134 traversals: 9 `rain`, 62 `sun`; SDK support for INS/RTK pose files | GNSS payload not acquired; raw observations, usable standalone solution and independent reference require inspection. Rain tags do not measure intensity or antenna wetness. First weather-associated screening candidate. |
| [Boreas](https://github.com/utiasASRL/pyboreas) | Repeated multiweather routes; documented postprocessed Applanix trajectory at 200 Hz; data CC BY 4.0 | Documented trajectory is a GNSS/INS batch solution, not an independent GNSS-only test. No raw IQ/RINEX was verified. Useful scene/reference context; not a primary GNSS-error validation dataset yet. |
| [UrbanNav](https://github.com/IPNL-POLYU/UrbanNavDataset) | Documented raw RINEX and separate SPAN/POS LV reference; urban obstruction/recovery sequences | No verified rain/wet-forest labels; reference is also GNSS/INS and has limitations. Useful receiver-observation baseline, not a rain-mechanism test. Existing browser use of its geometry does not validate weather. |

Official RobotCar metadata were cloned and inventoried at commit
`16ce3329223ca418fe5106277b91aea8d9b672b2`. Reproduce with:

```bash
python experiments/catalog_robotcar_weather.py --sdk /path/to/robotcar-dataset-sdk
```

`output/public-gnss-data/robotcar_weather_inventory.csv` includes every traversal,
source permalink and original tags. The companion provenance file saves all
134 tag-file hashes and the nine rain sequence IDs. Do not exclude `poor-gps`
or `no-gps` traversals: that would select on the outcome of interest. Do not
classify a traversal as dry merely because it lacks a rain tag. The 62 `sun`
labels are a screening pool, not 62 verified dry-antenna controls.

After publication of runtime revision 9, the RobotCar website and Boreas S3
listing both return HTTP 200. [Two Boreas native receiver chunks and reference poses](boreas_gnss_import.md)
have been acquired; 631 checksum-verified GGA records were extracted. Their
weather label and the positioning reference conventions remain unverified.
RobotCar's GPS archive links use a newly discovered host,
`mrgdatashare.robots.ox.ac.uk`, now present in runtime revision 10. The payload
endpoint still returns 503. The official downloads page announces withdrawal
of the dataset until further notice, dated **2026-10-22**. Oxford payload
acquisition is paused; no additional environment publication is requested.
No rainy trajectory result or calibrated weather effect is claimed. A separate
[public non-weather RINEX control](public_rinex_baseline.md) has now been
processed: 120 epochs, with 115 paired RTKLIB solutions. This checks the CPU
observation pipeline and does not validate weather effects.
The Boreas SDK's example `gps_post_process.csv` is empty;
it must not be treated as a real sample recording. No wet-forest dataset with
all required observation/weather/reference fields has yet been verified.

## Confirmatory evaluation design

Before collecting results, version the dataset manifests, timestamp/coordinate
conversions, receiver settings, policy and comparison methods. Required steps:

1. Verify actual payload headers, sample rate, time system, observation loss,
   standalone GNSS solution and reference construction/quality. Check antenna
   lever arms and exclude only predeclared reference-invalid intervals; retain
   receiver failures in the availability denominator.
2. For road weather comparisons, match route segment and direction, account for
   satellite geometry, season, speed and hardware. Different-day weather
   comparisons remain observational. Road results cannot establish forest or
   radome wetting calibration. Ground/foliage/antenna effects need separate
   measured constraints or sensitivity ranges.
3. Split by recording day/route episode, not adjacent frames. Freeze development,
   validation and final test IDs before tuning. All existing synthetic seeds
   are development only. New noise seeds alone cannot establish transfer to
   new geometry, receiver hardware, or real weather.
4. Use identical input observations for equal-weight LS, causal weighted LS,
   the frozen gate and the proposed intervention. Reproduce a relevant
   published baseline before making a superiority claim. Also evaluate with a
   conventional acquisition/DLL/PLL receiver: today's calibrated code follower
   and isolated PRN channels do not reproduce a complete receiver front end.
5. Report all-input availability, conditional error distributions and
   error-over-threshold counts together. Report first sustained recovery,
   subsequent outages, and censoring when the recording ends. Fix position
   error thresholds, allowable risk margin and recovery continuity interval
   using application needs and data cadence **before** final-test evaluation.
   The existing 100 m / five 1 ms positions are diagnostic settings, not
   operational safety or weather-recovery requirements.
6. For uncertainty, report empirical coverage and bound size; the previous
   nominal-95% covariance result was about 93.2% covered, so calibration is not
   established. Use paired episode/day-level uncertainty intervals, not a
   binomial interval over thousands of correlated epochs. With only nine
   rain-labelled traversals, rare-event risk claims will be limited.
7. Record wall time, sample throughput, peak memory, CPU/thread budget and
   warmup costs. Parallel runs must preserve seed-based pairing and numerical
   results. CPU speed is a separate contribution requiring comparable accuracy
   and workloads; absence of a GPU is not a novelty claim.

Success means a predeclared improvement on unseen episodes with the error-risk
constraint satisfied, plus reproducible comparisons and documented model
validity. Until those tests and the closest-literature review are complete,
the defensible result is the state-isolation diagnostic and dataset inventory.
