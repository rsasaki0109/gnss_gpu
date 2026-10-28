# Public RINEX → CPU positioning → RTKLIB comparison

Executed 2026-10-24. This checks real-observation integration on one public
development recording, not weather effects or true positioning accuracy. It
follows the [wet-recovery research protocol](research_wet_recovery_protocol.md).

## Input and comparator

RTKLIB's public test data include GSI Japan station **0759**, Trimble 5700,
GPS RINEX 2.10 observations on **2005-04-02**, 00:00:00–00:59:30 GPST:
120 epochs at nominal 30 s cadence with L1/C1/L2/P2 observations. Timestamps
include small receiver-clock offsets. There is no C/N0 observation code,
weather label, independent reference trajectory or tracking state.

Observation, navigation and upstream license notice are pinned to commit
`71db0ffa0d9735697c6adfd06fdf766d0e5ce807` in the
[source manifest](public_rinex_baseline_sources.json). Downloads are checked
against fixed byte counts and SHA-256 before writing. Third-party payloads and
binaries remain under local `output/` / research directories, outside versioned
source. RTKLIB's notice describes BSD terms plus additional clauses; original
data-provider redistribution terms were not independently established. Retain
the downloaded notices.

RTKLIB **2.4.2** was built from the same commit with its supplied GCC makefile.
The comparator is standalone broadcast GPS L1, with a 15° elevation mask,
broadcast ionosphere, Saastamoinen troposphere and explicit GDOP limit 30.
Configuration and binary hashes are saved before evaluation.

## CPU pipeline

[The experiment](../experiments/run_public_rinex_baseline.py) uses existing CPU
parsers, broadcast ephemeris and least-squares solvers. It retains actual GPS C1
measurements, healthy navigation with TOE age <=2 h, and satellites above 15°.
It computes transmit-time satellite position, clock/TGD, one Sagnac rotation,
Klobuchar ionosphere and Saastamoinen troposphere. It solves position and clock
using equal weights and, separately, relative sigma `1/sin(elevation)`.
The latter is an engineering weight, not calibrated absolute uncertainty.
Missing C/N0 or ranges are not synthesized.

The header's approximate coordinate initializes geometry/atmosphere; it is not
independent truth. Observation/navigation satellite IDs both use strings such
as `G03`. RINEX receiver time is converted to solution GPST by removing the
estimated receiver clock, then quantized to RTKLIB's configured 1 ms output
precision. Matches are exact and unique after this clock conversion; no nearest
sample, interpolation or fitted time shift is used. RTKLIB's `pntpos.c` also
applies this solution-time correction.

## Results

| Method | Positions / input epochs | Paired with RTKLIB | RMS 3-D difference from RTKLIB | Maximum difference |
| --- | ---: | ---: | ---: | ---: |
| CPU equal-weight LS | 120 / 120 | 115 | 0.5853 m | 1.0214 m |
| CPU elevation-weighted LS | 120 / 120 | 115 | 0.8923 m | 2.2043 m |
| RTKLIB standalone | 115 / 120 | — | — | — |

These are **algorithm agreement measures, not true errors**. Smaller difference
does not establish that equal weighting is more accurate. RTKLIB adds its own
variance/screening models; these are not identical implementations of one fit.

RTKLIB rejects the final five epochs (00:57:30–00:59:30). Its diagnostic trace
explicitly reports GDOP 31.7, 34.9, 38.5, 42.8 and 47.5, exceeding 30. The
ungated CPU solver still returns positions using five satellites, with small
postfit residual RMS (0.41–0.50 m). Small residuals alone do not establish good
positioning geometry. Extra CPU outputs are not an availability improvement
without an independent error/risk assessment.

There is no confirmed weather attribution or recovery event. The 120 epochs
are correlated samples from one development recording, not independent trials.

## Reproduction

Install CPU dependencies plus `dev` (requests) and `viz` (matplotlib) extras.
From the repository with `PYTHONPATH=python:.`:

```bash
git clone https://github.com/tomojitakasu/RTKLIB.git /workspace/rtklib-reference
git -C /workspace/rtklib-reference checkout 71db0ffa0d9735697c6adfd06fdf766d0e5ce807
make -C /workspace/rtklib-reference/app/rnx2rtkp/gcc -j4
python experiments/run_public_rinex_baseline.py --fetch \
  --rtklib-binary /workspace/rtklib-reference/app/rnx2rtkp/gcc/rnx2rtkp
```

`output/public-rinex-baseline/` contains all-epoch `positions.csv`, comparison
figure `public_rinex_baseline.png`, RTKLIB solution/configuration/diagnostic
trace and provenance. Wall times cover different scopes and do not support a
speed comparison. Focused tests cover clock sign, timestamp rollover, nonfinite
clock rejection and duplicate epochs. Together with existing transmission-time,
ephemeris-wrapper, position and weighted-position checks: **24 tests pass**.

This establishes a working public-observation CPU comparison. Weather-specific
validation still needs suitable labelled observations and an independently
assessed reference. [Boreas receiver logs have now been acquired and imported](boreas_gnss_import.md)
after the network update. This non-weather control recording does not replace
weather-specific validation.
