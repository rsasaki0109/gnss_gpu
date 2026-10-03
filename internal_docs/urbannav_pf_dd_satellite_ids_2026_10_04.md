# UrbanNav PF: the DD terms had never run (2026-10-04)

## Finding

The PF smoother's base-relative updates (DD pseudorange, widelane, DD carrier)
pair rover and base rows by `(system_id, prn)`. `libgnsspp` builds the rover
rows with `preprocess_spp_file`, and its `CorrectedMeasurement` never exposed
`prn`. `getattr(m, "prn", 0)` returned 0 for every row, so all rows of a
constellation collapsed into one "satellite", fewer than 4 common satellites
remained, and every DD update returned `None` without a warning:

```
[mupf_dd] DD-AFV used 0/12200 epochs, skip 12200/12200
[dd_pseudorange] used 0/12200 epochs, skip 12200/12200
```

Every UrbanNav PF number recorded since at least 2026-10-03 (the resampler
ablation, the RTK baselines, the README row) is a PF without its DD terms. It
also explains why the calibrated base coordinate did not move the PF output.

## Fix

- gnssplusplus-library #557 (develop `304798e7`) exposes `prn` and
  `satellite_id` on `CorrectedMeasurement` from its existing `identity` field.
- gnss_gpu pins `third_party/gnssplusplus` to `304798e7`. The pin also brings
  #556 (`gnss_live` / `gnss_replay` read `--base-ecef X Y Z` in order on MSVC).
- `require_measurement_satellite_ids` makes a DD-enabled run fail fast when the
  rows have no `prn`, instead of silently running without DD.

The PPC canonical reproduction still uses `62bd0b73`, as its record states.

## Results

Preset `odaiba_stop_detect`, calibrated base (`--base-ecef`,
`configs/urbannav/tokyo_base_cref0001.json`), libgnsspp from develop
`304798e7`, seeds 42 and 101–104 (mean ± sd). Rates count every reference
epoch; epochs without output fail. "Gap fill" uses libgnss++ `low-cost` RTK
(calibrated base) where it outputs and the PF elsewhere.

With `prn`, DD-AFV runs on 10,897 of 12,200 Odaiba epochs and 18,102 of
20,056 Shinjuku epochs; DD pseudorange runs on the 1 Hz base epochs (1,205 and
1,971). The pinned `62bd0b73` with the same binding patch gives identical
seed-42 results.

| Odaiba | SMTH P50 | SMTH RMS | <0.5 m | <1 m | <3 m | <5 m |
|---|---:|---:|---:|---:|---:|---:|
| PF, DD dead (seed 42, before) | 2.25 | 13.56 | 3.5% | 13.5% | 63.3% | 81.5% |
| PF, DD running (5 seeds) | 1.50 ± 0.08 | 4.06 ± 0.05 | 9.0% | 29.6% | 73.1% | 86.7% |
| libgnss++ RTK + PF gap fill (5 seeds) | — | 3.55 | 62.4% | 68.5% | 79.7% | 87.8% |
| RTKLIB demo5 (calibrated base) | 0.34 | 40.89 | 50.5% | 56.8% | 69.0% | 86.6% |

| Shinjuku | SMTH P50 | SMTH RMS | <0.5 m | <1 m | <3 m | <5 m |
|---|---:|---:|---:|---:|---:|---:|
| PF, DD dead (seed 42, before) | 5.97 | 13.02 | 0.6% | 2.2% | 17.5% | 38.3% |
| PF, DD running (5 seeds) | 1.72 ± 0.08 | 5.92 ± 0.03 | 7.0% | 23.1% | 71.8% | 81.9% |
| libgnss++ RTK + PF gap fill (5 seeds) | — | 5.89 | 59.7% | 68.8% | 81.3% | 84.2% |
| RTKLIB demo5 (calibrated base) | 1.32 | 11.75 | 41.0% | 44.4% | 63.4% | 73.5% |

libgnss++ `low-cost` RTK alone (calibrated base) reaches 74.5% (Odaiba) and
76.4% (Shinjuku) within 5 m; the PF gap fill adds 13.3 and 7.8 points.
Seed-to-seed spread of the gap-fill rates is 0.6 points or less.

- The April 2026 headline for this preset (1.36 m / 4.11 m on 12,228 epochs)
  is now within reach: RMS matches, and P50 is 0.14 m higher on today's
  12,184-epoch data. The non-reproducibility recorded on 2026-10-03 was this
  bug, not the data version alone.
- libgnss++ RTK + PF gap fill now beats RTKLIB demo5 at every threshold on both
  routes.
