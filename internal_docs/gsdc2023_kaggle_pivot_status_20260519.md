# GSDC2023 Kaggle Pivot Status - 2026-05-19

## Active Target

The active GSDC2023 target is now sub-meter-class accuracy.

Working definition:

- Primary local score proxy: `0.5 * (P50 + P95)` horizontal error.
- Sub-meter gate: score proxy `< 1.0 m` on credible train/held-out trip sets.
- Submission gate: coordinate-sane output, full-train aggregate improvement,
  and no obvious phone/run regression pocket large enough to threaten private
  score.

Implication: code-only WLS/robust-WLS and CT motion-prior variants are not
enough.  The main path must use ADR/carrier-phase information: TDCP, cycle-slip
handling, DD carrier/LAMBDA, and local FGO/RBPF.

## Data And Reproduction State

- Raw Kaggle data is available at `/tmp/gsdc_data/gsdc2023/sdc2023`.
- Exact historical reproduction is not ready because the two required reference submissions are missing:
  - `../ref/gsdc2023/results/test_parallel/20260421_0555/submission_20260421_0555.csv`
  - `../ref/gsdc2023/results/test_parallel/20260423_1450/submission_20260423_1450.csv`
- Current readiness snapshot: `experiments/results/gsdc2023_pivot_readiness_20260519.json`.

## Submission History Recovery

- `kaggle competitions submissions -c smartphone-decimeter-2023 --page-size 200 -v` works and was saved to `experiments/results/gsdc2023_kaggle_submissions_20260519.csv`.
- The installed Kaggle CLI/SDK exposes submission listing and metadata (`competition_submissions`, `GetSubmission`) but no submission-body download command.
- The historical raw submission URL returned by Kaggle metadata requires a website-authenticated session; API-key authentication was not sufficient in local checks.

## Local Candidate Screen

Local `gsdc2023_submission*.csv` files were screened into:

- `experiments/results/gsdc2023_local_submission_screen_20260519.csv`
- `experiments/results/gsdc2023_safe_unsubmitted_shortlist_20260519.csv`

Result:

- 7 local files matched the submission shape.
- `gsdc2023_submission_v2.csv` is rejected by coordinate sanity (`1137` out-of-bounds rows; includes a large lat/lon excursion).
- 6 files remain as structurally safe, unsubmitted, manual-review candidates:
  - `gsdc2023_submission.csv`
  - `gsdc2023_submission_v12.csv`
  - `gsdc2023_submission_v13.csv`
  - `gsdc2023_submission_v15.csv`
  - `gsdc2023_submission_v22.csv`
  - `gsdc2023_submission_v3.csv`

These files are old PF-family candidates and have no current positive score signal. Treat a Kaggle submit from this list as exploratory scoring, not as a likely private-floor improvement.

Historical clues from git:

- `gsdc2023_submission.csv` was submitted as `4.207` public / `5.144` private.
- `gsdc2023_submission_v2.csv` is a known negative (`10.15 m`) and now also fails coordinate sanity.
- `gsdc2023_submission_v3.csv` is a known negative (`30.76 m`).
- `gsdc2023_submission_v12.csv`, `v13`, `v15`, and `v22` are retained local PF-family artifacts, but no positive submitted score was found locally.

## Robust WLS Follow-Up

Full-train robust WLS evaluation was rerun:

```bash
PYTHONPATH=.:python python3 experiments/exp_gsdc2023_robust_wls.py \
  --output experiments/results/gsdc2023_robust_wls_eval_20260519_full.csv \
  --no-test-submission-on-win
```

Result:

- evaluated run/phone pairs: `156`
- P50 wins: `85/156` (`54.5%`)
- mean P50: WLS `81.80 m`, robust WLS `81.80 m`
- median P50: WLS `2.42 m`, robust WLS `2.42 m`
- mean delta: P50 `-0.00 m`, RMS `-0.00 m`
- no test submission was generated because the configured threshold was not met.

Conclusion: current robust WLS is reproducible but not worth spending a Kaggle submission.

## CT-RBPF/FGO Bridge Hook

The GSDC2023 raw bridge now has an opt-in `fgo_ct_rbpf` source hook:

```bash
PYTHONPATH=.:python python3 experiments/validate_fgo_gsdc2023_raw.py \
  --data-root /tmp/gsdc_data/gsdc2023/sdc2023 \
  --trip train/2023-05-25-19-10-us-ca-sjc-be2/pixel5 \
  --position-source gated \
  --chunk-epochs 15 \
  --ct-rbpf-fgo \
  --ct-rbpf-motion-sigma-m 0.2
```

Use `--position-source gated --ct-rbpf-fgo` to add the candidate to source selection, or
`--position-source fgo_ct_rbpf --ct-rbpf-fgo` to export it directly.

Smoke status:

- `experiments/results/gsdc2023_ct_rbpf_fgo_smoke_20260519/bridge_metrics.json` contains `ct_rbpf_fgo_enabled=true` and `fgo_ct_rbpf` chunk candidates.
- Direct source smoke on 15 epochs completed with `source mix: fgo_ct_rbpf=15`.
- A bounded train sweep is available at `experiments/results/gsdc2023_ct_rbpf_fgo_eval_20260519_limit12.csv`.
  - trips: `12`
  - CT rows: `36` (`sigma=0.1,0.2,0.5`)
  - local score-proxy wins: `1/36`
  - PR-MSE wins: `0/36`
  - selected CT epochs: `40`, all from `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl` with `sigma=0.1`
  - best observed delta: `-0.445 m` selected score proxy versus base gated
- A full-train 200-epoch screen is available at `experiments/results/gsdc2023_ct_rbpf_fgo_eval_20260519_all200_sigma01.csv`.
  - trips: `156`, CT rows: `156` (`sigma=0.1`)
  - local score-proxy wins: `2/156`
  - PR-MSE wins: `3/156`
  - selected CT epochs: `300`
  - score-proxy delta mean: `-0.0017 m`, min: `-0.312 m`, max regression: `+0.236 m`
  - selected CT rows: `7/156`, concentrated in `mi8` (`180` epochs) plus `pixel4/pixel4xl/pixel5` (`40` each)
  - best score-proxy wins:
    - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl`: `-0.312 m`, CT epochs `40`
    - `train/2022-02-24-18-29-us-ca-lax-o/mi8`: `-0.309 m`, CT epochs `40`

Current limitation: this is the Kaggle-side adapter and source-selection path. It reuses the existing GSDC FGO/VD solver with a CT motion-prior override; the full PPC-style RBPF particle emitter is not yet ported into the GSDC observation pipeline.

## Carrier/Base Readiness Audit

Full-train carrier usability is now measured at 200 epochs per trip:

- Output: `experiments/results/gsdc2023_carrier_phase_usability_20260519_all200.csv`
- trips: `156`
- mean valid ADR observations per epoch: `14.82`
- mean continuous valid ADR pairs per interval: `13.06`
- mean TDCP pairs per interval: `13.00`
- median valid ADR ratio: `0.714`
- trips with at least `8` TDCP pairs per interval: `141/156`
- strongest phones by valid ADR density: `pixel7pro`, `pixel7`, `pixel6pro`,
  `pixel5`, `sm-s908b`

Base/DD readiness was also materialized locally:

- Added `experiments/audit_gsdc2023_base_readiness.py`.
- Rebuilt local `settings_train.csv` / `settings_test.csv`.
- Applied the existing SLAC/VDCY Base1 heuristic:
  - train: `SLAC=133`, `VDCY=23`
  - test: `SLAC=34`, `VDCY=6`
- Generated local approximate base metadata:
  `/tmp/gsdc_data/gsdc2023/base/base_position.csv` and `base_offset.csv`.
- Downloaded train BRDC files for `65` courses.
- Downloaded/decompressed train base RINEX obs for `42` unique base-days.
- Readiness output:
  `experiments/results/gsdc2023_base_readiness_20260519_train_ready.csv`.
- Result: `156/156` train trips are now `base_correction_ready`.

DD carrier support against the downloaded public daily CORS files:

- Added `experiments/audit_gsdc2023_dd_carrier_support.py`.
- Smoke output:
  `experiments/results/gsdc2023_dd_carrier_support_20260519_smoke.csv`.
- Public daily CORS base cadence is `30.0 s`.
- First 5 train trips, 200 epochs each, with `0.6 s` nearest-base snap:
  - mean base snap coverage: `0.067`
  - mean DD coverage: `0.065`
  - mean DD pairs when available: `5.87`

Read: rover carrier continuity is good, but public daily CORS is too sparse for
a full-rate DD carrier source.  The practical sub-meter path should either
fetch high-rate/1 Hz base observations or use sparse DD carrier anchors with
1 Hz TDCP/FGO motion.

Sparse DD-carrier anchor probe:

- Added `experiments/eval_gsdc2023_dd_carrier_anchor.py`.
- Default smoke output:
  `experiments/results/gsdc2023_dd_carrier_anchor_eval_20260519_smoke.csv`.
  - first 5 train trips, 200 epochs each
  - accepted anchor epochs: `53`
  - score wins: `1/5`
  - mean score delta: `+0.0020 m`
- Tight `min_dd_pairs=7` smoke output:
  `experiments/results/gsdc2023_dd_carrier_anchor_eval_20260519_smoke_min7.csv`.
  - accepted anchor epochs: `22`
  - score wins: `0/5`
  - mean score delta: `0.0000 m`
- Smoothed correction output:
  `experiments/results/gsdc2023_dd_carrier_anchor_eval_20260519_smoke_smooth.csv`.
  - score wins: `1/5`
  - mean score delta: `+0.0038 m`
- Smoothed `min_dd_pairs=7` output:
  `experiments/results/gsdc2023_dd_carrier_anchor_eval_20260519_smoke_min7_smooth.csv`.
  - score wins: `1/5`
  - mean score delta: `+0.0016 m`

Read: fixed-ambiguity sparse DD anchors are plumbing/diagnostic value, not a
submission candidate.  Smoothing the sparse corrections over neighboring epochs
also does not improve the 5-trip smoke.  The next meaningful accuracy step needs
high-rate/1 Hz base observations or a native joint factor implementation that
can use carrier constraints without spreading bad sparse corrections.

High-rate base downloader status:

- Added `experiments/fetch_gsdc2023_highrate_base_obs.py`.
- It enumerates CDDIS high-rate 1 Hz 15-minute slots from actual GSDC course
  UTC spans, so it does not need to fetch a whole base-day.
- It now also discovers RINEX3 long-name 15-minute files from public directory
  indexes, e.g. GFZ ISDC names like
  `SITE00CCC_R_YYYYDDDHHMM_15M_01S_MO.crx.gz`.
- Dry-run example:

```bash
PYTHONPATH=.:python python3 experiments/fetch_gsdc2023_highrate_base_obs.py \
  --data-root /tmp/gsdc_data/gsdc2023/sdc2023 \
  --splits train \
  --course 2023-05-25-19-10-us-ca-sjc-be2 \
  --dry-run --limit 1
```

Dry-run output for that SJC course:

- `slots=3`
- `slac145t00.23d.gz`
- `slac145t15.23d.gz`
- `slac145t30.23d.gz`

GFZ ISDC public high-rate index check:

- `https://isdc-data.gfz.de/gnss/data/highrate/2023/145/` is reachable.
- The checked SJC/SLAC window did not expose `SLAC` or `VDCY` RINEX3
  high-rate files in that index.
- The compact CDDIS-style GFZ fallback path returned 404 for
  `slac145t15.23d.gz`.

Actual fetch reached CDDIS/Earthdata but received the Earthdata login HTML. The
script now detects this and fails with an explicit auth message.  Before this
route can produce `SLAC_1hz.obs` / `VDCY_1hz.obs`, configure `~/.netrc` for
`urs.earthdata.nasa.gov` or set `EARTHDATA_USERNAME` and
`EARTHDATA_PASSWORD`.

DD carrier audit/eval can now be pointed at a high-rate course file as soon as
one is available:

```bash
PYTHONPATH=.:python python3 experiments/audit_gsdc2023_dd_carrier_support.py \
  --data-root /tmp/gsdc_data/gsdc2023/sdc2023 \
  --base-obs-template '{base}_1hz.obs' \
  --require-base-obs-template
```

Bridge candidate integration:

- Added source `fgo_dd_carrier`.
- Enable it with `BridgeConfig(dd_carrier_fgo_enabled=True)` or CLI
  `experiments/validate_fgo_gsdc2023_raw.py --dd-carrier-fgo`.
- Direct source:
  `--position-source fgo_dd_carrier --dd-carrier-fgo`.
- Gated/auto source selection sees it as a chunk candidate and records
  `dd_carrier_accepted_anchor_epochs`, `dd_carrier_dd_epochs`,
  `dd_carrier_base_snapped_epochs`, and `dd_carrier_dd_pairs_mean` in
  `bridge_metrics.json`.
- Smoke on `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`, 40 epochs,
  daily 30 s base:
  - `anchors=3`, `dd_epochs=3`, `snapped=3`, `pairs_mean=5.33`
  - direct `fgo_dd_carrier` RMS2D `1.573 m`
  - plain FGO RMS2D `1.577 m`
- Added `experiments/eval_gsdc2023_dd_carrier_fgo.py`.
  - Smoke output:
    `experiments/results/gsdc2023_dd_carrier_fgo_eval_20260519_smoke.csv`
  - first 2 train trips, 80 epochs each, gated mode:
    score wins `0/2`, selected DD epochs `0`, accepted anchors `9`

TDCP/FGO sweep status:

- Added `experiments/eval_gsdc2023_tdcp_fgo_sweep.py`.
- Gated smoke:
  `experiments/results/gsdc2023_tdcp_fgo_sweep_20260520_smoke.csv`
  - first 2 train trips, 80 epochs, 6 variants
  - selected output stayed baseline for all variants
  - score wins versus default: `0/12`
- Direct-FGO smoke:
  `experiments/results/gsdc2023_tdcp_fgo_sweep_20260520_direct_fgo_smoke.csv`
  - best 2-trip mean: `tdcp_weight_scale=1e-7`,
    `tdcp_geometry_correction=False`
  - mean score: `2.9116 m` versus default `2.9666 m`
  - mixed by trip, so this is only a sweep lead, not a default/submission
    setting
- Broader direct-FGO small grid:
  `experiments/results/gsdc2023_tdcp_fgo_sweep_20260520_limit12_e80_direct_fgo_smallgrid.csv`
  - 12 train trips, 80 epochs, 6 variants
  - `tdcp_weight_scale=1e-7`, geometry correction on:
    mean delta `-0.0010 m`, wins `6/12`, worst regression `+0.0196 m`
  - `tdcp_weight_scale=1e-6`, geometry correction on:
    mean delta `+0.0144 m`
  - geometry correction off is unstable:
    `1e-7` off mean delta `+0.3068 m`, and `1e-6` off mean delta
    `+12.2350 m`
  - read: keep TDCP geometry correction on globally.  `1e-7` scale is the
    only safe-looking lead, but the gain is too small to change defaults yet.
- Targeted `tdcp_weight_scale=1e-7`, geometry-on follow-ups:
  - `experiments/results/gsdc2023_tdcp_fgo_sweep_20260520_trips13_36_e80_direct_fgo_scale1e7.csv`
  - `experiments/results/gsdc2023_tdcp_fgo_sweep_20260520_trips61_96_e80_direct_fgo_scale1e7.csv`
  - `experiments/results/gsdc2023_tdcp_fgo_sweep_20260520_trips97_132_e80_direct_fgo_scale1e7.csv`
  - combined with the initial 12-trip smoke: 108 train trips, 80 epochs each
  - overall: wins `44`, losses `51`, ties `13`
  - median delta `0.0 m`; mean is dominated by a catastrophic `sm-g988b`
    regression (`+1581.64 m`)
  - phone read:
    - `pixel4`: mean `-0.0040 m`, wins `7/11`, worst `+0.0139 m`
    - `mi8`: mean `-0.0048 m`, wins `8/16`, worst `+0.0380 m`
    - `pixel4xl`: mean `-0.0029 m`, wins `3/8`, worst `+0.0142 m`
    - `pixel5`: mean `+0.0015 m`, wins `10/29`, worst `+0.0523 m`
    - `pixel7pro`: mean `+0.0046 m`, wins `4/10`, worst `+0.0497 m`
    - `pixel6pro`: no effect in these runs
    - Samsung/A-series are not safe for this scale; `sm-g988b` and `sm-a226b`
      are explicit risk phones
  - read: `1e-7` is not a global default.  It is at most a tiny
    phone-limited candidate for `pixel4`, `pixel4xl`, and maybe `mi8`, and
    needs gated-selector validation before any submission use.

Phone-limited TDCP-scale candidate:

- Added bridge source candidate `fgo_tdcp_scale`.
- Enable with `BridgeConfig(tdcp_scale_candidate_enabled=True)` or CLI
  `--tdcp-scale-candidate`.
- Default candidate:
  - `tdcp_scale_candidate_weight_scale=1e-7`
  - `tdcp_scale_candidate_phones=("pixel4", "pixel4xl", "mi8")`
- The default selected/FGO path remains unchanged; this is only an additional
  gated candidate.
- Smoke checks:
  - `pixel4` produced `fgo_tdcp_scale` candidate metadata.
  - `pixel5` did not produce the candidate, as intended.
- Gated allowed-phone smoke:
  `experiments/results/gsdc2023_tdcp_scale_candidate_gated_20260520_allowed24_e80.csv`
  - 24 allowed-phone trips, 80 epochs
  - selected `fgo_tdcp_scale` epochs: `0`
  - score wins vs default: `0/24`
- Read: plumbing is safe, but the standard gated policy does not choose this
  weak TDCP-scale candidate.  Do not submit as-is.

Selector diagnostic follow-up:

- `experiments/eval_gsdc2023_tdcp_fgo_sweep.py` now writes chunk-level guard
  diagnostics for FGO and `fgo_tdcp_scale`: candidate count, raw-WLS PR-MSE
  guard blocks, baseline-gap guard passes, quality-margin passes, and mean
  deltas versus baseline/FGO.
- Diagnostic rerun:
  `experiments/results/gsdc2023_tdcp_scale_candidate_gated_diag_20260520_allowed24_e80.csv`
  - selected epochs stayed `baseline=1760`, `fgo=80`, `fgo_no_tdcp=80`,
    `fgo_tdcp_scale=0`
  - FGO and `fgo_tdcp_scale` were both blocked by raw-WLS PR-MSE guard on
    `24/24` chunks
  - both passed the baseline-gap guard on `14/24` chunks and the quality-margin
    check on `22/24` chunks
  - `tdcp_scale_mean_mse_delta_vs_fgo` mean was only `-0.0021`
- Dual-frequency audit of the 7 allowed-phone rows where direct FGO beat
  baseline:
  - all 7 were blocked by `raw_wls.mse_pr < fgo.mse_pr`
  - 3/7 also passed the baseline-gap guard
  - 4/7 failed the baseline-gap guard
  - `fgo_tdcp_scale` stayed near normal FGO, so the scale candidate is not an
    independent accuracy source.
- Read: the next non-base selector experiment, if pursued, should be a narrow
  train-backed FGO-over-raw-WLS proxy rescue.  It must be treated carefully
  because older raw-WLS/FGO proxy exceptions had negative Kaggle A/B evidence.

FGO-over-raw-WLS proxy rescue analyzer:

- Added `experiments/analyze_gsdc2023_fgo_proxy_rescue.py`.
- allowed24 threshold output:
  `experiments/results/gsdc2023_fgo_proxy_rescue_thresholds_20260520_allowed24_e80.csv`
- Best narrow filters on this small set:
  - `ratio<=1.05`, `gapRatio<=2.0`, `qDelta<=-0.20`: selected `2`, wins `2`,
    sum score delta `-6.784 m`, worst `-0.644 m`
  - `ratio<=1.20`, `gapRatio<=1.25`, `qDelta<=-0.35`: selected `3`, wins `3`,
    sum score delta `-6.490 m`, worst `-0.336 m`
- Read: these thresholds are candidates for broader train diagnostics only.
  They should not be promoted into gated selection until a wider phone/course
  sweep shows no regression pocket.

Broader 108-trip proxy-rescue diagnostic:

- Diagnostic CSV:
  `experiments/results/gsdc2023_fgo_proxy_rescue_diag_20260520_trips108_e80.csv`
- Threshold CSV:
  `experiments/results/gsdc2023_fgo_proxy_rescue_thresholds_20260520_trips108_e80.csv`
- The allowed24 filters did not remain clean:
  - `ratio<=1.05`, `gapRatio<=2.0`, `qDelta<=-0.20`: selected `3`, wins `2`,
    losses `1`, sum score delta `-6.091 m`, worst `+0.693 m`
  - `ratio<=1.20`, `gapRatio<=1.25`, `qDelta<=-0.35`: selected `8`, wins `4`,
    losses `4`, sum score delta `-9.138 m`, worst `+2.575 m`
- Top broad threshold selected only one row:
  `ratio<=1.10`, `gapRatio<=1.25`, `qDelta<=-0.40`, selected `1`, wins `1`,
  delta `-5.621 m`.
- Read: there is a positive FGO-over-raw-WLS pocket, but the current
  truth-free guards are too weak for runtime selection.  Do not add this
  rescue to gated selection without an additional phone/course/regime guard.

Phone-group proxy-rescue diagnostic:

- `experiments/analyze_gsdc2023_fgo_proxy_rescue.py` now supports
  `--phone-group` filters.
- Phone-group output:
  `experiments/results/gsdc2023_fgo_proxy_rescue_thresholds_phonegroups_20260520_trips108_e80.csv`
- `pixel4_family` (`pixel4,pixel4xl`) stayed clean:
  - `ratio<=1.05`, `gapRatio<=2.0`, `qDelta<=-0.20`: selected `2`, wins `2`,
    sum score delta `-6.784 m`, worst `-0.644 m`
  - `ratio<=1.20`, `gapRatio<=1.25`, `qDelta<=-0.35`: selected `3`, wins `3`,
    sum score delta `-6.490 m`, worst `-0.336 m`
- `allowed_scale` (`pixel4,pixel4xl,mi8`) best small rule:
  `ratio<=1.15`, `gapRatio<=1.5`, `qDelta<=-0.35`, selected `3`, wins `3`,
  sum score delta `-7.737 m`, worst `-0.336 m`.
- Loss rows in the broad unfiltered rules came from `pixel5` and `pixel7pro`.
  Any runtime proxy-rescue experiment should therefore be opt-in and
  phone-limited.

Runtime opt-in implementation:

- Added disabled-by-default `--fgo-raw-wls-proxy-rescue`.
- Default phone gate: `pixel4`.
- Runtime default thresholds:
  `ratio<=1.20`, `gapRatio<=1.25`, `qDelta<=-0.35`,
  `mseDeltaVsBaseline<=0.0`.
- Smoke results:
  - `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`, 80 epochs,
    dual-frequency gated with rescue: selected `fgo=80`; score `3.0866 m`
    versus baseline `3.6190 m`.
  - `train/2020-08-13-21-42-us-ca-mtv-sf-280/pixel5`, same rescue flag but
    default phone gate: selected `baseline=80`, preserving the known pixel5
    loss row.
- 108-trip gated output:
  `experiments/results/gsdc2023_fgo_proxy_rescue_gated_20260520_trips108_e80.csv`
  - changed trips: `2/108`
  - changed wins/losses: `2/0`
  - selected epochs: `baseline 8160 -> 8000`, `fgo 320 -> 480`,
    `fgo_no_tdcp 160 -> 160`
  - score delta across changed rows: `-0.8686 m`
  - mean score delta across all 108 rows: `-0.0080 m`
  - changed rows were both `pixel4`:
    - `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`: `-0.5324 m`
    - `train/2020-08-04-00-19-us-ca-sb-mtv-101/pixel4`: `-0.3362 m`
  - no non-allowed phone gained FGO epochs.
- Read: safe on this 108-trip slice, but small.  Validate on a larger train
  slice before treating it as a submission candidate.
- Full train 156-trip output:
  - no-rescue baseline:
    `experiments/results/gsdc2023_fgo_proxy_rescue_diag_20260520_train156_e80.csv`
  - rescue enabled:
    `experiments/results/gsdc2023_fgo_proxy_rescue_gated_20260520_train156_e80.csv`
  - changed trips: `2/156`
  - changed wins/losses: `2/0`
  - selected epochs: `baseline 11840 -> 11680`, `fgo 480 -> 640`,
    `fgo_no_tdcp 160 -> 160`
  - score delta across changed rows: `-0.8686 m`
  - mean score delta across all 156 rows: `-0.0056 m`
  - changed rows were the same two `pixel4` trips as the 108-trip run
  - no non-allowed phone gained FGO epochs.
- Read: full-train e80 confirms safety, but the effect is too small to be a
  submission candidate by itself.
- Pixel4-only full-train confirmation:
  `experiments/results/gsdc2023_fgo_proxy_rescue_gated_pixel4only_20260520_train156_e80.csv`
  matched the broader allowed-scale full-train run exactly: `0/156` changed
  rows versus
  `experiments/results/gsdc2023_fgo_proxy_rescue_gated_20260520_train156_e80.csv`.
  The runtime rescue phone gate was narrowed to `pixel4` only.  The separate
  `fgo_tdcp_scale` candidate phone gate remains `pixel4,pixel4xl,mi8`.

Test submission bridge generator:

- Added `experiments/build_gsdc2023_bridge_submission.py`.
- Added `--jobs` and `--resume-existing` for cached per-trip resume from
  `bridge_positions.csv` / `bridge_metrics.json`.
- Smoke output:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_smoke_partial.csv`.
- Pixel4 test e80 partial output:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_testpixel4_e80_partial_20260520.csv`
  - processed the two test `pixel4` trips for 80 epochs
  - selected sources: `baseline=159`, `fgo=0`, `raw_wls=0`
  - coordinate sanity passed
- Full test generation is currently too slow for the interactive loop.  A
  full 40-trip run and a full two-pixel4-trip run were stopped before
  completion.  The next engineering step is cached per-trip or parallel
  generation before using this as a submission candidate builder.
- Resume smoke:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_testpixel4_e80_partial_resume_20260520.csv`
  reused both cached pixel4 bridge outputs with `--jobs 2 --resume-existing`
  and matched the original e80 partial CSV exactly in keys/lat/lon.
- Full test pixel4 partial output:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_testpixel4_full_partial_20260520.csv`
  - processed both test `pixel4` trips with `--jobs 2`
  - patched `3070/3070` pixel4 sample rows; coordinate sanity passed
  - selected sources: `baseline=2870`, `fgo=200`, `raw_wls=0`
  - the FGO block was the first 200 epochs of
    `2021-09-14-20-32-us-ca-mtv-k/pixel4`
  - rescue guard values for that block:
    `fgo/raw_wls_mse_ratio=1.1803`, `quality_delta=-0.5971`,
    `mse_delta_vs_baseline=-3.8807`
- Full pixel4 resume confirmation:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_testpixel4_full_partial_resume_20260520.csv`
  reused both full cached bridge outputs and matched the original full pixel4
  partial CSV exactly in keys/lat/lon.
- Full 40-trip test bridge run:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_full_partial_20260520.csv`
  - all 40 test trips were solved/cached under
    `experiments/results/gsdc2023_bridge_gated_rescue_pixel4_full_20260520`
  - strict assembly found 24 sample timestamps missing from bridge outputs
  - partial assembly patched `71912/71936` rows
  - selected-source row counts among directly patched rows:
    `baseline=70071`, `raw_wls=190`, `fgo=1021`, `fgo_no_tdcp=630`
- Missing timestamp fill:
  `experiments/results/gsdc2023_bridge_gated_rescue_pixel4_full_missing_rows_20260520/missing_bridge_timestamp_rows.csv`
  - materialized 24 rows across 12 trips from nearest selected bridge rows
  - source side counts: `nearest_selected_previous=16`,
    `nearest_selected_next=8`
  - max nearest gap: `11000 ms`
- Improved missing timestamp fill:
  `experiments/materialize_gsdc2023_missing_bridge_timestamp_rows.py` now
  linearly interpolates lat/lon when a missing sample timestamp is bracketed
  by bridge rows, and only falls back to nearest selected row outside the
  bridge time span.
  - interpolation output:
    `experiments/results/gsdc2023_bridge_gated_rescue_pixel4_full_missing_rows_interp_20260520/missing_bridge_timestamp_rows.csv`
  - interpolated rows: `22/24`
  - nearest-next fallback rows: `2/24`
- Original full submission-shaped candidate:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_full_20260520.csv`
  - rows: `71936/71936`
  - sample keys match exactly
  - nonfinite lat/lon rows: `0`
  - coordinate bounds pass
  - local screen output:
    `experiments/results/gsdc2023_local_submission_screen_bridge_20260520.csv`
  - screen result: coordinate sane, not a submitted filename, not a duplicate
    submitted local SHA
- Preferred interpolated full candidate:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_full_interp_20260520.csv`
  - differs from nearest-fill candidate in `22` rows
  - max row change versus nearest-fill candidate: `174.47 m`
  - rows: `71936/71936`
  - sample keys match exactly
  - coordinate screen passed; not a submitted filename; not a duplicate
    submitted local SHA
  - SHA256:
    `3593aa03a42d626a8d7e5fc7b269207980878f5dcc8ee66ff6f780fc3de04dda`
  - screen output:
    `experiments/results/gsdc2023_local_submission_screen_bridge_interp_20260520.csv`
- Pre-submit trip manifest:
  `experiments/results/gsdc2023_bridge_gated_rescue_pixel4_full_interp_trip_manifest_20260520.csv`
  - submission-row source totals: `baseline=70071`, `raw_wls=190`,
    `fgo=1021`, `fgo_no_tdcp=630`, `interpolated_missing=24`,
    `missing_unfilled=0`
  - shortlist output:
    `experiments/results/gsdc2023_safe_unsubmitted_shortlist_bridge_interp_20260520.csv`
  - shortlist status for the interpolated full candidate: `review_manually`
    because safe screen gates pass, but no exact-candidate leaderboard signal
    exists yet.
- Kaggle submission result:
  - file:
    `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_full_interp_20260520.csv`
  - submitted: 2026-05-20
  - message: `bridge gated rescue pixel4 full interp 20260520`
  - status: `COMPLETE`
  - public/private: `3.993 / 4.821`
  - submission history snapshot:
    `experiments/results/gsdc2023_kaggle_submissions_20260520_bridge.csv`
  - post-submit local screen:
    `experiments/results/gsdc2023_local_submission_screen_bridge_postsubmit_20260520.csv`
    marks the file as submitted and duplicate submitted local SHA
  - read: worse than the current `mtv700` family private floor
    (`~4.710-4.713`), but better than the old `gsdc2023_submission.csv`
    historical score (`4.207 public / 5.144 private`).

Bridge failure analysis:

- Artifact directory:
  `experiments/results/gsdc2023_bridge_failure_analysis_20260520/`
- The mtv700 submission CSV body is not available locally, so direct row/trip
  coordinate A/B against the private-floor mtv700 candidate could not be
  produced.  The analysis used submission-log scores plus a rough coordinate
  comparison against old local `experiments/results/gsdc2023_submission.csv`
  only for context.
- Best mtv700 private reference in the submission log:
  `submission_mtv700_plus_p3p25_sjc_only_f25_39_20260509.csv` =
  `public=3.686`, `private=4.710`.  The bridge candidate is therefore
  `+0.307` public and `+0.111` private worse than that reference.
- Source mix: `baseline=70071`, `raw_wls=190`, `fgo=1021`,
  `fgo_no_tdcp=630`, `interpolated_missing=24`, `missing_unfilled=0`.
  Only `200` rows are the pixel4 FGO rescue block.
- Read: the regression is more likely broad bridge/base mismatch plus risky
  non-pixel4 selected-source rows than pixel4 rescue alone.  Highest-risk
  pockets are `pixel6pro` FGO rows, `pixel5` `fgo_no_tdcp` rows, and the full
  bridge baseline replacement.
- Next probe should not be another full bridge submit.  Recover or reconstruct
  the mtv700/private-floor CSV body, then test small mtv700-stack deltas one
  source family at a time: pixel4 rescue only, pixel6pro FGO off/on, pixel5
  `fgo_no_tdcp` off/on, and raw-WLS off/on.

Recovery follow-up:

- Local/workspace filename search and git tracked filename history did not
  find the mtv700/private-floor CSV body.
- Kaggle submission metadata exposes raw paths such as
  `/submissions/52481427/52481427.raw`, but an SDK-authenticated non-API raw
  request returned HTTP `401` with `{"error":{"message":"Unauthenticated"}}`.
  The current API token path still cannot recover submitted CSV bodies.

Private-floor reconstruction audit:

- Added `experiments/audit_gsdc2023_private_floor_reconstruction.py`.
- Audit output:
  `experiments/results/gsdc2023_private_floor_reconstruction_audit_20260520/`
- Result:
  - `private_floor_reconstructable_from_available_files=false`
  - private-floor score rows (`private <= 4.713`): `90`
  - exact private-floor local CSV matches: `0`
  - local old `gsdc2023_submission*.csv` candidate count: `7`
  - missing prerequisite count: `5`
- Missing prerequisites:
  `private_floor_builder_input`, `private_floor_pixel5_patch`,
  `matlab_reference_submission`, `matlab_candidate_submission`, and
  `matlab_bridge_root`.
- Read: the score history preserves many `mtv700`/private-floor labels and
  scores, but the exact CSV bodies and the known source artifacts needed to
  regenerate them are absent in this workspace.  The old local
  `gsdc2023_submission*.csv` files are not score-backed private-floor
  substitutes.
- Gate: do not submit a source-family or bridge-derived Kaggle candidate until
  this audit becomes reconstructable or an exact private-floor CSV body is
  supplied.

Local proxy-base selection:

- Added `experiments/select_gsdc2023_local_proxy_base.py`.
- Output:
  `experiments/results/gsdc2023_local_proxy_base_selection_20260520/`
- Selected dry-run proxy base:
  `experiments/results/gsdc2023_submission_v15.csv`
  - local cluster rank: `1`
  - mean pairwise distance to other sane local old candidates: `0.976 m`
  - nearest neighbor: `gsdc2023_submission_v13.csv` at `0.092 m`
  - `submit_allowed=false`
- Read: `v15` is a local medoid for dry-run comparisons only.  It is not a
  recovered mtv700/private-floor body and must not be treated as a leaderboard
  base.

Local submission provenance audit:

- Added `experiments/audit_gsdc2023_local_submission_provenance.py`.
- Output:
  `experiments/results/gsdc2023_local_submission_provenance_audit_20260520/`
- Result:
  - local `gsdc2023_submission*.csv` candidates: `7`
  - local score-backed private-floor candidates: `0`
  - all local candidates classify as `legacy_pf_local_unscored`
  - git-added time for the local candidates:
    `2026-04-23T06:55:06+09:00`
  - earliest private-floor score row in current score history:
    `2026-04-30 21:26:48.877000`
    (`submission_best_pixel5_3p375_ablate_ablate_old_misc_20260501.csv`)
- Read: local `v15` is a useful dry-run medoid, but it predates the
  score-backed private-floor/basecorr lineage and cannot be promoted to
  mtv700/private-floor without an external exact body or a clean
  reconstruction.

Basecorr private-floor lineage audit:

- Added `experiments/audit_gsdc2023_basecorr_private_floor_lineage.py`.
- Output:
  `experiments/results/gsdc2023_basecorr_private_floor_lineage_audit_20260520/`
- Result:
  - builder presets checked: `33`
  - score-history matched presets: `26`
  - score-backed private-floor builder presets: `13`
  - exact local private-floor candidate bodies: `0`
  - local rebuild blockers: missing `basecorr_builder_input` and
    `pixel5_patch`
- Read: the score history confirms real `basecorr` private-floor lineage in
  the builder presets, but the current workspace still lacks the two CSV
  inputs needed to regenerate those scored bodies.  Prioritize recovering
  `source_selection_lowbaseline_submission_probe_20260430` artifacts or an
  exact scored CSV body before any new source-family submit.

Private-floor recovery dependency audit:

- Added `experiments/audit_gsdc2023_private_floor_recovery_dependencies.py`.
- Output:
  `experiments/results/gsdc2023_private_floor_recovery_dependencies_20260520/`
- Result:
  - artifacts checked: `10`
  - available artifacts: `0`
  - available recovery routes: `0/4`
  - direct private-floor builder ready: `false`
  - missing core artifacts: `base_0555_submission`,
    `current_1450_submission`, `basecorr_builder_input`, `pixel5_patch`
- Read: current workspace recovery is fully blocked at the artifact layer.
  Restoring the two `../ref/gsdc2023/results/test_parallel` submissions can
  reopen the 2026-04-24 best-submission rebuild path, but the submit-relevant
  basecorr private-floor path still needs the direct builder input and Pixel5
  patch, or an exact scored CSV body.

Private-floor recovery one-shot:

- Added `experiments/run_gsdc2023_private_floor_recovery.py`.
- Current output:
  `experiments/results/gsdc2023_private_floor_recovery_run_20260520/`
- Behavior:
  - rerun recovery dependency audit
  - when the direct builder input and Pixel5 patch are present, rerun basecorr
    score-lineage audit and select score-backed private-floor presets
  - rebuild selected basecorr candidates
  - screen rebuilt candidates locally
  - optionally run the submit guard with the recovery dependency summary
- Current result: `status=blocked`,
  `build_skipped_reason=direct_private_floor_builder_not_ready`.
- Read: after restoring artifacts, rerun the one-shot without
  `--allow-unready`; missing artifacts exit nonzero, restored artifacts proceed
  to build/screen and remain non-submit until the guard passes.

Source-family ablation fallback:

- Added `experiments/build_gsdc2023_source_family_ablation_candidates.py`.
  It reads row-level `SelectedSource` from cached bridge `bridge_positions.csv`
  files plus interpolated missing-row metadata, then emits `only` and `revert`
  candidates for source families.
- Old-base dry run output:
  `experiments/results/gsdc2023_source_family_ablation_oldbase_bridge_20260520/`
  - generated candidates: `12`
  - families: `pixel4_fgo=200 rows`, `pixel6pro_fgo=521`,
    `pixel5_fgo_no_tdcp=600`, `raw_wls_all=190`,
    `interpolated_missing_all=24`, `nonbaseline_all=1865`
  - screen output:
    `experiments/results/gsdc2023_local_submission_screen_source_family_ablation_oldbase_bridge_20260520.csv`
  - screen result: all 12 coordinate-sane; no submitted filename; no duplicate
    submitted local SHA
- Read: these old-base probes validate the machinery but are not mtv700-stack
  submit candidates.  Use the same builder with mtv700/private-floor as
  `--reference` if that CSV body becomes available.

Local-proxy v15 source-family dry run:

- Output:
  `experiments/results/gsdc2023_source_family_ablation_localproxy_v15_bridge_20260520/`
- Reference:
  `experiments/results/gsdc2023_submission_v15.csv`
- Target:
  `experiments/results/gsdc2023_submission_bridge_gated_rescue_pixel4_full_interp_20260520.csv`
- Generated candidates: `12`
- Screen output:
  `experiments/results/gsdc2023_local_submission_screen_source_family_ablation_localproxy_v15_bridge_20260520.csv`
- Screen result: all 12 coordinate-sane; no submitted filename; no duplicate
  submitted local SHA.
- Ranking artifacts:
  `experiments/results/gsdc2023_source_family_ablation_localproxy_v15_bridge_20260520/source_family_ablation_ranking_localproxy_v15_bridge_20260520.md`,
  `.csv`, and `.json`.
- Read: the source-family order does not change under the v15 proxy.  This is
  useful for analysis but not for submission.  First real probe still requires
  a recovered/reconstructed private-floor base and should be `pixel4_fgo` only.

Private-floor submit guard:

- Added `experiments/guard_gsdc2023_private_floor_submit.py`.
- Purpose: convert the private-floor reconstruction audit into a hard
  candidate-screen guard before any Kaggle submit.
- Guard behavior:
  - block when
    `private_floor_reconstructable_from_available_files=false`
  - block when the optional recovery dependency summary says the basecorr
    private-floor line is not recoverable from the current workspace
  - block source-family dry runs whose summary `reference` is not a
    private-floor/mtv700 base
  - also preserve normal screen blockers such as coordinate sanity, submitted
    filename, and duplicate submitted SHA
- Local-proxy v15 guard output:
  `experiments/results/gsdc2023_submit_guard_localproxy_v15_bridge_20260520/`
  - `submit_allowed=false`
  - `blocked_count=12`
  - blockers: `private_floor_not_reconstructable`,
    `recovery_dependencies_not_available`, and
    `source_family_reference_not_private_floor`
- Required submit rule: run this guard with `--fail-on-blocked` for any
  source-family/bridge candidate screen before spending a Kaggle submit.

Source-family ranking:

- Added `experiments/summarize_gsdc2023_source_family_ablation.py`.
- Ranking artifacts:
  `experiments/results/gsdc2023_source_family_ablation_oldbase_bridge_20260520/source_family_ablation_ranking_oldbase_bridge_20260520.md`,
  `.csv`, and `.json`.
- Policy:
  - do not submit full bridge
  - do not submit old-base dry-run candidates
  - require a mtv700/private-floor reference body or reconstructed private-floor
    base
  - probe one source family at a time
- First probe once a private-floor base exists: `pixel4_fgo` only (`200` rows,
  one trip).
- Follow-up order: `pixel6pro_fgo` off/on, then `pixel5_fgo_no_tdcp`
  independently, then hold `raw_wls_all` unless there is stronger evidence.
  Treat `interpolated_missing_all` as a fill-policy issue, not standalone
  signal.  Reject `nonbaseline_all` because it repeats the broad bridge failure
  mode.

1 Hz base access check:

- Current environment has no `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` pair
  and no usable `urs.earthdata.nasa.gov` entry in `~/.netrc`.
- GFZ public high-rate index checks did not expose `SLAC`/`VDCY` for checked
  days (`2023/145` and `2021/341`).
- `fgo_dd_carrier` remains blocked as a sub-meter candidate until authenticated
  CDDIS/Earthdata or another usable 1 Hz base source is available.

## Next Action

Recommended next step:

1. If credentials are available, fetch one SLAC/VDCY 1 Hz course file and rerun
   DD support with `--base-obs-template '{base}_1hz.obs'`.
2. If credentials are not available, do not submit the 30 s-base
   `fgo_dd_carrier` candidate.  The conservative phone-limited TDCP-scale
   candidate has now been tested through gated selection and was not selected;
   next useful non-base work needs either a stronger carrier/TDCP candidate or
   a narrowly validated selector rescue, not just a TDCP-weight-scale tweak.

Do not spend a Kaggle submission on the current CT-only candidate.

## TDCP error-state correction smoother (post-v8 lever) — 2026-05-24/25

The earlier naive 4-parameter TDCP least-squares solve broke on Android
`HardwareClockDiscontinuityCount` increments (mean TDCP delta 1971 m vs GT
15.6 m).  The deployed approach sidesteps the clock-state problem entirely:
the bridge applies TDCP **geometry correction against `kaggle_wls`**, so each
inter-epoch TDCP solve estimates the *difference between consecutive WLS
position errors* (a correction increment), not absolute displacement.  The
receiver clock nuisance largely cancels across the geometry-removed interval.

Pipeline (all committed in PR #70):

- `experiments/eval_gsdc2023_tdcp_correction_smoother.py` — per-interval TDCP
  solve (`_estimate_interval`, Huber IRLS) → quality gate (pair count, postfit
  RMS, condition number) → tridiagonal error-state smoother
  `min_c  Σ‖c_i‖²/σ_anchor² + Σ‖(c_{i+1}-c_i)-d_i‖²/σ_tdcp²` per E/N axis.
  Applies standalone, after the v8 chain (`_v8`), or on top of it (`_on_v8`).
- `experiments/apply_gsdc2023_tdcp_to_submission.py` — production apply: rebuild
  TDCP from test raw data, time-match to a submission CSV, add the correction
  on top of submission positions (graceful `time_mismatch` / `no_tdcp` /
  `build_failed` fallbacks).
- `experiments/eval_gsdc2023_tdcp_row_gate.py` — row-level aggressive/conservative
  gate sweep.
- `experiments/merge_gsdc2023_tdcp_adaptive_submission.py` — adaptive merge:
  A32-family phones and LAX/pixel5 trips take the conservative variant, the rest
  aggressive, with an optional per-row aggressive-vs-conservative displacement
  gate.

Train verdict (full 41-trip grid sweep, full epochs,
`gsdc2023_tdcp_onv8_grid_full_candidates_20260524_summary.csv`):

- raw_wls 4.781 m → v8_chain 4.394 m → **best TDCP_on_v8 3.968 m**
  (`tdcp_a4_t0p05_c30_r0p1_p6_d3_on_v8`).
- **TDCP_on_v8 gain vs v8 = −0.43 m / −9.7 %** — the single largest post-process
  lever in the stack (Hampel −5.6 cm, accel −15 cm, snap −4.2 cm, heading
  −1.6 cm, Kalman −9.6 cm all together ≈ −34 cm; TDCP alone ≈ −43 cm), and far
  beyond the original −3…−8 cm hope.

Deployed submission (Kaggle freeze release, highest priority):
`experiments/results/gsdc2023_submission_cauchy_pairwise_hampel_accel3_snap_hdg45_kalman_tdcp_onv8_adaptive_rowgate_fine_20260525.csv`
(71936 rows, coordinate-sanity PASS).  It changes **62218 / 71936 rows
(86.5 %)** from v8, mean change 1.17 m / p95 2.98 m / max 19.6 m, touching
38 / 40 test trips — a broad, train-verified correction, not a few-row tweak.

Unit coverage: `tests/test_eval_gsdc2023_tdcp_correction_smoother.py` exercises
the deterministic numerical core (tridiagonal solve vs dense solve, correction
integration / anchor pull / invalid-interval chain split / max-delta clamp,
ECEF↔LLA round trip, ENU rotation, quality gating).
