# WP3b — Doppler robustness, multi-GNSS audit, IMU preintegration (native FGO on PPC tokyo run1)

This builds directly on `results/wp3a/WP3A_REPORT.md` (chunked native `fgo_gnss_lm_vd`
backbone on PPC tokyo/run1, `E:\datasets\PPC-Dataset-data\tokyo\run1`). WP3a's variant
(a) (PR + motion + clock-drift, GPS-only, `--doppler off`) is the baseline for all
comparisons below: **WLS 2D 95.73 m, FGO 2D 94.52 m, 3D 115.26 m**.

## D1 — Doppler blow-up: root cause + fix

### Root cause

Variant (b) (`--doppler in-repo`, no robustness) blew up on chunk 5 (epochs
5000:6000 of the WP3a chunking): full-run FGO 2D went from 94.52 m to 546.02 m,
with chunk 5 alone reporting `mse_pr = 9.25e5`.

Per-epoch Doppler residuals were dumped against a robust (Huber-IRLS)
per-epoch WLS velocity/clock-drift fit for chunk 5
(`experiments/diag_doppler_chunk5.py`). After fixing a LOS-sign bug in the
*diagnostic itself* (it must match `fgo.cu:doppler_prediction_vd`'s
`los = (sat - rx) / |sat - rx|` convention), typical Doppler residuals in
chunk 5 are small (median ≈ 1.6 m/s) — there is no GLONASS-FDMA / wrong-wavelength
/ sat-clock-drift-sign bug in the underlying observations.

A dedicated isolation test on chunk 5 alone
(`experiments/verify_doppler_gate_chunk5.py`, native VD solver run 5 ways)
shows the blow-up is **not** driven by a small number of static per-epoch
outliers:

| Variant | 2D RMS (chunk 5, 1000 epochs) |
|---|---|
| (a) no Doppler | 119.80 m |
| (b) in-repo Doppler, no robustness | **1844.57 m** |
| (b′) Doppler + D1 median/MAD gate (3σ, 360/6471 obs = 5.6% gated) | 1844.58 m (unchanged) |
| (b″) Doppler + native Huber kernel (`doppler_huber_k=5.0`) | **114.15 m** |
| (b‴) Doppler + gate + Huber | 114.18 m (no extra benefit over Huber alone) |

The per-epoch outlier gate removes 5.6% of observations and changes the
converged 2D RMS by 0.0005% — the gate is not touching whatever is actually
causing the blow-up. The native Huber kernel alone fixes it completely and
even beats the no-Doppler baseline. This points to the real driver being the
*native L2 Doppler factor's sensitivity during LM iterations*, not discrete
bad observations: the VD state is cold-started with `vx=vy=vz=0` (no velocity
initialization from WLS/Doppler), and with `lm_damping=0` (pure undamped
Gauss-Newton, matching WP3a's call), the first few iterations' velocity
corrections are large and transiently blow up the *position* through the
motion-factor velocity/position coupling before the ~6500 Doppler
observations in the chunk pull it back — an outlier-robust kernel that
down-weights large residuals *during* those early iterations fixes exactly
this, whereas a one-shot pre-solve gate (computed once, before any of those
transient iterations) cannot.

### Fix

Enable the already-existing native Huber kernel:
`fgo_gnss_lm_vd(..., doppler_huber_k=5.0)` (exposed as
`--doppler-huber-k 5.0` in `validate_fgo_ppc.py`). No gating needed —
gate+Huber gives no measurable improvement over Huber alone, so the simpler
one-parameter fix is preferred.

### Full-run confirmation (variant b′: PR + motion + Doppler + Huber)

```
set PYTHONPATH=python
set PYTHONUNBUFFERED=1
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler in-repo --doppler-huber-k 5.0 --fgo-iters 8 ^
  --export-csv results/wp3b/tokyo_run1_fgo_doppler_huber.csv
```

| Variant | WLS 2D | FGO 2D | FGO 3D | AllRMS (scorer, 3D) | coverage |
|---|---|---|---|---|---|
| WP3a (a) PR+motion (no Doppler) | 95.73 m | 94.52 m | 115.26 m | 115.26 m | 97.9% (11676/11928) |
| WP3a (b) PR+motion+Doppler (no robustness) | 95.73 m | 546.02 m | 595.66 m | 595.66 m | 97.9% |
| **WP3b (b′) PR+motion+Doppler+Huber (this fix)** | **95.73 m** | **90.04 m** | **110.22 m** | **110.22 m** | 97.9% |

**Success criterion met**: variant (b′) FGO 2D (90.04 m) is strictly better
than variant (a) (94.52 m), and also better than (a)'s 3D (110.22 m vs
115.26 m). Doppler *does* help once the L2 factor is made robust to its own
transient-iteration sensitivity.

## D2 — Multi-GNSS audit

### Root cause of `sats=11`

`PPCDatasetLoader.load_experiment_data(...)` defaults to `systems=("G",)`
(GPS-only). The run-log `sats=11` line prints `max_sats` (the epoch-wise
maximum), not the median — the *median* GPS-only satellite count is only
**8.0**.

Raw RINEX observation-slot histogram (`rover.obs`, all epochs, before any
nav/ephemeris filtering) shows the Septentrio mosaic-X5 receiver is
tracking a BeiDou-majority signal mix, not GPS-majority:

| System | Raw obs slots | Share |
|---|---|---|
| C (BeiDou) | 155 687 | 40.3% |
| G (GPS) | 86 841 | 22.5% |
| E (Galileo) | 67 039 | 17.3% |
| R (GLONASS) | 51 514 | 13.3% |
| J (QZSS) | 25 358 | 6.6% |

No elevation mask is applied anywhere in `load_experiment_data` (confirmed
by code read — not the bottleneck). Ephemeris/nav parsing for
R/E/C/J already works (`read_nav_rinex_multi` + `Ephemeris`); `n_clock` in
`validate_fgo_ppc.run_fgo_on_ppc_native` is already derived dynamically from
the constellations actually observed.

### Sat-count audit (`experiments/diag_sat_counts_d2.py`, data-only, no FGO)

```
set PYTHONPATH=python
python experiments/diag_sat_counts_d2.py
```

| `systems=` | n_epochs | median sats | mean sats | min / max |
|---|---|---|---|---|
| `G` (current default) | 11676 | **8.0** | 7.38 | 4 / 11 |
| `GRECJ` (all 5) | 11924 | **28.0** | 28.40 | 4 / 42 |

Enabling all 5 constellations moves the median from 8.0 to 28.0 sats/epoch —
matching the task's ~20-30 expectation for a multi-GNSS Septentrio receiver,
and confirming `systems=("G",)` is indeed the D2 root cause.

### Effect on variant (a) metrics — full-run result

A naive first attempt to run the full `--systems GRECJ` variant at the
default 1000-epoch chunk size (`n_state = 1000 × 12 = 12000` for
`n_clock=5`) did not finish a *single* chunk after 40+ minutes (vs ~15
min/chunk for the `n_clock=1` Doppler+Huber run) — the dense per-chunk LM
solve scales roughly with `n_state^3`, and `n_state` grows both with
`n_clock` (state stride 8→12) and chunk size. Added a `--chunk-epochs`
override to `validate_fgo_ppc.py` (`_solve_fgo_vd_chunked` already accepted a
chunk-size argument; it just wasn't exposed) — `--chunk-epochs 250` cuts a
single 5-clock chunk from a projected ~60 min to ~52 s (≈48×), making the
full 48-chunk run take ~40 min instead of ~12 h. This is documented as a
solver-perf backlog item (the host LM solve appears to be a dense,
non-blocked Hessian solve; block-diagonal/sparse exploitation of the
per-epoch structure would remove the need for this workaround) rather than a
change to `fgo.cu`.

```
set PYTHONPATH=python
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler off --systems GRECJ --chunk-epochs 250 --fgo-iters 8 ^
  --export-csv results/wp3b/tokyo_run1_fgo_multi_gnss_pr_motion.csv
```

| Variant | median sats | WLS 2D | FGO 2D | FGO 3D | coverage |
|---|---|---|---|---|---|
| WP3a (a) `systems=G` | 8.0 | 95.73 m | 94.52 m | 115.26 m | 97.9% (11676/11928) |
| WP3b `systems=GRECJ`, PR+motion | 28.0 | **108.68 m** | **114.78 m** | **131.27 m** | **100.0%** (11924/11928) |

**Finding: more satellites does not mean better accuracy here.** Both the
raw WLS and the FGO solution get *worse* (2D RMS 94.52 m → 114.78 m) despite
3.5× the median satellite count, while *coverage* improves from 97.9% to
100% (more epochs clear the ≥4-satellite usability threshold). Root-cause
hypotheses, in order of suspected contribution:

1. **No elevation mask** (`load_experiment_data` applies none) — the extra
   BeiDou/QZSS/Galileo/GLONASS satellites likely include more low-elevation,
   multipath-prone signals that a GPS-only, higher-elevation-median set does
   not have.
2. **No per-constellation code-bias / TGD / inter-frequency-bias modeling**
   beyond the free per-system clock offset (`n_clock`) — the C1x
   pseudorange codes used per `_PSEUDORANGE_CODE_PREFERENCES` differ per
   constellation and are not cross-calibrated.
3. Possible lower broadcast-ephemeris fidelity for non-GPS constellations in
   this codebase's RINEX nav parsing (not verified further; would need a
   per-constellation residual breakdown).

This is reported as the D2 finding per the task's own fallback clause
("or a documented finding..."); an elevation mask
(`observation_min_elevation_deg`, already a pattern used elsewhere in this
repo, e.g. `gsdc2023_bridge_config.py`) is the natural next experiment and is
flagged as follow-up work rather than attempted here given the compute
budget already spent on this run.

## D3 — IMU preintegration adapter

### Adapter (`experiments/ppc_imu_adapter.py`)

`ppc_imu_to_processed()` converts `PPCDatasetLoader.load_imu()` output
(gyro in deg/s, GPS-TOW-seconds timebase) into the `ProcessedIMU` structs
expected by `experiments/gsdc2023_imu.py:preintegrate_processed_imu`:

- **Units**: accelerometer columns are already SI (m/s²); gyro columns
  (`Ang Rate {X,Y,Z} (deg/s)`) are converted to rad/s.
- **Timebase**: PPC's `imu.csv` reuses the same GPS-TOW aliases as
  `reference.csv` — i.e. it ships already synchronized to GPS time of week;
  only the seconds→milliseconds unit conversion is needed for
  `preintegrate_processed_imu`.
- **Axes/mounting**: PPC does not document an IMU→vehicle boresight, so a
  zero static mounting angle is assumed (IMU axes == vehicle body axes), with
  heading derived purely from the GNSS-implied velocity direction via the
  `delta_frame="ecef"` mode (flat-vehicle / zero roll-pitch approximation).
  This is a documented approximation, not a calibrated boresight.

`build_ppc_imu_preintegration()` / `load_ppc_imu_preintegration()` orchestrate
loading + preintegration against the GNSS epoch time series, producing an
`IMUPreintegration` that is sliced per chunk with
`imu_preintegration_segment_with_bias_jacobians` inside
`_solve_fgo_vd_chunked` — chunk boundaries are handled the same way the
kinematic state already is (each chunk re-slices the preintegration by its
own `[start:end)` epoch range, so segments never straddle a chunk boundary).
The resulting `imu_delta_p` / `imu_delta_v` / `imu_delta_t` feed
`fgo_gnss_lm_vd` as loosely-coupled position/velocity priors
(`imu_position_sigma_m` / `imu_velocity_sigma_mps`); no attitude/bias states
are added to the VD state width (kept at the existing 8-wide
`[x,y,z,vx,vy,vz,clk,drift]` — the `delta_frame="ecef"` preintegration mode
folds attitude into the delta computation itself, so no extra solver state
is needed to consume it).

### Sigma sensitivity finding

`imu_delta_p` is **not** total displacement — per the preintegration
convention (`pos_{t+1} = pos_t + v_t·dt + ½g·dt² + R·delta_p`), it is only
the acceleration-driven residual on top of the constant-velocity/gravity
term, and is therefore legitimately tiny at ~0.2 s GNSS spacing (measured on
a 1000-epoch chunk: `delta_p` mean |·| = 6 mm, max 33 mm; `delta_v` mean |·|
= 0.06 m/s, max 0.46 m/s — see `experiments/diag_imu_effect.py`).

A first attempt used `imu_position_sigma_m=5.0, imu_velocity_sigma_mps=2.0`
(matching the CLI defaults' "safe" scale) and produced an essentially inert
factor: mean position shift over 1000 epochs vs no-IMU was **1.5 cm**,
`mse_pr` unchanged to 5 significant figures. A sigma sweep on a smaller
chunk found:

| `pos_sigma_m` / `vel_sigma_mps` | mean position shift | max shift | Δ`mse_pr` |
|---|---|---|---|
| 5.0 / 2.0 | 0.015 m | 0.32 m | +0.00% |
| 0.5 / 0.2 | 0.67 m | 8.6 m | +0.04% |
| 0.05 / 0.05 | 3.5 m | 34.6 m | +0.5% |
| 0.02 / 0.10 | 3.7 m | 35.6 m | +0.5% |

Tighter sigmas give the IMU factor real influence but also measurably worsen
the PR residual fit, which — combined with the zero-mounting-angle/no-boresight
approximation above — is a sign that a very tight sigma would force
uncalibrated IMU bias into the trajectory. `pos_sigma_m=0.5,
vel_sigma_mps=0.2` was chosen as the reported configuration: visible,
bounded influence without over-trusting an uncalibrated integration.

### Variant (c): PR + motion + Doppler(Huber) + IMU — full run

```
set PYTHONPATH=python
python -u experiments/validate_fgo_ppc.py --no-rtklib --vd --run tokyo/run1 --max-epochs 0 ^
  --doppler in-repo --doppler-huber-k 5.0 ^
  --imu --imu-position-sigma-m 0.5 --imu-velocity-sigma-mps 0.2 --fgo-iters 8 ^
  --export-csv results/wp3b/tokyo_run1_fgo_imu_doppler_huber.csv
```

| Variant | WLS 2D | FGO 2D | FGO 3D | AllRMS (scorer, 3D) | coverage |
|---|---|---|---|---|---|
| WP3a (a) PR+motion | 95.73 m | 94.52 m | 115.26 m | 115.26 m | 97.9% |
| WP3b (b′) PR+motion+Doppler+Huber | 95.73 m | 90.04 m | 110.22 m | 110.22 m | 97.9% |
| **WP3b (c) PR+motion+Doppler+Huber+IMU** | **95.73 m** | **85.82 m** | **105.73 m** | **105.73 m** | 97.9% |

IMU adds a further, incremental improvement on top of (b′): 2D RMS
90.04 m → 85.82 m (−4.7%), 3D 110.22 m → 105.73 m (−4.1%). Coverage is
unchanged from (a)/(b′) (`systems=G`, 11676/11928 epochs = 97.9%) since
variant (c) does not enable multi-GNSS — see D2 for why combining the two
fixes was deliberately not attempted in one run (multi-GNSS alone already
regressed accuracy here, so stacking it with Doppler+IMU would confound
which change caused what). The overall improvement chain across this report
is monotonic: **94.52 m (a) → 90.04 m (b′) → 85.82 m (c)**, a cumulative
**9.2%** 2D-RMS reduction over the WP3a baseline.

## Artifacts

| File | Description |
|---|---|
| `results/wp3b/tokyo_run1_fgo_doppler_huber.csv` | Variant (b′) trajectory |
| `results/wp3b/tokyo_run1_fgo_multi_gnss_pr_motion.csv` | D2 multi-GNSS trajectory |
| `results/wp3b/tokyo_run1_fgo_imu_doppler_huber.csv` | Variant (c) trajectory |
| `results/wp3b/score_doppler_huber.json` | (b′) scorer JSON |
| `results/wp3b/score_multi_gnss_pr_motion.json` | D2 scorer JSON |
| `results/wp3b/score_imu_doppler_huber.json` | (c) scorer JSON |
| `results/wp3b/scores.csv` | Combined scorer rows |

## Code touched

- `experiments/validate_fgo_ppc.py` — Huber-Doppler / gating (D1), dynamic
  multi-clock `n_clock`/`sys_kind` (D2), IMU wiring (D3), `--chunk-epochs`
  override (perf).
- `experiments/ppc_imu_adapter.py` (new) — PPC IMU → `ProcessedIMU` →
  `IMUPreintegration` adapter (D3.1).
- `experiments/diag_doppler_chunk5.py`, `experiments/verify_doppler_gate_chunk5.py`
  (new) — D1 diagnosis/verification.
- `experiments/diag_sat_counts_d2.py` (new) — D2 sat-count audit.
- `experiments/diag_imu_effect.py` (new) — D3 IMU sigma-sensitivity sweep.
- `tests/test_ppc_imu_adapter.py` (new), `tests/test_validate_fgo_ppc_native.py`
  (extended) — unit tests.

## Tests

```
set PYTHONPATH=python
python -m pytest -p no:xonsh tests/test_ppc_imu_adapter.py tests/test_validate_fgo_ppc_native.py tests/test_score_vs_inuex35.py -q
```

**Result:** 29 passed (`test_ppc_imu_adapter.py` + `test_validate_fgo_ppc_native.py` = 22
IMU-adapter/native-chunking/Doppler-gate tests, plus `test_score_vs_inuex35.py` = 7
pre-existing scorer tests, unaffected by this work).
