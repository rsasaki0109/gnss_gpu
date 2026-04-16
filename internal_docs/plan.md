# gnss_gpu Handoff Plan

Last updated: 2026-04-17 (local)  
Current branch: `main` (clean vs HEAD at last update)  
Last known HEAD: `3388eb2`  
Intended reader: **Claude / next coding agent** (Cursor, Copilot, etc.)

---

## 1. Executive Summary

This document is **not** the old “artifact packaging / paper asset” plan. The active work is **GNSS signal simulation, broadcast ephemeris, and end-to-end (E2E) evaluation** on real UrbanNav + PLATEAU subsets.

**Where the repo stands conceptually:**

- Broadcast ephemeris propagation is used on real nav data. `Ephemeris.compute_batch()` **re-selects the best nav message per epoch** (no “first ephemeris stuck for the whole batch”).
- `UrbanSignalSimulator` takes **`sat_clk`** and applies satellite clock in the simulated pseudorange model.
- **Windows** editable installs: `py::ssize_t`, local `M_PI` replacements, `os.add_dll_directory()` for CUDA, `.pyd` ignored — native extensions are expected to build and load on MSVC.
- **Acquisition** returns **fractional** circular lag (samples) via **3-point parabolic** interpolation around the FFT correlation peak; at zero IF, reported Doppler is a **magnitude** (sign ambiguous).
- **E2E** reconstructs pseudorange from acquisition lag + 1 ms ambiguity resolution + **`sat_clk`**, with a **`2 km` gate** before WLS. This is **not** injecting geometric truth as pseudorange.
- **Post-acquisition refinement** (experiments): GPU `batch_correlate` (**E/P/L**) on the **same 1 ms** IF buffer with **DLL** (code phase in chips) and **PLL** (`atan2(PQ, PI)` on carrier phase in cycles). **No** `scalar_tracking_update` time advance — frozen epoch, iterative nudges only.
- **Weighted WLS**: per-sat weights from **prompt power**, **acquisition peak/second-peak ratio**, and **|DLL| power discriminator** after refinement (`gnss_gpu.e2e_helpers.compute_e2e_wls_weights`; `experiments/e2e_utils.py` is now a back-compat shim).
- **E2E helpers** live in **`python/gnss_gpu/e2e_helpers.py`** (promoted from `experiments/`). Top-level re-exports: `compute_e2e_wls_weights`, `acquisition_lag_to_code_phase_chips`, `code_phase_chips_to_acquisition_lag`, `refine_acquisition_code_lag_dll`, `refine_acquisition_code_lags_dll_batch`, `refine_acquisition_code_lags_diagnostic_batch`, `dump_e2e_diagnostics_csv`, `pseudorange_to_code_phase_chips`, `acquisition_code_phase_to_pseudorange`.
- **Per-channel diagnostics**: `refine_acquisition_code_lags_diagnostic_batch` returns a dict with final E/P/L IQ, code/carrier phase+freq, `prompt_power`, `dll_abs`, and a rough `cn0_est_db` (1 ms coherent, not calibrated). `dump_e2e_diagnostics_csv` writes a 15-column CSV. `exp_e2e_positioning.py` has `--diagnostics-csv PATH` (per-scenario `{stem}_{name}.csv`).
- **CLI** on `experiments/exp_e2e_positioning.py` and `experiments/exp_e2e_trajectory.py`: `--dll-gain`, `--pll-gain`, `--n-iter`, `--correlator-spacing`, trajectory `--max-epochs`, and positioning `--diagnostics-csv`, `--gain-schedule`, `--n-coherent-ms`.

**What is *not* claimed:** a full receiver tracking loop over many milliseconds, navigation bit alignment, or sub-centimeter code tracking. The E2E path is **honest physics + acquisition-grade + short coherent refinement**.

**Main remaining leverage (high level):**

1. ~~Move proven E2E helpers from `experiments/` into **`python/gnss_gpu/`**~~ — **done** (`gnss_gpu/e2e_helpers.py`; `experiments/e2e_utils.py` is a shim).
2. **Longer coherent integration** or multiple ms (nav bit / data wipe issues) if you need another step in accuracy. **Biggest remaining accuracy lever.**
3. **Adaptive gains** vs `cn0_est_db` / `dll_abs` — diagnostic tap is in place, schedule is not.
4. **Warnings / CI hygiene** (low priority; see §8.4/§8.5).
5. Optional: stronger **carrier** aiding (PLL gain schedules, Costas, etc.) — current PLL is minimal.

---

## 2. What Was Done (Chronological Clusters)

### 2.1 Ephemeris and satellite clock

- `Ephemeris.compute_batch()` reselects ephemerides per epoch; tests in `tests/test_ephemeris.py`.
- `UrbanSignalSimulator`: `sat_clk` in `compute_epoch` / `simulate_trajectory`; pseudorange uses `range + rx_clock_bias - c * sat_clk + atmo_delay` (see code for exact path).
- Experiments pass `sat_clk`: `verify_real_ephemeris.py`, `exp_e2e_positioning.py`, `exp_e2e_trajectory.py`.
- **Why it matters:** ignoring `sat_clk` is catastrophic on real RINEX (e.g. Odaiba nav: median `|c·τ|` on the order of **tens of km**; max **~200+ km** — see earlier notes in git history / summaries).

### 2.2 Native Windows / MSVC / CUDA loading

- Pybind: `ssize_t` → `py::ssize_t` where needed.
- CUDA: explicit `M_PI`-style constants where MSVC was flaky.
- `python/gnss_gpu/__init__.py`: `add_dll_directory` for package dir and `CUDA_PATH`/`CUDA_HOME` `bin`.
- `.gitignore`: `*.pyd` and local build artifacts.

### 2.3 WLS / Multi-GNSS consistency

- CPU single-epoch WLS aligned with batch/GPU **receive-time ECEF range** model (removed inconsistent Sagnac mismatch that caused ~31 m bias in tests).

### 2.4 Acquisition

- Parabolic sub-sample peak; fractional `code_phase` in sample units (circular lag).
- Zero IF: Doppler reported as **magnitude** (not arbitrary search sign).

### 2.5 E2E pseudorange reconstruction (baseline)

- `experiments/e2e_utils.py`: lag ↔ chips, ambiguity resolution, `2 km` gate.
- Pseudorange for WLS includes clock handling consistent with experiments.

### 2.6 Post-acquisition refinement and weighted WLS (current experiments)

- **`refine_acquisition_code_lags_dll_batch`**: batch GPU correlate per iteration; **DLL** updates code phase; **PLL** updates `carrier_phase` when `pll_gain > 0`; optional **`return_lock_metrics`** for final prompt power and |DLL|.
- **`compute_e2e_wls_weights`**: combines prompt power, acquisition SNR ratio, and DLL magnitude into **mean-normalized** weights for `wls_position`.
- **`exp_e2e_positioning.py` / `exp_e2e_trajectory.py`**: wire the above; **argparse** for tuning without editing `e2e_utils.py`.

---

## 3. Current Quantitative Status (indicative)

Numbers **vary** with noise, acquisition draws, and which satellites pass the gate. Treat as **ballpark**, not regression baselines unless you freeze seeds and pin data.

### 3.1 Tests

```powershell
$env:PYTHONPATH='python'; python -m pytest tests -q
```

Expect on the order of **440+ passed** (exact count changes when tests are added). On some Windows setups, a broken **xonsh/pytest** plugin may require:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests -q -p pytest
```

Warnings: **clean** as of commit after `3388eb2` — `datetime.utcnow()` →
`datetime.now(timezone.utc)`, `pytest-asyncio` loop scope pinned to
`function` in `pyproject.toml`, matplotlib scatter drops `cmap` when no
color data is passed. `pytest -W default` shows 0 warnings.

### 3.2 Single-epoch E2E (`exp_e2e_positioning.py`)

```powershell
$env:PYTHONPATH='python'; python experiments\exp_e2e_positioning.py
```

**Typical recent range** (DLL+PLL+weights, default CLI):

- Open Sky position error: **~2–6 m** (single draw)
- Urban Odaiba: **~8–12 m** when only a subset of satellites acquires through the gate

Older doc baseline (**~9 m** open sky) was **before** DLL/PLL+weights; do not compare blindly to old screenshots.

**Artifacts:** `experiments/results/e2e_positioning/e2e_positioning.png`

### 3.3 Trajectory E2E (`exp_e2e_trajectory.py`)

```powershell
$env:PYTHONPATH='python'; python experiments\exp_e2e_trajectory.py
```

**Example summary** (defaults: `n_iter=15`, `dll_gain=0.22`, `pll_gain=0.18`, `max_epochs=30`):

| Scenario | RMS [m] | P50 [m] | P95 [m] | Avg NLOS | Avg Acq |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open Sky | **~13** | ~11 | ~21 | 0.0 | ~8 |
| Odaiba | **~22–31** | ~14–28 | ~39–58 | ~0.9 | ~6 |
| Shinjuku | **~90–95** | ~40–65 | ~130–190 | ~2.0 | ~6 |

Urban ordering **Open < Odaiba < Shinjuku** (severity) has been consistent in recent runs.

**Artifacts:** `experiments/results/e2e_positioning/e2e_trajectory_cdf.png`

### 3.4 LOS/NLOS verification

```powershell
$env:PYTHONPATH='python'; python experiments\verify_real_ephemeris.py
```

Large PLATEAU meshes + BVH; runtime **~1 s/frame** order of magnitude on a representative GPU setup.

**Artifact:** `experiments/results/los_nlos_verification/los_nlos_real_ephemeris.gif`

### 3.5 Short sanity checks

- `compute_batch` vs `compute` spot checks on `G01` (same TOW) matched at **0 m / 0 s** delta in past verification.
- Sub-sample acquisition + roundtrip tests: see `tests/test_e2e_utils.py`, `tests/test_signal_sim.py`.

---

## 4. Files That Matter Most Right Now

### 4.1 Ephemeris + urban sim

- `python/gnss_gpu/ephemeris.py`
- `python/gnss_gpu/urban_signal_sim.py`

### 4.2 E2E pipeline

- `python/gnss_gpu/e2e_helpers.py` — **single source of truth** for lag/chips, DLL/PLL batch, WLS weights, diagnostic tap, CSV dump
- `experiments/e2e_utils.py` — back-compat shim (re-exports from `gnss_gpu.e2e_helpers`)
- `experiments/exp_e2e_positioning.py` — argparse + `--diagnostics-csv`
- `experiments/exp_e2e_trajectory.py` — argparse + `--max-epochs`

### 4.3 Acquisition

- `include/gnss_gpu/acquisition.h`
- `src/acquisition/acquisition.cu`
- `python/gnss_gpu/acquisition.py` (+ bindings as wired in the build)

### 4.4 Tracking (used by E2E correlator)

- `src/tracking/tracking.cu` — `batch_correlate` / EPLL math reference  
- `python/gnss_gpu/_tracking_bindings.cpp`

### 4.5 WLS

- `src/positioning/wls.cu`

### 4.6 Tests (focused)

- `tests/test_ephemeris.py`
- `tests/test_urban_signal_sim.py`
- `tests/test_signal_sim.py`
- `tests/test_e2e_utils.py`

### 4.7 Short numbers note (optional)

- `experiments/results/real_ephemeris_e2e_summary.md` — may be shorter than this file; verify dates.

---

## 5. Repo Churn Expectations

Likely **modified** when you work on GNSS sim / E2E:

- `python/gnss_gpu/ephemeris.py`, `urban_signal_sim.py`
- `experiments/e2e_utils.py`, `exp_e2e_*.py`
- `src/acquisition/acquisition.cu`, `include/gnss_gpu/acquisition.h`
- `tests/test_*.py` as above

Likely **modified when experiments rerun** (if tracked):

- `experiments/results/e2e_positioning/*.png`
- `experiments/results/los_nlos_verification/*.gif`

**Ignored on purpose:** `experiments/data/`, `*.pyd` — do not delete these ignores without cause.

---

## 6. Reproduction Cookbook

### 6.1 Build

```powershell
python -m pip install -e .
```

After edits to `src/**/*.cu`, `include/**/*.h`, `python/gnss_gpu/_*_bindings.cpp`.

### 6.2 Focused tests

```powershell
$env:PYTHONPATH='python'; python -m pytest tests/test_ephemeris.py tests/test_urban_signal_sim.py tests/test_signal_sim.py tests/test_e2e_utils.py -q
```

### 6.3 Full suite

```powershell
$env:PYTHONPATH='python'; python -m pytest tests -q
```

### 6.4 Data fetch (if missing)

```powershell
python experiments\fetch_urbannav_subset.py --run Odaiba --output-dir experiments\data\urbannav
python experiments\fetch_urbannav_subset.py --run Shinjuku --output-dir experiments\data\urbannav
python experiments\fetch_plateau_subset.py --run-dir experiments\data\urbannav\Odaiba --preset tokyo23 --output-dir experiments\data\plateau_odaiba --mesh-radius 1
python experiments\fetch_plateau_subset.py --run-dir experiments\data\urbannav\Shinjuku --preset tokyo23 --output-dir experiments\data\plateau_shinjuku --mesh-radius 1
```

### 6.5 Experiments

```powershell
$env:PYTHONPATH='python'; python experiments\verify_real_ephemeris.py
$env:PYTHONPATH='python'; python experiments\exp_e2e_positioning.py
$env:PYTHONPATH='python'; python experiments\exp_e2e_trajectory.py
```

**Tuning (examples):**

```powershell
python experiments\exp_e2e_positioning.py --pll-gain 0 --n-iter 20
python experiments\exp_e2e_trajectory.py --max-epochs 10 --dll-gain 0.2
```

---

## 7. Behavioral Notes (do not regress lightly)

### 7.1 `AcquisitionResult.code_phase` is circular lag (samples)

Over one 1 ms C/A period. **0** and **n_samples** are nearly equivalent lock points — use **circular** error in tests where needed.

### 7.2 E2E reconstruction is approximate

Ambiguity resolution + gate + `sat_clk` handling — **not** a full tracking receiver over many ms.

### 7.3 Keep the **`2 km` gate** unless you replace it with real lock detection

It prevents occasional **multi-km** false locks from dominating urban error statistics.

### 7.4 Post-acquisition DLL/PLL is **not** time-stepped tracking

The same 1 ms buffer is re-correlated; **do not** confuse with `scalar_tracking_update` advancing epoch time. If you wire `scalar_tracking_update`, you need a different story (multiple ms of data or coherent replay).

### 7.5 Weights are **relative** precision

`compute_e2e_wls_weights` mean-normalizes; global scale does not change WLS solution, relative spread does.

---

## 8. Recommended Next Tasks (for the next agent)

### 8.1 ~~Promote E2E helpers into the package~~ — **DONE** (commit `b679cad`)

- `python/gnss_gpu/e2e_helpers.py` is the canonical location.
- `experiments/e2e_utils.py` is a thin re-export shim.

### 8.2 **Longer integration / multi-ms** — **first cut DONE (sim-only)**

- `experiments/exp_e2e_positioning.py --n-coherent-ms N` generates an
  N-ms IQ buffer; acquisition runs on the first 1 ms, refinement uses the
  full N ms. `correlate_kernel` (CUDA) already integrates coherently over
  any buffer length, and the simulator currently emits `nav_bit = +1`
  constant, so no nav-bit wipe is required for simulation.
- **Still open:** real-RINEX nav-bit extraction / data wipe for
  non-simulated flows. That path requires a bit-sync step (e.g., 20 ms
  buffer + Costas-variant detection) and is materially larger than the
  sim-only enabler.

### 8.3 **PLL/DLL tuning and diagnostics** (medium) — **diagnostic tap DONE, tuning pending**

- ~~Per-channel final correlator outputs / CSV~~ — done in commit `3388eb2`
  (`refine_acquisition_code_lags_diagnostic_batch` + `dump_e2e_diagnostics_csv` +
  `exp_e2e_positioning.py --diagnostics-csv`).
- **Still open:** adaptive gains vs `cn0_est_db` / `dll_abs`; schedule PLL gain by
  lock quality; Costas variants.

### 8.4 ~~Warnings cleanup~~ — **DONE**

- `datetime.utcnow()` → `datetime.now(timezone.utc)` in `nmea_writer.py`
- `asyncio_default_fixture_loop_scope = "function"` in `pyproject.toml`
- `plots.py` drops `cmap` when no color data is supplied to `scatter`.
- Full suite: `444 passed, 6 skipped, 0 warnings`.

### 8.5 **CI / regression**

- Optional: one **smoke** job that runs GPU-marked tests only when CUDA is present; document `PYTHONPATH` and plugin workarounds for Windows.

---

## 9. Suggested Reading Order (cold start)

1. `python/gnss_gpu/e2e_helpers.py` — current behavior in one place  
2. `experiments/exp_e2e_positioning.py` — argparse + flow (`--diagnostics-csv`)  
3. `python/gnss_gpu/ephemeris.py` — `compute_batch`  
4. `python/gnss_gpu/urban_signal_sim.py` — `sat_clk`  
5. `src/acquisition/acquisition.cu` — parabolic peak  
6. `src/tracking/tracking.cu` — correlator + discriminator reference  
7. `tests/test_e2e_utils.py`

---

## 10. Known Non-Goals (unless the user redirects)

- Redesigning marketing / paper website assets  
- Rewriting the **entire** CUDA tracking stack “for elegance” without experiment proof  
- Injecting **ideal geometric pseudoranges** back into E2E to fake better accuracy  
- Switching the whole positioning core to a different solver without a scoped reason  

---

## 11. One-Paragraph Handoff

Real-data ephemeris + `sat_clk` are integrated into simulation and E2E; Windows native builds are in a workable state; acquisition outputs **fractional** code lag; E2E uses ambiguity resolution and a **2 km** gate. On top of acquisition, experiments run **GPU E/P/L** refinement with **DLL + PLL** on a **single 1 ms** buffer (no fake time advance), then **weighted WLS** using prompt power, acquisition SNR ratio, and DLL magnitude. Trajectory Open Sky RMS is on the order of **~10–15 m** class in recent default runs (not centimeter-grade). The next sensible steps are **lifting helpers into `python/gnss_gpu/`**, optional **longer coherent integration**, and **maintenance** (warnings/CI) — not re-breaking the honest E2E story.
