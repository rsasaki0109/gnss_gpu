# WP14 — Local verification of the gnssplusplus GTSAM TC-FGO backend

**Verdict: CONFIRMED** — the README's claimed numbers reproduce locally to the
last printed digit on all 3 Tokyo PPC runs (with one important
metric-definition caveat for the cross-tool "beats inuex35" comparison; see
"Honest caveats" below).

## What was verified

Upstream README (`third_party/gnssplusplus/README.md`, "GNSS/IMU
Tightly-Coupled FGO vs tightly-coupled-gnss-imu-fgo") claims, on PPC Tokyo
runs 1-3 with the dataset tactical IMU (`gnss_fgo_parity` recommended preset):
`<50cm` 56.8/80.5/72.8, fix 54.7/78.0/72.2, FixRMS 0.89/0.63/0.29 — versus
inuex35 56.7/69.9/67.9, 49.5/60.8/59.4, 0.82/0.28/0.21.

Source under test: submodule `third_party/gnssplusplus`, branch
`develop-plus-local` @ `1d1ca57` (= origin/develop@09fec9a + 2 opt-in
cherry-picks, both default-off and not activated by the preset). Submodule
working tree clean; **no library source was modified**.

## Build configuration

- Harness: `results/wp14/harness/` — out-of-tree CMake wrapper that
  `add_subdirectory()`s the unmodified submodule and additionally builds
  `wp14_fgo_dump` = verbatim copy of `apps/gnss_fgo_parity.cpp` plus one added
  `--pos-out <csv>` flag (dumps the fixed-lag run's per-epoch solutions as
  `tow,ecef_x,ecef_y,ecef_z,fix` for the campaign scorer; reporting-only, does
  not touch the solve). The library's own `gnss_fgo_parity` was built
  unmodified from the same tree and cross-checked (below).
- Toolchain: Visual Studio 17 2022 (MSVC 14.38.33130), x64 Release,
  `CMAKE_CXX_FLAGS="-DWIN32 -D_WINDOWS -W3 -GR -EHsc -utf-8"`.
- Eigen: vcpkg `C:/vcpkg/installed/x64-windows` (Eigen 3.4.0).
- pybind11 3.0.4 / Python 3.12.10 (project hard-requires them at configure).
- GTSAM:
  - Attempt 1: `E:/repro_tc_fgo_build` (repro-workspace GTSAM
    develop@3c2f54c28, bundled Eigen) — **rejected at compile time**:
    `gtsam/base/Vector.h(66) static_assert: 'GTSAM was built against a
    different version of Eigen'` (bundled-Eigen GTSAM vs project's vcpkg
    Eigen).
  - Attempt 2 (used): `E:/gtsam/install/CMake` — GTSAM 4.3.0
    (develop@a3b8a34), built with `GTSAM_USE_SYSTEM_EIGEN=ON` against the same
    vcpkg Eigen, Boost/TBB/MKL OFF; provides the inuex35 GNSS factors
    (`CarrierPhaseFactor.h`, `PseudorangeFactor.h`, `GnssCommon.h`) that
    `src/algorithms/fgo_gtsam_backend.cpp` requires.
- Expected-and-documented link note: `/FORCE:MULTIPLE` (LNK4088) — the
  library's own CMakeLists declares this as the standard GTSAM-on-Windows
  workaround for duplicated STL COMDAT instantiations.
- CHOLMOD absent (Eigen sparse fallback, library default). Runtime:
  `E:/gtsam/install/bin` (gtsam.dll) on PATH.

## Commands

Per run (README recommended preset; `--ref` and `--pos-out` are
reporting-only; see `results/wp14/run_wp14.sh`):

```
wp14_fgo_dump --rover <run>/rover.obs --base <run>/base.obs --nav <run>/base.nav \
  --imu <run>/imu.csv --ref <run>/reference.csv \
  --fixed-lag 5 --multi-freq --partial-ar --hold --elev-mask 25 --snr-mask 30 \
  --imu-preset-tactical --cmc --cmc-level 0.75 --cp-hold --cp-hold-res 2.0 \
  --exc-recovery --ddpr-anchor --fde --varerr \
  --pos-out results/wp14/tokyo_<run>_fgo_gtsam.csv
```

Campaign scorer:

```
cd C:\Users\rsasa\Workspace\old\gnss_gpu
python experiments/score_vs_inuex35.py --traj results/wp14/tokyo_<run>_fgo_gtsam.csv \
  --format csv --city tokyo --run <run> --out-json results/wp14/score_<run>.json
```

Data: `datasets/PPC-Dataset-data/tokyo/run{1,2,3}` (full runs, all epochs).

## Results

### A. Parity tool's own metrics (2D horizontal error vs reference.csv — the README's metric family)

| Run | Metric | WP14 local | README claim | Match |
|---|---|---:|---:|---|
| run1 | <50cm (2D) | **56.82%** | 56.8% | exact |
| run1 | fix-rate | **54.71%** (6513/11905) | 54.7% | exact |
| run1 | FixRMS (2D) | **0.888 m** | 0.89 m | exact |
| run2 | <50cm (2D) | **80.54%** | 80.5% | exact |
| run2 | fix-rate | **77.97%** (7132/9147) | 78.0% | exact |
| run2 | FixRMS (2D) | **0.628 m** | 0.63 m | exact |
| run3 | <50cm (2D) | **72.80%** | 72.8% | exact |
| run3 | fix-rate | **72.24%** (11049/15294) | 72.2% | exact |
| run3 | FixRMS (2D) | **0.287 m** | 0.29 m | exact |

Solver wall time 314.6 / 282.7 / 353.2 s (run1/2/3); epochs solved
11905/11928, 9147/9151, 15294/15301 (remainder NONE-status; coverage
99.8/100.0/100.0%). All runs GO, nonfinite=0.

Cross-check: the **unmodified** `gnss_fgo_parity.exe` (library's own build
system, same tree) was run on full run1 with the identical preset —
metrics byte-identical to `wp14_fgo_dump`'s (n_fixed=6513, FixRMS 0.888455 m,
<50cm 56.8165%), confirming the dump copy did not alter behavior
(`tokyo_run1_parity_unmodified.log`).

### B. Campaign scorer (3D ECEF error, `score_vs_inuex35.py`) — same scorer on both systems

inuex35 columns are the local same-machine repro (`repro_tc_fgo/results/tc_run{n}.npz`,
scored with identical definitions); inuex35's README values in parentheses.

| Run | Metric | gnss++ GTSAM TC-FGO | inuex35 local (README) | Delta vs local |
|---|---|---:|---:|---:|
| run1 | coverage | 99.8% (11905/11928) | 100% | — |
| run1 | <50cm (3D) | 49.5% | 52.4% (56.7%) | **-2.9 pp** |
| run1 | fix% | **54.7%** | 46.9% (49.5%) | **+7.8 pp** |
| run1 | FixRMS (3D) | 1.730 m | 0.974 m (—) | worse |
| run1 | AllRMS (3D) | 38.51 m | 47.95 m | better |
| run1 | PPC official | 52.48% | — | — |
| run2 | coverage | 100.0% (9147/9151) | 100% | — |
| run2 | <50cm (3D) | **77.3%** | 69.9% (69.9%) | **+7.4 pp** |
| run2 | fix% | **78.0%** | 60.8% (60.8%) | **+17.2 pp** |
| run2 | FixRMS (3D) | 1.055 m | 0.277 m (—) | worse |
| run2 | AllRMS (3D) | 9.68 m | 32.08 m | better |
| run2 | PPC official | 77.28% | — | — |
| run3 | coverage | 100.0% (15294/15301) | 100% | — |
| run3 | <50cm (3D) | 66.0% | 67.9% (67.9%) | **-1.9 pp** |
| run3 | fix% | **72.2%** | 59.4% (59.4%) | **+12.8 pp** |
| run3 | FixRMS (3D) | 0.982 m | 0.211 m (—) | worse |
| run3 | AllRMS (3D) | 13.20 m | 34.52 m | better |
| run3 | PPC official | 62.49% | — | — |

For completeness, the local inuex35 repro re-scored under the parity tool's
2D-horizontal definition: <50cm 58.32/73.92/73.75%, FixRMS(2D)
0.637/0.114/0.109 m. Under that matched 2D definition gnss++ is
-1.5/+6.6/-1.0 pp on <50cm.

## Honest caveats

1. **Metric-family mismatch in the README table.** The parity tool computes
   2D horizontal error; inuex35's README <50cm values are 3D (the local
   inuex35 repro's 3D scores reproduce their README exactly on run2/run3:
   69.9/67.9). So the README's <50cm side-by-side compares gnss++ 2D against
   inuex35 3D. Under matched definitions (either both 2D or both 3D, same
   machine), the <50cm comparison becomes: run1 -1.5/-2.9 pp, run2 +6.6/+7.4
   pp, run3 -1.0/-1.9 pp (2D/3D deltas vs local repro). Fix-rate — whose
   definition is identical on both sides — beats inuex35 decisively on all
   three runs (+7.8/+17.2/+12.8 pp), and AllRMS is much lower on all three.
   run1/run3 <50cm deficits (1-3 pp) are within the known platform-variance
   band (the inuex35 repro itself moved -4.3 pp on run1 vs its own README).
2. FixRMS is behind inuex35 on all runs (README itself concedes this; the
   gap is wider in 3D because the vertical channel is included and the fixed
   population is much larger).
3. run1 coverage is 99.8% (23 NONE epochs); run2/run3 round to 100.0%
   (4 and 7 NONE epochs respectively).

## Verdict

**CONFIRMED.** All nine README headline numbers (3 runs x <50cm / fix /
FixRMS) reproduce exactly under the README's own metric definitions, from a
clean local build of the unmodified library (unmodified-binary cross-check on
run1 byte-identical). Under the campaign's stricter 3D scorer the picture is:
fix-rate and AllRMS beat inuex35 on all 3 runs; 3D <50cm beats on run2
(+7.4 pp) and trails by 1.9-2.9 pp on run1/run3 vs the same-machine inuex35
repro — within the documented platform-variance tolerance, but the "higher
<50cm fraction on all three runs" phrasing of the README should be read
against caveat 1.

## Deliverables

- `results/wp14/tokyo_run{1,2,3}_fgo_gtsam.csv` — per-epoch trajectories
  (tow, ECEF, fix flag; scorer `--format csv`).
- `results/wp14/tokyo_run{1,2,3}_parity.log` — full parity-tool output
  (diagnostics + metrics); `tokyo_run1_parity_unmodified.log` — unmodified
  binary cross-check.
- `results/wp14/score_run{1,2,3}.json` — campaign scorer outputs.
- `results/wp14/harness/` — wrapper CMakeLists + `wp14_fgo_dump.cpp` +
  `build_fgo/` build tree; `results/wp14/run_wp14.sh` — exact run commands.
- No commits made; submodule and parent repo source unchanged.
