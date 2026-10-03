# PPC canonical pipeline: local reproduction on Windows (2026-10-03)

The honest PPC result has been reproduced end to end on the Windows dev
machine, bit-for-bit at the score level:

- 11,031 safe FIX epochs with zero false FIX (2026-08-02 evidence:
  `ppc_causal_imu_motion_consensus_evidence_2026_08_02.json`);
- 58.521912% for the safe output;
- **59.222040%** after the causal FLOAT selector
  (`docs/ppc_causal_float_selector_evidence.json`).

Every per-route count and score matches the recorded values.

| Route | Safe FIX (rec.) | Safe output % (rec.) | Selected % (rec.) |
|---|---:|---:|---:|
| tokyo/run1 | 1257 (1257) | 65.2482 (65.2482) | 67.4483 (67.4483) |
| tokyo/run2 | 1907 (1907) | 81.1692 (81.1692) | 81.1833 (81.1833) |
| tokyo/run3 | 5154 (5154) | 78.2820 (78.2820) | 78.9118 (78.9118) |
| nagoya/run1 | 1160 (1160) | 51.4534 (51.4534) | 52.6397 (52.6397) |
| nagoya/run2 | 1344 (1344) | 28.9462 (28.9462) | 28.9646 (28.9646) |
| nagoya/run3 | 209 (209) | 46.0326 (46.0326) | 46.1845 (46.1845) |
| **mean / total** | **11031** | **58.521912** | **59.222040** |

The `gnss_fuse` candidate `.pos` files differ in hash from the recorded ones
(different compiler/build), but the selector output and every score are
identical.

## Inputs

- Data: `E:\datasets\PPC-Dataset-data` (6 routes; PLATEAU in
  `E:\datasets\plateau`).
- gnssplusplus `62bd0b73` (the gnss_gpu submodule pin until 2026-10-04;
  check it out explicitly, the pin has since moved), exported with
  `git archive` from the local clone
  `C:\Users\rsasa\Workspace\rtklib_v2_ws\gnssplusplus-library`.
- GTSAM: `E:\gtsam\install`. The native IMU FGO path rejects builds without
  the GTSAM backend (`--multisd-fgo-imu requires a build with the GTSAM
  backend`).

## Build (VS 2022 x64 developer shell, Ninja)

```bat
cmake -S <gnsspp-src> -B <gnsspp-src>\build312g -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGNSSPP_BUILD_PYTHON_BINDINGS=OFF -DGNSSPP_USE_CHOLMOD=OFF -DBUILD_TESTING=OFF ^
  -DEigen3_DIR=C:/vcpkg/installed/x64-windows/share/eigen3 ^
  -DGTSAM_DIR=E:/gtsam/install/CMake -DCMAKE_PREFIX_PATH=C:/vcpkg/installed/x64-windows
cmake --build <gnsspp-src>\build312g --target gnss_solve gnss_fuse -j 8
```

At runtime, put `E:\gtsam\install\bin` and `C:\vcpkg\installed\x64-windows\bin`
on `PATH`.

## Commands (Git Bash; set `PYTHONIOENCODING=utf-8` and `PYTHONPATH=python`)

1. Safe IMU PF/FGO tracker, all six routes. This is CPU-bound and takes about
   10 min per 12k-epoch route, about 55 min in total. The flags match the
   frozen 2026-08-02 policy; the runner defaults (`--top-k 4`,
   `--fix-min-streak 3`, gap 0) do **not**.

   ```bash
   python experiments/run_ppc_basin_fgo_six_route.py --binary <build312g>/apps/gnss_solve.exe \
     --data-root E:/datasets/PPC-Dataset-data --output-dir <out> --max-epochs 0 \
     --top-k 8 --quality-ranked-par --fix-min-streak 2 --validation-gap-tolerance 2 \
     --disjoint-holdout-consensus --causal-imu-motion-consensus \
     --native-imu --native-imu-aperture 0.30 --native-imu-fix-min-streak 2 --cuda-mode off
   ```

2. Safe output per route. The baseline is the route's own `gnss_solve` `.pos`.

   ```bash
   python experiments/compose_ppc_imu_safe_output.py --baseline-pos <out>/<route>_basin_fgo_k8_efull.pos \
     --tracker-csv <out>/<route>_basin_fgo_k8_efull.tracker.csv \
     --tracker-summary <out>/<route>_basin_fgo_k8_efull.tracker.json \
     --output <route>.safe_output.csv --summary <route>.safe_output.json
   ```

3. `gnss_fuse` FLOAT candidates (about 30 min with `--jobs 2`).

   ```bash
   python experiments/run_ppc_float_candidates.py --binary <build312g>/apps/gnss_fuse.exe \
     --expected-binary-sha256 <sha256 of that exe> --dataset-root E:/datasets/PPC-Dataset-data \
     --output-root <cand> --jobs 2
   ```

4. Causal FLOAT selector per route, then
   `experiments/evaluate_ppc_official_score.py --estimate ... --reference .../reference.csv`.

   ```bash
   python experiments/compose_ppc_causal_float_selector.py --route <route> \
     --safe-output <route>.safe_output.csv --safe-summary <route>.safe_output.json \
     --float-candidate-pos <cand>/<route>/float_candidate.pos \
     --candidate-manifest <cand>/<route>/run_manifest.json \
     --output <route>.selected.csv --summary <route>.selected.json
   ```

## Use

This is the baseline for the next PPC work in plan.md item 2: the 25%
safe-FIX milestone, 1,164 epochs short, with 4,038 epochs of candidate-oracle
headroom. Any change can now be measured against the exact 11,031 / 58.52% /
59.22% numbers on this machine.
