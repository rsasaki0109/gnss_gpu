# HANDOFF → Codex: PF-Only + IMU Campaign (FGO抜き) — 2026-07-18

Predecessor campaign: RB-FGO-PF (see `repro_tc_fgo\HANDOFF_CODEX.md`, banked win vs inuex35).
This campaign (started 2026-07-17) is a NEW direction set by the user:
**push the particle filter WITHOUT FGO, with IMU fusion, and with AR.**

## 0. Rules of the game

- **No FGO at runtime.** No sliding-window LM/GTSAM-style optimization anywhere in the
  positioning loop. **Rao-Blackwellization (per-particle/per-basin KF conditionals) is
  explicitly allowed** — that is the intended replacement for FGO conditionals.
- Commit as **rsasaki0109** (never jim/jim-auto). Branch per WP: `agent/wpNN-<slug>`,
  stacked (see §2). **No push / no PR without the user's go** — everything so far is local.
- Every WP has a spec in `internal_docs/task_wpNN_*.md` with explicit gates (G1/G2/G3)
  and an honest report in `results/wpNN/WPNN_REPORT.md` + live `PROGRESS.md`.
  **A measured negative with root-cause passes a gate; an unmeasured claim does not.**
  Keep this discipline — it has caught 4 real bugs so far.
- Roadmap: `internal_docs/pf_only_imu_roadmap_2026_07_17.md` (WP21-25).
  Targets: PPC Tokyo `<50cm_full%` beat inuex35 (56.7/69.9/67.9) without FGO;
  UrbanNav deep-urban mean ≤3.5 m (beats published TC-FGO 3.64 m, Wen 2021).

## 1. IMMEDIATE FIRST ACTION

Start from branch `agent/wp23a-carrier` (HEAD `d9e2463`, WP23a committed and complete).
Read the five WP reports (§2), then write the WP23b spec per §4 and begin there.

## 2. Branch stack & what each WP delivered (all local, stacked in order)

`main` → `agent/wp21-imu-preint` → `agent/wp22a-dd-imu` → `agent/wp22b-likelihood` → `agent/wp23a-carrier`

| WP | Commit | Deliverable | Headline result |
|---|---|---|---|
| WP21 | 87e58a3..3730799 | `python/gnss_gpu/imu_preintegration.py` (FGO-free preint, matches gsdc2023_imu.py to ~8e-17), `pf_imu_preint_adapter.py`, 3-arm ablation | preint default LOST to CV (97.4 vs 76.2 m AllRMS) — sigma_pos ignored heading error |
| WP21b | fa4474c | Heading-variance→sigma_pos (no hand floors), `set_velocity_covariance()` Σ_v feeding, IMUPredictor gravity-sign autodetect fix | preint beats CV 75.33 vs 76.16 m with all noise modeled; margin small because harness is raw-SPP |
| WP22a | d2ccaaf | `--imu {off,preint}` in `exp_ppc_ctrbpf_fgo.py`, ESS instrumentation | `+hybrid` RTK position-update dominates 82-98% of epochs → `<50cm_full%` byte-identical between IMU arms; ESS/N ~1e-4 everywhere |
| WP22b | 7d3f481 | Adaptive per-epoch tempering (`enable_epoch_tempering`, ESS/N target 0.10), C/N0+elev GMM w_los, particle-NLOS wiring | Tempering = only paying upgrade (ESS ×1510, AllRMS −1%). GMM +8-13% worse. Particle-NLOS +204-513% worse (undiff-PR 30 m gate). **Non-hybrid `<50cm_full%` = 0.0% in all 36 cells** |
| WP23a | d9e2463 | DD-PR weight-update stage (was never wired!), Suzuki MUPF schedule `rbpf+dd+cp+gate`, cycle-slip proxy gate, cloud-spread diagnostic, AFV sanity test | `<50cm_full%` still 0.0% on all 12 cells. Two root-caused mechanisms — see §3, items (e),(f) |

Read the five reports in order (`results/wp21/`, `results/wp22a/`, `results/wp22b/`,
`results/wp23a/`) — they are short and each one's conclusion motivates the next WP.

## 3. Hard-won findings (do not rediscover these)

a. **WP22b's `rbpf+dd+gate` baseline was ALREADY carrier-AFV, not DD-PR.**
   `DDCarrierComputer.DDResult` has no pseudorange field; DD-pseudorange was never a
   per-epoch weight update in the non-hybrid path until WP23a added it.
   (WP23A_REPORT §1 — the WP22b report's "DD-PR regime" language is wrong.)
b. **`+hybrid` masks the PF.** To measure PF changes, use non-hybrid variants.
   But note: non-hybrid AllRMS is 28-54 m, `<50cm_full%`=0 — the hybrid RTK floor
   (AllRMS 6.7-15.5 m, 6-10.6%) is currently the only sub-50cm supply.
c. **Weight degeneracy is solved**: adaptive tempering (bisection β to ESS/N 0.10),
   `_apply_pr_ess_guard` in `exp_ppc_ctrbpf_fgo.py`. Keep it on.
d. **Unrelated regression risk**: commit 81cd0a6 (#127, 2026-07-14) changed the
   Doppler-KF input for every `enable_rbpf_velocity_kf=True` variant; recorded
   baselines (e.g. run2 AllRMS 16.99/1.2%) moved to (13.24/10.6%) at HEAD.
   Archived `.pos` artifacts re-score to the old numbers.
e. **Fractional-cycle carrier AFV cannot work on a diffuse cloud** (WP23A §4a):
   post-PR cloud spread is 58-126× λ/2, so the periodic AFV likelihood just reshuffles
   weight among aliased local peaks. Measured, not speculated.
f. **`_apply_pr_ess_guard` has an inertness bug when chained per-stage** (WP23A §4b):
   its early-exit reverts a stage entirely when the ENTERING ESS/N is already below
   target — so the new DD-PR stage contributed literally zero on all grid cells.
   Diagnostic flag `--cp-mupf-resample-before-stage` (resample before each stage,
   as the validated GSDC MUPF track does) fixes it: AllRMS 29.26→20.68 m (−29%) on
   run2/off. **Fix this structurally before building more staged updates.**
g. **IMU preint plumbing is sound but starved**: it raises ESS/N ~5× and trims AllRMS
   a few % — the predict side is no longer the bottleneck; the likelihood/AR side is.
h. Particle-NLOS catastrophe was specifically the **undifferenced-PR 30 m gate**
   (DD-carrier gate alone: zero effect). Needs a redesign, not a retune.

## 4. WP23b — the next task (AR, the user explicitly wants this)

Spec not yet written. Requirements established by WP23A_REPORT §4 (read it):

1. **Integer-ambiguity basins as particle discrete states** with per-basin KF
   conditionals (NOT FGO). Design source: `internal_docs/rbpf_fgo_design.md` — take the
   basin/lineage/cumulative-marginal-likelihood/γ-mass machinery, replace the fixed-lag
   FGO conditional with a per-basin KF. The proven reference implementation (FGO-based,
   for reading only) is `repro_tc_fgo/rbpf_fgo.py`.
2. **GPU batch-LAMBDA reuse**: `src/ar/lambda_batch.cu` + `python/gnss_gpu/lambda_batch.py`
   (WP15, bit-identical to cssrlib mlambda) for top-K integer candidate generation.
3. **LAMBDA needs a float seed with a sane covariance** — NOT this PF's diffuse cloud.
   Options per WP23A §4: (a) an independent WLS/float-KF seed per epoch, or (b) pair
   with the hybrid RTK floor as the seed source. The per-basin KF's float state can
   serve once basins exist (chicken-and-egg: seed the first basins from option a/b).
4. **Fix §3(f) first** (tempering primitive), it will sit under every basin update.
5. **Fix decision = basin posterior mass γ>0.99**, calibrated; port false-fix guards
   from the RB-FGO-PF campaign (DDPR vote, coherent-shift lessons — see
   `internal_docs/fgo-first…` memory notes and `repro_tc_fgo/results/wp17-18 reports`).
6. Gate suggestion: run2 1200-epoch window, `<50cm_full%` > 0 from PF-only for the
   first time; then scale to the 3-run grid; compare vs hybrid floor (6-10.6%) and
   ultimately vs inuex35 (56.7/69.9/67.9 full-run).

## 5. Asset map (quick reference)

- Runner + all PF wiring for PPC: `experiments/exp_ppc_ctrbpf_fgo.py`
  (methods `rbpf+dd+gate`, `rbpf+dd+cp+gate`, `+hybrid` variants; `--imu {off,preint}`,
  `--enable-epoch-tempering`, `--cp-mupf-*` flags). ~12k lines, grep for `CTRBPFConfig`.
- Device PF API: `include/gnss_gpu/pf_device.h`, wrappers `python/gnss_gpu/pf_device_runtime.py`
  (`update_dd_pseudorange`, `update_dd_carrier_afv`, `update_dd_joint`,
  `pf_device_doppler_kf_update`, `set_velocity_covariance`, `get/set_log_weights`).
- IMU: `python/gnss_gpu/imu_preintegration.py`, `pf_imu_preint_adapter.py`, `imu.py`
  (gravity autodetect), `ins_ekf.py`.
- AR: `python/gnss_gpu/lambda_batch.py` (GPU), `lambda_ambiguity.py` (CPU LAMBDA).
- Scoring: `experiments/score_vs_inuex35.py` (use `--fix-statuses 1` per WP23a's
  score_grid.py), `python/gnss_gpu/ppc_score.py`. Grid scorers:
  `results/wp2{2b,3a}/score_grid.py`.
- Data: `datasets/PPC-Dataset-data/tokyo/run{1,2,3}` (5 Hz rover, 100 Hz IMU, truth);
  UrbanNav Tokyo under `data/urbannav/`; PLATEAU meshes `experiments/data/plateau_*`.
- 3D NLOS (unused so far in this campaign, roadmap WP22c): `pf_weight_3d_bvh`,
  `src/raytrace/`, `src/diffraction/utd.cu`.
- Literature basis (verified URLs in roadmap §4): Niimi RA-L 2026 (RBPF w/o AR, the
  architecture validation), Suzuki ICRA 2024 (multiple-update), Gupta&Gao (GMM),
  Wen 2021 (TC-FGO 3.64 m target), rbpf_fgo_design.md (basin machinery).

## 6. Environment notes

- Windows 11, PowerShell. Python env: the one `experiments/` README/existing runs use
  (PYTHONPATH=python). Runs are cheap: 1200-epoch × 50k particles ≈ 15-40 s each.
- Test pattern: `tests/test_imu_preintegration.py`, `test_pf_imu_preint_adapter.py`,
  `test_pf_device_wrapper.py`, `test_wp23a_dd_carrier_afv_sanity.py` + the WP22 suite
  (41 tests currently green). One pre-existing unrelated Windows path-separator failure
  exists in the broader PF suite — documented in WP21_REPORT, not ours.
- `.pos` outputs are gitignored by convention; CSVs + reports are committed.
