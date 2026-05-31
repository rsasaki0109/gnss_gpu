# CT-RBPF-FGO PPC port — 設計スケッチ

**作成日**: 2026-04-25 (HEAD: `76ca2d8` `feature/ppc-realtime-turing-target`)
**目的**: `plan.md §20.3 step 4` を実装するための短い設計メモ。実装は `experiments/exp_ppc_ctrbpf_fgo.py` (新規) に集約する。
**現状ベンチマーク**: libgnss++ hybrid honest aggregate **50.91%** (PPC 6 run合算、TURING 85.6% に対する gap +34.69pp)

---

## 1. 北極星 (再掲)

`internal_docs/plan.md` 冒頭で 2026-04-19 に設定された北極星:

> **A Continuous-Time Rao-Blackwellized Particle Filter with Factor Graph Optimization** (CT-RBPF-FGO)
>
> - **CT**: B-spline trajectory (control points)、任意時刻で (R, p, v, a) を解析的に query
> - **RB**: per-particle velocity を KF で marginalize (Doppler observation を per-particle linear-Gaussian update)
> - **PF**: per-particle NLOS rejection (LOS/NLOS 判定を各 particle の hypothesized position で独立実施)
> - **FGO**: weak-DD window で two-step FGO overlay (velocity FGO → position+TDCP FGO、太郎式)

UrbanNav Odaiba では submeter (1m 切り) を狙う構成。PPC では **honest 0.5m PPC pass% を 50.91% から +X pp** に伸ばす。

## 2. PPC のデータ前提

`PPCDatasetLoader.load_experiment_data()` (`python/gnss_gpu/io/ppc.py:273`) が返す:

| key | shape | 内容 |
|---|---|---|
| `sat_ecef` | list[(n_sat, 3)] | 衛星 ECEF (clock 補正済み) |
| `pseudoranges` | list[(n_sat,)] | sat_clk*c 加味済み PR |
| `weights` | list[(n_sat,)] | SNR 由来 |
| `carrier_phase` | list[(n_sat,)] | L1 (cycle) |
| `doppler_hz` | list[(n_sat,)] | L1 Doppler (Hz) |
| `system_ids` | list[(n_sat,)] | 0=G,1=R,2=E,3=C,4=J |
| `ground_truth` | (n_epochs, 3) | rover GT ECEF (usable epoch のみ) |
| `times` | (n_epochs,) | GPS TOW |
| `base_ecef` | (3,) | base 局 ECEF |

**注意**: `usable epoch` フィルタ(rover sat ≥ 4 + GT 時刻一致 ≤ 0.15s) があるため、`load_experiment_data` の `n_epochs` は ref CSV 行数より少ない。**honest PPC scoring** には ref CSV 全行を分母に取る必要があるので、PF が出ない epoch は `[0,0,0]` で埋める (`exp_ppc_libgnss_hybrid.py:268-278` のパターン)。

## 3. レイヤー別 PPC 設計

各レイヤーは段階的に有効化する knob を CLI に持たせる (`--enable-{ct,rb,pf-nlos,fgo}` 既定 off → 既定 on を後付け)。

### 3.1 PF (base layer)

- 既存の `ParticleFilterDevice` (`python/gnss_gpu/particle_filter_device.py`) と `position_update` / `correct_clock_bias` を再利用。
- state: `[x, y, z, cb]` (4D)、まずは UrbanNav 既存スタックと同じ。
- predict: SPP velocity (Doppler 由来) で短い dt 推進、`sigma_pos` は PPC 1Hz/5Hz に応じてチューニング。
- update: pseudorange row-wise update (既存 `pf.update`)、`weights` は SNR + elevation + system 別 scale。
- 出発点としては `run_pf_standard` 互換ロジックを書き起こす (PLATEAU BVH 依存をハズす)。

### 3.2 RB (per-particle velocity KF marginalize)

- codex14 で実装済の **proper RBPF (velocity KF)** を流用。`particle_filter_device.py` 内の Doppler KF hook が既存 (`--rbpf-velocity-kf`)。
- per-particle に `(v, P_v)` の 3D velocity Gaussian を持たせ、Doppler observation で linear-Gaussian update。
- gate: epoch-level に **DD pair 数 ≥ N、ESS ratio ≥ threshold、Doppler residual median ≤ threshold**。`internal_docs/plan.md §10.1 AAA (region-aware proper RBPF)` の方針。
- まず gate なしで全 epoch 有効化 → noise が乗る epoch は gate で off に切り替え。

### 3.3 CT (B-spline trajectory)

- B-spline control points (`spline_traj.py` か新規) で `(R, p, v, a)` を任意時刻で query。
- まずは **post-process** で PF 出力点列を control point とした **fixed-knot uniform cubic B-spline** に fit (回帰)、IMU 残差を Jacobian 解析で書き下し、最終 trajectory smoothing としてのみ使う。
- on-line CT (predict step を spline 解析微分で評価) は **次セッション以降**。最初は post-process で十分。

### 3.4 FGO (weak-DD window 局所 FGO)

- `python/gnss_gpu/local_fgo.py` の `solve_local_fgo` / `solve_local_fgo_with_lambda` をそのまま再利用。
- 入力: PF 出力 trajectory、DD carrier (DD pair が成立する epoch のみ)、DD pseudorange、undiff pseudorange (fallback)。
- weak-DD 検出は `detect_weak_dd_window` (existing API)。
- output: window 内の置換 trajectory + LAMBDA fix 統計。
- PF とは並行に走らせ、最終出力は **PF (smoother) と FGO の confidence-weighted blend**。最初は `if FGO converged & residual < threshold: replace`。

## 4. 段階的 milestone

| Phase | 内容 | 期待効果 |
|---|---|---|
| **0 (今回 scaffolding)** | PPC 用 PF runner (PR + Doppler + IMU stop) を smoke、`score_ppc2024` で honest score を 6 run 集計 | libgnss++ 50.91% に対し **PF baseline の honest %** を測る (たぶん下回る) |
| 1 | RBPF velocity KF を ON、gate なし | UrbanNav の codex14 と同等。subset 改善が PPC でも出るか確認 |
| 2 | region-aware gate (DD pairs ≥ N、ESS、residual med) | 全体 score 改善 |
| 3 | local FGO bridge (weak-DD window) を ON | 長 NLOS 区間の救済 |
| 4 | LAMBDA partial fix (`solve_local_fgo_with_lambda`) | DD carrier がそろう短区間で精密化 |
| 5 | post-process B-spline smoother | ボーナス。IMU 高頻度出力との 整合 |
| 6 | libgnss++ hybrid と confidence-weighted fusion | 50.91% を超える初期目標 |

Phase 6 で **honest aggregate 60% 超え** が直近の到達目標 (TURING 85.60% は遠いが、Python 単独 phase 1〜5 で +5〜10pp、libgnss++ fusion で更に +5pp 見込み)。

## 5. アンチパターン (再確認)

`plan.md §20.6` より:

- subset-chord PPC scores を信用しない。**全 rover epoch を `score_ppc2024` の denominator に入れる** (不在 epoch は `[0,0,0]` で埋め)
- `min_lock_count < 5` (DD lock 緩和) は false fix で全体悪化済み。**しない**
- Python post-process 単独 (FGO / IMU loose / CV / interp) の天井は ~44%。**libgnss++ C++ 改善か、ground-up CT-RBPF-FGO のみが +X pp の出口**

## 6. 出力契約

`experiments/exp_ppc_ctrbpf_fgo.py` の出力:

- `experiments/results/ppc_ctrbpf_fgo_runs.csv`: 6 run × phase 別 row、honest score (`hybrid_honest_ppc_pct`)、`pass_m`、`total_m`
- `experiments/results/libgnss_ctrbpf_pos/<city>_<run>_full.pos`: pos file (TOW、ECEF、status)
- 集約: print に `HYBRID honest aggregate` 行を 1 本

scoring は `gnss_gpu.ppc_score.score_ppc2024` を直接呼ぶ (libgnss++ hybrid と同じ式)。
