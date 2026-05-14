# GPT Pro 先生へのアドバイス依頼 — PPC2024 PF Ceiling 76.83% 飽和、 残 TURING gap 8.77pp の closing 戦略

**日時**: 2026-05-15
**現 ceiling**: 76.8299% (Phase 19ao 18-variant gici-open TC FGO pool)
**目標**: TURING (PPC2024 winner) 85.6% — 残 gap **8.77pp**
**問題**: 5 連続 marginal/regression で variant 追加路が完全飽和。 architectural breakthrough が必要。

---

## 1. プロジェクト背景

### 1.1 PPC2024 Competition

- **データ**: 6 run × 約 50 分の rover GNSS RINEX + IMU (200 Hz)、 base RINEX + nav
- **対象都市**: Tokyo Odaiba (urban canyon) × 3 run、 Nagoya 街区 × 3 run
- **観測**: GPS L1/L2 + Galileo E1/E5a + GLONASS L1/L2 + BDS B1I/B3 + QZSS L1/L2
  - **E5b 観測なし** (triple-freq UWL LAMBDA は不可)
- **rover**: Septentrio (high-quality dual/triple-freq)
- **scoring**: honest aggregate horizontal fix-rate (within 0.5m of truth、 4-fix only)
- **目標**: TURING (winner) 85.6% aggregate

### 1.2 現状達成

| Phase | 手法 | Aggregate | Δ from canonical |
|---|---|---:|---:|
| 11ep | libgnss++ + dev_demo5 | 71.48% | baseline |
| 17 | + develop branch demo5 port | 72.66% | +1.18pp |
| 19d | + GTSAM FGO 3-config diversity | 73.76% | +2.28pp |
| 19l | + gici-open TC FGO (format bug fix) | 75.38% | +3.90pp |
| 19u | + gici 3-variant (zeroarm + ratio25) | 76.03% | +4.55pp |
| 19al | + 14 more gici variants (17 total) | 76.83% | **+5.35pp** |
| 19ao | drop tightpr, +window5 (18 total) | **76.83%** | +5.35pp (飽和) |

**Per-run final state (Phase 19ao)**:

| Run | canonical | 19ao | Δ |
|---|---:|---:|---:|
| tokyo/run1 | 72.07% | **83.97%** | +11.90pp |
| tokyo/run2 | 85.67% | 86.47% | +0.80pp |
| tokyo/run3 | 81.60% | 81.77% | +0.17pp |
| nagoya/run1 | 66.53% | 67.69% | +1.16pp |
| nagoya/run2 | 43.98% | **44.21%** | +0.23pp |
| nagoya/run3 | 67.01% | 68.46% | +1.45pp |

**Bottleneck = nagoya/run2 (n/r2) で 44.21% に張り付き**。 全 19 variants で改善なし。

---

## 2. PPC pipeline 全体

```
PPC2024 raw (RINEX rover/base/nav + IMU 200Hz bin)
    │
    ├──→ libgnss++ (third_party/gnssplusplus、 my branch feature/demo5-parity-rebase)
    │       │ RTK + IMU-aware demo5 parity (--prefer-trusted-seed,
    │       │   --rtk-update-outlier-threshold, --ar-policy demo5-continuous)
    │       └─→ 6 variants → libgnss_rtk_pos_v5/ + libgnss_diag_phase10/{...}
    │
    ├──→ GTSAM FGO (gtsam_gnss_ws/、 ambiguity_resolution.cpp)
    │       │ Multi-iteration WLS + LAMBDA + ImuFactor pre-integration
    │       │ 3 config diversity: v2_gap, v14_snr38, v17_el25
    │       └─→ libgnss_diag_phase10/fgo_{v2_gap, v14_snr38, v17_el25}/
    │
    └──→ inuex35/gici-open `forppc2024` (gici_open_ws/)
            │ RTK + IMU TC FGO (Ceres-based、 sliding window=3)
            │ 17 variant pool (lever arm/AR ratio/PR-Phase outlier/SNR/elev/IMU/window)
            │ format-fix: nmea_to_ppc_pool_esdfix.py で `output_added=1` 列追加
            └─→ libgnss_diag_phase19/gici_full_<variant>/

    ALL ↓

PPC Selector (experiments/exp_ppc_ctrbpf_fgo.py)
    │ RBPF-velKF + DD + gate + hybrid + rtkdiag_pf
    │ Per-epoch best candidate selection via residual + ratio gate
    │   --rtkdiag-candidate-residual-rms-max 50.0
    │   --rtkdiag-candidate-ratio-min 1.0
    │   --rtkdiag-candidate-recenter-max-shift-m 10000.0
    │   --rtkdiag-candidate-emit-max-diff-m 0.4
    │ Per-run candidate blocking (per-run optimal config)
    │ 2000 particles
    └─→ Phase 19ao aggregate 76.83%
```

---

## 3. Exhausted な path (再試行禁止)

### 3.1 PPC-side 変更 (全て試行済、 marginal/regression)

| 試行 | Δpp | 結論 |
|---|---:|---|
| L1/L2 widelane cascade | +0.012pp | PPC selector で飽和 |
| L1/L5 widelane (Phase 18) | neutral | E5b 不在で UWL stage 不可 |
| WL pre-step AR (threshold sweep) | +0.000pp | 既存 pool で吸収 |
| WL-NL fallback (`--enable-wlnl-fallback`) | **-0.80pp** | geometrically wrong fixes が selector を誤誘導 |
| IFLC (Iono-Free LC) | -3.83pp | 短基線 + urban で IF noise 3x amplification |
| DD-PR LS anchor | -0.23pp | 既存 hybrid と重複 |
| Doppler + IMU 2-stage outlier rejection | -3 〜 -8.7pp | 既存 `\|code_res\|<3m` gate と重複、 good obs を削る |
| Switchable Constraints (`noiseModel::Robust` Cauchy) | offline +0.40pp / PF -0.0003pp | FGO 単独で有効、 既存 3-FGO pool が absorb |
| Reservoir Stein PF | -31.74pp on t/r1 | bandwidth 過大で particle collapse、 SVGD pathology |
| BVH NLOS hard-gating (PF3D phase1) | regressions | city model coverage 不足、 PDOP 劣化 |
| ML NLOS classifier (raw features) | cross-run AUC 0.56 | truth label 不在で sat-track memorization |

### 3.2 gici-open 17 variant pool (飽和の signature)

```
19al 17-variant: 76.8256%
+ himuba (sigma_ba=10/sigma_bg=3):  0.0000pp  redundant
+ tightpr (max_pr_err=1.5):        -0.0151pp  good obs を削る
+ window5 (max_window=5):          +0.0043pp  +2m on t/r1
```

PPC selector pool に追加 candidate を出しても per-epoch best は変わらない。 個別 variant が独立 diversity を提供できなくなった = **per-epoch optimal を absorb 済**。

### 3.3 17 variants の knob カタログ

すでに sweep 済の gici-open YAML knob:
- `gnss_extrinsics` (lever arm): default `[-0.670, 0.593, -1.216]`、 `[0, 0, 0]` (zeroarm)、 `[0.31, 0, 0.55]` (tcgif failed)
- `AR ratio`: 3.0 (default)、 2.5 (ratio25 winner)、 4.0 (ratio40)
- `max_pesudorange_error`: 2.5 (default)、 5.0 (loosepr)、 1.5 (tightpr regression)
- `max_phaserange_error`: 0.06 (default)、 0.12 (loosephase)
- `min_SNR`: 35 (default)、 30 / 40 / 45 (hisnr 系)
- `min_elevation`: 10 (default)、 12 (hielev)
- `min_acceleration`: 0.5 (default)、 0.1 (lowacc)
- `body_to_imu_rotation_std`: 5 (default)、 20 (imurot)
- `sigma_ba` / `sigma_bg`: 3.0 / 1.0 (default)、 10.0 / 3.0 (himuba)
- `max_window_length`: 3 (default)、 5 (window5 marginal)
- 2-knob combos: zeroarm+loosephase = combo (+0.21pp)、 4-knob = combo4 marginal

---

## 4. Bottleneck 詳細 — nagoya/run2

### 4.1 n/r2 の特性

- **総距離**: 4741m
- **fix-rate ceiling**: 44.21% (2096m pass)
- **全 19 variants で 44.21% 一定** — どの knob でも変わらない
- **canonical 11ep でも 43.98%** — 0.23pp しか改善せず
- **TURING 85.6% へは +41.4pp ギャップ** = +1960m が unrecoverable

### 4.2 推察される原因

- urban canyon の severe signal blockage / multipath
- ベースステーション (n/r2 は week 2323 / 2024-07-20) の特性
- IMU bias 推定の long-term drift
- ambiguity が repeatedly slip して NL LAMBDA が convergence しない

### 4.3 試行済 (effect なし)

- IMU bias sigma tuning (himuba: sigma_ba=10/sigma_bg=3)
- IMU rotation std (imurot: 20.0)
- max_window_length 拡張 (window5)
- Phase 11er (n/r2 専用 RTKDiag rescue policy、 +0.13pp aggregate)
- All AR ratio variants
- All lever arm variants
- residual select_mode override: -10.66pp regression

---

## 5. 残 path candidates (まだ試行していない / partial)

### 5.1 Architectural angles

| Path | 推定 +pp | 工数 | 状態 / blocker |
|---|---:|---|---|
| **CLAS / madoca PPP-AR** | +2-3pp | 1-2 週 rebase | PR #55 で infrastructure 投入済、 my branch と共通祖先なし |
| **BVH NLOS rejection** | +1-2pp | 1-2 週 | `python/gnss_gpu/bvh.py` 実装済、 city model data fetch denied |
| **multi-base / network RTK** | +1-2pp | 2-4 週 | base data 取得が必要 |
| **gnss_gpu native FGO** (gici 不要化) | architectural | ~月 | long-term refactor |
| **TripleFreq UWL E1-E5a-E5b** | +3-5pp | 2 週 | **dataset E5b 不在で不可** |

### 5.2 残された gici-open YAML knobs (未試行)

- `iono_modeling`: broadcast (default) / dual_frequency
- `tropo_modeling`: saastamoinen (default) / etc
- `solver_type`: ceres-based fixed
- `time_window_length_slow_motion`: 0.05 (default)
- `time_window_length_dynamic_motion`: 0.5 (default)
- `gnss_extrinsics_initial_std`: [0.05, 0.05, 0.05]
- `num_threads` (no fix-rate impact expected)
- `min_count_per_pair` (cycle slip)
- `cycle_slip_threshold`

### 5.3 PPC selector 側の未探索

- `--rtkdiag-candidate-emit-max-diff-m`: 0.4 (default) — tighter/looser sweep
- `--rtkdiag-candidate-recenter-max-shift-m`: 10000 (default、 essentially off)
- per-epoch best selection の logic 内部 (rule based)
- candidate residual proxy の choice (lowres: Fix=0.1 / Float=0.3 が現行)

---

## 6. GPT Pro 先生への質問

### 6.1 戦略レベル

**Q1**: n/r2 44.21% の構造的限界は **データ起因** (recoverable な fix が物理的に無い) か、 **アルゴリズム起因** (現 17 variants の能力不足) か。 切り分け方法はあるか?

**Q2**: PPC selector が **per-epoch best を absorb 済** とは、 残された +pp は **既存 candidate 間の選択 logic 改善** か **完全に新しい candidate type 追加** のどちらに向くべきか。

**Q3**: 8.77pp gap closing で、 most cost-effective な next step は?
- (a) CLAS/madoca PPP-AR rebase (1-2 週、 +2-3pp)
- (b) BVH NLOS (city model 提供待ち、 +1-2pp)
- (c) gnss_gpu native FGO (~月、 architectural)
- (d) 既存 17 variants の中で 1 つだけ keep する超選択 (selector の noise 削減)

### 6.2 技術レベル

**Q4**: gici-open Ceres-based TC FGO で **AR ratio = 2.5/3.0/4.0** + **lever arm 2 種** + **PR/Phase outlier 2 種** + **SNR 3 種** + **IMU bias 2 種** = 144 通り組合せのうち、 何%が新規 diversity を生むと期待できるか? (現 17/144 = 12% で飽和、 残 87% を機械的に試すべきか?)

**Q5**: n/r2 の 44.21% bottleneck で、 **IMU dead reckoning across loss-of-lock periods** (RTK fix 失敗中 IMU で span を埋める) が有効か? gici-open は ImuFactor を使うが、 PPC selector が "Fix=4" only を採用する logic で IMU-only solution は採用されない仕様。

**Q6**: **CLAS/madoca PPP-AR** で n/r2 は救えるか? PPP-AR は base station 不要で satellite-side correction、 base baseline の制約から解放されるが、 convergence time ~10-30 min は短い run (45 min) で十分か?

**Q7**: **city-model NLOS rejection** で n/r2 は救えるか? PR #55 で BVH 投入済、 OSM building footprints + Manhattan-world height extrusion で proceduralに 3D map 作れるが、 n/r2 の Nagoya 街区の精度はどの程度必要か?

### 6.3 Sanity check

**Q8**: 私の現 ceiling 76.83% は PPC2024 winner 85.6% と比べて **8.77pp 不足**。 winner が公開した手法 / 推察される手法から、 8.77pp の breakdown は何か? (multi-freq cascade / city-aided / multi-base / proprietary algorithms?)

**Q9**: 既存 17 variants pool は **過剰** か (selector noise を増やしているだけ)、 **不足** か (まだ orthogonal な knob がある)、 **適正** か (per-epoch optimal を達成済) か。 検証方法は?

**Q10**: PPC 4-fix gate (`Fix=4 only`) を **緩める** (Float も採用、 quality-weighted) と aggregate fix-rate に対する impact は? 競技ルール上 "fix" の定義が strict だが、 PPC dataset内の Float candidates を Fix の rescue として使う path は?

---

## 7. 主要 file references

### 7.1 PPC selector (gnss_gpu)
- `experiments/exp_ppc_ctrbpf_fgo.py` — メイン PF aggregator (5000 LOC、 PR #58 fallback_mode fix 済)
- `experiments/results/ppc_ctrbpf_fgo_phase19ao_*_runs.csv` — 最新結果
- `python/gnss_gpu/io/ppc.py` — PPC pool loader
- `internal_docs/gici_open_phase19_breakthrough.md` — 詳細 doc
- `internal_docs/plan.md` §0 — TURING 残戦略

### 7.2 libgnss++ (third_party/gnssplusplus)
- branch `feature/demo5-parity-rebase`、 develop port 済 (Phase 14-17)

### 7.3 gici-open (gici_open_ws)
- `build/gici_main` — TC FGO binary
- `option/{tokyo,nagoya}{1-3}_tc_run_<variant>.yaml` — 17+ variant configs
- `nmea_to_ppc_pool_esdfix.py` — format-fixed converter (output_added=1)

### 7.4 GTSAM FGO (gtsam_gnss_ws)
- `examples_cpp/ambiguity_resolution.cpp` — multi-iter WLS + LAMBDA + IMU
- 3 candidate dirs: `libgnss_diag_phase10/fgo_{v2_gap, v14_snr38, v17_el25}/`

---

## 8. 出力依頼

GPT Pro 先生、 以下を希望:

1. **Q1-Q10 への直接回答** (technical depth は専門レベル、 PPP-AR / FGO / RTK の数理含む)
2. **8.77pp gap の現実的 close 戦略** (1 週 / 1 月 / 1 半期 の time-budget 別 ROI 順)
3. **n/r2 44.21% に対する具体的 attack vector** (algorithm-side / data-side / hybrid)
4. **過去の RTK / urban canyon 競技で 80%+ 達成した手法の re-engineering 候補**
5. **私が見落としている angle**

技術的 push back / 異論大歓迎。 PPC2024 dataset が訓練データに含まれていない場合、 一般的な多周波 RTK + urban canyon mitigation 知識からの推論で OK。

以上、 よろしくお願いします。
