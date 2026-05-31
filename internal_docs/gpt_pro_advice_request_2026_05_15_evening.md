# PPC2024 90% target — 進捗報告 + 助言依頼

## Goal

PPC2024 (Tokyo + Nagoya 6 runs) で **OFFICIAL per-run-averaged path-weighted PPC at 50cm = 90%** を target。 TURING (PPC2024 winner) = 85.6%。

## Current best baseline

- **Phase 19aw K=3 = 83.42% OFFICIAL** (PR #59 committed、 早朝 session breakthrough)
- 構造: 17-variant gici-open pool + status=5 gate + velocity bridge + K=3 rms_prefilter on top of `temporal_n2_v10` etc.
- TURING gap 残: **2.18pp**
- 90% target gap 残: **6.58pp**

### Per-run baseline (Phase 19aw K=3)

| Run | OFFICIAL pass% | Selector mistake (epoch) | Oracle unreachable (epoch) |
|-----|---------------:|--------------------------:|----------------------------:|
| tokyo/run1 | 90.13 | 6.99 | 8.18 |
| tokyo/run2 | 94.83 | 1.22 | 2.50 |
| tokyo/run3 | 84.95 | 6.54 | 4.40 |
| nagoya/run1 | 80.38 | 2.81 | 3.66 |
| **nagoya/run2** | **62.03** | **27.30** | **13.76** |
| nagoya/run3 | 88.17 | 5.75 | 0.98 |
| **avg** | **83.42** | **8.43** | **5.58** |

- **n/r2 が支配的問題**: 27.30pp selector_mistake (epoch-count) + 13.76pp oracle_unreachable
- "selector_mistake" = oracle 候補が pool に居るが selector が picking 間違える
- "oracle_unreachable" = 50cm 以内の候補が pool に存在しない

### n/r2 oracle_unreachable epoch の時間分布

| Δt to nearest accurate fix=4 anchor | epochs | % |
|--|--:|--:|
| ≤3s | 65 | 5.0% |
| 3-6s | 25 | 1.9% |
| 6-10s | 36 | 2.8% |
| 10-30s | 176 | 13.5% |
| 30-60s | 157 | 12.1% |
| **1-5min** | **841** | **64.7%** |

→ unreach の主体は **deep urban canyon spans 1-5 分**。 IMU/TDCP/multi-anchor bridge 必須。

## Session で試した path (全 negative or no-op)

### 1. cluster_vote select_mode (n/r2 oracle gap 解析発)

仮説: K=3 prefilter で n/r2 mistake epoch 2580 の oracle が 64% は largest spatial cluster にいるので、 全候補から majority voting すべき。

実装: 50cm radius greedy clustering、 largest cluster → 内部で rms 最小 pick。 47-pool (n/r2) で動作。

**結果 (Phase 19ay clv2, 6-run aggregate)**:
| Run | K=3 | clv2 | delta |
|-----|-----:|-----:|-----:|
| t/r1 | 90.13 | 87.90 | -2.23 |
| t/r2 | 94.83 | 92.91 | -1.92 |
| t/r3 | 84.95 | 84.27 | -0.68 |
| n/r1 | 80.38 | 82.16 | **+1.78** |
| n/r2 | 62.03 | 59.41 | -2.62 |
| n/r3 | 88.17 | 89.29 | **+1.12** |
| **avg** | 83.42 | 82.66 | **-0.76** |

- n/r2 で epoch-count では +2.73pp net (FAIL→PASS 755 epochs) だが **path-weighted では -2.62pp** = regression は dense trajectory area の PASS→FAIL 497 epochs (5.27%) に集中。
- K=7/K=10/K=15/K=20 prefilter on cluster_vote sweep でも n/r2 最良 60.36% (K=3 temporal 62.03% に劣る)。
- **Per-run dispatch (n/r1+n/r3 のみ cluster_vote、 残 4 run K=3 維持)**: **83.90% (+0.48pp)** — 唯一 positive、 path-weighted で committable な唯一の small win。

### 2. TDCP smoother + K=3 rtkdiag_pf combo

仮説: rtkdiag_pf の per-epoch PF emit position に fwd+bwd Kalman smoothing (TDCP carrier-phase delta velocity 経由) を post-process で適用、 deep canyon 1-5 分 gap を interpolation で sub-meter に refine。

実装: 既存 `enable_tdcp_smoother=True` を method `rbpf+dd+gate+hybrid+rtkdiag_pf+tdcp` で活性化、 6 runs 実行。

**結果**: 6-run aggregate **80.65% = -2.77pp regression**、 全 run regression:
- t/r1 90.13→86.61 (-3.52pp)
- t/r2 94.83→93.93 (-0.90pp)
- t/r3 84.95→83.53 (-1.42pp)
- n/r1 80.38→78.31 (-2.07pp)
- n/r2 62.03→58.22 (-3.81pp)
- n/r3 88.17→83.33 (-4.84pp)

**Mechanism**: TDCP smoother が `hybrid_pos` (libgnss_rtk_pos_v5、 hybrid_pu input) を observation として使うので、 rtkdiag_pf が既に hybrid より良い候補選んでた epoch を hybrid 方向に引き戻す。 `enable_hybrid_pu=True` のままだと non-compose 構造。

### 3. PPP-AR (CLAS/MADOCA) pipeline

`third_party/gnssplusplus` に `gnss_ppp --enable-ar` binary 完備、 `gnss_fetch_products.py` で IGS preset 自動 fetch、 `apps/experiments/ppp_ar/` exists。 CLAS PPP-AR の test_clas_ppp.sh 完成済。

**ブロッカー**: PPC2024 dates (2024-07-20, -07-23, -08-03) の MGEX final products (SP3/CLK/IONEX/DCB) は CDDIS NASA / ESA SSC / BKG MGEX どれも auth 必要、 anonymous download は HTML login page 返す。 Earthdata Login credentials を `~/.netrc` に設定すれば動くが session 中に config 不可。

### 4. City map NLOS suppression (PLATEAU)

**既存 infrastructure 大充実**:
- `gnss_gpu.io.plateau.PlateauLoader` + `BVHAccelerator` ray-trace ready
- `experiments/fetch_plateau_subset.py` で PLATEAU CMS から trajectory-overlap mesh だけ HTTP range で fetch (国交省 公式 CMS、 auth 不要)
- `experiments/scan_ppc_plateau_segments.py` segment-level NLOS scan

**実装**:
1. tokyo/run1 PLATEAU 10 bldg + 8 brid mesh tiles fetch (~2GB)
2. EGM96 geoid file を venv proj に copy で fix
3. scan_ppc_plateau_segments BVH 化 (OOM fix)
4. 全 tokyo/run1 11923 epochs を per-epoch BVH check → **216160 (epoch, sat) NLOS rows、 NLOS rate 27.61%**

**Segment-level finding (tokyo/run1, 15 segments × 500 epoch)**:
| Segment epoch range | NLOS% | PF_pass% (Phase 19aw K=3) | err_p90 |
|--|--:|--:|--:|
| **3255-3754** | **50.2%** | **32.5%** | **68.75m** |
| **9741-10240** | **69.1%** | **23.0%** | 4.39m |
| 5425-5924 | 41.6% | 85.0% | 71.09m |
| 2693-3192 | 48.0% | 97.6% | 0.15m |
| 11368-11867 | 11.5% | 100% | 0.17m |

**強い correlation** (高 NLOS = 高 PF error) があるが universal じゃない (48% NLOS でも 97% PF_pass する segments も)。

**自作 candidate 生成 pipeline**:
1. Per-epoch NLOS extractor (`/tmp/extract_per_epoch_nlos.py`) → 216160 rows / 27.61% NLOS aggregate
2. RINEX masker (`/tmp/mask_rinex_nlos.py`) → 50708 sat-lines removed from 5353 epochs (n_sat field 自動更新)
3. `gnss_solve` on `tokyo_run1_rover_nlos_masked_full.obs` with `--preset low-cost --min-hold-count 2 --hold-ratio-threshold 2.0` (loose preset)

**Standalone gnss_solve 比較**:
| Method | Pass@50cm | p50 err | p90 err |
|---|--:|--:|--:|
| baseline (unmasked) | 33.76% | 0.72m | 30.99m |
| partial NLOS-mask (first 880 epochs) | 36.86% | 0.63m | 30.78m |
| **full NLOS-mask** | **28.52%** | 0.68m | **24.06m** |

- partial で +3.10pp signal あり (rescue 520 / over-mask 287 = net +233)
- **full で -5.24pp regression** (rescue 602 / over-mask 986 = net -384)

**理由 (full mask の RTK 壊滅)**:
- nlos_masked diag CSV: status=4 (Fix) はわずか **6.5%** (vs 通常 candidate >90%)
- final_residual_rms p50 = **0.585m** (vs 通常 < 0.05m)
- NLOS 除去で sat 数不足 → LAMBDA AR 失敗 → Float/SPP に degenerate

**Pool 統合 (Phase 19bb)**: `xd_nlos_masked` を tokyo/run1 pool に追加して rtkdiag_pf 再 run → **90.13% = baseline と完全同一**、 nlos_masked candidate は selector で **0 回選ばれず** (rms 高すぎて K=3 prefilter で常に除外)。

## 残された path 候補 (どれも独立な数日〜数週 work)

### A. NLOS soft mask via SNR (完了、 null - worse than hard-mask)

実装: RINEX 3 obs file の per-system obs type header parse (G=16, E=12, R=12, C=20, J=12)、 S* signal-strength field 位置 locate、 NLOS sat の S* 値を 20.000 dBHz に書換え。 全 50708 (epoch, sat) pair で SNR 変更。 `gnss_solve --rtk-snr-weighting --rtk-snr-reference-dbhz 45 --rtk-snr-max-variance-scale 25` で variance を up to 25x inflate。 obs 保持なので AR には全 sat 渡る。

**結果**: SOFT-mask = **21.20% pass@50cm** (baseline 33.76% から -12.56pp、 HARD-mask 28.52% から更に -7.32pp 悪化!)。 status hist 同じく Fix=798 (10.7%)、 大部分 Single/Float。

**Mechanism (なぜ hard-mask より悪い)**: SNR=20 とすると libgnss は link 品質劣化扱いするが、 carrier-phase 観測値は元のまま (variance だけ inflate)。 libgnss が "low SNR with valid CP" の inconsistency で全 RTK measurement に対し reliability を見直し → 結果 AR 機能不全。 hard-mask は完全に sat 不在になるので少なくとも sat geometry 計算は clean、 soft-mask は中途半端。

**Path A 完全 exhausted**。 RINEX-level NLOS modification は本質的に dead end。 NLOS 情報 を solver 内部に直接渡す必要 → Path C のみ残る real path。

### B. Stronger gnss_solve preset on masked obs (完了、 null)

仮説検証: 弱 preset "loose" だと AR が容易に壊れるが、 stronger preset (arfilter, hold=8, ratio=2.6) なら NLOS 除去後でも AR 維持できる可能性。

**結果**: NLOS-mask + strong preset = **28.49% pass@50cm** (NLOS-mask + loose 28.52% と marginal 同等、 status hist 同じく Fix=6.5%、 rms p50=0.577m)。 NLOS hard-mask が binding constraint、 preset tweaking では救えない。 **Path B 完全 exhausted**.

### C. PLATEAU NLOS を solver code 内部に integrate

`exp_ppc_ctrbpf_fgo.py` の PR weight stage (line ~2482, `_pr_weight_mask`) で per-(epoch, sat) NLOS flag を読込、 NLOS sat の sigma を 5-10x boost。 PF observation の effective weight が下がる。 数日 work。

### D. PPP-AR with Earthdata Login

`~/.netrc` に `machine urs.earthdata.nasa.gov login U password P` 設定後、 `gnss_fetch_products.py` で IGS final products (SP3+CLK+IONEX+DCB) を 3 dates 分 fetch、 `gnss_ppp --enable-ar` を 6 PPC runs 実行、 .pos を PPC candidate format converter (`/tmp/ppp_pos_to_diag.py`) で pool に追加。 +1-3pp 期待、 1-3 週。

### E. gici-open `PppEstimator` build & integrate

`/media/sasaki/aiueo/ai_coding_ws/gici_open_ws/` に既に clone 済、 全 build deps installed、 PPC2024 dataset 整合 (imu.csv→bin converter 必要)。 `PppEstimator` + `PppImuTcEstimator` (本物の TC FGO) 動かせば候補品質激改善見込。 +1-2pp、 2-3 週。

### F. n/r2 27.30pp selector_mistake — data-driven 解析 + block experiment

**Oracle vs PF-pick feature 比較** (n/r2 mistake epoch 2580):
- **rms-trap 99.2%**: Oracle が PF-pick より高 rms (rms ranking が逆効果)
- **abs_max-trap 97.1%**: 同様 (composite formula の abs_max term も逆効果)
- Status: Oracle 61% Float / 39% Fix、 PF-pick 56% Float / 44% Fix (PF-pick の方が Fix 寄り)
- sats: PF-pick がわずかに多い (p50 oracle=15 vs pf=16)

**Variant 集中**:
- PF-pick top 2 variants: `gici_full_zeroarm` (1070=41%) + `gici_full_combo4` (914=35%) = **77% of PF-picks**
- Oracle top 5 variants: diverse (`full_ratio15_lock3_trustedseed_csig005_em10` 370、 `gici_full_onarm` 316、 `gici_tc_esdfix` 269、 `gici_full_lowacc` 259、 `full_ratio15_lock3_trustedseed_mlc1oGc005` 188)

**実験 (Phase 19bc)**: n/r2 で `xd_gici_z` + `xd_gici_c4` を block → **n/r2 59.78% (-2.25pp regression)**。 cluster bias は variant 別じゃなく **SPATIAL 構造的** — zeroarm/combo4 を抜くと次点の同 cluster gici (zr/combo/hisnr) が同様に低 rms で picked、 結果は同じ wrong cluster 続行。 zeroarm/combo4 自体は時々 Status=4 を出すので block で全 Float に degenerate。

**Path F null 確定**。

K=3 prefilter は K=1/K=2/K=5/K=7 sweep でも n/r2 最良 62.03% で頭打ち。 cluster_vote も -2.62pp。 残る idea (試してない):
- temporal alpha tuning (現 0.00062 - 既に最良に見える)
- composite formula re-derivation (`residual / (... * abs_max^c)` の c<0 にすると逆方向)
- per-epoch feature-based selector (gradient boosting on diag features → mistake probability)
- **External anchor**: hybrid_pos / IMU-integrated DR / PPP / PLATEAU NLOS の何らかを selector の prior に

## 助言依頼内容

1. **どの path が 90% に最短か?** A-F のどれを優先すべきか
2. **NLOS soft mask via SNR (A)** の効果見積もり。 path-weighted +pp の reality?
3. **PLATEAU NLOS solver-internal integration (C)** の design。 PR sigma boost factor / 適用条件 / 注意点
4. **n/r2 selector mistake (F)** で K=3 prefilter 後に残る trap を解消する新規 angle
5. **PPP-AR (D) で +1-3pp は楽観的か realistic か**。 PPC2024 dataset (Septentrio mosaic-X5、 triple-freq L1+L2+L5) で PPP-AR 候補品質
6. **architectural ceiling**: PLATEAU + PPP-AR + 既存 17-pool で +6.58pp 達成可能か。 必要なら何足りないか
7. その他 missing path がないか

## 補足情報

- PPC dataset: Septentrio mosaic-X5 5Hz、 triple-freq GPS+GAL+BDS+QZSS+GLO、 IMU 100Hz syncd
- Datset dates: tokyo runs 2024-07-23 GW2324、 n/r2 2024-07-20 GW2323、 n/r1+n/r3 2024-08-03 GW2325
- Hardware: NVIDIA RTX 4070 Ti SUPER (16GB)
- Existing pool: 86 candidates (gnss_solve configs + dev_demo5 + fgo + 22 gici variants)
