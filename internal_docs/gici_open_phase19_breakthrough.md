# gici-open Phase 19 Breakthrough — PPC2024 PF Ceiling 76.83%

**Last updated**: 2026-05-15 JST
**Status**: 17-variant pool ceiling 確定 / 飽和 (5 連続 marginal+regression)
**Outcome**: Phase 11ep canonical 71.48% → Phase 19al **76.83%** = **+5.35pp / +2516m**, TURING gap **8.77pp**

## 概要

PPC2024 6-run aggregate honest fix-rate を、 既存 libgnss++ ベースの canonical 73.76% (Phase 19d 3-FGO) から、 inuex35/gici-open `forppc2024` branch の **TC FGO output を 17 variant pool に組み込む**ことで **76.83%** に押し上げた。 +3.07pp は PPC 1 セッションでの最大 single-session gain。

ブレイクスルーは 2 段:

1. **Format bug fix** (Phase 19l): 当初 gici-open は PPC selector に `0.00pp redundant` と判定されていたが、 NMEA→PPC pool 変換 script が CSV header に `output_added` 列を書いていなかったため `_rtkdiag_candidate_gate` が常に false で **selector から不可視**だった。 converter に `output_added=1` 列を追加し再走で **+1.62pp**。
2. **Multi-variant pool** (Phase 19s-19al): 単一 default config (esdfix lever arm) に加えて、 lever arm、 AR ratio、 PR/Phase outlier、 SNR、 elevation、 IMU bias、 IMU rotation 等の 17 異なる gici config を pool に併走させ、 PPC selector が per-epoch best を自動選択することで **更に +1.45pp** 取得。

## 累積 trajectory

| Phase | 内容 | Aggregate | Δ from previous |
|---|---|---:|---:|
| 11ep | canonical (libgnss++ + dev_demo5) | 71.48% | baseline |
| 19d | 3-FGO config diversity (v2_gap + v14_snr38 + v17_el25) | 73.76% | **+2.28pp** |
| 19l | gici single (default esdfix) + format-fix | 75.38% | **+1.62pp** |
| 19u | gici 3-variant (+zeroarm +ratio25) | 76.03% | +0.65pp |
| 19x | +loosepr | 76.11% | +0.08pp |
| 19y | +loosephase | 76.21% | +0.10pp |
| 19z | +ratio40 | 76.23% | +0.02pp |
| 19ab | +combo (zeroarm+loosephase 2-knob) | 76.44% | **+0.21pp** (combo extension) |
| 19ak | 16 variants (+combo4 +lprlph +zr +onarm +lowacc +hisnr40/45/30 +hielev) | 76.79% | +0.35pp 累積 |
| **19al** | **17 variants (+imurot)** | **76.83%** | **+0.03pp** (NEW CEILING) |
| 19am | +himuba (sigma_ba=10/sigma_bg=3) | 76.83% | **0.00pp redundant** |
| 19an | +tightpr (max_pr_err=1.5) | 76.81% | **-0.015pp regression** |

5 連続 marginal/regression で pool 飽和確定。

## 17 variants の詳細

ベース binary: `/media/sasaki/aiueo/ai_coding_ws/gici_open_ws/build/gici_main` (inuex35/gici-open `forppc2024` branch、 既 built)。

| Variant | YAML knob | 効果 (single t/r1) | 主に効く run |
|---|---|---:|---|
| default (esdfix) | `gnss_extrinsics: [-0.670, 0.593, -1.216]`, `AR ratio: 3.0`, `max_pr_err: 2.5`, `max_phase_err: 0.06` | baseline 78.36% | 全 run |
| zeroarm | `gnss_extrinsics: [0, 0, 0]` | +0.94pp | t/r1 dominant |
| ratio25 | `AR ratio: 2.5` | +1.46pp | t/r1 best single |
| loosepr | `max_pr_err: 5.0` | +0.08pp | t/r1, t/r2 |
| loosephase | `max_phase_err: 0.12` | +0.10pp | nagoya 系 (n/r3 +0.69pp) |
| ratio40 | `AR ratio: 4.0` (tight) | +0.02pp | t/r1 marginal |
| combo | zeroarm + loosephase (2-knob) | +0.21pp | combo extension breakthrough |
| combo4 | loosepr+loosephase+ratio25+zeroarm | +0.02pp | marginal |
| lprlph | loosepr + loosephase | +0.02pp | marginal |
| zr | zeroarm + ratio25 | +0.002pp | marginal |
| onarm | partial lever arm restoration | +0.05pp | marginal |
| lowacc | `min_acceleration: 0.1` | +0.04pp | marginal |
| hisnr40 | `min_SNR: [40, 40]` | +0.20pp | t/r1 |
| hisnr45 | `min_SNR: [45, 45]` | +0.10pp | t/r1 marginal |
| hisnr30 | `min_SNR: [30, 30]` | +0.05pp | marginal |
| hielev | `min_elevation: 12.0` | +0.04pp | marginal |
| imurot | `body_to_imu_rotation_std: 20.0` | +0.03pp | n/r3 +0.24pp |

**Combo rule (重要)**: 2-knob combo は **異なる run を target する knob 組合せ**で新 diversity。 zeroarm (t/r1 effective) + loosephase (nagoya effective) = +0.21pp (effective)。 同 run target 同士 (zr = 両 t/r1) は redundant。 4-knob combo は too "averaged" で個別効果 dilute。

## Per-run 結果 (Phase 19al)

| Run | canonical (11ep) | Phase 19l (gici single) | Phase 19al (17-variant) | Δ 11ep → 19al |
|---|---:|---:|---:|---:|
| tokyo/run1 | 72.07 | 78.33 | **84.04** | **+11.97pp** |
| tokyo/run2 | 85.67 | 86.26 | 86.47 | +0.80pp |
| tokyo/run3 | 81.60 | 81.67 | 81.77 | +0.17pp |
| nagoya/run1 | 66.53 | 67.47 | 67.69 | +1.16pp |
| nagoya/run2 | 43.98 | 44.02 | **44.21** | +0.23pp |
| nagoya/run3 | 67.01 | 67.25 | 68.46 | +1.45pp |
| **aggregate** | **71.48** | **75.38** | **76.83** | **+5.35pp** |

t/r1 が dominant gain (+11.97pp = +1235m on 10318m total)。 gici TC FGO + IMU integration が urban canyon multipath で libgnss RTK の frequent LAMBDA failure を埋める。

## Format bug の詳細 (Phase 19l breakthrough)

**問題**: `experiments/exp_ppc_ctrbpf_fgo.py:1606-1624` の `_rtkdiag_candidate_gate` が `row.get("output_added", "0") == "1"` を要求するが、 旧 `nmea_to_ppc_pool.py` が出力した CSV header は `final_valid` のみで `output_added` 列なし → 全 epoch で gate false → gici 候補が pool 入っても **selector から不可視**。

**修正**:
```python
# nmea_to_ppc_pool_esdfix.py
writer.writerow([..., f"{stds[0]:.3f}", f"{stds[1]:.3f}", f"{stds[2]:.3f}", 1])  # output_added=1
header = [..., "esd_std_pe", "esd_std_pn", "esd_std_pu", "output_added"]
```

**確認**: Phase 19l 再走で 6 run 全 positive、 aggregate **+1.6211pp**。 過去の "gici 0.00pp redundant" 結論 (project_gici_open_phase19e_integration.md) を完全に取り消し。

## 構造的限界

### n/r2 (nagoya/run2) ceiling 44.21%

全 19 variants で 44.21% に張り付き、 全 knob (lever arm/AR/SNR/IMU/PR/Phase/window/elevation) で改善なし。 PPC selector は per-epoch optimal を既に達成済、 残 epoch (2645m / 4741m = 55.79%) は libgnss++ + gici 全 candidate でも recoverable でない。

n/r2 を 70% に上げると aggregate +2.6pp 期待だが、 これには architectural breakthrough が必要:
- multi-base / network RTK
- PPP-AR (CLAS / madoca PR #55 経由)
- city-model NLOS rejection (BVH ray tracing、 PR #55 で infrastructure 投入済)
- triple-frequency LAMBDA cascade (UWL E5a-E5b、 dataset 制約で不可)

### 飽和の signature

Phase 19am (+himuba) で aggregate 完全同値 (pass/total identical)、 Phase 19an (+tightpr) で 7m 損失 (t/r1)。 PPC selector pool に追加 candidate を出しても per-epoch best は変わらない。 個別 variant が独立な diversity を提供できなくなった = pool が **per-epoch optimal を absorb 済**。

## Failure path 記録

| Variant / Phase | Δ aggregate | 理由 |
|---|---:|---|
| loosenl (Phase 19aa, NL fixation 0.9→0.7) | -0.09pp | wrong fixes 増 |
| tcgif lever arm `[0.31, 0, 0.55]` (Phase 19v) | -0.004pp | gici が早期 exit、 partial coverage |
| residual select_mode on n/r2 (Phase 19ah) | -10.66pp | catastrophic regression |
| tightpr (Phase 19an, max_pr_err=1.5) | -0.015pp | 既存 \|res\|<3m gate と重複、 good obs を削る |

## Pipeline 全体

```
PPC2024 dataset (RINEX + IMU bin)
    │
    ├──→ libgnss++ 6 variants (canonical Phase 17 pool)
    │      └─→ experiments/results/libgnss_rtk_pos_v5/
    │      └─→ experiments/results/libgnss_diag_phase10/dev_demo5_trusted_o3/
    │      └─→ experiments/results/libgnss_diag_phase10/demo5_continuous_nojump/
    │
    ├──→ GTSAM FGO 3-config (Phase 19a-d)
    │      └─→ experiments/results/libgnss_diag_phase10/fgo_{v2_gap, v14_snr38, v17_el25}/
    │
    └──→ gici-open TC FGO 17 variants (NEW, Phase 19l-al)
           gici_open_ws/build/gici_main
            │
            └─→ test_<run>_<variant>.txt  (NMEA + GPESD)
                  │
                  └─→ nmea_to_ppc_pool_esdfix.py  (CSV + .pos)
                        │
                        └─→ experiments/results/libgnss_diag_phase19/gici_full_<variant>/
                              │
                              └─→ exp_ppc_ctrbpf_fgo.py PPC selector
                                    (RBPF-velKF + DD + gate + hybrid + rtkdiag_pf)
                                    │
                                    └─→ 76.83% aggregate (Phase 19al)
```

## 主要ファイル

- `/media/sasaki/aiueo/ai_coding_ws/gici_open_ws/build/gici_main`: TC FGO binary
- `/media/sasaki/aiueo/ai_coding_ws/gici_open_ws/option/{tokyo,nagoya}{1-3}_tc_run_<variant>.yaml`: 17 variant × 6 run YAMLs
- `/media/sasaki/aiueo/ai_coding_ws/gici_open_ws/nmea_to_ppc_pool_esdfix.py`: format-fixed converter (output_added=1 列追加版)
- `experiments/results/libgnss_diag_phase19/gici_full_<variant>/`: 17 PPC pool candidate dirs
- `experiments/exp_ppc_ctrbpf_fgo.py`: PPC selector (PR#58 fallback_mode=hybrid 修正済)
- `/tmp/run_phase19al_17v.sh`: 6-run PF aggregator script (latest canonical best)

## 再現コマンド

```bash
cd /media/sasaki/aiueo/ai_coding_ws/gnss_gpu
source .venv/bin/activate

# (1) gici binary 6-run × 17-variant 生成 (~20-30 min each、 並列で ~30-60 min)
cd /media/sasaki/aiueo/ai_coding_ws/gici_open_ws
for v in zeroarm ratio25 loosepr loosephase ratio40 combo combo4 lprlph zr onarm lowacc hisnr hisnr45 hisnr30 hielev imurot; do
  for r in tokyo1 tokyo2 tokyo3 nagoya1 nagoya2 nagoya3; do
    nohup ./build/gici_main option/${r}_tc_run_${v}.yaml > /dev/null 2>&1 &
  done
  wait
done

# (2) NMEA → PPC pool 変換 (output_added=1 含む format-fix)
for v in <variant>; do
  for run in tokyo1 tokyo2 tokyo3 nagoya1 nagoya2 nagoya3; do
    case $run in tokyo*) city=tokyo;; nagoya*) city=nagoya;; esac
    n=${run##$city}; key=${city}_run${n}
    OUT=/media/sasaki/aiueo/ai_coding_ws/gnss_gpu/experiments/results/libgnss_diag_phase19/gici_full_${v}
    mkdir -p $OUT
    .venv/bin/python nmea_to_ppc_pool_esdfix.py test_${run}_${v}.txt $OUT/${key}_full \
      --gps-week <week> --utc-date <date>
  done
done

# (3) PF aggregator (Phase 19al 17-variant)
bash /tmp/run_phase19al_17v.sh
```

GPS week / UTC date map (`forppc2024` dataset):
- tokyo/run{1,2,3}: week 2324, date 2024-07-23
- nagoya/run1: week 2325, date 2024-08-03
- nagoya/run2: week 2323, date 2024-07-20
- nagoya/run3: week 2325, date 2024-08-03

## 関連メモ / プラン

- `internal_docs/plan.md` §0 (TURING 残戦略)
- 記憶: `~/.claude/projects/-media-sasaki-aiueo-ai-coding-ws-gnss-gpu/memory/project_gici_breakthrough_2026_05_14.md`
- 過去 (誤): `~/.claude/projects/-media-sasaki-aiueo-ai-coding-ws-gnss-gpu/memory/project_gici_open_phase19e_integration.md` (format-bug 発覚前の "gici 0.00pp redundant" 結論を含む、 取り消し済)
- PR#58 regression fix: `~/.claude/projects/-media-sasaki-aiueo-ai-coding-ws-gnss-gpu/memory/project_ppc_pf_script_regression_2026_05_14.md`

## 残 path (TURING gap 8.77pp 完全 close は数月 task)

| Path | 推定 +pp | 工数 | 状態 |
|---|---:|---|---|
| n/r2-specific NLOS gate (BVH + city model) | +1-2pp | 1-2 週 | data fetch denied、 OSM building footprints で代用必要 |
| CLAS / madoca PPP-AR (PR #55 経由) | +2-3pp | rebase 1-2 週 | submodule に branch 投入済、 my branch と共通祖先なし |
| multi-base / network RTK | +1-2pp | 2-4 週 | 別 base station data 取得必要 |
| triple-freq LAMBDA cascade (UWL E5a-E5b) | +3-5pp | 2 週 | dataset に E5b 観測なし、 構造的に不可 |
| gnss_gpu native FGO (gici 不要化) | architectural | ~月 | long-term |

直近、 推奨 ROI 順: **CLAS/madoca PPP-AR → BVH NLOS (city data 待ち) → architectural**。
