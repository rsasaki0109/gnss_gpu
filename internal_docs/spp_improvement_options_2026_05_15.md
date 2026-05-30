# SPP 精度改善 options — GPT Pro 諮問用 (2026-05-15 夜)

## 背景

- **タスク**: PPC2024 OFFICIAL (per-run-averaged path-weighted pass@50cm) **90%** が target
- **直近 breakthrough**: Path F (LightGBM path-weighted ranker + Viterbi sequence) で **LORO sim 85.25% / 実装 t/r1 90.84%, t/r2 95.08%** (まだ続行中)
- **次の path 候補**: ユーザー指摘 **「SPP の精度改善もありかも」** = 下流の hybrid_pu / RTKDIAG / ranker 全てに乗算的に効く基盤改善

## 現状の SPP (gnss_gpu repo)

- 主経路: `experiments/exp_urbannav_baseline.py::run_wls()` (PPC pipeline の `wls_positions` seed)
  - `MultiGNSSSolver` (CUDA、`src/positioning/multi_gnss.cu`) で multi-constellation WLS
  - 各 system に bias clock 1 つずつ (GPS, GLONASS, Galileo, BeiDou, QZSS)
  - 失敗時 GPS-only fallback (`_solve_single_system_epoch`)
  - 10 km jump filter
  - Per-system weight scaling (`weight_scale_by_system`)
- 補正 (`python/gnss_gpu/spp.py`、 153 行):
  - Sat clock 2-pass iteration
  - Earth rotation (Sagnac)
  - Tropo: **Saastamoinen** (simplified、 lat/alt/el のみ、 T/P/e は固定)
  - Iono: **Klobuchar** (broadcast nav)
  - TGD (group delay)
- 別オプション `python/gnss_gpu/robust_spp.py` (160 行): IRLS (Huber/Cauchy)、 threshold=15m、 ただし PPC pipeline で default 経路ではない
- 観測 weight: elevation-based ($1/\sin^2 \text{el}$ 系) + per-system scaling

## SPP の弱点 (推定/観測)

- **`candidate_vs_spp_m` ranker importance が下位** (47K vs cluster_size 9.9M)
  - つまり「SPP との距離」が selector 信号として弱い = SPP 精度が dm 単位の差別化を許さない
- **Klobuchar iono** は半経験式、 urban canyon multipath + iono storm に対して残差 1-3m
- **Saastamoinen Tropo** は天頂遅延 ~2.3m → mapping function ($1/\sin el$) で低仰角時に増幅、 surface T/P/e が固定
- **TGD** は GPS broadcast 値のみ、 multi-constellation の per-signal bias (DCB) を per-frequency まで分解してない
- **Receiver clock** は per-system 1 つ、 per-signal (L1C/A vs L1C, etc.) bias 未モデル
- **Doppler 情報未活用** (PR のみ、 carrier doppler は smoothing せず破棄)
- **Pseudorange smoothing 無し** (epoch-by-epoch、 Hatch filter 不在)
- **Multi-frequency IFLC** (L1+L2 or L1+L5) は **存在せず** L1 のみ
- **Outlier rejection は 10km jump filter のみ** (sigma-based IRLS は `robust_spp.py` 単独実装で default 経路非使用)

## 既知の Path 候補

### A. Doppler / Pseudorange smoothing (Hatch filter)
- carrier phase 由来の Doppler を使った PR smoothing
- short-term noise + multipath を ~1 order 減らせる (一般文献)
- 実装難易度: 中、 carrier の cycle slip 検出が必要
- 期待: +0.5-2m SPP improvement → ranker `candidate_vs_spp_m` 改善 → +0.1-0.5pp OFFICIAL

### B. IRLS outlier rejection を default 経路に
- `robust_spp.py` IRLS (Cauchy c=15m) を `MultiGNSSSolver` の後段に挿入
- 実装難易度: 低 (既に code はある、 wrapping のみ)
- 期待: 残差 cleaning で +0.5-1m、 +0.1-0.3pp

### C. Multi-frequency Iono-Free (IFLC)
- L1+L2 (GPS) or L1+L5 (Galileo/GPS) で iono 残差 elimination
- IFLC noise が 2.5-3x 増えるが iono delay 完全消去
- 既存 dataset で L2/L5 available か要確認 (PPC2024)
- 実装難易度: 高、 SPP 全 rewrite 必要
- 期待: iono storm 時の +1-3m gain、 通常時は noise 増で trade-off

### D. MGEX precise orbits/clocks
- IGS final products (orbits ~2.5cm RMS、 clocks 75ps RMS = ~22mm)
- broadcast nav の vs IGS final の差は orbits ~50cm、 clocks ~1ns = ~30cm
- 認証必要: CDDIS Earthdata Login (今夜ユーザー pause)、 BKG/IGN/ESA (試した、 認証 wall)
- 実装難易度: 中 (RINEX SP3/CLK parser + nav→SP3 interpolation)
- 期待: SPP +30-60cm gain、 ただし near-real-time products は ~6h 遅延 (PPC2024 は post-processed = OK)

### E. Tropo Saastamoinen → GPT2w / VMF1
- mapping function VMF1 + numerical met (NWP-based) で精度 +1-3cm at high elevation、 低仰角 +10-30cm
- 実装難易度: 中 (VMF1 grid file 必要)
- 期待: SPP +5-30cm (cm-dm scale)、 base for 他の improvement

### F. Per-signal DCB (Differential Code Bias)
- MGEX DCB products で L1C/A vs L2C, etc. bias を per-sat per-day で補正
- 認証必要 (D と同じ MGEX wall)
- 実装難易度: 低 (DCB CSV テーブル look up)
- 期待: SPP +5-20cm

### G. PLATEAU NLOS-based weight inflation at SPP level
- 既に extract した NLOS mask (`/tmp/tokyo_run1_nlos_per_epoch.csv` 等) を SPP の per-sat weight に soft-multiply
- NLOS sat は weight /= 3-5
- 実装難易度: 低 (run_wls の weights 引数前で multiply)
- 期待: urban canyon (n/r2, t/r1) で +0.5-2m、 +0.3-1pp OFFICIAL

### H. Receiver clock per-signal (RNX 3 multi-signal)
- L1C/A, L1C, L1P, etc. ごとに小さい RX clock bias
- 実装難易度: 中 (state vector を per-signal に拡張)
- 期待: +5-20cm、 multi-signal RX 限定

## GPT Pro への問い

1. **優先順位**: A-H のうち、 90% target に最短で効くのはどれ? それぞれ "実装工数 vs 期待 +pp" の matrix を引いてほしい
2. **Doppler smoothing (A)** は cycle slip detection の robustness が肝。 urban canyon (multipath) で安全に使う recipe は? (例: Vondrak filter? GFC indicator?)
3. **MGEX without CDDIS (D)**: 認証なしで取れる alternate source は? (BKG/IGN 試したが 404/403)、 ESA RTS stream?
4. **PLATEAU NLOS at SPP (G)** は実装が一番軽い。 NLOS sat を完全除外 vs weight /= k のどちらが robust? (既に hard-mask は -5.24pp という結果あり、 SPP layer なら違うかも)
5. **既存 ranker (Path F) との相乗効果**: SPP 改善が ranker feature `candidate_vs_spp_m` を一段強化する。 ranker re-train が必要か、 既 model で十分か?
6. **PPC2024 dataset 固有の制約**: L1 only か L1+L2 か (要確認)、 Galileo/QZSS の L5 available? このために IFLC (C) は使えるか?
7. **基盤改善 vs selector 改善**: Path F (selector) が +1.83pp 達成、 PLATEAU C は -2.6pp で失敗。 SPP layer は selector 改善と "vertical" な独立軸、 両方 stack できる? それとも ranker が SPP 改善を吸収してしまう?

## Path F 実況 (2026-05-15 19:35 時点)

- t/r1 90.84% (LORO sim 89.59% から +1.25pp)
- t/r2 95.08% (LORO sim 94.32% から +0.76pp)
- 残: t/r3, n/r1, n/r2, n/r3
- LORO sim aggregate 85.25% に対して、 実装が **+0.76-1.25pp 上振れ** している = pipeline filter (output_added 等) が ranker pick を更に良くしている兆候

## 添付ファイル

- 実装: `experiments/exp_ppc_ctrbpf_fgo.py` (selector + ranker integration)
- 訓練: `experiments/train_selector_ranker.py`
- 特徴抽出: `experiments/extract_selector_training_features.py`
- 予測: `experiments/results/selector_ranker_predictions.csv` (214MB, 3.28M rows)
- 模型: `experiments/results/selector_ranker_model.txt` (LightGBM 200 round binary)

## Feature importance (final model)

```
cluster_size_50cm: 9910939  <- 12x dominant
status:             802227
label:              622536
abs_max:            500545
rms:                363808
sats:               328141
ratio:              290991
dist_to_median_m:   260037
n_candidates_in_epoch: 245798
update_rows:        229806
baseline_m:         193627
rank_by_rms:        189717
candidate_jump_m:    50630
candidate_vs_spp_m:  47056   <- SPP 信号が weak!
spp_sats:            25446
spp_pdop:            21221
pdop:                 1052
spp_valid:               0
```

`candidate_vs_spp_m` の低 importance こそが「SPP 改善で ranker をもう一段引き上げられる」根拠。
