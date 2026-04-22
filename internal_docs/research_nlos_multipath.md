# 1m 切りに向けた技術調査 (2026-04-08)

## 現状
- Odaiba P50=1.65m, Shinjuku P50=3.13m (PF 100K)
- TDCP, smoother, carrier-phase smoothing は限界到達
- Urban canyon の NLOS + multipath が根本原因

---

## Area 1: Non-Gaussian 尤度モデル (NLOS 対応)

### 1A. Gaussian Mixture Model (GMM) 尤度 ★★★ 最有望
**出典**: Sadiki et al., "GNSS Positioning Enhancement Based on NLOS Multipath Biases Estimation Using Gaussian Mixture Noise"; NASA "Robust Positioning in the Presence of Multipath and NLOS GNSS Signals" (2019)

**コアアイデア**: pseudorange error を LOS (narrow Gaussian) + NLOS (wide Gaussian or uniform) の混合分布でモデル化。

```
p(pr_error) = w_los * N(0, sigma_los^2) + w_nlos * N(mu_nlos, sigma_nlos^2)
```

- LOS: sigma_los ≈ 3-5m, mu=0
- NLOS: sigma_nlos ≈ 20-50m, mu_nlos > 0 (positive bias, reflection always adds delay)
- w_los, w_nlos: C/N0 や elevation から推定

**PF 実装**:
```python
# 現在: log_weight += -0.5 * (residual / sigma)^2
# 変更: log_weight += log(w_los * N(residual, sigma_los) + w_nlos * N(residual - mu_nlos, sigma_nlos))
```

**期待効果**: NLOS 衛星の悪影響を尤度で吸収。Papers report 30-50% improvement in urban.
**実装難度**: 低。CUDA カーネル 1 関数変更のみ。
**リスク**: mu_nlos, sigma_nlos のチューニングが都市依存。

### 1B. Skew-Normal / Skew-t 分布 ★★
**出典**: Wen & Hsu, 3DMA GNSS (PolyU Hong Kong), "likelihood-based ranging using skew-normal distribution"

**コアアイデア**: NLOS error は常に正 (反射=遅延追加) → 非対称分布。skew-normal は正方向に裾が長い。

```
p(residual) = skew_normal(residual; xi=0, omega, alpha)
alpha > 0 → 正方向に偏り (NLOS bias)
```

**PF 実装**: Gaussian と同程度。`log_pdf_skew_normal()` を CUDA に追加。
**期待効果**: GMM より物理的に正しいが、チューニングパラメータ増。
**実装難度**: 低〜中。

### 1C. Switchable Constraints ★
**出典**: Sünderhauf & Protzel (2012), FGO 文脈だが PF にも適用可

**コアアイデア**: 各衛星に on/off スイッチ変数を追加。outlier 衛星は自動的に off。
**PF 実装**: 状態次元が増えるので PF には不向き。**スキップ推奨**。

---

## Area 2: NLOS 検出 / 衛星選択

### 2A. ML-based NLOS 分類 + 重み付け ★★★ 最有望
**出典**: Li et al. (2023), "ML-based GNSS signal classification and weighting", Satellite Navigation; Sun et al. (2025) "stacking ensemble learning"; Xu et al. (2023) "Robust design of ML-based GNSS NLOS detector with multi-frequency features"

**特徴量** (全て gnssplusplus から取得可能):
1. C/N0 (SNR)
2. Elevation angle
3. Pseudorange residual (SPP からの残差)
4. Pseudorange rate vs Doppler の乖離
5. Carrier phase の有無
6. Multi-frequency 一致性 (L1 vs L2)

**手法**: XGBoost or Random Forest で LOS/NLOS を分類 (精度 93-98%)。
NLOS 確率を重みに反映: `weight *= (1 - p_nlos)` or `sigma *= (1 + k * p_nlos)`

**期待効果**: 44% horizontal accuracy improvement (Li et al. 2023)
**実装難度**: 中。学習データが必要 (UrbanNav に 3DMA ラベルあり?)。推論は軽量 (XGBoost)。
**問題**: 学習データのラベル付けに 3D 建物モデルが必要。

### 2B. Residual-based adaptive weighting (学習不要) ★★★ 即実装可
**出典**: Pseudorange acceleration weighting (2024), PMC; consistency checking

**コアアイデア**: 
- **方法1**: SPP 残差が大きい衛星を downweight: `w = 1 / (1 + (residual / threshold)^2)`
- **方法2**: epoch 間の pseudorange 変化率 (acceleration) が異常な衛星を downweight
- **方法3**: RAIM subset testing — 全 n-1 部分集合で位置を計算、一致しない衛星を除外

**期待効果**: 75-80% improvement reported (acceleration weighting)
**実装難度**: 低。Python レベルで即実装可。
**メリット**: 学習データ不要、3D モデル不要、リアルタイム可。

### 2C. 3DMA (3D Mapping-Aided) ★
**出典**: Wen & Hsu PolyU, "Grid-based 3DMA GNSS with clustering"

**コアアイデア**: 3D 建物モデルで各衛星の LOS/NLOS を幾何学的に予測。
**問題**: 3D 建物データが必要。Odaiba/Shinjuku 用の OSM 3D はあるかもしれないが、精度不明。
**実装難度**: 高。**後回し推奨**。

---

## Area 3: マルチパス推定

### 3A. Code-Minus-Carrier (CMC) ベースの補正 ★★★ 最有望
**出典**: "GNSS Code Multipath Mitigation by Cascading Measurement Monitoring Techniques" (2018); "GNSS Urban Positioning with Multipath Mitigation Using Duration Time of TDCP" (2024)

**コアアイデア**: CMC = pseudorange - carrier_phase * wavelength (+ ambiguity + iono)。
carrier phase は multipath にほぼ影響されないので、CMC の変動 ≈ code multipath。
iono-free dual-freq で iono をキャンセルすれば CMC 変動 ≈ pure multipath。

```python
CMC = PR_code - (f1^2*L1*lam1 - f2^2*L2*lam2)/(f1^2-f2^2)
# CMC の epoch 間変動をスムージングして multipath estimate
mp_estimate = lowpass_filter(CMC - mean(CMC))
corrected_PR = PR_code - mp_estimate
```

**期待効果**: 直接的な multipath 除去。
**実装難度**: 低。L1/L2 は既に取得済み (urbannav.py 拡張済み)。
**問題**: Galileo/QZSS は L2 なし → GPS のみ。iono-free CMC は Odaiba (Trimble) で有効。

### 3B. Pseudorange acceleration weighting ★★
**出典**: "Reduction of Multipath Effect in GNSS Positioning by Applying Pseudorange Acceleration as Weight" (2024)

**コアアイデア**: multipath は時間的に変動 → pseudorange の 2 階差分 (acceleration) が大きい衛星は multipath 汚染。

```python
pr_accel = PR(t) - 2*PR(t-1) + PR(t-2)
weight = 1 / (1 + (pr_accel / threshold)^2)
```

**期待効果**: 75-80% improvement (横/縦)
**実装難度**: 低。3 エポック分のバッファだけ。

### 3C. 状態拡張 (per-satellite multipath delay) ★
**コアアイデア**: PF 状態に衛星ごとの multipath delay を追加: state = [x, y, z, cb, mp1, mp2, ...]
**問題**: 状態次元が衛星数に比例 → particle depletion が深刻化。**PF には不向き**。スキップ。

---

## 推奨実装順序

| 優先度 | 手法 | 期待効果 | 難度 | 備考 |
|--------|------|---------|------|------|
| **1** | **2B: Residual adaptive weighting** | 大 | 低 | 学習不要、即実装 |
| **2** | **3B: PR acceleration weighting** | 大 | 低 | multipath 直接対策 |
| **3** | **1A: GMM 尤度** | 中〜大 | 低 | CUDA カーネル変更 |
| **4** | **3A: CMC multipath 補正** | 中 | 低 | GPS only, L2 必要 |
| **5** | **2A: ML NLOS 分類** | 大 | 中 | 学習データ必要 |
| **6** | **1B: Skew-normal** | 中 | 低〜中 | GMM の代替 |

**最初に 2B + 3B を同時実装**（Python レベル、30分で可）→ 効果測定 → **1A (GMM) を CUDA に実装**が最適ルート。

---

## Sources
- [ML-based GNSS signal classification and weighting (Li et al. 2023)](https://link.springer.com/article/10.1186/s43020-023-00101-w)
- [Pseudorange acceleration weighting (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548491/)
- [GNSS Code Multipath Mitigation cascading techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022099/)
- [Robust Positioning NLOS GNSS (NASA 2019)](https://ntrs.nasa.gov/api/citations/20190034171/downloads/20190034171.pdf)
- [GMM NLOS Biases Estimation (Sadiki)](https://www.academia.edu/66853001)
- [3DMA GNSS PolyU (Wen & Hsu)](https://navi.ion.org/content/69/2/navi.515)
- [ML NLOS detector multi-frequency (Xu 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10416230/)
- [Multipath urban canyon 3D ray-tracing (2023)](https://link.springer.com/article/10.1007/s10291-023-01590-7)
- [Innovation-based multipath detection GNSS/INS (2024)](https://arxiv.org/html/2409.03433v1)
- [UrbanNav Related Publications](https://github.com/IPNL-POLYU/UrbanNavDataset/blob/master/docs/RELATED_PUBLICATIONS.md)
