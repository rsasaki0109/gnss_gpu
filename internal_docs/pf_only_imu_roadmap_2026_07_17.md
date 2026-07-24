# PF-Only + IMU Roadmap (FGO抜き) — 2026-07-17

## 0. 方針

FGO(スライディングウィンドウLM/GTSAM系)を一切使わず、Particle Filterを突き詰める。
Rao-Blackwellization(粒子ごとのKF)は「FGOではない」ので全面的に使う。
IMUは粒子ごとのKF伝播モデル(プレインテグレーション)として複合する。

**根拠(文献調査 2026-07-17):**
- Niimi et al. (PLANS 2025 / RA-L 2026): RBPF(位置=粒子、速度/クロックドリフト=粒子ごとKF、粒子ごとNLOS棄却、Student's-t)がAR無しで RTKLIB RTK超え(3D <0.3m率 68.5% vs 52.7%)。現状のPF-without-FGOのSOTA。
- Ng et al. 2024 (J.Nav): Canary Wharf車載でグリッドフィルタ(~19m)がFGO変種(~22.5m)に勝った実例あり。仮説ドメインのフィルタはFGOに構造的に負けない。
- Wen et al. 2021: UrbanNav系でTC-EKF 8.03m → TC-FGO 3.64m(ただし計算 ~1000倍)。**PFがUrbanNav deep-urbanで平均≤3.5mを出せばTC-FGOの公表値を実時間クラスで抜く。**
- 文献に「PF vs FGOのUrbanNavガチンコ比較」は存在しない。「GPUで粒子ごとBVHレイトレース尤度」も存在しない。両方うちの資産で埋められる空白。
- 本キャンペーンの結論(inuex35_tc_fgo_benchmark.md:251): inuex35の強みはFGOの精巧さではなく **RTK FIX供給 + エポック横断IMUメモリ**。→ PFで攻めるべきはこの2点。

## 1. ターゲット数値

| ベンチ | 指標 | 現状(PF単体) | 目標 |
|---|---|---|---|
| PPC Tokyo run1/2/3 | <50cm_full% (3D) | 3.0 / 1.2 / 3.2 | inuex35 (56.7/69.9/67.9) 超え。理想は RB-FGO-PF (59.6/78.7/78.1) にFGO無しで肉薄 |
| UrbanNav deep-urban | 平均水平誤差 | — | ≤3.5 m (TC-FGO公表値 3.64m 超え、実時間) |
| GSDC2023 | score | — | 参考トラック(PFのtop-3入り前例なし) |

## 2. フェーズ計画

### WP21: IMU-RBPF コア(主戦場)
IMUプレインテグレーションを粒子ごとKFの伝播モデルに。
- `tc_fgo.py` のプレインテグレーション(Δp, Δv, ΔR, 共分散)を抽出して独立モジュール化(FGO非依存の数学)。
- `PFDeviceState` の粒子ごとKF状態を拡張: 現行 `{mu_v, Sigma_v}` → `{v, clock drift, accel/gyro bias}`(姿勢は当面 `INSEKF`/相補フィルタで窓外伝播、粒子ごとには持たない — Schön/Gustafssonのmarginalized-PFテンプレート)。
- 予測: `x_new ~ N(x + R(θ)·Δp_preint, Q_preint + dt²Σ_v)` をCUDAカーネル化(`predict.cu` 拡張)。
- Doppler KF更新(`pf_device_doppler_kf_update`)は既存。TDCPを粒子ごとKF測定に追加。
- **ゲート: PPCでIMU有無のablation。<50cm_full% が有意に動くこと。**

### WP22: 粒子ごとNLOS棄却 + C/N0駆動GMM尤度
Niimi方式 + Gupta&Gao GMM + 検証済みC/N0予測器の統合。
- 既存の per-particle NLOS threshold + Huber(`pf_device_weight`)を、Niimi式「各粒子の仮説位置で残差評価→粒子ごとに棄却集合を変える」完全形に拡張。
- GMMカーネル(`pf_device_weight_gmm`)の混合重み `w_los` を衛星ごとにC/N0予測器(ccaf92cで検証済み)+仰角+残差一貫性で駆動(Wang 2022: 0.53m の特徴量セット)。
- BVHレイトレースLOS/NLOS事前(`pf_weight_3d_bvh` + UTD回折)をPPC/UrbanNav本番経路に接続(PLATEAUメッシュ既存)。**「粒子ごとレイトレース尤度のGPU PF」は文献に存在しない=論文ネタ①。**

### WP23: 搬送波位相 + AR(盆地RBPF、格上げ 2026-07-17)
ARを一級市民に。ただしFGO条件付けではなく**粒子ごと(盆地ごと)KF条件付け**で。
- 整数アンビギュイティ盆地を粒子の離散状態として運ぶ(rbpf_fgo_design.md のFGO条件付けを**KF条件付けに置換** — RB-FGO-PFで実証済みの構造をFGO抜きで再現)。
- GPU batch-LAMBDA(WP15, `lambda_batch.cu`)でtop-K整数候補生成、累積周辺尤度重み、盆地層別リサンプリング。
- **FIX宣言 = 盆地事後質量 γ>0.99**(校正済み判定。単一エポックのratio testを全廃)。false-fixガードはrbpf_fgoの教訓(DDPR投票、コヒーレントシフト対策)を移植。
- 入口として Suzuki ICRA 2024 の multiple-update(尤度の鋭さ順の逐次更新)を既存 DD carrier AFV カーネル(`pf_device_weight_dd_carrier_afv`)の呼び出し順序として実装(盆地化前のフロート段でも効く)。
- 多様性維持にSVGD/Reservoir-Stein(既存)を検討。
- ARターゲット: fix% / FixRMS で RB-FGO-PF(43-75% / 0.10-0.15m)に肉薄、false-fix ≤1%。

### WP24: 固定ラグ平滑化 + 出力ポリシー
「エポック横断メモリ」をFFBSi/fixed-lag(既存 ~40モジュール)で。IMU入りの遷移密度になって初めてFFBSiが効くはず(過去のFFBSi単体0%改善はCVモデルの遷移が緩すぎたのが一因と仮説)。
- 固定ラグL=2〜10sでのオンライン平滑化 + 出力(実時間性を捨てない範囲)。
- ゲート: FFBSi milestone の再評価(IMU遷移密度で mean delta が正に転じるか)。

### WP25: ベンチ公開 + 論文/OSS
- 論文①: GPU粒子ごとBVHレイトレース尤度PF + UrbanNav/PPCでのPF vs FGO head-to-head(両方とも文献に不在)。
- 論文②(後続): GNSS疑似距離のend-to-end微分可能PF(deep-sets残差エンコーダ→粒子ごと尤度、stop-gradient resampler、PyDPF足場)— これも不在。
- OSS: 「保守されてる疑似距離ドメインPF」はOSSに一つも無い(gnss_lib_py/taroz/IPNL全滅確認済み)。この空白がリポジトリのポジション。

## 3. リスクと対策
- 鋭い搬送波尤度が粒子多様性を殺す → multiple-update順序制御 + SVGD移動。
- fp32重みスキャンは >1e5粒子で偏る(Murray et al.)→ 既定のMetropolis resamplingを維持、scanはfp64。
- 姿勢を粒子に入れると次元爆発 → 窓外INSEKF伝播で回避、必要になったらheadingのみ粒子化。
- IMUバイアス可観測性(GNSS劣化区間)→ バイアスは粒子ごとKF内でrandom-walk、事前を強めに。

## 4. 主要参考文献(取得検証済み)
- Niimi et al., RBPF without IAR, PLANS 2025, arXiv:2506.03537 / RA-L 2026, DOI 10.1109/lra.2025.3641151
- Suzuki, Multiple Update PF (carrier phase), ICRA 2024, arXiv:2403.03394
- Gupta & Gao, GMM-likelihood PF for multi-fault, T-ITS 2021 / EURASIP JASP 2024, arXiv:2101.06380
- Zhong & Groves, Multi-epoch 3DMA Bayesian filtering, NAVIGATION 69(2) navi.515, 2022
- Ng, Zhong, Groves, Hsu, Grid-based 3DMA + FGO, J.Nav 77, 2024(グリッドがFGOに勝つ例)
- Wen et al., FGO vs EKF, NAVIGATION 68(2):315, 2021(TC-FGO 3.64m の目標値)
- Murray, Lee, Jacob, Parallel resampling, JCGS 2016, arXiv:1301.4019
- Koide et al., MegaParticles (1M粒子SVGD), ICRA 2024, arXiv:2404.16370
- Chen & Li, DPF survey, arXiv:2302.09639; PyDPF arXiv:2510.25693
- Schön, Karlsson, Gustafsson, Marginalized PF in practice, IEEE Aerospace 2006
