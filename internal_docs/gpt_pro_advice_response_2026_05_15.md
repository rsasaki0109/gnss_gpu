# GPT Pro 先生からの回答 — PPC2024 残 8.77pp closing 戦略

**日時**: 2026-05-15
**前提**: 質問書 [`gpt_pro_advice_request_2026_05_15.md`](gpt_pro_advice_request_2026_05_15.md)
**結論**: **「Fix=4 candidate を増やす」ではなく「正しい軌跡候補を出す → Fix status と別に採用 → ambiguity/NLOS/IMU を discrete hypothesis として扱う」が主戦場**。

---

## 0. 最重要 sanity check (即実行必須)

**公式 PPC2024 score は Fix フラグを見ない**。 提出座標の **3D 50cm 以内** だった走行距離割合、 **6 走行の距離割合平均**、 停止時除外、 リアルタイム想定 (未来観測不可) という規定。

私の現 selector が **`Fix=4 only` を採用基準** にしているなら、 それは **競技ルール上不要な自己制約**。 Float / DR / velocity-integrated / map-aided pose を「Fix ではないから捨てる」のは score 上の損失。

**Action 0 (1日以内)**: 私の per-run 値を単純平均すると 76.83% にならない。 local aggregate metric が公式 (6走行距離割合平均) と完全一致しているか **要確認**。 ここで mismatch あれば、 TURING 85.6% との gap の意味が変わる。

---

## 1. Q1-Q10 への回答 (要約)

### Q1: n/r2 44.21% は物理限界か?

**物理限界ではなく、 現 candidate family の構造限界**。 全 19 variants が同じ ceiling に張り付くのは、 連続 knob (AR ratio / lever arm / SNR / elevation / IMU bias sigma) では n/r2 の failure mode を変えられていないことを示す。

切り分け 5 本:
1. **truth-seeded RTK/FGO**: truth 近傍から初期化、 LAMBDA / residual validation が通るか
2. **truth-positioned ambiguity observability test**: truth 位置で DD carrier phase residual を計算、 integer N に近いか
3. **status-free oracle**: 既存 19 candidates の Fix flag 無視で 0.5m 判定、 oracle vs selected の gap
4. **current-candidate oracle vs selected gap**: 0.5pp 以下なら selector 飽和、 2pp 以上なら改善余地
5. **LOS geometry lower bound**: truth + city model で usable LOS dual-freq sat count、 PDOP、 cycle-slip rate

### Q2: 既存 selector 改善 vs 新 candidate 追加?

**新 candidate type 追加が本命**。 ただし Fix=4 gate 撤廃 + Float/DR rescue は selector 改善ではなく、 **実質的に新 candidate type 追加**。

| 改善対象 | 期待 |
|---|---|
| 同じ gici variants 間の ranker 改善 | 小 |
| Fix=4 gate 撤廃、 Float/DR/velocity/FGO-predicted pose を評価対象に | 中〜大 |
| partial AR / sampling AR / discrete ambiguity candidate | **大** |
| soft NLOS / 3DMA / map likelihood candidate | 中 |
| PPP-AR / MADOCA independent candidate | 中 |

### Q3: ROI 順

| 期間 | 最優先 | 期待値 |
|---|---|---|
| **1週間** | scoring 再確認、 Fix=4 gate 撤廃、 velocity/IMU bridge、 candidate oracle 診断、 constellation subset AR | **+1〜4pp** |
| 1か月 | native FGO / gici 改造で Turing/3rd-place 型 factor set + partial/sampling AR | +3〜7pp |
| 半期 | ambiguity + LOS/NLOS + IMU bias を discrete-hypothesis 化した統合 FGO/RBPF、 soft 3DMA、 MADOCA candidate | +6〜10pp |

CLAS/MADOCA は **独立 candidate として有益だが、 n/r2 の 44%→80% を単独で救う主役ではない**。 city-model NLOS は **hard gate ではなく soft likelihood / variance inflation / multi-hypothesis** として入れるなら価値あり。

### Q4: 144 通り gici knob を機械的に試すべきか?

**やめるべき**。 残り 127 通りのうち独立 diversity を生むのは 1〜5%、 期待 0〜0.3pp。 理由は knob が独立でないため (AR ratio / SNR / elevation / outlier threshold は同じ「衛星を少し削る / ambiguity を少し厳しくする」方向に相関)。

例外: **constellation subset / satellite leave-one-out** は YAML knob より ambiguity failure mode を変える。 2位解法は GQEBR → GQEB → GQER → GQB → GQR → GQ subset を試している。

### Q5: IMU dead reckoning across loss-of-lock は有効か?

**有効。 ただし「Fix を増やす」ではなく「提出位置を 0.5m 以内に保つ」用途**。

2位チーム: 最後 fixed pose + 速度から次 epoch 予測、 **東京1 3秒 / 東京2 4秒 / 東京3 7秒 / 名古屋1/2/3 6秒** の velocity integration。

実装方針: Fix=4 only gate をやめて以下に分離:

| 種別 | RTK status | selector status |
|---|---|---|
| RTK fixed | Fix | high-confidence GNSS |
| RTK float but FGO/NIS/Doppler consistent | Float | rescue candidate |
| last-fix + Doppler/IMU/NHC bridge | None | short-gap bridge (2-8秒) |
| IMU-only long bridge | None | low confidence / decay |

橋渡し: **2〜8秒から開始**。 30秒級は車両拘束 (NHC / heading / ZUPT / 道路 / 高度) なしでは 50cm 維持厳しい。 TURING の factor list にも ZUPT / heading / non-holonomic constraint が入っている。

### Q6: CLAS/MADOCA PPP-AR で n/r2 は救えるか?

**主役にはならない。 独立 candidate として 1〜3pp**。

MADOCA-PPP open-sky 性能: kinematic 収束 1800s 以下、 horizontal 12cm / vertical 16cm (95%)。 2024 report で iono correction ON で 360-510s 収束例あり。 ただし **urban canyon (n/r2 相当) で同じ収束性は期待できない**。

CLAS: cm 級補強だが L6D 専用受信機 + 10-20秒遅延、 補助的利用が想定。

入れるなら、 **PPP-AR fixed / PPP float / RTK fixed / DR bridge を並列 candidate にして selector が epoch ごと選ぶ形**。

### Q7: city-model NLOS rejection で n/r2 は救えるか?

**hard rejection では救えない。 soft weighting / multi-hypothesis なら救える可能性**。

OSM footprint + Manhattan height extrusion は **variance inflation** には使えるが、 **hard LOS/NLOS 判定には粗い**。 必要精度:

| 用途 | 必要地図精度 |
|---|---|
| open/urban 判定 | footprint 5〜10m、 height 10m でも可 |
| NLOS soft down-weight | footprint 2〜5m、 height 5〜10m |
| **hard satellite exclusion** | **footprint 1〜2m、 height 2〜5m、 façade 精度必要** |
| ray-tracing correction | façade geometry、 反射面、 受信機横位置精密 |

n/r2 で hard-gating regression が出ているので、 次は **NLOS 確率で variance を膨らませる**:

$$\sigma^2_{\Phi,s} \leftarrow \sigma^2_{\Phi,s} \cdot \exp(\alpha \cdot p_{\text{NLOS},s})$$

### Q8: TURING 85.6% との 8.77pp の breakdown 推定

| 差分要素 | 推定 gap |
|---|---:|
| Fix=4 gate 由来の取りこぼし、 Float/DR/velocity rescue 不足 | **1〜3pp** |
| course-wise / epoch-wise AR ratio、 QZS/system variance、 fixed residual validation | 1〜3pp |
| ZUPT / heading / NHC / relative ambiguity/frequency factor 不足 | 1〜3pp |
| partial AR / constellation subset / satellite exclusion bank | 1〜2pp |
| nonparametric ambiguity posterior sampling | 1〜3pp |
| soft NLOS / 3DMA / map likelihood | 0.5〜2pp |
| PPP-AR / MADOCA independent candidate | 0.5〜2pp |

重要: TURING だけでなく 2位・3位も「単純 RTK fix 率最大化」ではない。 2位 = velocity integration + satellite subset、 3位 = **ambiguity posterior sampling + FGO prediction / Float dual-seed**。

### Q9: 17 variants pool は過剰 / 不足 / 適正?

**同一 candidate family としては適正〜過剰。 全体としては不足**。

検証順:
1. leave-one-out selector score → 消しても変わらない variant は noise
2. pairwise candidate distance (0.5m 以上離れる epoch 数) → 少なければ重複
3. AR accepted set Jaccard → 衛星/ambiguity が同じなら重複
4. oracle saturation curve (5→10→17 candidates) → 飽和なら過剰
5. selected-vs-oracle gap → oracle 高ければ selector 問題
6. cluster pruning → 代表 variant のみで score 維持なら過剰

次に足すべきは **別 knob ではなく、 別 hypothesis**: Float/DR trajectory、 constellation subset、 partial AR、 sampling AR、 PPP-AR、 soft NLOS。

### Q10: PPC 4-fix gate を緩める impact

**最も見落とされている angle**。 公式 score は Fix flag ではなく 3D 50cm 以内。

やるべき: **「Float を無条件採用」ではなく、 RTK status と submission confidence を分離**。

$$\text{accept}(x_t) = [\text{NIS}_{\text{GNSS}} < \tau_1] \land [\|\Delta x_t - \Delta x_{\text{IMU/Doppler}}\| < \tau_2] \land [\text{cov}_{xy} < \tau_3] \land [\text{jump} < \tau_4]$$

この gate を通った Float/DR を Fix=4 と偽装する必要はなく、 selector 内部で「scoreable candidate」として扱う。 特に **n/r2 では、 最後 fixed anchor から 2〜6秒の bridge を拾うだけで距離 score に効く** (2位資料 6秒 bridge が外部証拠)。

---

## 2. 1週間プラン (期待 +1〜4pp)

実行順:

### Action 1: 公式 metric 完全一致確認 (1日)
- 公式: 3D 50cm、 距離割合、 6走行平均、 停止時除外
- 私の per-run 値の単純平均が aggregate と一致しないので、 local scorer の集計方法を公式と完全一致させる
- mismatch あれば 76.83% の数字自体が再校正必要

### Action 2: status-free oracle 出力 (1日)
- 既存 19 candidates の Fix flag 無視、 各 epoch で最良 candidate の 0.5m oracle を計算
- oracle vs current selected の gap で「selector 改善余地」と「candidate 不足」を分離

### Action 3: velocity/IMU bridge candidate (2-3日)
- last fixed pose + Doppler/velocity + yaw/NHC で 2, 4, 6, 8秒 sweep
- n/r2 では **6秒** を第一候補 (2位再実装)
- PPC selector に新 candidate type として追加

### Action 4: constellation subset bank (1-2日)
- n/r2 限定で GQEBR, GQEB, GQER, GQB, GQR, GQ のような衛星系 subset candidate
- gici knob より多様性が出る (failure mode 変更)

### Action 5: Fix validation tighten / submission widen (1日)
- wrong fix は selector を壊すので fixed residual validation を厳しく
- 一方、 Float/DR は別 confidence で拾う

**+0.5pp 未満なら**: 現 pool と gate の限界がほぼ確定 → 1か月プランへ。

---

## 3. 1か月プラン (期待 +3〜7pp)

**Native FGO 小型 MVP**。 gici を置き換えるのではなく、 **n/r2 専用に失敗モードを変える candidate を出す**。

優先 factor:

$$\min_{X,N} \sum \rho_P(r_P) + \sum \rho_\Phi(r_\Phi(N)) + \sum \rho_D(r_D) + \sum r_{\text{IMU}}^T \Sigma^{-1} r_{\text{IMU}} + \sum r_{\text{NHC}} + \sum r_{\text{ZUPT}} + \sum r_{\Delta N}$$

**重要**: N を「一度 LAMBDA で決めたら終わり」にしない。 **3位解法のように Float 解と FGO prediction の両方を seed にした sampling ambiguity posterior** を candidate として出す。 n/r2 に最も刺さる。

同時に city model を **hard gate ではなく soft weight**。 OSM procedural extrusion でも variance inflation なら regression リスク低い。

---

## 4. 半期プラン (期待 +6〜10pp)

**Unified multi-hypothesis TC FGO/RBPF**。

状態:
$$X_t = [p_t, v_t, R_t, b_t^a, b_t^\omega, c_t, \dot{c}_t]$$

離散状態:
$$Z_t = [N_{s,t}, \text{LOS}_{s,t}, \text{slip}_{s,t}, \text{candidate\_mode}_t]$$

観測:
$$P, \Phi, D, \text{IMU}, \text{NHC}, \text{ZUPT}, \text{height}, \text{map visibility}, \text{PPP correction}$$

n/r2 の本質 (ambiguity が repeatedly slip / NLOS が satellite ごと混ざる / IMU bridge が短時間有効) を同モデルで扱える。 Factor graph 単体より、 **discrete ambiguity/LOS hypothesis を持つ RBPF または beam search FGO** が向く。

---

## 5. n/r2 専用 attack vector

### Algorithm-side
| Attack | 目的 |
|---|---|
| status-free oracle / Float rescue | Fix=4 gate 由来取りこぼし除去 |
| 2〜8秒 velocity/IMU bridge | loss-of-lock gap を距離 score に変換 |
| constellation subset bank | 悪い衛星系・基準衛星の影響を外す |
| partial AR by satellite leave-one-out | NLOS/cycle-slip 衛星だけを落として AR |
| sampling ambiguity posterior | LAMBDA ratio failure を迂回 |
| fixed residual validation | wrong fix の selector 汚染を防ぐ |
| NHC / heading / ZUPT | IMU drift と lateral drift を抑える |

### Data-side
| Attack | 目的 |
|---|---|
| base coordinate / APC / PCV 再確認 | systematic cm〜dm bias 除去 |
| GLONASS IFB / inter-system bias 再確認 | mixed constellation DD の bias 除去 |
| MADOCA/IGS correction candidate | base-independent 候補追加 |
| PLATEAU / GSI height / DEM | vertical constraint と wrong ambiguity 抑制 |
| OSM/PLATEAU building soft visibility | NLOS variance inflation |

### Hybrid
| Attack | 目的 |
|---|---|
| candidate confidence calibration | residual/ratio/cov vs correctness を学習 |
| road/height/NHC prior | Float/DR candidate を scoreable にする |
| map-aided ambiguity sampling | NLOS と ambiguity を同時に扱う |

---

## 6. 見落としている angle (top 4)

### 6.1 Fix-rate competition だと思い込むこと
公式は **位置誤差 competition**。 Fix=4 は内部品質ラベルにすぎない。 2位チームが「fix 率を高めるより正解ルート上で高精度に測位することが重要」と明記して velocity integration を入れているのはこの点。

### 6.2 per-run 表と aggregate の metric 整合性
公式 metric は 6走行の **距離割合平均**。 私の per-run 19ao 値を単純平均しても 76.83% にならない場合、 local aggregate が「全距離一括 / horizontal only / Fix=4 only / 3D official」のどれかと混ざっている可能性。 **最初に score script を公式 metric に完全一致させる必要あり**。

### 6.3 n/r2 専用 ratio / satellite-count adaptive ratio / constellation subset
私は AR ratio 2.5/3/4 を global sweep 済だが、 **「名古屋2 だけ ratio=3」「衛星数に応じて ratio threshold を変える」「衛星系 subset を変える」** は global ratio variant と failure mode が違う。 TURING + 2位 の両方がこの近傍を使用。

### 6.4 wrong fix を減らす + scoreable non-fix を増やす を同時に
WL-NL fallback が regression したのは、 geometry 的に wrong な fix が selector を汚染したから。 これは「fix を増やす」方向の危険性を示す。 逆に **Float/DR/velocity bridge は Fix ではないが scoreable な軌跡候補として扱うべき**。

---

## 7. 結論

n/r2 に対しては、

> **「もっと固定する」ではなく**
> **「短いギャップを軌跡で埋める」**
> **「悪い衛星を subset/hypothesis で外す」**
> **「ambiguity を single LAMBDA decision にしない」**

が勝ち筋。
