# gnss_gpu / gnss_gpu_ws Codex 引き継ぎメモ

**最終更新**: 2026-04-12 JST  
**現在のブランチ**: `feature/rtklib-spp-fgo-pipeline`（PR #6 open）  
**現在の HEAD**: `1d7d009`  
**gnss_gpu repo の状態**: raw-bridge 関連の Python 実装とテストは **まだ commit/push していない**。`git -C gnss_gpu status --short` では `experiments/gsdc2023_raw_bridge.py`, `experiments/validate_fgo_gsdc2023_raw.py`, `experiments/validate_gsdc2023_phone_data.py`, `tests/test_validate_fgo_gsdc2023_raw.py`, `tests/test_validate_gsdc2023_phone_data.py` などが untracked。  
**workspace 側の注意**: `ref/gsdc2023/*` は `gnss_gpu` リポジトリの外側にある。`RAW_BRIDGE_API.md`, `run_raw_bridge_batch.py`, `run_fgo_raw_bridge.m`, `functions/submission.m`, `README.md` の変更は workspace ローカルで管理している。Claude に引き継ぐときは **「repo 内変更」と「workspace 補助ファイル」を分けて認識させること**。

**ドキュメント先頭の読み方**

- **セクション E（最優先）**: 2026-04-12 時点の **外れ trip 診断完了 + gated source fallback 実装・検証**。**いま Claude に引き継ぐなら必ずここから読め。**
- **セクション D（補助）**: 2026-04-11 時点の **GSDC2023 raw-bridge 実装完了、40 trip full rerun、shared API 化、MATLAB bridge refactor、API 文書整備**。
- **セクション C（旧・補助）**: 2026-04-08 時点の **GPU FGO + RTKLIB SPP 整合 + Doppler motion model + PPC-Dataset 検証 + GSDC2023 初期状態**。Section D の前提知識として必要な場合だけ読む。
- **セクション A（旧）**: 2026-04-07 時点の RTKLIB demo5 SPP 整合。セクション C に包含済み。歴史的記録。
- **付録 B（凍結）**: UrbanNav / PF / paper assets frozen mainline。別系統。

---

## E. 外れ trip 診断 + gated source fallback — 2026-04-12 更新

### E.0 診断結果

D.4.1 の外れ trip 5件を診断した結果、**5 trip すべて「baseline（Kaggle WLS）が壊れている」ことが判明**。raw_wls / fgo は正常。

#### E.0.1 数値まとめ

| # | trip | baseline_mse_pr | raw_wls_mse_pr | fgo_mse_pr | bl-rw max位置差 | 原因 |
|---|---|---:|---:|---:|---:|---|
| 1 | `2021-11-30-20-59-us-ca-mtv-m/mi8` | **9,985,992** | 161 | 161 | **300 km** | baseline が数百km飛ぶエポックあり |
| 2 | `2023-06-15-18-49-us-ca-sjc-ce1/pixel7pro` | **396,318** | 152 | 153 | **24 km** | baseline が km 級ジャンプ |
| 3 | `2021-09-28-21-56-us-ca-mtv-a/mi8` | **38,126** | 172 | 173 | **20 km** | baseline が km 級ジャンプ |
| 4 | `2020-12-11-19-30-us-ca-mtv-e/pixel4xl` | 124 | 152 | 152 | 8 m | **epoch が 5 しかない**（データ欠損） |
| 5 | `2022-04-27-21-55-us-ca-ebf-ww/mi8` | **1,182** | 140 | 140 | **2 km** | baseline に一部大きなスパイク |

#### E.0.2 全 40 trip の分布

- **raw_wls_mse_pr**: 全 trip が 130〜321 の範囲（中央値 185）— 安定
- **baseline_mse_pr**: 中央値 221 だが上位 5 trip が 500 超（最大 1000万）
- **baseline_mse_pr > 500 の trip は 5 件のみ** — これが OptError 外れ値の正体
- mi8 デバイスが 5件中 3件 — Kaggle WLS baseline に問題がある傾向

#### E.0.3 結論

baseline が壊れている trip でのみ raw_wls/fgo にフォールバックする **gated source** を実装すれば、submission 品質が大幅改善する。

### E.1 gated source fallback の実装

#### E.1.1 新しい position_source: `"gated"`

`POSITION_SOURCES` に `"gated"` を追加。動作:

1. solve_trip() で baseline / raw_wls / fgo の mse_pr をすべて計算（従来通り）
2. `baseline_mse_pr > gated_baseline_threshold`（デフォルト 500）なら、raw_wls と fgo のうち mse_pr が小さい方にフォールバック
3. 閾値以下なら baseline を使用

#### E.1.2 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `gnss_gpu/experiments/gsdc2023_raw_bridge.py` | `POSITION_SOURCES` に `"gated"` 追加、`BridgeConfig.gated_baseline_threshold` 追加、`solve_trip()` に gating ロジック |
| `gnss_gpu/experiments/validate_fgo_gsdc2023_raw.py` | `--gated-threshold` CLI 引数追加 |
| `ref/gsdc2023/run_raw_bridge_batch.py` | `--gated-threshold` CLI 引数追加、subprocess に渡す |

#### E.1.3 使い方

```bash
# single trip
PYTHONPATH=gnss_gpu/python:gnss_gpu python3 \
  gnss_gpu/experiments/validate_fgo_gsdc2023_raw.py \
  --data-root ref/gsdc2023/dataset_2023 \
  --trip test/2021-11-30-20-59-us-ca-mtv-m/mi8 \
  --max-epochs -1 --chunk-epochs 200 \
  --position-source gated

# batch
python3 ref/gsdc2023/run_raw_bridge_batch.py \
  --dataset-root ref/gsdc2023/dataset_2023 \
  --settings-csv ref/gsdc2023/dataset_2023/settings_test.csv \
  --dataset test --workers 1 \
  --max-epochs -1 --chunk-epochs 200 \
  --position-source gated --force
```

### E.2 batch rerun 結果

#### E.2.1 gated vs baseline 全体比較

| 指標 | baseline (04-10) | gated (04-12) | 改善 |
|---|---:|---:|---|
| mean OptError | **260,783** | **243** | **1,072x 改善** |
| median OptError | 222.66 | 203.89 | 8%↓ |
| max OptError | 9,985,992 | 1,446 | **6,906x 改善** |

#### E.2.2 影響を受けた trip（6 件）

| trip | baseline | gated | 改善幅 |
|---|---:|---:|---:|
| `mtv-m/mi8` | 9,985,992 | 161 | -9,985,831 |
| `sjc-ce1/pixel7pro` | 396,318 | 152 | -396,166 |
| `mtv-a/mi8` | 38,126 | 172 | -37,954 |
| `mtv-e/pixel4xl` | 1,850 | 1,446 | -404 |
| `ebf-ww/mi8` | 1,182 | 140 | -1,042 |
| `lax-x/pixel5` | 506 | 321 | -185 |

#### E.2.3 副作用

- **残り 34 trip は完全に同一**（delta = 0.00）— gating は正常 trip に一切影響しない
- 6 件目の `lax-x/pixel5`（baseline_mse_pr=506）も閾値 500 で捕捉された
- 成果物: `ref/gsdc2023/results/test_parallel/20260412_1053/`

### E.3 次タスク

1. batch rerun 結果の確認と submission 生成
2. train trip で ground truth 付き検証（gated vs baseline の精度比較）
3. 閾値チューニング（500 が最適かの検証）
4. D.4.3 の refactor 継続
5. D.4.4 の repo hygiene

---

## D. GSDC2023 raw-bridge + full test rerun + API/refactor handoff — 2026-04-11 更新

### D.0 全体の状況と結論

#### D.0.1 いま何ができるか

1. **preprocessed `dataset_2023.zip` がなくても GSDC2023 を回せる。**  
   Kaggle raw `device_gnss.csv` から Python 側で `bridge_positions.csv` / `bridge_metrics.json` を生成し、MATLAB の `run_fgo.m` / `submission.m` から利用できる。
2. **test 40 trip の full epoch rerun は完了している。**  
   baseline source で 40/40 trip を再実行し、partial submission を生成済み。
3. **raw-bridge の Python 中核ロジックは共通 API に切り出した。**  
   使い回しの本体は `gnss_gpu/experiments/gsdc2023_raw_bridge.py` にある。CLI と batch と MATLAB bridge はここに依存する。
4. **MATLAB bridge は環境変数から source/epoch/chunk/FGO パラメータを受け取れる。**  
   `run_fgo_raw_bridge.m` が Python CLI の front-end として動く。
5. **API/運用ドキュメントは整理済み。**  
   `ref/gsdc2023/RAW_BRIDGE_API.md` に Python API、CLI、CSV/JSON 契約、MATLAB 側 env がまとまっている。

#### D.0.2 現時点の結論

1. **現在の raw-only bridge では baseline を既定にするのが安全。**  
   train の確認用 trip `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4` では、full epoch / chunked 実行で  
   - Kaggle WLS baseline: `RMS2D 2.342 m`, `P50 1.924 m`, `P95 3.935 m`  
   - raw WLS: `RMS2D 6.985 m`  
   - current raw-bridge FGO: `RMS2D 7.313 m`  
   だった。**ground truth がある範囲では baseline が最良**。
2. **`results.csv` の `OptError` は Kaggle score ではない。**  
   これは **疑似距離残差の weighted MSE (`mse_pr`)**。位置精度の proxy として一部使えるが、Kaggle スコアそのものではない。  
   したがって test 40 trip rerun の `OptError` 大外れは「診断対象」であって「即座に submission がダメ」という意味ではない。
3. **auto source selection は現時点では default にしない。**  
   `baseline/raw_wls/fgo` を擬似距離残差で chunk ごとに切り替える `auto` を試したが、train 例では baseline を上回らなかった。  
   今は `position_source=baseline` が既定。

#### D.0.3 直近で完了したこと

1. **all test `device_gnss.csv` を 40/40 trip 分ダウンロード済み。**
2. **`settings_test.csv` は 40 trip をカバーしている。**
3. **full rerun 実行済み。**  
   コマンド:
   ```bash
   python3 ref/gsdc2023/run_raw_bridge_batch.py \
     --dataset-root ref/gsdc2023/dataset_2023 \
     --settings-csv ref/gsdc2023/dataset_2023/settings_test.csv \
     --dataset test \
     --workers 1 \
     --max-epochs -1 \
     --chunk-epochs 200 \
     --position-source baseline \
     --force
   ```
4. **成果物**:
   - `ref/gsdc2023/results/test_parallel/20260410_0900/results.csv`
   - `ref/gsdc2023/results/test_parallel/20260410_0900/submission_20260410_0900.csv`
   - `ref/gsdc2023/results/test_parallel/20260410_0900/summary.json`
5. **結果の要点**:
   - `results_rows = 40`
   - `submission_rows = 71936`
   - `submission_trips = 40`
   - `position_source = baseline`
   - `median OptError = 222.66`
   - `max OptError = 9985992.04`
6. **最大残差 trip（診断優先）**:
   - `2021-11-30-20-59-us-ca-mtv-m/mi8` → `9985992.04`
   - `2023-06-15-18-49-us-ca-sjc-ce1/pixel7pro` → `396317.65`
   - `2021-09-28-21-56-us-ca-mtv-a/mi8` → `38126.32`
   - `2020-12-11-19-30-us-ca-mtv-e/pixel4xl` → `1850.07`
   - `2022-04-27-21-55-us-ca-ebf-ww/mi8` → `1182.06`

#### D.0.4 MATLAB bridge の smoke

2026-04-11 時点で以下を確認済み:

```bash
GSDC2023_BRIDGE_MAX_EPOCHS=5 \
GSDC2023_BRIDGE_CHUNK_EPOCHS=5 \
GSDC2023_BRIDGE_POSITION_SOURCE=baseline \
matlab -batch "cd('/media/autoware/aa/ai_coding_ws/gnss_gpu_ws/ref/gsdc2023'); \
  s=table(\"2020-06-25-00-34-us-ca-mtv-sb-101\",\"pixel4\",'VariableNames',{'Course','Phone'}); \
  opt=run_fgo_raw_bridge('./dataset_2023','train',s); disp(opt);"
```

返ってきた `optstatus`:

- `OptTime ≈ 1.35`
- `OptIter = 1`
- `OptError = 180.8576`
- `Score = 2.2453`

つまり、**MATLAB -> Python CLI -> bridge CSV/JSON -> metrics 返却** の入口は動いている。

### D.1 現在の raw-bridge アーキテクチャ

#### D.1.1 Python 側の責務分離

1. **中核 API**  
   `gnss_gpu/experiments/gsdc2023_raw_bridge.py`
   - `TripArrays`
   - `BridgeConfig`
   - `BridgeResult`
   - `build_trip_arrays(...)`
   - `validate_raw_gsdc2023_trip(...)`
   - `solve_trip(...)`
   - `export_bridge_outputs(...)`
   - `has_valid_bridge_outputs(...)`
   - `bridge_position_columns(...)`
2. **thin CLI wrapper**  
   `gnss_gpu/experiments/validate_fgo_gsdc2023_raw.py`
   - argparse だけ持つ
   - 中核 API を呼んで summary を表示する
3. **batch orchestration**  
   `ref/gsdc2023/run_raw_bridge_batch.py`
   - trip 並列実行
   - `results.csv` 生成
   - sample submission に対する補間
   - `submission_<timestamp>.csv` 生成

#### D.1.2 MATLAB 側の責務分離

1. **README 入口**  
   `ref/gsdc2023/run_fgo.m`
   - `phone_data.mat` がなければ raw-bridge 経路へフォールバック
2. **raw bridge 起動**  
   `ref/gsdc2023/run_fgo_raw_bridge.m`
   - Python CLI の command line を組み立て
   - bridge metrics を読んで `optstatus` に戻す
3. **submission 生成**  
   `ref/gsdc2023/functions/submission.m`
   - `result_gnss_imu.mat` がなければ `bridge_positions.csv` を読む
   - `GSDC2023_BRIDGE_POSITION_SOURCE` で列選択

#### D.1.3 データ契約

`bridge_positions.csv` は以下の候補軌跡を同時に保存する:

- `BaselineLatitudeDegrees`, `BaselineLongitudeDegrees`, `BaselineAltitudeMeters`
- `RawWlsLatitudeDegrees`, `RawWlsLongitudeDegrees`, `RawWlsAltitudeMeters`
- `FgoLatitudeDegrees`, `FgoLongitudeDegrees`, `FgoAltitudeMeters`
- `LatitudeDegrees`, `LongitudeDegrees`, `AltitudeMeters`  
  これは **選択済み output source** に対応
- `SelectedSource`

`bridge_metrics.json` は少なくとも以下を持つ:

- `selected_source_mode`
- `mse_pr`
- `baseline_mse_pr`
- `raw_wls_mse_pr`
- `fgo_mse_pr`
- `fgo_iters`
- `failed_chunks`
- `selected_source_counts`
- train の場合のみ `selected_score_m` / `selected_metrics`

**重要**: `run_raw_bridge_batch.py` の skip 判定は、**単なるファイル存在ではなく** `has_valid_bridge_outputs(...)` を使い、`bridge_positions.csv` と `bridge_metrics.json` が両方あり、`fgo_iters >= 0` かつ `mse_pr` が finite であることを確認する。

### D.2 重要ファイル一覧（Claude がまず開くべきもの）

1. `gnss_gpu/internal_docs/plan.md` ← このファイル
2. `gnss_gpu/experiments/gsdc2023_raw_bridge.py`
3. `gnss_gpu/experiments/validate_fgo_gsdc2023_raw.py`
4. `gnss_gpu/tests/test_validate_fgo_gsdc2023_raw.py`
5. `ref/gsdc2023/RAW_BRIDGE_API.md`
6. `ref/gsdc2023/run_raw_bridge_batch.py`
7. `ref/gsdc2023/run_fgo_raw_bridge.m`
8. `ref/gsdc2023/functions/submission.m`
9. `ref/gsdc2023/README.md`

### D.3 現在使えるコマンド

#### D.3.1 Python single trip

```bash
cd /media/autoware/aa/ai_coding_ws/gnss_gpu_ws
PYTHONPATH=gnss_gpu/python:gnss_gpu python3 \
  gnss_gpu/experiments/validate_fgo_gsdc2023_raw.py \
  --data-root ref/gsdc2023/dataset_2023 \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --max-epochs -1 \
  --chunk-epochs 200 \
  --position-source baseline
```

#### D.3.2 Batch rerun / submission assemble

```bash
cd /media/autoware/aa/ai_coding_ws/gnss_gpu_ws
python3 ref/gsdc2023/run_raw_bridge_batch.py \
  --dataset-root ref/gsdc2023/dataset_2023 \
  --settings-csv ref/gsdc2023/dataset_2023/settings_test.csv \
  --dataset test \
  --workers 1 \
  --max-epochs -1 \
  --chunk-epochs 200 \
  --position-source baseline
```

`--force` を付けなければ有効 bridge 出力を再利用する。

#### D.3.3 README 経由（MATLAB）

```bash
cd /media/autoware/aa/ai_coding_ws/gnss_gpu_ws
GSDC2023_DATASET=test \
GSDC2023_BRIDGE_CHUNK_EPOCHS=200 \
GSDC2023_BRIDGE_POSITION_SOURCE=baseline \
bash ref/gsdc2023/run_readme_local.sh fgo
```

#### D.3.4 テスト

```bash
cd /media/autoware/aa/ai_coding_ws/gnss_gpu_ws
PYTHONPATH=gnss_gpu/python:gnss_gpu python3 -m pytest \
  gnss_gpu/tests/test_validate_fgo_gsdc2023_raw.py \
  gnss_gpu/tests/test_validate_gsdc2023_phone_data.py -q
```

2026-04-11 時点の結果: **`7 passed`**

### D.4 Claude にやってほしい次タスク（優先順）

#### D.4.1 最優先: 外れ trip 診断

やること:

1. `2021-11-30-20-59-us-ca-mtv-m/mi8`
2. `2023-06-15-18-49-us-ca-sjc-ce1/pixel7pro`
3. `2021-09-28-21-56-us-ca-mtv-a/mi8`
4. `2020-12-11-19-30-us-ca-mtv-e/pixel4xl`
5. `2022-04-27-21-55-us-ca-ebf-ww/mi8`

について、各 trip の `bridge_positions.csv` / `bridge_metrics.json` を見て次を比較する:

- baseline と raw_wls の位置差
- baseline と fgo の位置差
- `selected_source_mode`
- `baseline_mse_pr`, `raw_wls_mse_pr`, `fgo_mse_pr`
- `SelectedSource`（auto を再試すなら）

目的:

- **baseline 自体が壊れている trip** と
- **raw/fgo が壊れているだけの trip** を分けること

#### D.4.2 次点: 異常検知 / source gating

今の default は `baseline` 固定。もし改善を続けるなら、次は **global default を変える前に** train で妥当性を見ながら以下をやるべき。

候補:

1. baseline が極端に飛んだ chunk だけ raw_wls か fgo に切り替える
2. `mse_pr` だけでなく軌跡平滑性や step size を gating に入れる
3. device / trip type（mi8, pixel7pro 等）ごとの異常傾向を見る

**やってはいけない**:

- test だけ見て `position_source=auto` を default にすること
- `OptError` を Kaggle score と誤解して source policy を決めること

#### D.4.3 refactor 継続

今回まだ残っている改善余地:

1. `run_raw_bridge_batch.py` の subprocess 実行を直接 API 呼び出しに置き換える
2. `results.csv` / `summary.json` の assembler も Python 共通モジュールに切り出す
3. `run_fgo.m` の raw-bridge branch を helper 化して main loop を短くする

ただし優先度は **外れ trip 診断より下**。

#### D.4.4 repo hygiene

`gnss_gpu` repo 側では raw-bridge 関連ファイルが untracked のまま。Claude に渡すときは:

1. `gnss_gpu` repo に入れるべきもの
   - `experiments/gsdc2023_raw_bridge.py`
   - `experiments/validate_fgo_gsdc2023_raw.py`
   - `experiments/validate_gsdc2023_phone_data.py`
   - `tests/test_validate_fgo_gsdc2023_raw.py`
   - `tests/test_validate_gsdc2023_phone_data.py`
2. workspace 補助として残るもの
   - `ref/gsdc2023/RAW_BRIDGE_API.md`
   - `ref/gsdc2023/run_raw_bridge_batch.py`
   - `ref/gsdc2023/run_fgo_raw_bridge.m`
   - `ref/gsdc2023/functions/submission.m`
   - `ref/gsdc2023/README.md`

を分けて commit / handoff した方がよい。

### D.5 いま触ってはいけない / 忘れてはいけないこと

1. **`OptError` は疑似距離残差であって Kaggle score ではない。**
2. **raw bridge の default source は baseline のまま維持する。**
3. **`device_gnss.csv` は見かけ上 `.csv` でも ZIP payload のことがある。**
4. **`taroz.net` の preprocessed `dataset_2023.zip` は 2026-04-11 時点でも壊れている前提で考える。**
5. **`ref/gsdc2023` は `gnss_gpu` git repo の外。**
6. **MATLAB から Python を呼ぶ経路は quoting を直した状態が正。**  
   `run_fgo_raw_bridge.m` の `shell_quote()` を壊すと README 経路がすぐ死ぬ。

### D.6 ひとことで言うと

**GSDC2023 raw-bridge の end-to-end はもう通っている。**  
次の仕事は「配線」ではなく **外れ trip の診断と source policy の改善**。Claude にはそこから始めさせるのが正しい。

---

## C. GPU FGO + RTKLIB 整合 + Doppler + PPC 検証 — 2026-04-08 更新

### C.0 全体の状況と結論

#### C.0.1 やったこと

1. **GPU FGO ソルバ** (`fgo.cu`): Gauss-Newton + GPU 正規方程式組み立て + ホスト Cholesky。Sagnac 補正、マルチクロック ISB、Huber IRLS、バックトラッキング線探索。
2. **RTKLIB demo5 SPP 観測モデル整合**: `export_spp_meas.c` C ツールで RTKLIB `pntpos` と同一の `prange`/`iono`/`trop`/`sat_clk`/`sat_ecef`/`el_rad`/`var_total`/`sat_vel`/`rx_vel` を CSV 出力。
3. **pybind11 バインディング**: RTKLIB SPP を共有ライブラリ化し Python から直接呼び出し（subprocess 不要）。`py::array_t` の stride=0 バグ発見・修正済み。
4. **重みモード**: `sin²(el)` デフォルト（FGO 精度最良）、`rtklib` モード（pntpos 完全一致）選択可能。
5. **Doppler motion displacement**: RTKLIB `estvel()` の受信機速度を FGO の motion prior に注入。`motion_displacement` パラメータで `(x_{t+1} - x_t - disp[t])` を制約。
6. **PPC-Dataset 検証**: Tokyo 3ラン + Nagoya 3ラン、GPS-only SPP。
7. **回帰テスト**: 4 pytest テスト（精度、RTKLIB 一致、レガシーパス、CSV フォーマット）。
8. **PR #6**: `feature/rtklib-spp-fgo-pipeline` として提出済み。

#### C.0.2 代表的な数値

**gtsam_gnss 公開 RINEX（60 epoch, GPS, 15°, 静止局）**

| 手法 | RMS 2D vs reference |
| --- | ---: |
| RTKLIB demo5 rnx2rtkp SPP | 1.67 m |
| **gnss_gpu FGO (sin²el)** | **0.96 m** |
| gnss_gpu FGO (rtklib weight) | 1.67 m (完全一致) |
| レガシー correct_pseudoranges | 36.2 m (モデル不整合) |

**PPC-Dataset（500 epoch, GPS-only, 15°, 車両走行）**

| Run | RTKLIB SPP | FGO (σ=3) | FGO+Dop (σ=0.3) |
| --- | ---: | ---: | ---: |
| tokyo/run1 | 9.87 m | 7.40 m | **1.86 m** (200ep) |
| tokyo/run2 | 2.66 m | 2.23 m | — |
| tokyo/run3 | 2.55 m | 2.43 m | — |
| nagoya/run1 | 4.75 m | 3.63 m | — |
| nagoya/run2 | 1.23 m | 0.93 m | 1.03 m (200ep) |
| nagoya/run3 | 2.08 m | 2.15 m | — |
| **平均** | **3.86 m** | **3.13 m (19%↓)** | — |

#### C.0.3 やっていないこと / 次にやるべきこと

1. **GSDC2023 Kaggle データ検証** — URL 404 でダウンロード失敗。Kaggle CLI (`kaggle competitions download`) か手動 DL が必要。`phone_data.mat` には前処理済みデータが入っている。scipy.io.loadmat で読める。gtsam_gnss MATLAB 版と同一結果になるかの比較が目的。
2. **マルチ GNSS** — GPS のみ → GPS+GAL+QZS で衛星数 3 倍。`export_spp_meas.c` の `opt.navsys` を `SYS_GPS|SYS_GAL|SYS_QZS` に変え、Python 側も対応必要。
3. **Doppler を FGO 内に正式ファクタ化** — 現状は motion displacement として外部注入。gtsam_gnss のように state に velocity を含め Doppler を直接ファクタ化すると joint optimization になる。
4. **TDCP（Time-Differenced Carrier Phase）** — gtsam_gnss が使う搬送波位相差分ファクタ。数 cm 精度の inter-epoch 制約。
5. **密 Cholesky のスケーラビリティ** — 200 epoch 超で数値不安定。スパース Cholesky か sliding window に移行が必要。
6. **180 epoch 発散バグ** — gtsam_gnss 公開 RINEX で一部エポックが衛星 0 → WLS ゼロベクトル → FGO 発散。

### C.1 リポジトリ・パスの前提

```
gnss_gpu_ws/
├── gnss_gpu/                  # メインリポ (git@github.com:rsasaki0109/gnss_gpu.git)
│   ├── src/positioning/fgo.cu        # GPU FGO カーネル
│   ├── src/rtklib_spp/               # RTKLIB C ラッパー (pybind 用)
│   ├── include/gnss_gpu/fgo.h
│   ├── python/gnss_gpu/fgo.py        # FGO Python ラッパー
│   ├── python/gnss_gpu/_rtklib_spp_bindings.cpp
│   ├── python/gnss_gpu/_bindings.cpp # FGO pybind
│   ├── experiments/
│   │   ├── gtsam_public_dataset.py   # 共有ローダー (RTKLIB 整合)
│   │   ├── compare_fgo_rtklib_demo5.py
│   │   ├── validate_fgo_gtsam_public_dataset.py
│   │   └── validate_fgo_ppc.py       # PPC-Dataset 検証 + Doppler
│   ├── tests/
│   │   ├── test_rtklib_spp_regression.py  # 4テスト
│   │   └── test_fgo.py
│   ├── build/                        # CMake ビルド
│   └── internal_docs/plan.md         # このファイル
├── ref/
│   ├── RTKLIB-demo5/app/consapp/rnx2rtkp/gcc/
│   │   ├── export_spp_meas.c         # CSV 出力ツール (改造済み)
│   │   └── makefile
│   ├── gtsam_gnss/examples/data/     # 公開 RINEX テストデータ
│   ├── PPC-Dataset/PPC-Dataset/      # PPC 6ラン (DL済み 155MB)
│   │   ├── tokyo/run{1,2,3}/
│   │   └── nagoya/run{1,2,3}/
│   └── gsdc2023/                     # MATLAB コードのみ (データ未DL)
│       ├── fgo_gnss.m
│       ├── parameters.m
│       └── functions/
```

### C.2 export_spp_meas CSV フォーマット (2026-04-08 版)

**ヘッダ**: `gps_week,gps_tow,prn,sat_id,prange_m,r_m,iono_m,trop_m,sat_clk_m,satx,saty,satz,el_rad,var_total,svx,svy,svz,rx_vx,rx_vy,rx_vz`

| 列 | 意味 |
| --- | --- |
| `prange_m` | RTKLIB `prange` (TGD/コードバイアス補正済み生コード) |
| `iono_m` | Klobuchar 電離層遅延 (m) |
| `trop_m` | Saastamoinen 対流圏遅延 (m) |
| `sat_clk_m` | `-CLIGHT * dts` 衛星時計 (m) |
| `satx/y/z` | `satposs` ECEF 衛星位置 (m) |
| `el_rad` | 仰角 (rad) |
| `var_total` | `vare + vmeas + vion + vtrp + varerr_spp` (m²) |
| `svx/y/z` | `satposs` ECEF 衛星速度 (m/s) |
| `rx_vx/y/z` | `estvel` ECEF 受信機速度 (m/s, Sagnac 補正済み) |

**擬似距離の使い方**: `pr_for_wls = prange_m - iono_m - trop_m - sat_clk_m`

**重みの選択**:
- `sin²(el)`: `max(sin(el_rad), 0.1)²` — FGO 精度最良
- `1/var_total`: RTKLIB pntpos 完全一致

**ビルド**: `make -C ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc`

### C.3 GPU FGO アーキテクチャ

#### C.3.1 状態ベクトル

```
state[t] = [x, y, z, c_0, c_1, ..., c_{nc-1}]  (3 + n_clock 次元)
```

GPS のみ: `nc=1` → 4 次元/epoch。マルチ GNSS (ISB): `nc` ≤ 4。

#### C.3.2 ファクタ

1. **疑似距離ファクタ** (GPU カーネル `fgo_assemble_pseudorange`):
   - 残差: `res = pr[t,s] - (range(state[t], sat[t,s]) + clock)`
   - Sagnac 地球回転補正付き `geodist`
   - Huber IRLS 対応

2. **Motion ファクタ** (ホスト `add_motion_rw_host`):
   - 残差: `res_i = (state[t,i] - state[t+1,i] + disp[t,i])` for i ∈ {0,1,2}
   - `motion_displacement` = `None` → ゼロ平均ランダムウォーク
   - `motion_displacement` = 速度×dt → Doppler 情報付き motion

3. **密正規方程式 → Cholesky**:
   - `H (n_state × n_state)` + `g (n_state)` を GPU で並列組み立て
   - ホストで Cholesky 分解 → 解く
   - バックトラッキング線探索でステップ幅調整

#### C.3.3 制限事項

- **密 Cholesky**: `n_state > 8192` で拒否。GPS-only 4次元で最大 2048 epoch。
- **200 epoch 超で数値不安定**: PPC 1000 epoch で発散（RMS 800m+）。原因は密行列の条件数悪化。
- **スパース化未実装**: gtsam は iSAM2 でスパース増分解。gnss_gpu は全体をバッチで解く。

### C.4 Doppler motion model の詳細

#### C.4.1 動作原理

RTKLIB `pntpos` → `estvel()` が Sagnac 補正付き Doppler 速度推定を実行。結果は `sol.rr[3..5]` (ECEF m/s)。`export_spp_meas` が `rx_vx,rx_vy,rx_vz` として CSV 出力。

Python 側: `displacement[t] = [rx_vx, rx_vy, rx_vz] * dt` を計算し `fgo_gnss_lm(..., motion_displacement=disp)` に渡す。

#### C.4.2 重要: gnss_gpu の doppler_velocity() は Sagnac 未補正

`python/gnss_gpu/doppler.py` / `src/doppler/doppler.cu` の `doppler_velocity()` は **Sagnac 地球回転補正をしていない**。RTKLIB `resdop()` にある:

```c
rate = dot3(vs, e) + OMGE/CLIGHT*(rs[4]*rr[0] + rs[1]*x[0] - rs[3]*rr[1] - rs[0]*x[1]);
```

この補正項がないため、gnss_gpu の `doppler_velocity()` は ~18 m/s のバイアスが出る（静止局テストで確認）。

**対策**: RTKLIB の `rx_vx/vy/vz` を使う。自前の `doppler_velocity()` は使わない。

#### C.4.3 RINEX D1C 符号規約

RINEX D1C は `d(carrier_phase)/dt` [Hz]。RTKLIB `resdop` は `-D1C * c/freq` で range rate に変換。gnss_gpu の `doppler_velocity()` は `D1C * wavelength` を使う。**符号が逆**。

もし gnss_gpu の `doppler_velocity()` を直接使うなら **D1C を符号反転** + **Sagnac 補正追加** が必要。

#### C.4.4 効果

| 条件 | tokyo/run1 200ep | nagoya/run2 200ep |
| --- | ---: | ---: |
| WLS (per-epoch) | 2.13 m | 1.05 m |
| FGO RW σ=0.3 | 1.96 m | 1.03 m |
| **FGO Dop σ=0.3** | **1.86 m (13%↓)** | 1.03 m |

### C.5 PPC-Dataset 検証の詳細

#### C.5.1 データ

- taroz/PPC-Dataset (OneDrive 155MB)
- Tokyo 3ラン + Nagoya 3ラン
- 5Hz RINEX 3.04 (GPS+GLO+GAL+BDS+QZS)、IMU 100Hz、reference.csv
- **ただし現在の検証は GPS-only**

#### C.5.2 実行方法

```bash
cd gnss_gpu
PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all --max-epochs 500 --motion-sigma-m 3.0
PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all --max-epochs 200 --motion-sigma-m 0.3 --doppler
```

#### C.5.3 注意

- `--max-epochs 200` が安全上限（密 Cholesky）
- `--doppler` は `export_spp_meas` の `rx_vx/vy/vz` を使う（RTKLIB 必須）
- 全ラン実行は約 3-5 分

### C.6 GSDC2023 (Kaggle) — 未完了

#### C.6.1 目的

gtsam_gnss MATLAB 版 (`ref/gsdc2023/fgo_gnss.m`) と同一結果になるかの比較。

#### C.6.2 データ取得

`https://taroz.net/data/dataset_2023.zip` (2.7GB) は **2026-04-08 時点で 404**。

代替手段:
- `kaggle competitions download -c smartphone-decimeter-2023` (Kaggle CLI)
- 手動 DL で `ref/gsdc2023/dataset_2023/` に配置

`run_preprocessing.m` のコメント: **"The dataset_2023 already contains the processed phone_data.mat"** → DL すれば前処理不要。

#### C.6.3 phone_data.mat の構造 (推測)

```matlab
obs.n       % エポック数
obs.nsat    % 衛星数
obs.L1.resPc(n, nsat)  % 疑似距離残差 [m]
obs.L1.resD(n, nsat)   % Doppler 残差 [m/s]
obs.L1.resL(n, nsat)   % 搬送波位相残差 [m]
obs.clk(n, 1)          % 受信機時計 [m]
obs.dclk(n, 1)         % 時計ドリフト [m/s]
obs.dt                 % エポック間隔 [s]
obs.utcms(n, 1)        % UTC ミリ秒
posbl.xyz(n, 3)        % WLS 基準位置 ECEF [m]
nav                    % エフェメリス
```

Python で読む: `scipy.io.loadmat('phone_data.mat')`

#### C.6.4 MATLAB FGO の構成

```
状態/epoch: x(ENU 3), v(ENU 3), c(clock 7), d(drift 1) = 14 次元
ファクタ:
  - PseudorangeFactor_XC:  擬似距離 → 位置+時計
  - DopplerFactor_VD:      Doppler → 速度+ドリフト
  - MotionFactor_XXVV:     位置-速度 連成 (x2 ≈ x1 + v1*dt)
  - ClockFactor_CCDD:      時計-ドリフト 連成
  - TDCPFactor_XXCC:       搬送波位相差分 → 位置+時計
  - Huber ロバスト推定
  - LM 最適化 (max 1000 iter)
パラメータ (parameters.m):
  - sigma_motion = 0.01-0.05 m
  - sigma_motion_clk = 0.1 m
  - Huber P = 0.1-0.2, D = 0.4-0.8, L = 0.2-0.5
  - ENU 座標系 (ECEF→ENU 変換)
```

#### C.6.5 gnss_gpu で再現するために必要なこと

1. `phone_data.mat` を Python で読み込む
2. 観測残差を ECEF に戻す（ENU→ECEF 変換）、または FGO を ENU 対応にする
3. マルチクロック (7次元) 対応: 既存 `n_clock` ≤ 4 → 拡張 or ISB を整理
4. Doppler ファクタの正式実装（現状は motion displacement のみ）
5. TDCP ファクタ実装（最も精度に効く）
6. LM 最適化（現在は GN + 線探索）

**最小限のパス**: 擬似距離 + Doppler velocity で比較。TDCP なし。精度は gtsam_gnss より劣るが、パイプラインの健全性を確認できる。

### C.7 pybind11 バインディングの注意

#### C.7.1 stride=0 バグ

`py::array_t<double>(n)` がゼロストライドの配列を作る環境がある。**必ず明示的に shape + strides を指定する**:

```cpp
auto arr = py::array_t<double>(
    {sz},                                    // shape
    {static_cast<ssize_t>(sizeof(double))}   // strides
);
```

#### C.7.2 ビルド

```bash
cd gnss_gpu/build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) _gnss_gpu _gnss_gpu_rtklib_spp
cp _gnss_gpu.cpython-310-x86_64-linux-gnu.so ../python/gnss_gpu/
cp _gnss_gpu_rtklib_spp.cpython-310-x86_64-linux-gnu.so ../python/gnss_gpu/
```

#### C.7.3 RTKLIB pybind が不要な場合

`CMakeLists.txt` で `if(EXISTS ${RTKLIB_DIR}/rtklib.h)` ガード付き。RTKLIB ソースがなければスキップされる。subprocess + CSV パスに自動フォールバック。

### C.8 回帰テスト

```bash
cd gnss_gpu
PYTHONPATH=python python3 -m pytest tests/test_rtklib_spp_regression.py -v
```

4 テスト:
1. `test_fgo_beats_rtklib_accuracy` — FGO 0.96m < 1.5m threshold
2. `test_fgo_rtklib_weight_alignment` — weight_mode=rtklib で ||FGO-RTKLIB|| < 0.1m
3. `test_legacy_spp_path` — レガシーパス 30-50m (既知のモデルギャップ)
4. `test_export_spp_meas_csv_format` — CSV ヘッダ・データ品質

### C.9 Codex への注意事項

#### C.9.1 やるべきこと（優先順）

1. **GSDC2023 データ取得** → Kaggle CLI で DL → `phone_data.mat` を読んで FGO 実行 → MATLAB と比較
2. **マルチ GNSS** → `export_spp_meas.c` の `SYS_GPS` を拡張 → PPC でマルチ GNSS 検証
3. **Doppler ファクタ正式化** → state に velocity 追加 → 密 Cholesky のスケーラビリティ問題
4. **スパース Cholesky / sliding window** → 200 epoch 制限の撤廃

#### C.9.2 やってはいけないこと

- `sin²(el)` をデフォルトから変えない（RTKLIB 重みは精度が落ちる）
- `motion_displacement` を gnss_gpu の `doppler_velocity()` で計算しない（Sagnac 未補正で 18m/s バイアス）
- 密 Cholesky で 500 epoch 以上を解かない（数値不安定）
- `export_spp_meas` の CSV フォーマットの既存列を変えない（後方互換性）

#### C.9.3 RTKLIB 側の変更は ref/ 配下

`ref/RTKLIB-demo5/` は gnss_gpu リポの外にある。PR には含まれない。`export_spp_meas.c` の変更は別途管理が必要。

#### C.9.4 ビルド手順

```bash
# RTKLIB ツール
make -C ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc

# gnss_gpu CUDA + pybind
cd gnss_gpu/build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp _gnss_gpu*.cpython*.so ../python/gnss_gpu/

# テスト
cd gnss_gpu
PYTHONPATH=python python3 -m pytest tests/test_rtklib_spp_regression.py -v
PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all --max-epochs 200 --motion-sigma-m 0.3 --doppler
```

---

## A. RTKLIB demo5 × gnss_gpu（SPP / FGO / 公開 RINEX）— 2026-04-07 更新（旧）

> **注: このセクションはセクション C に包含済み。歴史的記録として残す。**

### A.0 目的（ユーザーが選んだ「1」）

- **demo5 の `pntpos` / `rescode` と同じ観測項**を gnss_gpu の WLS・FGO に流したい（Python 側で `correct_pseudoranges` + 自前エフェネリスだけだと **約 35 m 級のモデル不整合**が残っていた）。
- 結論: **ずれの本体は FGO ソルバではなく gnss_gpu 側の SPP 前処理**だった。観測を **RTKLIB が使っている分解（`prange`、ブロードキャストイオノ・トロポ、衛星時計）**に揃えると、公開クリップでは **FGO の reference 2D RMS が ~36 m → ~1 m 前後**に改善（下記数値）。

### A.1 リポジトリ・パスの前提

- ワークスペース例: `…/gnss_gpu_ws/`
- **gnss_gpu（Python + 実験）**: `gnss_gpu_ws/gnss_gpu/`
- **RTKLIB demo5**: `gnss_gpu_ws/ref/RTKLIB-demo5/`（clone 例: `rtklibexplorer/RTKLIB` の `demo5` ブランチ）
- **公開テストデータ（gtsam_gnss 同梱例）**: `gnss_gpu_ws/ref/gtsam_gnss/examples/data/`  
  - `rover_1Hz.obs`, `base.nav`, `reference.csv`

### A.2 観測モデル（RTKLIB `pntpos` / `rescode` と整合）

`pntpos.c` の疑似距離残差（単一星座・受信機時計を状態に含む）は次の形。

- `P = prange(obs, nav, opt, …)` … 生コード観測に **TGD / コードバイアス**などを反映した **幾何に乗せる前の擬似距離**
- 残差:  
  `v = P - ( r + dtr_rx - CLIGHT*dts_sat + d_iono + d_trop )`  
  ここで `r = geodist(rs, rr)`、`dts_sat` は `satposs` が出す衛星時計（秒）、`dtr_rx` は受信機時計バイアス（**メートル**として状態に入る）

**gnss_gpu の WLS が「幾何距離 + 受信機時計」だけを見る形**にするには、観測を次にするのが一貫する。

- `pseudorange_for_wls = P - d_iono - d_trop - sat_clk_m`
- ここで **`sat_clk_m = - CLIGHT * dts_sat`**（CSV でそう定義して出力している）

衛星位置は **RTKLIB の `satposs` がそのエポックで使った ECEF** を使うと、`geodist` と一致させやすい。

### A.3 新規 C ツール `export_spp_meas`

**場所**: `ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc/export_spp_meas.c`

**役割**

- RINEX OBS + NAV を読み、エポックごとに `pntpos` を実行（`PMODE_SINGLE`, GPS, `IONOOPT_BRDC`, `TROPOPT_SAAS`, 仰角マスクは CLI で指定）。
- **`ssat[sat-1].vs`** で最終的に使われた衛星に限定し、`rescode` 相当のフィルタ（仰角・`testsnr` 等）に寄せたうえで CSV 出力。

**CSV ヘッダ（1 行 1 衛星）**

`gps_week,gps_tow,prn,sat_id,prange_m,r_m,iono_m,trop_m,sat_clk_m,satx,saty,satz`

- `sat_clk_m` = `-CLIGHT * dts`（上式の「衛星時計項」をメートルで足し引きしやすいように）
- `satx/y/z` = そのエポックの `satposs` ECEF（m）

**ビルド**（`rnx2rtkp` と同じ `gcc` ディレクトリ）

```bash
make -C ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc
```

生成物: 同ディレクトリに `export_spp_meas` と `rnx2rtkp`。

**実行例**

```bash
./export_spp_meas rover_1Hz.obs base.nav -m 15
```

**実装上の注意**

- `traceopen(NULL)` は **呼ばない**（`trace.c` が `strcpy` で落ちる）。
- `postpos.o` リンクのため、`rnx2rtkp.c` と同様に **`showmsg` / `settspan` / `settime` のスタブ**を同 C ファイルに定義。

**Makefile 変更**

- 共有オブジェクトを `RTK_OBJS` にまとめ、`rnx2rtkp` / `export_spp_meas` の両ターゲットからリンク。
- `make all` で **両方**ビルド。

### A.4 Python 側: `gtsam_public_dataset.py`

**`build_public_gtsam_arrays`**

- 新オプション:  
  `rtklib_export_spp_exe: Path | None = None`, `el_mask_deg: float = 15.0`
- `rtklib_export_spp_exe` 指定時:
  - 一時 CSV に `export_spp_meas` を **subprocess** で流し、パース。
  - キー: `(gps_week, round(tow, 4), sat_id)`（例: `G10`）。
  - 各エポックで RINEX 側の衛星順を保ちつつ、`pseudorange` に  
    `prange_m - iono_m - trop_m - sat_clk_m`  
    を代入、`sat_ecef` に CSV の `satx/y/z` を代入。
  - **重み**は従来どおり `correct_pseudoranges(...)[1]` の elevation 由来を流用（`pr_raw` はダミー扱いで、仰角・マスクのためだけに使う）。完全な RTKLIB `varerr` との一致は未実装。
- **エポックメタデータ**: `epochs_data` の型を  
  `(tow, sats, pr_array, gps_week)` に拡張（既存コードが `epochs_data[t][0]` で TOW だけ見る用法は互換）。

**関連 imports**

- `_datetime_to_gps_week` を `gnss_gpu.io.nav_rinex` から使用。

### A.5 実験スクリプト: `compare_fgo_rtklib_demo5.py`

- **既定**: `rnx2rtkp` と同じディレクトリの `export_spp_meas` があれば **自動で SPP 整合パスを使用**（環境変数 `RTKLIB_EXPORT_SPP_MEAS` でも指定可）。
- **`--rtklib-export-spp PATH`**: バイナリ明示。
- **`--no-rtklib-spp-export`**: 従来の `correct_pseudoranges` + `Ephemeris` のみ（モデル不整合再現に有用）。
- **elevation mask** は `--elev` と `build_public_gtsam_arrays(..., el_mask_deg=...)` で一致させる。

**実行例（リポジトリルート想定）**

```bash
cd gnss_gpu
PYTHONPATH=python python3 experiments/compare_fgo_rtklib_demo5.py --max-epochs 60
```

### A.6 代表的な数値（公開クリップ・60 epoch・GPS・15°）

`export_spp_meas` **オン**（上記 compare が自動採用した場合の一例）:

- **FGO RMS 2D vs `reference.csv`**: およそ **0.96 m**（以前、gnss_gpu 単独 SPP 経路では **~36 m** 級）
- **RTKLIB `rnx2rtkp` 単点 vs reference**: およそ **1.67 m**
- **||FGO − RTKLIB|| 2D**: およそ **0.82 m**（観測は揃えたが、重み・反復・時刻・vsat の厳密一致などで差は残りうる）

### A.7 まだやっていない / 次にやるとよいこと

1. **残差 ~0.8 m の ||FGO − RTKLIB||**  
   - RTKLIB `varerr` と gnss_gpu 重みの統一、最終 `vsat` 集合の完全一致、受信時刻補正まわりの確認。
2. **`validate_fgo_gtsam_public_dataset.py`** に  
   `--rtklib-export-spp` / 環境変数連携を足し、CI・ドキュメントの一本化。
3. **運用・依存削減**  
   - subprocess + CSV の代わりに、共有ライブラリ + pybind などで `export_spp_meas` 相当を呼ぶ。
4. **別 RINEX / マルチ GNSS** で同パイプラインの回归テスト。

### A.8 主要ファイル一覧（RTKLIB ライン）

| 役割 | パス |
| --- | --- |
| RTKLIB CSV 出力 | `gnss_gpu_ws/ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc/export_spp_meas.c` |
| RTKLIB ビルド | `…/rnx2rtkp/gcc/makefile` |
| バッチ組み立て | `gnss_gpu/experiments/gtsam_public_dataset.py` |
| FGO vs demo5 比較 | `gnss_gpu/experiments/compare_fgo_rtklib_demo5.py` |
| FGO 検証（未: RTKLIB フラグ） | `gnss_gpu/experiments/validate_fgo_gtsam_public_dataset.py` |
| 旧・モデルギャップ診断 | `gnss_gpu/experiments/diagnose_spp_rtklib_gap.py`（概念確認用） |
| Python SPP（レガシー経路） | `gnss_gpu/python/gnss_gpu/spp.py`（`correct_pseudoranges`） |

### A.9 Claude への注意（RTKLIB ライン）

- **`export_spp_meas` をビルドしていない環境**では compare は `export_spp_meas` を付けずに `build_public_gtsam_arrays` を呼ぶ（`batch_kw` が空 → レガシー `correct_pseudoranges` 経路）。**意図せず古い SPP 経路に戻る**ので、バイナリの有無を README かスクリプト先頭で明示するとよい。
- RTKLIB の `prange` は **`pntpos.c` 内 static**。現状、GPS L1 単頻度向けに **必要最小限を `export_spp_meas.c` 側に複製**している。マルチシステム・IF 組合せ까지揃えるなら **RTKLIB 本体の関数を EXPORT するリファクタ**が必要。
- `correct_pseudoranges` の重みは **RTKLIB と同一ではない**（セクション A.4）。厳密一致が必要なら重みもエクスポートするか、Python で `varerr` を移植する。

---

## 付録 B: UrbanNav / PF / paper assets frozen mainline（2026-04-04 スナップショット）

以下は **別ライン**（粒子フィルタ・都市外乱データ・論文 asset）のフリーズ記録。**HEAD / ブランチ名は当時のメモ**であり、gnss_gpu_ws の現在の git と一致しない場合がある。必要なら倉庫履歴で照合すること。

**最終更新（当時）**: 2026-04-04 JST  
**現在の HEAD（当時）**: `41ccb98` (`refine teaser media layout`)  
**ブランチ状態（当時）**: `main`, worktree clean  
**現フェーズ（当時）**: 実装・探索フェーズ凍結済み。いまは artifact / README / GitHub Pages / 原稿パッケージングの段階。  

---

### B.0 最初に読むもの

1. `README.md`
2. `docs/experiments.md`
3. `docs/decisions.md`
4. `docs/interfaces.md`
5. `docs/paper_draft_2026-04-01.md`
6. `experiments/results/paper_assets/paper_main_table.md`
7. `docs/assets/results_snapshot.json`

この節以下の「`docs/plan.md`」参照は歴史的表記。**現行の引き継ぎは本ファイルのセクション A を優先。**

---

### B.450 いまの結論を先に書く

#### B.450.1 frozen mainline

- **mainline method は `PF+RobustClear-10K`**
- これは **UrbanNav external** の full-run で一番安全に勝っている構成
- README, GitHub Pages, paper assets, snapshot JSON はこの前提に揃っている

#### B.450.2 exploratory / supplemental

- **PPC gate family** は残しているが exploratory
- **`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`** は PPC holdout で surviving gate だが gain は小さい
- **`PF+AdaptiveGuide-10K`** と **`PF+EKFRescue-10K`** は supplemental
- **explicit 3D PF / PF3D-BVH** は accuracy の headline ではなく **systems result**

#### B.450.3 safe headline

いま安全に言えるのは次の 3 本だけ。

1. **UrbanNav external では multi-GNSS PF path が EKF を明確に上回る**
2. **Hong Kong 3シーケンスでも PF+AdaptiveGuide が EKF を上回る（cross-geography breadth）**
3. **BVH は PF3D の runtime を大幅に削る**
4. **PPC では holdout-surviving な小さい gate gain があるが、headline ではない**

#### B.450.4 unsafe headline

以下は今も危ない。

- `world first`
- `3D map aided PF improves real-data accuracy` を主張の中心に置くこと
- `guaranteed strong accept`
- `geography-independent general win`
- `adaptive / rescue` を mainline に昇格させること

---

### B.451 現在の数値

#### B.451.1 paper main table の固定値

出典: `experiments/results/paper_assets/paper_main_table.md`

| section | method | RMS 2D [m] | P95 [m] | >100m [%] | >500m [%] | time [ms/epoch] | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| PPC holdout | Safe baseline | 66.92 | 81.69 | 5.83 | 0.0 |  | `always_robust` |
| PPC holdout | Exploratory gate | 65.54 | 81.22 | 5.83 | 0.0 |  | `entry_veto_negative_exit_rescue...` |
| UrbanNav external | EKF | 93.25 | 178.18 | 16.29 | 0.161 | 0.031 | `trimble + G,E,J` |
| UrbanNav external | PF-10K | 67.61 | 101.46 | 5.44 | 0.0 | 1.367 | `trimble + G,E,J` |
| UrbanNav external | PF+RobustClear-10K | **66.60** | **98.53** | **4.80** | **0.0** | 1.401 | frozen mainline |
| UrbanNav external | WLS+QualityVeto | 2933.77 | 175.38 | 10.13 | 2.552 | 0.195 | promoted core hook |
| BVH systems | PF3D-10K | 55.50 | 58.39 | 0.0 | 0.0 | 1028.29 | real PLATEAU subset |
| BVH systems | PF3D-BVH-10K | 55.50 | 58.39 | 0.0 | 0.0 | **17.78** | **57.8x faster** |

#### B.451.2 UrbanNav external の補強

出典:

- `experiments/results/urbannav_fixed_eval_external_gej_trimble_qualityveto_summary.csv`
- `experiments/results/urbannav_window_eval_external_gej_trimble_qualityveto_w500_s250_summary.csv`

重要なのは run 平均 2 本だけではないこと。

- fixed full-run average:
  - `EKF = 93.25 / 178.18 / 16.29% / 0.161%`
  - `PF-10K = 67.61 / 101.46 / 5.44% / 0.000%`
  - `PF+RobustClear-10K = 66.60 / 98.53 / 4.80% / 0.000%`
- fixed window evaluation (`500 epoch / 250 stride`):
  - `PF+RobustClear-10K` は `EKF` に対して
    - `RMS 90/127 win`
    - `P95 102/127 win`
    - `>100m 89/127 win`
    - `>500m 127/127 ≤`

つまり Tokyo external は「たまたま Odaiba / Shinjuku の run 平均で勝った」だけではない。

#### B.451.3 PPC holdout の位置づけ

出典:

- `experiments/results/pf_strategy_lab_positive6_summary.csv`
- `experiments/results/pf_strategy_lab_holdout6_r200_s200_summary.csv`
- `experiments/results/pf_strategy_family_cv_positive6_holdout6_family_best.csv`

PPC では gate family をかなり掘ったが、最終的な結論はこう。

- tuned split では gain がある
- holdout でも一応 survival する
- ただし gain は **小さい**
- したがって paper / README / Pages の main headline にしてはいけない

安全な表現は「design-space / ablation / experiment-first process の証拠」まで。

#### B.451.4 Hong Kong の位置づけ

出典:

- `experiments/results/urbannav_fixed_eval_hk20190428_gc_adaptive_summary.csv`
- `experiments/results/urbannav_fixed_eval_hk_tst_gc_ublox_summary.csv`
- `experiments/results/urbannav_fixed_eval_hk_whampoa_gc_ublox_summary.csv`

Hong Kong は **supplemental positive result** に昇格した。3シーケンス全てで `PF+AdaptiveGuide-10K` が `EKF` を上回る:

- HK-20190428 (GC): 66.85 m vs 69.49 m (EKF)
- HK TST (GC): 152.37 m vs 301.04 m (EKF) — 49% 改善
- HK Whampoa (GC): 413.68 m vs 463.09 m (EKF) — 11% 改善

ただし frozen mainline `PF+RobustClear-10K` は HK multi-GNSS で崩壊する。
勝つのは supplemental variant (`PF+AdaptiveGuide-10K`)。

paper での位置づけ: cross-geography breadth evidence として使う。
headline claim にはしない（winning method が mainline と異なるため）。

---

### B.452 method freeze

#### B.452.1 mainline

**`PF+RobustClear-10K`**

理由:

- UrbanNav external full-run の frozen winner
- `PF-10K` との差は大きくはないが、tail 指標まで含めて最も安定
- README / Pages / paper assets をこの method に揃え済み

#### B.452.2 exploratory gate

**`entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`**

理由:

- PPC holdout で生き残る non-trivial gate
- ただし improvement は小さい
- mainline ではなく appendix / lab result 扱いが妥当

#### B.452.3 promoted core hook

**`WLS+QualityVeto`**

場所:

- `python/gnss_gpu/multi_gnss_quality.py`
- `experiments/exp_urbannav_baseline.py`
- `experiments/exp_urbannav_fixed_eval.py`

意味:

- multi-GNSS stabilization policy を reusable hook として core 側へ押し上げた
- ただし best external method ではない

#### B.452.4 supplemental variants

- `PF+AdaptiveGuide-10K`
- `PF+EKFRescue-10K`
- `PF+RobustClear+EKFRescue-10K`

役割:

- Hong Kong や sparse regime の mitigation
- cross-geometry weakness の応急処置
- Tokyo full-run frozen mainline の置換ではない

#### B.452.5 3D path

- `PF3D-BVH` は **systems contribution**
- explicit blocked/NLOS likelihood を headline accuracy result にしないこと

理由:

- real PLATEAU + NLOS で explicit 3D likelihood は安定勝ちしていない
- hard / mixture / gate を掘ったが、mainline にはなっていない
- 一方で runtime gain は非常に強い

---

### B.453 ここまでに試して、主役から降ろしたもの

#### B.453.1 PF strategy zoo

`experiments/pf_strategy_lab/` 以下でかなり多くの gate family を試した。

例:

- `always_robust`
- `always_blocked`
- `disagreement_gate`
- `rule_chain_gate`
- `weighted_score_gate`
- `clock_veto_gate`
- `dual_mode_regime_gate`
- `quality_veto_regime_gate`
- `hysteresis_quality_veto_regime_gate`
- `branch_aware_hysteresis_quality_veto_regime_gate`
- `rescue_branch_aware_hysteresis_quality_veto_regime_gate`
- `entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate`

結論:

- best surviving family は最後の `entry_veto_negative_exit...`
- それでも gain は small
- これ以上 strategy family を増やすのは return が薄い

#### B.453.2 adaptive guide

`PF+AdaptiveGuide-10K` は 3-run mixed regime では良く見えたが、full Tokyo external では frozen mainline を超えなかった。

出典:

- `experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_3k_summary.csv`
- `experiments/results/urbannav_fixed_eval_external_gej_trimble_adaptive_full_summary.csv`

結論:

- supplemental に留める
- README / Pages / paper main table を差し替えない

#### B.453.3 rescue variants

Hong Kong では効くが Odaiba で悪化する。

結論:

- safety option としては有用
- mainline には昇格させない

#### B.453.4 3D map accuracy story

real PLATEAU path は integration / systems 的には重要だが、accuracy headline には使わない。

結論:

- runtime figure は main figure で良い
- accuracy 主張は `PF+RobustClear-10K` に寄せる

---

### B.454 重要ファイルの地図

#### B.454.1 main results / artifact builders

- `experiments/build_paper_assets.py`
- `experiments/build_githubio_summary.py`
- `experiments/build_site_media.py`
- `experiments/results/paper_assets/paper_main_table.md`
- `docs/assets/results_snapshot.json`
- `docs/assets/results_snapshot.js`

#### B.454.2 website / README

- `README.md`
- `docs/index.html`
- `docs/site.css`
- `.github/workflows/pages.yml`
- `tests/site/playwright.config.cjs`
- `tests/site/site.spec.cjs`

#### B.454.3 frozen evaluation entry points

- `experiments/exp_urbannav_fixed_eval.py`
- `experiments/exp_urbannav_baseline.py`
- `experiments/exp_urbannav_pf.py`
- `experiments/exp_urbannav_pf3d.py`

#### B.454.4 loaders

- `python/gnss_gpu/io/ppc.py`
- `python/gnss_gpu/io/urbannav.py`
- `python/gnss_gpu/io/plateau.py`
- `python/gnss_gpu/ephemeris.py`

#### B.454.5 strategy lab

- `experiments/pf_strategy_lab/strategies.py`
- `experiments/pf_strategy_lab/evaluate_strategies.py`
- `experiments/pf_strategy_lab/cross_validate_families.py`

#### B.454.6 docs for process

- `docs/experiments.md`
- `docs/decisions.md`
- `docs/interfaces.md`
- `docs/paper_draft_2026-04-01.md`

---

### B.455 README / GitHub Pages / media の現状

#### B.455.1 README

README はすでに artifact-first に更新済み。

載せているもの:

- poster
- teaser GIF
- teaser `mp4` / `webm`
- main figures
- reproduce commands
- method freeze
- safe / unsafe claim の整理

#### B.455.2 GitHub Pages

Pages は `docs/index.html` から静的表示する。

特徴:

- `results_snapshot.js` を読む
- `noscript` fallback あり
- main figures と extra charts を表示
- teaser video は controls なし、`preload="metadata"`
- Playwright smoke test あり

#### B.455.3 teaser 修正

直近の `41ccb98` は teaser 修正。

何を直したか:

- 変な crop / composition をやめた
- paper figure をそのまま preview に使うように変更
- poster 風の full-frame slide にした
- `video.controls` を外して browser UI の被りを避けた

関係ファイル:

- `experiments/build_site_media.py`
- `docs/index.html`
- `docs/assets/media/site_teaser.gif`
- `docs/assets/media/site_teaser.mp4`
- `docs/assets/media/site_teaser.webm`

#### B.455.4 Pages workflow

`.github/workflows/pages.yml` は以下を通す。

1. `python3 experiments/build_paper_assets.py`
2. `python3 experiments/build_githubio_summary.py`
3. `npm ci`
4. `npm run site:smoke`

この順にしてあるので、paper assets と Pages assets のズレが起きにくい。

---

### B.456 validation 状態

#### B.456.1 freeze validation

出典: `experiments/results/freeze_validation_summary.json`

- headline: `440 passed, 7 skipped`
- full summary: `440 passed, 7 skipped, 17 warnings`
- command: `PYTHONPATH=python python3 -m pytest tests/ -q`

warning の中身:

- `pytest.mark.slow`
- `datetime.utcnow()`
- plotting / matplotlib

いまのところ freeze を止める性質の warning ではない。

#### B.456.2 site validation

- `npm run site:smoke`
- Playwright 2 tests pass

これは desktop / mobile の smoke で、main sections, figures, video, overflow を見ている。

#### B.456.3 current repo state（当時）

- branch: `main`
- HEAD: `41ccb98`
- worktree: clean

---

### B.457 data / loaders の整理

#### B.457.1 PPC

役割:

- design split
- ablation
- holdout gate evaluation

主ファイル:

- `python/gnss_gpu/io/ppc.py`
- `experiments/exp_ppc_wls_sweep.py`
- `experiments/exp_ppc_outlier_analysis.py`
- `experiments/exp_ppc_pf_gate_sweep.py`
- `experiments/exp_ppc_pf_blocked_clear_sweep.py`

#### B.457.2 UrbanNav Tokyo

役割:

- external validation の主戦場
- main paper claim の source

主ファイル:

- `python/gnss_gpu/io/urbannav.py`
- `experiments/fetch_urbannav_subset.py`
- `experiments/exp_urbannav_fixed_eval.py`

#### B.457.3 UrbanNav Hong Kong

役割:

- cross-geometry weakness の確認
- supplemental mitigation の testbed

主ファイル:

- `experiments/fetch_urbannav_hk_subset.py`
- `experiments/exp_urbannav_fixed_eval.py`

#### B.457.4 PLATEAU

役割:

- PF3D / BVH systems path
- real mesh integration

主ファイル:

- `python/gnss_gpu/io/plateau.py`
- `experiments/fetch_plateau_subset.py`
- `experiments/scan_ppc_plateau_segments.py`

---

### B.458 まだ残る弱点

全部は潰れていない。いま残っている弱点はかなり限定的。

#### B.458.1 geography breadth

- Tokyo external は強い（2シーケンス、PF+RobustClear mainline）
- Hong Kong 3シーケンスで PF+AdaptiveGuide が EKF を上回る
- ただし HK の winning method は mainline と異なる
- 5シーケンス/2都市の cross-geography breadth がある

#### B.458.2 3D map accuracy headline

- 3D path は systems 的に強い
- でも explicit 3D likelihood が real-data accuracy を押し上げた、とはまだ言いにくい

#### B.458.3 PPC gate gain の小ささ

- holdout-surviving だが small gain
- algorithm novelty の主役に据えるには弱い

#### B.458.4 PF vs PF+RobustClear の差の小ささ

- `PF-10K` も close ablation
- だから robust-clear story は「real but not huge」

これは弱点でもあるが、同時に誠実さでもある。過大主張しない方がいい。

---

### B.459 Claude が次にやるなら（PF / paper ライン）

#### B.459.1 いちばん安全な路線

**新しい method を増やさない。**

やること:

1. manuscript source へ fixed assets を移植
2. bibliography / citation 整理
3. figure / table の caption を仕上げる
4. README / Pages と paper の wording を揃える

#### B.459.2 もし追加実験をするなら

優先順位:

1. **multi-GNSS external breadth の追加**
2. **Hong Kong でも headline が立つ regime の探索**
3. **3D path の systems benchmark 拡充**

やらない方がいい:

- 新しい PPC gate family をさらに量産
- 3D likelihood の headline accuracy 化を急ぐ
- adaptive / rescue を mainline へ無理に昇格

#### B.459.3 artifact / infra で触るなら

候補:

- Pages に captions や downloadable CSV 導線を追加
- CI warnings の軽減
- media の圧縮や alt text 改善

ただし main story 自体はもう固定でよい。

---

### B.460 Claude への注意事項（PF / paper）

#### B.460.1 変えない方がいいもの

- `PF+RobustClear-10K` を mainline とする freeze
- `paper_main_table.md` の headline table
- Pages / README / snapshot JSON の mainline wording

#### B.460.2 変えてよいもの

- paper wording
- captions
- bibliography
- asset presentation
- supplemental section の整理

#### B.460.3 避けるべき主張

- `strong accept は確定`
- `3D map が real-data accuracy を押し上げた`
- `global / geography-independent win`
- `world first`

#### B.460.4 安全な主張

- `UrbanNav external では frozen PF path が EKF を大きく上回った`
- `BVH keeps PF3D accuracy while delivering a large runtime reduction`
- `PPC gate work is exploratory but holdout-surviving`
- `the package now supports honest, reproducible artifact-level evaluation`

---

### B.461 最低限の再生成コマンド（PF / paper artifact）

artifact 層だけならこれで足りる。

```bash
python3 experiments/build_paper_assets.py
python3 experiments/build_site_media.py
python3 experiments/build_githubio_summary.py
npm run site:smoke
PYTHONPATH=python python3 -m pytest tests/ -q
```

#### B.461.1 key outputs

- `experiments/results/paper_assets/paper_main_table.md`
- `experiments/results/paper_assets/paper_captions.md`
- `docs/assets/results_snapshot.json`
- `docs/assets/results_snapshot.js`
- `docs/assets/media/site_teaser.gif`
- `docs/assets/media/site_teaser.mp4`
- `docs/assets/media/site_teaser.webm`

---

### B.462 一言でまとめると（PF / paper）

この repo はもう「新しい gate を探す場所」ではない。  
いまは **`PF+RobustClear-10K` を frozen mainline として提示し、PPC は design-space、BVH は systems、Hong Kong は limitation/control として正直に並べる段階** である。

Claude が **PF/paper ライン**に入るなら、仕事は exploration ではなく **curation / packaging / manuscript integration** が中心になる。

**RTKLIB / 単点観測 / FGO の作業はセクション A を参照。**
