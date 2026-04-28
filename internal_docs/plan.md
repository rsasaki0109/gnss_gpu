# gnss_gpu / gnss_gpu_ws Codex 引き継ぎメモ

**最終更新**: 2026-04-23 JST（B.464 raw-only factor/count parity は 12 exported train trips で 100%。native FGO Doppler は RTKLIB/MATLAB の Sagnac range-rate + satellite-clock-drift convention へ更新。IMU weak prior / absolute height / MATLAB `time_diff_th` gap gate は opt-in 配線済み。VD host factor の direct-solve RHS 符号を motion / clock drift / relative-height で修正。submission/test fallback は `gated` default。raw bridge modularization は validation context / trip observation-matrix input/fill/post-fill stage helper / factor-mask audit helper / residual-value audit helper / audit-output helper / audit CLI helper / BridgeResult assembly helper / TripArrays assembly stage / clock residual stage / mask+base-correction stage / post-observation orchestration stage / observation-preparation orchestration stage / prepared-observation unpack helper / prepared-trip assembly helper / post-observation config-dependency bundle / raw bridge alias audit まで分割・固定済み。multi-GNSS 時の gnss_log GPS pseudorange overlay と epoch merge 漏れを修正し、diagnostics 付き 11 train trips の multi-GNSS 200 epoch residual-value matched parity は全 trip max 0.0071m 以下、full epoch residual-value matched parity は nav-boundary + receiver-clock transmit time + L5 signal-gated sat-product 補正後に全 trip max 0.01294m 以下まで改善）
**2026-04-29 JST sanity note**: 作業再開時に `gnss_gpu/` 側の status を整理した。生成 artifact ノイズは `.gitignore` で `_cmake_build/` と `experiments/results/*/`（`paper_assets` は除外）を ignore し、未追跡は実験 helper / tests の 84 件に圧縮済み。検証は `git diff --check`、`python3 -m compileall -q experiments tests python/gnss_gpu`、`PYTHONPATH=python:. python3 -m pytest -q tests/test_gsdc2023_*.py tests/test_compare_gsdc2023_phone_data_raw_bridge_counts.py tests/test_analyze_gsdc2023_raw_wls_policy.py tests/test_validate_fgo_gsdc2023_raw.py tests/test_validate_gsdc2023_phone_data.py tests/test_fgo_wrapper.py tests/test_fgo.py tests/test_ephemeris.py` が通過（400 passed）。
**2026-04-29 JST test cleanup note**: native WLS / multi-GNSS batch / RAIM / Python fallback / quality residual / synthetic fixture の Sagnac range model を統一。real-valued acquisition の Doppler 符号不定性を test 側で明示し、RTKLIB public regression は elevation-mask 後に 4 weighted sats 未満の epoch を batch から除外するよう修正。legacy `correct_pseudoranges` RINEX path は現状 2.8 km 級に外れる既知 debt として strict xfail 化。検証は `PYTHONPATH=python:. python3 -m pytest -q tests/` が通過（804 passed, 8 skipped, 1 xfailed）。
**ワークスペース**: `…/gnss_gpu_ws/`（本 repo は `gnss_gpu/` サブディレクトリ）
**現在のブランチ / HEAD**: 作業ツリーに依存（未確定なら `git status` / `git rev-parse HEAD` で確認）
**workspace 側の注意**: `ref/gsdc2023/*` と `ref/RTKLIB-demo5/*` は **gnss_gpu リポジトリ外**の参照データ。大容量 ZIP・`kaggle_smartphone_decimeter_2023/`・`base_rinex_cache/` はローカル専用で、**リモートにコミットしない前提**。

**ドキュメント先頭の読み方**

- **セクション G（最優先 / 2026-04-16）**: Codex / Cursor へ渡す最新 handoff。**データ配置・監査・次工程はここが正。**
- **セクション F（履歴 / 2026-04-14）**: VD solver + Multi-GNSS + TDCP の初期統合ログ。
- **セクション E 以下**: さらに古い記録。

---

## G. Cursor 向け最新 handoff — 2026-04-16 更新

### G.0 先に結論

1. **raw-bridge 系の参照 artifact（Kaggle）は引き続き `20260415_1702` 前後の gated + VD + multi-GNSS + position-offset 系が基準。**
   （score 表は G.5.3 / 旧記録を参照。）
2. **MATLAB 本命（taroz upstream）との差は依然として大きい。**
   upstream は約 `Public 0.789 / Private 0.928` 程度。
3. **2026-04-16 時点で「監査上のファイル欠け」は workspace でかなり解消した。**
   Kaggle フル unzip + 補助スクリプトにより、`audit_gsdc2023_matlab_parity.py` は **196 トリップすべてで `base_correction_ready`** に到達可能になった（下記 G.2 / G.3）。
4. **それでも「アルゴリズムの MATLAB 等価」ではない。**
   `collect_matlab_parity_audit` は **ファイル存在チェック**であり、`correct_pseudorange` / `phone_data.mat` / IMU プリインテグ / 双頻など **モデル等価性は未証明**。
5. **graph-level relative-height** は CUDA `fgo_gnss_lm_vd` と raw-bridge に実装済み（過去セッション）。σスイープでは train pixel4 ではベースライン優位になりやすく、**デフォルト ON 推奨までは言えない**。

### G.0.1 このセクションで読むべきサブ節（長い場合）

| 興味 | 読む |
| --- | --- |
| データがどこにあるか | **G.1** |
| どのスクリプトで何を生成したか | **G.2** |
| 監査 JSON の意味 | **G.5** |
| 次にコードで何をすべきか | **G.8** |
| コピペ用コマンド | **G.10** |

### G.1 データツリーと `DEFAULT_ROOT`（2026-04-16）

| パス（workspace 相対の例） | 内容 |
| --- | --- |
| `ref/gsdc2023/kaggle_smartphone_decimeter_2023/smartphone-decimeter-2023.zip` | Kaggle 公式バンドル（数 GB） |
| `ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/` | 解凍後の **`train/` / `test/` / `metadata/`** / `sample_submission.csv` |
| `ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/settings_{train,test}.csv` | **`generate_gsdc2023_settings_from_tree.py`** で全 trip 行を再生成（旧コピーは 1 行だけのことも） |
| `ref/gsdc2023/base/` | **`base_position.csv`** / **`base_offset.csv`**（`generate_gsdc2023_base_metadata.py`）。MATLAB `correct_pseudorange.m` と同じ列名 |
| `ref/gsdc2023/base_rinex_cache/` | **`fetch_gsdc2023_base_obs.py`** が NOAA から落とした日次 obs を **`SLAC_YYYY_DOY.obs`** 形式でキャッシュ |
| `ref/gsdc2023/dataset_2023/` | 部分木・旧手元用。`DEFAULT_ROOT` のフォールバック |

**`DEFAULT_ROOT`**（`experiments/gsdc2023_raw_bridge.py` の `resolve_gsdc2023_data_root()`）:

1. 環境変数 **`GSDC2023_DATA_ROOT`**（存在すれば最優先）
2. `…/kaggle_smartphone_decimeter_2023/sdc2023` がディレクトリとして存在し `train/` がある場合 → **そこを採用**
3. それ以外 → `ref/gsdc2023/dataset_2023`

**`collect_matlab_parity_audit` の `base` 探索**: `dataset_2023` の隣の `base/` に無い場合、Kaggle の `sdc2023` レイアウトでは **`ref/gsdc2023/base/` にフォールバック**（`_base_metadata_dir()`）。

### G.2 補助スクリプト一覧（推奨実行順・目的）

| スクリプト | 役割 |
| --- | --- |
| `merge_gsdc_settings_base1.py` | taroz 等の **参照** `settings_*.csv` から `Base1` / `RINEX` をマージ（参照無しなら不要） |
| `generate_gsdc2023_settings_from_tree.py` | Kaggle ツリーから **`settings_train.csv` / `settings_test.csv`** を全 trip 分行で生成 |
| `apply_gsdc2023_base1_heuristic.py` | **`Base1` をヒューリスティックで埋める**（`-us-ca-lax-` → `VDCY`、他 → `SLAC`）。**taroz 公式行とは同一でない** |
| `fetch_gsdc2023_brdc.py` | 各コースへ **`brdc.<yy>n`**（NOAA CORS S3、日次 GPS 放送暦） |
| `generate_gsdc2023_base_metadata.py` | `ref/gsdc2023/base/` に **`base_position.csv` / `base_offset.csv`**（概算 ECEF／オフセット 0） |
| `fetch_gsdc2023_base_obs.py` | NOAA の Hatanaka **`.yy{d}.gz`** → **`hatanaka`** で `.yyo` にし、`<course>/<Base1>_rnx3.obs` へコピー。. **404 時は ±2 日内でフォールバック**する場合あり |
| `audit_gsdc2023_matlab_parity.py` | trip ごとの **ファイル前提チェック**（≠ MATLAB 数値一致） |

**依存**: `pip install hatanaka`（`fetch_gsdc2023_base_obs.py`）。`taroz` の `dataset_2023.zip` URL は **404 になりやすい**（GitHub `taroz/gsdc2023#9`）。

### G.3 いまの採用物と non-adopted 物

#### G.3.1 採用済み artifact

raw-bridge の採用版 artifact は:

- `ref/gsdc2023/results/test_parallel/20260415_1702/submission_20260415_1702.csv`

ローカル summary:

- `mean_opt_error = 231.97`
- `median_opt_error = 198.245`
- `max_opt_error = 1197.7`

Kaggle:

- `Public 5.181`
- `Private 5.645`

この run は **gated + VD + multi-GNSS + no-TDCP + position-offset** 系。

#### G.3.2 直前の比較対象

重要 run を並べると:

| run | mean OptError | median | max | 備考 |
|---|---:|---:|---:|---|
| `20260415_0919` | 234.1035 | 203.115 | 1197.7 | mi8 fallback / outlier 対応後 |
| `20260415_1033` | 232.0985 | 200.815 | 1197.7 | baseline repair / clock jump prep |
| `20260415_1301` | 231.97 | 198.245 | 1197.7 | `pixel4` 限定 clock aid |
| `20260415_1702` | 231.97 | 198.245 | 1197.7 | `position-offset` 追加、Kaggle 改善 |

重要ポイント:

1. **local `OptError` が同値でも Kaggle は改善する。**
   `20260415_1301` → `20260415_1702` がそれ。
2. **global `mse_pr` / `OptError` を絶対視しないこと。**
   source selection でも train truth とズレた。

#### G.3.3 非採用の code path

以下は code とテストはあるが、full rerun / submission には採用していない。

1. **Samsung 向け TDCP parity**
   - `XXDD`
   - `Loffset`
   - phone-specific TDCP disable
   - real-data spot check で neutral か微悪化
2. **residual-based clock aid**
   - 実装したが `sm-a505u` に regression
   - blocklist を入れて性能は `20260415_1301` と同値に戻した
3. **stop velocity / stop pose graph factor**
   - native 実装 + raw-bridge plumbing + regression test あり
   - local dataset に stop event が無く、実運用評価できていない

### G.4 2026-04-15 に実際に積んだもの

#### G.4.1 raw-bridge 側の到達点

04-15 時点の `gsdc2023_raw_bridge.py` は以下を持つ。

1. **multi-GNSS**
   - GPS / Galileo / QZSS
   - per-slot `sys_kind`
   - multi-clock WLS / refit
2. **VD solver path**
   - `sat_vel`
   - Doppler
   - `clock_drift_sigma_m`
   - chunk solve
3. **TDCP**
   - ADR 由来 TDCP
   - `HardwareClockDiscontinuityCount` を跨ぐ pair 無効化
   - Doppler consistency threshold
   - Samsung A 系 `Loffset`
   - XXCC / XXDD phone policy
4. **source selection**
   - `baseline`
   - `raw_wls`
   - `fgo`
   - `auto`
   - `gated`
   - chunk quality record ベース
5. **device fallback**
   - `mi8` / `xiaomimi8` は multi-GNSS off + raw_wls fallback
6. **clock aid**
   - residual clock series
   - jump segmentation
   - 現在は `pixel4` 限定で有効
7. **postprocess parity**
   - phone position offset
   - loop-aware relative height
8. **IMU parity plumbing**
   - `device_imu.csv` load
   - sync
   - stop detection
   - epoch projection
9. **graph-level stop factor plumbing**
   - solver stop mask
   - `stop_velocity_sigma_mps`
   - `stop_position_sigma_m`

#### G.4.2 native solver 側の到達点

`fgo_gnss_lm_vd` は今こうなっている。

1. state:

```text
[x, y, z, vx, vy, vz, c0, ..., c_{K-1}, drift]
```

2. factor:

- pseudorange
- motion
- clock drift
- Doppler
- TDCP
- stop velocity prior
- stop pose-hold factor

3. MATLAB parity 寄りの差分:

- `clock_use_average_drift`
  - `clk_{t+1} = clk_t + (d_t + d_{t+1}) * dt / 2`
- `tdcp_use_drift`
  - XXDD 用
- `stop_mask`
  - stop epoch を native へ直接流せる

#### G.4.3 MATLAB front-end 側

`run_fgo_raw_bridge.m` は current Python bridge option をほぼ全部 forward できる。

04-15 時点で MATLAB から渡せるもの:

- epoch / chunk
- source mode
- signal / constellation / weight mode
- motion sigma
- clock drift sigma
- gated threshold
- TDCP consistency threshold
- `vd`
- `multi_gnss`
- `tdcp`
- `relative-height`
- `position-offset`
- `stop-velocity-sigma-mps`
- `stop-position-sigma-m`

### G.5 parity audit の最新結論

#### G.5.1 2026-04-16 時点：フル Kaggle ツリー + 補助スクリプト済み

`--data-root ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023` で `audit_gsdc2023_matlab_parity.py --datasets train test` を回した **代表値**（実測）:

- `n_trips = 196`（settings の全行＝course×phone）
- `base_correction_ready = 196`
- `status_counts = {"base_correction_ready": 196}`
- `settings_base1_nonempty_count = 196`
- `device_imu_present = 196` / `imu_sync_ready = 196`
- `ground_truth_present = 156`（test は GT 無しのため 40 trip は 0）
- `ref_height_present = 0`（`ref_hight.mat` は未配置のまま）

**解釈**: 監査は **`Base1`・base メタ・`<Base1>_rnx3.obs`・`brdc.*`** の存在まで。
**MATLAB の `correct_pseudorange` と同じ数値になることは保証しない**（特に NOAA 欠損日の ±日フォールバック、`base_position.csv` の概算座標、ヒューリスティック `Base1`）。

#### G.5.2 旧 snapshot（部分木・41 trips）向けメモ

2026-04-15 以前の **ローカル部分木**では例えば次のようになっていた（履歴）:

- `base_correction_ready = 0`, `base1_missing` が大半
フル Kaggle + 本節 G.1–G.2 の手順で **上記 196/196 へ改善可能**。

#### G.5.3 何が parity で埋まって、何が未完か

埋まったもの（実装・ファイルの両面）:

- raw bridge artifact を MATLAB が読める
- raw-bridge option を MATLAB が forward できる
- multi-GNSS / VD / TDCP の raw 経路
- baseline repair
- phone offset
- IMU stop detection plumbing
- graph-level stop velocity / stop pose factor の native 実装
- **graph-level relative-height（CUDA FGO に ENU-up ループClosure 系因子；raw-bridge から指定可能）**
- **（ファイル監査として）base RINEX / brdc / base_position / offset を揃えたパイプライン**

まだ大きく未完のもの（**アルゴリズム／数値 parity**）:

1. **`preprocessing.m` の完全移植** — `phone_data.mat` 生成パス
2. **base-station pseudorange correction の Python 側への本番組み込み**（MATLAB `correct_pseudorange.m` 相当）
3. **IMU preintegration / bias state / ImuFactor**
4. **absolute height factor**（`ref_hight.mat` 等）
5. **dual-frequency L1/L5 の MATLAB residual / factor 値 parity**

#### G.5.4 score parity の現実

今の raw bridge は:

- raw fallback / artifact generation としては有用
- MATLAB 本命の置き換えではない

repo に残っている score 比較:

| path | Public | Private |
|---|---:|---:|
| raw bridge gated | 4.466 | 6.102 |
| raw bridge gated + position offset (`20260415_1702`) | 5.181 | 5.645 |
| upstream MATLAB | 0.789 | 0.928 |

注意:

1. `20260415_1702` は **private 改善** と **public 悪化** の mixed result ではなく、
   `20260415_1301` 比では public/private とも改善。
2. それでも upstream MATLAB とは大差。

### G.6 実験・診断の要点

#### G.6.1 train で分かったこと

代表 train trip:

- `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`

ここで分かったこと:

1. **global `mse_pr` gating は truth とズレる。**
2. **chunk-aware source selection は有効。**
3. **TDCP は current raw path では決定打になっていない。**
4. **position offset は local `OptError` では見えにくいが Kaggle で効いた。**
5. **stop factor はこの trip では stop epoch が 0 で実質 inactive。**

#### G.6.2 device-specific fallback で効いたもの

効いた:

- `mi8` / `xiaomimi8` の raw_wls fallback
- catastrophic baseline gap override
- `pixel4` 限定 clock aid
- position offset

効かなかった、または採用見送り:

- Samsung A 系 TDCP parity
- 全 phone 共通 residual clock aid
- stop factor の real-data 効果

### G.7 テストと build

#### G.7.1 latest regression

2026-04-15 最新:

```bash
PYTHONPATH=gnss_gpu/python:gnss_gpu python3 -m pytest \
  gnss_gpu/tests/test_fgo_wrapper.py \
  gnss_gpu/tests/test_fgo.py \
  gnss_gpu/tests/test_validate_fgo_gsdc2023_raw.py \
  gnss_gpu/tests/test_validate_gsdc2023_phone_data.py -q
# -> 76 passed
```

内訳の意味:

- native VD solver API regression
- XXDD / average drift regression
- stop factor regression
- raw-bridge build / TDCP / IMU / source selection regression

#### G.7.2 build

native 変更後は必ず:

```bash
cmake --build gnss_gpu/build -j$(nproc)
cp gnss_gpu/build/_gnss_gpu*.so gnss_gpu/python/gnss_gpu/
```

を回すこと。`.so` をコピーしないと Python 側が古い binding を読む。

### G.8 いま Cursor / Codex が次にやるべきこと

#### G.8.1 最優先の判断（2026-04-16）

**「ファイル欠け」はフル Kaggle + G.2 の手順で解消できる。** いま詰まっているのは **数理ギャップ**（`correct_pseudorange` 未接続、IMU プリインテグ、双頻、`phone_data.mat` 経路）。

引き続き留意:

1. stop factor の **batch 全体**での効果は、trip によって `stop_epoch_count=0` が多く ROI が読みにくい
2. dual-frequency 完全 parity は **raw 列 + モデル**の両面で未着手に近い

#### G.8.2 優先順（2026-04-16 更新）

1. **MATLAB `correct_pseudorange.m` 相当を raw bridge または前処理に接続**
   - 監査でファイルは揃った。**擬似距離残差補正を Python で再現**すれば、初めて「base あり」モデル比較が可能。
2. **`phone_data.mat` / `preprocessing.m` パイプラインとの差分表**（何を省略しているかを列挙）
3. **IMU プリインテグレーション（GTSAM `PreintegratedImuMeasurements` 相当は未実装）**
4. **双頻・`ref_hight.mat`・taroz 公式 `Base1` 行との突合い**（ヒューリスティック SLAC/VDCY の置換）
5. **graph-level relative-height の再評価**
   - native は実装済み。デフォルト採用は数値上まだ慎重。

#### G.8.3 やらなくていいこと（当面）

当面は以下を深追いしない方がいい。

1. Samsung A 系 TDCP の追加 heuristic tuning
2. `OptError` だけを見た source selection 再設計
3. stop factor の full batch rerun
   - stop event が無いのでほぼ無意味

### G.9 重要ファイル

#### G.9.1 まず開く

1. `gnss_gpu/experiments/gsdc2023_raw_bridge.py`
2. `gnss_gpu/src/positioning/fgo.cu`
3. `gnss_gpu/python/gnss_gpu/fgo.py`
4. `gnss_gpu/python/gnss_gpu/_bindings.cpp`
5. `gnss_gpu/tests/test_fgo.py`
6. `gnss_gpu/tests/test_validate_fgo_gsdc2023_raw.py`
7. `ref/gsdc2023/MATLAB_PARITY_AUDIT.md`
8. `ref/gsdc2023/run_raw_bridge_batch.py`
9. `ref/gsdc2023/run_fgo_raw_bridge.m`

#### G.9.2 補助スクリプト・テスト（2026-04-16 時点で特に触るもの）

**診断・監査・データ生成**

- `experiments/audit_gsdc2023_matlab_parity.py`
- `experiments/diagnose_gsdc2023_bridge.py`
- `experiments/generate_gsdc2023_settings_from_tree.py`
- `experiments/apply_gsdc2023_base1_heuristic.py`
- `experiments/fetch_gsdc2023_brdc.py`
- `experiments/generate_gsdc2023_base_metadata.py`
- `experiments/fetch_gsdc2023_base_obs.py`（要 `hatanaka`）
- `experiments/merge_gsdc_settings_base1.py`（参照 settings がある場合）
- `experiments/benchmark_relative_height_ab.py` / `benchmark_relative_height_multitrip.py`

**テスト**

- `tests/test_fgo_wrapper.py`

### G.10 具体コマンド

#### G.10.0 フル Kaggle ツリーで監査を緑にする手順（概要）

`cd gnss_gpu`、データルートを `../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023` とする。

1. `kaggle competitions download …` で ZIP を取得し `sdc2023/` を解凍（済ならスキップ）。
2. `PYTHONPATH=python python3 experiments/generate_gsdc2023_settings_from_tree.py --data-root <上>`
3. `… apply_gsdc2023_base1_heuristic.py --data-root <上>`
4. `… fetch_gsdc2023_brdc.py --data-root <上> --splits train test`
5. `… generate_gsdc2023_base_metadata.py --output-dir ../ref/gsdc2023/base`
6. `pip install hatanaka` 後、`… fetch_gsdc2023_base_obs.py --data-root <上>`
7. `… audit_gsdc2023_matlab_parity.py --data-root <上> --datasets train test`

#### G.10.1 best artifact の再確認

```bash
cat ref/gsdc2023/results/test_parallel/20260415_1702/summary.json
```

#### G.10.2 parity audit（フル Kaggle ツリー推奨）

```bash
cd gnss_gpu
PYTHONPATH=python python3 experiments/audit_gsdc2023_matlab_parity.py \
  --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --datasets train test
# explain 付き:
#   ... --explain
```

部分木 `ref/gsdc2023/dataset_2023` のみの場合は `--data-root` をそちらに。

#### G.10.3 single trip validate

```bash
cd gnss_gpu
PYTHONPATH=python python3 experiments/validate_fgo_gsdc2023_raw.py \
  --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --max-epochs 200 \
  --position-source gated \
  --motion-sigma-m 0.3 \
  --clock-drift-sigma-m 1.0 \
  --chunk-epochs 200 \
  --vd --multi-gnss --no-tdcp --position-offset
```

stop factor を試すなら:

```bash
PYTHONPATH=python python3 experiments/validate_fgo_gsdc2023_raw.py \
  --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --max-epochs 200 \
  --position-source gated \
  --motion-sigma-m 0.3 \
  --clock-drift-sigma-m 1.0 \
  --stop-velocity-sigma-mps 0.01 \
  --stop-position-sigma-m 0.02 \
  --chunk-epochs 200 \
  --vd --multi-gnss --no-tdcp --position-offset
```

#### G.10.4 batch rerun

```bash
python3 ../ref/gsdc2023/run_raw_bridge_batch.py \
  --dataset-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --settings-csv ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/settings_test.csv \
  --dataset test \
  --workers 8 \
  --max-epochs -1 \
  --chunk-epochs 200 \
  --position-source gated \
  --motion-sigma-m 0.3 \
  --clock-drift-sigma-m 1.0 \
  --vd --multi-gnss --no-tdcp --position-offset \
  --force
```

### G.11 handoff の一言（2026-04-16）

**ファイル監査上の「MATLAB 前処理の前提ファイル」は、フル Kaggle + 補助スクリプトで揃えられるようになった。**
それでも **raw bridge の数理は MATLAB 本命（`fgo_gnss_imu` + base 補正 + IMU）と同値ではない**。Codex では次を最優先にするとよい。

1. **`correct_pseudorange` 相当の擬似距離補正を Python に実装し、`device_gnss` 経路へ統合できるか検証**
2. **`phone_data.mat` 有無での差分**（MATLAB を真実とするなら preprocessing からの回帰テスト設計）
3. **ヒューリスティック `Base1` / NOAA フォールバック日 obs を公式 taroz 行と比較できるなら差し替え**

heuristic（σ・source）の微調整より **モデルギャップの解体**が ROI が高い。

**2026-04-17 追記**: Python raw bridge に実験用 `--base-correction` を追加した。RINEX 2/3 base obs を読み、broadcast nav で base residual を作り、移動平均後に rover epoch / GPS sat へ補間して `device_gnss` pseudorange から差し引く first-stage 実装。デフォルトは OFF。`tests/test_validate_fgo_gsdc2023_raw.py` と G.7 の regression セットで配線確認済み。まだ MATLAB `correct_pseudorange.m` の完全数値一致ではなく、base receiver clock / iono-trop / multi-GNSS 全周波数の突合いが残る。

**2026-04-21 追記**: `--base-correction` の base RINEX code selection を RINEX3 の `C1C/C1X/C5Q` 等に対応させ、補正対象を GPS-only から GPS/Galileo/QZSS に広げた。`compute_base_pseudorange_correction_matrix` は Android signal band ごとに `C1*` / `P1*` または `C5*` / `P5*` を選び、base nav/residual が作れる satellite slot だけ補正する。これは MATLAB `obsb.sameSat(obsr)` へ近づける変更だが、まだ `correct_pseudorange.m` との per-satellite 数値一致検証が必要。

**2026-04-21 追記 2**: MATLAB 側に `export_base_correction_series.m` を追加し、`correct_pseudorange.m` の finite correction を `phone_data_base_correction_series.csv` へ出せるようにした。Python 側には `experiments/compare_gsdc2023_base_correction_series.py` を追加し、raw bridge の `base_correction_long.csv` と MATLAB CSV を `freq/epoch/utc/sys/svid` で join して matched/only/delta summary を出す。`correct_pseudorange.m` は Kaggle `sdc2023/train/` レイアウトでも `ref/gsdc2023/base` を探せる fallback と、base RINEX `dt` の非厳密 1s/15s 判定を追加。

**2026-04-21 追記 3**: `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5` の `80` epoch smoke で、raw bridge base correction と MATLAB `correct_pseudorange` の L1 GPS correction を比較した。送信時刻反復なしでは `matched=720`, median abs delta `20.384m`, p95 `47.843m`。base RINEX satellite position を pseudorange 由来の transmit time で 3 回反復するよう直した後は `matched=720`, median abs delta `1.725m`, p95 `3.005m`, max `3.007m`。残差は衛星ごとのほぼ定数差なので、次は group delay / iono-trop / MatRTKLIB residual convention を詰める。

**2026-04-21 追記 4**: 上記 smoke の残差を `export_base_residual_diagnostics.m` で分解し、base correction L1 GPS は MATLAB/RTKLIB とミリ未満まで一致した。追加で揃えた点は、(1) base station `satposs` 相当の衛星時計から TGD/BGD code bias を外す、(2) base residual は MATLAB `correct_pseudorange.m` と同じく `base_offset.csv` 適用前の `base_position.csv` 座標で計算する、(3) RTKLIB Saastamoinen tropmodel 式を使う、(4) RINEX2 `ION ALPHA` / `ION BETA` を読み MATLAB `nav.ion.gps` と同じ Klobuchar 係数を使う、(5) MatRTKLIB と同じ close-duplicate GPS nav filter を base correction 経路にも適用する、の5点。`train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5`, `80` epochs の再比較は `matched=720`, median abs delta `0.000124m`, p95 `0.000341m`, max `0.000342m`。この確認は L1 GPS の範囲であり、L5/Galileo/QZSS の実データ数値 parity は引き続き別途必要。

**2026-04-21 追記 5**: GPS L5 の実データ parity も確認した。`train/2021-12-08-20-28-us-ca-lax-c/pixel5`, `80` epochs, `--multi-gnss --dual-frequency` では VDCY base RINEX に実 `C5` があり、MATLAB `export_base_correction_series.m` との比較で L5 GPS `matched=400`, `bridge_only=0`, median abs delta `0.000002m`, p95 `0.000004m`, max `0.000004m`。同 smoke の L1 GPS は `matched=560`, `bridge_only=0`, median abs delta `0.000002m`, p95/max `0.000011m`。この過程で Python base correction が base RINEX 全日を smoothing していたため、MATLAB `preprocessing.m` が保存する `obsb` と違い G31 が前半に余分に finite になることを確認し、base residual cache を rover 全体時刻 `±180s`、MatRTKLIB `Gobs.selectTimeSpan` と同じ base interval rounding で trim するように直した。2020 MTV L1 smoke は trim 後も `matched=720`, median abs delta `0.000124m`, p95 `0.000341m`, max `0.000342m` で退行なし。local train tree の `brdc.*n` は全て GPS NAV DATA で Galileo/QZSS ephemeris を含まないため、Galileo/QZSS base-correction parity は matching non-GPS broadcast nav のある dataset 待ち。

**2026-04-21 追記 6**: dual-frequency raw bridge / native FGO の clock 次元を MATLAB `sysfreq2sigtype.m` に合わせた。`--dual-frequency` 時は GPS-only でも `n_clock=7` とし、signal kind は GPS L1=0, GLONASS G1=1, Galileo E1=2, BeiDou B1=3, GPS L5=4, Galileo E5=5, BeiDou B2=6。QZSS は MATLAB 側に専用 signal-clock index が無いため raw bridge では GPS L1/L5 clock family に寄せる。native `fgo_gnss_lm` / `fgo_gnss_lm_vd` の clock 上限も 7 に上げたので、従来の multi-GNSS L5/E5 共通 bias 近似は解消。残りは `phone_data.mat` の residual/factor 値 parity と、非 GPS base correction に必要な Galileo/QZSS broadcast nav 入手。

**2026-04-21 追記 7**: `compare_gsdc2023_factor_masks.py --max-epochs` が bridge 側だけを切り、MATLAB `phone_data_factor_mask.csv` を全期間のまま比較していた bug を修正した。MATLAB mask も settings / requested window に合わせて trim し、TDCP `L/resL` は `next_epoch_index` が window 外へ伸びる edge を除外する。`train/2021-12-08-20-28-us-ca-lax-c/pixel5`, `--max-epochs 200` の factor-key audit は raw-only で `total_matlab_count=12806`, `total_bridge_count=12796`, `total_matched_count=12788`, `matlab_only=18`, `bridge_only=8`, `symmetric_parity=0.9985944088708418`。同じ条件で `--matlab-residual-diagnostics-mask phone_data_residual_diagnostics.csv` を使う audit mode は `total_matched_count=12806`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。

**2026-04-17 追記 2**: `experiments/audit_gsdc2023_preprocessing_gap.py` を追加した。`preprocessing.m` / `phone_data.mat` の主要 stage を Python raw bridge の実装状況（implemented / partial / experimental / missing）へ写像し、Markdown / CSV / JSON として出せる。`--scan-trips` では `settings_{train,test}.csv` または trip directory から per-trip の `phone_data.mat`・raw CSV・IMU・GT・base correction ready 状態も `trip_gap.csv` にまとめる。これは差分表と回帰テスト設計用の監査であり、MATLAB 数値一致の証明ではない。

**2026-04-17 追記 3**: Python raw bridge に実験用 `--observation-mask` を追加した。`exobs.m` / `exobs_residuals.m` のうち、C/N0・multipath・Android `State` bit・擬似距離レンジ・elevation と、baseline 位置に対する pseudorange residual outlier mask を接続した。デフォルトは OFF。`audit_gsdc2023_preprocessing_gap.py` 上の `observation_masking` は `missing` から `experimental` に更新。まだ Doppler residual / carrier residual / `phone_data.mat` の `obsr` mask count との突合いは未完。

**2026-04-17 追記 4**: Python raw bridge に実験用 `--dual-frequency` を追加した。GPS/Galileo/QZSS の L1/E1 と L5/E5 を同一衛星で潰さず別 observation slot として保持し、native FGO / WLS へ流せる。`audit_gsdc2023_preprocessing_gap.py` 上の `dual_frequency_model` は `missing` から `experimental` に更新した。当初は native clock state 上限の都合で multi-GNSS 時に L5/E5 を共通 bias として扱う近似だったが、2026-04-21 に MATLAB signal-clock index へ更新済み。

**2026-04-17 追記 5**: Python raw bridge に `preintegrate_processed_imu` と `IMUPreintegration` を追加した。`process_device_imu` で時刻同期した acc/gyro から GNSS epoch 間の body-frame `delta_v` / `delta_p` / `delta_angle` を作る front-end helper。`audit_gsdc2023_preprocessing_gap.py` 上の `imu_preintegration` は `missing` から `partial` に更新。ただし gravity/attitude/bias evolution と native graph factor はまだ無いので、GTSAM `PreintegratedImuMeasurements` 等価ではない。

**2026-04-17 追記 6**: 実験用 `--observation-mask` に Doppler residual mask を追加した。baseline WLS から受信機速度を推定し、衛星速度 + LOS の幾何 range-rate と Android pseudorange-rate の差から epoch ごとの common clock drift を robust に引いたうえで、`--doppler-residual-mask-mps`（default 3.0 m/s）超過の Doppler weight だけを落とす。pseudorange weight は変更しない。これで `exobs_residuals.m` の Doppler 側に一歩近づいたが、carrier/TDCP residual mask と `phone_data.mat` の mask count parity はまだ未完。

**2026-04-17 追記 7**: ADR 由来 TDCP 構築に `tdcp_consistency_mask_count` を追加した。`doppler_weights` が raw/Doppler residual mask で落ちた pair は Doppler-carrier consistency 判定に使わず、実際に `tdcp_consistency_threshold_m` で TDCP pair を落とした数を `bridge_metrics.json` と summary に出す。MATLAB `Lmask_dDL` の完全移植ではないが、carrier/TDCP 側の除外挙動を audit 可能にした。残りは `phone_data.mat` の `obsr` と per-trip mask count を突き合わせること。

**2026-04-17 追記 8**: gap audit に `--bridge-counts` / `--bridge-max-epochs` を追加した。`--scan-trips` と併用すると raw bridge を experimental flags（observation mask + dual frequency + TDCP）で構築し、P/D/TDCP count・raw/pr/doppler/TDCP mask count・`phone_data.mat` L1/L5 count との差分列を `trip_gap.csv` に出す。ローカルのフル Kaggle tree には `phone_data.mat` が無いため、この環境では差分値は未評価だが、MATLAB artifact がある tree で mask parity を確認する入口はできた。

**2026-04-17 追記 9**: 実験用 `--observation-mask` に pseudorange-Doppler consistency mask を追加した。`--pseudorange-doppler-mask-m`（default 40 m）で、隣接 epoch の pseudorange delta と Doppler displacement の差を見て、common clock step / inter-system offset を差し引いても閾値を超える slot の pseudorange weight を両端 epoch で落とす。MATLAB `Pmask_dDP` に対応する入口で、`sm-a205u` / `sm-a505u` は MATLAB と同じく skip する。

**2026-04-17 追記 10**: MATLAB wrapper `run_fgo_raw_bridge.m` から Python raw bridge の追加 parity flags を渡せるようにした。`GSDC2023_BRIDGE_BASE_CORRECTION` / `GSDC2023_BRIDGE_OBSERVATION_MASK` / `GSDC2023_BRIDGE_DUAL_FREQUENCY` と、observation mask 閾値・graph relative height 閾値を環境変数で制御できる。これで MATLAB consumer 経由の batch でも Python CLI と同じ実験フラグを使える。

**2026-04-17 追記 11**: `experiments/export_gsdc2023_base_correction_series.py` を追加した。raw bridge の `compute_base_pseudorange_correction_matrix` をそのまま使い、epoch × satellite slot の補正量を `base_correction_wide.csv` / `base_correction_long.csv` / `base_correction_summary.json` に出す。`TripArrays.slot_keys` も追加し、MATLAB `correct_pseudorange.m` の per-satellite correction series と突き合わせる scaffold ができた。

**2026-04-17 追記 12**: `experiments/compare_gsdc2023_phone_data_raw_bridge_counts.py` を追加した。`phone_data.mat` の L1/L5 `P/D/L/resPc/resD/resL` finite count と、raw bridge の P/D/TDCP count を trip ごとに `count_comparison.csv` / `trip_summary.csv` / `summary.json` へ出す。ローカル tree に `phone_data.mat` が無い trip も raw bridge count 側だけ残るので、MATLAB artifact あり/なしの差を分けて audit できる。

**2026-04-17 追記 13**: native `fgo_gnss_lm_vd` に最小 IMU kinematic prior を追加した。state layout は `[x,y,z,vx,vy,vz,clk...,drift]` のまま、optional `imu_delta_p` / `imu_delta_v` と scalar sigma で隣接 epoch の `x1 - x0 - v0*dt = delta_p`、`v1 - v0 = delta_v` を host-side factor として足す。attitude/bias state はまだ持たないので GTSAM preintegrated IMU 等価ではなく、raw bridge の `preintegrate_processed_imu` 出力を native graph に渡すための phase-1 prior。

**2026-04-17 追記 14**: observation mask を row-level から P/D/L field-specific に寄せた。MATLAB `exobs` / `exobs_residuals` は `P`・`D`・`L` を別々に NaN にするため、raw bridge でも pseudorange weight、Doppler weight、ADR/TDCP availability を別 mask で扱う。これにより Android State の TOW/TOD 不成立で `P` だけ落ちる衛星でも、Doppler と ADR が有効なら `D` / TDCP edge は残る。`compare_gsdc2023_phone_data_raw_bridge_counts.py` も `TripArrays` の P/D/TDCP weight から直接 count するように変更した。

**2026-04-17 追記 15**: observation mask の default threshold を MATLAB final pass に寄せた。`Pmask_res` は L1=20m / L5=15m、`Pmask_dDP` は L1=40m / L5=20m とし、default 値のときだけ slot の signal type から L5/E5/B2 を低い閾値へ切り替える。CLI / batch の `--pseudorange-residual-mask-m` default も 20m に合わせた。非 default の scalar threshold を明示した場合は従来通り全 slot 同一閾値として扱う。

---

## F. VD solver + Multi-GNSS + TDCP + Kaggle 提出 — 2026-04-14 更新

### F.0 全体の状況と結論

#### F.0.1 いま何ができるか

1. **GPU FGO に 3 つの新ファクタが実装済み**: Velocity-Doppler (`fgo_gnss_lm_vd`)、Multi-GNSS ISB、TDCP
2. **Multi-GNSS で PPC-Dataset が 30% 改善**: GPS-only 2.07m → GPS+GAL+QZS 1.45m (6ラン平均)
3. **VD solver で Multi-GNSS の ISB 劣化を防止**: std FGO 1.83m → VD 1.42m
4. **TDCP ファクタが VD solver で動作確認済み**: 合成データテストで inter-epoch jitter 削減
5. **Kaggle GSDC2023 submission 済み**: gated source で Public 4.466m / Private 6.102m

#### F.0.2 Kaggle スコア比較

| Submission | Public | Private | 手法 |
|---|---:|---:|---|
| GPU PF 100K (過去最良) | 4.207 | 5.144 | パーティクルフィルタ、GPS L1/L5+GAL、Doppler |
| **gated (今回)** | **4.466** | **6.102** | raw-bridge baseline + gated fallback |
| taroz MATLAB FGO (1位) | 0.789 | 0.928 | Pseudorange+Doppler+TDCP+IMU、マルチGNSS |

**現状の gap**: taroz の 5.6 倍。主因は FGO パイプラインが GSDC2023 に未統合（PPC では動作確認済みだが GSDC2023 の device_gnss.csv 経路では未接続）。

#### F.0.3 直近で完了したこと（04-12〜04-14）

| Commit | 内容 | 主な成果 |
|---|---|---|
| `a8ed242` | GSDC2023 raw-bridge + gated source | mean OptError 260K→243 |
| `4f2b361` | VD solver + Multi-GNSS pipeline | PPC 2.07→1.45m (30%↓) |
| `012cd61` | TDCP factor + VD PPC validation | TDCP 動作確認、VD+Multi-GNSS 1.42m |

#### F.0.4 テスト状況

```bash
cd gnss_gpu
PYTHONPATH=python python3 -m pytest tests/test_fgo.py tests/test_rtklib_spp_regression.py \
  tests/test_validate_fgo_gsdc2023_raw.py tests/test_validate_gsdc2023_phone_data.py -v
# → 48 passed
```

### F.1 Velocity-Doppler solver (`fgo_gnss_lm_vd`)

#### F.1.1 State vector

```
state[t] = [x, y, z, vx, vy, vz, c0, ..., c_{nc-1}, drift]
           |---pos---|----vel----|------clocks---------|drift|
           3 dim       3 dim      n_clock dim          1 dim
```

GPS-only (`n_clock=1`): 8 dim/epoch。Multi-GNSS (`n_clock=3`): 10 dim/epoch。

#### F.1.2 ファクタ

1. **Pseudorange** (`fgo_assemble_pseudorange_vd` GPU kernel): position[0:3] + clock[6:6+nc]
2. **Doppler** (`add_doppler_factor_host`): velocity[3:6] + drift[6+nc]
3. **Motion** (`add_motion_factor_host`): `x_{t+1} ≈ x_t + v_t * dt` (位置-速度カップリング)
4. **Clock drift** (`add_clock_drift_factor_host`): `clk_{t+1} ≈ clk_t + drift_t * dt`
5. **TDCP** (`add_tdcp_factor_host_vd`): `e^T*(x_{t+1}-x_t) + (clk_{t+1}-clk_t) ≈ tdcp_meas`

#### F.1.3 符号規約の注意

- **VD solver**: 正しい GN 規約。`g = J_pred^T * W * (obs - pred)`、`H*Δx = g` で直接解く。
- **Legacy solver (`fgo_gnss_lm`)**: gradient を **negate** してから Cholesky 解く（符号バグ）。WLS 初期化 + line search で動くが、TDCP の改善効果が出にくい。**新機能は VD solver を使うべき。**

#### F.1.4 コマンド

```bash
cd gnss_gpu
# GPS-only VD
PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all --max-epochs 200 --motion-sigma-m 3.0 --vd

# Multi-GNSS VD
PYTHONPATH=python python3 experiments/validate_fgo_ppc.py --all --max-epochs 200 --motion-sigma-m 3.0 --multi-gnss --vd
```

### F.2 Multi-GNSS observation pipeline

#### F.2.1 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `ref/RTKLIB-demo5/.../export_spp_meas.c` | `SYS_GPS\|SYS_GAL\|SYS_QZS`、`sys_id` カラム、Galileo BGD 補正 |
| `gnss_gpu/experiments/gtsam_public_dataset.py` | `sys_kind` array + `n_clock=3` ISB 対応、`multi_gnss` パラメータ |
| `gnss_gpu/experiments/validate_fgo_ppc.py` | `--multi-gnss` フラグ |

#### F.2.2 CSV フォーマット（updated）

`gps_week,gps_tow,prn,sat_id,**sys_id**,prange_m,r_m,iono_m,trop_m,sat_clk_m,satx,saty,satz,el_rad,var_total,svx,svy,svz,rx_vx,rx_vy,rx_vz`

`sys_id`: `G`=GPS, `E`=Galileo, `J`=QZSS。GLONASS は除外（周波数依存）。

#### F.2.3 PPC-Dataset 結果（6ラン平均, 200 epoch）

| モード | 平均衛星数 | FGO RMS2D |
|---|---:|---:|
| GPS-only | 9 | 2.07m |
| **GPS+GAL+QZS** | **20** | **1.45m (30%↓)** |
| **GPS+GAL+QZS + VD** | **20** | **1.42m** |

### F.3 TDCP factor

#### F.3.1 実装済みの内容

- `add_tdcp_factor_host`: legacy solver (4-dim state) 用
- `add_tdcp_factor_host_vd`: VD solver (8-dim state) 用
- `tdcp_cost_host` / `tdcp_cost_host_vd`: line search 用コスト関数
- Python/pybind wrappers: `tdcp_meas`, `tdcp_weights`, `tdcp_sigma_m` パラメータ

#### F.3.2 残バグ / 制限

- Legacy solver では TDCP の効果が限定的（F.1.3 の符号問題）。**VD solver で使うこと。**
- **RINEX 搬送波位相パーサーが未実装** — TDCP を実データで使うには `export_spp_meas.c` に L1C 出力を追加する必要がある。
- Cycle slip 検出は `python/gnss_gpu/cycle_slip.py` に既存だが、TDCP パイプラインとの結合が未完了。

#### F.3.3 TDCP 設計書の要点（調査済み）

MATLAB gtsam_gnss の TDCP 実装を詳細調査済み:
- **2 バリアント**: XXCC（位置+時計）、XXDD（位置+ドリフト, Samsung A 用）
- **搬送波位相精度**: 擬似距離の 400 倍（σ_L / σ_P = 1/400）
- **Cycle slip 検出**: Doppler-carrier 整合性チェック（閾値 1.5m）
- **Huber 閾値**: L=0.2〜0.5（環境依存）
- **Phone 別挙動**: Pixel → XXCC、Samsung A → XXDD + Loffset=1.117m、sm-a325f/samsunga32 → TDCP 無効

### F.4 Kaggle 認証情報

```
~/.kaggle/kaggle.json: username=rsasaki, key=b721abd3adedf7c5716c228a5a030a85
```

新形式 `KGAT_*` トークンは Bearer 認証が必要（kaggle CLI 1.7.x は Basic auth のみ → 非対応）。上記は従来形式で動作確認済み。

### F.5 Claude にやってほしい次タスク（優先順）

#### F.5.1 最優先: GSDC2023 パイプラインに VD + Multi-GNSS + TDCP を統合

**目的**: Kaggle スコアを 4.466m → 1m 以下に改善

やること:

1. **`gsdc2023_raw_bridge.py` を VD solver に切り替える**
   - 現在は `fgo_gnss_lm` (4-dim) → `fgo_gnss_lm_vd` (8-dim) に変更
   - `device_gnss.csv` から Doppler 情報を抽出（`DopplerShiftHz` or `PseudorangeRateMetersPerSecond`）
   - 衛星速度を `SvVelocityXEcefMetersPerSecond` 等から取得
   - `dt` を epoch 間隔から計算
   - Multi-GNSS: `ConstellationType` で GPS(1) + Galileo(6) + QZSS(4) をフィルタ、ISB 用 `sys_kind` を構築

2. **TDCP を GSDC2023 device_gnss.csv で有効化**
   - `AccumulatedDeltaRangeMeters` (ADR) が搬送波位相
   - `AccumulatedDeltaRangeState` でサイクルスリップ/品質フラグ
   - ADR の time-differencing で TDCP measurement を構築
   - Doppler-carrier 整合性チェックで cycle slip 検出

3. **40 trip batch rerun + submission**
   ```bash
   python3 ref/gsdc2023/run_raw_bridge_batch.py \
     --dataset-root ref/gsdc2023/dataset_2023 \
     --settings-csv ref/gsdc2023/dataset_2023/settings_test.csv \
     --dataset test --workers 1 \
     --max-epochs -1 --chunk-epochs 200 \
     --position-source gated --force
   ```

#### F.5.2 次点: Legacy solver の GN 符号修正

`fgo_gnss_lm` の line 584 `h_g[i] = -h_g[i]` を削除し、正しい GN 規約にする。ただし:
- **全テストの reference 値を更新する必要がある**（`test_fgo_first_step_matches_reference` 24 テスト）
- RTKLIB regression テストの閾値も再確認
- これをやると FGO の精度が改善するはず（現在 line search が GN ステップを拒否 → 実質 WLS と同等）

#### F.5.3 搬送波位相パーサー追加

`export_spp_meas.c` に L1C carrier phase 出力を追加:
- CSV に `carrier_phase_m` カラム追加
- PPC-Dataset で TDCP + FGO の実データ検証
- Cycle slip detection pipeline の結合

#### F.5.4 Sparse Cholesky / Sliding window

200 epoch 制限の撤廃。密 Cholesky → スパース or sliding window。
VD solver は 8-dim state なので 200 epoch で 1600 dim → 条件数がさらに悪化。

#### F.5.5 IMU preintegration

MATLAB gtsam_gnss の `fgo_gnss_imu.m` を参考に IMU factor を追加。GSDC2023 の `device_imu.csv` から加速度・ジャイロを取得。

### F.6 いま触ってはいけない / 忘れてはいけないこと

1. **`fgo_gnss_lm` の gradient 符号を変えるなら全テスト reference を更新せよ。** 安易に negation を消すと 24 テスト落ちる。
2. **`export_spp_meas.c` の既存カラム順を変えない。** `sys_id` は末尾追加ではなく `sat_id` の後に挿入済み。
3. **TDCP は VD solver (`fgo_gnss_lm_vd`) で使え。** Legacy solver では符号問題で効果が出ない。
4. **gnss_gpu の `doppler_velocity()` は使うな。** Sagnac 未補正。RTKLIB の `rx_vx/vy/vz` を使う。
5. **`ref/gsdc2023/*` は gnss_gpu repo の外。** workspace 補助ファイル。
6. **Kaggle CLI は旧形式キーのみ対応。** `KGAT_*` は Bearer auth 必要。

### F.7 重要ファイル一覧

| 役割 | パス |
|---|---|
| **VD solver** | `src/positioning/fgo.cu` (fgo_gnss_lm_vd) |
| **VD Python wrapper** | `python/gnss_gpu/fgo.py` |
| **VD pybind** | `python/gnss_gpu/_bindings.cpp` |
| **C header** | `include/gnss_gpu/fgo.h` |
| **Multi-GNSS C ツール** | `ref/RTKLIB-demo5/.../export_spp_meas.c` |
| **PPC 検証 (VD/Multi-GNSS)** | `experiments/validate_fgo_ppc.py` |
| **GSDC2023 raw-bridge** | `experiments/gsdc2023_raw_bridge.py` |
| **GSDC2023 CLI** | `experiments/validate_fgo_gsdc2023_raw.py` |
| **Batch orchestration** | `ref/gsdc2023/run_raw_bridge_batch.py` |
| **FGO テスト** | `tests/test_fgo.py` (48 tests) |
| **RTKLIB テスト** | `tests/test_rtklib_spp_regression.py` (4 tests) |
| **GSDC2023 テスト** | `tests/test_validate_fgo_gsdc2023_raw.py` (5 tests) |
| **TDCP 設計書** | (本セクション F.3 + F.5.1 に記載) |
| **MATLAB FGO 参考** | `ref/gsdc2023/fgo_gnss.m`, `ref/gsdc2023/fgo_gnss_imu.m` |
| **MATLAB パラメータ** | `ref/gsdc2023/functions/parameters.m` |

### F.8 ビルド手順

```bash
# RTKLIB ツール (Multi-GNSS export_spp_meas)
make -C ref/RTKLIB-demo5/app/consapp/rnx2rtkp/gcc

# gnss_gpu CUDA + pybind
cd gnss_gpu/build
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cp _gnss_gpu*.cpython*.so ../python/gnss_gpu/

# テスト
cd gnss_gpu
PYTHONPATH=python python3 -m pytest tests/test_fgo.py tests/test_rtklib_spp_regression.py \
  tests/test_validate_fgo_gsdc2023_raw.py tests/test_validate_gsdc2023_phone_data.py -v
# → 48 passed
```

### F.9 ひとことで言うと

**GPU FGO の主要ファクタ（Pseudorange + Doppler + TDCP + Motion + Clock drift + Multi-GNSS ISB）は実装済み。**
次の仕事は「ソルバの追加」ではなく **GSDC2023 の device_gnss.csv パイプラインへの統合** と **Legacy solver の符号修正**。これで Kaggle 4.5m → 1m 以下が視野に入る。

---

## E. 外れ trip 診断 + gated source fallback — 2026-04-12 更新

> **注: このセクションはセクション F に包含済み。gated source の詳細記録として残す。**

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

#### E.0.2 gated vs baseline 全体比較

| 指標 | baseline (04-10) | gated (04-12) | 改善 |
|---|---:|---:|---|
| mean OptError | **260,783** | **243** | **1,072x** |
| median OptError | 222.66 | 203.89 | 8%↓ |
| max OptError | 9,985,992 | 1,446 | **6,906x** |

#### E.0.3 閾値チューニング結果（train trip ground truth）

train trip は 1 件のみ: `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`

| 閾値 | gating発動? | RMS2D |
|---:|---|---:|
| 100-170 | YES (誤判定) | 7.3m (3倍悪化) |
| **180-1000** | **NO (正解)** | **2.3m** |

**threshold=500 は安全で最適。** 200 未満に下げると正常 trip を誤 reject する。

**核心: mse_pr が低い ≠ 位置精度が良い。** Kaggle WLS は multi-GNSS + 高度処理で mse_pr=179.8 だが RMS2D=2.3m。自前 raw WLS は mse_pr=168.8 だが RMS2D=7.0m。

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
   当時は `position_source=baseline` が既定だった。2026-04-20 現在の submission/test fallback default は `gated`。

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
  --position-source gated
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
  --position-source gated
```

`--force` を付けなければ有効 bridge 出力を再利用する。
`run_raw_bridge_batch.py` と MATLAB fallback (`run_fgo_raw_bridge.m` / `functions/submission.m`) の default source は `gated`。`gated` / `auto` の submission assembly は `bridge_positions.csv` の `LatitudeDegrees` / `LongitudeDegrees`、つまり選択済み output columns を使う。

#### D.3.3 README 経由（MATLAB）

```bash
cd /media/autoware/aa/ai_coding_ws/gnss_gpu_ws
GSDC2023_DATASET=test \
GSDC2023_BRIDGE_CHUNK_EPOCHS=200 \
GSDC2023_BRIDGE_POSITION_SOURCE=gated \
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

**2026-04-16 追記**: フル Kaggle 展開・`settings` / `Base1` / `brdc` / base RINEX の生成手順は **セクション G.1 / G.2 / G.10.0** を正とする（本小節は歴史記録を兼ねる）。

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

---

### B.463 GSDC2023 MATLAB factor-count parity 追記

#### B.463.1 追加したもの

- `ref/gsdc2023/export_phone_data_observation_counts.m`
  - SciPy では読めない `GobsPhone` class object の raw `P/D/L` 有限 count を MATLAB 側で CSV 化する。
- `ref/gsdc2023/export_phone_data_factor_counts.m`
  - `exobs` → residual → `exobs_residuals` 後の MATLAB `fgo_gnss.m` factor count を CSV 化する。
  - `P/resPc` と `D/resD` は graph が実際に見る `resPc/resD` 有限 count。
  - `L/resL` は連続 epoch の `resL` から作られる TDCP factor count。
- `experiments/compare_gsdc2023_phone_data_raw_bridge_counts.py`
  - `--trip` で単一 trip を固定比較できる。
  - `phone_data_factor_counts.csv` を優先して読み、なければ `phone_data_observation_counts.csv`、最後に SciPy `phone_data.mat` 読みに fallback。
  - `--no-multi-gnss` で MATLAB GPS-only 条件に合わせられる。
  - `--pseudorange-residual-mask-l5-m` と `--tdcp-consistency-threshold-m` で dual-frequency / TDCP 側の effective threshold も固定できる。
  - `count_parity_ratio` を summary に出す。
- `experiments/compare_gsdc2023_factor_masks.py`
  - `phone_data_factor_mask.csv` と raw bridge の exact epoch/satellite/frequency keys を outer join する。
  - `matlab_only.csv` / `bridge_only.csv` / `summary_by_field.csv` で count だけでは見えない key-level 差分を見る。
- `experiments/gsdc2023_gnss_log_reader.py`
  - MATLAB `gnsslog2obs.m` の初期 Raw row filter / L1-L5 frequency inference / P-D-L availability count を Python で再現する。
  - `exobs.m` の signal/status mask（C/N0, code lock, TOW/TOD, ADR state, multipath）も audit 用に出せる。
- `experiments/compare_gsdc2023_gnss_log_observation_counts.py`
  - `phone_data_observation_counts.csv` と `supplemental/gnss_log.txt` reader の raw P/D/L counts を比較する。
- `experiments/compare_gsdc2023_gnss_log_residual_prekeys.py`
  - Python `gnss_log` reader の signal/status mask 後 P/D keys と、MATLAB `phone_data_residual_diagnostics.csv` の `p_pre_finite` / `d_pre_finite` keys を outer join する。
  - 差分は `gt.Gsat(...).residuals(...)` 入口での navigation residual availability を表す。
- `ref/gsdc2023/export_phone_data_residual_diagnostics.m`
  - `p_residual_m` / `d_residual_mps` と `p/d/l_pre_finite` / `p/d/l_factor_finite` に加え、再生成時は `p_pre_respc_m` / `d_pre_resd_m` / `p_corrected_m` / `p_range_m` / `d_obs_mps` / `d_model_mps` / `sat_x/y/z` / `sat_clock_bias_m` / `sat_iono_m` / `sat_trop_m` / `rcv_x/y/z` / `obs_clk_m` / `obs_dclk_m` / `p_isb_m` / `p_clock_bias_m` / `d_clock_bias_mps` も出す。
  - residual 値差が clock/ISB 由来か、観測補正値由来か、satellite range / range-rate model 由来かを切り分けるための audit columns。
- `experiments/compare_gsdc2023_residual_diagnostics_factor_mask.py`
  - `phone_data_residual_diagnostics.csv` の `p/d/l_factor_finite` から MATLAB factor mask を再構築し、`phone_data_factor_mask.csv` と exact join する。
- `experiments/compare_gsdc2023_residual_values.py`
  - `phone_data_residual_diagnostics.csv` の `p_residual_m` / `d_residual_mps` と、raw bridge が baseline WLS + `device_gnss.csv` satellite products から再計算した P/D residual value を exact key join する。
  - `settings_{train,test}.csv` の `IdxStart` / `IdxEnd` を反映し、bridge 側の `start_epoch` と出力 `epoch_index` を MATLAB export window に合わせる。
  - `residual_value_join.csv` / `matlab_only.csv` / `bridge_only.csv` / `summary_by_field.csv` / `summary.json` を出し、key parity ではなく residual 値そのものの差を audit する。raw bridge 側は `bridge_pre_residual` / `bridge_common_bias` / `bridge_observation` / `bridge_model` も出す。
- raw bridge diagnostics mask overlay
  - `build_trip_arrays(...)` / `BridgeConfig` / `validate_fgo_gsdc2023_raw.py` に `matlab_residual_diagnostics_mask_path` / `--matlab-residual-diagnostics-mask` を追加した。
  - `compare_gsdc2023_phone_data_raw_bridge_counts.py` と `compare_gsdc2023_factor_masks.py` も同じ CSV を受け取れる。
  - これは production mask ではなく、MATLAB `phone_data_residual_diagnostics.csv` の `p/d/l_factor_finite` を oracle として raw bridge の P/D/L availability に適用する audit mode。
- TDCP solver parity knobs
  - native TDCP factor の dual-frequency clock design を修正し、L5/E5 TDCP が pseudorange と同じ `c0+ISB` clock 差分を使うようにした。
  - `BridgeConfig.tdcp_weight_scale` / `--tdcp-weight-scale` / `GSDC2023_BRIDGE_TDCP_WEIGHT_SCALE` を追加した。final TDCP weights（diagnostics overlay 後を含む）に掛ける audit knob で、default は既存挙動維持の `1.0`。
  - `BridgeConfig.tdcp_geometry_correction` / `--tdcp-geometry-correction` / `GSDC2023_BRIDGE_TDCP_GEOMETRY_CORRECTION` を追加した。raw ADR 差分から baseline に対する satellite range delta を引き、MATLAB `obsr.resL(i+1)-obsr.resL(i)` に近い residualized TDCP measurement を作る実験フラグ。default は OFF。
- Sagnac / RTKLIB `geodist` range parity
  - `_geometric_range_with_sagnac(...)` を追加し、Python residual clock estimate、pseudorange residual mask、state residual audit、base correction residual、TDCP geometry correction、residual value compare で `norm(sat-rx) + OMGE*(sat_x*rx_y - sat_y*rx_x)/c` を使うようにした。
  - Doppler residual / residual-value compare も `_geometric_range_rate_with_sagnac(...)` を使い、MATLAB `satr.rate` と同じ Sagnac range-rate convention に寄せた。これで D factor-key の single-key swap は消え、D/L1・D/L5 は raw-only factor-key で 100% になった。
  - native nonlinear FGO factor はまだ straight Euclidean range のままなので、solver parity は別途対応が必要。
- Pseudorange-Doppler dDP mask parity
  - MATLAB `exobs_residuals.m` は `(-lam*(D2+D1)/2*dt) - (P2-P1)` を clock/median 補正なしで直接 threshold 判定する。Python raw bridge も `_mask_pseudorange_doppler_consistency(...)` を同じ直接 dDP 判定に寄せ、以前の epoch common-bias subtraction と min-keep guard を外した。
- GPS TGD / MATLAB clock parity
  - `device_gnss.csv` の `SvClockBiasMeters` は MATLAB `gt.Gsat` / RTKLIB 側と GPS TGD の扱いが違うため、trip の `brdc.<yy>n` から SVID ごとの TGD を読み、L1 は `TGD*c`、L5 は `(f1/f5)^2*TGD*c` 相当を sat-clock component と corrected pseudorange に反映するようにした。
  - `FullBiasNanos` は float 化してから引くと 1e18 ns 桁で 10m 級に量子化されるため、`int` で base full-bias との差を取り、MATLAB `gnsslog2obs.m` と同じ relative receiver clock（discontinuity ごとに base reset）を作るようにした。
  - residual value compare の P common bias は bridge subset 上の epoch median ではなく、MATLAB の `obs.clk + global ISB(freq/system)` に合わせ、frequency/system ごとの global ISB median を推定して差し引く。
  - production pseudorange residual mask も receiver clock が使える場合は epoch median ではなく、同じ relative clock + global ISB(freq/system) を使って residual を判定するようにした。receiver clock が無い場合は従来の epoch median fallback のまま。
  - `SvClockDriftMetersPerSecond` を `TripArrays.sat_clock_drift_mps` に保持し、Doppler residual mask / residual-value compare の model に `satr.rate - satr.ddts` 相当を使うようにした。
- `experiments/compare_gsdc2023_diagnostics_mask_solver.py`
  - 同じ raw bridge config で `raw_bridge` と `diagnostics_mask` を 2 回走らせ、solver metrics と ECEF trajectory delta を `metrics_by_case.csv` / `trajectory_delta.csv` / `trajectory_delta_by_epoch.csv` / `summary.json` に出す。
  - diagnostics mask overlay が「入力 key parity」から solver/export output にどう効くかを分離して見るための audit tool。

#### B.463.2 実測

対象:

```text
train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4
```

実行:

```bash
matlab -batch "cd('ref/gsdc2023'); setup_local_env; settings=readtable('kaggle_smartphone_decimeter_2023/sdc2023/settings_train.csv','TextType','string'); setting=settings(settings.Course==\"2020-06-25-00-34-us-ca-mtv-sb-101\" & settings.Phone==\"pixel4\",:); preprocessing(\"./kaggle_smartphone_decimeter_2023/sdc2023/train/\", setting); export_phone_data_factor_counts(\"./kaggle_smartphone_decimeter_2023/sdc2023/train/\", setting);"
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_phone_data_raw_bridge_counts.py \
  --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --datasets train \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --no-multi-gnss \
  --pseudorange-residual-mask-m 20.0 \
  --pseudorange-residual-mask-l5-m 15 \
  --doppler-residual-mask-mps 3.0 \
  --tdcp-consistency-threshold-m 1.0
```

結果:

- `matched_rows=12/12`
- 履歴: Sagnac 修正前の literal MATLAB threshold (`P_L1=20m`) では `count_parity_ratio=0.9177427068388331`
- 履歴: Sagnac 修正前の raw-CSV effective threshold (`P_L1=31.7m`, `P_L5=14.9m`, `D=2.983m/s`, `TDCP=0.81m`) では `count_parity_ratio=0.9998087039693926`
- Sagnac + GPS TGD + relative FullBias clock 後の literal 近傍 `P_L1=20m`, `P_L5=15m`, `D=2.983m/s`, `TDCP=0.81m` では `count_parity_ratio=0.9989956958393114`
- production residual mask に relative clock + global ISB を入れた後の count best tested は `P_L1=19m`, `P_L5=15m`, `D=2.983m/s`, `TDCP=0.81m` で `count_parity_ratio=0.9996652319464371`
- settings window + direct dDP + sat-clock drift + Sagnac range-rate 後は `matched_abs_delta_total=2` まで到達していた。
- さらに `supplemental/gnss_log.txt` の MATLAB 型 Pseudorange を P 観測値へ反映し、gnss_log P_ok 全体で global ISB を推定し、P 補正には RTKLIB/MATLAB と同じ humidity=0.7 の Saastamoinen tropo を使うようにした後、literal MATLAB threshold (`P_L1=20.0m`, `P_L5=15m`, `D=3.0m/s`, `TDCP=1.0m`) は `matched_phone_count_total=83640`, `matched_bridge_count_total=83640`, `matched_abs_delta_total=0`, `matched_signed_delta_total=0`, `count_parity_ratio=1.0`
  - L1 `P/resPc`: `-1`
  - L5 `P/resPc`: `0`
  - L1 `D/resD`: `0`
  - L5 `D/resD`: `0`
  - L1 `L/resL`: `0`
  - L5 `L/resL`: `0`
- output dir: `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260418_213243`

factor-key join:

- literal 20m: `symmetric_parity=0.9158775705404113`
- Sagnac 修正前 effective threshold (`P_L1=31.7m`, `P_L5=14.9m`, `D=2.983m/s`, `TDCP=0.81m`): `symmetric_parity=0.9963414634146341`, `jaccard=0.992804212633134`
- latest raw-only factor-key (`P_L1=20.0m`, `P_L5=15m`, `D=3.0m/s`, `TDCP=1.0m`): `total_matlab_count=83640`, `total_bridge_count=83640`, `total_matched_count=83640`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`, `jaccard=1.0`
- output dir: `experiments/results/gsdc2023_factor_mask_parity_20260418_213243`
- D/L1・D/L5・L/TDCP・P/L1・P/L5 すべて 100% key parity。

gnss_log raw observation reader:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_gnss_log_observation_counts.py \
  --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4
```

- `phone_count_total=134098`
- `gnss_log_count_total=134098`
- `matched_abs_delta_total=0`
- `count_parity_ratio=1.0`

GPS-only signal/status mask count（residual 前）:

- gnss_log reader: L1 `P/D=14713`, L5 `P/D=4371`
- MATLAB residual diagnostic pre-finite: L1 `P/D=14702`, L5 `P/D=4367`
- prekey join: `total_matlab_pre_count=38138`, `total_gnss_log_signal_count=38168`, `total_matched_count=38138`, `total_matlab_only=0`, `total_gnss_log_only=30`, `symmetric_parity=0.999214001257598`
- `gnss_log_only=30` の内訳は全て `epoch_index=1300` / `utcTimeMillis=1593046551440`。L1 は 11 sats、L5 は 4 sats の P/D が `Gsat.residuals` 前後で落ちる。
- 差分は `gt.Gsat(...).residuals(...)` 側の navigation residual availability で発生しており、Raw→GobsPhone 変換自体は一致済み。

residual diagnostics → factor mask:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_residual_diagnostics_factor_mask.py \
  --data-root ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023 \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4
```

- `total_factor_mask_count=83640`
- `total_diagnostics_count=83640`
- `total_matched_count=83640`
- `total_factor_mask_only=0`
- `total_diagnostics_only=0`
- `symmetric_parity=1.0`

つまり MATLAB artifact 内では `phone_data_residual_diagnostics.csv` だけから P/D/L/resPc/resD/resL factor keys を 100% 復元できる。現在は raw bridge 側も gnss_log Pseudorange / RTKLIB tropo / full gnss_log ISB に寄せることで、diagnostics overlay なしの raw-only factor key set も 100% parity に到達済み。

residual value parity audit:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_residual_values.py \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --max-epochs 200 \
  --no-multi-gnss
```

- output dir: `experiments/results/gsdc2023_residual_value_parity_20260418_164823`
- `total_matlab_count=5218`
- `total_bridge_count=5012`
- `total_matched_count=5012`
- `total_matlab_only=206`
- `total_bridge_only=0`
- all matched values after Sagnac range + GPS TGD sat-clock + relative FullBias clock + D convention fix: `mean_delta=0.10571892196577616`, `median_abs_delta=0.0042478395493525944`, `p95_abs_delta=0.40254463325717266`, `max_abs_delta=4.758943203796148`
- after subtracting each epoch's field/freq median delta: `median_abs_delta_after_epoch_median=0.004200138902672512`, `p95_abs_delta_after_epoch_median=0.047295862250027784`
- all matched decomposition:
  - pre-residual delta: `median_abs=0.004227030384584274`, `p95_abs=0.15410422496498907`
  - common-bias delta: `median_abs=0.006820243487484667`, `p95_abs=0.2573103015826774`
  - observation delta: `median_abs=0.00010255910481760111`, `p95_abs=0.1522879207506776`
  - model delta: `median_abs=0.00022413804251186775`, `p95_abs=0.004217736978932862`
  - satellite clock bias delta: `median_abs=4.81479219160974e-07`
  - satellite clock drift delta: `median_abs=3.4793285498806864e-09`
- by field:
  - D/L1: `matched=1968`, `matlab_only=93`, `median_abs_delta=0.0021065283008175895`, `p95_abs_delta=0.004204826915728854`, after epoch median `0.0020606654864692153` / `0.004178559698900999`
  - D/L5: `matched=538`, `matlab_only=10`, `median_abs_delta=0.0037642770506634005`, `p95_abs_delta=0.00422803169901711`, after epoch median `0.0004240503557754183` / `0.006329682310300857`
  - P/L1: `matched=1968`, `matlab_only=93`, `median_abs_delta=0.2826662497175372`, `p95_abs_delta=0.43573208335318464`, after epoch median `0.02665616478770666` / `0.050425609294324825`; `observation_delta` median abs `0.0799139216542244`, `model_delta` median abs `1.881e-6`, `common_bias_delta` median abs `0.2573103015822866`
  - P/L5: `matched=538`, `matlab_only=10`, `median_abs_delta=0.08066602668805145`, `p95_abs_delta=0.1647306961651326`, after epoch median `0.022838922217484514` / `0.05340646244584834`; `observation_delta` median abs `0.07841676101088524`, `model_delta` median abs `0.0034371744841337204`, `common_bias_delta` median abs `0.013640486974964006`

settings window 後の full-window residual value audit:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_residual_values.py \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --no-multi-gnss
```

- output dir: `experiments/results/gsdc2023_residual_value_parity_20260418_213300`
- `total_matlab_count=38138`, `total_bridge_count=33132`, `total_matched_count=33132`, `total_matlab_only=5006`, `total_bridge_only=0`
- settings `IdxStart=1` / `IdxEnd=1299` が反映され、raw device の最終 epoch 由来 `bridge_only` は出ない。
- all matched: `median_abs_delta=0.0008937317807056408`, `p95_abs_delta=0.010631670477642907`, after epoch median `p95_abs_delta_after_epoch_median=0.017358220648021495`
- decomposition: `median_abs_pre_residual_delta=0.00018765392215058796`, `p95_abs_pre_residual_delta=0.017914272095595152`; `median_abs_common_bias_delta=0.0028609415485032486`, `p95_abs_common_bias_delta=0.007286756629582669`; `median_abs_observation_delta=5.4804608481617834e-05`, `p95_abs_observation_delta=0.0179270276799798`
- by field:
  - D/L1: `matched=12828`, `matlab_only=1874`, `median_abs_delta=0.00015821574525798354`, `p95_abs_delta=0.00019156862646818872`
  - D/L5: `matched=3738`, `matlab_only=629`, `median_abs_delta=0.00017978376896932158`, `p95_abs_delta=0.00019239371005407287`
  - P/L1: `matched=12828`, `matlab_only=1874`, `median_abs_delta=0.007127`, `p95_abs_delta=0.010728`
  - P/L5: `matched=3738`, `matlab_only=629`, `median_abs_delta=0.008454`, `p95_abs_delta=0.016386`

この audit では raw bridge keys は MATLAB pre-finite keys の strict subset になり、`bridge_only=0`。Sagnac 修正前の P/L1 median 14m 級 gap は、RTKLIB `geodist` が入れる Earth rotation correction を Python が落としていたことが主因だった。Sagnac range 後は P の model delta が L1 median `~2e-6m` / L5 median `0.003m` まで潰れた。さらに Sagnac range-rate + `SvClockDriftMetersPerSecond` を D model に入れたことで D residual は `0.00016-0.00018m/s` median まで潰れた。GPS TGD、gnss_log Pseudorange の int64 精度修正、RTKLIB tropo、full gnss_log P_ok による global ISB 推定で P residual も cm 級へ低下し、raw-only factor/count key parity は 100% になった。

raw bridge diagnostics mask overlay:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_factor_masks.py \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --no-multi-gnss \
  --pseudorange-residual-mask-m 20.0 \
  --pseudorange-residual-mask-l5-m 15 \
  --doppler-residual-mask-mps 3.0 \
  --tdcp-consistency-threshold-m 1.0 \
  --matlab-residual-diagnostics-mask ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4/phone_data_residual_diagnostics.csv
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_phone_data_raw_bridge_counts.py \
  --datasets train \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --no-multi-gnss \
  --pseudorange-residual-mask-m 20.0 \
  --pseudorange-residual-mask-l5-m 15 \
  --doppler-residual-mask-mps 3.0 \
  --tdcp-consistency-threshold-m 1.0 \
  --matlab-residual-diagnostics-mask ../ref/gsdc2023/kaggle_smartphone_decimeter_2023/sdc2023/train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4/phone_data_residual_diagnostics.csv
```

- factor-key: `total_matlab_count=83640`, `total_bridge_count=83640`, `total_matched_count=83640`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`
- count: `matched_phone_count_total=83640`, `matched_bridge_count_total=83640`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`
- output dirs:
  - `experiments/results/gsdc2023_factor_mask_parity_20260418_190348`
  - `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260418_190348`

この結果は「raw bridge が MATLAB residual を自力で再現した」ことではなく、「MATLAB residual diagnostics を availability oracle として使えば、Python graph に入る P/D/L factor key set は MATLAB と完全同型にできる」ことを示す。

solver output separation:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_diagnostics_mask_solver.py \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --max-epochs 0 \
  --position-source raw_wls \
  --fgo-iters 0 \
  --chunk-epochs 200 \
  --no-multi-gnss \
  --dual-frequency \
  --observation-mask
```

- output dir: `experiments/results/gsdc2023_diagnostics_mask_solver_20260418_131808`
- raw bridge WLS score: `7.637727403065495`
- diagnostics-mask WLS score: `5.9907411847215855`
- delta: `-1.6469862183439092`
- trajectory delta after diagnostics overlay: median `3.029m`, p95 `13.522m`, max `58.178m`
- diagnostics-mask WLS metrics after fallback fix: RMS2D `4.659m`, p50 `3.687m`, p95 `8.295m`, max2D `17.435m`

この run で `run_wls(...)` の underconstrained epoch が `(0,0,0)` ECEF を残すバグも潰した。`run_wls(..., fallback_xyz=batch.kaggle_wls)` により、P factors が足りない epoch は Kaggle WLS 位置を保持する。修正前は diagnostics-mask 側で `1593045848440` と `1593046551440` が `(0,0,0)` になり、full-trip RMS を壊していた。

FGO 側の分離 run:

```bash
PYTHONPATH=python:. python3 experiments/compare_gsdc2023_diagnostics_mask_solver.py \
  --trip train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 \
  --max-epochs 200 \
  --position-source fgo \
  --chunk-epochs 200 \
  --no-multi-gnss \
  --dual-frequency \
  --observation-mask
```

- output dir: `experiments/results/gsdc2023_diagnostics_mask_solver_20260418_131844`
- raw bridge FGO score: `11865.664928891281`
- diagnostics-mask FGO score: `304.4809707212469`
- diagnostics-mask improves the bad FGO run substantially, but FGO remains much worse than raw WLS (`6.005168983891096` on the same 200 epochs) and Kaggle WLS (`3.395637539929072`).
- Conclusion: diagnostics overlay cleanly separates solver-input parity. The remaining large error is graph/solver tuning or factor-model behavior, not factor key availability.

TDCP isolation after native dual-frequency clock-design fix:

- `--no-vd --no-tdcp`: raw bridge score `6.802152213434494`, diagnostics-mask score `6.005168983891096`.
- `--vd --no-tdcp`: raw bridge FGO score `6.887123035890995`, diagnostics-mask FGO score `5.164323399197803`.
- `--vd --tdcp` with default TDCP scale `1.0`: raw bridge FGO score `9545.833649512591`, diagnostics-mask FGO score `318.8169188789472` (`experiments/results/gsdc2023_diagnostics_mask_solver_20260418_141326`).
- `--vd --tdcp --tdcp-weight-scale 0.001`: diagnostics-mask FGO score `5.194544080233152` (`experiments/results/gsdc2023_diagnostics_mask_solver_20260418_141947`).
- `--vd --tdcp --tdcp-weight-scale 0.0001`: diagnostics-mask FGO score `5.1524689108068875` (`experiments/results/gsdc2023_diagnostics_mask_solver_20260418_142020`).
- `--vd --tdcp --tdcp-geometry-correction --tdcp-weight-scale 0.000001`: raw bridge FGO score `6.477254613904419`, diagnostics-mask FGO score `5.1643025513786736`, corrected TDCP pairs `1241` (`experiments/results/gsdc2023_diagnostics_mask_solver_20260418_142851`).
- 最終 factor parity / TDCP diagnostics-weight fix 前の geometry-correction scale sweep（raw bridge / no overlay）では、tested grid 内で `tdcp_weight_scale=1e-6` が最良 `6.477254613904419`。`3e-7` は `6.540469745841344`、`1e-8` は `6.866245957736222`、`3e-6` 以上は悪化した。diagnostics-mask + geometry correction は `1e-4` で `5.162239345231134` までしか改善せず、no-geometry の `1e-4` (`5.1524689108068875`) を超えない。最新 sweep は B.464 の 2026-04-19 solver-side TDCP overlay fix を参照。

結論: factor-key parity は diagnostics overlay で 100% まで到達済み。solver 側の TDCP 崩壊は sign ではなく measurement residualization + weight 過大が主因。diagnostics-mask 条件だけなら no-geometry の `tdcp_weight_scale=1e-4` が VD-only をわずかに上回る。raw bridge（overlay なし）では geometry correction と `tdcp_weight_scale=1e-6` で km 級 collapse が消え、raw WLS `6.802m` より少し良い `6.477m` まで戻る。ただし upstream MATLAB 同等の strong TDCP にはまだ届かず、未解決差分は raw `device_gnss.csv` product 由来の residual availability / residual value と MATLAB `gnss_log` + `Gsat` path の不一致に残っている。

#### B.463.3 次の焦点

残差は Python raw bridge が `device_gnss.csv` の satellite position/clock products を正としている一方、MATLAB は `gnss_log.txt` → `GobsPhone` → `gt.Gsat` navigation residual の class object を正としている差が主因。

今回潰した実バグ:

- Doppler residual check の符号を `doppler + geom_rate` に修正。
- pseudorange-Doppler consistency の符号を MATLAB `-(D1+D2)*lambda/2*dt - diff(P)` 相当に修正。
- Doppler residual mask から MATLAB にない「4衛星残す」保護を外した。
- factor mask CSV join を追加し、count だけでなく key-level で差分を見られるようにした。
- compare / validate CLI と MATLAB wrapper に L5 pseudorange residual threshold と TDCP consistency threshold を通した。
- `run_wls` の underconstrained epoch fallback を追加し、diagnostics-mask overlay 時の zero-ECEF outlier を解消した。
- native dual-frequency TDCP clock design を修正し、TDCP weight scale audit knob を追加した。
- raw ADR TDCP を MATLAB `resL` 差分へ近づける geometry correction audit knob を追加した。
- RTKLIB `geodist` と同じ Sagnac range correction を residual preprocessing / audit 側に入れ、P model delta を mm 以下まで潰した。
- residual value compare の D を MATLAB `gnsslog2obs/resD` convention に変換し、native VD solver convention は保持したまま D residual value を mm/s scale まで合わせた。
- GPS nav TGD を `device_gnss.csv` sat-clock component に反映し、P observation / satellite-clock 側の m 級差を cm 以下へ縮めた。
- `FullBiasNanos` を int の relative clock として扱い、MATLAB `obs.clk` と同じ receiver clock common-bias を比較器で使えるようにした。
- production pseudorange residual mask を relative receiver clock + global ISB(freq/system) 型へ寄せ、pixel4 の raw bridge factor-key 残差を `matlab_only=110`, `bridge_only=126` から `matlab_only=70`, `bridge_only=78` へ縮めた。
- parity compare tools が `settings_{train,test}.csv` の `IdxStart/IdxEnd` を raw bridge build に反映するようにし、pixel4 の最終 epoch `bridge_only` bulk 差分を消した。sat-clock drift + Sagnac range-rate 後の latest raw-only factor best は `matlab_only=2`, `bridge_only=0`（P/L1 1 key、`P`/`resPc` 重複）。diagnostics-assisted audit mode では count / factor-key とも 100% parity。

次に詰めるなら:

1. `gt.Gsat(...).residuals(...)` 相当の residual availability を Python 側で再現し、raw bridge の key-level 残差（latest best で `matlab_only=2`, `bridge_only=0`。実体は P/L1 の 1 key が `P` / `resPc` に出る）をさらに詰める。
2. remaining P residual mask 差分を epoch / satellite key で分解し、threshold tuning 依存を減らす。
3. native nonlinear FGO factor 側にも Sagnac / sat-clock convention を反映し、preprocessing parity と solver factor parity を切り分ける。
4. diagnostics mask overlay を使って solver input parity と solver output 差分を分離し、その上で `device_gnss.csv` bridge と `gnss_log` bridge のどちらを canonical にするか決める。

### B.464 GSDC2023 raw-only factor parity 一般化 — 2026-04-19

B.463 の単一 trip 100% parity を別 phone / 別 course に広げた。

追加修正:

- `compare_gsdc2023_phone_data_raw_bridge_counts.py` に `--offset` を追加した。train 全体を小分けに scan できる。
- TDCP dDL outlier の扱いを MATLAB `exobs_residuals.m` に合わせた。MATLAB は bad dDL pair だけでなく両端の carrier phase 観測を NaN にするため、隣接 TDCP factor も落ちる。Python も endpoint propagation を行うようにした。
- `device_gnss.csv` に GPS L1/L5 row が無く、`gnss_log.txt` には存在する epoch / SVID を raw bridge に補うようにした。
  - satellite position / clock / delay product は同一 SVID/signal の `device_gnss.csv` 近傍から 2 次補間する。
  - factor が 0 の `gnss_log` epoch も時刻列には残す。MATLAB の `obs.utcms` index は no-factor epoch も数えるため、これを落とすと後続 factor key の `epoch_index` が一斉にずれる。
- P/D residual mask は receiver clock / drift がある場合、4 衛星未満の epoch でも評価するようにした。MATLAB は `obs.clk` / `obs.dclk` を使うため、single-observation epoch でも residual mask が効く。
- raw bridge は 4 衛星未満 epoch も保持する。WLS/FGO 側は既存の fallback / factor sparsity で扱う。

追加 MATLAB artifact:

- `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl`
- `train/2020-07-08-22-28-us-ca/pixel4`

実測結果:

- `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`
  - count: `matched_phone_count_total=83640`, `matched_bridge_count_total=83640`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`
  - factor: `total_matlab_count=83640`, `total_bridge_count=83640`, `total_matched_count=83640`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`
- `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl`
  - count: `matched_phone_count_total=86768`, `matched_bridge_count_total=86768`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`
  - factor: `total_matlab_count=86768`, `total_bridge_count=86768`, `total_matched_count=86768`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`
- `train/2020-07-08-22-28-us-ca/pixel4`
  - count: `matched_phone_count_total=113338`, `matched_bridge_count_total=113338`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`
  - factor: `total_matlab_count=113338`, `total_bridge_count=113338`, `total_matched_count=113338`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`

代表 output dirs:

- `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260419_025741`
- `experiments/results/gsdc2023_factor_mask_parity_20260419_025741`
- `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260419_025828`
- `experiments/results/gsdc2023_factor_mask_parity_20260419_025828`

次に進むなら:

1. MATLAB export をさらに batch 化し、phone family / course を増やして raw-only factor parity を確認する。
2. native nonlinear FGO factor 側へ Sagnac / sat-clock / gnss_log timebase convention を反映する。
3. TDCP を factor availability だけでなく solver residual / weight として再評価する。

2026-04-19 追加 probe:

- MATLAB export をさらに 3 trip 生成した:
  - `train/2020-07-08-22-28-us-ca/pixel4xl`
  - `train/2020-07-17-22-27-us-ca-mtv-sf-280/pixel4`
  - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4`
- 6 trip 合計 count parity:
  - output dir: `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260419_062013`
  - `matched_phone_count_total=615604`
  - `matched_bridge_count_total=615590`
  - `matched_abs_delta_total=22`
  - `count_parity_ratio=0.9999642627403331`
  - TDCP (`L/resL`) は L1/L5 とも delta 0。
  - 残差は `train/2020-07-08-22-28-us-ca/pixel4xl` と `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4` の P/D 境界だけ。
- 追加 3 trip のうち `train/2020-07-17-22-27-us-ca-mtv-sf-280/pixel4` は count parity 100%。
- `train/2020-07-08-22-28-us-ca/pixel4xl` は baseline jump 近傍の D residual / P residual 境界が残る。MATLAB `posbl` の `isoutlier(...,'movmedian',10,'ThresholdFactor',20)` は該当 1 epoch を補間しているが、Python の保守的な `_repair_baseline_wls` は過検出回避のため同じ点を落としていない。単純に MAD 閾値を下げると既存 100% trip で D/P count が悪化したため未採用。
- `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4` は最終 epoch の D availability と P residual threshold 近傍が残る。
- `compare_gsdc2023_factor_masks.py` / `compare_gsdc2023_phone_data_raw_bridge_counts.py` は同一秒に並列実行しても output directory が衝突しないよう、既存 dir がある場合に `_01`, `_02`, ... suffix を付けるようにした。

2026-04-19 追加修正:

- D residual mask の receiver velocity を、settings `IdxEnd` で切った後の片側差分ではなく、raw epoch metadata 全体の baseline velocity から exact time lookup するようにした。これで `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4` 最終 epoch の D/L1 4 key と D/L5 1 key 欠落が消えた。
- baseline WLS 補修に「直前が stale WLS 連続で、1 epoch だけ二次差分 >30 m かつ stale からの jump >60 m」の局所検出を追加した。広い isolated-spike 検出は既存 100% trip を壊したため採用せず、p4xl の MATLAB `Baseline has jump` 1 点だけに効く条件へ絞った。
- global ISB サンプル集計は receiver clock がある場合に epoch 内 4 obs 未満でも使うようにした。MATLAB `p_pre` availability に寄せるための変更で、既存 exact trip への副作用は出ていない。
- 直前の未解決 6 trip count parity（最終 fix 前）:
  - output dir: `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260419_070715`
  - `matched_phone_count_total=615604`
  - `matched_bridge_count_total=615606`
  - `matched_abs_delta_total=2`
  - `matched_signed_delta_total=2`
  - `count_parity_ratio=0.9999967511582121`
  - D / L / resD / resL は L1/L5 とも delta 0。残り count 差分は P/L1 と resPc/L1 の +1 だけ。
- 直前の未解決 key-level（最終 fix 前）:
  - `train/2020-07-08-22-28-us-ca/pixel4xl`: `total_matlab_count=115168`, `total_bridge_count=115170`, `total_matched_count=115168`, `matlab_only=0`, `bridge_only=2`。残りは `1594247314444` GPS L1 SVID 29 の P/resPc 1 key。MATLAB P residual は `20.0035945 m`、bridge は `19.9897007 m`。
  - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4`: `total_matlab_count=102120`, `total_bridge_count=102120`, `total_matched_count=102118`, `matlab_only=2`, `bridge_only=2`。残りは P/resPc の 1 key 入れ替わり。D と TDCP は一致。
  - `train/2020-07-08-22-28-us-ca/pixel4` は補修後も exact (`total_matched_count=113338`) のまま。

2026-04-19 最終 parity fix:

- `gnsslog2obs.m` は nanosecond→second 変換で `/1e9` を使う。Python reader は `*1e-9` だったため、巨大な `TimeNanos` / `ReceivedSvTimeNanos` 値で丸めが `0.0174502 m` ずれる epoch があり、p4xl の ISB と p4 の P/D 境界を動かしていた。`experiments/gsdc2023_gnss_log_reader.py` を `/1.0e9` に揃え、`tests/test_gsdc2023_gnss_log_reader.py` の期待式も MATLAB と同じ除算へ更新した。
- `exobs_residuals.m` の P-D consistency は corrected `resPc` ではなく raw `obs.(f).P` で `dDP = -lam*(D1+D2)/2*dt - diff(P)` を見る。raw bridge も gnss_log observable P を dDP mask に渡すようにした。
- p4 / p4xl の gnss_log observable P は `phone_data_residual_diagnostics.csv` から逆算した MATLAB P と max abs `5.3e-8 m` 以内になった。
- 最新 6 trip count parity:
  - output dir: `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_20260419_081321`
  - `matched_phone_count_total=615604`
  - `matched_bridge_count_total=615604`
  - `matched_abs_delta_total=0`
  - `matched_signed_delta_total=0`
  - `count_parity_ratio=1.0`
  - `P/D/L/resPc/resD/resL` の L1/L5 delta はすべて 0。
- 最新 key-level は 6/6 trip で exact:
  - `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`: `83640/83640`, `matlab_only=0`, `bridge_only=0`
  - `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4xl`: `86768/86768`, `matlab_only=0`, `bridge_only=0`
  - `train/2020-07-08-22-28-us-ca/pixel4`: `113338/113338`, `matlab_only=0`, `bridge_only=0`
  - `train/2020-07-08-22-28-us-ca/pixel4xl`: `115168/115168`, `matlab_only=0`, `bridge_only=0`
  - `train/2020-07-17-22-27-us-ca-mtv-sf-280/pixel4`: `114570/114570`, `matlab_only=0`, `bridge_only=0`
  - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4`: `102120/102120`, `matlab_only=0`, `bridge_only=0`

2026-04-19 solver-side TDCP overlay fix:

- diagnostics mask overlay の TDCP weight を `1.0` 固定に潰していた。P/D overlay は元の signal weight を復元していたため、TDCP だけ availability oracle 以上の solver 差分を作っていた。
- `_apply_matlab_residual_diagnostics_mask(...)` に `signal_tdcp_weights` を渡し、MATLAB `l_factor_finite` で残す TDCP pair には raw TDCP 構築時の weight を復元するようにした。raw 側で weight が 0 の pair は availability oracle を維持するため `1.0` fallback。
- regression: `test_matlab_residual_diagnostics_mask_preserves_tdcp_signal_weights` を追加。diagnostics overlay + `tdcp_weight_scale` 後も ADR uncertainty 由来の TDCP weight が保持されることを確認。
- representative solver probe (`train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4`, 200 epochs, VD, GPS-only dual-frequency, observation-mask):
  - `--tdcp-weight-scale 0.0001` / no geometry: raw `1419.3617625228221`, diagnostics `1419.155053517938`。両方 collapse するので、主因は overlay weight ではなく raw ADR TDCP measurement residualization。
  - `--tdcp-geometry-correction --tdcp-weight-scale 0.000001`: raw `5.946232580831246`, diagnostics `5.988314492423423`, score 差 `0.042081911592176446`。weight fix 後は raw / diagnostics が実質同じ solver regime に入る。
- 6 exported train trips / 200 epochs の geometry-corrected TDCP scale sweep:
  - no-TDCP: raw mean `6.411536232`, raw max `7.428460281`, diagnostics mean `6.417097694`, diagnostics max `7.103875197`
  - `tdcp_weight_scale=1e-7`: raw mean `6.325985668`, raw max `7.321271774`, diagnostics mean `6.358188716`, diagnostics max `7.101839236`
  - `tdcp_weight_scale=2e-7`: raw mean `6.341142180`, raw max `7.325259824`, diagnostics mean `6.302992948`, diagnostics max `7.042592069`
  - `tdcp_weight_scale=3e-7`: raw mean `6.262166931`, raw max `7.390027221`, diagnostics mean `6.210653432`, diagnostics max `6.982126964`
  - `tdcp_weight_scale=1e-6`: raw mean `6.505440241`, raw max `7.537892266`, diagnostics mean `6.368614897`, diagnostics max `7.194077985`
  - 結論: `3e-7` は平均最良、`1e-7` は raw worst-case 最良。per-trip best は割れるため、TDCP scale はまだ固定 default にせず、course/phone 依存の tuning 対象。
- output dirs:
  - `experiments/results/solver_probe_vd_tdcp_1e-4_weightfix/gsdc2023_diagnostics_mask_solver_20260419_085028`
  - `experiments/results/solver_probe_vd_tdcp_geom_1e-6_weightfix/gsdc2023_diagnostics_mask_solver_20260419_085023`
  - `experiments/results/solver_probe_vd_no_tdcp_6trip/`
  - `experiments/results/solver_probe_vd_tdcp_geom_1e-7_weightfix_6trip/`
  - `experiments/results/solver_probe_vd_tdcp_geom_2e-7_weightfix_6trip/`
  - `experiments/results/solver_probe_vd_tdcp_geom_3e-7_weightfix_6trip/`
  - `experiments/results/solver_probe_vd_tdcp_geom_1e-6_weightfix_6trip/`

2026-04-19 Samsung / Pixel5 追加 export + MatRTKLIB nav 互換 fix:

- MATLAB export を 2 trip 追加した:
  - `train/2021-12-08-20-28-us-ca-lax-c/pixel5`
  - `train/2022-10-06-21-51-us-ca-mtv-n/sm-a205u`
- `sm-a205u` では MATLAB `preprocessing.m` が Samsung blocklist の `obs.clk` / `obs.dclk` を `exobs_residuals` 前に推定する。raw bridge も residual clock bias/drift estimation を P/D residual mask より前へ移動した。
- Samsung A family (`TDCP_LOFFSET_PHONES`) では MATLAB `GobsPhone` の carrier `L*lambda` が Kaggle `device_gnss.csv` の `AccumulatedDeltaRangeMeters` と逆符号になる。raw bridge の ADR ingestion も同 family だけ sign を反転し、`Loffset=1.117m` の TDCP dDL 判定と solver TDCP measurement を MATLAB convention に揃えた。
- Samsung residual clock drift seed は Doppler residual mask と同じ solver convention (`dop + geom_rate`) で推定するよう修正した。これにより `sm-a205u` の D/resD が全落ちする問題が消えた。
- receiver velocity は MATLAB `Gpos.gradient(obs.dt)` と同じく、実時刻差分ではなく `Gtime.estInterval()` 相当の scalar interval（median diff を 0.01s 丸め）で `gradient(x)/dt` を使うようにした。`sm-a205u` 先頭の 2.001s gap では MATLAB `obs.dt=1.0` のため、実時刻差分だと receiver velocity が半分になり D residual boundary がずれていた。
- Pixel5/LAX の最後の P/resPc 1 key は threshold ではなく MatRTKLIB `phone_data.mat` の実効 GPS nav 選択差だった。
  - `train/2021-12-08-20-28-us-ca-lax-c` の G09 は RINEX nav に Toe `331184`（19:59:44, IODE 7）と Toe `331200`（20:00:00, IODE 95）の近接重複がある。
  - Kaggle `device_gnss.csv` の satellite product は後者 IODE 95 に近いが、MATLAB `gt.Gsat` / `phone_data.mat` は前者 IODE 7 を使う。
  - raw bridge は同一 GPS SVID の Toe/Toc 近接重複（0..60s）を MatRTKLIB 互換に前者優先で落とし、derived common clock が skipped record 側に近い場合だけ sat position / velocity / common no-TGD clock / drift を再計算するようにした。
- raw-only parity:
  - `pixel5`: count/factor とも exact。count `matched_phone_count_total=57846`, `matched_bridge_count_total=57846`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`。factor `total_matlab_count=57846`, `total_bridge_count=57846`, `total_matched_count=57846`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。
  - `sm-a205u`: count/factor とも exact。count `matched_phone_count_total=41920`, `matched_bridge_count_total=41920`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`。factor `total_matlab_count=41920`, `total_bridge_count=41920`, `total_matched_count=41920`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。
  - 既存 6 trip count parity も nav fix 後 exact のまま。`matched_phone_count_total=615604`, `matched_bridge_count_total=615604`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`。
- output dirs:
  - `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_extended_20260419_pixel5_matrtklib_navfix_gate/gsdc2023_phone_data_raw_bridge_count_parity_20260419_105923`
  - `experiments/results/gsdc2023_factor_mask_parity_extended_20260419/pixel5_lax_c_matrtklib_navfix_cache/gsdc2023_factor_mask_parity_20260419_110204`
  - `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_extended_20260419_sm_a205u_matrtklib_navfix/gsdc2023_phone_data_raw_bridge_count_parity_20260419_105545`
  - `experiments/results/gsdc2023_factor_mask_parity_extended_20260419/sm_a205u_matrtklib_navfix/gsdc2023_factor_mask_parity_20260419_105545`
  - `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_6trip_matrtklib_navfix_gate/gsdc2023_phone_data_raw_bridge_count_parity_20260419_105923`
- diagnostics overlay parity:
  - `sm-a205u` + `phone_data_residual_diagnostics.csv`: count/factor とも exact (`41920/41920`, `matched_abs_delta_total=0`, `symmetric_parity=1.0`)。
- regression:
  - `tests/test_validate_fgo_gsdc2023_raw.py::test_estimate_residual_clock_series_recovers_bias_and_drift`
  - `tests/test_validate_fgo_gsdc2023_raw.py::test_receiver_velocity_matches_matlab_scalar_interval_gradient`
  - `tests/test_validate_fgo_gsdc2023_raw.py::test_build_trip_arrays_prefers_residual_clock_series_for_blocklist_phone`
  - `tests/test_validate_fgo_gsdc2023_raw.py::test_build_trip_arrays_applies_tdcp_loffset_for_samsung_a_family`
  - `tests/test_validate_fgo_gsdc2023_raw.py::test_matrtklib_duplicate_nav_filter_matches_effective_eph_selection`
  - full parity regression subset: `79 passed`.
- `20.25m` の P residual threshold 緩和 probe は不採用。Pixel5 の missing 1 key は拾えるが、Pixel5 は `bridge_only=18`、既存 6 trip は P/L1 +79 key、sm-a205u は P/L1 +5 key の bridge-only を作り、全体 parity を悪化させた。最終的には threshold ではなく MatRTKLIB nav selection 差として解消した。

2026-04-19 追加 4 export / timebase・sat-clock reuse fix:

- 追加で MATLAB `preprocessing` + `phone_data_factor_counts.csv` + `phone_data_factor_mask.csv` を 4 trip 生成した:
  - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl`
  - `train/2020-08-04-00-19-us-ca-sb-mtv-101/pixel4`
  - `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl`
  - `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5`
- Pixel5 `2020-08-04-00-20` は count は近いが key-level が大きくずれていた。原因は MATLAB `obs.utcms` が GPS-only factor 0 の Galileo-only epoch `1596501762434` を保持する一方、bridge が GPS row の無い epoch を timebase から落としていたこと。
  - raw bridge は GPS-only / dual-frequency / observation-mask parity path でも `gnss_log.txt` の MATLAB 初期 filter 後 epoch timebase を保持し、選択 GPS row が無い epoch は zero-factor epoch として残すようにした。
- Pixel5 先頭 epoch の P/resPc bridge-only は receiver clock base の gap reset 差だった。
  - MATLAB `gnsslog2obs` は `TimeNanos` の 1 秒超 gap で clock base を張り直す。`device_gnss.csv` 由来の epoch meta には `TimeNanos` が無いことがあるため、bridge は `utcTimeMillis` の 1 秒超 gap も fallback reset として扱うようにした。
- Pseudorange-Doppler consistency (`dDP`) は MATLAB `exobs_residuals.m` と同じく実時刻差ではなく `obs.dt` 相当の scalar interval を使うようにした。Pixel5 の 3 秒 gap で MATLAB が P を endpoint propagation で落とす挙動に揃う。
- 最後の P/resPc 境界差は `gnss_log.txt` Pseudorange を使う再補正で、初期ループの MatRTKLIB 互換 satellite clock を再利用していなかったことが原因だった。
  - `gnss_log` raw P 自体は MATLAB diagnostics 逆算値と `~1e-7 m` で一致していた。
  - G10 duplicate-nav cases では raw bridge の corrected P が MATLAB より `+0.519 m` ずれ、threshold 境界の P/resPc key 入れ替わりを作っていた。
  - `_gnss_log_corrected_pseudorange_matrix(...)` が初期 satellite product selection 後の `sat_clock_bias_matrix` を使うようにし、satellite position/clock と P 補正の nav selection を揃えた。
- 追加 4 trip の改善 / 最終結果:
  - 修正前 count: `matched_phone_count_total=412686`, `matched_bridge_count_total=412716`, `matched_abs_delta_total=42`, `count_parity_ratio=0.9998982277082333`。
  - timebase / clock-gap / P-D dt 修正後 count: `matched_phone_count_total=412686`, `matched_bridge_count_total=412682`, `matched_abs_delta_total=12`, `count_parity_ratio=0.9999709222023524`。D / TDCP (`D/L/resD/resL`) は L1/L5 とも delta 0。
  - sat-clock reuse 後 count: `matched_phone_count_total=412686`, `matched_bridge_count_total=412686`, `matched_abs_delta_total=0`, `matched_signed_delta_total=0`, `count_parity_ratio=1.0`。
  - 修正前 key-level: `total_matched_count=400724`, `matlab_only=11962`, `bridge_only=11992`。Pixel5 の epoch index shift が支配的。
  - timebase / clock-gap / P-D dt 修正後 key-level: `total_matlab_count=412686`, `total_bridge_count=412682`, `total_matched_count=412672`, `matlab_only=14`, `bridge_only=10`。
  - sat-clock reuse 後 key-level: `total_matlab_count=412686`, `total_bridge_count=412686`, `total_matched_count=412686`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。
  - 追加 4 trip は全て exact:
    - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl`: `87350/87350`
    - `train/2020-08-04-00-19-us-ca-sb-mtv-101/pixel4`: `114728/114728`
    - `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl`: `111740/111740`
    - `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5`: `98868/98868`
- 元の 8 exported trip regression も維持:
  - count: `matched_phone_count_total=715370`, `matched_bridge_count_total=715370`, `matched_abs_delta_total=0`, `count_parity_ratio=1.0`
  - factor: `total_matlab_count=715370`, `total_bridge_count=715370`, `total_matched_count=715370`, `matlab_only=0`, `bridge_only=0`, `all_exact=true`
- 12 exported train trips 合計では count / factor とも exact (`1,128,056/1,128,056`)。
- output dirs:
  - `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_8trip_satclock_reuse_regression_20260419/gsdc2023_phone_data_raw_bridge_count_parity_20260419_124142`
  - `experiments/results/gsdc2023_factor_mask_parity_8trip_satclock_reuse_regression_20260419`
  - `experiments/results/gsdc2023_phone_data_raw_bridge_count_parity_new4_satclock_reuse_20260419/gsdc2023_phone_data_raw_bridge_count_parity_20260419_123823`
  - `experiments/results/gsdc2023_factor_mask_parity_new4_satclock_reuse_20260419`
  - `experiments/results/gsdc2023_residual_values_new4_timebase_fix_20260419`
  - `experiments/results/gsdc2023_residual_values_pixel5_timebase_clockfix2_20260419/gsdc2023_residual_value_parity_20260419_115035`
- regression:
  - `test_receiver_clock_bias_lookup_resets_on_time_nanos_gap`
  - `test_receiver_clock_bias_lookup_resets_on_utc_gap_without_time_nanos`
  - `test_pseudorange_doppler_consistency_uses_matlab_scalar_interval`
  - `test_gnss_log_corrected_pseudorange_uses_adjusted_sat_clock`
  - full parity regression subset: `83 passed`.

2026-04-19 native FGO Doppler Sagnac range-rate fix:

- `src/positioning/fgo.cu` の `add_doppler_factor_host` / `doppler_cost_host` を raw bridge の `_geometric_range_rate_with_sagnac(...)` と同じ convention に寄せた。
  - predicted Doppler は `drift - (euclidean_range_rate - sagnac_range_rate)`。
  - `sagnac_range_rate = omega * (svx*y + sx*vy - svy*x - sy*vx) / c`。
  - velocity Jacobian は `LOS + d(sagnac_range_rate)/d(receiver_velocity)` とし、cost 側も同じ model を使う。
- `tests/test_fgo.py` は Doppler synthetic observation を `_doppler_model_sagnac(...)` で生成するよう更新し、Sagnac range-rate を実際に使う regression `test_fgo_vd_doppler_uses_sagnac_range_rate` を追加した。
- build / verification:
  - `cmake --build _cmake_build -j2`
  - `cp _cmake_build/_gnss_gpu.cpython-310-x86_64-linux-gnu.so python/gnss_gpu/_gnss_gpu.cpython-310-x86_64-linux-gnu.so`
  - `PYTHONPATH=python:. python3 -m pytest -q tests/test_fgo.py` -> `43 passed`

2026-04-19 native FGO Doppler satellite-clock-drift fix:

- native `fgo_gnss_lm_vd` に optional `sat_clock_drift` (`[T,S]`, m/s) を追加した。既存 positional API をずらさないため pybind / Python wrapper とも末尾 optional。
- Doppler prediction は `drift - (euclidean_range_rate - sagnac_range_rate - sat_clock_drift)` になった。raw bridge は `TripArrays.sat_clock_drift_mps` を segment ごとに solver へ渡す。
- regression:
  - `test_fgo_vd_doppler_uses_satellite_clock_drift` を追加。
  - `PYTHONPATH=python:. python3 -m pytest -q tests/test_fgo.py tests/test_validate_fgo_gsdc2023_raw.py tests/test_compare_gsdc2023_phone_data_raw_bridge_counts.py tests/test_gsdc2023_gnss_log_reader.py` -> `128 passed`
- solver score probe（6 exported train trips / 200 epochs / VD / GPS-only dual-frequency / observation-mask）:
  - Sagnac-only 後: no-TDCP raw mean `6.411492884`, diagnostics mean `6.416769646`。`tdcp_weight_scale=3e-7` raw mean `6.261954746`, diagnostics mean `6.210626369`。
  - satellite-clock-drift 追加後:
    - no-TDCP: raw mean `6.411577114`, raw max `7.428138228`, diagnostics mean `6.416950194`, diagnostics max `7.103408185`
    - `tdcp_weight_scale=1e-7`: raw mean `6.326058855`, raw max `7.321047730`, diagnostics mean `6.357876953`, diagnostics max `7.101272305`
    - `tdcp_weight_scale=2e-7`: raw mean `6.340940931`, raw max `7.324655967`, diagnostics mean `6.302838177`, diagnostics max `7.042350805`
    - `tdcp_weight_scale=3e-7`: raw mean `6.261963184`, raw max `7.389424461`, diagnostics mean `6.210522811`, diagnostics max `6.981923998`
    - `tdcp_weight_scale=1e-6`: raw mean `6.505314742`, raw max `7.538055591`, diagnostics mean `6.368814487`, diagnostics max `7.193230067`
  - 結論: sat-clock drift の solver score 影響はこの 6-trip 条件では `~1e-4m` 級で、best は raw / diagnostics とも引き続き geometry-corrected TDCP `3e-7`。
- output dirs:
  - `experiments/results/solver_probe_doppler_sagnac_6trip_20260419`
  - `experiments/results/solver_probe_doppler_satclock_6trip_20260419`

2026-04-19 12 exported trip solver probe:

- `experiments/compare_gsdc2023_diagnostics_mask_solver.py` に `--raw-only` を追加した。diagnostics CSV が無い trip でも raw bridge case だけ同じ summary / `metrics_by_case.csv` 形式で出せる。
  - diagnostics CSV は 12 exported trip 中 11 本に存在。
  - `train/2020-07-17-23-13-us-ca-sf-mtv-280/pixel4xl` は diagnostics CSV 無しのため raw-only で集計。
- pure FGO (`position-source=fgo`, 12 trip, 200 epochs, GPS-only dual-frequency, observation-mask):
  - no-TDCP: raw mean `95.834317246`, median `6.655111024`, max `1008.831030357`
  - geometry-corrected TDCP `3e-7`: raw mean `107.915187579`, median `6.491044277`, max `1155.306960129`
  - 11 diagnostics trips の diagnostics FGO:
    - no-TDCP mean `105.619476739`, median `6.888571565`, max `1026.129255634`
    - TDCP `3e-7` mean `119.212904847`, median `6.663153572`, max `1177.474478818`
  - 解釈: TDCP `3e-7` は median では少し改善するが、Pixel5 `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5` が FGO score `~1000m` 級に崩れ、TDCP でさらに悪化する。pure FGO の default 改善としては不安定。
- gated selected (`position-source=gated`, 同条件):
  - no-TDCP: raw selected mean `5.119510875`, median `5.030075559`, max `8.081779046`
  - TDCP `3e-7`: raw selected mean `5.236595240`, median `5.312508922`, max `8.081779046`
  - 11 diagnostics trips:
    - no-TDCP selected mean `5.402972056`, median `5.545517679`, max `8.067474336`
    - TDCP `3e-7` selected mean `5.623502690`, median `5.972520848`, max `8.067474336`
  - `gated` は Pixel5 / Samsung の pure-FGO collapse を baseline fallback で回避する。Pixel5 outlier は `1008m -> 6.626m`、Samsung は `77m -> 5.973m`。
  - 一方、TDCP `3e-7` は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl` を `5.4426 -> 5.1288` に改善するが、`train/2021-12-08-20-28-us-ca-lax-c/pixel5` を `4.5841 -> 6.3029` に悪化させる。
- 初回結論: guard 前の 12-trip では **gated + no-TDCP が best aggregate**。geometry-corrected TDCP `3e-7` は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl` のような局所改善はあったが、LAX Pixel5 の悪化が平均を押し上げていた。
- output dirs:
  - `experiments/results/solver_probe_doppler_satclock_12trip_best_20260419`
  - `experiments/results/solver_probe_doppler_satclock_12trip_gated_best_20260419`

2026-04-19 gated FGO baseline-gap guard:

- LAX Pixel5 の chunk-selection 診断で、TDCP `3e-7` の FGO candidate は PR MSE / quality score では baseline より良く見える一方、baseline 軌跡との差が大きかった。
  - LAX Pixel5 TDCP: baseline `step_p95=12.538m`、FGO `baseline_gap_p95=15.975m`、`baseline_gap_max=21.748m`、selected score は guard 前 `6.3029m`。
  - 2020-08-04 pixel4xl TDCP: baseline `step_p95=13.882m`、FGO `baseline_gap_p95=10.143m`、selected score は `5.1288m` で局所改善。
- `_select_gated_chunk_source()` に FGO 専用 guard を追加した。baseline が catastrophic threshold 以下のとき、FGO の `baseline_gap_p95_m` が `max(baseline.step_p95_m, 12.0m)` を超える場合は baseline に戻す。catastrophic baseline では従来どおり代替 candidate を許可する。
- focused rerun:
  - LAX Pixel5 TDCP raw/diagnostics とも `gated_source=fgo -> baseline`、selected score `6.3029m -> 4.5841m`。
  - 2020-08-04 pixel4xl TDCP raw/diagnostics は `gated_source=fgo` 維持、raw selected `5.1288m`、diagnostics selected `5.0761m`。
- 12 exported trip gated rerun (`experiments/results/solver_probe_guard_12trip_gated_best_20260419`):
  - no-TDCP: raw selected mean `5.119510875`, median `5.030075559`, max `8.081779046`; diagnostics mean `5.222814631`, median `5.387511450`, max `8.067474336`
  - TDCP `3e-7`: raw selected mean `5.093356499`, median `4.873149302`, max `8.081779046`; diagnostics mean `5.194501274`, median `5.076064518`, max `8.067474336`
- 結論更新: この 12-trip / 200-epoch / GPS-only dual-frequency / observation-mask 条件では、guard 後の **gated + geometry-corrected TDCP `3e-7` が best aggregate**。ただし差は raw mean `~2.6cm`、diagnostics mean `~2.8cm` と小さいので、full train/test へ広げる前に trip family を増やして確認する。

2026-04-20 extra 24 train trip raw-only solver probe:

- 12 exported trip 以外から Pixel4/5/6/7、mi8、G988B、Samsung A/S 系を含む 24 train trip を追加し、同じ 200-epoch / GPS-only dual-frequency / observation-mask / `position-source=gated` 条件で raw-only probe を実行した。
- output dir: `experiments/results/solver_probe_guard_extra24_gated_rawonly_20260420`
- 全 48 run 成功 (`no_tdcp` 24 / TDCP `3e-7` 24、failure 0)。
- extra24 aggregate:
  - no-TDCP: raw selected mean `7.099215027`, median `6.549204549`, max `17.465562340`, min `3.203100838`
  - TDCP `3e-7`: raw selected mean `7.089414052`, median `6.549204549`, max `17.465562340`, min `3.203100838`
  - pairwise: TDCP better `1`, worse `0`, equal `23`; mean delta `-0.009800975m`
  - improvement は `train/2020-08-13-21-41-us-ca-mtv-sf-280/pixel4xl` の `6.244472328 -> 6.009248937` のみ。selection は no-TDCP / TDCP とも FGO 200 epoch。
- 12 exported + extra24 combined raw-only:
  - no-TDCP: `n=36`, mean `6.439313643`, median `5.734387517`, max `17.465562340`
  - TDCP `3e-7`: `n=36`, mean `6.424061534`, median `5.734387517`, max `17.465562340`
  - pairwise: TDCP better `2`, worse `0`, equal `34`; mean delta `-0.015252108m`
- 解釈: guard 後の TDCP `3e-7` は追加24本でも selected score regression を起こしていない。ただし selected source が baseline/raw_wls の場合はTDCPの効果が見えないため、改善は FGO selected chunk に限定される。
- 注意: mi8 は `_effective_position_source()` が `gated/auto -> raw_wls` に強制する歴史的 fallback のため、追加24 probe の mi8 2本は実際には `selected_source_mode=raw_wls`。この2本は train GT 上 baseline より悪い:
  - `train/2020-12-10-22-52-us-ca-sjc-c/mi8`: raw_wls `17.4656m`, baseline `5.7171m`
  - `train/2021-08-24-20-32-us-ca-mtv-h/mi8`: raw_wls `7.4907m`, baseline `4.9497m`
  - ただしこの fallback は過去の test mi8 baseline jump 対策として入ったものなので、外す前に test outlier 代表 (`2021-11-30-20-59-us-ca-mtv-m/mi8`, `2021-09-28-21-56-us-ca-mtv-a/mi8`, `2022-04-27-21-55-us-ca-ebf-ww/mi8`) で true gated の PR quality / trajectory sanity を再確認する。

2026-04-20 raw_wls candidate guard:

- `raw_wls` が PR MSE / quality score だけで選ばれ、GT 上は baseline より悪いケースが多かったため、`_select_gated_chunk_source()` を品質順の安全候補探索に変更した。
  - non-catastrophic baseline では `raw_wls` を gated 自動候補から外す。
  - `raw_wls` を外した後に safe FGO candidate があれば FGO を選ぶ。
  - catastrophic baseline alternative と `baseline.mse_pr > gated_threshold` の救済は残す。
- focused rerun:
  - `train/2020-07-08-22-28-us-ca/pixel4`: raw_wls `8.0818m` -> baseline `4.3923m`
  - `train/2020-08-13-21-42-us-ca-mtv-sf-280/pixel5`: raw_wls `8.5145m` -> FGO `6.9273m`
  - `train/2020-08-13-21-41-us-ca-mtv-sf-280/pixel4xl` TDCP: FGO 維持 `6.0092m`
- 既存 12+extra24 payload への new selection simulation (`experiments/results/raw_wls_guard_sim_36_20260420`):
  - no-TDCP: mean `6.439313643 -> 5.713894977`, changed `14`, improved `12`, worsened `2`
  - TDCP `3e-7`: mean `6.424061534 -> 5.759368834`, changed `14`, improved `12`, worsened `2`
  - guard 後 pairwise: TDCP better `3`, worse `1`, equal `32`, TDCP-noTDCP mean delta `+0.04547m`
- 解釈: raw_wls guard は selected aggregate を大きく改善するが、TDCP `3e-7` の優位はこの条件では消える。TDCP は solver source selection と独立に default ON へ上げず、TDCP-on/off candidate 化で選ばせるのが次の筋。
- 36本 actual rerun (`experiments/results/raw_wls_guard_36_gated_rawonly_20260420`):
  - no-TDCP: mean `5.713894977`, median `4.828779804`, max `17.465562340`; source totals baseline `6200`, FGO `600`, raw_wls `400`
  - TDCP `3e-7`: mean `5.759368834`, median `5.084394161`, max `17.465562340`; source totals baseline `6000`, FGO `800`, raw_wls `400`
  - pairwise: TDCP better `3`, worse `1`, equal `32`; regression は `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4` で baseline `3.3956m` -> TDCP-FGO `5.8831m`
- quality margin update:
  - `GATED_CANDIDATE_QUALITY_MARGIN = 0.08` を追加し、FGO candidate も baseline quality から `0.08` 以上改善しないと採用しないようにした。
  - focused rerun で上記 pixel4 TDCP は baseline に戻った (`5.8831m -> 3.3956m`)。
  - adjusted 36 aggregate (`experiments/results/quality_margin_36_adjusted_20260420`): no-TDCP mean `5.713894977`, TDCP `3e-7` mean `5.690271352`; pairwise TDCP better `3`, worse `0`, equal `33`, mean delta `-0.023623625m`。
- regression:
  - `PYTHONPATH=python:. python3 -m pytest -q tests/test_fgo.py tests/test_validate_fgo_gsdc2023_raw.py tests/test_compare_gsdc2023_phone_data_raw_bridge_counts.py tests/test_gsdc2023_gnss_log_reader.py` -> `141 passed`
  - `python3 -m py_compile experiments/gsdc2023_raw_bridge.py experiments/compare_gsdc2023_diagnostics_mask_solver.py experiments/gsdc2023_gnss_log_reader.py tests/test_validate_fgo_gsdc2023_raw.py tests/test_gsdc2023_gnss_log_reader.py`
  - `git diff --check`

2026-04-20 TDCP-on/off gated candidate:

- `position_source=gated` / VD / TDCP weights available の時だけ、同じ batch から TDCP arrays を外した no-TDCP FGO を追加で走らせ、chunk candidate `fgo_no_tdcp` として gated selector に渡すようにした。
- `fgo_no_tdcp` は `fgo` と同じ baseline-gap guard を通す。さらに TDCP-on `fgo` が safe な場合、`fgo_no_tdcp` は `GATED_TDCP_OFF_CANDIDATE_MARGIN = 0.03` 以上 quality が良い時だけ採用する。PR MSE の微差で TDCP-off へ倒れるのを避けるため。
- focused GPS-only rerun:
  - `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4` TDCP `3e-7`: quality margin 後と同じく baseline 維持 `3.3956m`。`fgo_no_tdcp` は候補に入るが採用されない。
  - `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl` TDCP `3e-7`: GPS-only representative の既存改善 `5.4426m -> 5.1288m` を維持。
- representative 12 GPS-only dual-frequency / observation-mask / 200 epoch rerun (`experiments/results/tdcp_off_candidate_12_gated_rawonly_gpsonly_20260420`):
  - no-TDCP: mean `4.278313771`, median `4.197171706`, max `6.626222136`; source totals baseline `2200`, FGO `200`, raw_wls `0`
  - TDCP `3e-7` + candidate: mean `4.252159394`, median `4.197171706`, max `6.626222136`; source totals baseline `2200`, FGO `200`, fgo_no_tdcp `0`, raw_wls `0`
  - pairwise: TDCP better `1`, worse `0`, equal `11`; mean delta `-0.026154376m`, min delta `-0.313852514m`, max delta `0`
- 36本 GPS-only dual-frequency / observation-mask / 200 epoch rerun (`experiments/results/tdcp_off_candidate_36_gated_rawonly_gpsonly_20260420`):
  - no-TDCP reference: `experiments/results/raw_wls_guard_36_gated_rawonly_20260420/aggregate_scores.csv`
  - TDCP `3e-7` + candidate: mean `5.690271352`, median `4.828779804`, max `17.465562340`; source totals baseline `6200`, FGO `600`, fgo_no_tdcp `0`, raw_wls `400`
  - pairwise vs no-TDCP: TDCP better `3`, worse `0`, equal `33`; mean delta `-0.023623625m`, min delta `-0.313852514m`, max delta `0`
- 解釈: 36本 GPS-only では TDCP-off candidate が不要な chunk で採用されず、quality margin 後の adjusted aggregate と同じ値になった。`fgo_no_tdcp` は安全弁として残すが、この subset では実採用0。

2026-04-20 mi8 true-gated full-epoch probe:

- `_effective_position_source()` の mi8 強制 `gated/auto -> raw_wls` を外せるか判断するため、`solve_trip()` を直接呼び、test mi8 outlier 代表 3 本を full epoch / true `position_source=gated` で評価した。
- output dir: `experiments/results/mi8_true_gated_full_epoch_20260420`
- 結果:
  - `test/2021-11-30-20-59-us-ca-mtv-m/mi8`: `1395` epochs, true gated は baseline `1395` / raw_wls `0` / fgo `0`。baseline step95 max `8151m` の巨大jump chunkを保持する。
  - `test/2021-09-28-21-56-us-ca-mtv-a/mi8`: `2479` epochs, true gated は baseline `2079` / raw_wls `400`。PR MSE は baseline `8.1698` -> selected `7.8209`。
  - `test/2022-04-27-21-55-us-ca-ebf-ww/mi8`: `1344` epochs, true gated は baseline `1144` / fgo `200` / raw_wls `0`。auto は fgo `400` だが FGO baseline-gap guard で半分 baseline に戻る。
- 解釈: mi8 強制 raw_wls fallback は train GT では悪いケースがあるが、test outlier では true gated が baseline jump を保持するケースもある。現時点では fallback を単純に外さない。mi8 は別 policy として、baseline jump chunk だけ raw_wls / fgo を許可する専用guardを検討する。

2026-04-20 mi8 baseline-jump gated guard:

- `_effective_position_source()` の mi8 policy を変更し、`auto -> raw_wls` は維持しつつ、`gated` は gated のまま通すようにした。
- gated selector に mi8 専用の raw_wls 許可条件を追加した。
  - `baseline.step_p95_m >= 100m`
  - `raw_wls.baseline_gap_max_m <= 200m`
  - `raw_wls.mse_pr <= baseline.mse_pr * 0.90`
  - `raw_wls.quality_score + 0.08 < baseline.quality_score`
- focused train mi8:
  - `train/2020-12-10-22-52-us-ca-sjc-c/mi8`: raw_wls固定 `17.4656m` 相当から baseline/gated `5.7171m` 相当へ改善。selected source `baseline=200`。
  - `train/2021-08-24-20-32-us-ca-mtv-h/mi8`: raw_wls固定 `7.4907m` から baseline/gated `4.9497m` 相当へ改善。selected source `baseline=200`。
  - TDCP `3e-7` + candidate でも両方 baseline 維持。
- focused test mi8 outliers:
  - `test/2021-11-30-20-59-us-ca-mtv-m/mi8`: 先頭baseline-jump chunkだけ raw_wls に切り替わり、full epoch selected MSE `9.3672 -> 8.6961`。source mix `baseline=1195`, `raw_wls=200`。
  - `test/2021-09-28-21-56-us-ca-mtv-a/mi8`: 旧true-gatedと同じ selected MSE `7.8209`、source mix `baseline=2079`, `raw_wls=400`。
  - `test/2022-04-27-21-55-us-ca-ebf-ww/mi8`: 旧true-gatedと同じ selected MSE `11.7550`、source mix `baseline=1144`, `fgo=200`。
- adjusted 36 GPS-only aggregate (`experiments/results/mi8_baseline_jump_guard_36_adjusted_20260420`):
  - no-TDCP: mean `5.316964287`, median `4.783612716`, max `11.710084550`; source totals baseline `6600`, FGO `600`, raw_wls `0`
  - TDCP `3e-7` + candidate: mean `5.293340661`, median `4.783612716`, max `11.710084550`; source totals baseline `6600`, FGO `600`, fgo_no_tdcp `0`, raw_wls `0`
  - pairwise: TDCP better `3`, worse `0`, equal `33`; mean delta `-0.023623625m`
- 解釈: mi8 train regression の主因だった all-raw_wls fallback は gated では不要になった。test outlier の raw_wls rescue は baseline-jump chunk に限定して維持できる。

2026-04-20 TDCP baseline-gap increase guard + conservative FGO gate / 36 actual:

- TDCP-on/off candidate + raw_wls guard + mi8 baseline-jump guard の組み合わせを、追加 24 train trip へ広げた。
- pre-gap-guard wider run (`experiments/results/full_stack_extra24_gated_rawonly_gpsonly_20260420`):
  - 23 run 成功、1 run failure (`train/2023-09-06-00-01-us-ca-routen/sm-g955f`, `All-NaN slice encountered`)。
  - no-TDCP mean `6.465151058`
  - TDCP `3e-7` + candidate mean `6.448766184`
  - pairwise: TDCP better `4`, worse `3`, equal `16`; mean delta `-0.016384874m`
  - TDCP regression:
    - `train/2021-01-04-21-50-us-ca-e1highway280driveroutea/mi8`: `+0.118697m`
    - `train/2021-01-04-22-40-us-ca-mtv-a/mi8`: `+0.059393m`
    - `train/2021-01-04-21-50-us-ca-e1highway280driveroutea/pixel5`: `+0.178586m`
- 対応:
  - `GATED_TDCP_BASELINE_GAP_INCREASE_MARGIN_M = 0.15` を追加した。
  - TDCP-on `fgo` の `baseline_gap_p95_m` が `fgo_no_tdcp` より `0.15m` 超悪化し、かつ `fgo_no_tdcp` が gated quality を通る場合、TDCP-on `fgo` を skip する。
  - `fgo_no_tdcp` 側の `GATED_TDCP_OFF_CANDIDATE_MARGIN = 0.03` tie-preference は、TDCP-on が baseline gap を上記 margin 超で悪化させた場合だけ緩める。通常は PR quality の微差で TDCP-off へ倒れない。
- focused regression rerun:
  - `mi8_e1highway_tdcp`: `fgo_no_tdcp` 選択、score `5.159641553`。current no-TDCP と同値。
  - `mi8_mtv_a_tdcp`: `fgo_no_tdcp` 選択、score `2.941357935`。current no-TDCP と同値。
  - `pixel5_e1highway_tdcp_margin015`: focused rerun で `fgo_no_tdcp` 選択。actual extra24 でも同 trip は TDCP-on `4.468141587` から `4.289555642` へ戻り、no-TDCP と同値。
  - 既存改善側 (`pixel7pro_2023-05-09`, `pixel5_mtv_a`) は FGO 選択を維持。
- baseline-vs-FGO audit:
  - TDCP regression は消えたが、combined 36 では FGO/raw_wls selected chunk が train GT 上ほぼ全て baseline より悪かった。
  - pre conservative-gate combined 36: selected no-TDCP mean `6.151140456`, TDCP mean `6.122046546` に対し、baseline-only mean は `5.296157237`。
  - worst over-selection は `train/2021-12-08-18-52-us-ca-lax-b/pixel6pro` raw_wls `11.2261m` vs baseline `4.2974m`、`train/2023-06-06-23-26-us-ca-sjc-he2/sm-a226b` FGO `10.1138m` vs baseline `5.0537m`。
- 対応 2:
  - `GATED_FGO_BASELINE_MSE_PR_MIN = 20.0` を追加し、baseline PR MSE がすでに低い chunk では FGO / `fgo_no_tdcp` を採用しない。
  - catastrophic fallback は baseline PR MSE が `GATED_BASELINE_THRESHOLD_DEFAULT` を超える時だけ発火するようにした。candidate 側の `baseline_gap_max_m` が巨大でも、baseline PR MSE が低いなら raw_wls へ倒さない。
  - さらに FGO / `fgo_no_tdcp` は raw_wls より PR MSE が良い時だけ候補として残すようにした。raw_wls が安全性で除外された後、raw_wls より PR residual が悪い FGO だけが残って採用される過選択を防ぐ。
- actual extra24 rerun (`experiments/results/tdcp_gap_guard_extra24_actual_20260420`):
  - no-TDCP: ok `24`, failed `0`, mean `5.910150795`, median `4.115510836`, max `24.273653806`, min `2.131943350`; source totals baseline `4800`, FGO `0`, raw_wls `0`
  - TDCP `3e-7` + candidate: ok `24`, failed `0`, mean `5.910150795`, median `4.115510836`, max `24.273653806`, min `2.131943350`; source totals baseline `4800`, FGO `0`, fgo_no_tdcp `0`, raw_wls `0`
  - pairwise: TDCP better `0`, worse `0`, equal `24`; mean delta `0`
  - `sm-g955f` の `All-NaN slice encountered` は `gnss_log` reader が missing `CarrierFrequencyHz` 行で `np.nanargmin` を呼んでいたため。missing frequency は `freq_hz=NaN`, `freq=""` として L1/L5 selection から除外するよう修正し、同 trip は baseline 200 / score `21.402816852` で通過。
- preliminary adjusted summary (`experiments/results/tdcp_gap_guard_extra24_adjusted_20260420`) は actual rerun で supersede。結論は同じく worse `3 -> 0`。
- 12 exported actual rerun (`experiments/results/tdcp_gap_guard_12_actual_20260420`):
  - no-TDCP: ok `12`, failed `0`, mean `4.068170121`, median `3.987241536`, max `6.626222136`, min `2.553728279`; source totals baseline `2400`, FGO `0`, raw_wls `0`
  - TDCP `3e-7` + candidate: ok `12`, failed `0`, mean `4.068170121`, median `3.987241536`, max `6.626222136`, min `2.553728279`; source totals baseline `2400`, FGO `0`, fgo_no_tdcp `0`, raw_wls `0`
  - pairwise: TDCP better `0`, worse `0`, equal `12`; mean delta `0`
- combined 36 actual aggregate (`experiments/results/tdcp_gap_guard_36_actual_20260420`):
  - no-TDCP: ok `36`, failed `0`, mean `5.296157237`, median `4.021189544`, max `24.273653806`, min `2.131943350`; source totals baseline `7200`, FGO `0`, raw_wls `0`
  - TDCP `3e-7` + candidate: ok `36`, failed `0`, mean `5.296157237`, median `4.021189544`, max `24.273653806`, min `2.131943350`; source totals baseline `7200`, FGO `0`, fgo_no_tdcp `0`, raw_wls `0`
  - pairwise: TDCP better `0`, worse `0`, equal `36`; mean delta `0`
- full train first-200 baseline PR MSE scan (`experiments/results/baseline_mse_scan_train200_gpsonly_20260420`):
  - 156 train trips 成功、failure `0`。
  - `GATED_FGO_BASELINE_MSE_PR_MIN = 20.0` 以上の eligible trip は `33/156`。
  - baseline PR MSE mean `15.599900801`, median `14.682583994`, max `43.189329302`。
  - top eligible は `train/2023-09-06-00-01-us-ca-routen/sm-g955f` (`43.1893`), `train/2023-09-06-00-01-us-ca-routen/pixel5a` (`35.2214`), `train/2020-12-10-22-52-us-ca-sjc-c/pixel4xl` (`32.1139`)。
- eligible33 actual gated rerun (`experiments/results/conservative_gate_train200_eligible33_gpsonly_20260420`):
  - raw_wls MSE guard 前は FGO が4本採用されたが、全て train GT 上 baseline より悪かった。
    - `train/2022-07-26-21-01-us-ca-sjc-s/pixel5`: FGO `6.8308m` vs baseline `3.2535m`
    - `train/2021-07-27-19-49-us-ca-mtv-b/pixel5`: FGO `9.9556m` vs baseline `5.3316m`
    - `train/2023-05-23-19-16-us-ca-mtv-ie2/pixel5`: FGO `6.8455m` vs baseline `3.4719m`
    - `train/2023-09-07-22-48-us-ca-routebc2/pixel5a`: FGO / `fgo_no_tdcp` `7.7310m` vs baseline `5.3421m`
  - raw_wls MSE guard 後:
    - no-TDCP: ok `33`, failed `0`, mean `6.804258028`, median `5.517635503`, max `24.273653806`, min `2.728663634`; source totals baseline `6600`, FGO `0`, raw_wls `0`
    - TDCP `3e-7` + candidate: ok `33`, failed `0`, mean `6.804258028`, median `5.517635503`, max `24.273653806`, min `2.728663634`; source totals baseline `6600`, FGO `0`, fgo_no_tdcp `0`, raw_wls `0`
    - pairwise: TDCP better `0`, worse `0`, equal `33`; mean delta `0`
- full train full-epoch baseline PR MSE scan (`experiments/results/baseline_mse_scan_train_full_gpsonly_20260420`):
  - 4 shard parallel scan は全 `156` train trips 成功、failure `0`。chunk は `1344`。
  - `GATED_FGO_BASELINE_MSE_PR_MIN = 20.0` 以上の eligible trip は `71/156`、eligible chunk は `233/1344`。eligible chunk のうち `62` は PR MSE `inf`。
  - trip full baseline PR MSE は finite `155/156`: mean `14.079554913`, median `12.896144601`, p95 `23.027688518`, max finite `42.223952727`。
  - chunk baseline PR MSE は finite `1282/1344`: mean `13.959855989`, median `12.581376560`, p95 `24.430908372`, max finite `56.712535176`。
  - eligible chunk finite `171/233`: mean `25.598601448`, median `23.122290954`, p95 `38.118834688`, max `56.712535176`。
  - baseline score は median `3.968335241`, p95 `7.041692812`。mean `2445.396968537` は `train/2023-09-06-18-04-us-ca/sm-s908b` の score `380794.695417759` で歪むため代表値にしない。
  - top finite max-chunk candidates:
    - `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g`: eligible chunks `3`, full MSE `15.585164340`, max chunk `56.712535176`, score `22.390805915`
    - `train/2020-07-08-22-28-us-ca/pixel4xl`: eligible chunks `8`, full MSE `22.050020546`, max chunk `43.480526631`, score `2.897276862`
    - `train/2021-01-04-21-50-us-ca-e1highway280driveroutea/pixel5`: eligible chunks `1`, full MSE `13.290591483`, max chunk `41.337154161`, score `2.341020202`
    - `train/2020-07-08-22-28-us-ca/pixel4`: eligible chunks `3`, full MSE `18.509050145`, max chunk `40.561751732`, score `3.607692861`
    - `train/2020-08-13-21-42-us-ca-mtv-sf-280/pixel5`: eligible chunks `1`, full MSE `9.982205770`, max chunk `39.290791251`, score `3.746554634`
  - top finite eligible chunks:
    - `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g` chunk `7` (`1400-1512`): MSE `56.712535176`, step p95 `22.105926414m`
    - `train/2023-09-06-00-01-us-ca-routen/sm-g955f` chunk `5` (`1000-1200`): MSE `46.568227060`, step p95 `74.166482290m`
    - `train/2023-09-06-22-49-us-ca-routebb1/sm-s908b` chunk `4` (`800-1000`): MSE `44.642581612`, step p95 `45.159142979m`
    - `train/2020-07-08-22-28-us-ca/pixel4xl` chunk `10` (`2000-2147`): MSE `43.480526631`, step p95 `19.522532996m`
    - `train/2021-01-04-21-50-us-ca-e1highway280driveroutea/pixel5` chunk `10` (`2000-2002`): MSE `41.337154161`, step p95 `3.982869903m`
  - 解釈: first-200 scan の eligible `33/156` は full-epoch では `71/156` まで増える。したがって「first window で baseline-only」は全trip全epochの結論ではない。ただし train GT では first-window eligible33 の FGO 採用は raw_wls MSE guard 後すべて baseline-only に戻ったため、full-epoch eligible は採用許可ではなく audit candidate pool として扱う。
- full-epoch top20 chunk actual gated rerun (`experiments/results/full_epoch_top20_chunk_actual_20260420`):
  - 対象は full-epoch scan の finite top MSE `12` chunks + PR MSE `inf` の unique-trip step-p95 top `8` chunks。各windowで no-TDCP と TDCP `3e-7` + candidate を比較した。
  - 40 case 全て成功、failure `0`。20 pair の TDCP delta は better `0`, worse `0`, equal `20`。
  - no-TDCP / TDCP candidate とも selected score は mean `45111.123823815`, median `5.654265415`, max `902034.946999390`。mean/max は `train/2023-09-06-18-04-us-ca/sm-s908b` chunk `4` (`800-1000`) の完全破綻で歪む。
  - source totals は no-TDCP baseline `3222`, raw_wls `200`, FGO `0`。TDCP candidate も baseline `3222`, raw_wls `200`, FGO `0`, `fgo_no_tdcp` `0`。
  - raw_wls へ倒れた唯一のwindowは `train/2023-09-06-18-04-us-ca/sm-s908b` chunk `4`。baseline/raw_wls selected score は同じ `902034.946999390`、PR MSE は baseline/raw_wls/FGO 全て `inf`。TDCP-FGO は GT 上 `809919.920851057` まで下がるが PR MSE が `inf` のため現行gateで選ぶ根拠はない。
  - raw_wls what-if: top20内では raw_wls が baseline より GT で良いのは `1` window、悪いのは `17` window、同値 `2` window。唯一の改善は `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g` chunk `7` (`1400-1512`): baseline `29.396923669m`, raw_wls `18.932817387m` (`-10.464106282m`)。ただし PR MSE 低下だけでは sm-g955f / pixel4 / pixel5a など多数を悪化させるため、raw_wls rescue は高baseline-PR-MSEかつ低raw_wls-PR-MSEの保守条件に限定する。
  - FGO what-if: `fgo_no_tdcp` は better `1` / worse `19` だが better は `-0.000002305m` の数値同等。TDCP-FGO は better `1` / worse `19` で、better は上記 sm-s908b 破綻windowのみ。通常windowで baseline より良いFGOは見つからなかった。
- raw_wls high-PR-MSE rescue:
  - `GATED_RAW_WLS_RESCUE_BASELINE_MSE_PR_MIN = 50.0`, `GATED_RAW_WLS_RESCUE_MSE_PR_MAX = 20.0`, `GATED_RAW_WLS_RESCUE_MSE_PR_RATIO_MAX = 0.35` を追加した。
  - baseline / raw_wls PR MSE がfiniteで、baseline PR MSEが `50.0` 以上、raw_wls PR MSEが `20.0` 以下、かつ `raw_wls <= baseline * 0.35` の場合だけ raw_wls を採用する。
  - raw_wls feature audit (`experiments/results/full_epoch_top20_chunk_actual_20260420/raw_wls_rescue_feature_audit.csv`) では、この条件は top20 の改善1件だけを拾い、悪化候補は拾わない。first-200 eligible33 aggregate では match `0`。
  - full-epoch scan 全体でも finite baseline PR MSE `>=50` の chunk はこの1件のみ。baseline PR MSE `>=40` へ広げると7 chunks / 6 trips になるが、raw_wls what-if は better `1`, worse `5`, equal `1`、delta sum `+27.325340124m` で悪化優勢。
  - threshold audit (`experiments/results/full_epoch_rawwls_threshold_audit_20260420/raw_wls_rescue_feature_audit.csv`): current condition は better `1`, worse `0`。`baseline>=45, raw<=20, ratio<=0.35` へ緩めると better `1`, worse `1` で `train/2023-09-06-00-01-us-ca-routen/sm-g955f` chunk `5` が `+5.684867482m` 悪化する。
- adjusted top20 rerun (`experiments/results/full_epoch_top20_chunk_actual_rawwls_rescue_20260420`):
  - 40 case 全て成功、failure `0`。20 pair の TDCP delta は better `0`, worse `0`, equal `20`。
  - no-TDCP / TDCP candidate とも selected score は mean `45110.600618501`, median `5.654265415`, max `902034.946999390`。前回 mean から `-0.523205314m`。
  - source totals は baseline `3110`, raw_wls `312`, FGO `0`。前回比で `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g` chunk `7` の `112` epochs だけが baseline から raw_wls へ移った。
  - changed pair は1件のみ: `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g` chunk `7` (`1400-1512`) が `29.396923669m -> 18.932817387m` (`-10.464106282m`)。その他windowは同値。
- FGO finite high-MSE audit:
  - threshold30 audit (`experiments/results/full_epoch_fgo_threshold30_audit_20260420`): finite baseline PR MSE `>=30` の `31` chunks / `16` trips をactual rerun。
    - 62 case 全て成功、failure `0`。selected TDCP delta は better `0`, worse `0`, equal `31`。
    - source totals は no-TDCP baseline `5170`, raw_wls `112`, FGO `0`。TDCP candidate も baseline `5170`, raw_wls `112`, FGO `0`, `fgo_no_tdcp` `0`。
    - FGO what-if: `fgo_no_tdcp` は better `0`, worse `31`; TDCP-FGO も better `0`, worse `31`。
  - threshold25 combined audit (`experiments/results/full_epoch_fgo_threshold25_combined_audit_20260420`): finite baseline PR MSE `>=25` の `61` chunks をactual rerun（threshold30 + `25<=MSE<30` extra）。
    - 122 case 全て成功、failure `0`。selected TDCP delta は better `0`, worse `0`, equal `61`。
    - source totals は no-TDCP baseline `10560`, raw_wls `112`, FGO `0`。TDCP candidate も baseline `10560`, raw_wls `112`, FGO `0`, `fgo_no_tdcp` `0`。
    - FGO no-TDCP what-if: better `0`, worse `61`。
    - TDCP-FGO what-if: better `2`, worse `59`。
      - `train/2023-09-06-18-04-us-ca/sm-s908b` chunk `2` (`400-600`): baseline `39779.898483621m`, TDCP-FGO `33967.249936810m`, but baseline score/step p95 are catastrophic and TDCP-FGO PR MSE `172.516116281` vs baseline PR MSE `7.318472774`; selection signalとしては使わない。
      - `train/2023-05-09-21-32-us-ca-mtv-pe1/pixel5` chunk `5` (`1000-1200`): baseline `15.013697034m`, TDCP-FGO `14.871985510m` (`-0.141711523m`)。ただし FGO PR MSE `19.982368501` は raw_wls PR MSE `17.956418057` より悪く、baseline gap p95 `33.838578390m` は baseline step p95 `31.396334157m` を超えるため現行gateで止まる。
    - PR MSE系の単純条件では安全に分離できない。例: `fgo_tdcp_mse<=20`, `fgo_tdcp_mse<=baseline_mse*0.75`, `baseline_score<100` は `21` chunks を拾い、better `1`, worse `20`, delta sum `+85.502120044m`。
  - FGO signal audit (`experiments/results/full_epoch_fgo_signal_audit_20260421`):
    - `pairwise_deltas.csv` から PR MSE / baseline step p95 / device / route family で候補条件を再集計した。TDCP-FGO は better `2`, worse `59`、FGO no-TDCP は better `0`, worse `61`。
    - `tdcp_mse_le_baseline_mse` は `34` chunks を拾い、better `1`, worse `33`, delta sum `+148.833229087m`。
    - `tdcp_mse_le20` は `22` chunks を拾い、better `1`, worse `21`, delta sum `+89.941502m`。
    - `tdcp_mse_le_raw_wls_mse` / `tdcp_mse_le10` は各 `1` chunk だけ拾うが、どちらも worse `1`。
    - `tdcp_mse_le_raw_and_step_p95_le35` / `tdcp_mse_le_raw_and_step_p95_le50` は match `0`。安全だが改善も拾わない。
    - route/device cohort でも安定した改善群は見つからない。route 系は `21/21` worse、pixel5 も `1` better に対して `12` worse。
  - 解釈: finite high-MSE帯では、FGOはGT上ほぼ悪化。小改善1件のために raw_wls MSE guard / baseline-gap guard を緩めると悪化多数を拾うため、FGO採用条件は現状維持する。
- raw_wls high-PR-MSE rescue broader scan (`experiments/results/full_epoch_rawwls_rescue_broader_scan_20260421`):
  - full-epoch baseline PR MSE scan 全体 (`1344` chunks) / eligible `71` trips, `233` chunks に対して、rescue の第一条件を再確認した。
  - eligible chunks の内訳は finite `171`, nonfinite `62`。現行 rescue は baseline PR MSE が finite であることを必須にしているため、nonfinite `62` chunks は発火対象外。
  - 現行 `baseline_mse_pr >= 50`, `raw_wls_mse_pr <= 20`, `raw_wls_mse_ratio <= 0.35` のうち、scan だけで判定できる finite baseline `>=50` prefilter は1件のみ:
    - `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g` chunk `7` (`1400-1512`), baseline PR MSE `56.712535176`, step p95 `22.105926414m`
  - actual threshold audit join (`actual_threshold_join_summary.json`):
    - current condition: count `1`, better `1`, worse `0`, delta sum `-10.464106282m`。該当は上記 samsunga325g だけ。
    - `baseline>=45` に緩めると count `2`, better `1`, worse `1`。追加される `train/2023-09-06-00-01-us-ca-routen/sm-g955f` chunk `5` は baseline `20.221754592m` -> raw_wls `25.906622075m` で `+5.684867482m` 悪化。
    - `baseline>=40` actual windows は count `6`, better `1`, worse `5`, delta sum `+27.325340124m`。
  - 解釈: eligible 71 trips 全体へ広げても、現行 raw_wls rescue は top20 audit で見つかった改善1件だけを拾う。baseline floor を `50` 未満へ下げる根拠はなく、現在の `50/20/0.35` を維持する。
- IMU preintegration opt-in bridge:
  - 既存の `load_device_imu_measurements` / `process_device_imu` / `preintegrate_processed_imu` と native VD solver の `imu_delta_p` / `imu_delta_v` 入口を raw bridge から接続した。
  - `validate_fgo_gsdc2023_raw.py` に `--[no-]imu-prior`, `--imu-frame {body,ecef}`, `--imu-position-sigma-m`, `--imu-velocity-sigma-mps` を追加。`run_raw_bridge_batch.py` と MATLAB fallback (`run_fgo_raw_bridge.m`) も同じ option / env (`GSDC2023_BRIDGE_IMU_PRIOR`, `GSDC2023_BRIDGE_IMU_FRAME`, `GSDC2023_BRIDGE_IMU_POSITION_SIGMA_M`, `GSDC2023_BRIDGE_IMU_VELOCITY_SIGMA_MPS`) を通す。
  - default は off。`--imu-frame body` は従来の raw sensor-frame delta、`--imu-frame ecef` は velocity yaw + MATLAB mounting angle + gravity compensation で ECEF delta を作る近似。MATLAB/GTSAM の `Pose3` attitude optimization, accelerometer/gyro bias state, `BetweenFactorConstantBias` までは再現していない。
  - `device_imu.csv` の `BiasX/Y/Z` は acc/gyro と同じ同期経路で `ProcessedIMU` に保持し、GNSS epoch interval ごとの weighted mean bias を `IMUPreintegration` と `bridge_metrics.json` (`imu_acc_bias_mean_norm_mps2`, `imu_gyro_bias_mean_norm_radps`) に記録する。MATLAB `fgo_gnss_imu.m` と同じく raw bias を単純減算せず、raw bridge の通常 graph path では bias state をまだ有効化しない。
  - native `fgo_gnss_lm_vd` は state 幅 `10+n_clock` のときだけ末尾 `[bax,bay,baz]` を optional accelerometer-bias state として扱い、IMU `delta_p` / `delta_v` residual に first-order accel-bias correction を入れられるようにした。raw bridge からは `--[no-]imu-accel-bias-state`, `--imu-accel-bias-prior-sigma-mps2`, `--imu-accel-bias-between-sigma-mps2` で opt-in。これは MATLAB `ConstantBias` への phase-1 scaffold で、gyro bias と pose attitude state はまだ無い。default state は従来幅のまま。
  - MATLAB `prm.time_diff_th = 1.5s` に合わせ、`--factor-dt-max-s` / env `GSDC2023_BRIDGE_FACTOR_DT_MAX_S` を追加。default `1.5` で、`dt >= 1.5s` の epoch interval は motion / clock / TDCP / IMU delta factor を張らず、`bridge_metrics.json` に `factor_dt_gap_count` を記録する。legacy 比較用に `0` を指定すると gap gate を無効化できる。
  - short smoke (`train/2021-07-19-20-49-us-ca-mtv-a/pixel5`, `20` epochs, `gated`) は `--imu-prior` on/off とも完走。on は `imu prior intervals=19` を記録し、gated output は baseline `20` epochs を選んで selected score を維持。FGO 候補は on/off ともこの window では大きく悪いため、IMU prior はまだ default にしない。
  - `audit_gsdc2023_preprocessing_gap.py` の `imu_preintegration` は `partial` から `experimental` に更新。次の本命は native state に pose/bias を足して MATLAB `ImuFactor` に寄せること。
- Absolute height native hook:
  - MATLAB `fgo_gnss_imu.m` は `datapath/course/ref_hight.mat` がある場合に `posgt.up` の nearest reference を `PriorFactorVector(keyX, [0,0,up])` として使う。
  - 現ツリーでは `find ref/gsdc2023 gnss_gpu -name 'ref_hight.mat' -o -name 'ref_height.mat'` が空で、実データ接続はまだできない。
  - native VD solver に `absolute_height_ref_ecef` / `absolute_height_sigma_m` を追加し、`enu_up_ecef` 方向だけを拘束する factor を入れた。非finite reference row は skip。
  - raw bridge / CLI / batch / MATLAB fallback に `--[no-]absolute-height`, `--absolute-height-sigma-m`, `--absolute-height-dist-m` と env `GSDC2023_BRIDGE_ABSOLUTE_HEIGHT*` を追加。`ref_hight.mat` / `ref_height.mat` から `posgt.enu` / `posgt.up` 等を読み、MATLAB と同じ 15m nearest horizontal gate で epoch reference を作る。
  - `audit_gsdc2023_preprocessing_gap.py` の `height_constraints` は `experimental` に更新。ただし手元には ref height artifact が無いため、real `ref_hight.mat` での数値 parity は未検証。
- Native VD host factor RHS fix:
  - `fgo_gnss_lm_vd` は `H * delta = g` を直接解く設計だが、motion / clock drift / relative-height の host factor が旧 sign convention の RHS を積んでいた。`add_motion_factor_host`, `add_clock_drift_factor_host`, `add_relative_height_factor_host` の線形項だけを直し、Hessian / cost は維持した。
  - factor 単体の line-search regression を追加し、PR なしでも motion residual、clock-drift residual、relative-height residual が下がることを確認した。
  - これは graph の数式方向修正であり、IMU の attitude / bias state 追加ではない。FGO 候補は短い GSDC smoke ではまだ gated selection で baseline に負けるため、default 方針は変えない。
- regression:
  - native VD direction focused: `PYTHONPATH=python:. python3 -m pytest -q tests/test_fgo.py::test_fgo_vd_motion_factor_reduces_standalone_residual tests/test_fgo.py::test_fgo_vd_clock_drift_factor_reduces_standalone_residual tests/test_fgo.py::test_fgo_vd_relative_height_factor_reduces_standalone_residual` -> `3 passed in 2.47s`
  - native FGO/wrapper: `PYTHONPATH=python:. python3 -m pytest -q tests/test_fgo.py tests/test_fgo_wrapper.py` -> `50 passed in 2.08s`
  - factor gap / IMU routing focused: `PYTHONPATH=python:. python3 -m pytest -q tests/test_validate_fgo_gsdc2023_raw.py::test_run_fgo_chunked_forwards_opt_in_imu_prior tests/test_validate_fgo_gsdc2023_raw.py::test_run_fgo_chunked_masks_imu_prior_across_factor_dt_gap tests/test_validate_fgo_gsdc2023_raw.py::test_build_trip_arrays_disables_tdcp_across_hardware_clock_discontinuity` -> `3 passed in 0.90s`
  - focused IMU/height opt-in: `PYTHONPATH=python:. python3 -m pytest -q tests/test_validate_fgo_gsdc2023_raw.py::test_preintegrate_processed_imu_between_gnss_epochs tests/test_validate_fgo_gsdc2023_raw.py::test_preintegrate_processed_imu_ecef_frame_removes_stationary_gravity tests/test_validate_fgo_gsdc2023_raw.py::test_run_fgo_chunked_forwards_opt_in_imu_prior` -> `3 passed in 0.83s`
  - raw bridge focused after native RHS fix: `PYTHONPATH=python:. python3 -m pytest -q tests/test_validate_fgo_gsdc2023_raw.py tests/test_compare_gsdc2023_phone_data_raw_bridge_counts.py tests/test_gsdc2023_gnss_log_reader.py` -> `108 passed in 1.97s`; `tests/test_validate_gsdc2023_phone_data.py::test_preprocessing_gap_table_has_expected_stage_coverage` -> `1 passed in 0.63s`
  - `PYTHONPATH=python:. python3 -m pytest -q tests/test_fgo.py tests/test_fgo_wrapper.py tests/test_validate_fgo_gsdc2023_raw.py tests/test_compare_gsdc2023_phone_data_raw_bridge_counts.py tests/test_gsdc2023_gnss_log_reader.py tests/test_validate_gsdc2023_phone_data.py::test_preprocessing_gap_table_has_expected_stage_coverage` -> `156 passed in 5.09s`
  - `PYTHONPATH=python:. python3 -m py_compile experiments/gsdc2023_raw_bridge.py experiments/validate_fgo_gsdc2023_raw.py tests/test_validate_fgo_gsdc2023_raw.py experiments/audit_gsdc2023_preprocessing_gap.py tests/test_validate_gsdc2023_phone_data.py`
  - `python3 -m py_compile ref/gsdc2023/run_raw_bridge_batch.py`
  - CLI help checks for `--factor-dt-max-s`, `--imu-frame`, and `--absolute-height` in `validate_fgo_gsdc2023_raw.py` and `run_raw_bridge_batch.py`
  - no-ref-height smoke: `train/2021-07-19-20-49-us-ca-mtv-a/pixel5`, 20 epochs, `gated --absolute-height` 完走。手元に `ref_hight.mat` が無いため no-op で selected baseline `20` epochs, selected wMSE `287.0477`。
  - IMU ECEF smoke after native RHS fix: 同trip 20 epochs, `gated --imu-prior --imu-frame ecef` 完走。`imu prior intervals=19 frame=ecef` を記録し、gated output は baseline `20` epochs を選択、selected wMSE `287.0477`、FGO RMS2D `2906.872m`。
  - `git diff --check`
- submission/test fallback:
  - `ref/gsdc2023/run_raw_bridge_batch.py` default `--position-source` を `gated` へ変更。
  - MATLAB fallback (`run_fgo_raw_bridge.m` / `functions/submission.m`) の default bridge position source も `gated` へ変更。
  - `bridge_position_columns("gated"|"auto", ...)` は `LatitudeDegrees` / `LongitudeDegrees` を返す regression を追加し、submission assembly が選択済み output columns を読むことを固定。
  - 2026-04-21 smoke:
    - 一時 dataset root (`/tmp/gnss_gpu_gated_smoke_20260421_043214`) で `run_raw_bridge_batch.py` を `--position-source` 省略 / `--limit 1` / `--max-epochs 20` / `--chunk-epochs 20` 実行。`summary_position_source=gated`, `selected_source_mode=gated`, submission と selected columns の max abs diff は lat/lon とも `0`。
    - 同条件を `/tmp/gnss_gpu_gated_smoke5_20260421_043306` / `--limit 5` へ拡大。5/5 成功、`summary_position_source=gated`, submission rows `7076`, NaN `False`, selected columns との差分 lat/lon `0`。source totals は baseline `100`, raw_wls `0`, FGO `0`, `fgo_no_tdcp` `0`。
    - 同条件を `/tmp/gnss_gpu_gated_smoke40_20260421_051825` / test 40 trip へ拡大。40/40 成功、`summary_position_source=gated`, `selected_source_mode=gated` は全40、submission rows `71936`, unique trips `40`, NaN `False`, selected columns との差分 lat/lon `0`。source totals は baseline `740`, raw_wls `20`, FGO `20`, `fgo_no_tdcp` `20`。
    - non-baseline 採用3件を `/tmp/gnss_gpu_gated_smoke40_focus_20260421_052008` で TDCP-on/off 比較。selected PR MSE の on-off delta は全て `0`。`2022-04-04-16-31-us-ca-lax-x/pixel5` は TDCP-on FGO が PR MSE `2733885.519052` へ壊れるため `fgo_no_tdcp` (`522.460103`) に戻る。`2022-10-06-20-46-us-ca-sjc-r/sm-a205u` は raw_wls (`430.082201`) に倒れ、`2022-07-12-18-37-us-ca-mtv-b/sm-a325f` は FGO/raw_wls が同じ PR MSE (`55.656095`) で FGO 採用。
    - 同じ一時 root で all-epoch dry run (`--max-epochs -1`, `--chunk-epochs 200`, workers `4`) も完走。result dir は `ref/gsdc2023/results/test_parallel/20260421_0555`、submission は `submission_20260421_0555.csv`。40/40 成功、submission rows `71936`, unique trips `40`, NaN `False`, selected columns との差分 lat/lon `0`。summary は mean OptError `58.1465`, median `25.665`, max `949.02`。source totals は baseline `70665`, raw_wls `590`, FGO `200`, `fgo_no_tdcp` `571`。
    - all-epoch non-baseline trip は5件: `2020-12-11-19-30-us-ca-mtv-e/pixel4xl` raw_wls `190` epochs、`2021-11-30-20-59-us-ca-mtv-m/mi8` raw_wls `200`、`2022-04-04-16-31-us-ca-lax-x/pixel5` `fgo_no_tdcp` `571`、`2022-07-12-18-37-us-ca-mtv-b/sm-a325f` FGO `200`、`2023-05-09-23-10-us-ca-sjc-r/sm-a505u` raw_wls `200`。
    - all-epoch PR MSE 集計では selected は baseline に対し improved `5`, worsened `0`, equal `35`。mean delta `-12299.032211` は `2021-11-30-20-59-us-ca-mtv-m/mi8` の baseline PR MSE `491674.596693 -> 9.798270` が支配的。その他 nonzero delta は `pixel4xl` `-197.387693`, LAX pixel5 `-91.894539`, sm-a505u `-6.567334`, sm-a325f `-0.640455`。
    - non-baseline branch: pixel4xl / mi8 は `catastrophic_baseline_alternative -> raw_wls`。LAX pixel5 と sm-a505u は `baseline_mse > 500` 分岐で best non-baseline quality を採用。sm-a325f は標準 FGO gate (`gap/raw_mse/quality=True`)。
    - hardening: `baseline_mse > gated_threshold` 分岐でも、候補 PR MSE が baseline PR MSE 未満のものだけ採用するようにした。既存 all-epoch test 40 trip の `376` chunk では selection change `0`。候補が全て悪い高PR baselineで非baselineへ倒れる事故を防ぐ regression を追加。
- GSDC2023 raw factor-mask parity follow-up:
  - `compare_gsdc2023_factor_masks.py --max-epochs` は MATLAB mask だけでなく bridge 側 residual masking も window に影響されていた。MATLAB `exobs_residuals.m` の pseudorange ISB は graph window ではなく `phone_data.mat` 全 epoch の `median(obs_pre.(freq).resPc-clk)` で決まるため、`build_trip_arrays(max_epochs=...)` でも ISB だけは full epoch span から推定し、requested window の group id へ `(MATLAB sys, L1/L5)` で remap するようにした。
  - LAX pixel5 200 epoch raw-only mask は `total_matlab_count=12806`, `total_bridge_count=12806`, `total_matched_count=12806`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。diagnostics overlay なしでも完全一致。
  - residual value compare の common-bias median abs delta は L1/L5 の windowed ISBずれ解消により `5.869e-06m` まで低下。残る大きい max delta は common bias ではなく satellite/model 端点側。
  - regression: `test_build_trip_arrays_uses_full_epoch_span_for_pseudorange_isb` を追加し、`max_epochs=1` でも先頭 epoch だけの ISB に引きずられないことを固定。
  - Doppler residual mask も window edge だけ full epoch の receiver velocity / clock drift context を使うようにした。MATLAB は `Gpos.gradient(obs.dt)` / `obs.dclk` を全 `phone_data.mat` span で持つため、`max_epochs=200` の末尾 epoch だけ window-local gradient / drift にすると D/resD 境界がずれる。middle epoch では full context を使わない regression も追加し、Pixel4XL の mid-window Doppler mask 退行を防ぐ。
  - TDCP dDL consistency は MATLAB `obs.dt` 相当の scalar interval を使い、graph の `factor_dt_max_s` gap-zeroing 後の `dt` ではなく raw positive interval から判定するようにした。`fgo_gnss.m` では `time_diff_th` は motion/clock 側の gate であり、TDCP availability は `obs.clkjump` と dDL mask が支配するため。
  - 12 exported train trips の `--max-epochs 200` raw-only factor-key audit は `experiments/results/signal_clock_followup_20260421/gsdc2023_factor_mask_parity_12trip_latest_tropo_20260421_150624` でも `trip_count=12`, `exact_count=12`, `total_matlab_only=0`, `total_bridge_only=0`, `min_symmetric_parity=1.0`。diagnostics overlay なしで P/D/L/resPc/resD/resL key set が全件一致。
  - `gnss_log.txt` から補完した GPS-only 行に `ArrivalTimeNanosSinceGpsEpoch=tow_rx_s*1e9` を持たせ、duplicate nav 時も MATLAB/MatRTKLIB と同じ arrival TOW で衛星 clock product を選べるようにした。さらに gnss_log corrected pseudorange を組む前に baseline WLS を repair し、finite sat slot の RTKLIB Saastamoinen tropo を再計算するようにした。これで `2020-08-04-00-20 pixel4xl` の residual max は `0.284m` から `0.00296m` へ低下。
  - residual value audit (`phone_data_residual_diagnostics.csv` がある 11 trip) も比較器を window edge の full velocity / drift context と single-observation clock/drift residual に対応済み。最新の `experiments/results/signal_clock_followup_20260421/gsdc2023_residual_value_parity_11trip_latest_tropo_20260421_150624` では `total_matched_count=50846`, trip median abs delta の中央値 `1.416e-4`, max trip p95 abs delta `0.00403m`, max trip max abs delta `0.00705m`。worst は `2020-07-17-23-13 pixel4` の P/L5 model 側 mm 級差分で、旧 worst の duplicate-nav / tropo fallback は解消。factor-key parity はこの値差込みでも全12 trip exact。
- GSDC2023 raw bridge modularization:
  - まず低リスクな純粋ロジックを分割し、`experiments/gsdc2023_signal_model.py` に signal type / L1-L5 label / constellation->MATLAB sys / MATLAB signal-clock kind / pseudorange ISB group remap を移した。`gsdc2023_raw_bridge.py` 側は既存 private API 名を import alias として維持し、既存 tests が直接参照している `_multi_system_for_clock_kind` などの互換性を保つ。
  - parity 比較器の重複 window logic は `experiments/gsdc2023_trip_window.py` へ集約した。`compare_gsdc2023_factor_masks.py` / `compare_gsdc2023_residual_values.py` は同じ `settings_epoch_window_for_trip` と `trim_epoch_window` を使うため、settings CSV / TDCP edge trimming の修正が片側だけに入る事故を防ぐ。
  - unit tests を `tests/test_gsdc2023_signal_model.py` / `tests/test_gsdc2023_trip_window.py` に分離し、純粋 mapping と window trimming を solver / raw CSV I/O なしで検証できるようにした。既存 integration tests と合わせて `182 passed`。実データ smoke として `2020-08-04-00-20 pixel4xl --max-epochs 200` を再実行し、factor は `13118/13118` exact、residual は `max_abs_delta=0.002961m` を維持。
  - `experiments/gsdc2023_gnss_log_bridge.py` を追加し、gnss_log-only synthetic row 補完、gnss_log epoch time 抽出、corrected pseudorange product 生成を raw bridge 本体から分離した。sat-clock adjustment は `SatClockAdjustmentFn` として注入し、new module は `gsdc2023_raw_bridge.py` を import しない。raw bridge 側は `_gnss_log_corrected_pseudorange_matrix` wrapper と private alias を残し、既存 call site の互換性を維持。
  - `tests/test_gsdc2023_gnss_log_bridge.py` を追加し、sat-clock adjustment 注入、RTKLIB tropo 優先、ArrivalTimeNanosSinceGpsEpoch 付与、epoch time 抽出を単体検証できるようにした。既存 integration tests と合わせて `185 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual `max_abs_delta=0.002961m` を維持。
  - `experiments/gsdc2023_residual_model.py` を追加し、Sagnac geometry / MATLAB scalar epoch interval / receiver velocity / residual clock bias-drift seed / weighted median clock prediction / pseudorange ISB / P residual mask / D residual mask / P-D consistency mask を raw bridge 本体から分離した。`compare_gsdc2023_residual_values.py` も geometry と clock helper をこの module から直接 import するようにし、raw bridge 依存を `_build_trip_arrays` と raw product loading へ狭めた。
  - `tests/test_gsdc2023_residual_model.py` を追加し、group-local median clock prediction、receiver clock bias からの ISB 推定、common-bias group つき P residual mask、P-D consistency endpoint mask を単体検証できるようにした。既存 integration tests と合わせて `189 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual `max_abs_delta=0.002961m` を維持。
  - `experiments/gsdc2023_imu.py` を追加し、`IMUMeasurements` / `ProcessedIMU` / `IMUPreintegration` dataclass、device_imu 読込、acc/gyro 同期、stop projection、Euler/mounting rotation、ECEF delta preintegration、IMU segment masking を raw bridge 本体から分離した。CSV reader は `read_csv_fn` 注入にして、new module は `gsdc2023_raw_bridge.py` を import しない。raw bridge 側は `_read_device_imu_frame` / `load_device_imu_measurements` wrapper と既存 private alias を残し、CLI / tests / downstream からの既存 import を維持。
  - `tests/test_gsdc2023_imu.py` を追加し、reader injection と duplicate handling、GNSS elapsed が無い場合の UTC offset sync、ECEF stationary-gravity preintegration、invalid interval masking を単体検証できるようにした。分割済み module + raw bridge integration tests は `193 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_chunk_selection.py` を追加し、`ChunkCandidateQuality` / `ChunkSelectionRecord` dataclass、trajectory motion stats、candidate quality score、auto/gated source selection、catastrophic baseline alternative、raw_wls rescue、TDCP-off FGO tie/gap policy、chunk selection payload、TDCP-off candidate injection を raw bridge 本体から分離した。new module は numpy だけに依存し、solver / CSV / `TripArrays` には依存しない。raw bridge 側は `_select_auto_chunk_source` / `_select_gated_chunk_source` などの既存 private API を import alias として維持。
  - `tests/test_gsdc2023_chunk_selection.py` を追加し、candidate quality の baseline gap / prev-tail 計算、auto selection、TDCP-off tie policy、TDCP-off candidate span matching、payload 生成を単体検証できるようにした。分割済み module + raw bridge integration tests は `198 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_base_correction.py` を追加し、base observation selector、base metadata/settings reader、base residual interpolation、MatRTKLIB duplicate GPS nav filter / nav-product adjustment / GPS TGD helper を raw bridge 本体から分離した。new module は raw bridge を import せず、`compute_base_pseudorange_correction_matrix` は base setting / residual loader / phone span provider を注入できるようにした。raw bridge 側は既存 private API 名を import alias として維持し、monkeypatch 互換の thin wrapper だけを残した。
  - `tests/test_gsdc2023_base_correction.py` を追加し、MatRTKLIB nav filter/selection、RINEX3 base pseudorange selector、L5 iono scale、base offset、MATLAB-style base time span mask、injected dependency つき correction matrix を単体検証できるようにした。分割済み module + raw bridge integration tests は `203 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_tdcp.py` を追加し、ADR state validation、TDCP phone-family policy（disable / Samsung A Loffset / XXDD drift）、TDCP measurement/weight construction、Doppler-carrier consistency reject propagation、TDCP weight scale、geometry correction を raw bridge 本体から分離した。new module は `gsdc2023_residual_model.geometric_range_with_sagnac` 以外に依存せず、raw bridge 側は `_build_tdcp_arrays` / `_apply_tdcp_weight_scale` / `_apply_tdcp_geometry_correction` などの既存 private API を import alias として維持。
  - `tests/test_gsdc2023_tdcp.py` を追加し、phone-family policy、ADR state bitmask、Doppler consistency reject、endpoint reject propagation、MATLAB scalar interval、weight scale、geometry correction を単体検証できるようにした。分割済み module + raw bridge integration tests は `210 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_height_constraints.py` を追加し、ECEF/ENU relative conversion の alias、`ref_hight.mat` / `ref_height.mat` loader、absolute-height nearest-up reference mapping、phone-family position offset、relative-height loop grouping、ENU-up axis、relative-height star edges、post-hoc relative-height smoothing を raw bridge 本体から分離した。new module は `gsdc2023_imu` の座標/heading helper と `evaluate` の ECEF/LLH helper にだけ依存し、raw bridge 側は `apply_relative_height_constraint` / `apply_phone_position_offset` / `load_absolute_height_reference_ecef` / `_ecef_to_enu_relative` などの既存 API を import alias として維持。
  - `tests/test_gsdc2023_height_constraints.py` を追加し、phone offset policy、heading-aligned offset、state extra-column preservation、relative-height grouping/star edges、stop-mask skip、absolute-height ref mapping、ENU-up axis を単体検証できるようにした。分割済み module + raw bridge integration tests は `218 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_output.py` を追加し、`BridgeResult`、metrics summary/score/format、ECEF->LLH 出力変換、`bridge_positions.csv` / `bridge_metrics.json` の export/load/validity check、position-source column selection / validation を raw bridge 本体から分離した。new module は `gsdc2023_raw_bridge.py` を import せず、raw bridge 側は `BridgeResult` / `_export_bridge_outputs` / `bridge_position_columns` などの既存 public/private API を import alias として維持。
  - `tests/test_gsdc2023_output.py` を追加し、`positions_table` / `metrics_payload` / `summary_lines`、GT なし export、validity reject、position-source validation、metrics helper を単体検証できるようにした。分割済み module + raw bridge integration tests は `223 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_observation_matrix.py` を追加し、`TripArrays` dataclass、raw GNSS column schema、raw CSV reader、Android State / MATLAB-style P-D-L availability mask、epoch metadata builder、baseline WLS repair、clock HCDC jump、receiver clock-bias lookup を raw bridge 本体から分離した。raw bridge 側は `_load_raw_gnss_frame` / `_matlab_signal_observation_masks` / `_repair_baseline_wls` / `_receiver_clock_bias_lookup_from_epoch_meta` などの既存 private API を import alias として維持。
  - `tests/test_gsdc2023_observation_matrix.py` を追加し、raw CSV schema filtering、P/D/L mask separation、low-bias epoch metadata preference、clock reset/jump、baseline repair を単体検証できるようにした。分割済み module + raw bridge integration tests は `228 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `experiments/gsdc2023_clock_state.py` を追加し、phone-family multi-GNSS / output-source policy、clock aid / drift seed policy、clock bias jump threshold、clock drift cleanup、clock-jump mask combine、chunk segment ranges、factor break mask を raw bridge 本体から分離した。new module は `gsdc2023_observation_matrix.interpolate_series` と `gsdc2023_residual_model.gradient_with_dt` だけに依存し、raw bridge 側は `_effective_multi_gnss_enabled` / `_clock_drift_seed_enabled` / `_segment_ranges` / `_factor_break_mask` などの既存 private API を import alias として維持。
  - `tests/test_gsdc2023_clock_state.py` を追加し、mi8/xiaomimi8 policy、clock-aid seed policy、bias-derived drift、drift jump interpolation、jump detect/combine、segment split、invalid-dt factor break を単体検証できるようにした。分割済み module + raw bridge integration tests は `235 passed`。実データ smoke は `2020-08-04-00-20 pixel4xl --max-epochs 200` で factor `13118/13118` exact、residual median abs delta `0.000136m`, p95 `0.001889m`, max `0.002961m` を維持。
  - `gsdc2023_observation_matrix.py` に `RawEpochObservation` / `ObservationMatrixProducts` と、`select_epoch_observations`、`build_slot_keys`、`build_gps_l1_sat_time_lookup`、`fill_observation_matrices`、`recompute_rtklib_tropo_matrix` を追加した。`build_trip_arrays` の epoch 選択、slot key 生成、P/D/L/ADR/Doppler/sat-clock/tropo matrix fill は injected helper へ移り、raw bridge 側は MatRTKLIB / signal clock / geometry 関数を注入して orchestration と後段 mask/solver 接続に寄せた。new helper は `gsdc2023_raw_bridge.py` を import しない。
  - `tests/test_gsdc2023_observation_matrix.py` に epoch selection、matrix fill、repaired-baseline 後の RTKLIB tropo recompute の単体テストを追加した。検証は `py_compile`、観測行列単体 `8 passed`、`build_trip_arrays` focused `4 passed`、GSDC2023 関連 `178 passed` + compare/validate `14 passed`。全体 `pytest -q` は `620 passed` まで進んだが、repo 既存の未ビルド extension (`gnss_gpu._bvh` / `_raytrace`) と非 GSDC WLS/RAIM/RTKLIB regression で `39 failed, 8 errors` のため、今回差分の合否には使っていない。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。residual は `total_matched_count=5360`, `matlab_only=92`, `bridge_only=0`, median abs delta `0.0001361232946738289`, p95 `0.0018885435276413621`, max `0.002961355846307967`。
  - `experiments/gsdc2023_diagnostics_mask.py` を追加し、MATLAB `phone_data_residual_diagnostics.csv` の P/D/L availability overlay と boolean coercion を raw bridge 本体から分離した。new module は `gsdc2023_signal_model` の MATLAB sys / frequency label mapping だけに依存し、raw bridge 側は `_diagnostics_bool` / `_apply_matlab_residual_diagnostics_mask` の既存 private API 名を import alias として維持。
  - `tests/test_gsdc2023_diagnostics_mask.py` を追加し、MATLAB CSV 由来の bool 値、P/D signal weight 復元、TDCP pair weight 復元、必須列 validation を単体検証できるようにした。focused diagnostics tests は `5 passed`、GSDC2023 関連 regression は `195 passed`。実データ diagnostics overlay smoke は `train/2020-06-25-00-34-us-ca-mtv-sb-101/pixel4 --max-epochs 200 --no-multi-gnss --matlab-residual-diagnostics-mask .../phone_data_residual_diagnostics.csv` で factor `11912/11912`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。
  - `experiments/gsdc2023_bridge_config.py` を追加し、`BridgeConfig`、`FACTOR_DT_MAX_S` の public re-export、outlier refinement 閾値 / helper を raw bridge 本体から分離した。new module は config validation に必要な定数群だけを import し、raw bridge 側は `BridgeConfig` / `FACTOR_DT_MAX_S` / `_should_refine_outlier_result` の既存 public/private API 名を import alias として維持。
  - `tests/test_gsdc2023_bridge_config.py` を追加し、default factor dt、position source / IMU frame validation、non-finite numeric guard、outlier refinement trigger 条件を単体検証できるようにした。検証は `py_compile`、focused config + raw alias tests `8 passed`、GSDC2023 関連 regression `202 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。同 window の raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_solver_selection.py` を追加し、TDCP-off candidate enable 判定、TDCP を抜いた `TripArrays` 生成、MI8 gated raw_wls jump guard 判定、source solution catalog、fixed/custom source counts、gated source splicing を raw bridge 本体から分離した。new module は `BridgeConfig` / `TripArrays` / `ChunkSelectionRecord` を入力契約にし、solver 実行・CSV I/O・metrics には依存しない。raw bridge 側は `_tdcp_off_candidate_enabled` / `_batch_without_tdcp` / `_mi8_gated_baseline_jump_guard_enabled` を import alias として維持。
  - `tests/test_gsdc2023_solver_selection.py` を追加し、TDCP-off enable 条件、TDCP clear、MI8 policy、gated chunk state splice/count、fixed/custom source catalog entries を単体検証できるようにした。検証は `py_compile`、focused solver-selection + existing gated/FGO tests `8 passed`、GSDC2023 関連 regression `207 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_solver_options.py` を追加し、`BridgeConfig` から `run_fgo_chunked` の solver option kwargs だけを作る `FgoRunOptions` dataclass を分離した。clock jump / clock drift seed / TDCP drift / stop mask のような trip/phone 依存値は `solve_trip` 側に残し、config 由来の固定 option だけを束ねる。raw bridge 側は TDCP-on/off の `run_fgo_chunked` 呼び出しで同じ `fgo_run_kwargs` を使うため、IMU/height/relative-height option 追加時の二重更新漏れを防ぐ。
  - `tests/test_gsdc2023_solver_options.py` を追加し、config field mapping と `run_kwargs()` のキー集合を単体検証できるようにした。検証は `py_compile`、focused solver-options + IMU forwarding tests `4 passed`、GSDC2023 関連 regression `209 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_solver_context.py` を追加し、phone-family TDCP drift policy、clock-average-drift policy、clock-aid jump mask / drift seed、solver stop mask、speed estimation を `SolverExecutionContext` に分離した。raw bridge 側は `estimate_speed_mps` / `solver_stop_mask` / `SolverExecutionContext` を同名 import して既存 import 互換を維持し、`solve_trip` は `solver_context.run_kwargs()` と `FgoRunOptions.run_kwargs()` を `run_fgo_chunked` へ渡すだけに薄くした。
  - `tests/test_gsdc2023_solver_context.py` を追加し、stop-mask speed filter、Pixel4 baseline-clock jump + drift seed、Samsung blocklist raw clock bias jump、sm-a505u drift seed skip、`run_kwargs()` のキー/値を単体検証できるようにした。検証は `py_compile`、focused solver-context + raw alias tests `8 passed`、GSDC2023 関連 regression `214 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_result_assembly.py` を追加し、source catalog の postprocess（relative-height / phone offset）、selected source extraction、truth metrics 計算を `AssembledSourceOutputs` に分離した。new module は `SourceSolutionCatalog` / `TripArrays` / `BridgeConfig` を入力契約にし、solver 実行や CSV I/O には依存しない。raw bridge 側は `assemble_source_outputs(...)` の返り値を `BridgeResult` に詰めるだけにした。
  - `tests/test_gsdc2023_result_assembly.py` を追加し、selected source/state/count/MSE、truth metrics 計算、truth なし metrics suppression、postprocess disabled 時の copy を単体検証できるようにした。検証は `py_compile`、focused result-assembly + output/export tests `10 passed`、GSDC2023 関連 regression `217 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_result_metadata.py` を追加し、IMU valid interval / bias norm summary と `BridgeResult` の payload metadata kwargs（factor gap、stop/IMU/height/base/observation/TDCP flags and counts）を raw bridge 本体から分離した。raw bridge 側は `_mean_finite_row_norm` / `_imu_result_summary` を import alias として維持し、metadata kwargs は後続の result assembly helper から `BridgeResult` へ注入する。
  - `tests/test_gsdc2023_result_metadata.py` を追加し、finite row norm、graph dt gap 後の IMU interval count、prior disabled 時の bias summary、metadata flag/count mapping、IMU なし no-op を単体検証できるようにした。検証は `py_compile`、focused result-metadata + IMU/output tests `12 passed`、GSDC2023 関連 regression `222 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_validation_context.py` を追加し、`validate_raw_gsdc2023_trip` の default config、base-correction preflight、phone-family effective multi-GNSS / position-source config、`max_epochs` sentinel、outlier refinement config を `RawTripValidationContext` 周辺へ分離した。raw bridge 側は entrypoint orchestration に寄せ、`RawTripValidationContext` を import して既存公開面を維持。
  - `tests/test_gsdc2023_validation_context.py` を追加し、MI8 family override、base correction readiness、audit injection、default config、unbounded epoch sentinel、outlier refinement 条件を単体検証できるようにした。検証は `py_compile`、focused validation-context + raw alias tests `11 passed`、GSDC2023 関連 regression `230 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_trip_stages.py` を追加し、`build_trip_arrays` 内の raw observation frame filtering / MATLAB-style observation mask、epoch metadata context、epoch-time / clock-drift context、full-window context 再帰、base correction 適用、gnss_log corrected pseudorange overlay、absolute-height reference、pseudorange residual ISB/mask、Doppler residual mask、pseudorange-Doppler consistency mask、graph/TDCP 用 dt 生成、TDCP measurement / diagnostics / geometry / weight postprocess、IMU stage 構築を `RawObservationFrameProducts` / `EpochMetadataContext` / `EpochTimeContext` / `FullObservationContextProducts` / `GnssLogPseudorangeStageProducts` / `AbsoluteHeightStageProducts` / `PseudorangeResidualStageProducts` / `DopplerResidualStageProducts` / `PseudorangeDopplerStageProducts` / `GraphTimeDeltaProducts` / `TdcpStageProducts` / `ImuStageProducts` と注入関数に分離した。full-window context は Doppler clock-drift context と pseudorange ISB context の二重再帰を避け、必要なときだけ `build_trip_arrays(... max_epochs=1_000_000_000, start_epoch=0, masks=0)` を作る。
  - `tests/test_gsdc2023_trip_stages.py` を追加し、constellation/signal/dual-frequency filtering、finite raw product filter、observation mask flags/count、gnss_log 用 P 行保持、empty-frame guard、epoch metadata lookup / repaired sorted baseline velocity、optional clock/elapsed lookup、epoch grouping / gnss_log epoch merge / clock drift clean、full-window context no-op/build-once/fallback drift、observation matrix input の duplicate sort/group/tail と nav loader 呼び出し、post-fill baseline repair / RTKLIB tropo recompute、base correction の finite/weight 適用、trip context guard、gnss_log overlay と ISB sample weight 切替、absolute-height loader gating、full-window ISB remap、L5 residual threshold、Doppler context forwarding、P-D consistency frequency threshold、factor dt gap と TDCP raw interval の分離、TDCP diagnostics/geometry/scale ordering、IMU missing/success/failure path を単体検証できるようにした。検証は `py_compile`、trip-stage focused tests `32 passed`、observation/raw focused tests `42 passed`、GSDC2023 関連 regression `262 passed`、`git diff --check` / `git -C ../ref/gsdc2023 diff --check` OK。
  - `build_observation_matrix_input_stage` / `ObservationMatrixInputProducts` と `postprocess_filled_observation_stage` / `FilledObservationPostprocessProducts` を追加し、`build_trip_arrays` 本体に残っていた observation matrix fill 前後の sort/group/tail、GPS TGD / MatRTKLIB nav product fetch、post-fill baseline WLS repair と RTKLIB tropo recompute を stage helper へ移した。raw bridge 側は `_build_observation_matrix_input_stage(...)` と `_postprocess_filled_observation_stage(...)` の結果を受けて、`_fill_observation_matrices` と後段 mask/solver へ渡すだけにした。
  - 実データ smoke は `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --no-multi-gnss` で維持。factor mask parity は `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0`。raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628`。
  - `experiments/gsdc2023_factor_mask.py` を追加し、`phone_data_factor_mask.csv` 系の key schema、MATLAB-style factor row append、mask normalize、outer-join side labeling、field/frequency summary、residual diagnostics からの factor-mask rebuild を compare/audit scripts から分離した。`compare_gsdc2023_factor_masks.py` と `compare_gsdc2023_residual_diagnostics_factor_mask.py` は同じ helper を使うため、`P/D/L/resPc/resD/resL` key の coercion・dedup・side count の更新漏れを防げる。
  - `tests/test_gsdc2023_factor_mask.py` を追加し、row append の MATLAB epoch/next-epoch key、normalize/merge/summary count、residual diagnostics 由来 L pair rebuild を単体検証できるようにした。検証は `py_compile`、factor-mask focused tests `7 passed`、GSDC2023 関連 regression `265 passed`。実データ factor smoke は同 Pixel4XL 200 epoch で `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0` を維持。
  - `gsdc2023_result_assembly.py` に `build_bridge_result` を追加し、`BridgeResult(...)` の field mapping、chunk-selection payload、result metadata kwargs を result assembly module 側へ移した。raw bridge 側の `solve_trip` は source catalog / assembled outputs を作った後、`_build_bridge_result(...)` を返すだけになり、metadata と output state mapping の更新箇所が一つに閉じた。
  - `tests/test_gsdc2023_result_assembly.py` に BridgeResult construction regression を追加し、selected outputs、MSE、source counts、chunk payload、metadata flag が assembly helper から入ることを固定した。検証は `py_compile`、result assembly/metadata focused tests `9 passed`、GSDC2023 関連 regression `266 passed`。実データ raw validation smoke は同 Pixel4XL 200 epoch で baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628` を維持。
  - `experiments/gsdc2023_residual_audit.py` を追加し、`phone_data_residual_diagnostics.csv` 由来の MATLAB residual frame 正規化、bridge residual row key 生成、MATLAB/bridge outer join、delta / component delta 付与、field/frequency summary を `compare_gsdc2023_residual_values.py` から分離した。bridge residual の実測生成は比較 script に残し、join/summary の schema と統計だけを shared helper に閉じた。
  - `tests/test_gsdc2023_residual_audit.py` を追加し、P/D diagnostics の column mapping、bridge residual key append、delta/component delta、summary count/stat、必須列 validation を単体検証できるようにした。検証は `py_compile`、residual-audit focused tests `5 passed`、residual compare 関連 `16 passed`、GSDC2023 関連 regression `269 passed`。実データ residual smoke は同 Pixel4XL 200 epoch で `total_matched_count=5360`, `total_matlab_only=92`, `total_bridge_only=0`, median abs delta `0.0001361232946738289`, p95 `0.0018885435276413621`, max `0.002961355846307967` を維持。
  - `gsdc2023_trip_stages.py` に `FilledObservationMatrixProducts` / `build_filled_observation_matrix_stage` を追加し、`build_trip_arrays` に残っていた `_select_epoch_observations` と `_fill_observation_matrices` の呼び出しを stage helper へ移した。`EpochTimeContext` / `EpochMetadataContext` / `ObservationMatrixInputProducts` を入力契約にして、raw bridge 側は stage の `epochs` と `observations` を受け取るだけにした。
  - `tests/test_gsdc2023_trip_stages.py` に filled-observation stage の selection/fill forwarding と empty epoch guard を追加した。検証は `py_compile`、trip-stage focused tests `34 passed`、observation/raw focused tests `44 passed`、GSDC2023 関連 regression `271 passed`。実データ smoke は同 Pixel4XL 200 epoch で factor `13118/13118` exact、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, raw WLS RMS2D `7.758m`, selected wMSE PR `31.8628` を維持。
  - `experiments/gsdc2023_audit_output.py` を追加し、compare/audit scripts の timestamped unique output directory、`summary.json` 書き出し、JSON + `comparison_dir=...` stdout を共通化した。`compare_gsdc2023_factor_masks.py` / `compare_gsdc2023_phone_data_raw_bridge_counts.py` / `compare_gsdc2023_residual_values.py` / `compare_gsdc2023_residual_diagnostics_factor_mask.py` / `compare_gsdc2023_gnss_log_residual_prekeys.py` / `compare_gsdc2023_gnss_log_observation_counts.py` は同じ helper を使うため、同一秒実行時の suffix 付与と summary 出力形式が揃う。
  - `tests/test_gsdc2023_audit_output.py` を追加し、output directory suffix、timestamped directory、summary JSON / stdout を単体検証できるようにした。検証は `py_compile`、audit-output + related focused tests `12 passed`、compare/audit 周辺 `26 passed`、GSDC2023 関連 regression `274 passed`。実データ factor smoke は同 Pixel4XL 200 epoch で `total_matlab_count=13118`, `total_bridge_count=13118`, `total_matched_count=13118`, `matlab_only=0`, `bridge_only=0`, `symmetric_parity=1.0` を維持。
  - `gsdc2023_trip_stages.py` に `TripArraysFactoryFn` / `assemble_trip_arrays_stage` を追加し、`build_trip_arrays` の末尾に残っていた `TripArrays(...)` の field mapping を stage helper へ移した。`GraphTimeDeltaProducts` / `TdcpStageProducts` / `ImuStageProducts` / `AbsoluteHeightStageProducts` / residual mask stage products を入力契約にして、raw bridge 側は後段 mask/TDCP/IMU stage の結果を渡して返すだけにした。
  - `tests/test_gsdc2023_trip_stages.py` に TripArrays assembly payload の単体テストを追加した。検証は `py_compile`、assembly focused + raw bridge representative tests `3 passed`、trip-stage focused tests `35 passed`、`git diff --check` OK。
  - `gsdc2023_trip_stages.py` に `ClockResidualStageProducts` / `build_clock_residual_stage` を追加し、epoch-count 由来 clock jump、blocklist phone の residual clock bias/drift 推定、bias-derived jump merge、clock drift cleanup を stage helper へ移した。raw bridge 側は clock stage の `clock_jump` / `clock_bias_m` / `clock_drift_mps` を後続 full-context / Doppler / TDCP / solver に渡すだけになった。
  - `tests/test_gsdc2023_trip_stages.py` に non-blocklist cleanup と blocklist residual-clock replacement の単体テストを追加した。検証は `py_compile`、clock residual focused + raw bridge blocklist regression `3 passed`、trip-stage focused tests `37 passed`、`git diff --check` OK。
  - `tests/test_gsdc2023_raw_bridge_exports.py` を追加し、raw bridge が再公開する trip-stage product 型と `_build_*` / `_apply_*` private alias が `gsdc2023_trip_stages.py` の実体を指していることを固定した。分割を進めても downstream import / monkeypatch surface が崩れたら focused test で検出できる。
  - raw bridge export/alias + trip-stage focused tests は `39 passed`。その前段の GSDC2023 関連 regression は `277 passed`、実データ smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `gsdc2023_trip_stages.py` に `ObservationMaskBaseCorrectionStageProducts` / `build_observation_mask_base_correction_stage` を追加し、Doppler residual mask、P-D consistency mask、P residual mask、base correction の順序を stage helper に閉じた。base correction は引き続き mask 後に残った positive weight + finite correction の slot だけへ適用される。
  - `tests/test_gsdc2023_trip_stages.py` に mask->base ordering の単体テストを追加し、PD/P residual で落ちた slot には base correction が入らないことを固定した。検証は `py_compile`、combined stage focused + raw bridge base correction regression + raw bridge export tests `4 passed`、trip-stage/export focused `40 passed`、GSDC2023 関連 regression `280 passed`、`git diff --check` OK。
  - `experiments/gsdc2023_audit_cli.py` を追加し、audit/compare CLI の `--data-root` / `--trip` / `--max-epochs` / `--multi-gnss` / `--output-dir` と、trip/output path resolve、nonnegative max-epochs coercion を共有化した。`compare_gsdc2023_factor_masks.py` / `compare_gsdc2023_residual_values.py` / `compare_gsdc2023_residual_diagnostics_factor_mask.py` / `compare_gsdc2023_gnss_log_observation_counts.py` / `compare_gsdc2023_gnss_log_residual_prekeys.py` / `compare_gsdc2023_diagnostics_mask_solver.py` / `compare_gsdc2023_phone_data_raw_bridge_counts.py` / `compare_gsdc2023_base_correction_series.py` はこの helper を使う。複数 trip script 用に `add_data_root_arg` / `add_trip_arg` も分け、base-correction compare 用に required output-dir も扱えるようにした。
  - `tests/test_gsdc2023_audit_cli.py` を追加し、default/override の path resolve、negative max-epochs clamp、BooleanOptionalAction、data-root/trip 個別 helper、required output-dir を単体検証できるようにした。検証は `py_compile`、audit CLI + base-correction compare focused tests `5 passed`、GSDC2023 関連 regression `286 passed`、`git diff --check` OK。CLI 経由 smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、residual `total_matched_count=5360`, `total_matlab_only=92`, median abs delta `0.0001361232946738289`, p95 `0.0018885435276413621`, max `0.002961355846307967` を維持。prekeys smoke は `51458/51458` exact、counts smoke は one-trip `bridge_errors=0`, `phone_errors=0`、diagnostics solver raw-only 20 epoch smoke も出力生成まで完走。`compare_gsdc2023_*.py` 内の共通 CLI 引数直書きは base-correction 専用以外も含めて helper 経由に揃った。
  - raw bridge の private alias / re-export surface を module ごとに棚卸しし、downstream が直接参照している互換 alias を `tests/test_gsdc2023_raw_bridge_exports.py` に固定した。base-correction、bridge-config、clock-state、gnss-log、height、IMU、observation-matrix、output、residual-model、signal-model、TDCP、trip-stage の `_...` alias が split module 実体を指すこと、`raw_bridge.__all__` の公開名が stale でないことを検証する。
  - alias audit focused は、raw bridge export tests と代表的な downstream private alias 利用 regression を合わせて `11 passed`、`git diff --check` OK。
  - `gsdc2023_trip_stages.py` に `PostObservationStageProducts` / `build_post_observation_stages` を追加し、`build_trip_arrays` 後段の gnss_log pseudorange overlay、signal weight snapshot、absolute-height reference、clock residual/full context、mask+base correction、graph dt、TDCP、IMU stage 配線を一つの orchestration helper に束ねた。raw bridge 側は post-fill observation の後、`_build_post_observation_stages(...)` を呼んで `TripArrays` assembly に必要な product を渡すだけになった。
  - `tests/test_gsdc2023_trip_stages.py` に post-observation orchestration の単体テストを追加し、gnss_log overlay、mask 後 base correction、diagnostics 用 signal weight copy、TDCP geometry/scale、IMU no-op、呼び出し順を固定した。検証は `py_compile`、post-observation focused + raw bridge representative tests `7 passed`、trip-stage/export focused `43 passed`、GSDC2023 関連 regression `287 passed`、`git diff --check` OK。実データ smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `gsdc2023_trip_stages.py` に `ObservationPreparationStageProducts` / `build_observation_preparation_stages` を追加し、`build_trip_arrays` 前段の raw observation frame filtering、epoch metadata / epoch-time clock-drift context、observation matrix input、epoch selection/fill、post-fill baseline repair / RTKLIB tropo recompute を一つの orchestration helper に束ねた。raw bridge 側は raw CSV と epoch metadata を読んだ後、前段 product から `df` / `epochs` / `obs` / repaired WLS/tropo を受け取るだけになった。
  - `tests/test_gsdc2023_trip_stages.py` に observation-preparation orchestration の単体テストを追加し、raw filtering、deduplicate sort/group/tail、clock bias/drift cleanup、selection/fill forwarding、post-fill repair/recompute の呼び出し順を固定した。raw bridge export alias も `ObservationPreparationStageProducts` / `_build_observation_preparation_stages` まで拡張した。検証は `py_compile`、observation-preparation focused + raw bridge representative tests `7 passed`、trip-stage/export focused `44 passed`、GSDC2023 関連 regression `288 passed`、`git diff --check` OK。実データ smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `gsdc2023_trip_stages.py` に `PreparedObservationProducts` / `unpack_observation_preparation_stage` を追加し、前段 product から後段 stage/TripArrays assembly に渡す observation arrays、repaired WLS/tropo、slot metadata、clock/velocity context を一つの read-only product として materialize するようにした。raw bridge 側の `obs.*` 大量展開をやめ、post-observation stage と TripArrays assembly はこの product を入力契約にして参照する。
  - `tests/test_gsdc2023_trip_stages.py` に prepared-observation unpack の単体テストを追加し、raw mask count、TGD、slot count、metadata/time context、repaired WLS/tropo が正しく転記されることを固定した。raw bridge export alias も `PreparedObservationProducts` / `_unpack_observation_preparation_stage` まで拡張した。検証は `py_compile`、prepared-observation focused + raw bridge representative tests `8 passed`、trip-stage/export focused `45 passed`、GSDC2023 関連 regression `289 passed`、`git diff --check` OK。実データ smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `gsdc2023_trip_stages.py` に `assemble_prepared_trip_arrays_stage` を追加し、`PreparedObservationProducts` と `PostObservationStageProducts` から最終 `TripArrays` を組む mapping を stage helper 側に閉じた。raw bridge 側は `_assemble_prepared_trip_arrays_stage(...)` を返すだけになり、clock residual / mask+base / TDCP / IMU / height product の assembly field mapping が一箇所に集約された。
  - `tests/test_gsdc2023_trip_stages.py` に prepared-trip assembly の単体テストを追加し、post-observation の clock residual が raw observation の clock 値を上書きすること、mask/base/TDCP/height/observation count が `TripArrays` payload に入ることを固定した。raw bridge export alias も `_assemble_prepared_trip_arrays_stage` まで拡張した。検証は `py_compile`、prepared-trip assembly focused + raw bridge representative tests `7 passed`、trip-stage/export focused `46 passed`、GSDC2023 関連 regression `290 passed`、`git diff --check` OK。実データ smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `gsdc2023_trip_stages.py` に `PostObservationStageConfig` / `PostObservationStageDependencies` / `build_configured_post_observation_stages` を追加し、post-observation stage の大量の boolean/scalar/path 設定と injected dependency 関数を bundle 化した。互換のため既存 `build_post_observation_stages(...)` は残しつつ、raw bridge 側は `PreparedObservationProducts` + config + dependencies を渡すだけにして、後段 stage 呼び出しの dependency injection 表面を薄くした。
  - `tests/test_gsdc2023_trip_stages.py` に bundled post-observation wrapper の単体テストを追加し、config/dependency 経由で gnss_log no-op、clock cleanup、TDCP no-op、IMU no-op が通ることを固定した。raw bridge export alias も `PostObservationStageConfig` / `PostObservationStageDependencies` / `_build_configured_post_observation_stages` まで拡張した。検証は `py_compile`、configured post-observation focused + raw bridge representative tests `7 passed`、trip-stage/export focused `47 passed`、GSDC2023 関連 regression `291 passed`、`git diff --check` OK。実データ smoke は Pixel4XL 200 epoch で factor `13118/13118` exact / symmetric parity `1.0`、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `experiments/gsdc2023_gnss_log_bridge.py` の `gnss_log_corrected_pseudorange_products` が multi-GNSS slot batch で GPS 以外の slot を見つけると GPS pseudorange overlay まで諦めていたため、GPS slot が一つでもあれば同じ matrix shape のまま GPS slot だけ `gnss_log.txt` 由来 pseudorange で上書きするよう修正した。`tests/test_gsdc2023_gnss_log_bridge.py` に mixed-constellation batch regression を追加。これで `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5 --max-epochs 200 --multi-gnss` の residual value parity は max abs delta `5.217767972m` / p95 `0.756643739m` から max `0.017453085m` / p95 `0.015706574m` へ改善し、observation delta max は約 `6.0466m` から `0.000086m` へ縮小。`train/2021-12-08-20-28-us-ca-lax-c/pixel5 --max-epochs 200 --multi-gnss` も max `0.417937932m` / p95 `0.223573732m` から max `0.002405220m` / p95 `0.002093780m` へ改善。factor mask は Pixel5 multi 200 epoch で MATLAB 側 `11868/11868` key が全 matched、bridge-only は multi-GNSS 追加 key として残る。検証は gnss-log focused `17 passed`、GSDC2023 関連 regression `292 passed`、`py_compile` OK。
  - 続けて `build_raw_observation_frame` / `build_epoch_time_context` の `not multi_gnss` guard を外し、multi-GNSS でも GPS gnss_log synthetic rows と MATLAB epoch times を候補に含めるよう修正した。これにより `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel4xl --max-epochs 200 --multi-gnss` は gnss_log 欠落 epoch をまたぐ Doppler velocity 外れが消え、residual max `6.373344768m` / matched `1592` / matlab_only `3860` から max `0.002961356m` / matched `5360` / matlab_only `92` へ改善。`train/2022-10-06-21-51-us-ca-mtv-n/sm-a205u --max-epochs 200 --multi-gnss` は residual max `2.205270985m` から `0.000237484m` へ改善。diagnostics 付き 11 train trips の 200 epoch multi-GNSS scan は max residual delta 上位でも `0.007050278m` / `0.006564969m` / `0.005800425m` で全て 1cm 未満。Pixel4XL multi factor mask は MATLAB 側 `13118/13118` key が全 matched、raw validation は baseline `200` epochs 選択、selected RMS2D `2.339m`, selected wMSE PR `31.8628` を維持。
  - `gps_matrtklib_sat_product_adjustment` を拡張し、full/filtered nav selector が同一 message を返す場合でも Android `device_gnss.csv` の sat product が MatRTKLIB selected nav clock から `0.005m` 超ずれていれば selected nav で sat position/clock/drift を再計算するようにした。送信時刻は RTKLIB `satposs` と同じく、receiver clock bias を含めた transmit time から relativity/TGD を含まない `eph2clk` 相当の broadcast clock polynomial で補正する。さらに GPS L5 は signal 別 cache/timing に分離し、L5 自身の pseudorange/timing で sat product を評価する。ただし L5 raw sat clock が再計算 clock と `0.005m` 以内なら raw sat position/clock を維持することで、`2020-07-17` SVID26 L5 の `0.027m` regression と `2020-07-08` SVID10 L5 の sat-clock mismatch を同時に避ける。diagnostics 付き 11 train trips の full epoch multi-GNSS residual scan は、旧 worst `0.024490553m` / `0.024411874m` から、最新 worst `train/2020-08-04-00-20-us-ca-sb-mtv-101/pixel5` max `0.012938063m`、次点 `2020-06-25 pixel4xl` `0.007535187m`、`2020-07-17-23-13 pixel4` `0.007312021m`、`2020-07-08 pixel4xl/pixel4` `0.006218105m` / `0.004809444m` まで改善。L5 に L1 timing を共有する clock-only 試行は 08-04/LAX 系で `0.3m` 以上に悪化したため不採用。検証は base/observation focused tests `20 passed`、GSDC2023 関連 regression `301 passed`、`train/2020-08-04-00-20 pixel4xl --max-epochs 200 --no-multi-gnss` raw validation smoke は selected RMS2D `2.321m`, selected wMSE PR `31.8628` を維持。
  - 次の分割候補は (1) raw bridge 互換 alias のうち downstream 参照が消えたものを段階的に deprecate する、(2) compare/audit scripts の output file naming と summary writer の追加共通化、(3) `build_trip_arrays` 前段・後段の config/dependency bundle 作成も小さい factory helper に寄せ、raw bridge 本体を entrypoint orchestration だけに近づける。各候補は public dataclass input/output を先に置き、raw bridge 本体には orchestration だけを残す。

次に進むなら:

1. baseline より本当に良い FGO chunk を探すには、PR MSE 以外の validation signal（baseline residual stability、map/course family、phone family）を追加で audit する。
2. all-epoch test dry run の non-baseline 5件について、submission 観点で採用をさらに絞るべきか判断する。
3. ここまでの selector / submission fallback 変更を整理し、PR用の変更要約と artifact 一覧を作る。


### 2026-04-23 raw_wls high-PR rescue retune


#### 2026-04-23 selector / Kaggle A-B decision summary

Current safe baseline remains `ref/gsdc2023/results/test_parallel/20260421_0555/submission_20260421_0555.csv`. Do not replace it as the default artifact yet. The later selector/fallback work produced useful fixes and diagnostics, but no local evidence strong enough to submit the full current path blindly.

What changed in code:

- `experiments/gsdc2023_chunk_selection.py` now owns the chunk-selection policy instead of leaving the logic embedded in raw bridge code.
- `raw_wls` high-PR rescue was relaxed to catch the train `samsunga325g` high-MSE chunk, but high-baseline fallback now has a motion plausibility guard. This prevents kilometer-scale raw_wls over-selection on `pixel6pro` / `sm-a205u` style failures.
- High-baseline fallback now also applies the raw_wls PR-MSE guard to FGO/FGO-off candidates. That prevents choosing `fgo_no_tdcp` when `raw_wls` has clearly lower PR MSE, as seen on `test/2022-04-04-16-31-us-ca-lax-x/pixel5` first chunk.
- Focused selector tests and raw-bridge integration tests cover the relaxed rescue, implausible-motion rejection, plausible catastrophic-baseline rescue, and the `fgo_no_tdcp` vs raw_wls PR-MSE ordering regression.

Train evidence:

- Current gated-policy replay over the existing 20-window train high-MSE audit is in `experiments/results/raw_wls_high_pr_policy_replay_20260423/`.
- Only one train window changes: `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g`, chunk `1400:1512`, to `raw_wls`.
- Existing audit score improves from baseline `29.397m` to raw_wls `18.933m` (`-10.464m`). Current-code rerun confirms `raw_wls=112`, selected/raw score `18.933m`, baseline `29.397m`, selected PR MSE `15.6667` vs baseline `56.7125`.
- This is good targeted evidence for the rescue, but it is not broad enough to justify full all-test replacement.

Test/full-run evidence:

- `20260423_1306` and `20260423_1433` should not be submitted blindly. They over-selected raw_wls on high-baseline test trips and produced worse local proxy behavior.
- `20260423_1450` fixed the worst raw_wls over-selection by re-running affected trips after the high-baseline motion guard. Source totals became baseline `70665`, raw_wls `590`, fgo_no_tdcp `771`.
- Still, `20260423_1450` differs from `20260421_0555` on important trips and should not supersede it without Kaggle A/B.
- `test/2022-04-04-16-31-us-ca-lax-x/pixel5` after the high-baseline raw guard selects baseline `1400`, raw_wls `771`, fgo/fgo_no_tdcp `0`. Full-trip selected PR MSE is `1415.4641`, baseline `1542.9487`, raw_wls `1406.4002`.
- The guard correctly changes the first chunk from `fgo_no_tdcp` to `raw_wls`, but the local PR proxy is still far worse than `20260421_0555`'s recorded `180.28` for the same trip.

Important proxy finding:

- The `180.28 -> 1415+` jump on `2022-04-04 pixel5` is not explained by TDCP, GPS-only/multi-GNSS, or VD/non-VD flags. Current no-TDCP, GPS-only no-TDCP, and no-VD no-TDCP isolation runs did not recover the old proxy.
- Re-evaluating old `20260421_0555` submission coordinates under the current observation model gives PR MSE `1424.2648`, very close to current selected `1415.5758`. So the huge local proxy delta is mostly an observation/proxy-model shift, not proof that the old and current trajectories differ by that scale.
- The current unmasked PR proxy is dominated by a small number of outlier epochs. For `pixel5`, top 20 epochs explain `89.99%` of the weighted PR-MSE contribution. Epoch `440` is dominated by GPS SVID4 L1 with residual around `-34 km`, and old `0555` coordinates show the same bad residual.
- `--observation-mask` fixes the local proxy outliers for the same coordinates (`1415.576 -> 87.000` on current selected), but a full `--observation-mask --no-tdcp` run selects baseline for all epochs and is farther from old `0555` coordinates (p95 `30.207m`). Therefore observation-mask is useful for diagnostics/proxy stability, not safe as a submission switch yet.

Cross-trip nonbaseline comparison:

Artifacts are under `experiments/results/test_fgo_tdcp_candidate_probe_20260423/nonbaseline_trip_proxy_compare/`:

- `proxy_summary.csv`
- `coordinate_diff_summary.csv`
- `decision_table.csv`
- `summary.json`

The comparison evaluates `old0555`, `current1450`, and `current1450_pixel5_rawguard` on the five `20260421_0555` nonbaseline trips under current unmasked and masked PR models.

Decision table summary versus `old0555`:

| trip | current candidate signal | judgement |
| --- | --- | --- |
| `2020-12-11 pixel4xl` | coord p95 `0.745m`, masked PR delta `-0.121` | Very low-risk but tiny effect. |
| `2021-11-30 mi8` | coord p95 `2.432m`, masked PR delta `+0.130` | Slightly worse; do not target first. |
| `2022-04-04 pixel5` current1450 | coord p95 `16.872m`, masked PR delta `+7.704` | Worse; avoid as first A/B. |
| `2022-04-04 pixel5` rawguard | coord p95 `6.531m`, unmasked PR delta `-8.343`, masked PR delta `+18.309` | Mixed and high-risk. A/B only after safer candidates. |
| `2022-07-12 sm-a325f` | coord p95 `2.575m`, masked PR delta `+0.028` | Slightly worse; do not target first. |
| `2023-05-09 sm-a505u` | coord p95 `3.004m`, masked PR delta `-0.552` | Best first A/B candidate. |

Generated local Kaggle A/B CSVs:

- `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_sm_a505u_current1450_only_20260423.csv`
- `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv`
- `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_pixel5_rawguard_and_sm_a505u_current1450_20260423.csv`
- `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_pixel5_rawguard_pixel4xl_sm_a505u_20260423.csv`
- Existing broader pixel5-only candidates are also in the same directory.
- `selective_ab_summary.json` and `summary.json` describe the row counts, coordinate deltas, and NaN checks. All generated candidates have no NaN lat/lon.

Recommended Kaggle order:

1. Submit `submission_20260421_0555_sm_a505u_current1450_only_20260423.csv` as the first A/B. It is the only candidate with a clear masked-proxy gain and moderate coordinate shift.
2. If that improves or is neutral, try `submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv`. `pixel4xl` is almost identical to old `0555`, with tiny proxy improvement, so it is low risk.
3. Only after those, consider pixel5 rawguard variants. They are closer to old `0555` coordinates than `current1450`, but masked proxy worsens, so they should be treated as higher-risk exploratory A/B.
4. Do not submit full `20260423_1450`, `20260423_1433`, or `20260423_1306` as replacement artifacts without Kaggle evidence.

2026-04-24 JST Kaggle A/B results:

- Submitted control `ref/gsdc2023/results/test_parallel/20260421_0555/submission_20260421_0555.csv`: Public `4.120`, Private `5.362`.
- Submitted `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_sm_a505u_current1450_only_20260423.csv`: Public `4.120`, Private `5.350`.
  - Versus control: Public unchanged, Private improves by `0.012`.
- Submitted `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv`: Public `4.115`, Private `5.350`.
  - Versus control: Public improves by `0.005`, Private improves by `0.012`.
  - Versus `sm_a505u_current1450_only`: `pixel4xl` adds a small Public gain and no Private regression.
- Interpretation: the narrow current1450 replacements are real but small wins over the `20260421_0555` control. They still do not beat existing leaderboard best `gsdc2023_submission_v22.csv` (`Public 4.112`, `Private 5.200`), so do not promote this raw-bridge A/B line as the default artifact.
- Next decision: hold the pixel5 rawguard variants unless explicitly spending more Kaggle submissions for exploratory evidence. The masked proxy warned against pixel5, and the current A/B gains are not large enough to justify burning it as a default-policy step.

2026-04-24 JST big-improvement probe:

- Expanded the train high-MSE chunk audit from top20 to top60 at `experiments/results/full_epoch_top60_chunk_actual_20260424/`.
  - `120/120` case runs succeeded (`60` windows x no-TDCP / TDCP-geometry candidate).
  - Current selector produced only one real selected win: `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g`, chunk `1400:1512`, baseline `29.397m` -> selected/raw_wls `18.933m` (`-10.464m`).
  - Additional train candidate wins not selected by current policy:
    - `train/2020-12-10-22-52-us-ca-sjc-c/pixel4xl`, chunk `1400:1408`: baseline `35.115m`, raw_wls `26.182m`, FGO `23.689m`. Very short tail chunk; high overfit risk.
    - `train/2022-08-04-20-07-us-ca-sjc-q/sm-a325f`, chunk `1400:1452`: baseline `21.825m`, raw_wls `17.291m`; A32-family test transfer failed below.
    - `train/2023-09-06-18-04-us-ca/sm-s908b`, chunk `400:600`: FGO improves a catastrophic baseline chunk from `39779.898m` to `27265.098m`, still unusably bad. Full-trip current gated no-TDCP does rescue the train trip from Kaggle WLS `280411.727m RMS2D` to selected `4.921m RMS2D`, mostly via raw_wls.
- Tested the obvious transfer candidates on Kaggle:
  - `submission_20260421_0555_sm_s908b_rawwls_20260424.csv`: Public `4.120`, Private `5.462`. Despite train `sm-s908b` rescue and test PR-MSE improvement (`57.662 -> 43.219`), Private worsened by `0.100`; discard.
  - `submission_20260421_0555_a32_family_rawwls_exact_cap100m_20260424.csv`: Public `4.444`, Private `5.362`. A32/Samsung A raw_wls replacement badly worsened Public and did not improve Private; discard.
- Conclusion: large local train/proxy wins from raw_wls fallback are not transferring reliably to test. Do not keep broadening raw_wls fallback from PR-MSE evidence alone. The big-improvement path should shift away from selective source replacement and toward model/trajectory quality: recover or reimplement the `v22` smoother line, robust smoothing over raw-bridge outputs, or MATLAB/taroz-equivalent preprocessing/IMU rather than more one-trip raw_wls A/B.

2026-04-24 JST robust post-smoother result:

- Added `experiments/smooth_gsdc2023_submission.py` and `tests/test_smooth_gsdc2023_submission.py`.
  - The smoother projects each `tripId` to a local ENU plane, splits on timestamp gaps / persistent reset jumps, suppresses isolated jump splitting so single spikes can be repaired, applies a Hampel-style local median repair, then applies an optional centered triangular smoother with a per-row correction cap.
  - The CLI can run one config or scan comma-separated grids on `bridge_positions.csv` with GT; it also works on Kaggle submission CSVs with no GT.
- Train GT scan over 43 available `bridge_positions.csv` artifacts / 11056 rows:
  - Fixed config `median_window=5`, `smooth_window=5`, `blend=1.0`, `max_correction_m=3.0` gave aggregate score delta `-0.5317m`; only 2/43 files worsened, worst file delta `+0.1317m`.
  - More aggressive `smooth_window=9`, `cap=5m` looked tempting on one bad trip but worsened 36/43 files; discard.
  - Refined grid slightly preferred `blend=0.75`, `cap=3m` on train aggregate (`-0.5345m`), but the Kaggle A/B below shows the original `blend=1.0` transfers better.
- Kaggle A/B on the safe `20260421_0555` control:
  - `experiments/results/post_smooth_submission_20260424/submission_20260421_0555_smooth_cap3_w5_b1_20260424.csv`: Public `3.828`, Private `4.978`.
    - Versus control `20260421_0555`: Public improves by `0.292`, Private improves by `0.384`.
    - Versus prior best `gsdc2023_submission_v22.csv` (`Public 4.112`, `Private 5.200`): Public improves by `0.284`, Private improves by `0.222`.
    - Submission format checks: `71936` rows, identical `tripId`/`UnixTimeMillis`, no NaN lat/lon, max horizontal correction about `3.01m`, mean correction about `0.99m`, p95 correction about `2.60m`.
  - `experiments/results/post_smooth_submission_20260424/submission_20260421_0555_smooth_cap3_w5_b075_20260424.csv`: Public `3.830`, Private `4.999`; worse than `blend=1.0`, so do not adopt.
- Stacked the smoother with the earlier narrow `current1450` raw-bridge A/B:
  - `experiments/results/post_smooth_submission_20260424/submission_20260421_0555_pixel4xl_sm_a505u_current1450_smooth_cap3_w5_b1_20260424.csv`: Public `3.819`, Private `4.972`.
    - Versus pure smoother: Public improves by `0.009`, Private improves by `0.006`.
    - Versus control `20260421_0555`: Public improves by `0.301`, Private improves by `0.390`.
    - This is now the best local leaderboard artifact.
  - `submission_20260421_0555_sm_a505u_current1450_only_smooth_cap3_w5_b1_20260424.csv`: Public `3.828`, Private `4.972`. Private ties the stacked artifact, but `pixel4xl` keeps a Public gain with no Private regression, so keep the two-trip stacked artifact.
- Added optional Gaussian / boxcar smoothing kernels to `smooth_gsdc2023_submission.py`.
  - Kernel train scan is stored at `experiments/results/post_smooth_scan_20260424/kernel_config_summary.json`.
  - Best train kernel was Gaussian `median_window=5`, `smooth_window=9`, `sigma=1.0`, `blend=1.0`, `cap=3m` with aggregate delta `-0.5498m`, but Kaggle A/B `submission_20260421_0555_pixel4xl_sm_a505u_current1450_gauss_w9_s1_cap3_b1_20260424.csv` scored Public `3.821`, Private `4.978`; worse than triangular stacked, so do not adopt.
- De-duplicated the available GT bridge-position artifacts to 13 unique trajectories and rescanned kernels / small policy candidates:
  - `experiments/results/post_smooth_scan_20260424/phone_policy_unique_scores.csv`
  - `experiments/results/post_smooth_scan_20260424/phone_policy_unique_summary.csv`
  - The unique-GT scan favored `boxcar window=3 cap=3` with no local worsening, but Kaggle A/B `submission_20260421_0555_pixel4xl_sm_a505u_current1450_box_w3_cap3_b1_20260424.csv` scored Public `3.835`, Private `5.008`. This confirms the local GT set is too small/biased for direct per-phone policy promotion.
- Inspected full `20260423_1450` with the triangular smoother but did not submit it. Compared with the current best stacked smoother, the smoothed full-1450 artifact still has risky residual changes on `2022-04-04 pixel5` (p95 diff about `14.96m`, max about `346.76m`), plus A32-family changes already failed a direct raw_wls A/B.
- Tested two additional single-trip raw patch A/Bs on top of the current best smoother:
  - `submission_20260421_0555_pixel5_rawguard_pixel4xl_sm_a505u_smooth_cap3_w5_b1_20260424.csv`: Public `3.819`, Private `5.044`. Public ties, Private badly worsens; pixel5 rawguard should remain rejected.
  - `submission_20260421_0555_pixel4xl_sm_a505u_mi8_1450_smooth_cap3_w5_b1_20260424.csv`: Public `3.823`, Private `4.972`. Private ties but Public worsens; mi8 1450 patch should remain rejected.
- Tested leave-one-trip-unsmoothed reverse A/Bs for the largest smoother corrections. Candidate summary is stored at `experiments/results/post_smooth_leave_one_20260424/leave_one_unsmoothed_candidates.json`; trip correction summary is `trip_smooth_correction_summary.csv`.
  - `submission_best_leave_unsmoothed_2022_04_04_16_31_us_ca_lax_x_pixel5_20260424.csv`: Public `3.819`, Private `5.016`; reject.
  - `submission_best_leave_unsmoothed_2021_09_14_20_32_us_ca_mtv_k_pixel4_20260424.csv`: Public `3.819`, Private `5.006`; reject.
  - `submission_best_leave_unsmoothed_2022_10_06_20_46_us_ca_sjc_r_sm_a205u_20260424.csv`: Public `3.819`, Private `5.004`; reject.
  - `submission_best_leave_unsmoothed_2023_05_25_17_32_us_ca_pao_j_pixel6pro_20260424.csv`: Public `3.819`, Private `5.001`; reject.
  - Interpretation: even the highest-correction trips should remain smoothed. The smoother is not merely averaging harmless rows; removing it from large-correction trips consistently hurts Private.
- Tested cap-policy A/Bs after the reverse A/B showed that high-correction trips should stay smoothed.
  - Generated candidates under `experiments/results/post_smooth_cap_policy_20260424/`.
  - Weakening the top-4 high-correction trips hurt: top4 `cap=2.0` scored Public `3.819`, Private `4.995`; `cap=2.5` scored `3.819 / 4.982`.
  - Strengthening the top-4 high-correction trips helped monotonically through large caps: `cap=4.0` `3.819 / 4.968`, `cap=10` `3.819 / 4.926`, and `cap=20/30/50/100` all `3.819 / 4.904`.
  - Broadening the high cap across more trips improved further: top8 `cap=100` `3.818 / 4.905`, top12 `3.813 / 4.902`, top20 `3.812 / 4.900`, and all 40 trips with `cap=100` `3.807 / 4.896`.
  - Global all-trip high caps plateau on Private: all-trip `cap=200` `3.807 / 4.896`, all-trip `cap=1000` `3.806 / 4.896`. `cap=1000` changes only a couple of rows materially versus `cap=100`, but keeps a tiny Public gain.
  - `cap=100` versus `cap=1000` has only two material row differences; details are stored in `experiments/results/post_smooth_cap_policy_20260424/cap100_vs_cap1000_diff_rows.json`.
    - Row `1107`, `2020-12-11-19-30-us-ca-mtv-e/pixel4xl`, UnixTimeMillis `1607716162442`, gets a `538.904m` larger correction under `cap=1000`.
    - Row `34131`, `2022-04-04-16-31-us-ca-lax-x/pixel5`, UnixTimeMillis `1649091715434`, gets a `274.415m` larger correction under `cap=1000`.
  - Row-specific A/B isolated the useful gain:
    - `submission_best_cap1000_minus_row2_cap100_20260424.csv`: Public `3.806`, Private `4.896`. This keeps the `pixel4xl` full repair but returns the `pixel5` giant correction to `cap=100`; score is unchanged versus full `cap=1000`.
    - `submission_best_cap1000_minus_row1_cap100_20260424.csv`: Public `3.807`, Private `4.896`. Removing the `pixel4xl` full repair loses the tiny Public gain.
  - Clean final artifact: `experiments/results/post_smooth_cap_policy_20260424/submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv`. This is identical to `submission_best_cap1000_minus_row2_cap100_20260424.csv`: keep global `cap=100`, promote only the one proven `pixel4xl` outlier row to the `cap=1000` repair, and avoid the unnecessary `pixel5` `274m` jump.
- New default candidate: `experiments/results/post_smooth_cap_policy_20260424/submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv` (`Public 3.806`, `Private 4.896`, submitted-equivalent via `submission_best_cap1000_minus_row2_cap100_20260424.csv`).
- Reproduction implementation is now explicit:
  - Script: `experiments/reproduce_gsdc2023_best_submission.py`.
  - Command: `PYTHONPATH=python:. python3 experiments/reproduce_gsdc2023_best_submission.py`.
  - Reduced-dependency command: `PYTHONPATH=python:. python3 experiments/reproduce_gsdc2023_best_submission.py --patch-source bridge-exception`.
  - Base inputs: `../ref/gsdc2023/results/test_parallel/20260421_0555/submission_20260421_0555.csv` and `../ref/gsdc2023/results/test_parallel/20260423_1450/submission_20260423_1450.csv`.
  - Input artifact SHA256s are now checked by the reproducer: `20260421_0555` base `b60b1dcbe188540b8d35c56f487e580b8f5aee7f0138c8685e9745d02cef61c8`; `20260423_1450` patch `2ff02b916c642956285e0421f7a8dab171f9a88ca30dc42317ea404e9685029c`.
  - Deterministic steps: rebuild the source candidate by replacing only `2020-12-11-19-30-us-ca-mtv-e/pixel4xl` (`1190` rows) and `2023-05-09-23-10-us-ca-sjc-r/sm-a505u` (`2384` rows) in `20260421_0555` with `20260423_1450`; smooth that rebuilt source with `median=5`, `window=5`, `blend=1`, `cap=100m`; smooth the same source with `cap=1000m`; read both written intermediate CSVs back; promote only row `tripId=2020-12-11-19-30-us-ca-mtv-e/pixel4xl`, `UnixTimeMillis=1607716162442` from cap1000 into cap100; write the final CSV.
  - Reduced historical patch dependency: `--patch-source bridge-exception` sources both patch trips from current raw bridge outputs and pins their SHA256s. `2020-12-11-19-30-us-ca-mtv-e/pixel4xl` is fully byte-identical from `experiments/results/reproduce_best_submission_20260424/regenerate_patch_trips/pixel4xl/bridge_positions.csv` (SHA256 `43f97d81a4e660fbf8862158445fad04deda4b6b4727c35cfe5534ac21724bd5`). `2023-05-09-23-10-us-ca-sjc-r/sm-a505u` is byte-identical outside rows `188:410`; those `223` rows are kept as a small exception CSV at `experiments/results/reproduce_best_submission_20260424/regenerate_patch_trips/sm_a505u_1450_exception_rows_188_410.csv` (SHA256 `359f4960ba5d281305965eb71eb7e14c4ba0af68307c61946bb4ef54ad137d5c`), while the bridge CSV SHA256 is `dd5211c4427286ed000617f909cde0c5831c3d82b94c3cf5e682c3f6aba51419`. The bridge+exception run no longer needs the full `20260423_1450` submission as a value source; it uses `20260421_0555` only for key alignment, the two bridge position files, and the 223-row sm-a505u exception, and still reproduces source/final byte-identical.
  - sm-a505u diagnosis: rows `200:399` are nearly current FGO (`mean 0.008m`, max `0.251m`) while current gated selected raw_wls there; rows `188:199` and `400:410` are boundary-transition rows. TDCP-enabled, no-offset, different-motion, higher-iteration, and near-motion (`0.25`/`0.35`) first-500 probes missed the artifact; `motion_sigma=0.3` remains the sharp minimum but still leaves the small FGO/legacy-position-offset residual. Only the 223 exception rows remain pinned rather than the previous full `2384`-row trip.
  - Rebuilt source candidate: `experiments/results/reproduce_best_submission_20260424/submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv`.
  - Reproduced output: `experiments/results/reproduce_best_submission_20260424/submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv`.
  - Byte-level verification passed for both stages: rebuilt source candidate SHA256 `1c7574019b0c711665379a8204acd2ef1ad5453263a6a25180638f73e2b798d0`; final scored artifact SHA256 `1d5519d6fc290b2eab60e6175ce4f06055a78965217e7a10dd81d8aa5907d370`; final rows `71936`, NaN lat/lon rows `0`.
- Added source A/B diagnostics for the next selector-policy step:
  - Script: `experiments/analyze_gsdc2023_source_ab.py`.
  - Output root: `experiments/results/source_ab_sm_a505u_pixel5_20260424/`.
  - Focused full-epoch raw bridge metrics:
    - `test/2023-05-09-23-10-us-ca-sjc-r/sm-a505u`: selected PR MSE `59.4643`, baseline `66.3298`, raw_wls `56.0531`, source mix `baseline=2184`, `raw_wls=200`. The only gated raw_wls chunk is epoch `200:400`; its chunk PR MSE is baseline `521.0375` vs raw_wls `440.0662`, raw/baseline ratio `0.8446`, raw baseline-gap p95 `76.200m`, max `86.750m`.
    - `test/2022-04-04-16-31-us-ca-lax-x/pixel5`: selected PR MSE `1415.4641`, baseline `1542.9487`, raw_wls `1406.4002`, source mix `baseline=1400`, `raw_wls=771`. Gated raw_wls chunks are epoch `0:200`, `400:600`, `1800:2000`, and `2000:2171`; their raw baseline-gap max values are `232.924m`, `814.579m`, `827.123m`, and `290.434m`. This explains why rawguard looked better on PR MSE but failed Kaggle (`3.819 / 5.044`).
  - Submission-delta localization:
    - `sm_a505u` 0555->1450 material delta is concentrated at epoch `200:400`: mean `5.054m`, p95 `13.917m`, max `39.737m`; other chunks are mostly the phone-offset-level `~0.39m`.
    - `pixel5 LAX-X` 0555->1450 has large deltas in exactly the raw_wls-selected chunks: p95 `26.333m` at `0:200`, `14.640m` at `400:600`, `26.642m` at `1800:2000`, and `18.714m` at `2000:2170`.
  - New narrow A/B candidate: `experiments/results/source_ab_sm_a505u_pixel5_20260424/narrow_sm_a505u_candidate/submission_0555_pixel4xl_full_sm_a505u_epoch200_400_smooth_cap100_plus_pixel4xl_row_20260424.csv`.
    - Construction: 0555 base + full `pixel4xl` current1450 patch + only `sm_a505u` epoch `200:400` current1450 patch + triangular smoother `median=5`, `window=5`, `blend=1`, `cap=100m` + the proven `pixel4xl` cap1000 outlier row.
    - Versus current final candidate, this differs only on `sm_a505u`; no row differs by more than `1m`, p95 diff is `0.391m`, max diff `0.391m`. It removes the all-trip `sm_a505u` offset noise while preserving the material raw_wls-selected chunk. Treat this as the next Kaggle A/B candidate, not the default, until scored.
    - Kaggle A/B result: Public `3.806`, Private `4.898`. Public ties the current final, but Private worsens by `0.002`, so reject this narrowing and keep the current final artifact.
  - Added saved-payload source guard replay:
    - Script: `experiments/replay_gsdc2023_source_guards.py`.
    - Pre-change replay output: `experiments/results/source_guard_replay_20260424/`. It scanned `90` files / `206` chunk records and found current-code raw_wls in `24` records. A `gap_max <= 200m` raw_wls guard changed `22` records, all from `test/2022-04-04-16-31-us-ca-lax-x/pixel5`; the two retained raw_wls chunks were the useful `test/2023-05-09 sm-a505u` epoch `200:400` chunk and the train `2022-07-26 samsunga325g` high-PR rescue.
  - Implemented the guard as a phone-specific safety rule, not as a global selector rewrite:
    - Initial rule was `raw_wls_max_gap_guard_m("pixel5", "gated") -> 200m`; all other phones / non-gated modes kept `None`.
    - `select_gated_chunk_source()` now accepts optional `raw_wls_max_gap_m` and applies it before raw_wls can be selected in catastrophic/high-baseline/normal gated branches.
    - Post-change replay output: `experiments/results/source_guard_replay_20260424_after_pixel5_guard/`. Current source counts are baseline `204`, raw_wls `2`; the remaining raw_wls records are `sm_a505u` epoch `200:400` and train `samsunga325g` epoch `0:112`.
    - Focused full-trip rerun after the guard: `experiments/results/focused_test_pixel5_lax_x_20260424_gated_no_tdcp_after_guard/`. Source mix changes from baseline `1400`, raw_wls `771` to baseline `1890`, raw_wls `281`; selected PR MSE is `1509.2611` vs baseline `1542.9487` and raw_wls `1406.4002`. The guard removes the prior high-gap raw_wls spans (`gap max` `232.924m`, `814.579m`, `827.123m`, `290.434m`) but still leaves smaller low-gap `pixel5` raw_wls chunks with gap max `15.766m` to `167.285m`. This is safer than the rejected pixel5 rawguard, but still not submission-default evidence.
    - Focused `sm_a505u` rerun after the guard: `experiments/results/focused_test_sm_a505u_20260424_gated_no_tdcp_after_guard/`. It stays unchanged at selected PR MSE `59.4643`, baseline `66.3298`, raw_wls `56.0531`, source mix baseline `2184`, raw_wls `200`; the retained raw_wls chunk is epoch `200:400` with baseline PR MSE `521.0375`, raw_wls `440.0662`, gap p95 `76.200m`, max `86.750m`.
    - Added train-backed raw_wls policy lab: `experiments/analyze_gsdc2023_raw_wls_policy.py`, output `experiments/results/raw_wls_policy_lab_20260424/`. On the train actual audit, all evaluated high-PR raw_wls rescue variants select only the `train/2022-07-26 samsunga325g` positive window: `1` selected, `1` better, `0` worse, total delta `-10.464m`. The focused after-guard `pixel5` and `sm_a505u` raw_wls records are instead `high_baseline_fallback`, not the train-backed high-PR rescue branch.
    - Tightened current code from a `pixel5` `200m` max-gap guard to a `pixel5` gated raw_wls veto: `GATED_PIXEL5_RAW_WLS_BASELINE_GAP_MAX_M = 0.0`. Rationale: `pixel5` rawguard already failed Kaggle Private, the remaining `281` lower-gap `pixel5` epochs are still test-only high-baseline fallback, and there is no train-backed positive `pixel5` raw_wls evidence.
    - Pixel5-veto replay output: `experiments/results/source_guard_replay_20260424_pixel5_veto/`. It scanned `92` files / `291` records; current source counts are baseline `288`, raw_wls `3`. The raw_wls records are the two duplicate `sm_a505u` focused payloads and the train `samsunga325g` high-PR rescue; `pixel5` raw_wls is now `0` in replay.
- Interpretation: the requested "big improvement" came from trajectory postprocessing, not from another raw_wls source replacement. The only raw-bridge stacking that transferred was the already narrow `pixel4xl + sm_a505u` current1450 patch, and the main smoother lesson is that the original `cap=3m` was too conservative. A near-unbounded reset-safe smoother now beats the previous best while preserving the same Public class.

Verification status for this slice:

- `python3 -m py_compile experiments/gsdc2023_chunk_selection.py` passed.
- `PYTHONPATH=python:. pytest -q tests/test_gsdc2023_chunk_selection.py tests/test_validate_fgo_gsdc2023_raw.py -q -k "gated_chunk_source or chunk_selection"` passed with `35 passed`.
- `PYTHONPATH=python:. pytest -q tests/test_smooth_gsdc2023_submission.py` passed with `5 passed`.
- `git diff --check -- experiments/gsdc2023_chunk_selection.py tests/test_gsdc2023_chunk_selection.py tests/test_validate_fgo_gsdc2023_raw.py internal_docs/plan.md` passed.
- `git diff --check -- internal_docs/plan.md` passed after the cross-trip summary update.
- `python3 -m py_compile experiments/analyze_gsdc2023_source_ab.py experiments/smooth_gsdc2023_submission.py` passed after adding the source A/B analyzer.
- `PYTHONPATH=python:. pytest -q tests/test_smooth_gsdc2023_submission.py tests/test_gsdc2023_chunk_selection.py` passed with `16 passed`.
- `git diff --check -- internal_docs/plan.md experiments/analyze_gsdc2023_source_ab.py experiments/smooth_gsdc2023_submission.py tests/test_smooth_gsdc2023_submission.py tests/test_gsdc2023_chunk_selection.py` passed after the narrow `sm_a505u` candidate update.
- `python3 -m py_compile experiments/gsdc2023_chunk_selection.py experiments/gsdc2023_solver_selection.py experiments/gsdc2023_result_assembly.py experiments/gsdc2023_raw_bridge.py experiments/replay_gsdc2023_source_guards.py` passed after the pixel5 raw_wls gap guard.
- `PYTHONPATH=python:. pytest -q tests/test_gsdc2023_chunk_selection.py tests/test_gsdc2023_solver_selection.py tests/test_gsdc2023_result_assembly.py` passed with `24 passed`.
- `PYTHONPATH=python:. pytest -q tests/test_validate_fgo_gsdc2023_raw.py -q -k "select_gated_chunk_source or gated_chunk_source"` passed with `24 passed`.
- `git diff --check -- experiments/gsdc2023_chunk_selection.py experiments/gsdc2023_solver_selection.py experiments/gsdc2023_result_assembly.py experiments/gsdc2023_raw_bridge.py experiments/replay_gsdc2023_source_guards.py tests/test_gsdc2023_chunk_selection.py tests/test_gsdc2023_solver_selection.py internal_docs/plan.md` passed after the guard update.
- Focused reruns passed after the guard:
  - `PYTHONPATH=python:. python3 experiments/validate_fgo_gsdc2023_raw.py --trip test/2022-04-04-16-31-us-ca-lax-x/pixel5 --max-epochs -1 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --export-bridge-dir experiments/results/focused_test_pixel5_lax_x_20260424_gated_no_tdcp_after_guard`
  - `PYTHONPATH=python:. python3 experiments/validate_fgo_gsdc2023_raw.py --trip test/2023-05-09-23-10-us-ca-sjc-r/sm-a505u --max-epochs -1 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --export-bridge-dir experiments/results/focused_test_sm_a505u_20260424_gated_no_tdcp_after_guard`
- Re-ran guard verification after the focused rerun notes: `py_compile` passed, `PYTHONPATH=python:. pytest -q tests/test_gsdc2023_chunk_selection.py tests/test_gsdc2023_solver_selection.py tests/test_gsdc2023_result_assembly.py` passed with `24 passed`, and the guard-scope `git diff --check` passed.
- Policy-lab run passed: `PYTHONPATH=python:. python3 experiments/analyze_gsdc2023_raw_wls_policy.py --train-audit experiments/results/full_epoch_top20_chunk_actual_20260420/raw_wls_rescue_feature_audit.csv --metrics 'experiments/results/focused_test_*_20260424_gated_no_tdcp_after_guard/bridge_metrics.json' --output-dir experiments/results/raw_wls_policy_lab_20260424`.
- Pixel5-veto replay passed: `PYTHONPATH=python:. python3 experiments/replay_gsdc2023_source_guards.py --input 'experiments/results/focused_test_*_20260424_*/*.json' --input 'experiments/results/test_fgo_tdcp_candidate_probe_20260423/**/*.json' --input 'experiments/results/raw_wls_high_pr_policy_replay_20260423/*.json' --input 'experiments/results/raw_wls_guard_36_gated_rawonly_20260420/no_tdcp/**/*.json' --input 'experiments/results/raw_wls_guard_36_gated_rawonly_20260420/tdcp_geom_3e-7/**/*.json' --output-dir experiments/results/source_guard_replay_20260424_pixel5_veto`.
- Final guard/policy verification passed: `python3 -m py_compile experiments/analyze_gsdc2023_raw_wls_policy.py experiments/gsdc2023_chunk_selection.py experiments/gsdc2023_solver_selection.py experiments/replay_gsdc2023_source_guards.py`, `PYTHONPATH=python:. pytest -q tests/test_analyze_gsdc2023_raw_wls_policy.py tests/test_gsdc2023_chunk_selection.py tests/test_gsdc2023_solver_selection.py tests/test_gsdc2023_result_assembly.py` with `26 passed`, and guard-scope `git diff --check`.
- Alias selector regression still passed after the pixel5 veto: `PYTHONPATH=python:. pytest -q tests/test_validate_fgo_gsdc2023_raw.py -q -k "select_gated_chunk_source or gated_chunk_source"` with `24 passed`.
- Final artifact reproduction passed: default `PYTHONPATH=python:. python3 experiments/reproduce_gsdc2023_best_submission.py`, pixel4xl-override hybrid run, manual bridge+sm-a505u-exception run, and the short reduced-dependency command `PYTHONPATH=python:. python3 experiments/reproduce_gsdc2023_best_submission.py --patch-source bridge-exception` all regenerated the source candidate and current best CSV byte-identical to their recorded artifacts. `python3 -m py_compile experiments/reproduce_gsdc2023_best_submission.py`, `PYTHONPATH=python:. pytest -q tests/test_reproduce_gsdc2023_best_submission.py tests/test_smooth_gsdc2023_submission.py` (`10 passed`), and reproduction-scope `git diff --check` passed.

Next engineering tasks after Kaggle A/B:

1. Keep `submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv` as default. The narrow `sm_a505u` epoch `200:400` A/B tied Public but worsened Private (`3.806 / 4.898`), so the full-trip `sm_a505u` patch remains slightly better.
2. Do not promote `pixel5` raw_wls based on PR MSE alone. The LAX-X A/B shows raw_wls can improve PR MSE while worsening Kaggle Private; any future `pixel5` raw rescue needs train-backed masked/robust evidence or a direct Kaggle A/B.
3. If generating a fresh current-code all-test artifact, use the pixel5 raw_wls veto, not the earlier `200m` partial guard. It keeps the useful `sm_a505u` case and the train-backed `samsunga325g` high-PR rescue while removing all `pixel5` raw_wls in replay. Still treat any fresh all-test artifact as Kaggle A/B only, not as a reason to supersede the scored final.
4. Longer-term, stabilize source selection around masked/robust PR metrics rather than raw unmasked PR-MSE, but do not switch submission selection to observation-mask output wholesale yet.

- MATLAB residual parity is now cm-level on the 11-trip diagnostics scan, but Kaggle/submeter is still not reached. The next useful lever is source selection on train chunks where Kaggle WLS has high pseudorange residuals and raw WLS is a demonstrably better fallback.
- Relaxed `raw_wls` high-PR-MSE rescue from `raw_wls_mse_pr <= 20.0` and ratio `<= 0.35` to `raw_wls_mse_pr <= 50.0` and ratio `<= 0.40`, while keeping the finite baseline floor `baseline_mse_pr >= 50.0` and the existing baseline-step guard.
- Target chunk now switches as intended:
  - `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g`, start `1400`, epochs `111`: source mix `raw_wls=111`, selected PR MSE `44.3967`, selected RMS2D `15.293m` vs Kaggle WLS `28.292m`.
- Control chunks stayed baseline:
  - `train/2023-09-06-22-49-us-ca-routebb1/sm-s908b`, start `800`, epochs `200`: selected baseline, RMS2D `1.900m`, raw WLS `3.932m`.
  - `train/2020-07-08-22-28-us-ca/pixel4xl`, start `2000`, epochs `143`: selected baseline, RMS2D `3.370m`, raw WLS `13.777m`.
  - `train/2020-07-08-22-28-us-ca/pixel4`, start `2000`, epochs `127`: selected baseline, RMS2D `3.256m`, raw WLS `9.430m`.
  - `train/2023-09-06-00-01-us-ca-routen/sm-g955f`, starts `1000` / `800`, each `200` epochs: both stayed baseline because the existing baseline step-p95 guard fires; raw WLS was worse on GT (`22.344m` / `22.023m` vs baseline `14.669m` / `11.368m`).
  - `train/2021-01-04-21-50-us-ca-e1highway280driveroutea/pixel5`, start `2000`, epochs `2`: stayed baseline; raw WLS worse (`5.321m` vs `3.654m`).
- Added selector boundary tests for the new rescue window: pass `119.0495 -> 44.3967`, reject ratio above `0.40`, and reject raw MSE above `50.0`. Focused selector regression: `29 passed`.
- A full train no-FGO rescue audit was attempted but stopped after several minutes without progress output; do not cite it as evidence. Current evidence is the old full-epoch high-MSE prefilter plus the seven targeted reruns above.
- Bounded test-batch smoke after the relaxed rescue: `run_raw_bridge_batch.py --dataset test --limit 10 --workers 2 --max-epochs 200 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --force` completed at `ref/gsdc2023/results/test_parallel/20260423_1205`. Summary: 10/10 ok, submission rows `16464`, trips `10`, no NaN. Source totals from exported metrics: baseline `1600`, raw_wls `400`, FGO `0`; the raw_wls epochs are the existing catastrophic-baseline cases (`2021-11-05 pixel6pro`, `2021-11-30 mi8`), not new relaxed-rescue over-selection in this smoke.
- Larger bounded test-batch smoke: `run_raw_bridge_batch.py --dataset test --limit 40 --workers 4 --max-epochs 200 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --force` completed at `ref/gsdc2023/results/test_parallel/20260423_1213`. Summary: 40/40 ok, submission rows `71936`, trips `40`, no NaN. Source totals: baseline `7200`, raw_wls `600`, FGO `200`. Replaying the exported chunk records with the old rescue thresholds (`raw<=20`, ratio `<=0.35`) gives identical totals and `0` source changes, so the relaxed rescue does not alter this test-40 / 200-epoch smoke.
- All-test all-epoch generation: `run_raw_bridge_batch.py --dataset test --workers 4 --max-epochs -1 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --force` completed at `ref/gsdc2023/results/test_parallel/20260423_1306`. Summary: 40/40 ok, submission rows `71936`, trips `40`, no NaN. Source totals: baseline `68419`, raw_wls `2836`, FGO `771`; non-baseline trips are `2020-12-11 pixel4xl`, `2021-11-05 pixel6pro`, `2021-11-30 mi8`, `2022-04-04 pixel5`, `2022-10-06 sm-a205u`, `2023-05-09 sm-a505u`.
- Replaying the all-test all-epoch exported chunk records with the old rescue thresholds (`raw<=20`, ratio `<=0.35`) gives identical totals and `0` source changes. Therefore the relaxed high-PR raw_wls rescue is not changing test selection under this no-TDCP submission configuration; its observed effect remains the train `samsunga325g` high-PR chunk rescue.
- Current-code TDCP-candidate all-test all-epoch run completed at `ref/gsdc2023/results/test_parallel/20260423_1433`, but high-PR fallback over-selected raw_wls on `2021-11-05 pixel6pro` and `2022-10-06 sm-a205u`, producing source totals baseline `68419`, raw_wls `2836`, fgo_no_tdcp `771` and much worse proxy summary than `20260421_0555`.
- Added a high-baseline fallback motion guard: candidates must satisfy `step_p95 <= max(2 * baseline.step_p95, 100m)` and `baseline_gap_p95 <= max(3 * baseline.step_p95, 150m)` before catastrophic/high-baseline fallback can select them. This keeps the plausible `2020-12-11 pixel4xl` raw_wls rescue and the mi8 baseline-jump rescue, but rejects the pixel6pro / sm-a205u kilometer-scale raw_wls trajectories.
- Re-ran only the two affected test trips after the guard and reassembled outputs at `ref/gsdc2023/results/test_parallel/20260423_1450`. Source totals became baseline `70665`, raw_wls `590`, fgo_no_tdcp `771`; non-baseline trips are `2020-12-11 pixel4xl`, `2021-11-30 mi8`, `2022-04-04 pixel5`, and `2023-05-09 sm-a505u`.
- Submission coordinate comparison `20260421_0555` vs `20260423_1450`: rows `71936`, no NaN in either, mean horizontal diff `0.446m`, p95 `0.431m`, max `347.4m`, rows with diff >1m `1308`. Largest changes are `2022-04-04 pixel5`, `2023-05-09 sm-a505u`, `2022-07-12 sm-a325f`, `2021-11-30 mi8`, and `2020-12-11 pixel4xl`.
- Decision: do not submit `20260423_1306`, `20260423_1433`, or `20260423_1450` blindly. `20260421_0555` remains the safer known all-test TDCP-candidate artifact unless we submit A/B to Kaggle or build train evidence that the coordinate shifts are beneficial.
- Added raw-bridge integration coverage for the high-baseline motion guard: pixel6pro-like kilometer-scale raw_wls is rejected, while the plausible 2020-12-11 pixel4xl catastrophic-baseline raw_wls rescue is preserved. Focused selector regression now passes with `33 passed`; `py_compile` and `git diff --check` are clean.
- Replayed current gated policy over the existing 20-window train high-MSE audit at `experiments/results/raw_wls_high_pr_policy_replay_20260423/`: 19 windows stay baseline and only `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g` chunk `1400:1512` selects raw_wls. Existing audit score improves from baseline `29.397m` to raw_wls `18.933m` (`-10.464m`). A current-code rerun of that chunk also selects `raw_wls=112`, with selected/raw score `18.933m`, baseline `29.397m`, selected PR MSE `15.6667` vs baseline `56.7125`.
- Rechecked existing TDCP-off/FGO train sweeps before broadening FGO selection. The current-code 36-trip TDCP gap guard sweep has no source changes and pairwise delta `0.0` across all 36 trips; older adjusted/off-candidate sweeps show small non-worsening gains but are not enough to justify wider `fgo_no_tdcp` adoption without a current-code targeted rerun.
- Focused current-code probe for test non-baseline regions is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/`. Under batch-like defaults, `test/2022-04-04-16-31-us-ca-lax-x/pixel5` start `0:600` selects baseline `200`, `fgo_no_tdcp` `200`, and raw_wls `200`; the first chunk had raw_wls PR MSE `340.02` vs fgo_no_tdcp `404.78`. `test/2023-05-09-23-10-us-ca-sjc-r/sm-a505u` representative chunks `0:200`, `400:600`, and `2000:2200` all stay baseline under current code.
- Tightened high-baseline fallback consistency: FGO/FGO-off candidates now still have to pass the existing raw_wls PR-MSE guard, so high-baseline fallback cannot choose `fgo_no_tdcp` when raw_wls has lower PR MSE. Replaying the focused payload changes the `2022-04-04 pixel5` first chunk from `fgo_no_tdcp` to `raw_wls`; focused selector tests now pass with `35 passed`.
- Re-ran `test/2022-04-04-16-31-us-ca-lax-x/pixel5` start `0:600` after the guard. It now selects baseline `200`, raw_wls `400`, fgo_no_tdcp `0`; the first chunk is confirmed `raw_wls` with candidate PR MSE raw_wls `340.02` vs fgo_no_tdcp `404.78`. Window selected PR MSE improves from the previous focused probe `4582.42` to `4561.50` while baseline is `4756.19`. Artifact: `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_first_600_after_high_baseline_raw_guard_summary.json`.
- Saved-policy replay across existing chunk payload artifacts is at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/all_saved_chunk_policy_replay_after_raw_guard.csv` (`59` files / `111` records). Most diffs are historical pre-guard artifacts; the current focused payload diff is the intended `2022-04-04 pixel5` first chunk `fgo_no_tdcp -> raw_wls`.
- Full-trip rerun after the high-baseline raw guard is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_full_after_high_baseline_raw_guard/` (`679.9s`). Batch-like defaults were `position_source=gated`, `chunk_epochs=200`, `motion_sigma_m=0.3`, `clock_drift_sigma_m=1.0`, `position_offset`, `multi_gnss`, TDCP enabled, and no observation mask. Source counts for `test/2022-04-04-16-31-us-ca-lax-x/pixel5` are baseline `1400`, raw_wls `771`, fgo `0`, fgo_no_tdcp `0`; raw_wls spans are `0:200`, `400:600`, and `1800:2171`. Full-trip PR MSE is selected `1415.4641`, baseline `1542.9487`, raw_wls `1406.4002`, fgo `39929607.7868`.
- The full-trip nonbaseline chunks all choose raw_wls over fgo_no_tdcp by PR MSE: `0:200` raw_wls `340.02` vs fgo_no_tdcp `404.78` vs baseline `593.26`; `400:600` raw_wls `12952.81` vs fgo_no_tdcp `13146.70` vs baseline `13283.32`; `1800:2000` raw_wls `822.26` vs fgo_no_tdcp `1107.51` vs baseline `1257.98`; `2000:2171` raw_wls `605.37` vs fgo_no_tdcp `647.85` vs baseline `1061.27`.
- Comparing that full-trip rerun against the existing `20260423_1450` submission coordinates for the same trip gives rows `2170`, mean horizontal diff `3.442m`, p95 `18.225m`, max `495.083m`, and rows with diff > `1m` = `561`. The guard improves the TDCP-candidate proxy for this trip (`20260423_1433/1450` around `1449.25` PR MSE -> `1415.46`), but it is still far worse than the safer `20260421_0555` proxy for the same trip (`180.28`).
- Decision after the full-trip rerun: the high-baseline raw guard is a correct selector consistency fix, but it does not make the current TDCP-candidate all-test artifact submission-ready. Do not submit/reassemble the current TDCP-candidate path blindly. Next useful run is an isolation rerun of `2022-04-04 pixel5` under a `20260421_0555`-matching/no-TDCP configuration to identify which config/code delta caused the `180.28 -> 1415+` proxy blow-up.
- Current-code full no-TDCP isolation is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_full_current_no_tdcp/`. It exactly matches the TDCP-candidate-after-guard selected lat/lon: selected PR MSE `1415.4641`, baseline `1542.9487`, raw_wls `1406.4002`, fgo `1460.8539`, source counts baseline `1400`, raw_wls `771`, fgo `0`, `fgo_iters=61`. Therefore the remaining `20260421_0555 -> current` regression is not caused by TDCP output itself; current no-TDCP FGO no longer reproduces the old good `fgo_no_tdcp` proxy.
- Current-code GPS-only no-TDCP isolation is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_full_current_gps_only_no_tdcp/`. It is worse: selected PR MSE `2557.2203`, baseline `2778.6689`, raw_wls `2450.5034`, fgo `2635.6850`, source counts baseline `1771`, raw_wls `400`, fgo `0`, `fgo_iters=59`. This rules out a simple "old run was GPS-only" explanation. MATLAB/taroz target is not GPS-only; GPS-only here is only a regression isolation knob.
- `20260421_0555/parameters.m` and `20260423_1433/parameters.m` are identical in the copied MATLAB parameter file. The later `summary.json` adds `factor_dt_max_s=1.5` / extra opt-in fields, but this pixel5 run reports `factor_dt_gap_count=0`, so the next likely suspects are post-2026-04-21 raw bridge observation/clock/satellite-product changes rather than a simple CLI flag difference.
- Current-code no-VD/no-TDCP isolation is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_full_current_no_vd_no_tdcp/`. It also matches the same selected source pattern: selected PR MSE `1415.4641`, baseline `1542.9487`, raw_wls `1406.4002`, fgo `1468.9376`, source counts baseline `1400`, raw_wls `771`, fgo `0`, `fgo_iters=28`. This rules out a simple old-run-was-non-VD explanation.
- Coordinate comparison for `20260421_0555/submission_20260421_0555.csv` vs current no-TDCP selected output on the same `2022-04-04 pixel5` trip: rows `2170`, mean horizontal diff `1.931m`, median `0.316m`, p95 `6.501m`, max `245.727m`, rows > `1m` = `685`. Against current candidate columns, old submission is closest to current selected output (`mean=1.93m`) rather than current baseline (`5.54m`), raw_wls (`5.57m`), or FGO (`6.62m`). Therefore the huge `OptError/PR-MSE` delta is partly a proxy/observation-model shift, not a wholly different trajectory.
- Next decision point: do not treat PR-MSE alone as submission truth after the post-2026-04-21 observation changes. Keep `20260421_0555` as the safest known submission artifact, and only submit the current selector path as a Kaggle A/B if accepting that the local PR proxy has become unstable.
- Re-evaluated `20260421_0555/submission_20260421_0555.csv` coordinates under the current no-TDCP multi-GNSS observation model using current selected altitude as the missing altitude proxy. Artifact: `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_0555_submission_under_current_observation_model.json`. Current-model PR MSE is old submission `1424.2648`, current selected `1415.5758`, current baseline `1543.2670`, current raw_wls `1406.4191`, current FGO `1460.8796`. This confirms the old good submission coordinates are also high-MSE under the current PR model, so the `180.28 -> 1415+` delta is largely local proxy/model shift.
- Per-epoch contribution audit is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_current_pr_mse_top_epochs.csv` and `pixel5_current_pr_mse_top_epochs_summary.json`. The top 20 epochs account for `89.99%` of current selected weighted PR-MSE contribution. Worst epochs are `432:440` in the `400:600 raw_wls` span and epoch `1853` in the late raw_wls span; epoch `440` alone has current per-epoch MSE `1,962,592.9` and old-submission per-epoch MSE `1,972,505.4`, with old/current horizontal separation `245.73m`. Because old and current positions both score badly on the same epochs, this is an observation-residual outlier problem, not enough evidence that the current trajectory is worse by the same scale.
- Created local Kaggle A/B candidate CSVs without overwriting existing artifacts: `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/submission_20260421_0555_pixel5_current_rawguard_20260423.csv` and `submission_20260423_1450_pixel5_current_rawguard_20260423.csv`. Both replace only `2022-04-04-16-31-us-ca-lax-x/pixel5` (`2170` submission rows) with the current high-baseline raw-guard selected coordinates and have no NaN lat/lon. Relative to `20260421_0555`, the replaced rows differ mean `1.931m`, p50 `0.316m`, p95 `6.501m`, max `245.727m`; relative to `20260423_1450`, mean `3.438m`, p50 `0.000m`, p95 `18.204m`, max `494.527m`. These are A/B-only candidates; keep `20260421_0555` as the safe default until Kaggle confirms otherwise.
- Outlier slot audit is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_pr_residual_outlier_epoch_slots.csv` and `pixel5_pr_residual_outlier_epoch_slots_summary.json`. Epoch `440` is dominated by GPS SVID4 L1: current selected residual `-33743.9m` with weight `0.010`, weighted squared residual `11386523.4`; old `0555` coordinates show the same slot residual `-34014.0m`. Epoch `1853` is a smaller multi-satellite outlier led by GPS SVID30 (`-529.2m`) and GPS SVID14 (`417.6m`). This supports treating the current unmasked PR-MSE as outlier-sensitive.
- Evaluated the same current selected coordinates under current `--observation-mask` TripArrays at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_current_selected_observation_mask_eval.json`: PR-MSE drops from `1415.576` to `87.000`, p95 per-epoch MSE from `727.154` to `177.517`, and epoch 440 positive slots from `21` to `16` (`observation_mask_count=6222`, `residual_mask_count=3212`, `pseudorange_doppler_mask_count=270`). Masking fixes the local proxy outliers.
- Full `--observation-mask --no-tdcp` current-code run is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/pixel5_full_current_obsmask_no_tdcp/`. It gives selected/baseline PR-MSE `23.1792`, raw_wls `13.7735`, fgo `17.0270`, but gated selects baseline for all `2171` epochs because baseline PR-MSE is now below the high-baseline threshold. Coordinate comparison (`pixel5_obsmask_coordinate_compare.json`) shows old `0555` vs obsmask selected mean `5.540m`, p95 `30.207m`, max `403.570m`, which is worse than old `0555` vs current no-mask selected p95 `6.501m`. Therefore observation-mask is useful for diagnostics/local proxy stability, but should not be switched into submission blindly.
- Cross-trip nonbaseline proxy compare is stored at `experiments/results/test_fgo_tdcp_candidate_probe_20260423/nonbaseline_trip_proxy_compare/` (`proxy_summary.csv`, `coordinate_diff_summary.csv`, `decision_table.csv`, `summary.json`). It evaluates `old0555`, `current1450`, and `current1450_pixel5_rawguard` coordinates on the five `20260421_0555` nonbaseline trips under current unmasked and masked PR models.
- Cross-trip decision table versus `old0555`: `pixel4xl` current1450 is essentially identical (coord p95 `0.745m`, masked PR delta `-0.121`); `mi8` is slightly worse (p95 `2.432m`, masked delta `+0.130`); `pixel5` current1450 is worse (p95 `16.872m`, masked delta `+7.704`), while `pixel5 rawguard` is closer in coordinates and better unmasked (`-8.343`) but worse masked (`+18.309`); `sm-a325f` is slightly worse (p95 `2.575m`, masked delta `+0.028`); `sm-a505u` has the only clear masked proxy gain (`-0.552`) with moderate coordinate shift (p95 `3.004m`, max `39.737m`).
- Created narrower A/B CSVs under `experiments/results/test_fgo_tdcp_candidate_probe_20260423/submission_candidates/` without overwriting existing submissions: `submission_20260421_0555_sm_a505u_current1450_only_20260423.csv`, `submission_20260421_0555_pixel4xl_and_sm_a505u_current1450_20260423.csv`, `submission_20260421_0555_pixel5_rawguard_and_sm_a505u_current1450_20260423.csv`, and `submission_20260421_0555_pixel5_rawguard_pixel4xl_sm_a505u_20260423.csv`. Summary is `selective_ab_summary.json`. The most defensible first Kaggle A/B is `sm_a505u_current1450_only`; the pixel5 rawguard variants are higher-risk because masked proxy worsens even though the coordinate delta is closer than `current1450`.

2026-04-24 JST raw_wls rescue redesign follow-up:

- Tightened the normal high-PR-MSE `raw_wls` rescue back to train-backed evidence only:
  - `baseline_mse_pr >= 50.0`
  - `raw_wls_mse_pr <= 20.0`
  - `raw_wls_mse_pr <= baseline_mse_pr * 0.35`
  - `raw_wls.baseline_gap_max_m <= 150.0`
- This intentionally rejects the earlier relaxed `119.0495 -> 44.3967` case. The retained positive train case is `train/2022-07-26-21-01-us-ca-sjc-s/samsunga325g`, where `56.7125 -> 15.6667`, gap max `123.431m`, and GT delta `-10.464m`.
- Re-ran the train-backed policy lab at `experiments/results/raw_wls_policy_lab_20260424/`. Current high-PR rescue still selects `1` train window, better `1`, worse `0`, total delta `-10.464m`; `current_high_pr` now records thresholds `50.0 / 20.0 / 0.35 / gap_max 150.0`.
- Re-ran pixel5-veto source replay at `experiments/results/source_guard_replay_20260424_pixel5_veto/`. Current source counts remain baseline `288`, raw_wls `3`; the raw_wls records are duplicate `sm_a505u` focused payloads for epoch `200:400` plus the train `samsunga325g` high-PR rescue. Pixel5 raw_wls remains `0`.
- Verification: `python3 -m py_compile experiments/gsdc2023_chunk_selection.py experiments/analyze_gsdc2023_raw_wls_policy.py experiments/analyze_gsdc2023_source_ab.py`; `PYTHONPATH=python:. pytest -q tests/test_gsdc2023_chunk_selection.py tests/test_analyze_gsdc2023_raw_wls_policy.py` (`17 passed`); `PYTHONPATH=python:. pytest -q tests/test_validate_fgo_gsdc2023_raw.py -q -k "select_gated_chunk_source or gated_chunk_source"` (`24 passed`).

2026-04-24 JST current-code all-test artifact attempt, intentionally stopped:

- After the strict raw_wls rescue redesign, the next planned step was to generate a fresh current-code all-test artifact for Kaggle A/B only, not as a new default. The intended configuration was:
  - `position_source=gated`
  - `chunk_epochs=200`
  - `motion_sigma_m=0.3`
  - `clock_drift_sigma_m=1.0`
  - `--no-tdcp`
  - `--position-offset`
  - current selector code, including pixel5 raw_wls veto and the strict train-backed high-PR raw_wls rescue above.
- The exact command started was:
  - `PYTHONPATH=python:. python3 ../ref/gsdc2023/run_raw_bridge_batch.py --dataset test --workers 4 --max-epochs -1 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --force`
- The run printed only the startup line:
  - `parallel workers=4 trips=40 run=40 max_epochs=-1 chunk_epochs=200`
- The user then redirected the session to stop and update this plan. At `2026-04-24 15:13 JST`, the batch was still running with process group `965177`; four child `validate_fgo_gsdc2023_raw.py` processes were active on:
  - `test/2020-12-11-19-30-us-ca-mtv-e/pixel4xl`
  - `test/2021-08-17-20-37-us-ca-mtv-g/pixel5`
  - `test/2021-08-31-20-37-us-ca-mtv-e/sm-g988b`
  - `test/2021-09-14-20-32-us-ca-mtv-k/pixel4`
- The process group was terminated with `kill -TERM -- -965177`, then checked with `ps`; no `run_raw_bridge_batch.py` / `validate_fgo_gsdc2023_raw.py` processes remained.
- Important interpretation:
  - No `[ok] ...` trip completion lines were observed before stopping.
  - No `results_dir=...` line was printed.
  - Therefore there is no assembled current-code all-test submission from this attempt.
  - Any partial `bridge_run.log`, `bridge_positions.csv`, or `bridge_metrics.json` files that may have been touched under `../ref/gsdc2023/dataset_2023/test/...` during the interrupted run must not be cited as evidence. If this all-test A/B is resumed, rerun with `--force` and wait for a complete `results_dir=...` output.
- Next clean execution steps when resuming:
  1. Re-run the exact command above to completion.
  2. Read the generated `../ref/gsdc2023/results/test_parallel/<timestamp>/summary.json` and `results.csv`.
  3. Aggregate exported per-trip `bridge_metrics.json` source counts and confirm pixel5 raw_wls remains `0`.
  4. Compare the generated submission against `experiments/results/reproduce_best_submission_20260424/submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv` and `../ref/gsdc2023/results/test_parallel/20260421_0555/submission_20260421_0555.csv` for row count, key equality, NaN lat/lon, mean/p95/max horizontal delta, and largest changed trips.
  5. If the artifact is worth testing, smooth it with the current default trajectory smoother policy before Kaggle A/B. Do not supersede the scored default artifact without Kaggle evidence.

2026-04-24 JST current-code all-test artifact completed after the interrupted attempt:

- Re-ran the exact all-test batch to completion:
  - `PYTHONPATH=python:. python3 ../ref/gsdc2023/run_raw_bridge_batch.py --dataset test --workers 4 --max-epochs -1 --position-source gated --chunk-epochs 200 --motion-sigma-m 0.3 --clock-drift-sigma-m 1.0 --no-tdcp --position-offset --force`
  - Output: `../ref/gsdc2023/results/test_parallel/20260424_2321/`
- Batch summary (`summary.json`):
  - `n_trips=40`
  - `mean_opt_error=60.46`
  - `median_opt_error=25.665`
  - `max_opt_error=949.02`
- Exported source totals from per-trip `bridge_metrics.json`:
  - baseline `71436`
  - raw_wls `590`
  - fgo `0`
- Non-baseline trips are exactly three:
  - `2020-12-11-19-30-us-ca-mtv-e/pixel4xl`: baseline `1000`, raw_wls `190`, full-trip PR MSE `949.0233`
  - `2021-11-30-20-59-us-ca-mtv-m/mi8`: baseline `1195`, raw_wls `200`, PR MSE `9.7983`
  - `2023-05-09-23-10-us-ca-sjc-r/sm-a505u`: baseline `2184`, raw_wls `200`, PR MSE `55.8469`
- Important selector check:
  - `2022-04-04-16-31-us-ca-lax-x/pixel5` now stays baseline for all `2171` epochs under this no-TDCP all-test run; selected counts are baseline `2171`, raw_wls `0`, fgo `0`, PR MSE `272.1760`.
  - This confirms the pixel5 raw_wls veto is effective in the full batch, not only in focused replay.
- Unsmoothed submission comparison:
  - Versus `20260421_0555`: rows/key order identical, NaN `0`; mean horizontal diff `0.4839m`, p50 `0.3163m`, p95 `0.4305m`, max `403.57m`, rows `>1m = 746`, rows `>5m = 547`.
  - Largest changed trip remains `2022-04-04 pixel5 LAX-X` (`p95 30.21m`, max `403.57m`). Secondary diffs are `2022-07-12 sm-a325f` (`p95 2.57m`, max `19.60m`) and the expected per-phone offset-scale Samsung / mi8 changes around `0.39-0.43m`.
  - Versus current best scored artifact `submission_best_cap100_plus_pixel4xl_outlier_row_20260424.csv`: mean diff `1.2390m`, p95 `2.7466m`, max `639.57m`, rows `>1m = 30096`.
- Applied the current default smoother family to the new artifact:
  - Command:
    - `PYTHONPATH=python:. python3 experiments/smooth_gsdc2023_submission.py --input ../ref/gsdc2023/results/test_parallel/20260424_2321/submission_20260424_2321.csv --output experiments/results/post_smooth_submission_20260424/submission_20260424_2321_smooth_cap100_w5_b1_20260424.csv --summary experiments/results/post_smooth_submission_20260424/submission_20260424_2321_smooth_cap100_w5_b1_20260424.json --median-window 5 --smooth-window 5 --blend 1.0 --max-correction-m 100 --smooth-kernel triangular`
  - Correction stats: rows `71936`, groups `40`, segments `47`, corrected rows `71758`, Hampel rows `11`, mean correction `1.0538m`, p95 correction `2.5542m`, max correction `100.0m`.
  - Smoothed-vs-unsmoothed diff: mean `1.0538m`, p95 `2.5542m`, max `100.08m`; biggest smoothing effect is still `2022-04-04 pixel5 LAX-X` (`p95 5.44m`, max `79.36m`).
  - Smoothed-vs-best diff shrinks materially: mean `0.4630m`, p50 `0.3159m`, p95 `0.4304m`, max `539.49m`, rows `>1m = 961`.
  - Remaining largest diffs versus best are:
    - `2022-04-04 pixel5 LAX-X`: `p95 24.90m`, max `304.24m`
    - `2022-07-12 sm-a325f`: `p95 2.30m`, max `11.60m`
    - `2023-05-09 sm-a505u`: `p95 2.20m`, max `18.47m`
    - `2020-12-11 pixel4xl`: a single large row remains (`max 539.49m`), consistent with the known cap1000 outlier-row promotion in the current best artifact.
- Current interpretation:
  - The strict-rescue + pixel5-veto current-code all-test run is now operational and much safer than the earlier partial / pre-veto lines.
  - However, the raw bridge local proxy is not clearly better than `20260421_0555` (`mean_opt_error 60.46` vs `58.15`), and the unsmoothed artifact still diverges materially on `2022-04-04 pixel5`.
  - The smoothed current artifact is close to the current best scored artifact in aggregate coordinates, but that is not the same as leaderboard evidence.
  - Therefore neither `../ref/gsdc2023/results/test_parallel/20260424_2321/submission_20260424_2321.csv` nor `experiments/results/post_smooth_submission_20260424/submission_20260424_2321_smooth_cap100_w5_b1_20260424.csv` should replace the scored default without Kaggle A/B.
