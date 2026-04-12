# gnss_gpu 引き継ぎメモ

**最終更新**: 2026-04-11 JST
**現在の HEAD**: `4923ee2` (`feature/carrier-phase-imu`)
**作業ツリー**: **dirty**。このファイルを含めて未コミット差分あり。
**現フェーズ**: `DD pseudorange` / `DD carrier` / quality gate / support-skip / undiff fallback / carrier anchor / preset 化 / refactor までは実装済み。
**今の主課題**: frozen preset `Odaiba SMTH P50=1.38m / RMS=5.04m` を regression test で守りつつ、精度改善を続ける。
**FGO**: 使わない。PF の枠内で詰める。
**PR #4**: 許可があるまで merge しない。

---

## 0. まず最初に読む順

Claude に渡すときは、最初にここだけ読めば十分という順番を固定しておく。

1. 本ファイルの **§1 現在の要約** と **§8 次にやるべきこと**
2. `internal_docs/pf_smoother_api.md`
3. `experiments/exp_pf_smoother_eval.py`
4. `experiments/run_pf_smoother_odaiba_reference.sh`
5. `tests/test_exp_pf_smoother_eval.py`
6. `internal_docs/interfaces.md` の `PF Smoother 実験 API` 節

この順に読めば、

- 何がもう実装済みか
- 今の frozen reference は何か
- どこがまだ不安定か
- 次に何を固定すべきか

が分かる。

---

## 1. 現在の要約

### 1.1 このメモの一番重要な結論

2026-04-08 版の「次は `DD pseudorange` を実装する」という結論は **もう古い**。

今は次の状態まで進んでいる。

- `DD pseudorange` は **実装済み**
- `DD carrier AFV` の multi-GNSS 化と smoother replay も **実装済み**
- fixed gate / adaptive gate / ESS gate / spread gate / support-skip / dynamic sigma まで **一通り実装して試した**
- `weak-DD epoch` 向けの `undiff fallback`、`carrier anchor`、`TDCP continuity propagation`、`tracked / hybrid fallback` まで **実装して試した**
- 実験ハブ `exp_pf_smoother_eval.py` はかなり肥大化したので、helper 分割と preset 化まで **実施済み**

つまり、今の引き継ぎ先がやるべきことは「DD pseudorange を新規実装すること」ではなく、

1. 既存の rescue stack を正しく理解する
2. frozen reference を再現可能に保つ
3. `5.54m` の historical best と現在 preset rerun の差を潰す

の 3 つ。

### 1.2 headline numbers

#### 歴史的ベストと現在の frozen rerun

| ラベル | 条件 | Odaiba SMTH P50 | Odaiba SMTH RMS | Shinjuku SMTH P50 | Shinjuku SMTH RMS | 状態 |
|---|---|---:|---:|---:|---:|---|
| **current frozen reference** | `--preset odaiba_reference` (sigma-pos=1.2, PU=1.9, dd-sigma=0.20, fallback-min-sats=4) | **1.38m** | **5.04m** | **2.55m** | **9.53m** | 2026-04-12 freeze 済み。再現確認済 |
| previous frozen (sigma-pos=1.2, PU=1.5) | sigma-pos=1.2, dd-sigma=0.20, fallback-min-sats=4 | 1.38m | 5.13m | 2.62m | 9.79m | 旧 preset |
| previous frozen (sigma-pos=2.0) | dd-sigma=0.20, fallback-min-sats=4, sigma-pos=2.0 | 1.51m | 5.48m | — | — | 旧 preset |
| old pre-DD baseline | 2026-04-08 の `ALL-IN + smoother` | 1.65m | 6.24m | 3.01m | 12.82m | 旧 baseline |

#### 解釈

- **headline best** は Odaiba `5.04m` / Shinjuku `9.53m` (frozen preset で再現可能)
- `sigma-pos` 2.0→1.2 が最大の改善源 (predict noise の削減で particle collapse を抑制)
- `position-update-sigma` 1.5→1.9 で追加改善 (urban canyon での bad SPP の影響を緩和)
- 両 dataset で一貫した改善: overfitting ではない

### 1.3 何が効いて、何が効いていないか

効いたもの:

- `DD pseudorange`
- `DD carrier` の quality gate
- `spread tighten-only` 的な締め方
- `support-skip`
- `undiff fallback`
- `sigma-pos` 削減 (2.0→1.2): particle collapse 抑制で tail error を大幅削減

少しは効いたが主戦場ではなかったもの:

- `carrier anchor`
- `TDCP/LOS continuity propagation`
- `tracked fallback`

ほぼ頭打ちだったもの:

- 閾値の細かい sweep だけ
- dynamic sigma の微調整だけ
- tracked row を強く boost する方向

---

## 2. 現在の workspace 状態

### 2.1 worktree はクリーンではない

今の `git status --short` は次の通り。

```text
 M experiments/exp_pf_smoother_eval.py
 M include/gnss_gpu/pf_device.h
 M internal_docs/interfaces.md
 M internal_docs/plan.md
 M python/gnss_gpu/_pf_device_bindings.cpp
 M python/gnss_gpu/dd_carrier.py
 M python/gnss_gpu/particle_filter_device.py
 M src/particle_filter/pf_device.cu
 M tests/test_cuda_streams.py
?? experiments/run_pf_smoother_odaiba_reference.sh
?? internal_docs/pf_smoother_api.md
?? python/gnss_gpu/dd_pseudorange.py
?? python/gnss_gpu/dd_quality.py
?? tests/test_dd_carrier.py
?? tests/test_dd_pseudorange.py
?? tests/test_dd_quality.py
?? tests/test_exp_pf_smoother_eval.py
```

重要なのは、

- 「この repo は clean」という古い前提は **誤り**
- Claude は `git status` を見て「壊れている」と誤解しやすいが、実際は会話中の実装差分が積み上がっているだけ
- 既存差分を戻してはいけない

### 2.2 この worktree で主要に触っているファイル

| ファイル | 役割 |
|---|---|
| `experiments/exp_pf_smoother_eval.py` | rescue stack の統合実験ハブ。今の中心 |
| `python/gnss_gpu/dd_pseudorange.py` | DD pseudorange 計算 |
| `python/gnss_gpu/dd_carrier.py` | DD carrier AFV 計算 |
| `python/gnss_gpu/dd_quality.py` | DD quality gate / scale helper |
| `python/gnss_gpu/particle_filter_device.py` | DD update / smoother replay / spread stat |
| `src/particle_filter/pf_device.cu` | DD pseudorange / DD carrier / spread kernels |
| `tests/test_exp_pf_smoother_eval.py` | preset / helper / dataset smoke regression |
| `internal_docs/pf_smoother_api.md` | API と状態遷移の source of truth |
| `experiments/run_pf_smoother_odaiba_reference.sh` | frozen reference 実行入口 |

### 2.3 注意点

- `experiments/results/pf_smoother_eval.csv` は full-run で上書きされる。会話中は毎回バックアップして復元している。
- `internal_docs/plan.md` の 2026-04-08 版に書いてあった「作業ツリーは clean」は今は明確に誤り。
- `libgnsspp` の import は pure Python wrapper より build binding を優先しないと smoke regression が不安定になる。`exp_pf_smoother_eval.py` 冒頭の `sys.path` 順修正はこのため。

---

## 3. ここまでに実装済みのもの

この節は「何をもう一度実装し直す必要がないか」を明示するために書く。

### 3.1 DD pseudorange

実装済みファイル:

- `python/gnss_gpu/dd_pseudorange.py`
- `src/particle_filter/pf_device.cu`
- `python/gnss_gpu/particle_filter_device.py`
- `tests/test_dd_pseudorange.py`

実装内容:

- rover/base raw RINEX を使った `DD pseudorange`
- constellation ごとの reference satellite 選択
- per-system / per-pair 構造
- duplicate row の整理
- optional な base interpolation
- GPU kernel での DD pseudorange weight update
- smoother forward/backward での replay

注意:

- 「DD pseudorange をこれから実装する」は古い TODO なのでやらない
- 改善の余地は gate / weight / dataset handling にあるのであって、配線そのものではない

### 3.2 DD carrier AFV

実装済みファイル:

- `python/gnss_gpu/dd_carrier.py`
- `src/particle_filter/pf_device.cu`
- `python/gnss_gpu/particle_filter_device.py`
- `tests/test_dd_carrier.py`

実装内容:

- multi-GNSS の per-system / per-pair DD carrier AFV
- smoother replay
- raw RINEX / optional base interpolation
- code 混在抑制のための system 共通 code path

制約:

- GLONASS FDMA wavelength はまだ扱っていないので skip する

### 3.3 DD quality gate

実装済みファイル:

- `python/gnss_gpu/dd_quality.py`
- `tests/test_dd_quality.py`

入っているもの:

- fixed threshold gate
- adaptive `median + k*MAD`
- ESS-based scaling
- spread-based scaling
- pair-count / metric scaling helper

これは「まだ仮の雑実装」ではなく、少なくとも sweep 可能な状態にある。

### 3.4 weak-DD rescue stack

今の `exp_pf_smoother_eval.py` は、DD が薄い epoch 向けに次の rescue 系を持っている。

1. `support-skip`
2. `undiff same-band carrier AFV fallback`
3. `carrier anchor` による pseudorange-like rescue
4. `TDCP continuity propagation`
5. `tracked fallback`
6. `hybrid tracked fallback`

ここまで全部試している。何も無いところから設計し直す必要はない。

### 3.5 preset 化と refactor

実装済み:

- `_CLI_PRESETS`
- `odaiba_reference`
- `_expand_cli_preset_argv`
- `_print_cli_presets`
- `_namespace_to_run_kwargs`
- `main(argv)`
- `run_pf_smoother_odaiba_reference.sh`

refactor 済み helper:

- `CarrierAnchorAttempt`
- `CarrierFallbackAttempt`
- `_attempt_carrier_anchor_pseudorange_update`
- `_attempt_dd_carrier_undiff_fallback`

つまり、今は「巨大な 1 ループを読むしかない」状態からは一歩進んでいる。

---

## 4. 2026-04-11 時点の code map

### 4.1 中心ファイル

#### `experiments/exp_pf_smoother_eval.py`

このファイルが今の実験ハブ。

重要な entry / helper:

- `_CLI_PRESETS`
  `odaiba_reference` の frozen flags を持つ
- `CarrierBiasState`
- `CarrierAnchorAttempt`
- `CarrierFallbackAttempt`
- `_attempt_carrier_anchor_pseudorange_update`
- `_attempt_dd_carrier_undiff_fallback`
- `run_pf_with_optional_smoother(...)`
- `build_arg_parser()`
- `main(argv=None)`

このファイルの役割:

- dataset load
- PF runtime orchestration
- DD pseudorange / DD carrier の gating と update
- weak-DD rescue
- smoother replay
- diagnostics dump
- preset 展開

#### `python/gnss_gpu/particle_filter_device.py`

役割:

- Python から CUDA kernel を呼ぶ low-level API
- DD pseudorange / DD carrier の update wrapper
- smoother の epoch store / replay
- spread statistic の取得

#### `src/particle_filter/pf_device.cu`

役割:

- DD pseudorange kernel
- per-pair reference を読む DD carrier kernel
- spread statistic kernel

### 4.2 ドキュメント

#### `internal_docs/pf_smoother_api.md`

ここが今の API / state-machine / call sequence の source of truth。

入っているもの:

- runtime API の引数表
- diagnostics の列定義
- `attempted / used / tracked-assisted` の定義
- `Epoch Flow`
- `Carrier Rescue Flow`
- `Tracker Lifecycle`
- `Call Sequence`

Claude に「全体像を短時間で掴ませる」にはこのファイルが一番効く。

#### `internal_docs/interfaces.md`

`PF Smoother 実験 API` の節から `pf_smoother_api.md` と frozen script へ飛べるようにしてある。

### 4.3 テスト

#### `tests/test_exp_pf_smoother_eval.py`

今の handoff で特に重要。

持っているもの:

- helper unit tests
- preset 展開 test
- late override test
- dataset-aware smoke regression

dataset-aware smoke regression の意図:

- Odaiba reference preset が最低限は動くこと
- `n_dd_used >= 1`
- `n_dd_pr_used >= 1`
- P50 が 0-20m の sane range にあること

この test は「best を保証する test」ではなく、「refactor で runtime が壊れていない」ことを保証する test。

---

## 5. 実験の流れと、どこまで詰めたか

ここは Claude が同じ sweep をやり直さないための節。

### 5.1 baseline から DD 系へ

2026-04-08 時点の旧 baseline は `ALL-IN + smoother` で、

- Odaiba: `P50=1.65m / RMS=6.24m`
- Shinjuku: `P50=3.01m / RMS=12.82m`

だった。

ここから、

- `SPP position_update` 依存がボトルネック
- undifferenced carrier AFV だけでは particle が carrier peak ambiguity を解けない
- したがって `DD pseudorange` が必要

という理解で進めていた。

その仮説自体は今も変わっていない。

### 5.2 DD pseudorange 導入後

`DD pseudorange` を raw RINEX ベースで入れて full-run した結果、

- `exact-only DD pseudorange`
- `sigma=0.5`

で

- `SMTH P50=1.48m / RMS=5.75m`

まで改善した。

この時点で、

- `DD pseudorange` は効く
- base interpolation は tail を悪化させやすい

という結論になった。

### 5.3 adaptive gate / ESS / spread

その後、

- adaptive gate
- ESS-linked gate
- spread-linked gate

を順に追加していった。

この流れで best は概ね次のように詰まった。

| 段階 | 代表結果 |
|---|---:|
| exact-only DD pseudorange | `1.48m / 5.75m` |
| adaptive DD pseudorange + adaptive DD carrier | `1.49m / 5.70m` |
| ESS-linked adaptive gate | `1.50m / 5.67m` |
| spread tighten-only | `1.49m / 5.63m` |
| carrier spread sweep best | `1.47m / 5.62m` |

重要なのは、

- gate の工夫は **確かに効いた**
- ただし `sub-meter` に押し切るほどではなかった

ということ。

### 5.4 support-skip

tail diagnostics を入れて見えたのは、

- worst epoch の多くで `DD pseudorange` が無い
- `DD carrier kept pairs <= 4`
- `ESS < 0.01`

という collapse パターンだった。

そこから `support-skip` を導入して、

- `ESS <= 0.01`
- `max_pairs <= 4`
- `raw AFV median >= 0.15 cycles`

のような低 support epoch では DD carrier update を切るようにした。

best はこの時点で、

- `SMTH P50=1.48m / RMS=5.61m`

まで改善した。

### 5.5 undiff fallback

さらに、

- DD carrier が薄い epoch で
- same-band undifferenced carrier AFV を fallback として使う

経路を入れたところ、

- `SMTH P50=1.49m / RMS=5.55m`

まで改善した。

ここはかなり重要で、現時点でも **最大の改善源のひとつ**。

### 5.6 carrier anchor / continuity / tracked 系

その後に試したもの:

- `carrier anchor`
- `windowed continuity`
- `TDCP/LOS motion prediction`
- `tracked fallback`
- `tracked hybrid fallback`
- tracked continuity-driven sigma

結果としては、

- 少し効く run はあった
- ただし大きく best を更新するほどではなかった
- tracked row を boost する方向は tail をむしろ悪化させやすかった

会話中に一度だけ、

- `SMTH P50=1.50m / RMS=5.54m`

を観測しているが、そこは **まだ完全に freeze できていない**。

---

## 6. 現在の frozen reference

### 6.1 入口

今の frozen reference は次の 2 通りで呼べる。

```bash
python3 experiments/exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --preset odaiba_reference
```

または

```bash
URBANNAV_DATA_ROOT=/tmp/UrbanNav-Tokyo bash experiments/run_pf_smoother_odaiba_reference.sh
```

追加引数は末尾に付くので、

```bash
bash experiments/run_pf_smoother_odaiba_reference.sh --max-epochs 10 --n-particles 5000
```

のように smoke もできる。

### 6.2 `odaiba_reference` preset の中身

2026-04-12 時点の frozen flags は概ね以下。

```text
--runs Odaiba
--n-particles 100000
--sigma-pos 1.2
--position-update-sigma 1.9
--predict-guide imu
--imu-tight-coupling
--residual-downweight
--pr-accel-downweight
--smoother
--dd-pseudorange
--dd-pseudorange-sigma 0.5
--dd-pseudorange-gate-adaptive-floor-m 4.0
--dd-pseudorange-gate-adaptive-mad-mult 3.0
--dd-pseudorange-gate-ess-min-scale 0.9
--dd-pseudorange-gate-ess-max-scale 1.1
--mupf-dd
--mupf-dd-sigma-cycles 0.20
--mupf-dd-base-interp
--mupf-dd-gate-adaptive-floor-cycles 0.25
--mupf-dd-gate-adaptive-mad-mult 3.0
--mupf-dd-gate-ess-min-scale 0.9
--mupf-dd-gate-ess-max-scale 1.1
--mupf-dd-gate-spread-min-scale 0.88
--mupf-dd-gate-spread-max-scale 1.0
--mupf-dd-gate-low-spread-m 3.0
--mupf-dd-gate-high-spread-m 8.0
--mupf-dd-skip-low-support-ess-ratio 0.01
--mupf-dd-skip-low-support-max-pairs 4
--mupf-dd-skip-low-support-min-raw-afv-median-cycles 0.15
--mupf-dd-fallback-undiff
--mupf-dd-fallback-sigma-cycles 0.10
--mupf-dd-fallback-min-sats 4
--carrier-anchor
--carrier-anchor-sigma-m 0.25
--carrier-anchor-max-residual-m 0.80
--carrier-anchor-max-continuity-residual-m 0.50
```

### 6.3 current frozen rerun の結果

2026-04-12 更新後 (sigma-pos=1.2, PU=1.9, dd-sigma=0.20, fallback-min-sats=4):

観測されたカウンタ:

- `[mupf_dd] DD-AFV used 11138/12252 epochs, skip 1114/12252`
- `[mupf_dd_support_skip] epochs=328`
- `[carrier_anchor] epochs=3`
- `[carrier_anchor_tdcp] propagated_rows=95`
- `[mupf_dd_fallback_undiff] epochs=831`
- `[dd_pseudorange] used 1215/12252 epochs, skip 11037/12252`

結果:

- `FWD P50=1.42m / RMS=5.74m`
- `SMTH P50=1.38m / RMS=5.04m`

再現確認済み (100K particles, 決定論的)。
Shinjuku でも確認: `SMTH P50=2.55m / RMS=9.53m` (旧 baseline 12.82m から -3.29m)。

### 6.4 どう読むべきか

現在の preset は headline best を frozen truth として保持している。

- regression reference として有用
- headline best Odaiba `5.04m` / Shinjuku `9.53m` を再現可能に保持
- 旧 pre-DD baseline `6.24m` / `12.82m` から大幅に改善

---

## 7. 検証とテスト

### 7.1 直近で通っているもの

#### preset / helper / dataset smoke

```bash
python3 -m py_compile experiments/exp_pf_smoother_eval.py tests/test_exp_pf_smoother_eval.py
python3 experiments/exp_pf_smoother_eval.py --list-presets
PYTHONPATH=python python3 -m pytest tests/test_exp_pf_smoother_eval.py -q
```

結果:

- `11 passed`

#### targeted CUDA smoke

```bash
PYTHONPATH=python python3 -m pytest tests/test_cuda_streams.py -k 'carrier_anchor_pseudorange_smoke or undiff_carrier_afv_smoke or smooth_with_dd_pseudorange_and_dd_carrier_smoke' -q
```

これは会話中に通している。

### 7.2 script smoke

```bash
bash experiments/run_pf_smoother_odaiba_reference.sh --max-epochs 3 --n-particles 5000
```

この 3 epoch smoke では、

- `[mupf_dd] DD-AFV used 3/3 epochs, skip 0/3`
- `[dd_pseudorange] used 1/3 epochs, skip 2/3`
- `FWD P50=0.75m RMS=0.78m`
- `SMTH P50=0.75m RMS=0.69m`

だった。

これは plumbing 確認用であって、headline result としては読まないこと。

### 7.3 smoke regression test の意味

`tests/test_exp_pf_smoother_eval.py` の dataset-aware smoke regression は、

- exact metric 再現 test ではない
- runtime が壊れていないことを見る test
- preset 展開と dataset load と DD update の最低限の sanity check

である。

よって、

- `5.54m` と `5.61m` の差をこの test だけで判断してはいけない

---

## 8. 次にやるべきこと

ここが一番重要。

### 8.1 優先順位

#### 優先度 A: best config の frozen preset 化 ✅ 完了 (2026-04-11)

sweep の結果、`mupf-dd-sigma-cycles` 0.25→0.20 と `mupf-dd-fallback-min-sats` 5→4 の 2 変更で
旧 historical best `5.54m` を上回る `5.48m` を達成。preset 更新済み、3 回再現確認済み。

#### 優先度 B: regression reference を test で守る

`odaiba_reference` を freeze したら、次にやるのは test の固定。

やること:

- `tests/test_exp_pf_smoother_eval.py` の smoke regression の bounds を現実に合わせて厳しくする
- counter range を固定する (DD used, fallback, anchor など)
- full-run regression の期待値を明記する

#### 優先度 C: その後で初めて精度改善

ここまで終わってから新しい modeling に戻る。

今の感触だと、

- tracked row をさらに boost する
- gate threshold をさらに sweep する

だけでは大きい改善は出にくい。

もし精度改善を再開するなら、

- weak-DD epoch の fallback 品質を上げる
- anchor / fallback 間の切替基準をもっと整える
- continuity の quality を diagnostics で定量化する

の順が良い。

#### 優先度 D: 停車検知 → dynamic sigma_pos で P50 1m 切り (2026-04-13 追加)

**発見**: epoch 別診断の結果、以下が判明。

| 条件 | エポック数 | P50 |
|---|---:|---:|
| DD=yes + IMU=yes | 7099 (58%) | **1.107m** |
| DD=yes + IMU=no | 4109 (34%) | 3.883m |
| DD=no + IMU=yes | 951 (8%) | 3.840m |
| DD=no + IMU=no | 69 (0.6%) | 8.291m |

DD+IMU が両方効くエポックは **既に P50=1.107m で 1m に近い**。
全体の P50=1.38m を引き上げているのは **IMU=no の 4178 epoch (34%)** で、これは **停車中** (IMU speed < 0.01 m/s で fallback)。

**改善方針**: 停車中は velocity=0 かつ sigma_pos を極小 (0.1-0.3m) にする。動いていないので particle depletion のリスクなし。

**実装箇所**: `experiments/exp_pf_smoother_eval.py` の predict ループ内、IMU fallback 分岐 (L1502-1504 付近)。

```python
# 現在: IMU speed < 0.01 → fallback to SPP velocity
# 変更: IMU speed < 0.01 → velocity=0, sigma_pos=0.1 (停車モード)
if speed_enu <= 0.01:
    velocity = np.zeros(3)  # stationary
    pf.sigma_pos = 0.1      # tight predict (not moving)
    used_imu = True          # IMU "detected stop" counts as used
    n_imu_used += 1
else:
    # existing IMU velocity code...
```

predict 後に sigma_pos を元に戻す:
```python
pf.predict(velocity=velocity, dt=dt)
pf.sigma_pos = original_sigma_pos  # restore
```

**CLI フラグ追加**: `--imu-stop-sigma-pos` (default=0.1)

**期待効果**: IMU=no epoch の P50 3.9m → ~1.5m。全体 P50 1.38m → ~1.1-1.2m。

**検証**: `--epoch-diagnostics-out` で IMU=no epoch の改善を確認。Shinjuku でも regression しないこと。

**注意**: frozen preset `odaiba_reference` は変更しない。新しい preset `odaiba_stop_detect` を作って検証。

### 8.2 現在の状況 (2026-04-13 更新)

優先度 A は完了。優先度 D (停車検知 dynamic sigma_pos) が最も有望な改善方向。

---

## 9. 既知の問題とハマりどころ

### 9.1 best config の漂流 (解決済み)

2026-04-11 に best config を preset として再固定済み (`5.48m`, 3 回再現確認)。
flag drift の原因は `mupf-dd-sigma-cycles` (0.25→0.20) と `mupf-dd-fallback-min-sats` (5→4) だった。

### 9.2 tracked fallback の actual use は少ない

診断では、

- tracked attempt は多い
- しかし actual use はかなり少ない
- candidate sat 数も `0-2` に偏ることが多い

という状態が見えている。

したがって、tracked 系は「存在しないよりは良い」可能性はあるが、現時点では主役ではない。

### 9.3 `libgnsspp` import の順序

`exp_pf_smoother_eval.py` 冒頭で `sys.path` を積む順番を間違えると、

- pure Python wrapper が先に来る
- build binding を期待する smoke regression が落ちる

という問題がある。

これは見落としやすい。

### 9.4 `pf_smoother_eval.csv` の上書き

experiment を回すと `experiments/results/pf_smoother_eval.csv` が変わる。

会話中は毎回バックアップして戻しているが、Claude が full-run を何本か回すならここは明示的に気をつけた方がいい。

### 9.5 `sub-meter` はまだ出ていない

ここは重要なので明記する。

- `sub-meter` はまだ未達
- ただし `old ALL-IN + smoother 6.24m` から `historical best 5.54m` までは縮めた
- 今の真の課題は「rescue stack を reproducible に保ちながら、さらに削る」こと

---

## 10. Claude に期待する振る舞い

この handoff の意図は「古い TODO に戻らせない」こと。

### 10.1 やってよいこと

- `odaiba_reference` の frozen condition を詰める
- `tests/test_exp_pf_smoother_eval.py` を強化する
- `pf_smoother_api.md` を参照して rescue flow を読む
- `exp_pf_smoother_eval.py` の helper 単位で整える

### 10.2 今はやらなくてよいこと

- `DD pseudorange` の新規設計
- `DD carrier` の全面書き直し
- FGO 導入
- PR #4 の merge
- 「worktree を clean に戻す」ための巻き戻し

### 10.3 もし精度改善に進むなら

再現条件を固定した後であれば、次の順がまだ筋がいい。

1. fallback quality を上げる
2. anchor / fallback の切替基準を diagnostics ベースで整理する
3. tracked continuity は veto 的に使う方向を考える

逆に、

- tracked 行をさらに強く boost
- gate threshold の微小 sweep を延々続ける

だけだと改善幅は小さい見込み。

---

## 11. 参考コマンド集

### preset 一覧

```bash
python3 experiments/exp_pf_smoother_eval.py --list-presets
```

### frozen reference full-run

```bash
URBANNAV_DATA_ROOT=/tmp/UrbanNav-Tokyo bash experiments/run_pf_smoother_odaiba_reference.sh
```

### frozen reference smoke

```bash
URBANNAV_DATA_ROOT=/tmp/UrbanNav-Tokyo bash experiments/run_pf_smoother_odaiba_reference.sh --max-epochs 10 --n-particles 5000
```

### preset helper regression

```bash
python3 -m py_compile experiments/exp_pf_smoother_eval.py tests/test_exp_pf_smoother_eval.py
PYTHONPATH=python python3 -m pytest tests/test_exp_pf_smoother_eval.py -q
```

### targeted CUDA smoke

```bash
PYTHONPATH=python python3 -m pytest tests/test_cuda_streams.py -k 'carrier_anchor_pseudorange_smoke or undiff_carrier_afv_smoke or smooth_with_dd_pseudorange_and_dd_carrier_smoke' -q
```

---

## 12. 一言で言うと

この repo の PF smoother 系は、もう「DD pseudorange をこれから実装する前段階」ではない。

今は、

- rescue stack は既にかなり入っている
- historical best `5.54m` は一度出ている
- だが frozen preset はまだ `5.61-5.62m`

という段階。

したがって Claude が最初にやるべきことは、

**best config の再固定と regression 化**

であって、新しい大改造ではない。
