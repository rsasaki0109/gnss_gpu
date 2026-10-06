# gnss_gpu 引き継ぎメモ

**最終更新**: 2026-10-02 JST（docs 整合: §2 不整合の棚卸し、CONTRIBUTING lint、decisions.md 相互リンク）／2026-09-26 JST（repo hygiene: plan.md 再構成、decisions.md 未決定事項の棚卸し、result artifact policy の enforcement、`experiments/archive/`）
**対象**: 次に作業する coding agent（Claude / Codex / Cursor）
**旧版**: 2026-04〜07 の研究ログ（PPC Phase 11–80、GSDC2023 MATLAB 等価、validation wave 詳細）は [`archive/plan_history_2026-04_to_2026-07.md`](archive/plan_history_2026-04_to_2026-07.md) に移動済み。他文書の「plan.md §B / §0 / Phase NN」参照は旧版を指す。

このファイルは **入口だけ** を置く。数値の source of truth は各行の出典文書。更新時は §2 と §4 を書き換え、詳細は個別文書に書くこと。

---

## 1. 最初に読む順

1. このファイル §2（canonical 数値）と §4（next）
2. [`decisions.md`](decisions.md) 末尾「現在の未決定事項」と D-037
3. 取り組むトラックの出典文書（§3 の表）
4. `README.md`（公開 contract）、`CONTRIBUTING.md`（commit / hook ルール）

---

## 2. Canonical 数値（2026-09-26 時点）

**注意: PPC には contract が 2 つある。混ぜないこと。**

| 対象 | 値 | contract / 条件 | 出典 |
|---|---|---|---|
| PPC 公式 score（honest、現行） | **59.222040%**（6 route 平均）、safe FIX 11,031 / 48,778、false 0 | 欠測 epoch を分母に残す honest scorer（#159）、causal FLOAT selector（#166） | `docs/ppc_causal_float_selector.md`, `docs/ppc_causal_float_selector_evidence.json` |
| PPC 公式 score（旧 ranker contract） | 86.205492%（Phase71） | 旧 ranker / 旧 scorer。honest と比較不可 | `internal_docs/ppc_current_status.md`（最終更新 2026-05-19） |
| PPC library FIX-rate | Tokyo 54.36% / Nagoya 69.38%（false 3 / 0） | WP176 satellite-PAR surplus、default-off route | `internal_docs/wp176_fix_rate_final_2026_07_31.md` |
| PPC GNSS-only MultiSD-FGO | Tokyo 52.29% / Nagoya 70.30%、false 0 | dual-holdout。WLNL oracle 天井 52.72/70.36% → GNSS-only で 70/80% は不可 | `internal_docs/multisd_fgo_ppc_research_2026_08_01.md` |
| PF-only RTK（v0.3） | Tokyo `<50cm` 46.5112%、FIX 10.87%；Nagoya FIX 18.07% | Tokyo run1 は operational audit（virgin holdout ではない） | `RELEASE_NOTES_v0.3.0.md`, `README.md` |
| RB-FGO-PF（Tokyo `<50cm_full%` 3D） | 59.6 / 78.7 / 78.1（run1/2/3） | inuex35 56.7/69.9/67.9 超え。runtime は `experiments/rbpf_fgo/`（2026-10-06 に `repro_tc_fgo` から凍結コピー） | `internal_docs/inuex35_tc_fgo_benchmark.md` |
| GSDC2023 Kaggle | best v13 public 3.224 / private 3.783 m | 2026-06-10 以降 GSDC commit なし | `docs/gsdc2023_solution.md` |
| UrbanNav Odaiba | PF100K P50 1.36 m / RMS 4.11 m（RTKLIB 2.67 / 13.08 m） | external mainline は PF+RobustClear-10K RMS 66.6 m（D-029） | `README.md` |
| PF3D-BVH on real UrbanNav | all-LOS RMS 2D 37.63 m、NLOS 枝付きは 80–85 m | Odaiba G-only 300 epoch 診断 | `decisions.md` D-037 |

既知の不整合（2026-10-02 棚卸し）:
- 解消: Nagoya PF-only `<50cm` の 69.55% は WP100（7/23）の旧値で、WP172/173（7/29）で 5,715/7,583 = 75.37% に更新済み（`wp172_nagoya_development_2026_07_29.json`）。README が正、`HANDOFF_CLAUDE_PF_ONLY_2026_07_23B.md` に superseded 注記を追加。69.55% は non-degradation floor。
- 一部解消: PPC 分母の 11,928 は inuex35 README の rover epoch 数を `experiments/score_vs_inuex35.py` に hard-code したもの（Tokyo run1 の inuex35 比較で使用）。PF-only の 11,924 / 7,583 は PF replay の full denominator。4 epoch / 19 epoch 差の原因はデータ未所持のため未確認。2 つの contract の epoch 数を混ぜて比較しないこと。
- 解消（2026-10-02）: `score_vs_inuex35.py` の `_ROVER_EPOCH_COUNTS` が Nagoya run1/2/3 に Tokyo の値（11928 / 9151 / 15301）をコピーしていた。inuex35 README は Tokyo のみなので Nagoya 行を削除し、reference 長に fallback させた。commit 済みの Nagoya 結果でこの値を使ったものは無い。
- 旧版 plan の GSDC 3.993/4.821 は古い bridge 提出の値。現行 best は上表。
- `internal_docs/ppc_current_status.md` は旧 ranker contract の文書として stale 注記を追加（本文は未更新）。
- 解消（2026-10-03）: `benchmarks/RESULTS.md` を Turing 世代 GPU で再計測し、README の速度表記も Device PF の実測（1M 粒子 32 ms）に更新。

---

## 3. トラック一覧（2026-07-03 → 09-26）

| トラック | 期間 / PR | 状態 | 出典 |
|---|---|---|---|
| Validation / refactor waves（input_validation、pybind header、PF device 分割、DD kernel fuse） | 7/3–7/12, #104–#116, #124–#126 | 完了 | `docs/common_input_shapes.md`, 旧版 plan |
| NLOS wave（PLATEAU per-epoch mask → PF/DD、ranker headroom） | 7/4, #117–#118 | 不採用（Δ=0、oracle headroom 0.0pp） | `fable5_nlos_wave2_advice_response_2026_07_04.md`, `nlos_pf_measurement_wiring.md` |
| inuex35 TC-FGO campaign / RB-FGO-PF | 7/6–7/11, #119–#123 | RB-FGO-PF milestone-2 達成。run3 false-fix 3.09% は 2026-10-06 に report floor `nb >= 12` で 0.05%（位置・`<50cm` は不変、fix 率 −4〜5 pp） | `inuex35_tc_fgo_benchmark.md`, `rbpf_fgo_design.md` |
| Structural audit / v0.2.0 | 7/14, #127–#129 | 完了 | CHANGELOG |
| Scenario engine / coverage map / demos / UTD CUDA | 7/17, #130–#136 | 採用（README「Simulate GNSS anywhere」） | CHANGELOG |
| PF-only RTK（WP21–WP173） | 7/17–7/29, #137, #150, #151 | promotion floor 達成、81/86% stretch 未達 | `HANDOFF_CLAUDE_PF_ONLY_2026_07_23B.md`, `pf_only_rtk_stretch_plan_2026_07_19.md` |
| v0.3.0 release platform（urban nav phase 0–6、wheel、ROS 2 soak） | 7/29, #138–#149 | released | `RELEASE_NOTES_v0.3.0.md`, `urban_navigation_phase*.md` |
| PF/FGO GPU（cuSOLVER）、WP176 FIX-rate | 7/31, #152–#155 | 採用（default-off） | `pf_fgo_gpu_2026_07_31.md`, `wp176_fix_rate_final_2026_07_31.md` |
| GNSS-only MultiSD-FGO、PPC safe/causal pipeline | 8/1–8/2, #156–#166 | 採用（FIX authority は safe IMU PF/FGO tracker のみ） | `multisd_fgo_ppc_research_2026_08_01.md`, `docs/ppc_pf_fgo_research_plan.md` |
| GPU onboarding CLI / run compare | 8/28, #167–#168 | 採用 | README |
| UrbanNav data loop / PLATEAU / PF3D-BVH NLOS | 9/16, #169 + direct commits | per-particle 3DMA 不採用（D-037）、opt-in のみ残す | `decisions.md` D-037 |
| Repo hygiene（artifact policy 強制、experiments 108 本 archive、`gnss_gpu.metrics` 抽出） | 9/26 | 完了（branch `agent/repo-hygiene-docs`） | このファイル, `results/ARTIFACT_POLICY.md`, `experiments/archive/README.md` |

---

## 4. Next（優先順）

### 研究

1. **D-037 後の pivot を 1 つ選ぶ**（`decisions.md`「現在の未決定事項」）。最小コストの試験は「3D マップ遮蔽率を PPC ranker / RTK 除外候補の feature として 1 列追加」。粒子ごとの遮蔽判定を尤度に入れる案は再提案しない（D-033, D-037）。
   - 2026-10-03 確認: 候補 (c) の ranker への NLOS 特徴量は 7 月の Wave 2 で既に検証済み（ranker 層は飽和、名古屋 run2 の oracle headroom 0.0 pp）、PF/DD への PLATEAU mask は Wave 1 で Δ=0。同じ形での再試行は見込みが薄い。
2. **PPC honest 59.22% → 25% safe-FIX milestone**（12,195 epoch、残 1,164）。候補: DD-reference 変更をまたぐ ambiguity evidence の持続、candidate oracle headroom 4,038 epoch。出典 `docs/ppc_pf_fgo_research_plan.md` L86–142。
   - 2026-10-03: 正規パイプライン（safe IMU PF/FGO tracker 11,031 FIX → 58.521912% → causal FLOAT selector 59.222040%）をWindows dev 機で全 route・全スコア完全再現。gnssplusplus `62bd0b73` を GTSAM 付きでビルドして使う。手順は `ppc_canonical_reproduction_windows_2026_10_03.md`。注意: `run_ppc_basin_fgo_six_route.py` の既定値（top-k 4 等）は凍結ポリシーと違う。
3. **#169 の実 UrbanNav データ e2e**（`gnss-gpu run --preset urbannav-pf`）が未実行。
4. ~~RB-FGO-PF run3 の false-fix 3.09% → ~2%~~（2026-10-06: report floor を `nb >= 12` に上げて 0.29/0.00/0.05%、leave-one-run-out で選定、D-040。per-cluster relinearization は WP19–37 で出荷不可と判明済み）。残: run3 の 0.53 m ずれ区間は位置が直っていない。FIX/FLOAT 食い違い時に FLOAT を出す案は WP38 で否定（区間内で FIX と FLOAT は 0.09 m で一致、両者に共通のバイアス）。WP39（D-041）で原因は E04/G09 の NLOS 化と特定したが、ずれは出荷版シードでしか起きない確率的な失敗だった。NLOS-AR 除外は run1 のみ確実に +3.0 pp OFFICIAL で、地図あり 2 パスのオプション扱い。以後、RB-FGO-PF の比較は 3 シード以上で行う。地図なしの C/N0 低下フラグ（基地局比 8 dB）で代替する案は WP40 で否定（run1 の改善が再現せず、3 シード平均 OFFICIAL 56.2/81.9/83.9）。runtime は `experiments/rbpf_fgo/` に凍結コピー済み（run1/run3 の先頭 400 エポックで出力がビット一致。GTSAM ビルド段は clean machine で未検証）。
5. PF-only に virgin holdout が無い（Tokyo run1 は operational audit）。

### エンジニアリング / docs

6. **`experiments/exp_ppc_ctrbpf_fgo.py` の分割**（2026-10-03 第 1–2 段完了: 12,680 → 8,066 行）。I/O・CLI パース（`ppc_ctrbpf_io.py`）、`CTRBPFConfig` / `_config_variants`（`ppc_ctrbpf_config.py`）、RTK diag の gate / sort / run-index policy（`ppc_ctrbpf_rtkdiag.py`）を AST 同一のまま移動し、元モジュールから全名 re-export。第 2 段で `main` 冒頭の argparse 定義 797 行を `ppc_ctrbpf_cli._build_arg_parser()` へ移動（`--help` とデフォルト値が完全一致）。残り: `_run_ctrbpf_on_segment`（3,427 行、うち epoch ループ本体 2,802 行）と `main` の run ループ（約 1,400 行）の内部分割。前提として 2026-10-03 に refactor guard `experiments/fingerprint_ctrbpf_segment.py` を追加（合成区間で 43 method を実行し全戻り値の SHA-256 を出す、同一 GPU で決定的。`_gnss_gpu_pf_device` 必須）。ただし合成入力で通るのは関数の 34%（DD 0–2%、IMU-TC 1%、RTK diag 37%、velocity KF 52%）。ループ内ブロックは PF・統計オブジェクト・数十のローカルを共有するため、引数を並べた関数抽出では結合が減らない。次の一手は (1) 合成 DD computer / IMU 入力で guard のカバレッジを上げる、(2) per-run 状態を 1 つの dataclass にまとめてからブロックを method 化する、import 元 50 ファイルを新モジュール直参照へ移すか。引き続き「数値挙動を変えない」commit に限定する。
7. ~~`experiments/gsdc2023_*` を package にまとめるか~~（2026-10-03 判断: 今は移さない。`decisions.md` D-038。335 import と 17 か所の module monkeypatch のため shim 移動は危険、GSDC 休眠中で便益が薄い）
8. ~~`benchmarks/RESULTS.md` の再計測~~（2026-10-03 完了: Turing 世代 6 GB GPU で全 native module を計測し 2026-04 の表は historical として併記。PF Device が標準 PF より遅かった原因は Megopolis が毎反復 16 double/粒子をコピーしていたこと。index 化で bit-identical のまま 153 → 32 ms、host-buffered PF の megopolis の GPU メモリリーク（4 buffer 未解放）も修正）。README の速度表記は Device PF の実測（32 ms、Turing）に更新。
9. ~~`CONTRIBUTING.md` の lint 指示を CI に合わせる / decisions.md 2 本の関係を明記~~（2026-10-02 完了）。repo 全体の ruff（`ruff check .` で 554 件）を CI 対象に広げるかは未決定。
10. CI: ~~coverage と pyright basic~~（2026-10-03 導入: full-suite に `--cov=gnss_gpu`（job summary + `coverage-xml` artifact）、PR gate の `typecheck` job で pyright basic を ratchet 方式で強制。負債 71 ファイルは 2026-10-03 に全て解消し exclude は空、`extraPaths: ["python"]` で installed copy ではなく repo source を解決）。self-hosted CUDA workflow は 2026-10-03 からネイティブ依存の test 一式と PF3D-BVH 短区間回帰（`tests/test_pf3d_bvh_short_segment.py`）を実行。合成 street canyon では完全な地図でも PF3D-BVH（2D RMS 10.6 m）が 3D 非考慮 PF（7.5 m）より悪い（未調整パラメータ、D-037 と同傾向）。
11. ~~GSDC bridge Doppler 2 件の strict xfail~~（2026-10-03 解消）。実装は正しく fixture が古かった: `4f7fc65`（6/6）で raw bridge が `doppler=+PseudorangeRate`・`clock_drift_mps` も正符号に変わったのに fixture が旧符号のまま、かつ `b0607ef`（5/8）で L-factor 用の `_build_trip_arrays(use_tdcp=True)` 呼び出しが増えていた。
12. ~~full suite を CI で回す~~（2026-10-03 完了: `.github/workflows/full-suite.yml`、毎日 03:00 JST + `workflow_dispatch`、ubuntu-latest / ネイティブ拡張なし。依存は `scripts/ci/requirements-full-suite.txt` に固定）。PR gate にはしていない。
13. ~~`tests/test_cuda_streams.py` の 11 件~~（2026-10-03 解消）。validation wave（`3ede3c0`）は `spread_pos=0` / `sigma_pos=0` の拒否を `test_pf_device_wrapper.py` で明示的に固定しているため契約は変えず、4 月の古い test 側を `NEAR_ZERO_SIGMA = 1e-12` に置換。self-hosted CUDA の test 一覧に追加。
14. **PF device リサンプリング**（D-039）。(a) ~~既定 `megopolis` の切替~~（2026-10-03 完了: 正しい Megopolis B=60 を既定に、旧版は `megopolis_legacy`。PPC と UrbanNav Odaiba で評価）。(b) `pf_device_resample_systematic` の u0 二重除算バグ（FFBSi 系に影響、未修正）。(c) `_resample` が毎回同じ seed（PPC では効果なし）。(d) ホスト側 `ParticleFilter` の megopolis も同じ欠陥（未評価）。
15. ~~README headline（Odaiba PF smoother 1.36 m / 4.11 m）の再現~~（2026-10-04 原因判明・解決: libgnsspp の `CorrectedMeasurement` に `prn` が無く、DD pseudorange / widelane / DD carrier が実データで一度も動いていなかった（Odaiba 0/12200 epoch）。gnssplusplus #557 で `prn` を公開し pin を develop `304798e7` に更新、DD 有効時に `prn` が無ければ即エラー。5 seed で SMTH P50 1.50 m / RMS 4.06 m、4 月の RMS と一致。`urbannav_pf_dd_satellite_ids_2026_10_04.md`）。以下は 2026-10-03 時点の誤った結論（記録として残す）: 再現不能。README を記録した `421d284` 自体でも今のデータでは SMTH P50 1.99 m / RMS 12.71 m で、gnss_gpu のコード変更は原因ではない。gnssplusplus の SPP 版は P50 を 0.3–0.7 m 動かすが RMS は説明しない。評価 epoch 数が 4 月 freeze の 12228 に対し現データでは常に 12184 で、4 月は別版の Odaiba データだった。4 月データと gnssplusplus pin `4932676` は手元に無く復元不能。README は現 main の値に更新済み（#191）。詳細: `resampler_ablation_urbannav_2026_10_03.md`。残作業: 公開サイト snapshot と demo を現データで再生成するか判断。
16. **UrbanNav PF の RTK 欠落区間精度**（2026-10-04 着手）。RTK FIXED epoch で粒子群を fix まわりに引き直す `--rtk-anchor-pos` を追加（前向き・後ろ向き両方、既定 off）。5 seed で RTK + PF 穴埋めの <5 m は Odaiba 87.8→88.4%、Shinjuku 84.2→84.4%、穴の中の P50 は −0.4〜−0.7 m。限界は FIX を失った後の約 1 m/s のずれ（IMU 誘導の速度誤差とみられる）。同日追記: 原因は 10 Hz SPP 差分による方位補正（雑音が移動量を上回る）。`--rtk-anchor-heading`（FIX 区間の差分でのみ方位補正、他はジャイロ）+ `--sigma-pos 0.1` で、アンカー付き PF 単体が Odaiba <5 m 98.1% / <3 m 93.0% / RMS 1.26 m、Shinjuku 91.8% / 87.8% / 2.42 m（5 seed、RTKLIB は 86.6% / 69.0%、73.5% / 63.4%）。注意: σ_pos 0.1 はこの 2 ルートで選んだ値。σ_pos は 2 ルート間の交差確認で 0.05–0.1 が最適、保守側の 0.1 で named preset `urbannav_rtk_anchored`（`--rtk-anchor-pos` 必須）を追加。同日 PPC 6 route で held-out 確認（`ppc_pf_rtk_anchor_2026_10_04.md`）: Tokyo は再調整なしで改善が再現、Nagoya はジャイロバイアス 0.15 deg/s で長い欠落が崩れた。`--imu-speed-source doppler`（gnssplusplus #558 で Doppler / 衛星速度を公開、pin `9d89a58a`）と `--imu-gyro-bias-zupt` を追加し、preset `rtk_anchored_doppler` で 6 route 平均 <5 m 88.4%（anchor のみ 84.5%、RTK 単独 78.6%）。ZUPT の UrbanNav での悪化は方位ではなく停止中の漂い（stop σ 0.1 m/epoch）が原因と判明し、stop σ 0.01 + ZUPT を両 preset に入れて統一（Odaiba 5 seed: <1/<3/<5 m 80.7/98.0/98.1%、RMS 0.83 m、PPC 平均 <5 m 88.9%）。2026-10-05 PPC 5 seed で確認: 6 route 平均 <0.5/<1/<3/<5 m = 67.9/72.8/83.9/88.9%（anchor のみより全閾値 +4.3〜5.3 点、seed sd ≤ 0.4）。例外 nagoya run1（<5 m −1.2 点、RMS 6.8 m）。原因は Doppler 速度の外れ値（マルチパス 1 本で 30 m/s 級）で、残差最大の行を除く頑健 LS にして 6 route 平均 <5 m 90.4%（seed 42）、nagoya run1 は依然 anchor のみを 1.3 点下回る。nagoya run1 の残りは 15 s の受信断（測定ゼロ）: 1 回の predict で直線外挿 + dt 非依存の σ_pos で 73 m ずれて戻れなかった。IMU guide を区間内のジャイロ経路平均方向に（既定）、受信断後の σ 拡大 `--predict-gap-velocity-sigma` を opt-in で追加（nagoya run1 は改善するが Shinjuku で悪化）。残差判定・FLOAT による立て直しは不採用（記録のみ）。代わりに平滑化の前向き/後ろ向きを各アンカーからの時間で重み付け（`--smoother-anchor-weighting`、両 preset に追加）し、5 seed で <1 m が Odaiba +3.7 / Shinjuku +2.7 / PPC 平均 +2.1 点、PPC RMS は +0.26 m。UrbanNav Hong Kong 2019-04-28 は held-out に不適（8 分・1 周波 u-blox・基準局 30 s 間隔で RTK FIX 0）。公開サイトの snapshot は 2026-10-05 に現在値へ更新（入力 `experiments/results/urbannav_current_checkpoint.json`）。記録 `urbannav_pf_rtk_anchor_2026_10_04.md`。

---

## 5. 作業ルール（要点）

- **PR 経由で main に入れる**。pre-push hook は main への直接 push を禁止している（9/16 の NLOS commit 5 件は PR なしで入ってしまった）。
- **AI の `Co-authored-by` trailer は付けない**（`.githooks/commit-msg` が reject する）。
- 採用判断は holdout / disjoint holdout / LORO を通してから。単一 pilot や合成データでの成功で既定値を昇格させない（D-009, D-010, D-037）。
- 負の結果も `decisions.md` に D-entry として残す。
- 生成物は `results/ARTIFACT_POLICY.md` に従う。commit する artifact は文書から参照されている必要があり、CI（`scripts/ci/check_artifact_policy.py`）で検査する。
- Validation / refactor の commit に score 改善を混ぜない。kernel の数値式を validation commit で変えない。

## 6. コマンド

```bash
# local smoke（CUDA rebuild 不要）
PYTHONPATH=python python -m pytest tests/test_*_wrapper.py -q
bash scripts/ci/run_python_smoke.sh
# full suite（環境依存のテストは skip、既知の drift は strict xfail。fail 0 が正常）
PYTHONPATH=python python -m pytest tests -q
# lint（CI と同じ）
python -m ruff check python/ --ignore=E501,F401
python tools/lint_repo_paths.py --max-path 180 --max-name 128
python scripts/ci/check_artifact_policy.py
# GPU onboarding
gnss-gpu doctor
gnss-gpu run --preset signal-acquisition
```

CI（`.github/workflows/ci.yml`）: actionlint、ruff、repo path lint、commit message lint、artifact policy check、python smoke、wrapper smoke、site smoke（playwright）、CUDA import smoke。self-hosted CUDA runtime は `cuda-runtime-self-hosted.yml`（`workflow_dispatch`）。
