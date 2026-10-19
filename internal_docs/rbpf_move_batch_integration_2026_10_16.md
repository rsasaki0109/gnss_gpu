# RBPF一括尤度のstep接続 (D227, 2026-10-16)

D226の明示C++/CUDA尤度APIを、凍結公開RBPFの完全な`step`へ接続する実験用callerを実装した。`MoveBatchStep`を対象instanceへ明示設定して使う。公開凍結source・製品source/PE・既定設定は変更しない。

元のmove loopだけを置換し、extend/shrinkを各held particleにつき各1回生成する。元の全move、重複candidateのprior、粒子順、乱数呼び出し順を保持する。候補を再生成してprefetchする方式は使わない。元cacheのgauge keyと最初の要求順で未評価尤度をまとめ、空仮説のburst scoreとshadow/debugによる既存cacheを引き継ぐ。
全候補の尤度取得後、元のweight/choice式を同じ粒子順で実行する。全粒子の選択後に状態を反映する元commit loop以降は無改変。抽出、maintenance、shadow、条件付き位置、FIX閾値、保持feedback、resampling、cluster state公開順を保持する。IRLS数、候補数、FP64精度、モデル式は変更しない。

source SHAと関係methodを検証し、別モデルやbind後のmethod変更を拒否する。strict/underflow-warningなどNumPyの非対応error policyでは、staging前に元の完全stepを呼び、例外と部分的な粒子/RNG更新も引き継ぐ。通常の実験domainは有限で整形済みの入力。任意の壊れた入力やユーザーcallbackの副作用について処理途中の順序まで同等と保証する汎用置換ではない。scorerのサイズ/dtype等の既存fallbackは全要求をscalarで計算し、切り捨てない。

## 実データprefix

既存のcustom GTSAM/cssrlib環境と実PPCデータを読み、shipped設定のTokyo run3、seed 20260710、400同期epochを元実装/GPU版で直列実行した。NumPyの合成oracleやGTSAMのmockではない。起動後に200回の実RBPF stepがあり、197回が有効、3回がskipだった。

- 報告位置、float位置、Nav covariance、条件付き位置/P、FIX、全粒子Hと選択tag、RNG state、basin順、保持commit/releaseとcounterが全200stepで一致。位置・共分散の配列bitは一致した。
- 保持commit 84、mass release 11、unseen release 2、resampling 27、reported FIX 193は双方同じ。400epochの`sol_xyz`/`flt_xyz`/`smode`もbit一致。
- 連続量すべてのbit一致ではない。保存NPZで差があるのはgamma、second_gamma、gamma_pair、ESS。全保存stateの最大差はESSの5.968558980384842e-13、gamma最大差8.77e-15だった。LLを元NumPyとbit一致とは扱わない。
- 197回の全batchがnative実行、scalar fallback 0。moveのunique数は14〜45、中央値21。12,608 particle更新、held 12,504、要求move 206,522を保持した。

prefixの測位誤差指標は双方同じだが、**1route・1seed・400epochであり、全区間・multi-seedの測位精度gateは未完了**。snapshotのNumPy配列はdtype/shape/全bytesを保存する。Python scalar NaNはJSON codecでcanonical化するため、そのNaN payloadまで一致したとの主張には使わない。
記録したRBPF call合計は元12.960s、GPU11.742s。ただしunpinnedで各1回の診断で、初回compile/cacheや測定順も異なる。約9.4%の差を採用可能な速度効果とは認定しない。runner全体のwall時間差をGPUの利得に数えない。

## 完全stepの合成比較

元の完全RBPF stepを実行し、graph抽出・時刻/衛星識別・integer-search入力だけを合成fixtureとした。graph solveは含まない。native CPU、GPU、stagingのみscalar、元実装、同一元実装A/Aを5round×3groupで回転比較。計450 timed step、270候補比較、90 A/A比較。候補生成・dedup/pack・全転送/返却・条件付き位置・mass/FIX判定・resamplingを含み、setup/状態restore/hash/snapshotは時間外。
自processのみ1許可CPU・1BLAS thread、通常priority、GPU/host非占有。主指標はpair短縮率のround内中央値→5round中央値。採否screenは元比10%以上かつ4/5round改善、A/A中央値abs5%未満。GPU固有のscreenにはC++比でも同条件を要求する。

| 粒子 / rows / groups / 実unique | 元 ms | C++ ms | GPU ms | C++対元 paired短縮 | GPU対元 paired短縮 | GPU対C++ paired短縮 |
|:---|---:|---:|---:|---:|---:|---:|
| 64 / 64 / 4 / 17 | 31.787 | 24.770 | 27.168 | 22.25% | 17.95% | -3.54% |
| 64 / 64 / 4 / 47 | 42.323 | 27.585 | 29.010 | 34.85% | 29.83% | -6.68% |
| 64 / 64 / 4 / 95 | 56.784 | 30.358 | 34.773 | 49.29% | 41.98% | -15.78% |
| 128 / 64 / 4 / 95 | 84.344 | 55.168 | 59.942 | 35.54% | 29.07% | -6.82% |
| 64 / 128 / 8 / 17 | 54.202 | 45.875 | 52.962 | 10.88% | 2.85% | -7.53% |
| 128 / 128 / 8 / 95 | 184.276 | 150.297 | 156.190 | 22.50% | 16.10% | -6.41% |

A/Aは全6条件でscreen内。C++/GPUの対元screenは各5/6条件で通過。64粒子/128rows/17uniqueは各3/5roundしか改善せず不合格。**GPU対C++は0/6条件で通過、今回のcallerではC++が優勢。** D226の尤度部品のGPU対C++合格条件を完全stepへ外挿しない。scalar A/AはGPU独自の全変動を測る対照ではない。再試行して不合格を上書きしない。
450callすべてで離散/RNGと条件付き位置/Pを比較し、scalar stagingは全保存数値bit一致。nativeの連続量は診断上の1e-10内で、精度閾値を変更して採用したものではない。

## 試験・メモリ・使い方

新65tests PASS、pytest設定warning1。3seed、4連続epoch、cluster/uniform/Gaussian/IRLS有無、shadow/spawn、resampling有無、maintenance release/fresh/reseed、skip、strict例外の部分state、実際のfeedback commit/releaseを確認した。profileで元・候補ともextend/shrinkがheld particleにつき各1callであることも検査した。
新callerのmemcheckは8 targeted tests PASS、warning1、0 error/0 leak。CU/DLLはD226と同一で再buildなし。D226のracecheck/synccheckを今回の新規PASS数へ加算しない。
最初のstrict試験は元実装の例外前RNG消費を未消費と期待し、61PASS/1FAILだった。元実装との部分state比較へ修正し、失敗source/logを保存した。数値許容差やモデルを緩めていない。実データ照合scriptのNPZ位置key誤記も、raw runを変更せず別auditorで修正した。

全moveを一時保持するためPythonメモリは増え得る。別のuntimed tracemalloc診断で128粒子/128rowsは元peak1,765,203B、C++2,092,238B、GPU1,904,876B。単回のPython allocation測定で、allocator再利用・native/CUDA分を含む安定した全体peakやCPU/GPU間のメモリ優劣とは扱わない。D226のcapacity512 workspaceはdevice1,143,352B＋pageable host1,143,352Bを保持する。

明示利用例（scorerの寿命は呼出し側で管理）:

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_move_batch import MoveBatchStep

saved_step = pf.step
with BatchScorer(explicit_dll_path) as scorer:
    pf.step = MoveBatchStep(pf, scorer, mode="native_cpu")  # "gpu"も明示選択可能
    try:
        result = pf.step(rtk, epoch_context)
    finally:
        pf.step = saved_step
```

再現helper: [完全step合成比較](../benchmarks/bench_rbpf_move_batch.py)、[実データrunner](../benchmarks/bench_rbpf_move_field.py)（既存custom依存環境で`--mode/--dll/--tc-src/--data-root/--output-dir`を指定）。全raw/初期失敗/plan/source/DLLは`dist/rbpf_move_batch_integration_20261016/manifest.json`で固定保存する。
記録: [全結果JSON](../benchmarks/rbpf_move_batch_integration_20261016.json)、[完了照合](../benchmarks/rbpf_move_batch_integration_completion_check_20261016.json)。次は同じモデルでの全route・multi-seedとgraphを含む統制比較。新しい精度改善モデルへの接続は別途source/model監査が必要。
