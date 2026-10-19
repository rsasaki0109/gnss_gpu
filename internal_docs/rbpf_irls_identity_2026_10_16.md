# 厳密に同じIRLS状態の再計算を省く (D242, 2026-10-16)

**実走で再計算を省き、保存state/軌跡bitsを保った。ただしPF全体の有効な高速化は未実証で、D236/製品/既定を維持する。** 合成screenはCPU数値基準1/7、GPU主0/7だが、全14CPU/GPU warm snapshotで省略0件だった。CPU+13.45%をこの省略による利得と認定しない。実走では同一状態が発生するため、適用される実測入力での別の制御測定が必要。

## 同一性の証明と実装

まず行列分解の重複を調べた。D226尤度DLLは既に、各normal-systemのLUをsolution/logdetへ共用していた。今回はIRLSの再重み付け後の`Aw`と`bw`が、直前に解いたsystemと全要素でbit同一なら、次のnormal/RHS構築・LU・solveと、その後の同じ状態の更新を省く。

元の最大2回のIRLS更新という規則を維持する。行列と右辺が同じなら、同じprior・row順で作るnormal/RHS、strict partial-pivotの選択/因子/解が同じになり、残りのweightも同じ状態を繰り返す。任意の1bitが変われば元の次のsolveを行う。収束epsilon、近似的な打切り、精度変更はない。正負zeroも区別するuint64比較を使い、CPUはmemcpy、GPUはbinary64 bit intrinsicで確認する。

再重み付けのFP式/演算順、ordered row sums、LU/pivots、FP64、FMA無効、各行/hypothesis、prior/group順、失敗判定を保つ。異なるhypothesis間の行列や解を再利用しない。元の2更新が異なる結果を作る場合は全て実行する。新しいtolerance、粒子/候補/prior/閾値/RNG変更はない。D236のcaller/move/canonical/packing/確率/choice/conditional/FIX/feedback/graphは無変更で、D237/D239/D240/D241等の不採用候補やprofilerを併用しない。

新しいCUDA分岐はblock全体で同じ判定を読み、changedはatomicとbarrierで集約し、counterはlane0だけが書く。初稿のcounter書込みはbuild前にlane0へ制限した。初稿source/proofも残し、その1か所を戻すと初稿、生成変更を戻すとD226 source全体になることを独立再実行した。decompose/solve_lu/prepare_row/normal_entry/solve_normal/residual/likelihood_row/finish/write_resultの9関数は全文一致する。

`IdentityBatchScorer`は元のpacking/fallback処理を使い、完了したnative batchだけcounterを読む。empty/unsupported/error/capacity fallbackで前batchの値を読まない。新しい`irls_identity`counterは外部metadataへ分け、全既存metadataを保つ。`IRLSIdentityStep`へは明示的なscorerを渡す。scalar制御ではNoneまたはIdentityBatchScorerを使う。

## 精度とGPU検査

108tests PASS、既知pytest設定warning1。54fixture条件×CPU/GPUで、全score・solution・normal・logdet・statusのbits、全既存diagnostic metadataと入力不変性を比較した。uniform/非uniform、Huber有効/無効、cluster、3seed、rows/groups16/1・64/4・128/8を含む。

厳密zero-residual/signed-zero fixtureのCPU/GPU各3shapeでは2つのduplicate solveを省き、全出力bitsを保つ。変化するweight/Huber無効の対照、m0/1/63/127・partial block、asymmetric/singular/tiny/huge prior、empty/unsupported/capacity fallbackとstale counterも確認する。3backend×3seed×4構成×4epochの全保存state/位置/P/RNG/graph event/全従来metadata、strict NumPy途中state/errorもD236に一致する。

初回107PASS/1FAILは試験の容量条件が不一致だった。従来capacity512のGPU値と、候補capacity1の既存scalar fallbackを比べ、既存のGPU対NumPy差1.1368683772161603e-13を拾った。別診断で従来capacity1と候補capacity1のscalar fallbackがbit同一/metadata同一/counter非staleを確認した。初版test/source/plan/logを保存し、容量条件だけを合わせて108testsを通した。controller/CUDA source/DLLは最初のbuild/testから不変で、数値許容差は緩和しない。

memcheck74targeted tests、racecheck12、synccheck12 PASS（各既知設定warning1）。error/leak0、race hazard/error/warning0、同期error0。GPU検査は直列実行し、検査processのCPU availabilityを継承した。一般versionやCPU ASanの保証へは拡張しない。

## 完全PF-stepの制御比較

従来CPU/候補CPU/候補GPU/従来GPU/CPU A/Aの5modeを直列回転。7条件×5round×3group、525timed call/35warm snapshotを保存した。1許可CPU/BLAS1thread/通常priority、host/GPU非占有。bit比較・barrier/atomic・追加counter転送/query/array確保を含む候補の全コストと元PF全体を測る。restore/warm/callback再束縛/hash/検証は時間外。callback bodyのASTと毎callのRTK template不変も確認した。extract/search/time/satellite/graph callbackはfixtureで、GTSAM graph solveの速度ではない。

paired group比率のround中央値→5round中央値を使い、p50比とは区別する。CPUはD236 CPU比10%以上、4/5round以上改善、CPU A/A中央値abs5%未満。GPU主screenはD236 GPUと候補CPUの双方へ同基準を要求する。

| particles/unique/rows/groups | feedback | CPU p50 ms D236→候補 | CPU paired短縮 | 改善round | GPU対D236 GPU | GPU対候補CPU | CPU A/A | CPU数値screen |
|:---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 64/6/64/4 | 1 | 31.756→25.878 | -2.25% | 2/5 | -17.48% | -17.11% | -10.14% | FAIL |
| 64/32/64/4 | 1 | 52.098→45.288 | +13.45% | 4/5 | +9.15% | +14.97% | +2.80% | PASS |
| 64/64/64/4 | 1 | 71.894→54.630 | -1.63% | 2/5 | -3.16% | +0.60% | -3.02% | FAIL |
| 128/32/64/4 | 1 | 53.849→50.757 | +5.15% | 3/5 | +4.95% | +9.27% | +5.73% | FAIL |
| 64/6/128/8 | 1 | 38.880→39.779 | +3.95% | 4/5 | -8.61% | -2.98% | +2.10% | FAIL |
| 128/128/128/8 | 1 | 584.787→630.534 | +0.64% | 3/5 | -7.00% | +6.91% | +3.92% | FAIL |
| 64/6/64/4 | 0 | 15.587→14.051 | +1.32% | 3/5 | +4.94% | -15.36% | +16.96% | FAIL |

CPU数値基準1/7、GPU同backend副0/7、GPU主0/7、A/A失敗3/7。CPU64/32/64/4は+13.4451%、4/5、A/A+2.8050%で数値基準に通るが、この条件も含め**全7条件のCPU/GPU warm snapshotで同一状態/省略回数は0**。同じfixtureを復元するtimed試行ではcounterを個別保存していないため、それを新しい観測へ数え直さない。少なくとも今回の省略による高速化の証拠には使えない。再compile/コード配置/host変動等の影響も分離していない。

全負値/小利得/不安定roundを保持し、完了測定を再試行しない。数値基準1/7はそのまま記録するが、この最適化への帰属可能な速度screenは0で、採用数0。既定とD236を維持する。

## 実走での適用頻度と正確性

実GTSAM/cssrlib Tokyo run3/seed20260710、400epoch入力、元cluster_relin0/1×CPU/GPU各200、計800paired PF step。凍結D236同backend/同構成に対し、全保存state JSONL行、軌跡NPZ全配列dtype/shape/bytes、全cfg、全既存caller metadataが一致する。入力/source/external tc/両DLL/既存3拡張もhash照合する。各prefix197valid/3skip。

| prefix | paired steps | native batches | native score hypotheses | identical hypotheses | skipped normal/LU solves |
|:---|---:|---:|---:|---:|---:|
| r3_seed0_identity_cpu_c0 | 200 | 197 | 4166 | 1401 | 2802 |
| r3_seed0_identity_gpu_c0 | 200 | 197 | 4166 | 1401 | 2802 |
| r3_seed0_identity_cpu_c1 | 200 | 197 | 4063 | 1329 | 2658 |
| r3_seed0_identity_gpu_c1 | 200 | 197 | 4063 | 1329 | 2658 |

これは延べnative score inputのcounterで、粒子数や論理move数ではない。default cluster0では各backend4166件中1401件で2802回のnormal/LU/solveを省けたが、counter比率を時間短縮率へ換算しない。実走call_secondsは各1回の非制御観測なので、旧実走との速度比較に使わない。合成負荷の省略0と実走の適用を区別する。次は実走score入力を固定replayとして新規収集し、適用される負荷のcomponent速度とPF全体の速度を別々に検証する。

全route/multi-seed測位精度・whole-graph速度gateは未完了。有限scalar/全数値配列bitsを照合し、Python scalar NaN payloadをcanonical化する既存codec制約は残る。

## メモリと保存

counterはhypothesisあたりint32を1個、既存host/device出力workspaceとreadbackへ加える。capacity512でhost/device stagingは各1,143,352→1,145,400bytes、各+2,048bytes。新しいcross-hypothesis/epoch cacheはない。Pythonのlast_skippedは最新batchのint32 arrayだけを保持し、最大4*capacity bytes。その他Python/native/CUDA allocator/stack/shared等のheap peakは実測しない。比較は旧1context+候補1context、各実走は候補1contextで、全確保/queryコストを測定に含める。

44prior reports/21archives2673files、36production sources/5installed extensions/PF build PE、旧D226数値DLLと3CPU拡張を再照合した。製品source/model/既定は変更しない。公開source/docsはUTF8 CRLFを基本とし、測定hash固定のprobe READMEだけ元のLFを保持する。tests/JSONはLF、Ruff/compile/diff/公開名scan/artifact policyを確認する。初稿/初版失敗試験/全control/実走/両DLLとsource/proofを保存する。

[probe/API](../benchmarks/probes/rbpf_irls_identity/README.md)、[全測定・独立監査](../benchmarks/rbpf_irls_identity_20261016.json)、[完了照合](../benchmarks/rbpf_irls_identity_completion_check_20261016.json)。省略の正確性・実走適用は確認済み、速度利得の帰属は未確認としてD236を維持する。
