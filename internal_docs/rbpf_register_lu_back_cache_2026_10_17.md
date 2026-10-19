# D255: 後退代入の解broadcastをレジスタに保持 (2026-10-17)

同期修正版D253を土台に、後退代入で解けたrowの値を一度だけ全warpへbroadcastしてレジスタに保持する。11×11の成功したguard内solveで、後退代入のdouble shuffle呼出しを55回から10回へ減らす。解けたlaneは以後その値を変えないため、元が繰り返し配っていた値と同じbitsを使える。row iは元と同じdescending i、row内ascending kのmultiply/subtractと最後のdivisionを実行する。tree和・FMA・近似逆数・演算順の変更はない。

全32laneがfull-mask shuffleに参加し、k>iのcacheは各rowで使う前に初期化済み。両status syncwarp、pivot比較/tie/交換、logdetの順序と途中失敗、generic fallbackとrange guardを維持。dimension3〜10は元generic。全観測・候補・粒子、非対称prior、IRLS回数、閾値/RNG、FP64/fmad=falseを保つ。前回D254の効果が小さかったidentity交換省略は含めない。

今回の追加cacheは速度効果不足として不採用。全scoreの増分+0.711/+0.227%は3%基準に届かず、A/A対照でも同程度以上の変動がある。shuffle呼出しの削減割合を全scoreの短縮率に換算できないことを、この実入力2設定で確認した。最適化CPUも超えず、製品・既定は維持する。

## 実入力の速度判断

同期修正版を同じ新実験内の対照として比較した。前回までの完了した候補の判断・時間を更新せず、絶対時間の比較や再測定による採用は行わない。今回の旧GPU比は+14.700/+11.237%、同期修正版からの増分は+0.711/+0.227%。base GPU条件は2/2、3%の増分条件は0/2、GPU component gateは0/2、CPU joint gateは0/2。製品・既定は維持する。

最適化CPU比は-68.606/-49.108%短縮（所要時間約1.686/1.491倍）。CPUからGPUへの切替はCPUを含むjoint gateで判断する。元GPUとの差を今回のcache変更だけの効果とは呼ばず、同期修正版との差を増分として扱う。

凍結した全394入力、10mode×5round×3group、59100timed/3940warmを1回の実験で比較。modeは最適化CPU、旧GPU、同DLL元GPU、新generic、同期修正版register、cache候補、および旧GPU/CPU/同期修正版/新候補の各A/A。rotate/reverse順序。全warmの同backend全diagnostic/status/score/input bits、全timed返却hashが一致。GPU warmは過去の凍結GPU基準とも照合した。

dedup/DD/cluster/pack/guard、全shuffle/cache/solve、転送/kernel/完了同期/返却、fallback/scatterはtimer内。workspace/warm/復元copy/hashは外。allowed CPU1本/BLAS1/normal priority/非exclusive、GPU処理は直列。timed callに追加event/counterなし。全197batch整数ns合計→round内3group paired率median→5round median。事前gateは旧GPU/同DLL元GPU/新generic各10%以上・各4/5round以上正、同期修正版からの増分3%以上・4/5round以上正、4種A/Aと同DLL元対照のabsmedian<5%。CPU比10%以上・4/5round以上正をjoint gateとして別判定。新genericはcode/resourcesが異なるので旧GPUとの等速対照にはしない。全round/rawを保持し、完了候補を再測定しない。

時間は197batch合計median÷197で個別batchのp50ではない。paired短縮率とは別統計量。

| cluster | CPU ms | 旧GPU ms | 同期修正版 ms | 新generic ms | cache候補 ms | 対旧GPU短縮 | 対同期修正版短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.8316 | 3.5859 | 3.0807 | 3.6245 | 3.0588 | +14.700% | +0.711% | -68.606% |
| 1 | 2.6848 | 4.5115 | 4.0132 | 4.5459 | 4.0032 | +11.237% | +0.227% | -49.108% |

| cluster | GPU A/A | CPU A/A | 同期修正版 A/A | cache候補 A/A | 同DLL元対照 | controls |
|:---:|---:|---:|---:|---:|---:|:---:|
| 0 | +0.776% | -1.526% | +0.132% | -0.666% | -0.046% | True |
| 1 | -0.137% | -0.266% | -0.172% | -0.236% | +0.116% | True |

全RTK/PF/FGO速度、GPU利用率やGPU化率は未確立。GTSAM solver/整数探索はCPUのまま。

## 精度・同期・機械語

289tests PASS。既存271testsの全score/normal/solution/logdet/status、fallback/例外/警告、context寿命と返却array所有権、3seed×4設定×4epochの全PF保存state/RNG/graph event比較を継承。raw6336行列はdimension3〜11、非対称/pivot/tie/特異/負det/NaN/Inf/overflow/subnormal/signed-zero/悪条件、guard内の後退代入Inf/NaN、6種の行交換patternを含む。

追加18test×32=576行列はrow0/3/6、指数54/100/499、符号±の3項相殺を作る。ascending-kでは先頭のunitが大きな減算で失われ、次に相殺し、最後の項が残る。順序を逆転すると0となるため、単なる許容差試験にしない。元generic・同期修正版・新候補のLU/pivot/解/logdet/statusをdtype/shape/bytesと既知の解で確認。合計6912行列、dimension11は1792行列。新たな数値許容差なし。

CUDA memcheck/racecheck/synccheck各46対象subset PASS、error/hazard/leak0。同じtestを3checkerで実行したもので138固有testとは数えない。各suiteに既存pytest asyncio設定warning1件。allocation/非同期障害の注入は未実施。

元numeric prefix/CPU body/通常kernel5144 normalized SASS命令は旧DLLと一致。新kernelは17328命令・64register/shared30904B/stack32B/spill0、同期修正版は18336命令・64register。rawは12136命令・96register/shared30904B/stack0/spill0。sourceの55→10は成功したguard内11×11後退代入の理論呼出し数で、全scoreの動的shuffle数やGPU利用率を測った値ではない。命令数を速度利得へ換算しない。

## 実GTSAMとGPUの精度

run3/seed20260710/400epoch、cluster0/1各1prefixを元D236 GPU基準と比較。400保存PF state、全trajectory配列のdtype/shape/bytes、全cfgと従来caller metadataが一致。各197valid/3skip、計394nonempty GPU score、scalar fallback0。fieldのみ全diagnosticを取得しdimension11が両設定で正と確認し、従来metadataだけをcallerへ返す。追加score callなし、guard-hitや動的shuffle counterなし。field時間を性能根拠にしない。全route/multiseed精度gate未実施。元NumPy/native間の既存丸め差と保存codecのscalar NaN payload正規化は継承。

通常scoreの追加buffer/転送/counterなし。staging host/device各1143352B/context、timing6contextsで各6860112B。raw probe最大device一時確保863232Bはtest専用。driver領域/CPU heap peak未測定。36製品source、5installed PE、PF build PE、70既報、34archives/5556filesを保全。

[結果](../benchmarks/rbpf_register_lu_back_cache_20261017.json)、[完了照合](../benchmarks/rbpf_register_lu_back_cache_completion_check_20261017.json)、[全score比較](../benchmarks/bench_rbpf_register_lu_back_cache.py)、[CUDA source](../benchmarks/probes/rbpf_register_lu_back_cache/register_lu.cuh)、[相殺の検査](../tests/test_rbpf_register_lu_back_cache.py)。
