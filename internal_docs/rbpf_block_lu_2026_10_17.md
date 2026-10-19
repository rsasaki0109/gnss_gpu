# D256: 128-thread blockで行列要素ごとにLUを分担 (2026-10-17)

11×11 normal solveのLUを、128-thread block全体へ分担した。121要素をthreadへ対応させ、最初のtrailing updateでは100要素を並行更新する。pivotはlane0の元のascending比較/tie順。row交換は列ごとに分担し、lower-columnの除算を行ごとに進め、barrier後に各trailing要素を更新する。各要素が受けるascending-jのdivision/multiply/subtract順は元どおり。D231の32-lane shared-row版、D253のrow-register版、D254/D255の細かな通信削減とは別の分担方法で、前回のmicrochangeは混ぜない。

全128threadが全collectiveへ入る。初期normal/RHSに非有限またはabs>2^500があればblock全体で元genericへ分岐する実装guardを継承。pivot失敗はlane0のlocal predicateをsyncthreads_orで全threadへ配り、共有statusを各warpが読みながら更新する競合を避ける。pivot/交換/除算/更新をblock barrierで区切る。途中失敗のLU/pivot/partial logdetを保ち、対角logを元j順で加算する。前進・後退代入はliteral original solve_luをlane0で実行し、callerのblock barrierで解/statusを渡す。[CUDA12.8.1 synchronization](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#synchronization-functions)。

全観測・候補・粒子、非対称prior、IRLS回数、閾値/RNG、FP64/fmad=falseを維持。tree和・FMA再結合・近似逆数・jitter・対称化なし。dimension3〜10は元generic。selector/API名は既存検証用ABIを再利用し、RegisterLUScorerのkernel=register/selector2がこのDLLではblock-element LUを選ぶ。mode0元kernel、mode1同じ新kernelのgeneric対照。名前からrow-register分担と解釈しない。raw probeも128threadへ変更し、配列形式・最大256行列・確保byte数は同じ。通常scoreに追加buffer/counter/転送なし。

今回のblock分担は効果不足として不採用。同期修正版からの差は-0.042/+1.015%で事前の3%増分基準を満たさず、block候補A/Aのmedian変動も約1.1/1.2%。単純に分担thread数を増やすだけでは、この2設定の全score時間はほぼ縮まらなかった。LU更新・逐次代入・block同期それぞれのwall timeは未測定で、原因の割合は断定しない。最適化CPUも超えず、製品・既定は維持する。

## 実入力の速度判断

旧GPU比は+12.715/+9.752%、同期修正版D253からの増分は-0.042/+1.015%。base GPU条件1/2、3%増分条件0/2、GPU component gate0/2、CPU joint gate0/2。製品・既定は維持。元GPUとの差をこの分担変更だけの効果とは呼ばず、同期修正版との差を追加効果として扱う。最適化CPU比は-68.821/-41.599%短縮、所要時間約1.688/1.416倍。

全394凍結入力、10mode×5round×3group、59100timed/3940warmを新しい1実験で比較。modeは最適化CPU、旧GPU、同DLL元GPU、新generic、同期修正版register、新block候補、および旧GPU/CPU/同期修正版/新候補のA/A。rotate/reverse順序。全warmの同backend score/全135diagnostic/status/input bits、全timed返却hashが一致。GPU warmは過去の凍結GPU基準とも照合。

dedup/DD/cluster/pack/guard、block vote/barrier/LU/代入、転送/kernel/完了同期/返却、fallback/scatterはtimer内。workspace/warm/復元copy/hashは外。allowed CPU1本/BLAS1/normal priority/非exclusive、GPU処理は直列。timed callに追加event/counterなし。全197batch整数ns合計→round内3group paired率median→5round median。事前gateは旧GPU/同DLL元GPU/新generic各10%以上・各4/5round以上正、同期修正版から増分3%以上・4/5round以上正、4種A/Aと同DLL元対照absmedian<5%。CPU比10%以上・4/5round以上正をjoint gateとして別判定。新genericのcode/resourcesは違うので旧GPUとの等速対照にはしない。完了候補の時間/採用判断は更新せず、絶対時間比較・採用のための再測定なし。全round/rawを保持。

時間は197batch合計median÷197で個別batchのp50ではない。paired短縮率とは別統計量。

| cluster | CPU ms | 旧GPU ms | 同期修正版 ms | 新generic ms | block候補 ms | 対旧GPU短縮 | 対同期修正版短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.9695 | 3.8177 | 3.3641 | 3.8152 | 3.2945 | +12.715% | -0.042% | -68.821% |
| 1 | 3.1092 | 4.9155 | 4.4277 | 4.9933 | 4.3993 | +9.752% | +1.015% | -41.599% |

| cluster | GPU A/A | CPU A/A | 同期修正版 A/A | block候補 A/A | 同DLL元対照 | controls |
|:---:|---:|---:|---:|---:|---:|:---:|
| 0 | +0.799% | -0.165% | +0.834% | +1.094% | +0.155% | True |
| 1 | +0.755% | +1.083% | +0.441% | +1.212% | +0.022% | True |

全RTK/PF/FGO速度、GPU利用率やGPU化率は未確立。GTSAM solver/整数探索はCPUのまま。新しい数値offload領域は増やしていない。

## 差分・同期検証

315tests PASS。既存289testsの全score/normal/solution/logdet/status、元fallback/例外/警告、context寿命と返却array所有権、3seed×4設定×4epochの全PF保存state/RNG/graph event比較を継承。raw6912行列はdimension3〜11、非対称/pivot/tie/特異/負det/NaN/Inf/overflow/subnormal/signed-zero/悪条件、guard内の後退代入Inf/NaN、6種の行交換pattern、元の減算順を検査する相殺ケースを含む。

追加22test×32=704行列は全11位置のpivot失敗と行置換あり/なしを作り、block全体の終了とpartial LU/logdet/pivots/RHSを元generic・同期修正版と比較。追加4testはcount1/63/128/256、計448行列のdimension3〜11、成功/特異/非有限RHS/負detを混在させ、block間独立性・最大raw batch・入力不変を確認。計8064行列比較、dimension11は2549比較。全配列のdtype/shape/bytesとstatusが一致し、新しい数値許容差はない。partial logdetの有限性/符号と未処理pivotの0も確認する。

CUDA memcheck/racecheck/synccheck各54対象subset PASS、error/hazard/leak0。全315testを各checkerで実行したとはせず、162固有testとも数えない。各suiteに既存pytest asyncio設定warning1件。allocation/非同期障害の注入は未実施。元numeric prefix/CPU body/通常kernel5144 normalized SASS命令は旧DLLと一致。新kernelは8880命令・64register/shared30904B/stack32B/spill0、rawは6432命令・96register/shared30904B/stack0/spill0。これらはsm75埋込みcubinの静的検査値。計測後のown-process照会で、current deviceのcompute capabilityは75、runtime versionは12080、driver API versionは13020。実際にロードされた/JITされたコードや動的resourceは取得していない。命令数やthread数を速度利得に換算せず、計測はその実行環境での全score時間として扱う。機種名は照会・記録しない。

## 実GTSAMとGPUの精度

run3/seed20260710/400epoch、cluster0/1各1prefixを元D236 GPU基準と比較。400保存PF state、全trajectory配列のdtype/shape/bytes、全cfgと従来caller metadataが一致。各197valid/3skip、計394nonempty GPU score、scalar fallback0。fieldのみ全diagnosticを取得しdimension11が両設定で正と確認、従来metadataだけをcallerへ返す。追加score call・guard-hit/block-LU counterなし。field時間は性能根拠にしない。全route/multiseed精度gate未実施。元NumPy/native間の既存丸め差と保存codecのscalar NaN payload正規化は継承。

staging host/device各1143352B/context、timing6contextsで各6860112B。raw probe最大device一時確保863232Bはtest専用。driver領域/CPU heap peak未測定。36製品source、5installed PE、PF build PE、72既報、35archives/5763filesを保全。

[結果](../benchmarks/rbpf_block_lu_20261017.json)、[完了照合](../benchmarks/rbpf_block_lu_completion_check_20261017.json)、[全score比較](../benchmarks/bench_rbpf_block_lu.py)、[CUDA source](../benchmarks/probes/rbpf_block_lu/block_lu.cuh)、[追加検査](../tests/test_rbpf_block_lu.py)。
