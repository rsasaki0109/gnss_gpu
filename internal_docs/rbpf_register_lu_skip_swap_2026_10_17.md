# D254: Pivotが動かない行交換通信の省略 (2026-10-17)

同期修正版D253を土台に、pivotが現在行を選ぶときのrow/RHS shuffleだけを省いた。全laneで同じpivot indexを使う条件分岐で、交換が必要なら全32laneが同じshuffleへ参加する。浮動小数点演算・比較と各要素の演算順、FP64/fmad=false、前回追加したstatus読出し後のsyncwarpを含む両barrierは維持する。全観測・候補・粒子、非対称prior、IRLS回数、閾値、RNGを変えない。dimension11以外と元のrange guard fallbackも維持。

今回、同期修正版を初めて新候補と同じ実験内で計測した。初版D253の不安全なDLL/速度記録は一切使わず、過去の判断・source・記録を更新しない。結果はGPU component gate 0/2、CPU joint gate 0/2。製品・既定は変更せず、全PF/FGO速度は未確立。

今回の交換省略単体は効果不足として不採用。旧GPU比は+14.048/+12.435%だが、同期修正版からの増分は+0.555/+0.458%で3%基準に届かず、対照の変動と同程度だった。旧GPUとの約14%/12%の差を今回の交換省略だけの効果とは解釈しない。元GPU/同DLL元/newgenericのbase条件は2/2通過、増分条件は0/2。最適化CPU比は-103.227/-61.654%短縮（所要時間約2.032/1.617倍）で両設定とも遅く、CPUからの切替を正当化しない。

## 差分検証

271tests PASS。既存265testsの全score/normal/solution/logdet/status、fallback・例外・警告、返却array所有権とcontext寿命、3seed×4設定×4epochの全PF保存state/RNG/graph event比較を継承。raw6144行列にはdimension3〜11、非対称、pivot tie、特異/負det、NaN/Inf、overflow/subnormal/signed-zero/悪条件、有限guard内の後退代入Inf/NaNを含む。さらに6種類×32=192行列で、交換なし・隣接交換・逆順・cycle・交換混在・signed-zeroを元genericと同期修正版と新候補で照合。合計6336行列、dimension11は1216行列。値の許容差は追加せずdtype/shape/bytesを比較した。

CUDA memcheck/racecheck/synccheck各28対象subset PASS、error/hazard/leak0。同じtestを3回別checkerで走らせたもので84固有testとは数えない。各suiteに既存pytest asyncio設定warning1件。allocation/非同期障害の注入は未実施。新kernelは17952 normalized SASS命令、70register/shared30904B/stack32B/spill0。同期修正版は18336命令・64register。raw probeは12848命令・96register/shared30904B/stack0/spill0。元kernel5144命令のnormalized SASS hashとnumeric prefix/CPU bodyは旧DLLと一致。命令/レジスタ数を速度利得には換算しない。

## 実入力の全score時間

凍結した全394入力、10mode×5round×3group、59100timed/3940warmを1回の実験で比較。modeは最適化CPU、旧GPU、同DLL元GPU、新generic、同期修正版register、新候補、および旧GPU/CPU/同期修正版/新候補の各A/A。順序をrotate/reverseする。全warmの同backend全diagnostic/status/score/input bits、全timed返却hashが一致し、GPUは過去の凍結GPU基準とも照合した。

dedup/DD/cluster/pack/guard、分岐・shuffle・solve、転送/kernel/完了同期/返却、fallback/scatterはtimer内。workspace構築/warm/復元copy/hashは外。allowed CPU1本/BLAS1/normal priority/非exclusive。GPU処理は直列、timed callに追加event/counterなし。全197batchの整数ns合計→round内3groupのpaired率median→5round median。事前gateは旧GPU/同DLL元GPU/新generic各10%以上・各4/5round以上正、同期修正版に対する増分3%以上・4/5round以上正、4種A/Aと同DLL元対照のabsmedian<5%。CPU比10%以上・4/5round以上正をjoint gateとして別判定する。新genericはcode/resourcesが違うため旧GPUとの等速対照にはしない。完了候補の再測定なし。

時間は197batch合計median÷197で個別batchのp50ではない。paired短縮率は別統計量。絶対時間を他実験と引き比べない。

| cluster | CPU ms | 旧GPU ms | 同期修正版 ms | 新generic ms | 新候補 ms | 対旧GPU短縮 | 対同期修正版短縮 | 対CPU短縮 |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.4999 | 3.5492 | 2.9283 | 3.4791 | 2.9229 | +14.048% | +0.555% | -103.227% |
| 1 | 2.2147 | 4.2138 | 3.6633 | 4.1553 | 3.6887 | +12.435% | +0.458% | -61.654% |

| cluster | GPU A/A | CPU A/A | 同期修正版 A/A | 新候補 A/A | 同DLL元対照 | controls |
|:---:|---:|---:|---:|---:|---:|:---:|
| 0 | -0.005% | +0.482% | +0.606% | +0.034% | -0.143% | True |
| 1 | +1.016% | -1.452% | +0.536% | +0.174% | +0.434% | True |

全round/raw値を結果JSON・private archiveへ保持する。非exclusive環境で対照gateとround分布を併記し、GPU利用率・GPU化率・全RTK/PF/FGO速度には換算しない。component gateを通らない設定の速度採用はしない。GTSAM solver/整数探索はCPUのまま。

## 実GTSAMとGPUの精度

run3/seed20260710/400epoch、cluster0/1を各1prefix、元D236 GPU基準と比較。400保存PF state、全trajectory配列のdtype/shape/bytes、全cfg・従来caller metadataが一致。各197valid/3skip、計394nonempty GPU score、scalar fallback0。fieldのみ全diagnosticを取得しdimension11が両設定で正であることを確認、従来metadataだけをcallerへ返す。追加score callなし。guard通過数/identity交換省略数の実測counterはなく、dimension11件数を省略回数とは呼ばない。field時間は性能根拠にしない。全route/multiseed精度gateは未実施。元NumPy/native間の既存丸め差と保存codecのscalar NaN payload正規化は継承する。

通常scoreへの追加buffer/転送/counterなし。staging host/device各1143352B/context、timing6contextsで各6860112B。raw probeの最大device一時確保863232Bはtest専用。driver領域/CPU heap peak未測定。36製品source、5installed PE、PF build PE、68既報、33archives/5352filesを保全した。

[結果](../benchmarks/rbpf_register_lu_skip_swap_20261017.json)、[完了照合](../benchmarks/rbpf_register_lu_skip_swap_completion_check_20261017.json)、[全score比較](../benchmarks/bench_rbpf_register_lu_skip_swap.py)、[CUDA source](../benchmarks/probes/rbpf_register_lu_skip_swap/register_lu.cuh)、[追加検査](../tests/test_rbpf_register_lu_skip_swap.py)。
